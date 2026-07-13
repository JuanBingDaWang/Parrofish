"""End-to-end local ingestion state, idempotency, and failure tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from writing_factory.kb.chunking import StructureChunker
from writing_factory.kb.files import ManagedFileStore
from writing_factory.kb.ingestion import IngestionService
from writing_factory.kb.models import ParsedDocument
from writing_factory.kb.parsing import TextParser
from writing_factory.llm.models import EmbeddingResult
from writing_factory.store import Database
from writing_factory.store.bm25_index import BM25Index
from writing_factory.store.kb_repository import KnowledgeBaseRepository
from writing_factory.store.vector_index import LanceVectorIndex


class TextRouter:
    def parse(self, document, *, progress, check_cancelled) -> ParsedDocument:
        check_cancelled()
        progress(48, "parsed")
        parsed = TextParser().parse(document.managed_path)
        return parsed.model_copy(update={"filename": document.filename})


class FakeEmbeddings:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def embeddings(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding failed")
        vectors = [[1.0, 0.0, 0.0] if "数字" in text else [0.0, 1.0, 0.0] for text in texts]
        return EmbeddingResult(vectors=vectors, model="fixture")


def _service(settings, embeddings):
    database = Database(settings.database_path)
    database.initialize()
    repository = KnowledgeBaseRepository(database)
    kb_id = repository.ensure_default()
    vectors = LanceVectorIndex(settings.lancedb_path)
    bm25 = BM25Index(repository)
    service = IngestionService(
        settings,
        repository,
        ManagedFileStore(settings.managed_documents_dir),
        TextRouter(),
        StructureChunker(child_target_chars=8),
        embeddings,
        vectors,
        bm25,
    )
    return database, repository, kb_id, vectors, bm25, service


def test_ingests_both_indexes_and_reuses_duplicate(settings, tmp_path: Path) -> None:
    embeddings = FakeEmbeddings()
    database, repository, kb_id, vectors, bm25, service = _service(settings, embeddings)
    source = tmp_path / "方法.txt"
    source.write_text("数字人文档案。\n\n田野调查访谈。", encoding="utf-8")
    progress: list[int] = []

    first = service.ingest(
        kb_id,
        source,
        progress=lambda percent, _message: progress.append(percent),
    )
    second = service.ingest(kb_id, source)

    ready_chunks = repository.ready_child_chunks(kb_id)
    dense = vectors.search(
        [1.0, 0.0, 0.0],
        allowed_doc_ids={first.doc_id},
        limit=2,
    )
    sparse = bm25.search(kb_id, "田野调查", limit=2)
    assert first.child_chunk_count == 2
    assert second.reused_document
    assert embeddings.calls == 1
    assert len(ready_chunks) == 2
    assert dense[0].text == "数字人文档案。"
    assert sparse[0].text == "田野调查访谈。"
    assert progress[-1] == 100
    with database.connection() as connection:
        statuses = [
            row[0]
            for row in connection.execute(
                "SELECT status FROM ingest_jobs ORDER BY created_at"
            ).fetchall()
        ]
    assert statuses == ["ready", "ready"]


def test_embedding_failure_is_not_retrievable(settings, tmp_path: Path) -> None:
    database, repository, kb_id, _vectors, _bm25, service = _service(
        settings, FakeEmbeddings(fail=True)
    )
    source = tmp_path / "失败.txt"
    source.write_text("不会发布的文本。", encoding="utf-8")

    with pytest.raises(RuntimeError, match="embedding failed"):
        service.ingest(kb_id, source)

    assert repository.ready_child_chunks(kb_id) == []
    with database.connection() as connection:
        row = connection.execute("SELECT status, error_message FROM ingest_jobs").fetchone()
    assert row["status"] == "failed"
    assert row["error_message"] == "RuntimeError"


def test_deletes_document_from_both_indexes_and_keeps_original(settings, tmp_path: Path) -> None:
    _database, repository, kb_id, vectors, bm25, service = _service(settings, FakeEmbeddings())
    source = tmp_path / "保留原件.txt"
    source.write_text("数字出版研究。\n\n公共文化服务。", encoding="utf-8")
    ingested = service.ingest(kb_id, source)
    managed = next(settings.managed_documents_dir.iterdir())
    artifact_dir = settings.mineru_artifacts_dir / managed.stem
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "result.zip").write_bytes(b"cached")

    result = service.delete_documents(kb_id, {ingested.doc_id})

    assert result.removed_count == 1
    assert result.orphaned_count == 1
    assert result.cleanup_failures == 0
    assert source.is_file()
    assert not managed.exists()
    assert not artifact_dir.exists()
    assert repository.list_documents(kb_id) == []
    assert not vectors.has_document(ingested.doc_id)
    assert bm25.search(kb_id, "数字出版", limit=3) == []
