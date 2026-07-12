"""LanceDB replacement/filtering and SQLite-backed BM25 tests."""

from __future__ import annotations

from pathlib import Path

from writing_factory.kb.chunking import StructureChunker
from writing_factory.kb.files import ManagedFileStore
from writing_factory.kb.models import Bibliography, Chunk, ParsedBlock, ParsedDocument
from writing_factory.store import Database
from writing_factory.store.bm25_index import BM25Index
from writing_factory.store.kb_repository import KnowledgeBaseRepository
from writing_factory.store.vector_index import LanceVectorIndex


def _child(chunk_id: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        chunk_index=0,
        char_start=0,
        char_end=len(text),
        chunk_kind="child",
        parent_id="parent",
    )


def test_lancedb_replaces_one_document_and_filters_allowed_docs(tmp_path: Path) -> None:
    index = LanceVectorIndex(tmp_path / "lance")
    first = _child("chunk_a", "doc_a", "档案数字化")
    second = _child("chunk_b", "doc_b", "田野调查")
    index.replace_document("doc_a", [first], [[1.0, 0.0, 0.0]])
    index.replace_document("doc_b", [second], [[0.0, 1.0, 0.0]])

    only_second = index.search([1.0, 0.0, 0.0], allowed_doc_ids={"doc_b"}, limit=5)
    index.replace_document(
        "doc_a", [first.model_copy(update={"text": "数字档案"})], [[0.9, 0.1, 0.0]]
    )
    only_first = index.search([1.0, 0.0, 0.0], allowed_doc_ids={"doc_a"}, limit=5)

    assert [hit.doc_id for hit in only_second] == ["doc_b"]
    assert [hit.chunk_id for hit in only_first] == ["chunk_a"]
    assert index.has_document("doc_a")


def test_bm25_rebuilds_from_ready_sqlite_chunks(settings, tmp_path: Path) -> None:
    database = Database(settings.database_path)
    database.initialize()
    repository = KnowledgeBaseRepository(database)
    kb_id = repository.ensure_default()
    source = tmp_path / "史料.txt"
    source.write_text("数字人文方法。\n\n田野调查方法。", encoding="utf-8")
    managed = ManagedFileStore(settings.managed_documents_dir).import_file(source)
    parsed = ParsedDocument(
        filename=source.name,
        format="txt",
        parser_name="fixture",
        parser_version="1",
        blocks=[
            ParsedBlock(order=0, block_type="text", text="数字人文方法。"),
            ParsedBlock(order=1, block_type="text", text="田野调查方法。"),
        ],
    )
    chunked = StructureChunker(child_target_chars=5).chunk(managed.doc_id, parsed)
    job_id = repository.create_job(kb_id, source)
    repository.save_document_and_chunks(
        kb_id=kb_id,
        job_id=job_id,
        managed=managed,
        bibliography=Bibliography(title="史料"),
        parsed=parsed,
        chunked=chunked,
    )
    repository.mark_ready(kb_id, managed.doc_id, job_id)

    first_process = BM25Index(repository).search(kb_id, "田野调查", limit=2)
    second_process = BM25Index(repository).search(kb_id, "田野调查", limit=2)

    assert first_process[0].text == "田野调查方法。"
    assert [hit.chunk_id for hit in second_process] == [hit.chunk_id for hit in first_process]
