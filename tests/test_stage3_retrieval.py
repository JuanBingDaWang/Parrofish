"""Stage-3 end-to-end contracts for filtering, caching, and API orchestration."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from writing_factory.eval import RecallCase, recall_at_k
from writing_factory.kb.models import Chunk, MetadataFilter, RetrievalRequest, SearchHit
from writing_factory.kb.retrieval import HybridRetriever
from writing_factory.llm.models import (
    ChatResult,
    EmbeddingResult,
    RerankItem,
    RerankResult,
)
from writing_factory.store import Database
from writing_factory.store.bm25_index import BM25Index
from writing_factory.store.kb_repository import KnowledgeBaseRepository
from writing_factory.store.vector_index import LanceVectorIndex


class _FakeSiliconFlow:
    def __init__(self, *, delayed_chat: bool = False) -> None:
        self.delayed_chat = delayed_chat
        self.embedding_batches: list[tuple[str, ...]] = []
        self.rerank_calls = 0
        self.chat_calls = 0
        self.chat_peak = 0
        self._chat_active = 0
        self._lock = threading.Lock()

    def chat(self, messages, **_kwargs) -> ChatResult:
        with self._lock:
            self.chat_calls += 1
            self._chat_active += 1
            self.chat_peak = max(self.chat_peak, self._chat_active)
        try:
            if self.delayed_chat:
                time.sleep(0.04)
            system = messages[0]["content"]
            content = (
                '{"queries":["原问题","具体子问题","具体子问题"]}'
                if "查询扩展" in system
                else "假设性答案"
            )
            return ChatResult(content=content, model="fake-chat")
        finally:
            with self._lock:
                self._chat_active -= 1

    def embeddings(self, texts, **_kwargs) -> EmbeddingResult:
        self.embedding_batches.append(tuple(texts))
        return EmbeddingResult(
            vectors=[[1.0, 0.0] for _text in texts],
            model="fake-embedding",
        )

    def rerank(self, _query, documents, *, top_n, **_kwargs) -> RerankResult:
        self.rerank_calls += 1
        return RerankResult(
            results=[
                RerankItem(index=index, relevance_score=1.0 - index / 100)
                for index in range(min(top_n, len(documents)))
            ],
            model="fake-reranker",
        )


class _MutableRepository:
    def __init__(self, chunk: Chunk) -> None:
        self.chunk = chunk
        self.version = "v1"

    def ready_child_chunks(self, _kb_id, *, filters=None) -> list[Chunk]:
        return [self.chunk]

    def retrieval_fingerprint(self, _kb_id: str) -> str:
        return self.version

    def parent_chunk(self, _parent_id: str):
        return None


class _MutableVectors:
    def __init__(self, repository: _MutableRepository) -> None:
        self.repository = repository
        self.allowed_scopes: list[set[str]] = []

    def search(self, _vector, *, allowed_chunk_ids=None, limit, **_kwargs) -> list[SearchHit]:
        chunk = self.repository.chunk
        allowed = {chunk.chunk_id} if allowed_chunk_ids is None else set(allowed_chunk_ids)
        self.allowed_scopes.append(allowed)
        if chunk.chunk_id not in allowed or limit <= 0:
            return []
        return [_search_hit(chunk, "dense")]


class _MutableBM25:
    def __init__(self, repository: _MutableRepository) -> None:
        self.repository = repository

    def search(self, _kb_id, _query, *, allowed_chunk_ids, limit, **_kwargs):
        chunk = self.repository.chunk
        if chunk.chunk_id not in allowed_chunk_ids or limit <= 0:
            return []
        return [_search_hit(chunk, "bm25")]


def _child(chunk_id: str, *, doc_id: str = "doc", parent_id: str | None = None) -> Chunk:
    text = f"{chunk_id} 的检索文本"
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        section_heading="引言",
        chunk_index=0,
        char_start=0,
        char_end=len(text),
        parent_id=parent_id,
        chunk_kind="child",
    )


def _search_hit(chunk: Chunk, source: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        score=1.0,
        rank=1,
        source=source,  # type: ignore[arg-type]
        section_heading=chunk.section_heading,
        parent_id=chunk.parent_id,
    )


def test_full_chain_parallelizes_expansion_and_batches_all_embeddings() -> None:
    repository = _MutableRepository(_child("chunk"))
    vectors = _MutableVectors(repository)
    bm25 = _MutableBM25(repository)
    siliconflow = _FakeSiliconFlow(delayed_chat=True)
    retriever = HybridRetriever(repository, vectors, bm25, siliconflow)  # type: ignore[arg-type]
    progress: list[tuple[int, str]] = []

    result = retriever.search(
        RetrievalRequest(kb_id="kb", query="原问题", parent_expand=False),
        progress=lambda value, message: progress.append((value, message)),
    )

    assert result.expanded_queries == ("具体子问题",)
    assert result.hits[0].matched_child_ids == ("chunk",)
    assert siliconflow.chat_calls == 2
    assert siliconflow.chat_peak == 2
    assert siliconflow.embedding_batches == [("原问题", "具体子问题", "假设性答案")]
    assert siliconflow.rerank_calls == 1
    assert progress[-1] == (100, "检索完成")


def test_cache_key_tracks_published_corpus_version() -> None:
    repository = _MutableRepository(_child("old"))
    siliconflow = _FakeSiliconFlow()
    retriever = HybridRetriever(
        repository,
        _MutableVectors(repository),
        _MutableBM25(repository),
        siliconflow,
    )  # type: ignore[arg-type]
    request = RetrievalRequest(
        kb_id="kb",
        query="检索",
        use_rewrite=False,
        use_hyde=False,
        use_rerank=False,
        parent_expand=False,
    )

    first = retriever.search(request)
    cached = retriever.search(request)
    repository.chunk = _child("new")
    repository.version = "v2"
    refreshed = retriever.search(request)

    assert first.hits[0].chunk_id == "old"
    assert cached.hits[0].chunk_id == "old"
    assert refreshed.hits[0].chunk_id == "new"
    assert len(siliconflow.embedding_batches) == 2


def test_metadata_filters_constrain_dense_and_sparse_results(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    repository = KnowledgeBaseRepository(database)
    kb_id = repository.ensure_default()
    chunks = [
        _insert_document(database, kb_id, "doc_a", "叶芃", 2024, "pdf", "引言", "实践性"),
        _insert_document(database, kb_id, "doc_b", "张三", 2023, "txt", "方法", "实践性"),
    ]
    vectors = LanceVectorIndex(settings.lancedb_path)
    for chunk in chunks:
        vectors.replace_document(chunk.doc_id, [chunk], [[1.0, 0.0]])
    siliconflow = _FakeSiliconFlow()
    retriever = HybridRetriever(repository, vectors, BM25Index(repository), siliconflow)  # type: ignore[arg-type]
    request = RetrievalRequest(
        kb_id=kb_id,
        query="实践性",
        filters=MetadataFilter(
            doc_ids={"doc_a"},
            authors={"叶芃"},
            years={2024},
            formats={"pdf"},
            section_headings={"引言"},
            chunk_kinds={"child"},
        ),
        use_rewrite=False,
        use_hyde=False,
        use_rerank=False,
        parent_expand=False,
    )

    result = retriever.search(request)

    assert {hit.chunk_id for hit in result.hits} == {"doc_a_child"}
    assert (
        repository.ready_child_chunks(
            kb_id,
            filters=MetadataFilter(authors={"叶芃"}, years={2024}),
        )[0].doc_id
        == "doc_a"
    )
    assert (
        repository.ready_child_chunks(
            kb_id,
            filters=MetadataFilter(chunk_kinds={"parent"}),
        )
        == []
    )
    assert repository.ready_child_chunks(kb_id, filters=MetadataFilter(doc_ids=set())) == []
    resolved = repository.ready_child_chunks_by_ids(
        kb_id,
        set(result.hits[0].matched_child_ids),
    )
    assert [(chunk.chunk_id, chunk.text) for chunk in resolved] == [("doc_a_child", "实践性")]


def _insert_document(
    database: Database,
    kb_id: str,
    doc_id: str,
    author: str,
    year: int,
    file_format: str,
    section: str,
    text: str,
) -> Chunk:
    parent_text = f"{section}\n{text}"
    child = Chunk(
        chunk_id=f"{doc_id}_child",
        doc_id=doc_id,
        text=text,
        section_heading=section,
        chunk_index=0,
        char_start=0,
        char_end=len(text),
        parent_id=f"{doc_id}_parent",
        chunk_kind="child",
    )
    with database.connection() as connection:
        connection.execute(
            """
            INSERT INTO documents(
                doc_id, sha256, filename, format, source_path, managed_path, bib_json,
                parser_name, parser_version, canonical_text, ingest_date, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'fixture', '1', ?, '2026-01-01', '2026-01-01')
            """,
            (
                doc_id,
                f"sha_{doc_id}",
                f"{doc_id}.{file_format}",
                file_format,
                f"/{doc_id}",
                f"/{doc_id}",
                json.dumps({"title": doc_id, "author": author, "year": year}),
                parent_text,
            ),
        )
        connection.execute(
            """
            INSERT INTO knowledge_base_documents(kb_id, doc_id, added_at, status)
            VALUES (?, ?, '2026-01-01', 'ready')
            """,
            (kb_id, doc_id),
        )
        connection.execute(
            """
            INSERT INTO chunks(
                chunk_id, doc_id, text, section_heading, chunk_index, char_start,
                char_end, parent_id, chunk_kind, metadata_json
            ) VALUES (?, ?, ?, ?, 0, 0, ?, NULL, 'parent', '{}')
            """,
            (f"{doc_id}_parent", doc_id, parent_text, section, len(parent_text)),
        )
        connection.execute(
            """
            INSERT INTO chunks(
                chunk_id, doc_id, text, section_heading, chunk_index, char_start,
                char_end, parent_id, chunk_kind, metadata_json
            ) VALUES (?, ?, ?, ?, 0, 0, ?, ?, 'child', '{}')
            """,
            (child.chunk_id, doc_id, text, section, len(text), child.parent_id),
        )
    return child


def test_retrieval_request_rejects_unbounded_or_non_positive_limits() -> None:
    with pytest.raises(ValidationError):
        RetrievalRequest(kb_id="kb", query="x", top_k=0)
    with pytest.raises(ValidationError):
        RetrievalRequest(kb_id="kb", query="x", dense_limit=101)


def test_checked_in_chinese_golden_set_has_perfect_bm25_recall_at_one() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "stage3_retrieval_golden.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    chunks = [
        _child(item["chunk_id"]).model_copy(
            update={"text": item["text"], "char_end": len(item["text"])}
        )
        for item in payload["documents"]
    ]
    repository = _GoldenRepository(chunks)
    bm25 = BM25Index(repository)  # type: ignore[arg-type]
    cases = [
        RecallCase(case["query"], frozenset(case["expected_chunk_ids"]))
        for case in payload["cases"]
    ]

    score = recall_at_k(cases, lambda query, k: bm25.search("kb", query, limit=k), k=1)

    assert score == 1.0


class _GoldenRepository:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def ready_child_chunks(self, _kb_id: str, *, filters=None) -> list[Chunk]:
        return self.chunks
