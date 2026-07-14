"""Read-only stage-3 hybrid retrieval with traceable child-chunk provenance."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any

from writing_factory.kb.fusion import fuse_many
from writing_factory.kb.models import (
    Chunk,
    MetadataFilter,
    RetrievalRequest,
    RetrievalResult,
    SearchHit,
)
from writing_factory.kb.parent_retriever import ParentExpander
from writing_factory.kb.rerank import Reranker
from writing_factory.kb.rewrite import QueryExpander
from writing_factory.llm import SiliconFlowClient
from writing_factory.store.bm25_index import BM25Index
from writing_factory.store.kb_repository import KnowledgeBaseRepository
from writing_factory.store.vector_index import LanceVectorIndex

ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], None]


def _no_progress(_percent: int, _message: str) -> None:
    pass


def _no_cancellation() -> None:
    pass


class DenseRetriever:
    """Embed queries in one batch and search the SQLite-approved child scope."""

    def __init__(
        self,
        repository: KnowledgeBaseRepository,
        vectors: LanceVectorIndex,
        siliconflow: SiliconFlowClient,
    ) -> None:
        self.repository = repository
        self.vectors = vectors
        self.siliconflow = siliconflow

    def search(
        self,
        kb_id: str,
        query: str,
        *,
        limit: int = 5,
        filters: MetadataFilter | None = None,
    ) -> list[SearchHit]:
        """Return dense results for one query."""

        results = self.search_many(kb_id, (query,), limit=limit, filters=filters)
        return results[0] if results else []

    def search_many(
        self,
        kb_id: str,
        queries: Sequence[str],
        *,
        limit: int,
        filters: MetadataFilter | None = None,
        allowed_chunks: Sequence[Chunk] | None = None,
    ) -> list[list[SearchHit]]:
        """Batch query embeddings while keeping a separate ranked list per query."""

        cleaned = [query for query in queries if query.strip()]
        if len(cleaned) != len(queries):
            raise ValueError("Dense query batch cannot contain blank queries")
        if not cleaned or limit <= 0:
            return []
        chunks = (
            list(allowed_chunks)
            if allowed_chunks is not None
            else self.repository.ready_child_chunks(kb_id, filters=filters)
        )
        if not chunks:
            return [[] for _query in cleaned]
        embedded = self.siliconflow.embeddings(cleaned, priority=20)
        if len(embedded.vectors) != len(cleaned):
            raise ValueError("Embedding result count does not match retrieval query count")
        allowed_doc_ids = {chunk.doc_id for chunk in chunks}
        allowed_chunk_ids = (
            {chunk.chunk_id for chunk in chunks}
            if filters is not None and filters.section_headings is not None
            else None
        )
        return [
            self.vectors.search(
                vector,
                allowed_doc_ids=allowed_doc_ids,
                allowed_chunk_ids=allowed_chunk_ids,
                limit=limit,
            )
            for vector in embedded.vectors
        ]


class SparseRetriever:
    """Expose the jieba/BM25 index under the same result contract."""

    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def search(
        self,
        kb_id: str,
        query: str,
        *,
        limit: int = 5,
        filters: MetadataFilter | None = None,
        allowed_chunk_ids: set[str] | None = None,
    ) -> list[SearchHit]:
        """Return sparse results within the same child scope as dense retrieval."""

        return self.index.search(
            kb_id,
            query,
            limit=limit,
            filters=filters,
            allowed_chunk_ids=allowed_chunk_ids,
        )


class HybridRetriever:
    """Compose rewrite/HyDE, hybrid RRF, parent expansion, and reranking.

    Persona data is deliberately absent: this component only retrieves facts. Every
    returned parent carries the exact child identifiers that caused the match.
    """

    CACHE_LIMIT = 128

    def __init__(
        self,
        repository: KnowledgeBaseRepository,
        vectors: LanceVectorIndex,
        bm25: BM25Index,
        siliconflow: SiliconFlowClient,
    ) -> None:
        self.repository = repository
        self.vectors = vectors
        self.bm25 = bm25
        self.siliconflow = siliconflow
        self.dense = DenseRetriever(repository, vectors, siliconflow)
        self.sparse = SparseRetriever(bm25)
        self.expander = ParentExpander(repository)
        self.rewriter = QueryExpander(siliconflow)
        self.reranker = Reranker(siliconflow)
        self._cache: OrderedDict[str, RetrievalResult] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _cache_key(self, request: RetrievalRequest) -> str:
        payload = _canonicalize(request.model_dump(mode="python"))
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        fingerprint = self.repository.retrieval_fingerprint(request.kb_id)
        return hashlib.sha256(f"{fingerprint}\x00{serialized}".encode()).hexdigest()

    def search(
        self,
        request: RetrievalRequest,
        *,
        progress: ProgressCallback = _no_progress,
        check_cancelled: CancellationCheck = _no_cancellation,
    ) -> RetrievalResult:
        """Run the full retrieval pipeline with cooperative task boundaries."""

        check_cancelled()
        if not request.query.strip():
            progress(100, "检索问题为空")
            return RetrievalResult(query=request.query)

        key = self._cache_key(request)
        cached = self._get_cached(key)
        if cached is not None:
            progress(100, "复用检索结果")
            return cached

        progress(5, "解析检索范围")
        chunks = self.repository.ready_child_chunks(request.kb_id, filters=request.filters)
        if not chunks:
            result = RetrievalResult(query=request.query)
            self._remember(key, result)
            progress(100, "没有符合条件的语料")
            return result

        check_cancelled()
        progress(12, "扩展检索问题")
        queries, hyde_query = self._expand_query(request)
        check_cancelled()
        progress(32, "生成查询向量")

        dense_queries = [*queries, *([hyde_query] if hyde_query else [])]
        allowed_chunk_ids = {chunk.chunk_id for chunk in chunks}
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrieval") as executor:
            dense_future = executor.submit(
                self.dense.search_many,
                request.kb_id,
                dense_queries,
                limit=request.dense_limit,
                filters=request.filters,
                allowed_chunks=chunks,
            )
            sparse_future = executor.submit(
                self._sparse_lists,
                request,
                queries,
                allowed_chunk_ids,
            )
            dense_lists = dense_future.result()
            sparse_lists = sparse_future.result()

        check_cancelled()
        progress(62, "融合稠密与稀疏结果")
        result_lists: list[Sequence[SearchHit]] = []
        for index in range(len(queries)):
            result_lists.extend((dense_lists[index], sparse_lists[index]))
        if hyde_query:
            result_lists.append(dense_lists[-1])
        candidate_limit = max(request.dense_limit, request.sparse_limit, request.top_k)
        fused = fuse_many(result_lists, limit=candidate_limit)

        if request.parent_expand:
            progress(74, "补充父级上下文")
            fused = self.expander.expand(fused)

        check_cancelled()
        if request.use_rerank and fused:
            progress(84, "重排候选证据")
            fused = self.reranker.rerank(request.query, fused, top_n=request.top_k)
        else:
            fused = [
                item.model_copy(update={"final_rank": rank})
                for rank, item in enumerate(fused[: request.top_k], start=1)
            ]

        result = RetrievalResult(
            query=request.query,
            expanded_queries=tuple(queries[1:]),
            hits=tuple(fused),
        )
        self._remember(key, result)
        progress(100, "检索完成")
        return result

    def _expand_query(self, request: RetrievalRequest) -> tuple[list[str], str | None]:
        if request.use_rewrite and request.use_hyde:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="query-expand") as executor:
                rewrite_future = executor.submit(
                    copy_context().run,
                    self.rewriter.rewrite,
                    request.query,
                )
                hyde_future = executor.submit(
                    copy_context().run,
                    self.rewriter.hyde_passage,
                    request.query,
                )
                rewritten = rewrite_future.result()
                hyde_query = hyde_future.result()
        else:
            rewritten = self.rewriter.rewrite(request.query) if request.use_rewrite else []
            hyde_query = self.rewriter.hyde_passage(request.query) if request.use_hyde else None
        return _deduplicate_queries((request.query, *rewritten)), hyde_query

    def _sparse_lists(
        self,
        request: RetrievalRequest,
        queries: Sequence[str],
        allowed_chunk_ids: set[str],
    ) -> list[list[SearchHit]]:
        return [
            self.sparse.search(
                request.kb_id,
                query,
                limit=request.sparse_limit,
                filters=request.filters,
                allowed_chunk_ids=allowed_chunk_ids,
            )
            for query in queries
        ]

    def _get_cached(self, key: str) -> RetrievalResult | None:
        with self._cache_lock:
            result = self._cache.get(key)
            if result is not None:
                self._cache.move_to_end(key)
            return result

    def _remember(self, key: str, result: RetrievalResult) -> None:
        with self._cache_lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self.CACHE_LIMIT:
                self._cache.popitem(last=False)


def _deduplicate_queries(queries: Sequence[str]) -> list[str]:
    unique: dict[str, str] = {}
    for query in queries:
        cleaned = query.strip()
        if cleaned:
            unique.setdefault(cleaned.casefold(), cleaned)
    return list(unique.values())


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(_canonicalize(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value
