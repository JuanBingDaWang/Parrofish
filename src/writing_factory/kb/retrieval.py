"""Stage-one dense and sparse retrieval entry points kept deliberately separate."""

from __future__ import annotations

from writing_factory.kb.models import SearchHit
from writing_factory.llm import SiliconFlowClient
from writing_factory.store.bm25_index import BM25Index
from writing_factory.store.kb_repository import KnowledgeBaseRepository
from writing_factory.store.vector_index import LanceVectorIndex


class DenseRetriever:
    """Embed a query and search only SQLite-published KB documents."""

    def __init__(
        self,
        repository: KnowledgeBaseRepository,
        vectors: LanceVectorIndex,
        siliconflow: SiliconFlowClient,
    ) -> None:
        self.repository = repository
        self.vectors = vectors
        self.siliconflow = siliconflow

    def search(self, kb_id: str, query: str, *, limit: int = 5) -> list[SearchHit]:
        """Return dense results; fusion and reranking arrive in stage 3."""

        chunks = self.repository.ready_child_chunks(kb_id)
        allowed_doc_ids = {chunk.doc_id for chunk in chunks}
        if not allowed_doc_ids or not query.strip():
            return []
        embedding = self.siliconflow.embeddings([query])
        return self.vectors.search(
            embedding.vectors[0],
            allowed_doc_ids=allowed_doc_ids,
            limit=limit,
        )


class SparseRetriever:
    """Expose the jieba/BM25 index under the same result contract."""

    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def search(self, kb_id: str, query: str, *, limit: int = 5) -> list[SearchHit]:
        """Return sparse results; RRF remains a stage 3 responsibility."""

        return self.index.search(kb_id, query, limit=limit)
