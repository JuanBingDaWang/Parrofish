"""Rerank fused candidates with SiliconFlow's bge-reranker endpoint."""

from __future__ import annotations

from collections.abc import Sequence

from writing_factory.kb.models import FusedHit
from writing_factory.llm import SiliconFlowClient


class Reranker:
    """Coarse-to-fine reranking: fuse first, rerank a small candidate set, keep top_n."""

    def __init__(self, siliconflow: SiliconFlowClient) -> None:
        self.siliconflow = siliconflow

    def rerank(
        self,
        query: str,
        hits: Sequence[FusedHit],
        *,
        top_n: int,
        use_cache: bool = True,
    ) -> list[FusedHit]:
        """Reorder candidates by relevance score and assign a new final rank."""

        if not hits or top_n <= 0:
            return []
        documents = [hit.text for hit in hits]
        result = self.siliconflow.rerank(
            query,
            documents,
            top_n=min(top_n, len(documents)),
            use_cache=use_cache,
            priority=20,
        )
        scored: list[FusedHit] = []
        seen_indices: set[int] = set()
        for item in result.results:
            if item.index < 0 or item.index >= len(hits) or item.index in seen_indices:
                raise ValueError("Reranker returned an invalid candidate index")
            seen_indices.add(item.index)
            original = hits[item.index]
            scored.append(
                original.model_copy(update={"rerank_score": item.relevance_score, "final_rank": 0})
            )
        scored.sort(key=lambda item: (-(item.rerank_score or 0.0), item.chunk_id))
        for index, item in enumerate(scored):
            scored[index] = item.model_copy(update={"final_rank": index + 1})
        return scored[:top_n]
