"""Expand child-chunk hits into their parent sections for LLM context."""

from __future__ import annotations

from collections.abc import Sequence

from writing_factory.kb.models import FusedHit
from writing_factory.store.kb_repository import KnowledgeBaseRepository


class ParentExpander:
    """Replace or augment child hits with the larger parent block they belong to.

    Child chunks give precise lexical/semantic hits but little context. The parent
    block (a whole section) is what should be handed to the LLM. We keep the child
    hit for traceability and mark the returned parent hit as expanded.
    """

    def __init__(self, repository: KnowledgeBaseRepository) -> None:
        self.repository = repository

    def expand(self, hits: Sequence[FusedHit]) -> list[FusedHit]:
        """Return one FusedHit per unique parent (or the child if it has none)."""

        if not hits:
            return []
        grouped: dict[str, list[FusedHit]] = {}
        for hit in hits:
            grouped.setdefault(hit.parent_id or hit.chunk_id, []).append(hit)

        expanded: list[FusedHit] = []
        for group_id, child_hits in grouped.items():
            best = max(child_hits, key=lambda item: (item.rrf_score, -item.final_rank))
            matched_child_ids = _matched_child_ids(child_hits)
            if best.parent_id is None:
                expanded.append(best.model_copy(update={"matched_child_ids": matched_child_ids}))
                continue
            parent_chunk = self.repository.parent_chunk(group_id)
            if parent_chunk is None:
                expanded.extend(
                    hit.model_copy(
                        update={
                            "matched_child_ids": hit.matched_child_ids or (hit.chunk_id,),
                        }
                    )
                    for hit in child_hits
                )
                continue
            modalities = {hit.source for hit in child_hits}
            source = best.source if len(modalities) == 1 else "hybrid"
            expanded.append(
                FusedHit(
                    chunk_id=parent_chunk.chunk_id,
                    doc_id=parent_chunk.doc_id,
                    text=parent_chunk.text,
                    source=source,
                    dense_rank=_minimum_defined(hit.dense_rank for hit in child_hits),
                    sparse_rank=_minimum_defined(hit.sparse_rank for hit in child_hits),
                    rrf_score=best.rrf_score,
                    rerank_score=best.rerank_score,
                    final_rank=0,
                    page_start=parent_chunk.page_start,
                    page_end=parent_chunk.page_end,
                    section_heading=parent_chunk.section_heading,
                    parent_id=None,
                    expanded_from_child=True,
                    matched_child_ids=matched_child_ids,
                )
            )
        expanded.sort(key=lambda item: (-item.rrf_score, item.chunk_id))
        return [
            item.model_copy(update={"final_rank": index})
            for index, item in enumerate(expanded, start=1)
        ]


def _matched_child_ids(hits: Sequence[FusedHit]) -> tuple[str, ...]:
    ordered: dict[str, None] = {}
    for hit in sorted(hits, key=lambda item: (-item.rrf_score, item.chunk_id)):
        for chunk_id in hit.matched_child_ids or (hit.chunk_id,):
            ordered.setdefault(chunk_id, None)
    return tuple(ordered)


def _minimum_defined(values) -> int | None:
    defined = [value for value in values if value is not None]
    return min(defined) if defined else None
