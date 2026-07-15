"""Reciprocal rank fusion for dense, sparse, and multi-query result lists."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from writing_factory.kb.models import FusedHit, SearchHit

DEFAULT_RRF_K = 60


@dataclass(slots=True)
class _FusionState:
    hit: SearchHit
    score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    seen_dense: bool = False
    seen_sparse: bool = False


def rrf_fuse(
    dense: Sequence[SearchHit],
    sparse: Sequence[SearchHit],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[FusedHit]:
    """Fuse one dense and one sparse ranked list."""

    return fuse_many((dense, sparse), k=k, limit=limit)


def fuse_many(
    result_lists: Iterable[Sequence[SearchHit]],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[FusedHit]:
    """Accumulate RRF contributions from every modality/query ranked list.

    A chunk that appears in several rewritten-query lists receives every reciprocal
    rank contribution. Duplicate occurrences inside one malformed list count once.
    """

    if k < 0:
        raise ValueError("RRF k must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("Fusion limit must be non-negative")

    states: dict[str, _FusionState] = {}
    for hits in result_lists:
        seen_in_list: set[str] = set()
        for fallback_rank, hit in enumerate(hits, start=1):
            if hit.chunk_id in seen_in_list:
                continue
            seen_in_list.add(hit.chunk_id)
            rank = hit.rank if hit.rank > 0 else fallback_rank
            state = states.get(hit.chunk_id)
            if state is None:
                state = _FusionState(hit=hit)
                states[hit.chunk_id] = state
            state.score += 1.0 / (k + rank)
            if hit.source == "dense":
                state.seen_dense = True
                state.dense_rank = _minimum_rank(state.dense_rank, rank)
            else:
                state.seen_sparse = True
                state.sparse_rank = _minimum_rank(state.sparse_rank, rank)

    fused: list[FusedHit] = []
    for chunk_id, state in states.items():
        source = "hybrid" if state.seen_dense and state.seen_sparse else state.hit.source
        fused.append(
            FusedHit(
                chunk_id=chunk_id,
                doc_id=state.hit.doc_id,
                text=state.hit.text,
                source=source,
                dense_rank=state.dense_rank,
                sparse_rank=state.sparse_rank,
                rrf_score=state.score,
                final_rank=0,
                page_start=state.hit.page_start,
                page_end=state.hit.page_end,
                section_heading=state.hit.section_heading,
                parent_id=state.hit.parent_id,
                matched_child_ids=(chunk_id,),
            )
        )

    fused.sort(key=lambda item: (-item.rrf_score, item.chunk_id))
    if limit is not None:
        fused = fused[:limit]
    return [item.model_copy(update={"final_rank": rank}) for rank, item in enumerate(fused, 1)]


def _minimum_rank(current: int | None, candidate: int) -> int:
    return candidate if current is None else min(current, candidate)
