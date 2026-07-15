"""Recall/precision evaluation for traceable retrieval regression cases."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from writing_factory.kb.models import FusedHit, SearchHit


@dataclass(frozen=True, slots=True)
class RecallCase:
    """One query and the chunk identifiers considered relevant."""

    query: str
    expected_chunk_ids: frozenset[str]


def _top_chunk_ids(hits: Sequence[SearchHit | FusedHit], k: int) -> set[str]:
    """Take the top-k chunk identifiers regardless of hit contract."""

    return {hit.chunk_id for hit in hits[:k]}


def recall_at_k(
    cases: Sequence[RecallCase],
    search: Callable[[str, int], Sequence[SearchHit | FusedHit]],
    *,
    k: int,
) -> float:
    """Compute macro recall over explicit expected chunk identifiers."""

    if not cases:
        raise ValueError("Recall evaluation requires at least one case")
    if k <= 0:
        raise ValueError("k must be positive")
    recalls: list[float] = []
    for case in cases:
        if not case.expected_chunk_ids:
            raise ValueError("Each recall case needs at least one expected chunk")
        retrieved = _top_chunk_ids(search(case.query, k), k)
        recalls.append(len(retrieved & case.expected_chunk_ids) / len(case.expected_chunk_ids))
    return sum(recalls) / len(recalls)


def evidence_recall_at_k(
    cases: Sequence[RecallCase],
    search: Callable[[str, int], Sequence[FusedHit]],
    *,
    k: int,
) -> float:
    """Recall exact child evidence even when retrieval returns expanded parents."""

    if not cases:
        raise ValueError("Evidence recall evaluation requires at least one case")
    if k <= 0:
        raise ValueError("k must be positive")
    recalls: list[float] = []
    for case in cases:
        if not case.expected_chunk_ids:
            raise ValueError("Each evidence recall case needs at least one expected chunk")
        retrieved: set[str] = set()
        for hit in search(case.query, k)[:k]:
            retrieved.add(hit.chunk_id)
            retrieved.update(hit.matched_child_ids)
        recalls.append(len(retrieved & case.expected_chunk_ids) / len(case.expected_chunk_ids))
    return sum(recalls) / len(recalls)


def precision_at_k(
    cases: Sequence[RecallCase],
    search: Callable[[str, int], Sequence[SearchHit | FusedHit]],
    *,
    k: int,
) -> float:
    """Compute macro precision: of the top-k retrieved, how many were relevant."""

    if not cases:
        raise ValueError("Precision evaluation requires at least one case")
    if k <= 0:
        raise ValueError("k must be positive")
    precisions: list[float] = []
    for case in cases:
        if not case.expected_chunk_ids:
            raise ValueError("Each precision case needs at least one expected chunk")
        retrieved = _top_chunk_ids(search(case.query, k), k)
        if not retrieved:
            precisions.append(0.0)
            continue
        precisions.append(len(retrieved & case.expected_chunk_ids) / len(retrieved))
    return sum(precisions) / len(precisions)


def parent_hit_rate(
    cases: Sequence[RecallCase],
    search: Callable[[str, int], Sequence[FusedHit]],
    *,
    k: int,
    parent_map: dict[str, str],
) -> float:
    """Fraction of cases whose top-k contains the parent of an expected child chunk.

    Useful when the retriever returns parent blocks but relevance is labelled on the
    original child chunks: a parent hit still satisfies the user's information need.
    """

    if not cases:
        raise ValueError("Parent-hit evaluation requires at least one case")
    if k <= 0:
        raise ValueError("k must be positive")
    rates: list[float] = []
    for case in cases:
        if not case.expected_chunk_ids:
            raise ValueError("Each parent-hit case needs at least one expected chunk")
        expected_parents = {parent_map.get(cid, cid) for cid in case.expected_chunk_ids}
        hits = search(case.query, k)[:k]
        retrieved = _top_chunk_ids(hits, k)
        traced_children = {chunk_id for hit in hits for chunk_id in hit.matched_child_ids}
        if retrieved & case.expected_chunk_ids or traced_children & case.expected_chunk_ids:
            rates.append(1.0)
        elif retrieved & expected_parents:
            rates.append(1.0)
        else:
            rates.append(0.0)
    return sum(rates) / len(rates)
