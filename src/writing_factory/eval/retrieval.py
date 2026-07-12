"""Recall-at-k evaluation for traceable retrieval regression cases."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from writing_factory.kb.models import SearchHit


@dataclass(frozen=True, slots=True)
class RecallCase:
    """One query and the chunk identifiers considered relevant."""

    query: str
    expected_chunk_ids: frozenset[str]


def recall_at_k(
    cases: Sequence[RecallCase],
    search: Callable[[str, int], Sequence[SearchHit]],
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
        retrieved = {hit.chunk_id for hit in search(case.query, k)[:k]}
        recalls.append(len(retrieved & case.expected_chunk_ids) / len(case.expected_chunk_ids))
    return sum(recalls) / len(recalls)
