"""Retrieval recall metric tests."""

from __future__ import annotations

from writing_factory.eval import RecallCase, recall_at_k
from writing_factory.kb.models import SearchHit


def _hit(chunk_id: str, rank: int) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        doc_id="doc",
        text="text",
        score=1.0,
        rank=rank,
        source="bm25",
    )


def test_recall_at_k_uses_expected_chunk_ids() -> None:
    cases = [
        RecallCase(query="甲", expected_chunk_ids=frozenset({"a"})),
        RecallCase(query="乙", expected_chunk_ids=frozenset({"b", "c"})),
    ]

    score = recall_at_k(
        cases,
        lambda query, _k: [_hit("a", 1)] if query == "甲" else [_hit("b", 1)],
        k=2,
    )

    assert score == 0.75
