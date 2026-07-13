"""Retrieval recall metric tests."""

from __future__ import annotations

from writing_factory.eval import RecallCase, evidence_recall_at_k, recall_at_k
from writing_factory.kb.models import FusedHit, SearchHit


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


def test_evidence_recall_uses_child_provenance_from_expanded_parent() -> None:
    cases = [RecallCase(query="甲", expected_chunk_ids=frozenset({"child"}))]
    parent = FusedHit(
        chunk_id="parent",
        doc_id="doc",
        text="父级上下文",
        source="hybrid",
        final_rank=1,
        expanded_from_child=True,
        matched_child_ids=("child",),
    )

    score = evidence_recall_at_k(cases, lambda _query, _k: [parent], k=1)

    assert score == 1.0
