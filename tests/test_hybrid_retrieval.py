"""Unit tests for the hybrid retrieval building blocks.

These tests target the deterministic, IO-free pieces of the retrieval pipeline
(RRF fusion, metadata filtering, parent expansion) so they run without a database
or any network access. The LLM-backed stages (rewrite/HyDE/rerank) are covered
separately with mocked clients.
"""

from __future__ import annotations

from writing_factory.kb.fusion import fuse_many, rrf_fuse
from writing_factory.kb.models import (
    Chunk,
    FusedHit,
    MetadataFilter,
    SearchHit,
)
from writing_factory.kb.parent_retriever import ParentExpander
from writing_factory.kb.rewrite import QueryExpander


def _search_hit(
    chunk_id: str,
    rank: int,
    *,
    doc_id: str = "doc",
    source: str = "bm25",
    parent_id: str | None = None,
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=f"text-{chunk_id}",
        score=1.0,
        rank=rank,
        source=source,  # type: ignore[arg-type]
        parent_id=parent_id,
    )


def _fused_hit(
    chunk_id: str,
    rrf_score: float,
    *,
    parent_id: str | None = None,
    doc_id: str = "doc",
) -> FusedHit:
    return FusedHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=f"text-{chunk_id}",
        source="hybrid",
        rrf_score=rrf_score,
        final_rank=0,
        parent_id=parent_id,
    )


def test_rrf_fuse_combines_dense_and_sparse() -> None:
    dense = [_search_hit("a", 1, source="dense"), _search_hit("b", 2, source="dense")]
    sparse = [_search_hit("b", 1, source="bm25"), _search_hit("c", 2, source="bm25")]

    fused = rrf_fuse(dense, sparse)

    ids = [hit.chunk_id for hit in fused]
    assert ids == ["b", "a", "c"]
    # "b" appears in both lists, so it must outrank the single-list hits.
    assert fused[0].chunk_id == "b"
    assert fused[0].source == "hybrid"
    # "b" is rank 2 in dense (a is rank 1) and rank 1 in sparse.
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 1


def test_rrf_fuse_marks_single_source_origin() -> None:
    dense = [_search_hit("a", 1, source="dense")]
    sparse: list[SearchHit] = []

    fused = rrf_fuse(dense, sparse)

    assert len(fused) == 1
    assert fused[0].source == "dense"
    assert fused[0].dense_rank == 1
    assert fused[0].sparse_rank is None


def test_rrf_fuse_assigns_final_rank() -> None:
    dense = [_search_hit("a", 1), _search_hit("b", 2)]
    sparse = [_search_hit("c", 1)]

    fused = rrf_fuse(dense, sparse)

    assert [hit.final_rank for hit in fused] == [1, 2, 3]


def test_rrf_fuse_respects_limit() -> None:
    dense = [_search_hit(chr(97 + i), i + 1) for i in range(5)]
    sparse: list[SearchHit] = []

    fused = rrf_fuse(dense, sparse, limit=2)

    assert len(fused) == 2
    assert [hit.chunk_id for hit in fused] == ["a", "b"]


def test_rrf_fuse_deterministic_scores() -> None:
    dense = [_search_hit("a", 1), _search_hit("b", 2)]
    sparse = [_search_hit("b", 1), _search_hit("c", 2)]

    first = rrf_fuse(dense, sparse)
    second = rrf_fuse(dense, sparse)

    assert [h.rrf_score for h in first] == [h.rrf_score for h in second]


def test_multi_query_rrf_accumulates_every_ranked_list() -> None:
    repeated = _search_hit("repeated", 1, source="dense")
    one_time = _search_hit("one-time", 1, source="dense")

    fused = fuse_many(([repeated], [repeated], [repeated], [one_time]))

    assert [hit.chunk_id for hit in fused] == ["repeated", "one-time"]
    assert fused[0].rrf_score == 3 / 61
    assert fused[1].rrf_score == 1 / 61


def test_metadata_filter_defaults_empty() -> None:
    filt = MetadataFilter()

    assert filt.doc_ids is None
    assert filt.authors is None
    assert filt.years is None
    assert filt.formats is None
    assert filt.section_headings is None
    assert filt.chunk_kinds is None


def test_metadata_filter_accepts_constraints() -> None:
    filt = MetadataFilter(
        doc_ids={"d1", "d2"},
        authors={"Smith"},
        years={2020},
        formats={"pdf"},
        section_headings={"引言"},
        chunk_kinds={"child"},
    )

    assert filt.doc_ids == {"d1", "d2"}
    assert filt.authors == {"Smith"}
    assert filt.years == {2020}
    assert filt.chunk_kinds == {"child"}


def test_query_rewrite_rejects_non_list_json_payload() -> None:
    assert QueryExpander._parse_queries('{"queries":"不是列表"}') == []


class _FakeRepository:
    """Minimal stand-in exposing only what ParentExpander calls."""

    def __init__(self, parents: dict[str, Chunk]) -> None:
        self._parents = parents

    def parent_chunk(self, parent_id: str) -> Chunk | None:
        return self._parents.get(parent_id)


def _chunk(chunk_id: str, text: str, *, parent_id: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        text=text,
        page_start=1,
        page_end=2,
        section_heading="引言",
        chunk_index=0,
        char_start=0,
        char_end=len(text),
        parent_id=parent_id,
        chunk_kind="parent" if parent_id is None else "child",
    )


def test_parent_expander_returns_child_when_no_parent() -> None:
    repo = _FakeRepository({})
    expander = ParentExpander(repo)
    hits = [_fused_hit("child1", 0.5, parent_id=None)]

    expanded = expander.expand(hits)

    assert len(expanded) == 1
    assert expanded[0].chunk_id == "child1"
    assert expanded[0].expanded_from_child is False


def test_parent_expander_replaces_child_with_parent() -> None:
    parent = _chunk("parent1", "完整的父级章节文本")
    repo = _FakeRepository({"parent1": parent})
    expander = ParentExpander(repo)
    child_hit = _fused_hit("child1", 0.9, parent_id="parent1")

    expanded = expander.expand([child_hit])

    assert len(expanded) == 1
    assert expanded[0].chunk_id == "parent1"
    assert expanded[0].text == "完整的父级章节文本"
    assert expanded[0].expanded_from_child is True
    assert expanded[0].parent_id is None
    assert expanded[0].matched_child_ids == ("child1",)


def test_parent_expander_keeps_best_child_per_parent() -> None:
    parent = _chunk("parent1", "父级文本")
    repo = _FakeRepository({"parent1": parent})
    expander = ParentExpander(repo)
    hits = [
        _fused_hit("child-a", 0.3, parent_id="parent1"),
        _fused_hit("child-b", 0.8, parent_id="parent1"),
    ]

    expanded = expander.expand(hits)

    assert len(expanded) == 1
    assert expanded[0].chunk_id == "parent1"
    assert expanded[0].rrf_score == 0.8
    assert expanded[0].matched_child_ids == ("child-b", "child-a")


def test_parent_expander_falls_back_when_parent_missing() -> None:
    repo = _FakeRepository({})
    expander = ParentExpander(repo)
    hits = [_fused_hit("child1", 0.4, parent_id="ghost")]

    expanded = expander.expand(hits)

    assert len(expanded) == 1
    assert expanded[0].chunk_id == "child1"
    assert expanded[0].expanded_from_child is False
