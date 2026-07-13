"""Stable source grouping and checkpoint hash tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from writing_factory.distill.sources import SourceCorpusBuilder


class FakeSourceRepository:
    def source_documents(self, kb_id, *, doc_ids=None):
        return [
            {
                "doc_id": "doc_a",
                "title": "A",
                "filename": "a.txt",
                "chunk_count": 2,
            }
        ]

    def ready_parent_chunks(self, kb_id, *, doc_ids=None):
        return [
            SimpleNamespace(
                chunk_id=f"chunk_{index}",
                doc_id="doc_a",
                text=character * 8_000,
                page_start=index,
                page_end=index,
                section_heading=None,
            )
            for index, character in enumerate(("甲", "乙"), start=1)
        ]


def test_source_hash_includes_grouping_strategy() -> None:
    repository = FakeSourceRepository()

    compact = SourceCorpusBuilder(repository, max_unit_characters=12_000).build("kb")
    broad = SourceCorpusBuilder(repository, max_unit_characters=24_000).build("kb")

    assert len(compact.units) == 2
    assert len(broad.units) == 1
    assert compact.source_hash != broad.source_hash


def test_source_builder_rejects_an_explicit_empty_selection() -> None:
    with pytest.raises(ValueError, match="at least one selected"):
        SourceCorpusBuilder(FakeSourceRepository()).build("kb", doc_ids=set())
