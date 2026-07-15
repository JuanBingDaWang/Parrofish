"""Opt-in real-corpus SiliconFlow regression for the complete stage-3 chain."""

from __future__ import annotations

import os

import pytest

from writing_factory.app import build_application
from writing_factory.kb.models import RetrievalRequest


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_RETRIEVAL_TESTS") != "1",
    reason="set RUN_LIVE_RETRIEVAL_TESTS=1 with LIVE_RETRIEVAL_QUERY",
)
def test_real_hybrid_retrieval_has_traceable_evidence() -> None:
    query = os.environ.get("LIVE_RETRIEVAL_QUERY", "").strip()
    expected_filename = os.environ.get("LIVE_RETRIEVAL_EXPECTED_FILENAME", "").strip()
    if not query:
        pytest.skip("LIVE_RETRIEVAL_QUERY is required")

    context = build_application()
    try:
        documents = {
            str(item["doc_id"]): str(item["filename"])
            for item in context.repository.list_documents(context.default_kb_id)
        }
        result = context.hybrid_retriever.search(
            RetrievalRequest(kb_id=context.default_kb_id, query=query)
        )
    finally:
        context.close()

    assert result.hits
    assert all(hit.matched_child_ids for hit in result.hits)
    if expected_filename:
        assert any(
            expected_filename.casefold() in documents.get(hit.doc_id, "").casefold()
            for hit in result.hits
        )
