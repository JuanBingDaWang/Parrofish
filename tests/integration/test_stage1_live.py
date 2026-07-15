"""Opt-in real MinerU, embedding, LanceDB, and BM25 ingestion test."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from writing_factory.app import build_application
from writing_factory.config import load_settings


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_INGEST_TESTS") != "1",
    reason="set RUN_LIVE_INGEST_TESTS=1 with LIVE_INGEST_FILE and LIVE_INGEST_QUERY",
)
def test_real_document_ingestion_and_retrieval(tmp_path: Path) -> None:
    source_value = os.environ.get("LIVE_INGEST_FILE")
    query = os.environ.get("LIVE_INGEST_QUERY")
    if not source_value or not query:
        pytest.skip("LIVE_INGEST_FILE and LIVE_INGEST_QUERY are required")
    source = Path(source_value).expanduser().resolve(strict=True)
    settings = load_settings().model_copy(
        update={
            "data_dir": tmp_path,
            "database_path": tmp_path / "live.db",
            "lancedb_path": tmp_path / "lancedb",
            "managed_documents_dir": tmp_path / "documents",
            "mineru_artifacts_dir": tmp_path / "mineru",
            "log_dir": tmp_path / "logs",
        }
    )
    context = build_application(settings)
    try:
        result = context.ingestion.ingest(context.default_kb_id, source)
        sparse = context.sparse_retriever.search(context.default_kb_id, query, limit=5)
        dense = context.dense_retriever.search(context.default_kb_id, query, limit=5)
    finally:
        context.close()

    assert result.child_chunk_count > 0
    assert sparse
    assert dense
