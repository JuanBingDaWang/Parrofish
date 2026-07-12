"""SQLite migration, cache, and privacy tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from writing_factory.store import ApiCallRecord, Database


def test_initializes_expected_stage_zero_tables(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.initialize()

    with database.connection() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert {
        "knowledge_bases",
        "documents",
        "chunks",
        "ingest_jobs",
        "api_cache",
        "api_calls",
    }.issubset(tables)
    assert journal_mode.lower() == "wal"


def test_cache_round_trip_and_api_record_have_no_payload(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.initialize()
    response = {"choices": [{"message": {"content": "private-result"}}]}
    database.set_cached_response("hash", "test", "chat", response)
    database.record_api_call(
        ApiCallRecord(
            call_id="call",
            request_hash="hash",
            provider="test",
            operation="chat",
            model="model",
            reasoning_effort="disabled",
            prompt_summary='{"character_count": 15}',
            cache_hit=False,
            status="success",
            duration_ms=10,
            result_summary='{"content_chars": 14}',
        )
    )

    assert database.get_cached_response("hash") == response
    with sqlite3.connect(database.path) as connection:
        record = connection.execute(
            "SELECT prompt_summary, result_summary FROM api_calls"
        ).fetchone()
    assert "private-result" not in "".join(record)
