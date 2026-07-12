"""Thread-safe SQLite access for metadata, API logs, and local cache."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from writing_factory.store.migrations import apply_migrations


def utc_now() -> str:
    """Return a sortable UTC timestamp."""

    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class ApiCallRecord:
    """Privacy-preserving metadata for one external service call."""

    call_id: str
    request_hash: str
    provider: str
    operation: str
    model: str | None
    reasoning_effort: str | None
    prompt_summary: str
    cache_hit: bool
    status: str
    duration_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    result_summary: str | None = None
    error_type: str | None = None
    created_at: str = ""


class Database:
    """Open short-lived SQLite connections so QThreads never share a handle."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        """Create the database and bring its schema to the current version."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            apply_migrations(connection)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection with atomic commit or rollback."""

        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_cached_response(self, cache_key: str) -> dict[str, Any] | None:
        """Return a non-expired response without exposing it to ordinary logs."""

        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT response_json FROM api_cache
                WHERE cache_key = ?
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (cache_key, utc_now()),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["response_json"])

    def set_cached_response(
        self,
        cache_key: str,
        provider: str,
        operation: str,
        response: Mapping[str, Any],
        expires_at: str | None = None,
    ) -> None:
        """Upsert a full response in the ignored local cache database."""

        payload = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO api_cache(
                    cache_key, provider, operation, response_json, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response_json = excluded.response_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (cache_key, provider, operation, payload, utc_now(), expires_at),
            )

    def record_api_call(self, record: ApiCallRecord) -> None:
        """Persist call metadata without raw prompts, results, or credentials."""

        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO api_calls(
                    call_id, request_hash, provider, operation, model,
                    reasoning_effort, prompt_summary, cache_hit, status,
                    input_tokens, output_tokens, total_tokens, duration_ms,
                    result_summary, error_type, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.call_id,
                    record.request_hash,
                    record.provider,
                    record.operation,
                    record.model,
                    record.reasoning_effort,
                    record.prompt_summary,
                    int(record.cache_hit),
                    record.status,
                    record.input_tokens,
                    record.output_tokens,
                    record.total_tokens,
                    record.duration_ms,
                    record.result_summary,
                    record.error_type,
                    record.created_at or utc_now(),
                ),
            )
