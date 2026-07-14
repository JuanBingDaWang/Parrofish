"""Opt-in live smoke test for the configured SiliconFlow account."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from writing_factory.config import load_settings
from writing_factory.llm import SiliconFlowClient
from writing_factory.store import Database


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_API_TESTS") != "1",
    reason="set RUN_LIVE_API_TESTS=1 to spend a small live request",
)
def test_chat_connection(tmp_path: Path) -> None:
    settings = load_settings().model_copy(
        update={
            "data_dir": tmp_path,
            "database_path": tmp_path / "live.db",
            "log_dir": tmp_path / "logs",
        }
    )
    database = Database(settings.database_path)
    database.initialize()
    client = SiliconFlowClient(settings, database)
    try:
        result = client.chat(
            [
                {"role": "system", "content": "Reply with only OK."},
                {"role": "user", "content": "Connection check."},
            ],
            thinking=False,
            temperature=0.0,
            max_tokens=8,
            seed=7,
            use_cache=False,
            stream=True,
        )
    finally:
        client.close()

    assert result.content.strip()
    assert result.model
    assert result.finish_reason == "stop"
