"""Explicit opt-in live contract test for the Bocha Web Search API."""

from __future__ import annotations

import os

import pytest

from writing_factory.config import load_settings
from writing_factory.llm.bocha import BochaClient
from writing_factory.store import Database


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_BOCHA_TESTS") != "1",
    reason="set RUN_LIVE_BOCHA_TESTS=1 to spend one small live web-search request",
)
def test_bocha_live_search_returns_traceable_url(tmp_path) -> None:
    settings = load_settings()
    database = Database(tmp_path / "bocha-live.db")
    database.initialize()
    client = BochaClient(settings, database)
    try:
        result = client.search("博查 AI 开放平台", count=1)
    finally:
        client.close()

    assert result.pages
    assert result.pages[0].url.startswith(("https://", "http://"))
