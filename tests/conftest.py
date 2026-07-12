"""Shared isolated configuration for unit and Qt tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import SecretStr

from writing_factory.config import Settings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Return settings that never touch repository runtime data."""

    data_dir = tmp_path / "data"
    return Settings(
        project_root=tmp_path,
        data_dir=data_dir,
        database_path=data_dir / "test.db",
        lancedb_path=data_dir / "lancedb",
        managed_documents_dir=data_dir / "documents",
        mineru_artifacts_dir=data_dir / "mineru",
        log_dir=tmp_path / "logs",
        siliconflow_api_key=SecretStr("sf-test-secret"),
        mineru_api_token=SecretStr("mineru-test-secret"),
        max_retries=2,
    )
