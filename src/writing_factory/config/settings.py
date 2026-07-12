"""Load application settings without exposing local secrets."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MINERU_BASE_URL = "https://mineru.net/api/v4"


class Settings(BaseModel):
    """Validated configuration shared by every application module."""

    model_config = ConfigDict(frozen=True)

    project_root: Path
    data_dir: Path
    database_path: Path
    lancedb_path: Path
    managed_documents_dir: Path
    mineru_artifacts_dir: Path
    log_dir: Path
    siliconflow_api_key: SecretStr = Field(repr=False)
    mineru_api_token: SecretStr = Field(repr=False)
    siliconflow_base_url: str = DEFAULT_SILICONFLOW_BASE_URL
    mineru_base_url: str = DEFAULT_MINERU_BASE_URL
    chat_model: str = "deepseek-ai/DeepSeek-V4-Flash"
    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    citation_style: Literal["gb-t-7714"] = "gb-t-7714"
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 180.0
    max_retries: int = 3
    min_request_interval_seconds: float = 0.0
    mineru_poll_interval_seconds: float = 3.0
    mineru_timeout_seconds: float = 600.0
    embedding_batch_size: int = 32

    @field_validator("siliconflow_base_url", "mineru_base_url")
    @classmethod
    def strip_url_suffix(cls, value: str) -> str:
        """Keep endpoint composition deterministic."""

        return value.rstrip("/")

    def ensure_runtime_directories(self) -> None:
        """Create directories that hold ignored, local runtime state."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.lancedb_path.mkdir(parents=True, exist_ok=True)
        self.managed_documents_dir.mkdir(parents=True, exist_ok=True)
        self.mineru_artifacts_dir.mkdir(parents=True, exist_ok=True)


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_local_keys(key_file: Path) -> tuple[str | None, str | None]:
    if not key_file.is_file():
        return None, None

    lines = [
        line.strip()
        for line in key_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    siliconflow_key = lines[0] if lines else None
    mineru_token = lines[1] if len(lines) > 1 else None
    return siliconflow_key, mineru_token


def _required_secret(name: str, value: str | None) -> SecretStr:
    if value is None or not value.strip():
        raise ValueError(f"Missing required secret: {name}")
    return SecretStr(value.strip())


def load_settings(
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load environment overrides, falling back to the ignored test key file."""

    root = (project_root or _default_project_root()).resolve()
    env = os.environ if environ is None else environ
    file_siliconflow_key, file_mineru_token = _read_local_keys(root / "key_test.txt")

    data_dir = Path(env.get("WRITING_FACTORY_DATA_DIR", root / "data"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    log_dir = Path(env.get("WRITING_FACTORY_LOG_DIR", root / "logs"))
    if not log_dir.is_absolute():
        log_dir = root / log_dir

    return Settings(
        project_root=root,
        data_dir=data_dir.resolve(),
        database_path=(data_dir / "writing_factory.db").resolve(),
        lancedb_path=(data_dir / "lancedb").resolve(),
        managed_documents_dir=(data_dir / "documents").resolve(),
        mineru_artifacts_dir=(data_dir / "mineru").resolve(),
        log_dir=log_dir.resolve(),
        siliconflow_api_key=_required_secret(
            "SILICONFLOW_API_KEY",
            env.get("SILICONFLOW_API_KEY", file_siliconflow_key),
        ),
        mineru_api_token=_required_secret(
            "MINERU_API_TOKEN",
            env.get("MINERU_API_TOKEN", file_mineru_token),
        ),
        siliconflow_base_url=env.get("SILICONFLOW_BASE_URL", DEFAULT_SILICONFLOW_BASE_URL),
        mineru_base_url=env.get("MINERU_BASE_URL", DEFAULT_MINERU_BASE_URL),
        chat_model=env.get("SILICONFLOW_CHAT_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
        embedding_model=env.get("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3"),
        rerank_model=env.get("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
    )
