"""Load application settings without exposing local secrets."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from writing_factory.config.secrets import (
    BOCHA_CREDENTIAL,
    MINERU_CREDENTIAL,
    SILICONFLOW_CREDENTIAL,
    CredentialStore,
    KeyringCredentialStore,
)

DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MINERU_BASE_URL = "https://mineru.net/api/v4"
DEFAULT_BOCHA_BASE_URL = "https://api.bochaai.com/v1"
CredentialSource = Literal["credential_store", "environment", "key_test", "missing"]


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
    bocha_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    siliconflow_credential_source: CredentialSource = "missing"
    mineru_credential_source: CredentialSource = "missing"
    bocha_credential_source: CredentialSource = "missing"
    siliconflow_base_url: str = DEFAULT_SILICONFLOW_BASE_URL
    mineru_base_url: str = DEFAULT_MINERU_BASE_URL
    bocha_base_url: str = DEFAULT_BOCHA_BASE_URL
    chat_model: str = "deepseek-ai/DeepSeek-V4-Flash"
    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    citation_style: Literal["gb-t-7714"] = "gb-t-7714"
    distillation_output_language: Literal["zh-CN"] = "zh-CN"
    siliconflow_max_concurrency: int = Field(default=3, ge=1, le=8)
    siliconflow_request_timeout_seconds: int = Field(default=900, ge=60, le=3600)
    siliconflow_total_timeout_seconds: int = Field(default=1800, ge=60, le=21600)
    siliconflow_stream_idle_timeout_seconds: int = Field(default=180, ge=30, le=1800)
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 180.0
    max_retries: int = 3
    min_request_interval_seconds: float = 0.0
    mineru_poll_interval_seconds: float = 3.0
    mineru_timeout_seconds: float = 600.0
    embedding_batch_size: int = 32

    @field_validator("siliconflow_base_url", "mineru_base_url", "bocha_base_url")
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


def _default_project_root(environ: Mapping[str, str]) -> Path:
    if getattr(sys, "frozen", False):
        if os.name == "nt":
            base = Path(
                environ.get("LOCALAPPDATA")
                or Path.home() / "AppData" / "Local"
            )
        else:
            base = Path(environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
        return base / "Parrofish"
    return Path(__file__).resolve().parents[3]


def _read_local_keys(key_file: Path) -> tuple[str | None, str | None, str | None]:
    if not key_file.is_file():
        return None, None, None

    lines = [
        line.strip()
        for line in key_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    siliconflow_key = lines[0] if lines else None
    mineru_token = lines[1] if len(lines) > 1 else None
    bocha_key = lines[2] if len(lines) > 2 else None
    return siliconflow_key, mineru_token, bocha_key


def _resolve_secret(
    *,
    vault_value: str | None,
    environment_value: str | None,
    file_value: str | None,
) -> tuple[SecretStr, CredentialSource]:
    for value, source in (
        (vault_value, "credential_store"),
        (environment_value, "environment"),
        (file_value, "key_test"),
    ):
        if value and value.strip():
            return SecretStr(value.strip()), source
    return SecretStr(""), "missing"


def load_settings(
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    credential_store: CredentialStore | None = None,
) -> Settings:
    """Load OS-vault, environment, and ignored-file configuration in that order."""

    env = os.environ if environ is None else environ
    root = (project_root or _default_project_root(env)).resolve()
    file_siliconflow_key, file_mineru_token, file_bocha_key = _read_local_keys(
        root / "key_test.txt"
    )
    store = credential_store
    if store is None and environ is None:
        store = KeyringCredentialStore()
    try:
        stored_siliconflow_key = store.get(SILICONFLOW_CREDENTIAL) if store else None
        stored_mineru_token = store.get(MINERU_CREDENTIAL) if store else None
        stored_bocha_key = store.get(BOCHA_CREDENTIAL) if store else None
    except Exception:
        stored_siliconflow_key = None
        stored_mineru_token = None
        stored_bocha_key = None
    siliconflow_key, siliconflow_source = _resolve_secret(
        vault_value=stored_siliconflow_key,
        environment_value=env.get("SILICONFLOW_API_KEY"),
        file_value=file_siliconflow_key,
    )
    mineru_token, mineru_source = _resolve_secret(
        vault_value=stored_mineru_token,
        environment_value=env.get("MINERU_API_TOKEN"),
        file_value=file_mineru_token,
    )
    bocha_key, bocha_source = _resolve_secret(
        vault_value=stored_bocha_key,
        environment_value=env.get("BOCHA_API_KEY"),
        file_value=file_bocha_key,
    )

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
        siliconflow_api_key=siliconflow_key,
        mineru_api_token=mineru_token,
        bocha_api_key=bocha_key,
        siliconflow_credential_source=siliconflow_source,
        mineru_credential_source=mineru_source,
        bocha_credential_source=bocha_source,
        siliconflow_base_url=env.get("SILICONFLOW_BASE_URL", DEFAULT_SILICONFLOW_BASE_URL),
        mineru_base_url=env.get("MINERU_BASE_URL", DEFAULT_MINERU_BASE_URL),
        bocha_base_url=env.get("BOCHA_BASE_URL", DEFAULT_BOCHA_BASE_URL),
        chat_model=env.get("SILICONFLOW_CHAT_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
        embedding_model=env.get("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3"),
        rerank_model=env.get("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        distillation_output_language=env.get("DISTILLATION_OUTPUT_LANGUAGE", "zh-CN"),
        siliconflow_max_concurrency=int(
            env.get(
                "SILICONFLOW_MAX_CONCURRENCY",
                env.get("DISTILLATION_MAP_CONCURRENCY", "3"),
            )
        ),
        siliconflow_request_timeout_seconds=int(
            env.get(
                "SILICONFLOW_REQUEST_TIMEOUT_SECONDS",
                env.get("FRAMEWORK_GENERATION_TIMEOUT_SECONDS", "900"),
            )
        ),
        siliconflow_total_timeout_seconds=int(
            env.get("SILICONFLOW_TOTAL_TIMEOUT_SECONDS", "1800")
        ),
        siliconflow_stream_idle_timeout_seconds=int(
            env.get("SILICONFLOW_STREAM_IDLE_TIMEOUT_SECONDS", "180")
        ),
    )
