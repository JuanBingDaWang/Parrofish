"""Configuration and credential safety tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from writing_factory.config import load_settings
from writing_factory.config import settings as settings_module


class MemoryCredentialStore:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})

    def get(self, name: str) -> str | None:
        return self.values.get(name)

    def set(self, name: str, value: str) -> None:
        self.values[name] = value

    def delete(self, name: str) -> None:
        self.values.pop(name, None)


def test_loads_two_raw_keys_without_exposing_them(tmp_path: Path) -> None:
    key_file = tmp_path / "key_test.txt"
    key_file.write_text("silicon-secret\nmineru-secret\n", encoding="utf-8")

    settings = load_settings(tmp_path, environ={})

    assert settings.siliconflow_api_key.get_secret_value() == "silicon-secret"
    assert settings.mineru_api_token.get_secret_value() == "mineru-secret"
    assert "silicon-secret" not in repr(settings)
    assert "mineru-secret" not in repr(settings)


def test_environment_overrides_local_key_file(tmp_path: Path) -> None:
    (tmp_path / "key_test.txt").write_text("file-one\nfile-two\n", encoding="utf-8")

    settings = load_settings(
        tmp_path,
        environ={
            "SILICONFLOW_API_KEY": "environment-one",
            "MINERU_API_TOKEN": "environment-two",
        },
    )

    assert settings.siliconflow_api_key.get_secret_value() == "environment-one"
    assert settings.mineru_api_token.get_secret_value() == "environment-two"


def test_os_credential_store_has_highest_priority(tmp_path: Path) -> None:
    (tmp_path / "key_test.txt").write_text("file-one\nfile-two\n", encoding="utf-8")
    store = MemoryCredentialStore(
        {
            "siliconflow-api-key": "vault-one",
            "mineru-api-token": "vault-two",
        }
    )

    settings = load_settings(
        tmp_path,
        environ={
            "SILICONFLOW_API_KEY": "environment-one",
            "MINERU_API_TOKEN": "environment-two",
        },
        credential_store=store,
    )

    assert settings.siliconflow_api_key.get_secret_value() == "vault-one"
    assert settings.mineru_api_token.get_secret_value() == "vault-two"
    assert settings.siliconflow_credential_source == "credential_store"
    assert settings.mineru_credential_source == "credential_store"


def test_missing_credential_keeps_ui_startable(tmp_path: Path) -> None:
    (tmp_path / "key_test.txt").write_text("only-one\n", encoding="utf-8")

    settings = load_settings(tmp_path, environ={})

    assert settings.siliconflow_credential_source == "key_test"
    assert settings.mineru_credential_source == "missing"
    assert settings.mineru_api_token.get_secret_value() == ""


def test_third_local_key_line_loads_bocha_without_exposing_it(tmp_path: Path) -> None:
    (tmp_path / "key_test.txt").write_text(
        "siliconflow-local\nmineru-local\nbocha-local\n",
        encoding="utf-8",
    )

    settings = load_settings(tmp_path, environ={})

    assert settings.bocha_api_key.get_secret_value() == "bocha-local"
    assert settings.bocha_credential_source == "key_test"
    assert "bocha-local" not in repr(settings)


def test_frozen_windows_app_uses_local_app_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(settings_module.os, "name", "nt")

    settings = load_settings(environ={"LOCALAPPDATA": str(tmp_path)})

    assert settings.project_root == tmp_path / "Parrofish"
    assert settings.data_dir == tmp_path / "Parrofish" / "data"
    assert settings.log_dir == tmp_path / "Parrofish" / "logs"
    assert settings.siliconflow_credential_source == "missing"
    assert settings.mineru_credential_source == "missing"


def test_distillation_defaults_to_simplified_chinese_and_three_map_workers(
    tmp_path: Path,
) -> None:
    (tmp_path / "key_test.txt").write_text("one\ntwo\n", encoding="utf-8")

    settings = load_settings(tmp_path, environ={})

    assert settings.distillation_output_language == "zh-CN"
    assert settings.siliconflow_max_concurrency == 3
    assert settings.siliconflow_request_timeout_seconds == 900


def test_siliconflow_request_timeout_can_be_configured(tmp_path: Path) -> None:
    (tmp_path / "key_test.txt").write_text("one\ntwo\n", encoding="utf-8")

    settings = load_settings(
        tmp_path,
        environ={"SILICONFLOW_REQUEST_TIMEOUT_SECONDS": "1200"},
    )

    assert settings.siliconflow_request_timeout_seconds == 1200


def test_legacy_framework_timeout_environment_name_is_still_read(tmp_path: Path) -> None:
    (tmp_path / "key_test.txt").write_text("one\ntwo\n", encoding="utf-8")

    settings = load_settings(
        tmp_path,
        environ={"FRAMEWORK_GENERATION_TIMEOUT_SECONDS": "1200"},
    )

    assert settings.siliconflow_request_timeout_seconds == 1200


def test_siliconflow_rejects_more_than_eight_concurrent_requests(tmp_path: Path) -> None:
    (tmp_path / "key_test.txt").write_text("one\ntwo\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="less than or equal to 8"):
        load_settings(
            tmp_path,
            environ={"SILICONFLOW_MAX_CONCURRENCY": "9"},
        )
