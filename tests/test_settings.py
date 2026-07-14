"""Configuration and credential safety tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from writing_factory.config import load_settings


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


def test_requires_both_credentials(tmp_path: Path) -> None:
    (tmp_path / "key_test.txt").write_text("only-one\n", encoding="utf-8")

    with pytest.raises(ValueError, match="MINERU_API_TOKEN"):
        load_settings(tmp_path, environ={})


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
