"""Configuration and credential safety tests."""

from __future__ import annotations

from pathlib import Path

import pytest

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
