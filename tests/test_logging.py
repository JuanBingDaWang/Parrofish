"""Local log redaction tests."""

from __future__ import annotations

import logging
from pathlib import Path

from writing_factory.config.logging import configure_logging, register_secret


def test_redacts_credentials_from_json_log(tmp_path: Path) -> None:
    secret = "never-write-this-token"
    configure_logging(tmp_path, (secret,))

    logger = logging.getLogger("test")
    logger.warning("credential=%s", secret)
    try:
        raise RuntimeError(f"failed with {secret}")
    except RuntimeError:
        logger.exception("provider failure")
    logging.shutdown()

    content = (tmp_path / "writing_factory.jsonl").read_text(encoding="utf-8")
    assert secret not in content
    assert "[REDACTED]" in content


def test_redacts_credentials_registered_after_startup(tmp_path: Path) -> None:
    secret = "runtime-secret-token"
    configure_logging(tmp_path, ())
    register_secret(secret)

    logging.getLogger("test").warning("updated credential=%s", secret)
    logging.shutdown()

    content = (tmp_path / "writing_factory.jsonl").read_text(encoding="utf-8")
    assert secret not in content
    assert "[REDACTED]" in content
