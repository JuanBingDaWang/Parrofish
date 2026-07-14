"""Application assembly smoke test."""

from __future__ import annotations

import weakref

from writing_factory.app import build_application


def test_builds_and_closes_application_context(settings) -> None:
    context = build_application(settings)
    try:
        assert settings.database_path.is_file()
        assert (settings.log_dir / "writing_factory.jsonl").is_file()
        assert weakref.ref(context)() is context
        assert context.get_framework_generation_timeout() == 900
        context.set_framework_generation_timeout(1200)
        assert context.get_framework_generation_timeout() == 1200
    finally:
        context.close()
