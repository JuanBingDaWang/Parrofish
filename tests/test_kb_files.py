"""Managed source-file storage tests."""

from __future__ import annotations

from pathlib import Path

from writing_factory.kb.files import ManagedFileStore


def test_imports_content_addressed_copy_and_deduplicates(tmp_path: Path) -> None:
    source = tmp_path / "论文.txt"
    source.write_text("中文史料", encoding="utf-8")
    store = ManagedFileStore(tmp_path / "managed")

    first = store.import_file(source)
    second = store.import_file(source)

    assert first.doc_id == second.doc_id
    assert first.managed_path == second.managed_path
    assert first.managed_path.read_text(encoding="utf-8") == "中文史料"
    assert first.source_path == source.resolve()
