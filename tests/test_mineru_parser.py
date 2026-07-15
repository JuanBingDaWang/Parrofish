"""MinerU polling, artifact caching, and parser routing tests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from writing_factory.kb.mineru_parser import DocumentParserRouter, MinerUDocumentParser
from writing_factory.kb.models import ManagedDocument
from writing_factory.llm.models import MinerUBatchUpload


class FakeMinerU:
    def __init__(self) -> None:
        self.polls = 0
        self.batch_creations = 0

    def create_batch_upload(self, filenames, *, model_version):
        self.batch_creations += 1
        return MinerUBatchUpload(batch_id="batch", file_urls=["https://upload.invalid"])

    def upload_file(self, upload_url: str, file_path: Path) -> None:
        assert file_path.is_file()

    def get_batch_result(self, batch_id: str):
        self.polls += 1
        state = "pending" if self.polls == 1 else "done"
        item = {"state": state}
        if state == "done":
            item["full_zip_url"] = "https://download.invalid/result.zip"
        return {"extract_result": [item]}

    def download_result(self, download_url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as bundle:
            bundle.writestr(
                "abc_content_list.json",
                json.dumps(
                    [
                        {
                            "type": "text",
                            "text": "数字史学",
                            "text_level": 1,
                            "page_idx": 0,
                        }
                    ],
                    ensure_ascii=False,
                ),
            )
        return destination


def _managed(tmp_path: Path, suffix: str = ".pdf") -> ManagedDocument:
    source = tmp_path / f"source{suffix}"
    source.write_text("fixture", encoding="utf-8")
    return ManagedDocument(
        doc_id="doc_fixture",
        sha256="a" * 64,
        filename=f"source{suffix}",
        format=suffix.removeprefix("."),
        source_path=source,
        managed_path=source,
    )


@pytest.mark.parametrize("suffix", [".pdf", ".docx", ".pptx"])
def test_polls_downloads_and_reuses_mineru_artifact(settings, tmp_path: Path, suffix: str) -> None:
    fake = FakeMinerU()
    parser = MinerUDocumentParser(settings, fake, sleep=lambda _seconds: None)
    progress: list[tuple[int, str]] = []

    first = parser.parse(
        _managed(tmp_path, suffix),
        progress=lambda value, text: progress.append((value, text)),
    )
    second = parser.parse(_managed(tmp_path, suffix))

    assert first.blocks[0].page == 1
    assert second.blocks[0].text == "数字史学"
    assert fake.batch_creations == 1
    assert progress[-1][0] == 48


def test_router_uses_utf8_fallback_for_txt(settings, tmp_path: Path) -> None:
    managed = _managed(tmp_path, ".txt")
    managed.managed_path.write_text("甲。\n\n乙。", encoding="utf-8")
    router = DocumentParserRouter(MinerUDocumentParser(settings, FakeMinerU()))

    parsed = router.parse(managed)

    assert [block.text for block in parsed.blocks] == ["甲。", "乙。"]
    assert parsed.filename == managed.filename
