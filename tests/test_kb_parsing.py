"""MinerU artifact adaptation and plain-text fallback tests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from writing_factory.kb.parsing import MinerUResultAdapter, TextParser


def test_adapts_mineru_pages_headings_and_tables(tmp_path: Path) -> None:
    archive = tmp_path / "result.zip"
    items = [
        {
            "type": "text",
            "text": "第一章",
            "text_level": 1,
            "page_idx": 0,
            "bbox": [1, 2, 3, 4],
        },
        {"type": "text", "text": "正文段落。", "page_idx": 0},
        {
            "type": "table",
            "table_caption": ["表一"],
            "table_body": "| 年份 | 数量 |\n|---|---|\n| 2020 | 2 |",
            "page_idx": 1,
        },
    ]
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("abc_content_list.json", json.dumps(items, ensure_ascii=False))
        bundle.writestr("full.md", "# fallback")

    parsed = MinerUResultAdapter().from_archive(archive, "资料.pdf")

    assert parsed.blocks[0].page == 1
    assert parsed.blocks[0].heading_level == 1
    assert parsed.blocks[1].section_heading == "第一章"
    assert parsed.blocks[2].page == 2
    assert parsed.blocks[2].text.startswith("表一\n| 年份")
    assert parsed.blocks[0].bbox == (1.0, 2.0, 3.0, 4.0)


def test_plain_text_fallback_is_explicit_utf8(tmp_path: Path) -> None:
    source = tmp_path / "资料.txt"
    source.write_text("第一段。\n\n第二段。", encoding="utf-8")

    parsed = TextParser().parse(source)

    assert [block.text for block in parsed.blocks] == ["第一段。", "第二段。"]
    assert parsed.parser_name == "text-loader"
