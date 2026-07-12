"""Structured MinerU result adaptation and UTF-8 fallback parsing."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from writing_factory.kb.models import ParsedBlock, ParsedDocument


class UnsupportedDocumentError(ValueError):
    """Raised when no approved parser covers a source format."""


class MinerUResultAdapter:
    """Convert MinerU content-list artifacts into provider-independent blocks."""

    def from_archive(self, archive_path: Path, filename: str) -> ParsedDocument:
        """Read structured blocks from a completed MinerU ZIP artifact."""

        archive = archive_path.resolve(strict=True)
        with zipfile.ZipFile(archive) as bundle:
            content_name = self._content_list_name(bundle.namelist())
            if content_name is None:
                return self._from_markdown(bundle, filename, archive)
            with bundle.open(content_name) as source:
                raw_items = json.load(source)
        if not isinstance(raw_items, list):
            raise ValueError("MinerU content_list.json must contain a list")
        blocks = self._adapt_items(raw_items)
        return ParsedDocument(
            filename=filename,
            format=Path(filename).suffix.lower().removeprefix("."),
            blocks=blocks,
            parser_name="mineru",
            parser_version="v4-vlm",
            artifact_path=archive,
        )

    @staticmethod
    def _content_list_name(names: list[str]) -> str | None:
        candidates = [
            name
            for name in names
            if name.endswith("content_list.json") and not name.endswith("content_list_v2.json")
        ]
        return sorted(candidates)[0] if candidates else None

    def _adapt_items(self, items: list[Any]) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        heading_stack: dict[int, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            block_type = str(item.get("type", "unknown"))
            text = self._item_text(item, block_type)
            if not text.strip():
                continue
            heading_level = item.get("text_level")
            if not isinstance(heading_level, int):
                heading_level = None
            if heading_level is not None:
                heading_stack[heading_level] = text.strip()
                heading_stack = {
                    level: value for level, value in heading_stack.items() if level <= heading_level
                }
            section_heading = (
                " > ".join(heading_stack[level] for level in sorted(heading_stack)) or None
            )
            page_index = item.get("page_idx")
            page = page_index + 1 if isinstance(page_index, int) else None
            bbox = self._bbox(item.get("bbox"))
            blocks.append(
                ParsedBlock(
                    order=len(blocks),
                    block_type=block_type,
                    text=text,
                    page=page,
                    heading_level=heading_level,
                    section_heading=section_heading,
                    bbox=bbox,
                    raw_metadata={
                        key: value
                        for key, value in item.items()
                        if key not in {"text", "table_body"}
                    },
                )
            )
        return blocks

    @staticmethod
    def _item_text(item: dict[str, Any], block_type: str) -> str:
        if block_type == "table":
            captions = item.get("table_caption") or []
            body = item.get("table_body") or item.get("text") or ""
            return "\n".join([*map(str, captions), str(body)]).strip()
        if block_type == "image":
            captions = item.get("image_caption") or []
            footnotes = item.get("image_footnote") or []
            return "\n".join(map(str, [*captions, *footnotes])).strip()
        return str(item.get("text") or "").strip()

    @staticmethod
    def _bbox(value: Any) -> tuple[float, float, float, float] | None:
        if not isinstance(value, list) or len(value) != 4:
            return None
        if not all(isinstance(item, (int, float)) for item in value):
            return None
        return tuple(float(item) for item in value)  # type: ignore[return-value]

    @staticmethod
    def _from_markdown(bundle: zipfile.ZipFile, filename: str, archive: Path) -> ParsedDocument:
        try:
            markdown = bundle.read("full.md").decode("utf-8")
        except KeyError as exc:
            raise ValueError("MinerU archive has neither content list nor full.md") from exc
        return ParsedDocument(
            filename=filename,
            format=Path(filename).suffix.lower().removeprefix("."),
            blocks=[ParsedBlock(order=0, block_type="markdown", text=markdown)],
            parser_name="mineru",
            parser_version="v4-vlm-markdown-fallback",
            artifact_path=archive,
        )


class TextParser:
    """UTF-8 parser for plain-text files outside MinerU coverage."""

    def parse(self, path: Path) -> ParsedDocument:
        """Split plain text on blank lines while preserving order."""

        source = path.resolve(strict=True)
        text = source.read_text(encoding="utf-8-sig")
        paragraphs = [part.strip() for part in text.replace("\r\n", "\n").split("\n\n")]
        blocks = [
            ParsedBlock(order=index, block_type="text", text=paragraph)
            for index, paragraph in enumerate(paragraphs)
            if paragraph
        ]
        return ParsedDocument(
            filename=source.name,
            format="txt",
            blocks=blocks,
            parser_name="text-loader",
            parser_version="1",
        )
