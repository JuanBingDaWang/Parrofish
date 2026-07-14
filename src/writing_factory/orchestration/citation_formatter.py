"""CSL-based bibliography formatting through citeproc-py."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import citeproc_styles
from citeproc import (
    Citation,
    CitationItem,
    CitationStylesBibliography,
    CitationStylesStyle,
    formatter,
)
from citeproc.source.json import CiteProcJSON

if TYPE_CHECKING:
    from writing_factory.kb.models import Bibliography


_STYLE_NAMES = {
    "gb-t-7714": "china-national-standard-gb-t-7714-2015-numeric",
    "apa": "apa",
    "mla": "modern-language-association",
}


def format_bibliography(bibliography: Bibliography | None, style: str) -> str:
    """Format one bibliography record with the configured CSL style."""

    if bibliography is None:
        return "[文献信息缺失]"
    style_name = _STYLE_NAMES.get(style, _STYLE_NAMES["gb-t-7714"])
    style_path = citeproc_styles.get_style_filepath(style_name)
    citation_style = CitationStylesStyle(style_path, validate=False)
    source = CiteProcJSON([_to_csl_item(bibliography)])
    rendered = CitationStylesBibliography(
        citation_style,
        source,
        formatter.plain,
    )
    rendered.register(Citation([CitationItem("source")]))
    entries = [str(item).strip() for item in rendered.bibliography()]
    if not entries:
        return "[文献信息缺失]"
    return re.sub(r"^\s*\[\d+\]\s*", "", entries[0])


def _to_csl_item(bibliography: Bibliography) -> dict[str, Any]:
    extra = bibliography.extra or {}
    item: dict[str, Any] = {
        "id": "source",
        "type": _csl_type(bibliography.document_type),
        "title": bibliography.title,
    }
    if bibliography.author:
        names = [
            name.strip() for name in re.split(r"[,，;；、]", bibliography.author) if name.strip()
        ]
        item["author"] = [{"literal": name} for name in names]
    if bibliography.year:
        item["issued"] = {"date-parts": [[bibliography.year]]}
    if bibliography.publisher_or_journal:
        if item["type"] == "book":
            item["publisher"] = bibliography.publisher_or_journal
        else:
            item["container-title"] = bibliography.publisher_or_journal

    mappings = {
        "volume": "volume",
        "issue": "issue",
        "pages": "page",
        "page": "page",
        "doi": "DOI",
        "url": "URL",
        "publisher_place": "publisher-place",
    }
    for source_key, csl_key in mappings.items():
        value = extra.get(source_key)
        if value not in (None, ""):
            item[csl_key] = str(value)
    return item


def _csl_type(document_type: str | None) -> str:
    normalized = (document_type or "").casefold()
    if normalized in {"book", "monograph", "图书", "专著"}:
        return "book"
    if normalized in {"thesis", "dissertation", "学位论文"}:
        return "thesis"
    if normalized in {"conference", "paper-conference", "会议论文"}:
        return "paper-conference"
    if normalized in {"web", "webpage", "网页"}:
        return "webpage"
    if normalized in {"report", "报告"}:
        return "report"
    return "article-journal"
