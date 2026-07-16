"""Reference list assembler: deduplicate EvidenceItems by doc_id, look up
bibliography metadata, and format citations per the chosen style (GB/T 7714,
APA, MLA).

Iron law compliance:
    #4 引用由代码拼装不由模型敲 — 本模块完全不调用 LLM，纯规则拼装
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from writing_factory.generate.models import (
    CitationStyle,
    EvidenceItem,
    PolishedSection,
    ReferenceItem,
    ReferenceList,
)
from writing_factory.orchestration.citation_formatter import format_bibliography

if TYPE_CHECKING:
    from writing_factory.store.kb_repository import KnowledgeBaseRepository

logger = logging.getLogger(__name__)


# ── 公开入口 ────────────────────────────────────────────────────


def assemble_reference_list(
    evidence_items: list[EvidenceItem],
    *,
    citation_style: CitationStyle,
    kb_repository: KnowledgeBaseRepository,
    kb_id: str,
) -> ReferenceList:
    """Produce a deduplicated, formatted ReferenceList from raw EvidenceItems.

    Args:
        evidence_items: All evidence items from the full paper (all sections).
        citation_style: One of gb-t-7714, apa, mla.
        kb_repository: Repository for looking up Bibliography records.
        kb_id: The knowledge base identifier.

    Returns:
        ReferenceList with items sorted by source_key.
    """
    if not evidence_items:
        return ReferenceList(items=[], style=citation_style)

    # 1. Collect unique doc_ids
    doc_ids: set[str] = {item.doc_id for item in evidence_items}

    # 2. Look up bibliographies
    bibs = kb_repository.get_bibliographies(kb_id, doc_ids)

    # 3. Group evidence items by doc_id, preserving source_keys
    #    We use a dict to deduplicate: doc_id → {source_keys, first_item}
    dedup: dict[str, dict] = {}
    for item in evidence_items:
        if item.doc_id not in dedup:
            dedup[item.doc_id] = {
                "source_keys": [],
                "first_chunk_id": item.chunk_id,
                "first_item": item,
            }
        dedup[item.doc_id]["source_keys"].append(item.source_key)

    # 4. Build formatted ReferenceItems
    items: list[ReferenceItem] = []
    for doc_id, info in dedup.items():
        first_item: EvidenceItem = info["first_item"]
        bib = bibs.get(doc_id) or _web_bibliography(first_item)
        source_keys = info["source_keys"]
        # Sort source_keys naturally: [S1], [S2], [S10] → [S1], [S2], [S10]
        source_keys.sort(key=_source_key_sort_key)

        citation_text = _format_citation(
            bib=bib,
            style=citation_style,
            source_key=source_keys[0] if len(source_keys) == 1 else source_keys,
        )

        items.append(
            ReferenceItem(
                source_key=", ".join(source_keys),
                citation_text=citation_text,
                doc_id=doc_id,
                chunk_id=info["first_chunk_id"],
                url=first_item.url,
            )
        )

    # Sort by the first source_key
    items.sort(key=lambda ri: _source_key_sort_key(ri.source_key.split(",")[0].strip()))

    return ReferenceList(items=items, style=citation_style)


def _web_bibliography(item: EvidenceItem):
    """Build code-owned webpage metadata for CSL formatting."""

    if item.source_type != "web" or not item.url:
        return None
    from writing_factory.kb.models import Bibliography

    year_match = re.search(r"(?:19|20)\d{2}", item.date_published or "")
    return Bibliography(
        author=item.site_name or None,
        title=item.title or "未命名网页",
        year=int(year_match.group(0)) if year_match else None,
        publisher_or_journal=item.site_name or None,
        document_type="web",
        extra={"url": item.url},
    )


def render_final_citation_markers(
    sections: list[PolishedSection],
    references: ReferenceList,
) -> list[PolishedSection]:
    """Replace internal ``[Sx]`` keys with final bibliography numbers in code."""

    marker_numbers: dict[str, int] = {}
    for number, reference in enumerate(references.items, 1):
        for source_key in reference.source_key.split(","):
            marker_numbers[source_key.strip()] = number

    rendered: list[PolishedSection] = []
    for section in sections:
        text = section.polished_text
        for source_key, number in sorted(
            marker_numbers.items(), key=lambda item: len(item[0]), reverse=True
        ):
            text = text.replace(f"[{source_key}]", f"[{number}]")
        rendered.append(section.model_copy(update={"polished_text": text}))
    return rendered


# ── 样式格式化 ──────────────────────────────────────────────────


def _format_citation(
    bib,
    style: CitationStyle,
    source_key: str | list[str],
) -> str:
    """Format a single bibliography entry per the chosen citation style.

    If bib is None (no metadata found), returns a minimal fallback.
    """
    _ = source_key
    return format_bibliography(bib, style)


def _format_gb7714(bib) -> str:
    """GB/T 7714-2015 格式。

    [序号] 作者. 题名[J]. 刊名, 年, 卷(期): 起止页码.
    [序号] 作者. 书名[M]. 出版地: 出版社, 年.
    """
    if bib is None:
        return "[文献信息缺失]"

    parts: list[str] = []

    # 作者
    author = _format_authors(bib.author) if bib.author else ""
    if author:
        parts.append(author + ".")

    # 题名
    title = bib.title or ""
    if title:
        parts.append(title + "[J]" if bib.document_type != "book" else title + "[M]")

    # 刊名/出版社
    if bib.publisher_or_journal:
        parts.append(bib.publisher_or_journal + ",")

    # 年份
    if bib.year:
        parts.append(str(bib.year) + ".")

    # extra fields
    extra = bib.extra if bib and bib.extra else {}
    if "volume" in extra:
        parts.append(extra["volume"])
    if "issue" in extra:
        parts[-1] = f"{parts[-1]}({extra['issue']})" if parts else extra["issue"]
    if "pages" in extra:
        parts.append(":" + extra["pages"] + ".")

    return " ".join(parts).rstrip(".") + "."


def _format_apa(bib) -> str:
    """APA 7th 格式。

    Author, A. A. (Year). Title. Journal, Volume(Issue), pages.
    Author, A. A. (Year). Title. Publisher.
    """
    if bib is None:
        return "[Bibliographic information missing]"

    parts: list[str] = []

    # Author
    author = _format_authors_apa(bib.author) if bib.author else ""
    if author:
        parts.append(author)

    # Year
    if bib.year:
        parts.append(f"({bib.year}).")
    else:
        parts.append("(n.d.).")

    # Title
    title = bib.title or ""
    if title:
        parts.append(title + ".")

    # Journal or Publisher
    if bib.publisher_or_journal:
        extra = bib.extra if bib.extra else {}
        if bib.document_type == "book":
            parts.append(bib.publisher_or_journal + ".")
        else:
            journal_part = bib.publisher_or_journal
            if "volume" in extra:
                journal_part += f", {extra['volume']}"
            if "issue" in extra:
                journal_part += f"({extra['issue']})"
            if "pages" in extra:
                journal_part += f", {extra['pages']}"
            parts.append(journal_part + ".")

    return " ".join(parts)


def _format_mla(bib) -> str:
    """MLA 9th 格式。

    Author. "Title." Journal, vol. #, no. #, year, pp. #-#.
    Author. Title. Publisher, year.
    """
    if bib is None:
        return "[Bibliographic information missing]"

    parts: list[str] = []

    # Author
    author = _format_authors_mla(bib.author) if bib.author else ""
    if author:
        parts.append(author + ".")

    # Title
    title = bib.title or ""
    if bib.document_type == "book":
        parts.append(f'"{title}."')
    else:
        parts.append(f'"{title},"')

    # Journal/Publisher
    if bib.publisher_or_journal:
        parts.append(bib.publisher_or_journal + ",")

    # Volume, issue, year, pages
    extra = bib.extra if bib.extra else {}
    if "volume" in extra:
        parts.append(f"vol. {extra['volume']},")
    if "issue" in extra:
        parts.append(f"no. {extra['issue']},")
    if bib.year:
        parts.append(f"{bib.year},")
    if "pages" in extra:
        parts.append(f"pp. {extra['pages']}.")

    result = " ".join(parts)
    return result.rstrip(",") + "."


# ── 作者名格式化辅助 ────────────────────────────────────────────


def _format_authors(author_raw: str) -> str:
    """GB/T 7714: 保留原始作者名（如 '张三, 李四' 或 'Smith J, Jones M'）。"""
    return author_raw.strip()


def _format_authors_apa(author_raw: str) -> str:
    """APA 7th: 如 'Smith, J., & Jones, M.'。

    简单启发式：如果输入是英文逗号分隔，按 APA 规则处理；
    中文作者保留原名。
    """
    authors = [a.strip() for a in author_raw.split(",") if a.strip()]
    if not authors:
        return author_raw.strip()

    # 检测是否含中文字符 — 中文作者保留原名
    if any("\u4e00" <= ch <= "\u9fff" for ch in author_raw):
        return author_raw.strip()

    # 英文作者：尝试 APA 格式
    formatted: list[str] = []
    for i, a in enumerate(authors):
        prefix = "& " if i == len(authors) - 1 and len(authors) > 1 else ""
        formatted.append(f"{prefix}{a}")
    return ", ".join(formatted)


def _format_authors_mla(author_raw: str) -> str:
    """MLA 9th: 第一位作者 'Last, First'，其余 'First Last'。

    简单启发式：中文保留原名，英文尝试倒置。
    """
    authors = [a.strip() for a in author_raw.split(",") if a.strip()]
    if not authors:
        return author_raw.strip()

    if any("\u4e00" <= ch <= "\u9fff" for ch in author_raw):
        return author_raw.strip()

    # 仅第一位倒置
    first = authors[0]
    if " " in first:
        parts = first.rsplit(" ", 1)
        first = f"{parts[1]}, {parts[0]}"

    if len(authors) == 1:
        return first
    if len(authors) == 2:
        return f"{first}, and {authors[1]}"
    return f"{first}, et al."


# ── 排序辅助 ────────────────────────────────────────────────────


def _source_key_sort_key(source_key: str) -> tuple[int, int]:
    """Sort source_keys naturally: [S1] < [S2] < [S10]."""
    import re

    numbers = re.findall(r"\d+", source_key)
    if not numbers:
        return (0, 0)
    return (int(numbers[0]), int(numbers[1]) if len(numbers) > 1 else 0)
