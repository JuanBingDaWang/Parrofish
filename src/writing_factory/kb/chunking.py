"""Structure-first parent/child chunking with exact canonical character spans."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

from writing_factory.kb.models import Chunk, ChunkedDocument, ParsedBlock, ParsedDocument

_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；!?;])")


@dataclass(frozen=True, slots=True)
class _BlockSpan:
    block: ParsedBlock
    start: int
    end: int


class StructureChunker:
    """Respect headings and pages before applying bounded character fallbacks."""

    def __init__(
        self,
        *,
        child_target_chars: int = 1000,
        child_max_chars: int = 1800,
        parent_max_chars: int = 8000,
    ) -> None:
        self.child_target_chars = child_target_chars
        self.child_max_chars = child_max_chars
        self.parent_max_chars = parent_max_chars

    def chunk(self, doc_id: str, parsed: ParsedDocument) -> ChunkedDocument:
        """Build immutable canonical text and two-level chunks."""

        canonical_text, spans = self._canonicalize(parsed.blocks)
        parents = self._build_parents(doc_id, canonical_text, spans)
        children = self._build_children(doc_id, canonical_text, spans, parents)
        self._validate(canonical_text, [*parents, *children])
        return ChunkedDocument(
            doc_id=doc_id,
            canonical_text=canonical_text,
            chunks=[*parents, *children],
        )

    def _canonicalize(self, blocks: list[ParsedBlock]) -> tuple[str, list[_BlockSpan]]:
        pieces: list[str] = []
        spans: list[_BlockSpan] = []
        cursor = 0
        for block in sorted(blocks, key=lambda item: item.order):
            text = block.text.replace("\r\n", "\n").replace("\r", "\n").strip()
            if not text:
                continue
            if pieces:
                pieces.append("\n\n")
                cursor += 2
            start = cursor
            pieces.append(text)
            cursor += len(text)
            spans.append(
                _BlockSpan(block=block.model_copy(update={"text": text}), start=start, end=cursor)
            )
        return "".join(pieces), spans

    def _build_parents(
        self,
        doc_id: str,
        text: str,
        spans: list[_BlockSpan],
    ) -> list[Chunk]:
        groups = self._group_spans(spans, parent=True)
        return [
            self._make_chunk(doc_id, text, group, index, "parent", None)
            for index, group in enumerate(groups)
        ]

    def _build_children(
        self,
        doc_id: str,
        text: str,
        spans: list[_BlockSpan],
        parents: list[Chunk],
    ) -> list[Chunk]:
        expanded: list[_BlockSpan] = []
        for span in spans:
            expanded.extend(self._split_oversized(span, text))
        children: list[Chunk] = []
        for parent in parents:
            parent_spans = [
                span
                for span in expanded
                if parent.char_start <= span.start and span.end <= parent.char_end
            ]
            for group in self._group_spans(parent_spans, parent=False):
                children.append(
                    self._make_chunk(
                        doc_id,
                        text,
                        group,
                        len(children),
                        "child",
                        parent.chunk_id,
                    )
                )
        return children

    def _group_spans(self, spans: list[_BlockSpan], *, parent: bool) -> list[list[_BlockSpan]]:
        groups: list[list[_BlockSpan]] = []
        current: list[_BlockSpan] = []
        for span in spans:
            if not current:
                current = [span]
                continue
            current_size = span.end - current[0].start
            heading_changed = span.block.section_heading != current[-1].block.section_heading
            page_changed = span.block.page != current[-1].block.page
            limit = self.parent_max_chars if parent else self.child_target_chars
            must_split = current_size > limit or heading_changed or (not parent and page_changed)
            if must_split:
                groups.append(current)
                current = [span]
            else:
                current.append(span)
        if current:
            groups.append(current)
        return groups

    def _split_oversized(self, span: _BlockSpan, canonical_text: str) -> list[_BlockSpan]:
        if span.end - span.start <= self.child_max_chars:
            return [span]
        source = canonical_text[span.start : span.end]
        pieces = _SENTENCE_BOUNDARY.split(source)
        results: list[_BlockSpan] = []
        piece_start = 0
        buffer_start = 0
        buffer_length = 0
        for piece in pieces:
            if buffer_length and buffer_length + len(piece) > self.child_target_chars:
                results.extend(self._bounded_span(span, buffer_start, buffer_length))
                buffer_start = piece_start
                buffer_length = 0
            buffer_length += len(piece)
            piece_start += len(piece)
        if buffer_length:
            results.extend(self._bounded_span(span, buffer_start, buffer_length))
        return results

    def _bounded_span(
        self, original: _BlockSpan, local_start: int, length: int
    ) -> list[_BlockSpan]:
        results: list[_BlockSpan] = []
        consumed = 0
        while consumed < length:
            size = min(self.child_max_chars, length - consumed)
            start = original.start + local_start + consumed
            end = start + size
            results.append(_BlockSpan(block=original.block, start=start, end=end))
            consumed += size
        return results

    def _make_chunk(
        self,
        doc_id: str,
        canonical_text: str,
        spans: list[_BlockSpan],
        index: int,
        kind: Literal["parent", "child"],
        parent_id: str | None,
    ) -> Chunk:
        start, end = spans[0].start, spans[-1].end
        chunk_text = canonical_text[start:end]
        pages = [span.block.page for span in spans if span.block.page is not None]
        identifier = hashlib.sha256(
            f"{doc_id}|{kind}|{start}|{end}|{chunk_text}".encode()
        ).hexdigest()[:32]
        return Chunk(
            chunk_id=f"chk_{identifier}",
            doc_id=doc_id,
            text=chunk_text,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            section_heading=spans[-1].block.section_heading,
            chunk_index=index,
            char_start=start,
            char_end=end,
            parent_id=parent_id,
            chunk_kind=kind,
            metadata={"block_types": sorted({span.block.block_type for span in spans})},
        )

    @staticmethod
    def _validate(canonical_text: str, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            if canonical_text[chunk.char_start : chunk.char_end] != chunk.text:
                raise ValueError(f"Chunk span mismatch: {chunk.chunk_id}")
