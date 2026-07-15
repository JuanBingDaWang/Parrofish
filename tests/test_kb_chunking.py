"""Structure-first chunking and exact span tests."""

from __future__ import annotations

from writing_factory.kb.chunking import StructureChunker
from writing_factory.kb.models import ParsedBlock, ParsedDocument


def test_chunks_preserve_exact_spans_pages_and_parent_links() -> None:
    parsed = ParsedDocument(
        filename="资料.pdf",
        format="pdf",
        parser_name="fixture",
        parser_version="1",
        blocks=[
            ParsedBlock(
                order=0,
                block_type="text",
                text="第一章",
                page=1,
                heading_level=1,
                section_heading="第一章",
            ),
            ParsedBlock(
                order=1,
                block_type="text",
                text="第一章的事实。",
                page=1,
                section_heading="第一章",
            ),
            ParsedBlock(
                order=2,
                block_type="text",
                text="第二页证据。",
                page=2,
                section_heading="第一章",
            ),
            ParsedBlock(
                order=3,
                block_type="text",
                text="第二章",
                page=2,
                heading_level=1,
                section_heading="第二章",
            ),
        ],
    )

    chunked = StructureChunker(child_target_chars=100).chunk("doc_1", parsed)
    parents = [chunk for chunk in chunked.chunks if chunk.chunk_kind == "parent"]
    children = [chunk for chunk in chunked.chunks if chunk.chunk_kind == "child"]

    assert len(parents) == 2
    assert len(children) == 3
    for child in children:
        assert chunked.canonical_text[child.char_start : child.char_end] == child.text
        parent = next(item for item in parents if item.chunk_id == child.parent_id)
        assert parent.char_start <= child.char_start < child.char_end <= parent.char_end
    assert {(child.page_start, child.page_end) for child in children} == {
        (1, 1),
        (2, 2),
    }


def test_oversized_block_is_bounded_and_stable() -> None:
    long_text = "这是一个需要保留出处的长句。" * 200
    parsed = ParsedDocument(
        filename="长文.txt",
        format="txt",
        parser_name="fixture",
        parser_version="1",
        blocks=[ParsedBlock(order=0, block_type="text", text=long_text)],
    )
    chunker = StructureChunker(child_target_chars=200, child_max_chars=260)

    first = chunker.chunk("doc_long", parsed)
    second = chunker.chunk("doc_long", parsed)
    children = [chunk for chunk in first.chunks if chunk.chunk_kind == "child"]

    assert all(len(chunk.text) <= 260 for chunk in children)
    assert [chunk.chunk_id for chunk in first.chunks] == [chunk.chunk_id for chunk in second.chunks]
