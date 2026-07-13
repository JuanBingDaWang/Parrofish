"""Typed contracts shared by parsing, chunking, indexing, and retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Bibliography(BaseModel):
    """Document-level source information used by later citation assembly."""

    model_config = ConfigDict(frozen=True)

    author: str | None = None
    title: str
    year: int | None = None
    publisher_or_journal: str | None = None
    document_type: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ParsedBlock(BaseModel):
    """One ordered MinerU or fallback-loader content block."""

    model_config = ConfigDict(frozen=True)

    order: int
    block_type: str
    text: str
    page: int | None = None
    heading_level: int | None = None
    section_heading: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict, repr=False)


class ParsedDocument(BaseModel):
    """Provider-independent structured parsing result."""

    model_config = ConfigDict(frozen=True)

    filename: str
    format: str
    blocks: list[ParsedBlock]
    parser_name: str
    parser_version: str
    artifact_path: Path | None = None


class ManagedDocument(BaseModel):
    """Immutable local copy and content-derived identity for an imported file."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    sha256: str
    filename: str
    format: str
    source_path: Path
    managed_path: Path


class Chunk(BaseModel):
    """A parent context block or child index block with exact text offsets."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None
    chunk_index: int
    char_start: int
    char_end: int
    parent_id: str | None = None
    chunk_kind: Literal["parent", "child"]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_span(self) -> Chunk:
        """Reject offset metadata that cannot describe this exact text."""

        if self.char_start < 0 or self.char_end < self.char_start:
            raise ValueError("Invalid character span")
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("Chunk text length does not match character span")
        return self


class ChunkedDocument(BaseModel):
    """Immutable canonical text plus its parent and child chunks."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    canonical_text: str
    chunks: list[Chunk]


class MetadataFilter(BaseModel):
    """Optional document-scoped constraints applied before retrieval."""

    model_config = ConfigDict(frozen=True)

    doc_ids: set[str] | None = None
    authors: set[str] | None = None
    years: set[int] | None = None
    formats: set[str] | None = None
    section_headings: set[str] | None = None
    chunk_kinds: set[Literal["parent", "child"]] | None = None


class SearchHit(BaseModel):
    """One traceable result from a dense or sparse index."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    source: Literal["dense", "bm25"]
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None
    parent_id: str | None = None


class FusedHit(BaseModel):
    """A retrieval result after hybrid fusion, optional parent expansion and rerank."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    source: Literal["dense", "bm25", "hybrid"]
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    final_rank: int
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None
    parent_id: str | None = None
    expanded_from_child: bool = False
    matched_child_ids: tuple[str, ...] = ()


class RetrievalRequest(BaseModel):
    """A single hybrid retrieval call with all stage-3 options."""

    model_config = ConfigDict(frozen=True)

    kb_id: str = Field(min_length=1)
    query: str
    top_k: int = Field(default=8, ge=1, le=50)
    filters: MetadataFilter | None = None
    use_rewrite: bool = True
    use_hyde: bool = True
    use_rerank: bool = True
    parent_expand: bool = True
    dense_limit: int = Field(default=40, ge=1, le=100)
    sparse_limit: int = Field(default=40, ge=1, le=100)


class RetrievalResult(BaseModel):
    """The full output of one hybrid retrieval, including the queries used."""

    model_config = ConfigDict(frozen=True)

    query: str
    expanded_queries: tuple[str, ...] = ()
    hits: tuple[FusedHit, ...] = ()


class IngestResult(BaseModel):
    """Stable identifiers and counts returned by a successful ingestion."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    kb_id: str
    doc_id: str
    child_chunk_count: int
    reused_document: bool = False
