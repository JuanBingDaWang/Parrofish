"""Typed response contracts returned by external service adapters."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TokenUsage(BaseModel):
    """Normalized token accounting across providers."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ChatResult(BaseModel):
    """One non-streaming chat completion."""

    model_config = ConfigDict(frozen=True)

    content: str
    model: str
    finish_reason: str | None = None
    reasoning_content: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    trace_id: str | None = None


class EmbeddingResult(BaseModel):
    """Dense vectors in the same order as their input texts."""

    model_config = ConfigDict(frozen=True)

    vectors: list[list[float]]
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)


class RerankItem(BaseModel):
    """One scored document from the rerank endpoint."""

    model_config = ConfigDict(frozen=True)

    index: int
    relevance_score: float
    document: Any | None = None


class RerankResult(BaseModel):
    """Ordered reranking output."""

    model_config = ConfigDict(frozen=True)

    results: list[RerankItem]
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)


class MinerUTask(BaseModel):
    """Normalized MinerU task state used by the ingestion pipeline."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    state: str | None = None
    full_zip_url: str | None = None
    err_msg: str | None = None
    raw_data: dict[str, Any] = Field(default_factory=dict, repr=False)


class MinerUBatchUpload(BaseModel):
    """Presigned upload URLs and their batch identifier."""

    model_config = ConfigDict(frozen=True)

    batch_id: str
    file_urls: list[str]
