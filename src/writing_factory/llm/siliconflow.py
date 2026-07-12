"""Typed SiliconFlow chat, embedding, and rerank client."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from writing_factory.config import Settings
from writing_factory.llm.base import ExternalServiceError, ServiceTransport
from writing_factory.llm.models import (
    ChatResult,
    EmbeddingResult,
    RerankItem,
    RerankResult,
    TokenUsage,
)
from writing_factory.store import Database

ReasoningEffort = Literal["high", "max"]


class SiliconFlowClient:
    """The only SiliconFlow entry point exposed to business modules."""

    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.transport = ServiceTransport(
            provider="siliconflow",
            base_url=settings.siliconflow_base_url,
            credential=settings.siliconflow_api_key,
            database=database,
            connect_timeout_seconds=settings.connect_timeout_seconds,
            read_timeout_seconds=settings.read_timeout_seconds,
            max_retries=settings.max_retries,
            minimum_interval_seconds=settings.min_request_interval_seconds,
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""

        self.transport.close()

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        thinking: bool,
        reasoning_effort: ReasoningEffort = "high",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        seed: int | None = None,
        use_cache: bool = True,
    ) -> ChatResult:
        """Run a deterministic or creative non-streaming chat request."""

        payload: dict[str, Any] = {
            "model": self.settings.chat_model,
            "messages": list(messages),
            "enable_thinking": thinking,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if thinking:
            payload["reasoning_effort"] = reasoning_effort
        if seed is not None:
            payload["seed"] = seed
        response = self.transport.request_json(
            "POST",
            "/chat/completions",
            operation="chat",
            payload=payload,
            model=self.settings.chat_model,
            reasoning_effort=reasoning_effort if thinking else "disabled",
            prompt_summary={
                "message_count": len(messages),
                "character_count": sum(len(item.get("content", "")) for item in messages),
                "thinking": thinking,
            },
            use_cache=use_cache,
        )
        try:
            choice = response["choices"][0]
            message = choice["message"]
            return ChatResult(
                content=message.get("content") or "",
                reasoning_content=message.get("reasoning_content"),
                finish_reason=choice.get("finish_reason"),
                model=response.get("model", self.settings.chat_model),
                usage=self._usage(response),
                trace_id=response.get("trace_id"),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise ExternalServiceError("SiliconFlow returned an invalid chat response") from exc

    def embeddings(
        self,
        texts: Sequence[str],
        *,
        use_cache: bool = True,
    ) -> EmbeddingResult:
        """Embed a batch while preserving input order."""

        response = self.transport.request_json(
            "POST",
            "/embeddings",
            operation="embedding",
            payload={"model": self.settings.embedding_model, "input": list(texts)},
            model=self.settings.embedding_model,
            prompt_summary={
                "item_count": len(texts),
                "character_count": sum(len(text) for text in texts),
            },
            use_cache=use_cache,
        )
        try:
            ordered = sorted(response["data"], key=lambda item: item["index"])
            vectors = [item["embedding"] for item in ordered]
        except (KeyError, TypeError) as exc:
            raise ExternalServiceError(
                "SiliconFlow returned an invalid embedding response"
            ) from exc
        return EmbeddingResult(
            vectors=vectors,
            model=response.get("model", self.settings.embedding_model),
            usage=self._usage(response),
        )

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
        return_documents: bool = False,
        use_cache: bool = True,
    ) -> RerankResult:
        """Rerank candidate documents through SiliconFlow's separate endpoint."""

        payload: dict[str, Any] = {
            "model": self.settings.rerank_model,
            "query": query,
            "documents": list(documents),
            "return_documents": return_documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        response = self.transport.request_json(
            "POST",
            "/rerank",
            operation="rerank",
            payload=payload,
            model=self.settings.rerank_model,
            prompt_summary={
                "document_count": len(documents),
                "query_chars": len(query),
                "document_chars": sum(len(document) for document in documents),
            },
            use_cache=use_cache,
        )
        try:
            items = [RerankItem.model_validate(item) for item in response["results"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalServiceError("SiliconFlow returned an invalid rerank response") from exc
        return RerankResult(
            results=items,
            model=response.get("model", self.settings.rerank_model),
            usage=self._usage(response),
        )

    @staticmethod
    def _usage(response: dict[str, Any]) -> TokenUsage:
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        return TokenUsage(
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            total_tokens=usage.get("total_tokens", 0) or 0,
        )
