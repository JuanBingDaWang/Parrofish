"""Typed SiliconFlow chat, embedding, and rerank client."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from pydantic import SecretStr

from writing_factory.config import Settings
from writing_factory.llm.base import ExternalServiceError, ServiceTransport
from writing_factory.llm.common import (
    DynamicConcurrencyGate,
    IncompleteStreamError,
    RetryableServiceError,
)
from writing_factory.llm.configuration import ChatStepConfig, ModelCatalogEntry
from writing_factory.llm.models import (
    ChatResult,
    EmbeddingResult,
    RerankItem,
    RerankResult,
    TokenUsage,
)
from writing_factory.store import Database

ReasoningEffort = Literal["high", "max"]
StepConfigProvider = Callable[[str], ChatStepConfig]
StreamObserver = Callable[[str, str], None]
_STREAM_OBSERVER: ContextVar[StreamObserver | None] = ContextVar(
    "siliconflow_stream_observer",
    default=None,
)
_STREAM_LABEL: ContextVar[str | None] = ContextVar(
    "siliconflow_stream_label",
    default=None,
)
_RUN_SETTINGS: ContextVar[tuple[str, str, str, dict[str, ChatStepConfig]] | None] = ContextVar(
    "siliconflow_run_settings",
    default=None,
)


class SiliconFlowClient:
    """The only SiliconFlow entry point exposed to business modules."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        concurrency_gate: DynamicConcurrencyGate | None = None,
        *,
        request_timeout_seconds: float | None = None,
        chat_model: str | None = None,
        embedding_model: str | None = None,
        rerank_model: str | None = None,
        step_config_provider: StepConfigProvider | None = None,
    ) -> None:
        self.chat_model = chat_model or settings.chat_model
        self.embedding_model = embedding_model or settings.embedding_model
        self.rerank_model = rerank_model or settings.rerank_model
        self._step_config_provider = step_config_provider
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
            concurrency_gate=concurrency_gate,
            default_request_timeout_seconds=(
                request_timeout_seconds
                if request_timeout_seconds is not None
                else settings.siliconflow_request_timeout_seconds
            ),
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""

        self.transport.close()

    def set_request_timeout(self, seconds: float) -> None:
        """Immediately apply the global timeout to subsequent SiliconFlow calls."""

        if seconds <= 0:
            raise ValueError("SiliconFlow 单次请求超时上限必须大于 0")
        self.transport.default_request_timeout_seconds = seconds

    def configure_provider(
        self,
        *,
        credential: SecretStr | None = None,
        base_url: str | None = None,
    ) -> None:
        """Apply credential or endpoint changes to subsequent requests."""

        if credential is not None:
            self.transport.set_credential(credential)
        if base_url is not None:
            self.transport.set_base_url(base_url)

    def configure_models(
        self,
        *,
        chat_model: str | None = None,
        embedding_model: str | None = None,
        rerank_model: str | None = None,
    ) -> None:
        """Apply active model IDs without rebuilding the client."""

        if chat_model:
            self.chat_model = chat_model
        if embedding_model:
            self.embedding_model = embedding_model
        if rerank_model:
            self.rerank_model = rerank_model

    def set_step_config_provider(self, provider: StepConfigProvider | None) -> None:
        self._step_config_provider = provider

    def step_config(self, step_id: str) -> ChatStepConfig:
        snapshot = _RUN_SETTINGS.get()
        if snapshot is not None and step_id in snapshot[3]:
            return snapshot[3][step_id]
        if self._step_config_provider is None:
            from writing_factory.llm.configuration import get_step_definition

            return get_step_definition(step_id).default
        return self._step_config_provider(step_id)

    @contextmanager
    def freeze_runtime_settings(self) -> Iterator[None]:
        """Keep one background run stable while settings remain editable."""

        from writing_factory.llm.configuration import STEP_DEFINITIONS

        profiles = {item.step_id: self.step_config(item.step_id) for item in STEP_DEFINITIONS}
        token = _RUN_SETTINGS.set(
            (self.chat_model, self.embedding_model, self.rerank_model, profiles)
        )
        try:
            yield
        finally:
            _RUN_SETTINGS.reset(token)

    def _active_model(self, kind: Literal["chat", "embedding", "reranker"]) -> str:
        snapshot = _RUN_SETTINGS.get()
        if snapshot is not None:
            return {"chat": snapshot[0], "embedding": snapshot[1], "reranker": snapshot[2]}[
                kind
            ]
        return {
            "chat": self.chat_model,
            "embedding": self.embedding_model,
            "reranker": self.rerank_model,
        }[kind]

    @contextmanager
    def observe_stream(self, observer: StreamObserver) -> Iterator[None]:
        """Route this execution context's stream activity to one UI worker."""

        token = _STREAM_OBSERVER.set(observer)
        try:
            yield
        finally:
            _STREAM_OBSERVER.reset(token)

    @contextmanager
    def stream_stage(self, label: str) -> Iterator[None]:
        """Tag nested concurrent stream events so UI output cannot interleave silently."""

        token = _STREAM_LABEL.set(label.strip() or None)
        try:
            yield
        finally:
            _STREAM_LABEL.reset(token)

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        thinking: bool,
        reasoning_effort: ReasoningEffort | None = "high",
        temperature: float = 0.2,
        max_tokens: int = 8192,
        seed: int | None = None,
        response_format: Literal["text", "json_object"] = "text",
        use_cache: bool = True,
        request_timeout_seconds: float | None = None,
        request_total_timeout_seconds: float | None = None,
        request_attempts: int | None = None,
        stream: bool = False,
        priority: int = 10,
        result_validator: Callable[[ChatResult], None] | None = None,
        step_id: str | None = None,
        step_max_tokens_multiplier: int = 1,
    ) -> ChatResult:
        """Run a deterministic or creative chat request, optionally over SSE."""

        if step_id is not None:
            profile = self.step_config(step_id)
            temperature = profile.temperature
            if profile.thinking is not None:
                thinking = profile.thinking
            reasoning_effort = (
                reasoning_effort
                if profile.reasoning_effort == "auto"
                else profile.reasoning_effort
            )
            max_tokens = min(131072, profile.max_tokens * max(1, step_max_tokens_multiplier))
            stream = profile.stream
            request_attempts = profile.retry_count + 1
            if profile.timeout_seconds is not None:
                request_total_timeout_seconds = profile.timeout_seconds

        active_chat_model = self._active_model("chat")
        payload: dict[str, Any] = {
            "model": active_chat_model,
            "messages": list(messages),
            "enable_thinking": thinking,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "response_format": {"type": response_format},
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if thinking and reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if seed is not None:
            payload["seed"] = seed

        def validate_response(response: dict[str, Any]) -> None:
            parsed = self._chat_result(response, streamed=stream)
            if response_format == "json_object":
                try:
                    decoded = json.loads(parsed.content)
                except ValueError as exc:
                    raise ValueError("SiliconFlow 返回的 JSON 对象不完整") from exc
                if not isinstance(decoded, dict):
                    raise ValueError("SiliconFlow 返回的 JSON 顶层必须是对象")
            if result_validator is not None:
                result_validator(parsed)

        observer = _STREAM_OBSERVER.get() if stream else None
        stream_label = _STREAM_LABEL.get()
        if observer is not None and not stream_label and step_id:
            try:
                from writing_factory.llm.configuration import get_step_definition

                stream_label = get_step_definition(step_id).name
            except ValueError:
                stream_label = step_id

        def notify(kind: str, text: str) -> None:
            if observer is None:
                return
            observer(f"{kind}::{stream_label}" if stream_label else kind, text)

        pending_content: list[str] = []
        last_content_publish = 0.0
        last_reasoning_publish = 0.0

        def publish_stream_event(chunk: dict[str, Any]) -> None:
            nonlocal last_content_publish, last_reasoning_publish
            if observer is None:
                return
            now = time.monotonic()
            stream_event = chunk.get("_stream_event")
            if stream_event in {"done", "clean_eof", "incomplete"}:
                if pending_content:
                    notify("content", "".join(pending_content))
                    pending_content.clear()
                if stream_event == "clean_eof":
                    notify("status", "服务端未发送 [DONE]，完整性校验通过，已接收本次输出")
                elif stream_event == "incomplete":
                    notify("status", "本次流式输出中断，正在重试")
                return
            choices = chunk.get("choices") or []
            if not choices:
                return
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning_content")
            content = delta.get("content")
            if (
                isinstance(reasoning, str)
                and reasoning
                and now - last_reasoning_publish >= 0.5
            ):
                # Keep private reasoning out of the UI; only expose activity.
                notify("reasoning", "activity")
                last_reasoning_publish = now
            if isinstance(content, str) and content:
                pending_content.append(content)
            if pending_content and (
                choices[0].get("finish_reason") is not None
                or now - last_content_publish >= 0.05
            ):
                notify("content", "".join(pending_content))
                pending_content.clear()
                last_content_publish = now

        prompt_summary = {
            "message_count": len(messages),
            "character_count": sum(len(item.get("content", "")) for item in messages),
            "thinking": thinking,
        }
        total_timeout = (
            request_total_timeout_seconds
            if request_total_timeout_seconds is not None
            else getattr(
                self.transport,
                "default_request_timeout_seconds",
                self.settings.siliconflow_request_timeout_seconds,
            )
        )
        deadline = time.monotonic() + total_timeout if total_timeout is not None else None
        attempt_limit = max(
            1,
            request_attempts
            if request_attempts is not None
            else getattr(self.transport, "max_retries", self.settings.max_retries),
        )
        fallback_enabled = stream and attempt_limit >= 2
        stream_attempts = attempt_limit - 1 if fallback_enabled else attempt_limit

        def remaining_timeout() -> float | None:
            if deadline is None:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise IncompleteStreamError("siliconflow request exceeded total timeout")
            return min(total_timeout, remaining) if total_timeout is not None else remaining

        response_is_streamed = stream
        try:
            response = self.transport.request_json(
                "POST",
                "/chat/completions",
                operation="chat",
                payload=payload,
                model=active_chat_model,
                reasoning_effort=(reasoning_effort or "auto") if thinking else "disabled",
                prompt_summary=prompt_summary,
                use_cache=use_cache,
                request_timeout_seconds=request_timeout_seconds,
                request_total_timeout_seconds=(
                    remaining_timeout() if fallback_enabled else request_total_timeout_seconds
                ),
                request_attempts=stream_attempts,
                stream_response=stream,
                priority=priority,
                response_validator=(
                    validate_response
                    if response_format == "json_object" or result_validator is not None
                    else None
                ),
                stream_event_callback=publish_stream_event if observer is not None else None,
            )
        except RetryableServiceError:
            if not fallback_enabled:
                raise
            if observer is not None:
                notify("status", "流式重试仍未完成，最后一次改用非流式请求")
            fallback_payload = dict(payload)
            fallback_payload["stream"] = False
            fallback_payload.pop("stream_options", None)
            response = self.transport.request_json(
                "POST",
                "/chat/completions",
                operation="chat",
                payload=fallback_payload,
                model=active_chat_model,
                reasoning_effort=(reasoning_effort or "auto") if thinking else "disabled",
                prompt_summary={**prompt_summary, "stream_fallback": True},
                use_cache=use_cache,
                request_timeout_seconds=request_timeout_seconds,
                request_total_timeout_seconds=remaining_timeout(),
                request_attempts=1,
                stream_response=False,
                priority=priority,
                response_validator=(
                    validate_response
                    if response_format == "json_object" or result_validator is not None
                    else None
                ),
            )
            response_is_streamed = False

        result = self._chat_result(response, streamed=response_is_streamed)
        if observer is not None and stream and not response_is_streamed and result.content:
            notify("content", result.content)
        if result_validator is not None:
            result_validator(result)
        return result

    def _chat_result(self, response: dict[str, Any], *, streamed: bool) -> ChatResult:
        """Parse one raw provider response into the stable chat contract."""

        if streamed and "chunks" in response:
            return self._streamed_chat_result(response)
        try:
            choice = response["choices"][0]
            message = choice["message"]
            return ChatResult(
                content=message.get("content") or "",
                reasoning_content=message.get("reasoning_content"),
                finish_reason=choice.get("finish_reason"),
                model=response.get("model", self._active_model("chat")),
                usage=self._usage(response),
                trace_id=response.get("trace_id"),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise ExternalServiceError("SiliconFlow returned an invalid chat response") from exc

    def _streamed_chat_result(self, response: dict[str, Any]) -> ChatResult:
        """Assemble OpenAI-compatible SSE deltas without exposing them to callers."""

        content: list[str] = []
        reasoning: list[str] = []
        finish_reason: str | None = None
        model = self._active_model("chat")
        usage: dict[str, Any] = {}
        trace_id: str | None = None
        try:
            for chunk in response["chunks"]:
                model = chunk.get("model") or model
                trace_id = chunk.get("trace_id") or trace_id
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                finish_reason = choice.get("finish_reason") or finish_reason
        except (KeyError, TypeError) as exc:
            raise ExternalServiceError("SiliconFlow returned invalid chat events") from exc
        return ChatResult(
            content="".join(content),
            reasoning_content="".join(reasoning) or None,
            finish_reason=finish_reason,
            model=model,
            usage=self._usage({"usage": usage}),
            trace_id=trace_id,
        )

    def embeddings(
        self,
        texts: Sequence[str],
        *,
        use_cache: bool = True,
        priority: int = 10,
        model: str | None = None,
    ) -> EmbeddingResult:
        """Embed a batch while preserving input order."""

        active_model = model or self._active_model("embedding")
        response = self.transport.request_json(
            "POST",
            "/embeddings",
            operation="embedding",
            payload={"model": active_model, "input": list(texts)},
            model=active_model,
            prompt_summary={
                "item_count": len(texts),
                "character_count": sum(len(text) for text in texts),
            },
            use_cache=use_cache,
            priority=priority,
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
            model=response.get("model", active_model),
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
        priority: int = 10,
        model: str | None = None,
    ) -> RerankResult:
        """Rerank candidate documents through SiliconFlow's separate endpoint."""

        active_model = model or self._active_model("reranker")
        payload: dict[str, Any] = {
            "model": active_model,
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
            model=active_model,
            prompt_summary={
                "document_count": len(documents),
                "query_chars": len(query),
                "document_chars": sum(len(document) for document in documents),
            },
            use_cache=use_cache,
            priority=priority,
        )
        try:
            items = [RerankItem.model_validate(item) for item in response["results"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalServiceError("SiliconFlow returned an invalid rerank response") from exc
        return RerankResult(
            results=items,
            model=response.get("model", active_model),
            usage=self._usage(response),
        )

    def list_models(self, sub_type: str) -> list[ModelCatalogEntry]:
        """Fetch one authenticated model category from SiliconFlow."""

        if sub_type not in {"chat", "embedding", "reranker"}:
            raise ValueError("模型类型必须是 chat、embedding 或 reranker")
        response = self.transport.request_json(
            "GET",
            f"/models?type=text&sub_type={sub_type}",
            operation="list_models",
            prompt_summary={"sub_type": sub_type},
            use_cache=False,
            request_attempts=2,
        )
        data = response.get("data")
        if not isinstance(data, list):
            raise ExternalServiceError("SiliconFlow 模型列表响应无效")
        return [ModelCatalogEntry.model_validate(item) for item in data]

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
