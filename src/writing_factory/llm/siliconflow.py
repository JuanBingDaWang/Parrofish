"""Typed SiliconFlow chat, embedding, and rerank client."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
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
CancellationCheck = Callable[[], None]
_STREAM_OBSERVER: ContextVar[StreamObserver | None] = ContextVar(
    "siliconflow_stream_observer",
    default=None,
)
_STREAM_LABEL: ContextVar[str | None] = ContextVar(
    "siliconflow_stream_label",
    default=None,
)
_CANCELLATION_CHECK: ContextVar[CancellationCheck | None] = ContextVar(
    "siliconflow_cancellation_check",
    default=None,
)
@dataclass(frozen=True, slots=True)
class _RunSettingsSnapshot:
    """Freeze model and timeout settings for one background operation."""

    chat_model: str
    embedding_model: str
    rerank_model: str
    step_configs: dict[str, ChatStepConfig]
    attempt_timeout_seconds: float
    total_timeout_seconds: float
    stream_idle_timeout_seconds: float


_RUN_SETTINGS: ContextVar[_RunSettingsSnapshot | None] = ContextVar(
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
        total_timeout_seconds: float | None = None,
        stream_idle_timeout_seconds: float | None = None,
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
        self.default_total_timeout_seconds = (
            total_timeout_seconds
            if total_timeout_seconds is not None
            else settings.siliconflow_total_timeout_seconds
        )
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
            stream_idle_timeout_seconds=(
                stream_idle_timeout_seconds
                if stream_idle_timeout_seconds is not None
                else settings.siliconflow_stream_idle_timeout_seconds
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

    def set_total_timeout(self, seconds: float) -> None:
        """Apply the soft budget used to decide whether another attempt may start."""

        if seconds <= 0:
            raise ValueError("SiliconFlow 整项调用总预算必须大于 0")
        self.default_total_timeout_seconds = seconds

    def set_stream_idle_timeout(self, seconds: float) -> None:
        """Apply the maximum gap allowed between two streamed data chunks."""

        if seconds <= 0:
            raise ValueError("SiliconFlow 流式空闲超时必须大于 0")
        self.transport.stream_idle_timeout_seconds = seconds

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
        if snapshot is not None and step_id in snapshot.step_configs:
            return snapshot.step_configs[step_id]
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
            _RunSettingsSnapshot(
                chat_model=self.chat_model,
                embedding_model=self.embedding_model,
                rerank_model=self.rerank_model,
                step_configs=profiles,
                attempt_timeout_seconds=float(
                    getattr(self.transport, "default_request_timeout_seconds", None)
                    or self.settings.siliconflow_request_timeout_seconds
                ),
                total_timeout_seconds=float(self.default_total_timeout_seconds),
                stream_idle_timeout_seconds=float(
                    getattr(
                        self.transport,
                        "stream_idle_timeout_seconds",
                        self.settings.siliconflow_stream_idle_timeout_seconds,
                    )
                ),
            )
        )
        try:
            yield
        finally:
            _RUN_SETTINGS.reset(token)

    def _active_model(self, kind: Literal["chat", "embedding", "reranker"]) -> str:
        snapshot = _RUN_SETTINGS.get()
        if snapshot is not None:
            return {
                "chat": snapshot.chat_model,
                "embedding": snapshot.embedding_model,
                "reranker": snapshot.rerank_model,
            }[kind]
        return {
            "chat": self.chat_model,
            "embedding": self.embedding_model,
            "reranker": self.rerank_model,
        }[kind]

    @contextmanager
    def observe_stream(
        self,
        observer: StreamObserver,
        *,
        check_cancelled: CancellationCheck | None = None,
    ) -> Iterator[None]:
        """Route stream activity and cooperative cancellation for one UI worker."""

        observer_token = _STREAM_OBSERVER.set(observer)
        cancellation_token = _CANCELLATION_CHECK.set(check_cancelled)
        try:
            yield
        finally:
            _CANCELLATION_CHECK.reset(cancellation_token)
            _STREAM_OBSERVER.reset(observer_token)

    @contextmanager
    def stream_stage(self, label: str) -> Iterator[None]:
        """Tag nested concurrent stream events so UI output cannot interleave silently."""

        normalized = label.strip() or "模型调用"
        token = _STREAM_LABEL.set(normalized)
        try:
            yield
        except Exception as exc:
            self._report_stream_error(normalized, exc)
            raise
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
        report_stream_error: bool = True,
    ) -> ChatResult:
        """Run chat and publish one sanitized failure event for the active UI stage."""

        label = _STREAM_LABEL.get()
        if not label and step_id:
            try:
                from writing_factory.llm.configuration import get_step_definition

                label = get_step_definition(step_id).name
            except ValueError:
                label = step_id
        try:
            return self._chat_impl(
                messages,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                response_format=response_format,
                use_cache=use_cache,
                request_timeout_seconds=request_timeout_seconds,
                request_total_timeout_seconds=request_total_timeout_seconds,
                request_attempts=request_attempts,
                stream=stream,
                priority=priority,
                result_validator=result_validator,
                step_id=step_id,
                step_max_tokens_multiplier=step_max_tokens_multiplier,
            )
        except Exception as exc:
            if report_stream_error:
                self._report_stream_error(label or "模型调用", exc)
            raise

    def _chat_impl(
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
                request_timeout_seconds = profile.timeout_seconds
            if profile.total_timeout_seconds is not None:
                request_total_timeout_seconds = profile.total_timeout_seconds

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
        check_cancelled = _CANCELLATION_CHECK.get()
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
        run_snapshot = _RUN_SETTINGS.get()
        default_attempt_timeout = (
            run_snapshot.attempt_timeout_seconds
            if run_snapshot is not None
            else getattr(
                self.transport,
                "default_request_timeout_seconds",
                self.settings.siliconflow_request_timeout_seconds,
            )
        )
        attempt_timeout = (
            request_timeout_seconds
            if request_timeout_seconds is not None
            else default_attempt_timeout
        )
        total_timeout = (
            request_total_timeout_seconds
            if request_total_timeout_seconds is not None
            else (
                run_snapshot.total_timeout_seconds
                if run_snapshot is not None
                else self.default_total_timeout_seconds
            )
        )
        stream_idle_timeout = (
            run_snapshot.stream_idle_timeout_seconds
            if run_snapshot is not None
            else getattr(
                self.transport,
                "stream_idle_timeout_seconds",
                self.settings.siliconflow_stream_idle_timeout_seconds,
            )
        )
        if attempt_timeout is None or attempt_timeout <= 0:
            raise ValueError("SiliconFlow 单次尝试上限必须大于 0")
        if total_timeout <= 0:
            raise ValueError("SiliconFlow 整项调用总预算必须大于 0")
        deadline = self._monotonic() + total_timeout
        attempt_limit = max(
            1,
            request_attempts
            if request_attempts is not None
            else getattr(self.transport, "max_retries", self.settings.max_retries),
        )
        fallback_enabled = stream and attempt_limit >= 2
        response: dict[str, Any] | None = None
        response_is_streamed = stream
        previous_error: RetryableServiceError | None = None

        for attempt_number in range(1, attempt_limit + 1):
            attempt_is_streamed = stream and not (
                fallback_enabled and attempt_number == attempt_limit
            )
            if attempt_number == 1:
                attempt_name = "流式尝试" if attempt_is_streamed else "非流式尝试"
            elif attempt_is_streamed:
                attempt_name = "流式重试"
            elif stream:
                attempt_name = "非流式兜底"
            else:
                attempt_name = "非流式重试"
            if attempt_number > 1:
                if attempt_is_streamed:
                    self._wait_before_retry(attempt_number - 1, check_cancelled)
                if self._monotonic() >= deadline:
                    message = (
                        f"整项调用总预算 {int(total_timeout)} 秒已耗尽，"
                        f"未启动第 {attempt_number}/{attempt_limit} 次{attempt_name}"
                    )
                    notify("status", message)
                    raise IncompleteStreamError(message) from previous_error
                notify(
                    "status",
                    f"正在启动第 {attempt_number}/{attempt_limit} 次{attempt_name}，"
                    f"单次上限 {int(attempt_timeout)} 秒",
                )

            attempt_payload = dict(payload)
            attempt_payload["stream"] = attempt_is_streamed
            if attempt_is_streamed:
                attempt_payload["stream_options"] = {"include_usage": True}
            else:
                attempt_payload.pop("stream_options", None)
            try:
                response = self.transport.request_json(
                    "POST",
                    "/chat/completions",
                    operation="chat",
                    payload=attempt_payload,
                    model=active_chat_model,
                    reasoning_effort=(reasoning_effort or "auto") if thinking else "disabled",
                    prompt_summary={
                        **prompt_summary,
                        "attempt_number": attempt_number,
                        "attempt_limit": attempt_limit,
                        "stream_fallback": stream and not attempt_is_streamed,
                    },
                    use_cache=use_cache,
                    request_timeout_seconds=attempt_timeout,
                    request_total_timeout_seconds=attempt_timeout,
                    request_attempts=1,
                    stream_response=attempt_is_streamed,
                    stream_idle_timeout_seconds=stream_idle_timeout,
                    priority=priority,
                    response_validator=(
                        validate_response
                        if response_format == "json_object" or result_validator is not None
                        else None
                    ),
                    stream_event_callback=(
                        publish_stream_event
                        if observer is not None and attempt_is_streamed
                        else None
                    ),
                    check_cancelled=check_cancelled,
                )
            except RetryableServiceError as exc:
                previous_error = exc
                notify(
                    "attempt_reset",
                    f"第 {attempt_number}/{attempt_limit} 次{attempt_name}未完整结束："
                    f"{str(exc).strip() or type(exc).__name__}",
                )
                if attempt_number >= attempt_limit:
                    raise
                continue
            response_is_streamed = attempt_is_streamed
            break

        if response is None:
            raise IncompleteStreamError("SiliconFlow 所有请求尝试均未返回完整结果")

        result = self._chat_result(response, streamed=response_is_streamed)
        if observer is not None and stream and not response_is_streamed and result.content:
            notify("content", result.content)
        if result_validator is not None:
            result_validator(result)
        if observer is not None and stream:
            notify("complete", "done")
        return result

    @staticmethod
    def _monotonic() -> float:
        """Return a monotonic clock through one testable boundary."""

        return time.monotonic()

    @staticmethod
    def _wait_before_retry(
        failed_attempt_number: int,
        check_cancelled: CancellationCheck | None,
    ) -> None:
        """Back off between streamed attempts while remaining cancellation-responsive."""

        deadline = time.monotonic() + min(8.0, 0.5 * (2 ** max(0, failed_attempt_number - 1)))
        while True:
            if check_cancelled is not None:
                check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

    @staticmethod
    def _report_stream_error(label: str, error: Exception) -> None:
        """Send one failure reason without turning cancellation into a business error."""

        if getattr(error, "_writing_factory_stream_error_reported", False):
            return
        check_cancelled = _CANCELLATION_CHECK.get()
        if check_cancelled is not None:
            try:
                check_cancelled()
            except Exception:
                return
        observer = _STREAM_OBSERVER.get()
        if observer is None:
            return
        detail = str(error).strip() or type(error).__name__
        detail = detail[:2000]
        try:
            error._writing_factory_stream_error_reported = True  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            observer(f"error::{label}", detail)
        except Exception:
            pass

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
            check_cancelled=_CANCELLATION_CHECK.get(),
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
            check_cancelled=_CANCELLATION_CHECK.get(),
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
