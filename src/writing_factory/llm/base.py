"""Retrying, cached, observable HTTP transport for external services."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any

import httpx
from pydantic import SecretStr
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, stop_before_delay
from tenacity.wait import wait_exponential_jitter

from writing_factory.llm.common import (
    DynamicConcurrencyGate,
    ExternalServiceError,
    RateLimiter,
    RetryableServiceError,
)
from writing_factory.store import ApiCallRecord, Database

logger = logging.getLogger(__name__)


class ServiceTransport:
    """Centralize auth, retries, cache, and privacy-preserving call records."""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        credential: SecretStr,
        database: Database,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        max_retries: int,
        minimum_interval_seconds: float = 0.0,
        http_client: httpx.Client | None = None,
        concurrency_gate: DynamicConcurrencyGate | None = None,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.database = database
        self.max_retries = max(1, max_retries)
        self.rate_limiter = RateLimiter(minimum_interval_seconds)
        self.concurrency_gate = concurrency_gate
        self._owns_client = http_client is None
        timeout = httpx.Timeout(
            connect=connect_timeout_seconds,
            read=read_timeout_seconds,
            write=read_timeout_seconds,
            pool=connect_timeout_seconds,
        )
        self._client = http_client or httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {credential.get_secret_value()}",
                "Accept": "application/json",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        """Release the owned connection pool."""

        if self._owns_client:
            self._client.close()

    def request_json(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        payload: Mapping[str, Any] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        prompt_summary: Mapping[str, Any] | None = None,
        use_cache: bool = False,
        request_timeout_seconds: float | None = None,
        request_total_timeout_seconds: float | None = None,
        request_attempts: int | None = None,
        stream_response: bool = False,
        priority: int = 10,
        response_validator: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Execute one JSON request and record only structural summaries."""

        normalized_payload = dict(payload or {})
        request_hash = self._request_hash(method, path, normalized_payload)
        call_id = str(uuid.uuid4())
        started_at = time.perf_counter()
        summary = json.dumps(prompt_summary or {}, ensure_ascii=False, sort_keys=True)

        if use_cache:
            cached = self.database.get_cached_response(request_hash)
            if cached is not None:
                try:
                    if response_validator is not None:
                        response_validator(cached)
                except Exception as exc:
                    self.database.quarantine_response(
                        call_id=call_id,
                        request_hash=request_hash,
                        provider=self.provider,
                        operation=operation,
                        response=cached,
                    )
                    self.database.delete_cached_response(request_hash)
                    logger.warning(
                        "discarded invalid cached response provider=%s operation=%s "
                        "error_type=%s call_id=%s",
                        self.provider,
                        operation,
                        type(exc).__name__,
                        call_id,
                    )
                else:
                    self._record_call(
                        call_id=call_id,
                        request_hash=request_hash,
                        operation=operation,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        prompt_summary=summary,
                        cache_hit=True,
                        status="success",
                        duration_ms=self._elapsed_ms(started_at),
                        response=cached,
                    )
                    return cached

        if request_total_timeout_seconds is not None and request_total_timeout_seconds <= 0:
            raise ValueError("request_total_timeout_seconds must be positive")

        try:
            request_once = self._request_sse_once if stream_response else self._request_once
            deadline = (
                time.monotonic() + request_total_timeout_seconds
                if request_total_timeout_seconds is not None
                else None
            )

            def execute_attempt() -> dict[str, Any]:
                attempt_timeout = request_timeout_seconds
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RetryableServiceError(
                            f"{self.provider} request exceeded total timeout"
                        )
                    attempt_timeout = (
                        remaining
                        if attempt_timeout is None
                        else min(attempt_timeout, remaining)
                    )
                return request_once(
                    method,
                    path,
                    normalized_payload,
                    attempt_timeout,
                    priority,
                )

            stop_policy = stop_after_attempt(max(1, request_attempts or self.max_retries))
            if request_total_timeout_seconds is not None:
                stop_policy |= stop_before_delay(request_total_timeout_seconds)
            response = Retrying(
                stop=stop_policy,
                wait=wait_exponential_jitter(initial=0.5, max=8.0),
                retry=retry_if_exception_type(RetryableServiceError),
                reraise=True,
            )(execute_attempt)
        except Exception as exc:
            safe_error = (
                exc
                if isinstance(exc, ExternalServiceError)
                else ExternalServiceError(f"{self.provider} request failed")
            )
            self._record_call(
                call_id=call_id,
                request_hash=request_hash,
                operation=operation,
                model=model,
                reasoning_effort=reasoning_effort,
                prompt_summary=summary,
                cache_hit=False,
                status="error",
                duration_ms=self._elapsed_ms(started_at),
                error_type=type(safe_error).__name__,
            )
            if safe_error is exc:
                raise
            raise safe_error from exc

        try:
            if response_validator is not None:
                response_validator(response)
        except Exception as exc:
            self.database.quarantine_response(
                call_id=call_id,
                request_hash=request_hash,
                provider=self.provider,
                operation=operation,
                response=response,
            )
            self._record_call(
                call_id=call_id,
                request_hash=request_hash,
                operation=operation,
                model=model,
                reasoning_effort=reasoning_effort,
                prompt_summary=summary,
                cache_hit=False,
                status="error",
                duration_ms=self._elapsed_ms(started_at),
                response=response,
                error_type=type(exc).__name__,
            )
            raise

        if use_cache:
            self.database.set_cached_response(
                request_hash,
                self.provider,
                operation,
                response,
            )
        self._record_call(
            call_id=call_id,
            request_hash=request_hash,
            operation=operation,
            model=model,
            reasoning_effort=reasoning_effort,
            prompt_summary=summary,
            cache_hit=False,
            status="success",
            duration_ms=self._elapsed_ms(started_at),
            response=response,
        )
        return response

    def _request_sse_once(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any],
        request_timeout_seconds: float | None = None,
        priority: int = 10,
    ) -> dict[str, Any]:
        """Collect one server-sent event response through the shared transport boundary."""

        self.rate_limiter.wait()
        timeout_kwargs = (
            {"timeout": request_timeout_seconds} if request_timeout_seconds is not None else {}
        )
        wall_started = time.monotonic()
        chunks: list[dict[str, Any]] = []
        done_received = False
        gate = self.concurrency_gate
        slot = gate.slot(priority=priority) if gate is not None else _null_slot()
        try:
            with slot:
                with self._client.stream(
                    method,
                    path,
                    json=payload or None,
                    **timeout_kwargs,
                ) as response:
                    self._raise_for_status(response)
                    for line in response.iter_lines():
                        if (
                            request_timeout_seconds is not None
                            and time.monotonic() - wall_started > request_timeout_seconds
                        ):
                            raise RetryableServiceError(
                                f"{self.provider} stream exceeded wall-clock timeout"
                            )
                        value = line.strip()
                        if not value.startswith("data:"):
                            continue
                        data = value[5:].strip()
                        if data == "[DONE]":
                            done_received = True
                            break
                        try:
                            decoded = json.loads(data)
                        except ValueError as exc:
                            raise ExternalServiceError(
                                f"{self.provider} returned invalid event JSON"
                            ) from exc
                        if not isinstance(decoded, dict):
                            raise ExternalServiceError(
                                f"{self.provider} returned an invalid event shape"
                            )
                        chunks.append(decoded)
        except httpx.TimeoutException as exc:
            raise RetryableServiceError(f"{self.provider} request timed out") from exc
        except httpx.TransportError as exc:
            raise RetryableServiceError(f"{self.provider} network error") from exc
        if not done_received:
            raise RetryableServiceError(
                f"{self.provider} event stream ended before [DONE]"
            )
        if not chunks:
            raise ExternalServiceError(f"{self.provider} returned an empty event stream")
        return {"chunks": chunks}

    def _request_once(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any],
        request_timeout_seconds: float | None = None,
        priority: int = 10,
    ) -> dict[str, Any]:
        self.rate_limiter.wait()
        timeout_kwargs = (
            {"timeout": request_timeout_seconds} if request_timeout_seconds is not None else {}
        )
        gate = self.concurrency_gate
        slot = gate.slot(priority=priority) if gate is not None else _null_slot()
        try:
            with slot:
                if method.upper() == "GET":
                    response = self._client.request(
                        method, path, params=payload or None, **timeout_kwargs
                    )
                else:
                    response = self._client.request(
                        method, path, json=payload or None, **timeout_kwargs
                    )
        except httpx.TimeoutException as exc:
            raise RetryableServiceError(f"{self.provider} request timed out") from exc
        except httpx.TransportError as exc:
            raise RetryableServiceError(f"{self.provider} network error") from exc

        self._raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise ExternalServiceError(f"{self.provider} returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ExternalServiceError(f"{self.provider} returned an invalid response shape")
        return data

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Apply the same sanitized status policy to JSON and SSE responses."""

        if response.status_code == 429 or response.status_code >= 500:
            if response.status_code == 429 and self.concurrency_gate is not None:
                self.concurrency_gate.note_rate_limit()
            raise RetryableServiceError(
                f"{self.provider} temporary error (HTTP {response.status_code})"
            )
        if response.is_error:
            raise ExternalServiceError(
                f"{self.provider} rejected the request (HTTP {response.status_code})"
            )

    def _record_call(
        self,
        *,
        call_id: str,
        request_hash: str,
        operation: str,
        model: str | None,
        reasoning_effort: str | None,
        prompt_summary: str,
        cache_hit: bool,
        status: str,
        duration_ms: int,
        response: Mapping[str, Any] | None = None,
        error_type: str | None = None,
    ) -> None:
        usage = self._response_usage(response)
        input_tokens = self._int_or_none(usage.get("prompt_tokens", usage.get("input_tokens")))
        output_tokens = self._int_or_none(
            usage.get("completion_tokens", usage.get("output_tokens"))
        )
        total_tokens = self._int_or_none(usage.get("total_tokens"))
        self.database.record_api_call(
            ApiCallRecord(
                call_id=call_id,
                request_hash=request_hash,
                provider=self.provider,
                operation=operation,
                model=model,
                reasoning_effort=reasoning_effort,
                prompt_summary=prompt_summary,
                cache_hit=cache_hit,
                status=status,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                duration_ms=duration_ms,
                result_summary=self._result_summary(response),
                error_type=error_type,
            )
        )
        logger.info(
            "provider=%s operation=%s status=%s cache_hit=%s duration_ms=%s",
            self.provider,
            operation,
            status,
            cache_hit,
            duration_ms,
        )

    def _request_hash(self, method: str, path: str, payload: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            {
                "provider": self.provider,
                "base_url": self.base_url,
                "method": method.upper(),
                "path": path,
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _result_summary(response: Mapping[str, Any] | None) -> str | None:
        if response is None:
            return None
        summary: dict[str, Any] = {"keys": sorted(response.keys())}
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            summary["content_chars"] = len(content) if isinstance(content, str) else 0
        data = response.get("data")
        if isinstance(data, list):
            summary["item_count"] = len(data)
        return json.dumps(summary, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _response_usage(response: Mapping[str, Any] | None) -> Mapping[str, Any]:
        """同时提取普通 JSON 和流式事件末尾的 token 用量。"""

        if response is None:
            return {}
        usage = response.get("usage")
        if isinstance(usage, Mapping):
            return usage
        chunks = response.get("chunks")
        if isinstance(chunks, list):
            for chunk in reversed(chunks):
                if isinstance(chunk, Mapping) and isinstance(chunk.get("usage"), Mapping):
                    return chunk["usage"]
        return {}

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return max(0, round((time.perf_counter() - started_at) * 1000))

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        return value if isinstance(value, int) else None


@contextmanager
def _null_slot():
    """避免为未配置并发闸门的其他服务改变传输行为。"""

    yield
