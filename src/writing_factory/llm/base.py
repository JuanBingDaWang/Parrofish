"""Retrying, cached, observable HTTP transport for external services."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import SecretStr
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential_jitter

from writing_factory.llm.common import (
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
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.database = database
        self.max_retries = max(1, max_retries)
        self.rate_limiter = RateLimiter(minimum_interval_seconds)
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

        try:
            response = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(initial=0.5, max=8.0),
                retry=retry_if_exception_type(RetryableServiceError),
                reraise=True,
            )(self._request_once, method, path, normalized_payload)
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

    def _request_once(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.rate_limiter.wait()
        try:
            if method.upper() == "GET":
                response = self._client.request(method, path, params=payload or None)
            else:
                response = self._client.request(method, path, json=payload or None)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise RetryableServiceError(f"{self.provider} network error") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise RetryableServiceError(
                f"{self.provider} temporary error (HTTP {response.status_code})"
            )
        if response.is_error:
            raise ExternalServiceError(
                f"{self.provider} rejected the request (HTTP {response.status_code})"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ExternalServiceError(f"{self.provider} returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ExternalServiceError(f"{self.provider} returned an invalid response shape")
        return data

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
        usage = response.get("usage", {}) if response else {}
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
    def _elapsed_ms(started_at: float) -> int:
        return max(0, round((time.perf_counter() - started_at) * 1000))

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        return value if isinstance(value, int) else None
