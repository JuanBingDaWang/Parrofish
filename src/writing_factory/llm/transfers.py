"""Credential-free transfer client for provider-signed artifact URLs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential_jitter

from writing_factory.llm.common import (
    ExternalServiceError,
    RateLimiter,
    RetryableServiceError,
)
from writing_factory.store import ApiCallRecord, Database

logger = logging.getLogger(__name__)


class FileTransferTransport:
    """Upload and download signed URLs without ever attaching API credentials."""

    def __init__(
        self,
        *,
        provider: str,
        database: Database,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        max_retries: int,
        minimum_interval_seconds: float = 0.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.provider = provider
        self.database = database
        self.max_retries = max(1, max_retries)
        self.rate_limiter = RateLimiter(minimum_interval_seconds)
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(
                connect=connect_timeout_seconds,
                read=read_timeout_seconds,
                write=read_timeout_seconds,
                pool=connect_timeout_seconds,
            ),
            follow_redirects=True,
        )

    def close(self) -> None:
        """Release the credential-free connection pool."""

        if self._owns_client:
            self._client.close()

    def upload_file(
        self,
        upload_url: str,
        file_path: Path,
        *,
        operation: str,
        content_type: str | None = None,
    ) -> None:
        """Upload a local file using exactly the headers allowed by its signature."""

        path = file_path.resolve(strict=True)
        metadata = {"bytes": path.stat().st_size, "suffix": path.suffix.lower()}
        self._run_transfer(
            operation=operation,
            request_fingerprint=f"upload:{metadata}",
            metadata=metadata,
            action=lambda: self._upload_once(upload_url, path, content_type),
        )

    def download_file(
        self,
        download_url: str,
        destination: Path,
        *,
        operation: str,
    ) -> Path:
        """Download atomically so interrupted artifacts are never treated as complete."""

        target = destination.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        try:
            self._run_transfer(
                operation=operation,
                request_fingerprint=f"download:{target.suffix.lower()}",
                metadata={"suffix": target.suffix.lower()},
                action=lambda: self._download_once(download_url, temporary),
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def _run_transfer(
        self,
        *,
        operation: str,
        request_fingerprint: str,
        metadata: dict[str, object],
        action,
    ) -> None:
        request_hash = hashlib.sha256(
            f"{self.provider}:{operation}:{request_fingerprint}".encode()
        ).hexdigest()
        call_id = str(uuid.uuid4())
        started_at = time.perf_counter()
        status = "success"
        error_type = None
        try:
            Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(initial=0.5, max=8.0),
                retry=retry_if_exception_type(RetryableServiceError),
                reraise=True,
            )(action)
        except Exception as exc:
            status = "error"
            safe_error = (
                exc
                if isinstance(exc, ExternalServiceError)
                else ExternalServiceError(f"{self.provider} file transfer failed")
            )
            error_type = type(safe_error).__name__
            self._record(
                call_id,
                request_hash,
                operation,
                metadata,
                status,
                started_at,
                error_type,
            )
            if safe_error is exc:
                raise
            raise safe_error from exc
        self._record(
            call_id,
            request_hash,
            operation,
            metadata,
            status,
            started_at,
            error_type,
        )

    def _upload_once(self, upload_url: str, path: Path, content_type: str | None) -> None:
        self.rate_limiter.wait()
        headers = {"Content-Type": content_type} if content_type is not None else None
        try:
            with path.open("rb") as source:
                response = self._client.put(upload_url, content=source, headers=headers)
        except (OSError, httpx.TimeoutException, httpx.NetworkError) as exc:
            raise RetryableServiceError(f"{self.provider} upload network error") from exc
        self._check_response(response, "upload")

    def _download_once(self, download_url: str, temporary: Path) -> None:
        self.rate_limiter.wait()
        try:
            with self._client.stream("GET", download_url) as response:
                self._check_response(response, "download")
                with temporary.open("wb") as target:
                    for block in response.iter_bytes():
                        target.write(block)
        except (OSError, httpx.TimeoutException, httpx.NetworkError) as exc:
            raise RetryableServiceError(f"{self.provider} download network error") from exc

    def _check_response(self, response: httpx.Response, action: str) -> None:
        if response.status_code == 429 or response.status_code >= 500:
            raise RetryableServiceError(
                f"{self.provider} {action} temporary error (HTTP {response.status_code})"
            )
        if response.is_error:
            raise ExternalServiceError(
                f"{self.provider} {action} was rejected (HTTP {response.status_code})"
            )

    def _record(
        self,
        call_id: str,
        request_hash: str,
        operation: str,
        metadata: dict[str, object],
        status: str,
        started_at: float,
        error_type: str | None,
    ) -> None:
        duration_ms = max(0, round((time.perf_counter() - started_at) * 1000))
        self.database.record_api_call(
            ApiCallRecord(
                call_id=call_id,
                request_hash=request_hash,
                provider=self.provider,
                operation=operation,
                model=None,
                reasoning_effort=None,
                prompt_summary=json.dumps(metadata, sort_keys=True),
                cache_hit=False,
                status=status,
                duration_ms=duration_ms,
                error_type=error_type,
            )
        )
        logger.info(
            "provider=%s operation=%s status=%s duration_ms=%s",
            self.provider,
            operation,
            status,
            duration_ms,
        )
