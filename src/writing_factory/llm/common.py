"""Shared sanitized errors and worker-thread rate limiting."""

from __future__ import annotations

import threading
import time


class ExternalServiceError(RuntimeError):
    """A sanitized provider failure safe to show in the desktop UI."""


class RetryableServiceError(ExternalServiceError):
    """A transient error eligible for bounded retry."""


class RateLimiter:
    """Serialize request starts when a minimum interval is configured."""

    def __init__(self, minimum_interval_seconds: float) -> None:
        self._minimum_interval = max(0.0, minimum_interval_seconds)
        self._last_request_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Wait only on the worker thread that is making the request."""

        with self._lock:
            delay = self._minimum_interval - (time.monotonic() - self._last_request_at)
            if delay > 0:
                time.sleep(delay)
            self._last_request_at = time.monotonic()
