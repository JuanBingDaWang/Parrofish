"""Shared sanitized errors and worker-thread rate limiting."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


class ExternalServiceError(RuntimeError):
    """A sanitized provider failure safe to show in the desktop UI."""


class RetryableServiceError(ExternalServiceError):
    """A transient error eligible for bounded retry."""


class IncompleteStreamError(RetryableServiceError):
    """An SSE response ended without a trustworthy terminal state."""

    def __init__(self, message: str, *, response: dict | None = None) -> None:
        super().__init__(message)
        self.response = response


class RateLimiter:
    """Serialize request starts when a minimum interval is configured."""

    def __init__(self, minimum_interval_seconds: float) -> None:
        self._minimum_interval = max(0.0, minimum_interval_seconds)
        self._last_request_at = 0.0
        self._lock = threading.Lock()

    def wait(self, check_cancelled: Callable[[], None] | None = None) -> None:
        """Wait only on the worker thread that is making the request."""

        with self._lock:
            delay = self._minimum_interval - (time.monotonic() - self._last_request_at)
            deadline = time.monotonic() + max(0.0, delay)
            while True:
                if check_cancelled is not None:
                    check_cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.1, remaining))
            self._last_request_at = time.monotonic()


@dataclass(order=True, slots=True)
class _ConcurrencyWaiter:
    """按优先级和到达顺序排列的等待请求。"""

    priority: int
    sequence: int
    token: object = field(compare=False)


class DynamicConcurrencyGate:
    """跨业务模块共享、可在运行时调整的全局请求并发上限。"""

    def __init__(self, limit: int) -> None:
        self._configured_limit = self._validated(limit)
        self._effective_limit = self._configured_limit
        self._recover_at = 0.0
        self._recovery_seconds = 30.0
        self._active = 0
        self._peak = 0
        self._sequence = 0
        self._waiters: list[_ConcurrencyWaiter] = []
        self._condition = threading.Condition()

    @staticmethod
    def _validated(value: int) -> int:
        if not 1 <= value <= 8:
            raise ValueError("SiliconFlow 最大并发数必须在 1 至 8 之间")
        return value

    @property
    def limit(self) -> int:
        with self._condition:
            return self._configured_limit

    @property
    def effective_limit(self) -> int:
        with self._condition:
            self._refresh_adaptive_limit()
            return self._effective_limit

    @property
    def active(self) -> int:
        with self._condition:
            return self._active

    @property
    def peak(self) -> int:
        with self._condition:
            return self._peak

    def set_limit(self, value: int) -> None:
        """对新请求应用上限；不取消已经在途的请求。"""

        limit = self._validated(value)
        with self._condition:
            self._configured_limit = limit
            if self._recover_at == 0.0:
                self._effective_limit = limit
            else:
                self._effective_limit = min(self._effective_limit, limit)
            self._condition.notify_all()

    def note_rate_limit(self) -> None:
        """收到 429 后临时降低有效并发，并在冷却后逐级恢复。"""

        with self._condition:
            self._effective_limit = max(1, self._effective_limit - 1)
            self._recover_at = time.monotonic() + self._recovery_seconds
            self._condition.notify_all()

    @contextmanager
    def slot(
        self,
        *,
        priority: int = 10,
        check_cancelled: Callable[[], None] | None = None,
    ) -> Iterator[None]:
        """等待一个槽位；较小的 priority 值优先。"""

        token = object()
        with self._condition:
            self._sequence += 1
            waiter = _ConcurrencyWaiter(priority, self._sequence, token)
            self._waiters.append(waiter)
            self._waiters.sort()
            try:
                while True:
                    if check_cancelled is not None:
                        check_cancelled()
                    self._refresh_adaptive_limit()
                    if self._active < self._effective_limit and self._waiters[0].token is token:
                        break
                    timeout = None
                    if self._recover_at:
                        timeout = max(0.01, self._recover_at - time.monotonic())
                    if check_cancelled is not None:
                        timeout = min(0.1, timeout) if timeout is not None else 0.1
                    self._condition.wait(timeout=timeout)
            except Exception:
                self._waiters = [item for item in self._waiters if item.token is not token]
                self._condition.notify_all()
                raise
            self._waiters.pop(0)
            self._active += 1
            self._peak = max(self._peak, self._active)
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def _refresh_adaptive_limit(self) -> None:
        now = time.monotonic()
        if not self._recover_at or now < self._recover_at:
            return
        self._effective_limit = min(
            self._configured_limit,
            self._effective_limit + 1,
        )
        self._recover_at = (
            now + self._recovery_seconds if self._effective_limit < self._configured_limit else 0.0
        )
