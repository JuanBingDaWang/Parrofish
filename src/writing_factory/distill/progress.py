"""Thread-safe weighted progress aggregation for parallel distillation branches."""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping


class WeightedProgress:
    """Map independently reported branch percentages onto one monotonic range."""

    def __init__(
        self,
        callback: Callable[[int, str], None],
        *,
        start: int,
        end: int,
        weights: Mapping[str, float],
    ) -> None:
        self._callback = callback
        self._start = start
        self._end = max(start, end)
        self._weights = {key: max(0.0, value) for key, value in weights.items()}
        if not any(self._weights.values()):
            self._weights = {key: 1.0 for key in weights}
        self._progress = {key: 0 for key in weights}
        self._reported = start
        self._lock = threading.Lock()

    def branch(self, key: str) -> Callable[[int, str], None]:
        if key not in self._weights:
            raise KeyError(key)

        def report(percent: int, message: str = "") -> None:
            self.update(key, percent, message)

        return report

    def update(self, key: str, percent: int, message: str = "") -> None:
        with self._lock:
            bounded = max(self._progress[key], min(100, max(0, percent)))
            self._progress[key] = bounded
            total_weight = sum(self._weights.values()) or 1.0
            fraction = sum(
                self._weights[name] * self._progress[name] / 100
                for name in self._weights
            ) / total_weight
            overall = self._start + round((self._end - self._start) * fraction)
            overall = max(self._reported, overall)
            self._reported = overall
        self._callback(overall, message)

    def complete(self, key: str, message: str = "") -> None:
        self.update(key, 100, message)
