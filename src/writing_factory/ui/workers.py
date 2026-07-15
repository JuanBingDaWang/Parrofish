"""Reusable QThread workers for all blocking service operations."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

logger = logging.getLogger(__name__)


def _no_stream(_kind: str, _text: str) -> None:
    pass


class TaskCancelled(RuntimeError):
    """Raised by cooperative work after a cancellation request."""


@dataclass(frozen=True, slots=True)
class TaskContext:
    """Thread-safe cancellation and progress hooks supplied to a task."""

    _cancel_event: threading.Event
    _progress_callback: Callable[[int, str], None]
    _stream_callback: Callable[[str, str], None] = _no_stream

    @property
    def is_cancelled(self) -> bool:
        """Return whether the UI requested cooperative cancellation."""

        return self._cancel_event.is_set()

    def check_cancelled(self) -> None:
        """Stop work at a safe boundary when cancellation was requested."""

        if self.is_cancelled:
            raise TaskCancelled("Task cancelled")

    def report_progress(self, percent: int, message: str = "") -> None:
        """Send bounded progress to the GUI thread."""

        self._progress_callback(max(0, min(100, percent)), message)

    def report_stream(self, kind: str, text: str) -> None:
        """Forward incremental model activity without blocking the worker thread."""

        if text:
            self._stream_callback(kind, text)

    def scaled(self, start: int, end: int, *, prefix: str = "") -> TaskContext:
        """Create a child context that maps its progress into one parent range."""

        lower = max(0, min(100, start))
        upper = max(lower, min(100, end))

        def report(percent: int, message: str) -> None:
            mapped = lower + round((upper - lower) * max(0, min(100, percent)) / 100)
            label = f"{prefix}{message}" if message else prefix.removesuffix(" · ")
            self.report_progress(mapped, label)

        return TaskContext(self._cancel_event, report, self._stream_callback)


class Worker(QObject):
    """Execute one callable in its assigned QThread."""

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    streamed = pyqtSignal(str, str)
    finished = pyqtSignal()

    def __init__(self, task: Callable[[TaskContext], Any]) -> None:
        super().__init__()
        self._task = task
        self._cancel_event = threading.Event()

    @pyqtSlot()
    def run(self) -> None:
        """Run the task and convert failures into sanitized UI signals."""

        context = TaskContext(self._cancel_event, self.progress.emit, self.streamed.emit)
        try:
            context.check_cancelled()
            result = self._task(context)
        except TaskCancelled:
            self.failed.emit("任务已取消")
        except Exception as exc:
            logger.exception("Background task failed: %s", type(exc).__name__)
            self.failed.emit(str(exc) or type(exc).__name__)
        else:
            self.succeeded.emit(result)
        finally:
            self.finished.emit()

    def cancel(self) -> None:
        """Request cooperative cancellation without blocking the GUI."""

        self._cancel_event.set()


class BackgroundTaskManager(QObject):
    """Own worker/thread lifetimes and expose one start method to the UI."""

    task_started = pyqtSignal(str)
    task_finished = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tasks: dict[str, tuple[QThread, Worker]] = {}

    def start(
        self,
        task: Callable[[TaskContext], Any],
        *,
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_progress: Callable[[int, str], None] | None = None,
        on_stream: Callable[[str, str], None] | None = None,
    ) -> str:
        """Move a worker to a fresh QThread and start it asynchronously."""

        task_id = str(uuid.uuid4())
        thread = QThread(self)
        worker = Worker(task)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._forget(task_id))
        if on_success is not None:
            worker.succeeded.connect(on_success)
        if on_error is not None:
            worker.failed.connect(on_error)
        if on_progress is not None:
            worker.progress.connect(on_progress)
        if on_stream is not None:
            worker.streamed.connect(on_stream)
        self._tasks[task_id] = (thread, worker)
        self.task_started.emit(task_id)
        thread.start()
        return task_id

    def cancel(self, task_id: str) -> None:
        """Request cancellation for one live task."""

        item = self._tasks.get(task_id)
        if item is not None:
            item[1].cancel()

    def cancel_all(self) -> None:
        """Request cancellation for all live tasks."""

        for _, worker in tuple(self._tasks.values()):
            worker.cancel()

    @property
    def active_count(self) -> int:
        """Return the number of live background threads."""

        return len(self._tasks)

    def _forget(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)
        self.task_finished.emit(task_id)
