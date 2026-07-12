"""QThread worker responsiveness and signal tests."""

from __future__ import annotations

import time

from PyQt6.QtCore import QTimer

from writing_factory.ui.workers import BackgroundTaskManager, TaskContext


def test_background_task_keeps_gui_event_loop_responsive(qtbot) -> None:
    manager = BackgroundTaskManager()
    timer_ticks: list[int] = []
    results: list[str] = []
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: timer_ticks.append(1))
    timer.start()

    def blocking_task(context: TaskContext) -> str:
        context.report_progress(50, "working")
        time.sleep(0.15)
        return "done"

    manager.start(blocking_task, on_success=results.append)
    qtbot.waitUntil(lambda: results == ["done"], timeout=2000)
    qtbot.waitUntil(lambda: manager.active_count == 0, timeout=2000)
    timer.stop()

    assert len(timer_ticks) >= 3
