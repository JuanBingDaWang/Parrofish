"""全局 SiliconFlow 并发闸门和运行时设置测试。"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from writing_factory.llm.common import DynamicConcurrencyGate
from writing_factory.store import Database, RuntimeSettingsRepository


def test_dynamic_gate_caps_all_parallel_workers() -> None:
    gate = DynamicConcurrencyGate(2)
    active = 0
    peak = 0
    lock = threading.Lock()

    def work() -> None:
        nonlocal active, peak
        with gate.slot():
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda _value: work(), range(6)))

    assert peak == 2
    assert gate.peak == 2


def test_dynamic_gate_applies_new_limit_without_recreation() -> None:
    gate = DynamicConcurrencyGate(3)
    gate.set_limit(1)

    assert gate.limit == 1


def test_rate_limit_reduces_only_effective_concurrency() -> None:
    gate = DynamicConcurrencyGate(4)

    gate.note_rate_limit()

    assert gate.limit == 4
    assert gate.effective_limit == 3


def test_runtime_concurrency_setting_persists_in_sqlite(tmp_path) -> None:
    database = Database(tmp_path / "settings.db")
    database.initialize()
    settings = RuntimeSettingsRepository(database)

    settings.set("siliconflow_max_concurrency", 5)

    assert RuntimeSettingsRepository(database).get("siliconflow_max_concurrency") == 5
