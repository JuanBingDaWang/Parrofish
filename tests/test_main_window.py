"""Desktop shell interaction tests."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtCore import Qt

from writing_factory.kb.models import IngestResult
from writing_factory.llm.models import ChatResult, TokenUsage
from writing_factory.ui.main_window import MainWindow


def test_connection_button_runs_check_in_background(qtbot) -> None:
    def check_connection() -> ChatResult:
        time.sleep(0.05)
        return ChatResult(
            content="OK",
            model="test-model",
            usage=TokenUsage(total_tokens=3),
        )

    window = MainWindow(check_connection)
    qtbot.addWidget(window)
    window.show()

    qtbot.mouseClick(window.check_button, Qt.MouseButton.LeftButton)
    assert not window.check_button.isEnabled()
    qtbot.waitUntil(lambda: window.siliconflow_status.text() == "可用", timeout=2000)

    assert window.check_button.isEnabled()
    assert "3 tokens" in window.statusBar().currentMessage()
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)


def test_document_import_updates_table_without_blocking(qtbot, tmp_path: Path) -> None:
    documents: list[dict[str, object]] = []
    source = tmp_path / "资料.txt"
    source.write_text("测试", encoding="utf-8")

    def ingest(path: Path, context) -> IngestResult:
        assert path == source
        context.report_progress(50, "索引中")
        time.sleep(0.05)
        documents.append(
            {
                "doc_id": "doc",
                "filename": source.name,
                "status": "ready",
                "chunk_count": 1,
                "ingest_date": "2026-07-12T10:00:00+00:00",
            }
        )
        return IngestResult(
            job_id="job",
            kb_id="kb",
            doc_id="doc",
            child_chunk_count=1,
        )

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        ingest_document=ingest,
        list_documents=lambda: documents,
    )
    qtbot.addWidget(window)
    window.show()

    page = window.knowledge_page
    page.start_ingestion(source)
    assert not page.import_button.isEnabled()
    qtbot.waitUntil(lambda: page.document_table.rowCount() == 1, timeout=2000)

    assert page.document_table.item(0, 0).text() == "资料.txt"
    assert page.document_table.item(0, 1).text() == "可检索"
    assert window.persona_page.source_table.rowCount() == 1
    assert page.import_button.isEnabled()
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)


def test_batch_document_import_runs_sequentially(qtbot, tmp_path: Path) -> None:
    documents: list[dict[str, object]] = []
    sources = [tmp_path / "一.pdf", tmp_path / "二.pdf"]
    for source in sources:
        source.write_bytes(b"fixture")
    active = 0
    max_active = 0
    received: list[Path] = []

    def ingest(path: Path, context) -> IngestResult:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        received.append(path)
        context.report_progress(50, "解析中")
        time.sleep(0.03)
        documents.append(
            {
                "doc_id": f"doc_{len(documents)}",
                "filename": path.name,
                "status": "ready",
                "chunk_count": 2,
                "ingest_date": "2026-07-12T10:00:00+00:00",
            }
        )
        active -= 1
        return IngestResult(
            job_id=f"job_{len(documents)}",
            kb_id="kb",
            doc_id=str(documents[-1]["doc_id"]),
            child_chunk_count=2,
        )

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        ingest_document=ingest,
        list_documents=lambda: documents,
    )
    qtbot.addWidget(window)
    window.show()

    window.knowledge_page.start_ingestions(sources)
    qtbot.waitUntil(
        lambda: window.knowledge_page.document_table.rowCount() == 2,
        timeout=3000,
    )

    assert received == sources
    assert max_active == 1
    assert window.persona_page.source_table.rowCount() == 2
    assert "2 个文件" in window.statusBar().currentMessage()
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)


def test_entering_persona_page_refreshes_external_document_changes(qtbot) -> None:
    documents: list[dict[str, object]] = []
    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        list_documents=lambda: documents,
    )
    qtbot.addWidget(window)
    window.show()
    assert window.persona_page.source_table.rowCount() == 0
    documents.append(
        {
            "doc_id": "doc_new",
            "filename": "新论文.pdf",
            "status": "ready",
            "chunk_count": 3,
        }
    )

    window.navigation.setCurrentRow(2)

    assert window.persona_page.source_table.rowCount() == 1
    assert window.persona_page.source_table.item(0, 1).text() == "新论文.pdf"


def test_persona_page_distills_checked_sources_in_background(qtbot) -> None:
    profiles: list[dict[str, object]] = []
    received: list[tuple[str, str, set[str]]] = []

    def distill(name, mode, doc_ids, context):
        received.append((name, mode, doc_ids))
        context.report_progress(60, "归并")
        time.sleep(0.05)
        profiles.append(
            {
                "name": name,
                "mode": mode,
                "status": "ready",
                "model_count": 3,
                "research_date": "2026-07-12",
            }
        )
        return SimpleNamespace(persona=SimpleNamespace(mental_models=[1, 2, 3]))

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        list_documents=lambda: [
            {
                "doc_id": "doc_one",
                "filename": "论文.pdf",
                "status": "ready",
                "chunk_count": 4,
            }
        ],
        distill_persona=distill,
        list_personas=lambda: profiles,
    )
    qtbot.addWidget(window)
    window.show()
    page = window.persona_page
    page.name_input.setText("叶芃")

    assert page.distill_button.isEnabled()
    page.start_distillation()
    qtbot.waitUntil(lambda: page.profile_table.rowCount() == 1, timeout=2000)

    assert received == [("叶芃", "person", {"doc_one"})]
    assert page.profile_table.item(0, 0).text() == "叶芃"
    assert page.profile_table.item(0, 3).text() == "3"
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)


def test_persona_fidelity_check_runs_in_background_and_refreshes_score(qtbot) -> None:
    profiles = [
        {
            "persona_id": "persona_one",
            "name": "叶芃",
            "mode": "person",
            "status": "ready",
            "model_count": 3,
            "fidelity_score": None,
            "research_date": "2026-07-12",
        }
    ]
    received: list[str] = []

    def evaluate(persona_id, context):
        received.append(persona_id)
        context.report_progress(50, "中性评分")
        time.sleep(0.05)
        profiles[0]["fidelity_score"] = 88
        return SimpleNamespace(total=88)

    window = MainWindow(
        lambda: ChatResult(content="OK", model="test"),
        evaluate_persona=evaluate,
        list_personas=lambda: profiles,
    )
    qtbot.addWidget(window)
    window.show()
    page = window.persona_page
    page.profile_table.selectRow(0)

    assert page.evaluate_button.isEnabled()
    qtbot.mouseClick(page.evaluate_button, Qt.MouseButton.LeftButton)
    assert not page.evaluate_button.isEnabled()
    qtbot.waitUntil(lambda: page.profile_table.item(0, 4).text() == "88/100", timeout=2000)

    assert received == ["persona_one"]
    assert page.evaluate_button.isEnabled()
    qtbot.waitUntil(lambda: window._tasks.active_count == 0, timeout=2000)
