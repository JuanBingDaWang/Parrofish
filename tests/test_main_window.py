"""Desktop shell interaction tests."""

from __future__ import annotations

import time
from pathlib import Path

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
    assert page.import_button.isEnabled()
