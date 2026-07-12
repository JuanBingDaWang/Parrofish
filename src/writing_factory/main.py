"""Desktop process entry point; business logic remains in service modules."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from writing_factory.app import build_application
from writing_factory.ui.main_window import MainWindow
from writing_factory.ui.theme import configure_application_font
from writing_factory.ui.workers import TaskContext


def main() -> int:
    """Start the PyQt6 event loop and close services on exit."""

    application = QApplication(sys.argv)
    application.setApplicationName("写作工厂")
    application.setOrganizationName("WritingFactory")
    configure_application_font(application)
    context = build_application()

    def check_siliconflow():
        return context.siliconflow.chat(
            [
                {"role": "system", "content": "Reply with only OK."},
                {"role": "user", "content": "Connection check."},
            ],
            thinking=False,
            temperature=0.0,
            max_tokens=8,
            seed=7,
            use_cache=False,
        )

    def ingest_document(source_path: Path, task_context: TaskContext):
        return context.ingestion.ingest(
            context.default_kb_id,
            source_path,
            progress=task_context.report_progress,
            check_cancelled=task_context.check_cancelled,
        )

    window = MainWindow(
        check_siliconflow,
        ingest_document=ingest_document,
        list_documents=lambda: context.repository.list_documents(context.default_kb_id),
    )
    application.aboutToQuit.connect(context.close)
    window.show()
    return application.exec()
