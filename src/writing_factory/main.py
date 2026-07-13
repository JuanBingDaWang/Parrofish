"""Desktop process entry point; business logic remains in service modules."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from writing_factory.app import build_application
from writing_factory.distill.models import PersonaSpec
from writing_factory.distill.quality import run_static_quality_check
from writing_factory.distill.serialization import render_persona_markdown
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
            priority=0,
        )

    def ingest_document(source_path: Path, task_context: TaskContext):
        return context.ingestion.ingest(
            context.default_kb_id,
            source_path,
            progress=task_context.report_progress,
            check_cancelled=task_context.check_cancelled,
        )

    def distill_persona(
        name,
        mode,
        doc_ids,
        control_doc_ids,
        domain,
        task_context: TaskContext,
    ):
        return context.distillation.distill(
            kb_id=context.default_kb_id,
            name=name,
            mode=mode,
            doc_ids=doc_ids,
            control_doc_ids=control_doc_ids,
            domain=domain,
            progress=task_context.report_progress,
            check_cancelled=task_context.check_cancelled,
        )

    def delete_documents(doc_ids: set[str], task_context: TaskContext):
        return context.ingestion.delete_documents(
            context.default_kb_id,
            doc_ids,
            progress=task_context.report_progress,
            check_cancelled=task_context.check_cancelled,
        )

    def delete_personas(persona_ids: set[str], task_context: TaskContext) -> int:
        task_context.check_cancelled()
        task_context.report_progress(20, "删除档案")
        removed = context.persona_repository.delete_personas(
            context.default_kb_id,
            persona_ids,
        )
        task_context.report_progress(100, "删除完成")
        return removed

    def save_persona(persona_id: str, persona: PersonaSpec) -> tuple[PersonaSpec, str]:
        report = run_static_quality_check(persona)
        # 人工编辑可以保留历史档案的原语言，其他结构与证据硬门仍必须通过。
        failed = [
            name
            for name, passed in report.checks.items()
            if not passed and name != "output_language"
        ]
        if failed:
            raise ValueError(f"档案未通过质量检查：{', '.join(failed)}")
        markdown = render_persona_markdown(persona)
        context.persona_repository.update_ready(
            persona_id=persona_id,
            persona=persona,
            markdown=markdown,
        )
        context.persona_repository.save_evaluation(
            persona_id=persona_id,
            evaluation_type="nuwa_static",
            result_json=report.model_dump_json(),
        )
        return persona, markdown

    def evaluate_persona(persona_id: str, task_context: TaskContext):
        task_context.check_cancelled()
        task_context.report_progress(10, "设计自检问题")
        result = context.fidelity.evaluate(persona_id)
        task_context.report_progress(100, "自检完成")
        return result

    window = MainWindow(
        check_siliconflow,
        ingest_document=ingest_document,
        list_documents=lambda: context.repository.list_documents(context.default_kb_id),
        delete_documents=delete_documents,
        distill_persona=distill_persona,
        evaluate_persona=evaluate_persona,
        list_personas=lambda: context.persona_repository.list_personas(context.default_kb_id),
        delete_personas=delete_personas,
        load_persona=context.persona_repository.load_ready,
        save_persona=save_persona,
        load_runtime_persona=context.persona_repository.load_runtime,
        list_persona_versions=context.persona_repository.list_versions,
        get_siliconflow_concurrency=lambda: context.siliconflow_gate.limit,
        set_siliconflow_concurrency=context.set_siliconflow_concurrency,
    )
    application.aboutToQuit.connect(context.close)
    window.show()
    return application.exec()
