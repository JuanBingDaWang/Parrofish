"""Desktop process entry point; business logic remains in service modules."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from writing_factory.app import build_application
from writing_factory.distill.fidelity_models import (
    FidelityStageProgress,
    encode_fidelity_progress,
)
from writing_factory.distill.models import PersonaSpec
from writing_factory.distill.options import DistillationOptions
from writing_factory.distill.quality import run_static_quality_check
from writing_factory.distill.serialization import render_persona_markdown
from writing_factory.generate.models import GenerationOptions
from writing_factory.generate.source_policy import build_persona_generation_source_policy
from writing_factory.ui.branding import configure_application_identity
from writing_factory.ui.main_window import MainWindow
from writing_factory.ui.theme import configure_application_font
from writing_factory.ui.workers import TaskContext


def main() -> int:
    """Start the PyQt6 event loop and close services on exit."""

    application = QApplication(sys.argv)
    configure_application_identity(application)
    configure_application_font(application)
    app_context = build_application()

    def check_siliconflow():
        return app_context.siliconflow.list_models("chat")

    def ingest_document(source_path: Path, task_context: TaskContext):
        with app_context.siliconflow.freeze_runtime_settings():
            return app_context.ingestion.ingest(
                app_context.default_kb_id,
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
        options: DistillationOptions,
        task_context: TaskContext,
    ):
        with (
            app_context.siliconflow.freeze_runtime_settings(),
            app_context.siliconflow.observe_stream(
                task_context.report_stream,
                check_cancelled=task_context.check_cancelled,
            ),
        ):
            return app_context.distillation.distill(
                kb_id=app_context.default_kb_id,
                name=name,
                mode=mode,
                doc_ids=doc_ids,
                control_doc_ids=control_doc_ids,
                domain=domain,
                options=options,
                progress=task_context.report_progress,
                check_cancelled=task_context.check_cancelled,
            )

    def resume_persona(persona_id: str, task_context: TaskContext):
        with (
            app_context.siliconflow.freeze_runtime_settings(),
            app_context.siliconflow.observe_stream(
                task_context.report_stream,
                check_cancelled=task_context.check_cancelled,
            ),
        ):
            return app_context.distillation.resume(
                kb_id=app_context.default_kb_id,
                persona_id=persona_id,
                progress=task_context.report_progress,
                check_cancelled=task_context.check_cancelled,
            )

    def upgrade_persona(
        base_persona_id: str,
        doc_ids: set[str],
        control_doc_ids: set[str],
        domain: str,
        options: DistillationOptions,
        task_context: TaskContext,
    ):
        with (
            app_context.siliconflow.freeze_runtime_settings(),
            app_context.siliconflow.observe_stream(
                task_context.report_stream,
                check_cancelled=task_context.check_cancelled,
            ),
        ):
            return app_context.distillation.upgrade(
                kb_id=app_context.default_kb_id,
                base_persona_id=base_persona_id,
                doc_ids=doc_ids,
                control_doc_ids=control_doc_ids,
                domain=domain,
                options=options,
                progress=task_context.report_progress,
                check_cancelled=task_context.check_cancelled,
            )

    def delete_documents(doc_ids: set[str], task_context: TaskContext):
        return app_context.ingestion.delete_documents(
            app_context.default_kb_id,
            doc_ids,
            progress=task_context.report_progress,
            check_cancelled=task_context.check_cancelled,
        )

    def retrieve(
        query: str,
        *,
        use_rewrite: bool,
        use_hyde: bool,
        context: TaskContext | None = None,
        **_extra: object,
    ):
        from writing_factory.kb.models import RetrievalRequest

        request = RetrievalRequest(
            kb_id=app_context.default_kb_id,
            query=query,
            use_rewrite=use_rewrite,
            use_hyde=use_hyde,
        )
        with app_context.siliconflow.freeze_runtime_settings():
            return app_context.hybrid_retriever.search(
                request,
                progress=context.report_progress if context is not None else lambda _p, _m: None,
                check_cancelled=context.check_cancelled if context is not None else lambda: None,
            )

    def delete_personas(persona_ids: set[str], task_context: TaskContext) -> int:
        task_context.check_cancelled()
        task_context.report_progress(20, "删除档案")
        removed = app_context.persona_repository.delete_personas(
            app_context.default_kb_id,
            persona_ids,
        )
        task_context.report_progress(100, "删除完成")
        return removed

    def send_chat_message(
        *,
        conversation_id: str,
        user_message: str,
        context: TaskContext,
    ):
        with (
            app_context.siliconflow.freeze_runtime_settings(),
            app_context.siliconflow.observe_stream(
                context.report_stream,
                check_cancelled=context.check_cancelled,
            ),
        ):
            return app_context.author_chat.send(
                conversation_id=conversation_id,
                user_message=user_message,
                progress=context.report_progress,
                check_cancelled=context.check_cancelled,
            )

    def verify_chat_message(*, message_id: str, context: TaskContext):
        context.report_progress(10, "准备中性核验")
        with (
            app_context.siliconflow.freeze_runtime_settings(),
            app_context.siliconflow.observe_stream(
                context.report_stream,
                check_cancelled=context.check_cancelled,
            ),
        ):
            result = app_context.author_chat.verify(message_id)
        context.report_progress(100, "中性核验完成")
        return result

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
        app_context.persona_repository.update_ready(
            persona_id=persona_id,
            persona=persona,
            markdown=markdown,
        )
        app_context.persona_repository.save_evaluation(
            persona_id=persona_id,
            evaluation_type="nuwa_static",
            result_json=report.model_dump_json(),
        )
        return persona, markdown

    def evaluate_persona(persona_id: str, task_context: TaskContext):
        task_context.check_cancelled()

        stage_labels = {
            "design": "设计自检问题",
            "answer": "档案盲测回答",
            "judge": "中性独立评判",
        }
        stage_progress = {
            "design": {"started": 5, "restored": 32, "completed": 32, "failed": 32},
            "answer": {"started": 36, "restored": 64, "completed": 64, "failed": 64},
            "judge": {"started": 68, "restored": 96, "completed": 96, "failed": 96},
        }

        def report_stage(event: FidelityStageProgress) -> None:
            percent = stage_progress[event.stage][event.state]
            task_context.report_progress(percent, encode_fidelity_progress(event))
            state_text = {
                "started": "开始",
                "restored": "从断点恢复",
                "completed": "完成",
                "failed": "失败",
            }[event.state]
            duration = _format_elapsed(event.duration_ms)
            suffix = "" if event.state == "started" else f" · 耗时 {duration}"
            task_context.report_stream(
                "status::档案自检",
                f"{stage_labels[event.stage]} · {state_text}{suffix}",
            )

        with (
            app_context.siliconflow.freeze_runtime_settings(),
            app_context.siliconflow.observe_stream(
                task_context.report_stream,
                check_cancelled=task_context.check_cancelled,
            ),
        ):
            result = app_context.fidelity.evaluate(
                persona_id,
                progress=report_stage,
                check_cancelled=task_context.check_cancelled,
            )
        task_context.report_progress(100, "自检完成")
        return result

    def run_writing_pipeline(
        persona_id: str,
        task_description: str,
        domain: str,
        context: TaskContext,
        task_id: str,
        selected_doc_ids: set[str],
        explicitly_allowed_persona_doc_ids: set[str],
        generation_options: dict | None = None,
        resume: bool = False,
    ) -> dict:
        from writing_factory.orchestration.pipeline_runner import (
            run_writing_pipeline_with_progress,
            summarize_writing_state,
        )

        app_context.project_repository.mark_task_status(task_id, "running")

        def persist_checkpoint(state: dict) -> None:
            app_context.project_repository.update_task_state(task_id, state)
            context.report_stream(
                "pipeline_state",
                json.dumps(summarize_writing_state(state), ensure_ascii=False),
            )

        try:
            with (
                app_context.siliconflow.freeze_runtime_settings(),
                app_context.siliconflow.observe_stream(
                    context.report_stream,
                    check_cancelled=context.check_cancelled,
                ),
            ):
                result = run_writing_pipeline_with_progress(
                    persona_id=persona_id,
                    task_description=task_description,
                    domain=domain,
                    context=context,
                    siliconflow=app_context.siliconflow,
                    retriever=app_context.hybrid_retriever,
                    persona_repository=app_context.persona_repository,
                    kb_repository=app_context.repository,
                    checkpoint_dir=app_context.settings.data_dir / "checkpoints",
                    kb_id=app_context.default_kb_id,
                    citation_style=app_context.settings.citation_style,
                    task_id=task_id,
                    selected_doc_ids=selected_doc_ids,
                    explicitly_allowed_persona_doc_ids=explicitly_allowed_persona_doc_ids,
                    generation_options=GenerationOptions.model_validate(generation_options or {}),
                    bocha_client=app_context.bocha,
                    web_search_result_count=app_context.get_web_search_result_count(),
                    state_callback=persist_checkpoint,
                    resume=resume,
                )
        except Exception as exc:
            if context.is_cancelled:
                app_context.project_repository.mark_task_status(task_id, "cancelled")
            else:
                app_context.project_repository.mark_task_status(task_id, "error", str(exc))
            raise
        app_context.project_repository.update_task_state(task_id, result)
        return result

    def preview_source_selection(
        persona_id: str,
        selected_doc_ids: set[str],
        explicitly_allowed_persona_doc_ids: set[str],
    ) -> dict[str, int]:
        policy = build_persona_generation_source_policy(
            persona_repository=app_context.persona_repository,
            persona_id=persona_id,
            selected_task_doc_ids=selected_doc_ids,
            explicitly_allowed_persona_doc_ids=explicitly_allowed_persona_doc_ids,
        )
        return {
            "selected_count": len(selected_doc_ids),
            "isolated_count": len(selected_doc_ids & policy.excluded_persona_doc_ids),
            "usable_count": len(policy.allowed_task_doc_ids),
        }

    def load_writing_task(task_id: str) -> dict[str, object] | None:
        from writing_factory.orchestration.pipeline_runner import load_latest_writing_state

        record = app_context.project_repository.get_task(task_id)
        if record is None or record.get("state") is not None:
            return record
        checkpoint_state = load_latest_writing_state(
            app_context.settings.data_dir / "checkpoints",
            task_id,
        )
        if checkpoint_state is not None:
            record["state"] = checkpoint_state
        return record

    def evaluate_generation(
        thesis_json: str,
        draft_json: str,
        context: dict,
        task_context: TaskContext,
    ) -> dict | None:
        """Run Stage 7 evaluation on a completed draft.

        This is optional — the writing pipeline can run without it.
        Returns a dict with keys: faithfulness, judge, judge_rationale, injection
        or None if evaluation is not available.
        """
        try:
            from writing_factory.eval.run_eval import EvaluationRunner
            from writing_factory.eval.traceability import evidence_context_from_state

            runner = EvaluationRunner(
                siliconflow=app_context.siliconflow,
                database=app_context.database,
            )

            # Parse draft text
            draft_data = json.loads(draft_json) if isinstance(draft_json, str) else draft_json
            sections = draft_data.get("sections", [])
            draft_text = "\n\n".join(
                s.get("polished_text", "") for s in sections if s.get("polished_text")
            )
            thesis_text = ""
            thesis_data = json.loads(thesis_json) if isinstance(thesis_json, str) else thesis_json
            if isinstance(thesis_data, dict):
                thesis_text = thesis_data.get("thesis_text", thesis_data.get("angle", ""))

            evidence_context = evidence_context_from_state(context)

            task_context.report_progress(10, "引用可溯性评估")
            traceability = runner.evaluate_traceability(
                context,
                persist=True,
                kb_id=app_context.default_kb_id,
                pipeline_run_id=context.get("task_id"),
            )

            with (
                app_context.siliconflow.freeze_runtime_settings(),
                app_context.siliconflow.observe_stream(
                    task_context.report_stream,
                    check_cancelled=task_context.check_cancelled,
                ),
            ):
                # Faithfulness
                task_context.report_progress(20, "忠实度评估")
                faithfulness_result = runner.evaluate_faithfulness(
                    question=thesis_text or "（无论点）",
                    answer=draft_text or "（无正文）",
                    context=evidence_context,
                    persist=True,
                    kb_id=app_context.default_kb_id,
                    pipeline_run_id=context.get("task_id"),
                )

                # LLM Judge
                task_context.report_progress(50, "裁判评分")
                judge_result = runner.evaluate_judge(
                    thesis=thesis_text or "（无论点）",
                    draft=draft_text or "（无正文）",
                    persist=True,
                    kb_id=app_context.default_kb_id,
                    pipeline_run_id=context.get("task_id"),
                )

                # Injection
                task_context.report_progress(80, "注入检测")
                injection_verdict = runner.check_injection(draft_text)

            task_context.report_progress(100, "评估完成")
            evaluation = {
                "faithfulness": round(faithfulness_result.score, 4)
                if faithfulness_result
                else None,
                "traceability": round(traceability.verified_support_ratio, 4),
                "hallucination_rate": round(traceability.hallucination_rate, 4),
                "traceability_passed": traceability.passed,
                "judge": round(judge_result.overall_score, 4) if judge_result else None,
                "judge_rationale": judge_result.judge_rationale if judge_result else None,
                "judge_error": judge_result.evaluation_error if judge_result else None,
                "injection": injection_verdict.risk_level if injection_verdict else None,
            }
            task_id = context.get("task_id")
            if task_id:
                app_context.project_repository.save_evaluation(task_id, evaluation)
            return evaluation
        except Exception as exc:
            logger = __import__("logging").getLogger(__name__)
            logger.exception("阶段 7 评估失败")
            task_context.report_progress(100, f"评估失败: {exc}")
            return {"error": str(exc)}

    window = MainWindow(
        check_siliconflow,
        ingest_document=ingest_document,
        list_documents=lambda: app_context.repository.list_documents(app_context.default_kb_id),
        delete_documents=delete_documents,
        distill_persona=distill_persona,
        resume_persona=resume_persona,
        upgrade_persona=upgrade_persona,
        evaluate_persona=evaluate_persona,
        list_personas=lambda: app_context.persona_repository.list_ready_personas(
            app_context.default_kb_id
        ),
        list_persona_profiles=lambda: app_context.persona_repository.list_personas(
            app_context.default_kb_id
        ),
        delete_personas=delete_personas,
        load_persona=app_context.persona_repository.load_ready,
        save_persona=save_persona,
        load_runtime_persona=app_context.persona_repository.load_runtime,
        list_persona_versions=app_context.persona_repository.list_versions,
        load_distillation_context=(
            lambda persona_id: app_context.persona_repository.load_run_context(
                app_context.default_kb_id, persona_id
            )
        ),
        get_siliconflow_concurrency=lambda: app_context.siliconflow_gate.limit,
        set_siliconflow_concurrency=app_context.set_siliconflow_concurrency,
        get_siliconflow_request_timeout=app_context.get_siliconflow_request_timeout,
        set_siliconflow_request_timeout=app_context.set_siliconflow_request_timeout,
        get_siliconflow_total_timeout=app_context.get_siliconflow_total_timeout,
        set_siliconflow_total_timeout=app_context.set_siliconflow_total_timeout,
        get_siliconflow_stream_idle_timeout=(app_context.get_siliconflow_stream_idle_timeout),
        set_siliconflow_stream_idle_timeout=(app_context.set_siliconflow_stream_idle_timeout),
        get_retrieval_option=app_context.get_retrieval_option,
        set_retrieval_option=app_context.set_retrieval_option,
        retrieve=retrieve,
        run_writing_pipeline=run_writing_pipeline,
        evaluate_generation=evaluate_generation,
        list_projects=app_context.project_repository.list_projects,
        create_project=lambda title, description: app_context.project_repository.create_project(
            kb_id=app_context.default_kb_id,
            title=title,
            description=description,
        ),
        update_project=lambda project_id, title, description: (
            app_context.project_repository.update_project(
                project_id,
                title=title,
                description=description,
            )
        ),
        delete_projects=app_context.project_repository.delete_projects,
        create_writing_task=lambda **kwargs: app_context.project_repository.create_task(
            kb_id=app_context.default_kb_id,
            citation_style=app_context.settings.citation_style,
            **kwargs,
        ),
        list_writing_tasks=app_context.project_repository.list_tasks,
        load_writing_task=load_writing_task,
        save_edited_draft=app_context.project_repository.save_edited_draft,
        delete_writing_tasks=app_context.project_repository.delete_tasks,
        preview_source_selection=preview_source_selection,
        list_chat_conversations=app_context.chat_repository.list_conversations,
        load_chat_conversation=app_context.chat_repository.load_conversation,
        create_chat_conversation=app_context.author_chat.create_conversation,
        rename_chat_conversation=app_context.chat_repository.rename_conversation,
        delete_chat_conversations=app_context.chat_repository.delete_conversations,
        list_chat_messages=app_context.chat_repository.list_messages,
        send_chat_message=send_chat_message,
        verify_chat_message=verify_chat_message,
        get_author_chat_recent_rounds=app_context.get_author_chat_recent_rounds,
        set_author_chat_recent_rounds=app_context.set_author_chat_recent_rounds,
        get_web_search_result_count=app_context.get_web_search_result_count,
        set_web_search_result_count=app_context.set_web_search_result_count,
        settings_backend=app_context.settings_service,
    )
    application.aboutToQuit.connect(app_context.close)
    window.show()
    if os.environ.get("PARROFISH_FROZEN_SMOKE_TEST") == "1":
        QTimer.singleShot(100, application.quit)
    return application.exec()


def _format_elapsed(duration_ms: int) -> str:
    """Format a stage duration for persistent self-check progress output."""

    total_seconds = max(0, duration_ms // 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
