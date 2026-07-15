"""Streaming wrapper around the LangGraph writing pipeline for UI progress reporting.

This module provides a single entry point that:
    1. Builds the graph (or loads cached)
    2. Creates initial state from high-level parameters
    3. Streams node execution, reporting progress via TaskContext
    4. Returns the final WritingState dict

Usage (via main.py closure):
    runner = run_writing_pipeline_with_progress(
        persona_id=...,
        task_description=...,
        domain=...,
        context=task_context,
        siliconflow=app_context.siliconflow,
        retriever=app_context.hybrid_retriever,
        persona_repository=app_context.persona_repository,
        kb_repository=app_context.repository,
        checkpoint_dir=app_context.settings.data_dir / "checkpoints",
        kb_id=app_context.default_kb_id,
        citation_style=app_context.settings.citation_style,
    )
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver

from writing_factory.generate.models import (
    GenerationContext,
    GenerationOptions,
    PolishedSection,
    VerifiedDraft,
)
from writing_factory.generate.source_policy import build_persona_generation_source_policy
from writing_factory.orchestration.errors import PipelineNodeError
from writing_factory.orchestration.graph import (
    build_writing_graph,
    close_writing_graph,
    create_initial_state,
)
from writing_factory.orchestration.state import (
    MAX_RECOVERY_REVISIONS_PER_SECTION,
    PIPELINE_STATUS_DONE,
    PIPELINE_STATUS_DRAFTING,
    PIPELINE_STATUS_ERROR,
    SECTION_STATUS_REVISING,
)

if TYPE_CHECKING:
    from writing_factory.kb.retrieval import HybridRetriever
    from writing_factory.llm.siliconflow import SiliconFlowClient
    from writing_factory.store.kb_repository import KnowledgeBaseRepository
    from writing_factory.store.persona_repository import PersonaRepository
    from writing_factory.ui.workers import TaskContext

logger = logging.getLogger(__name__)

# ── Human-readable step labels ────────────────────────────────

_NODE_LABELS: dict[str, str] = {
    "select_topic": "选题中",
    "build_framework": "构建文稿框架",
    "prefetch_evidence": "并发预取并冻结内容单元证据",
    "draft_section": "起草内容单元",
    "verify_section": "核对事实",
    "polish_section": "打磨文风",
    "prepare_next_section": "准备下一节",
    "prepare_revise_section": "准备修订",
    "parallel_reviews": "并行审查术语与全文结构",
    "term_consistency": "术语一致性审查",
    "structure_review": "结构审查",
    "global_polish": "全局一致性打磨",
    "assemble": "组装事实来源与最终稿",
}


def run_writing_pipeline_with_progress(
    *,
    persona_id: str,
    task_description: str,
    domain: str,
    context: TaskContext,
    # Dependencies
    siliconflow: SiliconFlowClient,
    retriever: HybridRetriever,
    persona_repository: PersonaRepository,
    kb_repository: KnowledgeBaseRepository,
    checkpoint_dir: Path,
    kb_id: str = "default",
    citation_style: str = "gb-t-7714",
    task_id: str | None = None,
    selected_doc_ids: set[str] | None = None,
    explicitly_allowed_persona_doc_ids: set[str] | None = None,
    generation_options: GenerationOptions | None = None,
    state_callback: Callable[[dict[str, Any]], None] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the full writing pipeline with progress reporting.

    Args:
        persona_id: Persona ID to use for style.
        task_description: Natural-language writing task description.
        domain: Research domain (optional, appended to task_description).
        context: TaskContext for progress/cancellation from the UI.
        siliconflow: SiliconFlow LLM client.
        retriever: Hybrid retriever for evidence lookup.
        persona_repository: Persona storage.
        kb_repository: Knowledge base storage (for reference assembly).
        checkpoint_dir: Directory for SQLite checkpointer.
        kb_id: Knowledge base ID.
        citation_style: Citation style string.

    Returns:
        Final WritingState dict (status, final_draft_json, sections, etc.).
    """
    resolved_task_id = task_id or f"task_{uuid4().hex}"
    graph = None

    # ── Build full task text ───────────────────────────────────
    full_task = task_description
    if domain:
        full_task = f"{task_description}\n内容领域：{domain}"

    source_policy = build_persona_generation_source_policy(
        persona_repository=persona_repository,
        persona_id=persona_id,
        selected_task_doc_ids=selected_doc_ids or set(),
        explicitly_allowed_persona_doc_ids=explicitly_allowed_persona_doc_ids or set(),
    )
    source_policy.require_nonempty()

    gen_ctx = GenerationContext(
        kb_id=kb_id,
        task_description=full_task,
        citation_style=citation_style,
        persona_id=persona_id,
        task_id=resolved_task_id,
        allowed_doc_ids=tuple(sorted(source_policy.allowed_task_doc_ids)),
        excluded_persona_doc_ids=tuple(sorted(source_policy.excluded_persona_doc_ids)),
        source_policy_id=source_policy.policy_id,
        generation_options=generation_options or GenerationOptions(),
    )

    # ── Create initial state ───────────────────────────────────
    initial = create_initial_state(
        context_json=gen_ctx.model_dump_json(),
        persona_id=persona_id,
        kb_id=kb_id,
    )

    # ── Stream execution ───────────────────────────────────────
    last_reported_percent = 2

    def report_progress(percent: int, message: str) -> None:
        nonlocal last_reported_percent
        last_reported_percent = max(last_reported_percent, max(0, min(100, percent)))
        context.report_progress(last_reported_percent, message)

    def report_activity(_percent: int, message: str) -> None:
        context.report_progress(last_reported_percent, message)

    config = {"configurable": {"thread_id": resolved_task_id}}

    try:
        context.check_cancelled()
        if not resume:
            report_progress(0, "初始化流水线")
            report_progress(2, "编译写作流水线图")
        graph = build_writing_graph(
            persona_repository=persona_repository,
            retriever=retriever,
            siliconflow=siliconflow,
            kb_repository=kb_repository,
            checkpoint_dir=checkpoint_dir,
            progress=report_activity,
            check_cancelled=context.check_cancelled,
        )
        if resume:
            current = graph.get_state(config)
            if current is not None:
                pending_node = current.next[0] if current.next else ""
                last_reported_percent = max(
                    last_reported_percent,
                    _checkpoint_progress(pending_node, dict(current.values)),
                )
            context.report_progress(last_reported_percent, "读取写作断点")
        graph_input = None if resume else initial
        stream_config = config
        if resume:
            stream_config = _legacy_resume_config(graph, config)
            if stream_config is not config:
                report_progress(last_reported_percent, "恢复升级前的失败断点")
            else:
                stream_config, recovered = _verification_recovery_config(graph, config)
                if recovered:
                    report_progress(
                        last_reported_percent,
                        "安全门失败节已进入受控恢复修订",
                    )
        for event in graph.stream(graph_input, stream_config):
            context.check_cancelled()

            # LangGraph streaming yields dicts like {node_name: state_updates}
            if not isinstance(event, dict):
                continue

            for node_name, state_updates in event.items():
                if not isinstance(state_updates, dict):
                    continue

                snapshot = graph.get_state(config)
                snapshot_values = dict(snapshot.values) if snapshot else dict(state_updates)
                snapshot_values["task_id"] = resolved_task_id
                if state_callback is not None:
                    state_callback(snapshot_values)

                label = _NODE_LABELS.get(node_name, node_name)
                section_idx = snapshot_values.get("current_section_index")
                total = len(snapshot_values.get("sections", []))
                if section_idx is not None and node_name in (
                    "draft_section",
                    "verify_section",
                    "polish_section",
                ):
                    label = (
                        f"{_NODE_LABELS.get(node_name, node_name)} "
                        f"(第 {section_idx + 1}/{total} 节)"
                    )

                pct = _checkpoint_progress(node_name, snapshot_values)
                report_progress(pct, label)

        # ── Graph finished — collect final state ───────────────
        report_progress(99, "正在获取最终结果")

        # Get the final state via get_state
        final_state = graph.get_state(config)
        state_dict = dict(final_state.values) if final_state else {}
        state_dict["task_id"] = resolved_task_id
        if state_dict.get("status") != PIPELINE_STATUS_DONE:
            detail = state_dict.get("error") or state_dict.get("status") or "最终状态为空"
            raise RuntimeError(f"写作流水线未正常完成：{detail}")

        report_progress(100, "流水线完成")
        return state_dict

    except Exception as exc:
        if context.is_cancelled:
            raise
        logger.exception("写作流水线执行异常")
        if graph is not None and state_callback is not None:
            snapshot = graph.get_state(config)
            if snapshot is not None:
                failed_state = dict(snapshot.values)
                failed_state["task_id"] = resolved_task_id
                state_callback(failed_state)
        report_progress(last_reported_percent, f"已停止：{exc}")
        raise
    finally:
        if graph is not None:
            close_writing_graph(graph)


def _legacy_resume_config(graph, config: dict[str, Any]) -> dict[str, Any]:
    """Rewind terminal error states written by the pre-short-circuit graph.

    New node exceptions leave the latest snapshot healthy with a pending node, so
    they use the ordinary thread config. Only legacy terminal ``error`` snapshots
    need a historical checkpoint id to retry the first node that failed.
    """

    current = graph.get_state(config)
    if (
        current is None
        or current.values.get("status") != PIPELINE_STATUS_ERROR
        or current.next
    ):
        return config
    for snapshot in graph.get_state_history(config):
        if snapshot.values.get("status") != PIPELINE_STATUS_ERROR and snapshot.next:
            logger.info(
                "恢复旧版终止错误断点: step=%s next=%s",
                snapshot.metadata.get("step"),
                snapshot.next,
            )
            return snapshot.config
    return config


def _verification_recovery_config(
    graph,
    config: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Turn a pending verification failure into a bounded repair checkpoint."""

    current = graph.get_state(config)
    if current is None or "fail_verification" not in current.next:
        return config, False

    state = dict(current.values)
    sections = list(state.get("sections", []))
    index = state.get("current_section_index")
    if not isinstance(index, int) or not 0 <= index < len(sections):
        raise PipelineNodeError("事实核对恢复", "安全门断点缺少有效的当前内容单元")

    section = dict(sections[index])
    verified_json = section.get("verified_draft_json")
    if not verified_json:
        raise PipelineNodeError("事实核对恢复", "安全门断点缺少结构化核对结果")
    verified = VerifiedDraft.model_validate_json(verified_json)
    failed = [item for item in verified.verified_claims if item.verdict != "supported"]
    if not failed:
        raise PipelineNodeError("事实核对恢复", "安全门断点中没有可供修复的失败论断")

    recovery_count = int(section.get("recovery_revision_count", 0))
    if recovery_count >= MAX_RECOVERY_REVISIONS_PER_SECTION:
        details = "；".join(
            f"{item.claim.claim_id}({','.join(item.claim.source_keys) or '无来源键'})："
            f"{item.verifier_rationale[:180]}"
            for item in failed[:5]
        )
        raise PipelineNodeError(
            "事实核对恢复",
            "额外恢复修订已用尽"
            f"（{recovery_count}/{MAX_RECOVERY_REVISIONS_PER_SECTION}）"
            f"，仍未通过：{details}",
        )

    section.update(
        {
            "status": SECTION_STATUS_REVISING,
            "recovery_revision_count": recovery_count + 1,
            "polished_section_json": None,
            "error": None,
        }
    )
    sections[index] = section
    updated_config = graph.update_state(
        config,
        {
            "sections": sections,
            "status": PIPELINE_STATUS_DRAFTING,
            "error": None,
        },
        as_node="prepare_revise_section",
    )
    logger.info(
        "恢复事实核对安全门断点: section=%s recovery=%d/%d failed_claims=%s",
        section.get("section_id"),
        recovery_count + 1,
        MAX_RECOVERY_REVISIONS_PER_SECTION,
        [item.claim.claim_id for item in failed],
    )
    return updated_config, True


def _checkpoint_progress(node_name: str, state: dict[str, Any]) -> int:
    """Compute overall progress exclusively from committed checkpoint state."""

    phase_minimums = {
        "select_topic": 3,
        "build_framework": 10,
        "prefetch_evidence": 14,
        "parallel_reviews": 88,
        "term_consistency": 88,
        "structure_review": 88,
        "global_polish": 95,
        "assemble": 99,
    }
    sections = state.get("sections", [])
    if not sections:
        return phase_minimums.get(node_name, 2)
    fractions = {
        "pending": 0.0,
        "drafting": 0.0,
        "drafted": 0.34,
        "verifying": 0.34,
        "verified": 0.67,
        "revising": 0.34,
        "polishing": 0.67,
        "polished": 1.0,
        "error": 0.0,
    }
    completed_equivalent = sum(
        fractions.get(str(section.get("status", "pending")), 0.0)
        for section in sections
    )
    section_progress = 14 + round(68 * completed_equivalent / len(sections))
    return max(section_progress, phase_minimums.get(node_name, 0))


def summarize_writing_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a compact UI-safe view without copying frozen evidence through Qt."""

    sections: list[dict[str, Any]] = []
    for section in state.get("sections", []):
        item = {
            "section_id": section.get("section_id", ""),
            "heading": section.get("heading", ""),
            "status": section.get("status", "pending"),
            "revision_count": section.get("revision_count", 0),
            "recovery_revision_count": section.get("recovery_revision_count", 0),
            "target_length_chars": section.get("target_length_chars"),
            "elapsed_seconds": section.get("elapsed_seconds", 0.0),
        }
        polished_json = section.get("polished_section_json")
        if polished_json:
            try:
                item["polished_text"] = PolishedSection.model_validate_json(
                    polished_json
                ).polished_text
            except ValueError:
                pass
        sections.append(item)
    return {
        "task_id": state.get("task_id"),
        "status": state.get("status"),
        "current_section_index": state.get("current_section_index", 0),
        "sections": sections,
    }


def load_latest_writing_state(
    checkpoint_dir: Path,
    task_id: str,
) -> dict[str, Any] | None:
    """Read a task checkpoint without compiling or executing the writing graph."""

    path = checkpoint_dir / "writing_checkpoints.db"
    if not path.is_file():
        return None
    connection = sqlite3.connect(
        f"file:{path.resolve().as_posix()}?mode=ro",
        uri=True,
        check_same_thread=False,
    )
    try:
        item = SqliteSaver(connection).get_tuple(
            {"configurable": {"thread_id": task_id}}
        )
        if item is None:
            return None
        state = dict(item.checkpoint.get("channel_values", {}))
        state["task_id"] = task_id
        return state
    finally:
        connection.close()
