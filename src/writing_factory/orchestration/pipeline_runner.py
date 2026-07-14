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
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from writing_factory.generate.models import GenerationContext
from writing_factory.generate.source_policy import build_generation_source_policy
from writing_factory.orchestration.graph import (
    build_writing_graph,
    close_writing_graph,
    create_initial_state,
)
from writing_factory.orchestration.state import (
    PIPELINE_STATUS_ERROR,
    SECTION_STATUS_DRAFTED,
    SECTION_STATUS_ERROR,
    SECTION_STATUS_POLISHED,
    SECTION_STATUS_VERIFIED,
)

if TYPE_CHECKING:
    from writing_factory.kb.retrieval import HybridRetriever
    from writing_factory.llm.siliconflow import SiliconFlowClient
    from writing_factory.store.kb_repository import KnowledgeBaseRepository
    from writing_factory.store.persona_repository import PersonaRepository
    from writing_factory.ui.workers import TaskContext

logger = logging.getLogger(__name__)

# ── Progress weight model ─────────────────────────────────────
# Estimated weight per major phase (percentage points out of 100)
# This gives a rough progress % as the graph executes.

_WEIGHT_TOPIC = 3
_WEIGHT_FRAMEWORK = 7
_WEIGHT_PER_SECTION = 68  # split among sections: draft+verify+polish
_WEIGHT_TERM_REVIEW = 5
_WEIGHT_STRUCTURE_REVIEW = 5
_WEIGHT_GLOBAL_POLISH = 7
_WEIGHT_ASSEMBLE = 5

# ── Human-readable step labels ────────────────────────────────

_NODE_LABELS: dict[str, str] = {
    "select_topic": "选题中",
    "build_framework": "构建论文框架",
    "draft_section": "起草章节",
    "verify_section": "核对事实",
    "polish_section": "打磨文风",
    "prepare_next_section": "准备下一节",
    "prepare_revise_section": "准备修订",
    "term_consistency": "术语一致性审查",
    "structure_review": "结构审查",
    "global_polish": "全局一致性打磨",
    "assemble": "组装参考文献与最终稿",
}


def _estimate_section_count(
    persona_repository: PersonaRepository,
    persona_id: str,
    task_description: str,
) -> int:
    """Estimate the number of sections for progress calculation.

    Falls back to 5 if we can't determine.
    """
    # Could do a quick LLM call here, but for now just use a reasonable default.
    # The outline-building phase will determine the actual count.
    _ = persona_repository
    _ = persona_id
    _ = task_description
    return 5


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
        full_task = f"{task_description}\n研究领域：{domain}"

    loaded = persona_repository.load_ready(persona_id)
    if loaded is None:
        raise ValueError(f"persona '{persona_id}' 未就绪")
    audit_persona, _markdown = loaded
    source_policy = build_generation_source_policy(
        persona=audit_persona,
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
    )

    # ── Create initial state ───────────────────────────────────
    initial = create_initial_state(
        context_json=gen_ctx.model_dump_json(),
        persona_id=persona_id,
        kb_id=kb_id,
    )

    # ── Estimate section count for progress ────────────────────
    estimated_sections = _estimate_section_count(persona_repository, persona_id, task_description)

    # Per-section weight
    per_section_weight = _WEIGHT_PER_SECTION / estimated_sections if estimated_sections > 0 else 10

    # ── Stream execution ───────────────────────────────────────
    # Progress tracking state
    base_progress = _WEIGHT_TOPIC + _WEIGHT_FRAMEWORK
    completed_sections = 0
    current_section_done_weight = 0.0  # 0-1 within current section

    config = {"configurable": {"thread_id": resolved_task_id}}

    try:
        context.report_progress(0, "初始化流水线")
        context.check_cancelled()
        context.report_progress(2, "编译写作流水线图")
        graph = build_writing_graph(
            persona_repository=persona_repository,
            retriever=retriever,
            siliconflow=siliconflow,
            kb_repository=kb_repository,
            checkpoint_dir=checkpoint_dir,
            progress=context.report_progress,
            check_cancelled=context.check_cancelled,
        )
        graph_input = None if resume else initial
        for event in graph.stream(graph_input, config):
            context.check_cancelled()

            # LangGraph streaming yields dicts like {node_name: state_updates}
            if not isinstance(event, dict):
                continue

            for node_name, state_updates in event.items():
                if not isinstance(state_updates, dict):
                    continue

                # Get status from state update
                status = state_updates.get("status", "")
                error = state_updates.get("error")

                # ── Report progress ────────────────────────────
                progress = _compute_progress(
                    node_name=node_name,
                    status=status,
                    state_updates=state_updates,
                    base_progress=base_progress,
                    per_section_weight=per_section_weight,
                    completed_sections=completed_sections,
                    current_section_done_weight=current_section_done_weight,
                    estimated_sections=estimated_sections,
                )
                if progress is not None:
                    base_progress, completed_sections, current_section_done_weight = progress

                # Update current_section_done_weight tracking
                if status == SECTION_STATUS_DRAFTED:
                    current_section_done_weight = 0.33
                elif status == SECTION_STATUS_VERIFIED:
                    current_section_done_weight = 0.66
                elif status == SECTION_STATUS_POLISHED:
                    current_section_done_weight = 1.0
                    completed_sections += 1
                elif status == SECTION_STATUS_ERROR:
                    current_section_done_weight = 0.0

                # Track completed sections from prepare_next_section
                if node_name == "prepare_next_section":
                    pass  # completed_sections already incremented above

                # Label
                label = _NODE_LABELS.get(node_name, node_name)
                section_idx = state_updates.get("current_section_index")
                total = state_updates.get("total_sections", estimated_sections)
                if section_idx is not None and node_name in (
                    "draft_section",
                    "verify_section",
                    "polish_section",
                ):
                    label = (
                        f"{_NODE_LABELS.get(node_name, node_name)} "
                        f"(第 {section_idx + 1}/{total} 节)"
                    )

                # Report to UI
                pct = min(int(base_progress + current_section_done_weight * per_section_weight), 99)
                context.report_progress(pct, label)

                # Handle error
                if status == PIPELINE_STATUS_ERROR:
                    error_msg = error or "未知流水线错误"
                    logger.error("写作流水线节点 %s 出错: %s", node_name, error_msg)
                    # Don't abort — let the graph finish, we'll catch the final state

        # ── Graph finished — collect final state ───────────────
        context.report_progress(99, "正在获取最终结果")

        # Get the final state via get_state
        final_state = graph.get_state(config)
        state_dict = dict(final_state.values) if final_state else {}
        state_dict["task_id"] = resolved_task_id

        context.report_progress(100, "流水线完成")
        return state_dict

    except Exception as exc:
        if context.is_cancelled:
            raise
        logger.exception("写作流水线执行异常")
        context.report_progress(100, f"流水线异常: {exc}")
        return {
            "status": PIPELINE_STATUS_ERROR,
            "error": str(exc),
            "persona_id": persona_id,
            "task_description": task_description,
            "task_id": resolved_task_id,
        }
    finally:
        if graph is not None:
            close_writing_graph(graph)


def _compute_progress(
    *,
    node_name: str,
    status: str,
    state_updates: dict[str, Any],
    base_progress: float,
    per_section_weight: float,
    completed_sections: int,
    current_section_done_weight: float,
    estimated_sections: int,
) -> tuple[float, int, float] | None:
    """Update progress tracking based on the current node.

    Returns (base_progress, completed_sections, current_section_done_weight) or None.
    """
    new_base = base_progress
    new_completed = completed_sections
    new_section_done = current_section_done_weight

    if node_name == "select_topic":
        new_base = _WEIGHT_TOPIC
    elif node_name == "build_framework":
        new_base = _WEIGHT_TOPIC + _WEIGHT_FRAMEWORK
    elif node_name == "draft_section":
        new_section_done = 0.0
    elif node_name == "verify_section":
        new_section_done = 0.33
    elif node_name == "polish_section":
        new_section_done = 0.66
    elif node_name == "prepare_next_section":
        new_completed += 1
        new_section_done = 0.0
    elif node_name == "term_consistency":
        new_base = _WEIGHT_TOPIC + _WEIGHT_FRAMEWORK + _WEIGHT_PER_SECTION
    elif node_name == "structure_review":
        new_base = _WEIGHT_TOPIC + _WEIGHT_FRAMEWORK + _WEIGHT_PER_SECTION + _WEIGHT_TERM_REVIEW
    elif node_name == "global_polish":
        new_base = (
            _WEIGHT_TOPIC
            + _WEIGHT_FRAMEWORK
            + _WEIGHT_PER_SECTION
            + _WEIGHT_TERM_REVIEW
            + _WEIGHT_STRUCTURE_REVIEW
        )
    elif node_name == "assemble":
        new_base = (
            _WEIGHT_TOPIC
            + _WEIGHT_FRAMEWORK
            + _WEIGHT_PER_SECTION
            + _WEIGHT_TERM_REVIEW
            + _WEIGHT_STRUCTURE_REVIEW
            + _WEIGHT_GLOBAL_POLISH
        )

    return new_base, new_completed, new_section_done
