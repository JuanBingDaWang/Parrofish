"""LangGraph StateGraph 构建 + SQLite checkpointer 装配。

图结构:
    select_topic → build_framework → draft_section → verify_section
        → conditional{polish | revise}
        → polish_section → conditional{next_section | assemble(→ phase 6)}
        → term_consistency → structure_review → global_polish → assemble → END

    revise 回路: draft_section ← revise (最多 MAX_REVISIONS_PER_SECTION 次)
    next_section 回路: draft_section ← next_section (遍历所有节)

阶段 6 注入点: per-section 循环结束后, 先做术语一致性→结构审查→全局打磨(1M 上下文),
            最后 assemble 拼装参考文献与最终稿。

设计铁律遵守:
    #5 断点续跑 — SQLite checkpointer 持久化状态
    #2 事实先冻结 — 阶段6 只做术语/过渡/结构优化, 不改事实
    #7 锚定论点 — 全局审查对照 thesis 做一致性检查
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from writing_factory.orchestration.nodes import (
    WritingPipeline,
    prepare_next_section,
    prepare_revise_section,
    should_continue_after_polish,
    should_continue_after_verify,
)
from writing_factory.orchestration.state import WritingState

if TYPE_CHECKING:
    from writing_factory.kb.retrieval import HybridRetriever
    from writing_factory.llm.siliconflow import SiliconFlowClient
    from writing_factory.store.kb_repository import KnowledgeBaseRepository
    from writing_factory.store.persona_repository import PersonaRepository

logger = logging.getLogger(__name__)


def build_writing_graph(
    *,
    persona_repository: PersonaRepository,
    retriever: HybridRetriever,
    siliconflow: SiliconFlowClient,
    kb_repository: KnowledgeBaseRepository,
    checkpoint_dir: Path,
    framework_generation_timeout_seconds: float = 900.0,
    progress: Callable[[int, str], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> StateGraph:
    """构建完整的 LangGraph 写作流水线，装配 SQLite checkpointer。

    Args:
        persona_repository: Persona 仓库。
        retriever: 混合检索器。
        siliconflow: SiliconFlow 统一客户端。
        kb_repository: 知识库仓库（用于引用拼装）。
        checkpoint_dir: SQLite checkpointer 存储目录。

    Returns:
        已编译的 StateGraph，可直接调用 .invoke() 或 .stream()。
    """
    # 依赖注入
    pipeline = WritingPipeline(
        persona_repository=persona_repository,
        retriever=retriever,
        siliconflow=siliconflow,
        kb_repository=kb_repository,
        framework_generation_timeout_seconds=framework_generation_timeout_seconds,
        progress=progress,
        check_cancelled=check_cancelled,
    )

    # 构建图
    builder = StateGraph(WritingState)

    # ── 节点注册 ──────────────────────────────────────────────
    builder.add_node("select_topic", pipeline.select_topic_node)
    builder.add_node("build_framework", pipeline.build_framework_node)
    builder.add_node("draft_section", pipeline.draft_section_node)
    builder.add_node("verify_section", pipeline.verify_section_node)
    builder.add_node("fail_verification", pipeline.fail_verification_node)
    builder.add_node("polish_section", pipeline.polish_section_node)
    builder.add_node("assemble", pipeline.assemble_node)

    # 阶段 6 — 一致性与全局打磨
    builder.add_node("term_consistency", pipeline.term_consistency_node)
    builder.add_node("structure_review", pipeline.structure_review_node)
    builder.add_node("global_polish", pipeline.global_polish_node)

    # 条件边过渡节点（只做纯状态更新，不调 LLM）
    builder.add_node("prepare_next_section", prepare_next_section)
    builder.add_node("prepare_revise_section", prepare_revise_section)

    # ── 边与条件路由 ──────────────────────────────────────────

    # 入口 → 选题
    builder.set_entry_point("select_topic")

    # 选题 → 框架
    builder.add_edge("select_topic", "build_framework")

    # 框架 → 起草第一节
    builder.add_edge("build_framework", "draft_section")

    # 起草 → 核对
    builder.add_edge("draft_section", "verify_section")

    # 核对 → 条件分支: polish 或 revise
    builder.add_conditional_edges(
        "verify_section",
        should_continue_after_verify,
        {
            "polish": "polish_section",
            "revise": "prepare_revise_section",
            "error": "fail_verification",
        },
    )

    # revise → 回到起草（重新起草/修订）
    builder.add_edge("prepare_revise_section", "draft_section")
    builder.add_edge("fail_verification", END)

    # 打磨 → 条件分支: next_section 或 assemble（进入阶段 6）
    builder.add_conditional_edges(
        "polish_section",
        should_continue_after_polish,
        {
            "next_section": "prepare_next_section",
            "assemble": "term_consistency",  # 改为阶段 6 入口
        },
    )

    # next_section → 起草下一节
    builder.add_edge("prepare_next_section", "draft_section")

    # ── 阶段 6 边 ────────────────────────────────────────────

    # 术语审查 → 结构审查
    builder.add_edge("term_consistency", "structure_review")

    # 结构审查 → 全局打磨
    builder.add_edge("structure_review", "global_polish")

    # 全局打磨 → 组装（生成参考文献列表 + 最终稿）
    builder.add_edge("global_polish", "assemble")

    # 组装 → 结束
    builder.add_edge("assemble", END)

    # ── 编译 + SQLite checkpointer ────────────────────────────
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        checkpoint_dir / "writing_checkpoints.db",
        check_same_thread=False,
    )
    checkpointer = SqliteSaver(connection)

    graph = builder.compile(checkpointer=checkpointer)
    logger.info("写作流水线图已编译，checkpointer: %s", checkpoint_dir)

    return graph


def close_writing_graph(graph) -> None:
    """Close the per-run SQLite connection owned by a compiled writing graph."""

    checkpointer = getattr(graph, "checkpointer", None)
    connection = getattr(checkpointer, "conn", None)
    if connection is not None:
        connection.close()


def create_initial_state(
    *,
    context_json: str,
    persona_id: str,
    kb_id: str,
) -> WritingState:
    """创建图调用的初始状态 dict。

    Args:
        context_json: GenerationContext.model_dump_json()。
        persona_id: Persona ID。
        kb_id: 知识库 ID。

    Returns:
        符合 WritingState 的初始 dict。
    """
    return WritingState(
        context_json=context_json,
        persona_id=persona_id,
        kb_id=kb_id,
        thesis_json=None,
        outline_json=None,
        sections=[],
        current_section_index=0,
        term_registry_json="{}",
        source_key_counter=0,
        accumulated_evidence_json="[]",
        claims_made_json="[]",
        reference_list_json=None,
        final_draft_json=None,
        status="topic_selecting",
        error=None,
    )
