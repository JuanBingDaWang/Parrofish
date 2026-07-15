"""Recoverable LangGraph orchestration and writing state.

Public API:
    - build_writing_graph() — 构建完整的 LangGraph 写作流水线
    - create_initial_state() — 创建图调用的初始状态
    - run_writing_pipeline_with_progress() — 带进度报告的流水线执行包装器
    - WritingState / SectionState — TypedDict 状态类型
    - WritingPipeline — 流水线节点集合（依赖注入）
    - assemble_reference_list — 引用拼装纯函数
    - review_term_consistency — 术语一致性审查
    - review_structure — 结构审查
    - run_global_polish — 全局一致性打磨（1M 上下文）
"""

from writing_factory.orchestration.consistency import (
    review_structure,
    review_term_consistency,
    run_global_polish,
)
from writing_factory.orchestration.graph import (
    build_writing_graph,
    close_writing_graph,
    create_initial_state,
)
from writing_factory.orchestration.nodes import WritingPipeline
from writing_factory.orchestration.pipeline_runner import run_writing_pipeline_with_progress
from writing_factory.orchestration.reference_assembler import (
    assemble_reference_list,
    render_final_citation_markers,
)
from writing_factory.orchestration.state import (
    MAX_RECOVERY_REVISIONS_PER_SECTION,
    MAX_REVISIONS_PER_SECTION,
    PIPELINE_STATUS_DONE,
    PIPELINE_STATUS_ERROR,
    PIPELINE_STATUS_EVIDENCE_PREFETCH,
    PIPELINE_STATUS_GLOBAL_POLISH,
    PIPELINE_STATUS_STRUCTURE_REVIEW,
    PIPELINE_STATUS_TERM_REVIEW,
    SectionState,
    WritingState,
)

__all__ = [
    "build_writing_graph",
    "create_initial_state",
    "close_writing_graph",
    "run_writing_pipeline_with_progress",
    "WritingState",
    "SectionState",
    "WritingPipeline",
    "assemble_reference_list",
    "render_final_citation_markers",
    "review_term_consistency",
    "review_structure",
    "run_global_polish",
    "MAX_RECOVERY_REVISIONS_PER_SECTION",
    "MAX_REVISIONS_PER_SECTION",
    "PIPELINE_STATUS_DONE",
    "PIPELINE_STATUS_EVIDENCE_PREFETCH",
    "PIPELINE_STATUS_ERROR",
    "PIPELINE_STATUS_TERM_REVIEW",
    "PIPELINE_STATUS_STRUCTURE_REVIEW",
    "PIPELINE_STATUS_GLOBAL_POLISH",
]
