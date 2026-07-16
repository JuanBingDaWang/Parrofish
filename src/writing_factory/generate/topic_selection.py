"""论旨锐化：persona + KB 检索 → 可论证的中心论旨。

这是生成流水线（阶段 4）的第一步，产出 ThesisStatement。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from writing_factory.generate.models import GenerationContext, ThesisStatement
from writing_factory.generate.persona_context import persona_context_for_genre
from writing_factory.generate.prompts import topic_selection_messages
from writing_factory.generate.source_policy import (
    enforce_retrieval_safety,
    task_document_filter,
)
from writing_factory.kb.models import RetrievalRequest

if TYPE_CHECKING:
    from writing_factory.kb.retrieval import HybridRetriever
    from writing_factory.llm.siliconflow import SiliconFlowClient
    from writing_factory.store.persona_repository import PersonaRepository

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], None]


def _no_progress(_percent: int, _message: str) -> None:
    pass


def _no_cancellation() -> None:
    pass


def build_direct_thesis(context: GenerationContext) -> ThesisStatement:
    """Create a stable thesis anchor without spending an LLM call."""

    if not context.persona_id:
        raise ValueError("persona_id 不能为空：写作任务必须指定 persona")
    task = context.task_description.strip()
    if not task:
        raise ValueError("写作任务不能为空")
    suggested_title = "" if context.generation_options.document_form == "paragraph" else task[:60]
    return ThesisStatement(
        suggested_title=suggested_title,
        thesis_text=task,
        angle="按用户给定的主题和要求直接展开，不额外改写选题角度。",
        kb_support_assessment=(
            "无事实构思模式不使用知识库，仅允许观点、框架和条件性假设。"
            if context.generation_options.evidence_mode == "conceptual_only"
            else "未执行 LLM 创作意图锐化；后续内容单元仍按所选质量步骤检索并处理事实证据。"
        ),
        persona_id=context.persona_id,
        genre=context.generation_options.genre,
        purpose="完成用户明确指定的非虚构写作任务",
        audience="以用户任务描述为准",
        desired_effect="满足用户指定的信息与表达目标",
    )


def select_topic(
    *,
    context: GenerationContext,
    persona_repository: PersonaRepository,
    retriever: HybridRetriever,
    siliconflow: SiliconFlowClient,
    progress: ProgressCallback = _no_progress,
    check_cancelled: CancellationCheck = _no_cancellation,
) -> ThesisStatement:
    """运行选题锐化流水线：persona + KB 检索 → 可论证的论点。

    流水线步骤：
        1. 加载 persona 档案
        2. 预检索 KB，验证角度可行性
        3. 构造 persona + 检索摘要 → LLM 选题消息
        4. 调用 LLM（thinking 模式，json_object 输出）
        5. 解析为 ThesisStatement

    Args:
        context: 生成上下文（必须含 kb_id、task_description、persona_id）
        persona_repository: persona 档案仓库
        retriever: 混合检索器
        siliconflow: SiliconFlow 客户端
        progress: 进度回调 (percent, message)
        check_cancelled: 取消检查回调，被取消时抛出异常

    Returns:
        ThesisStatement: 经 persona 锐化、KB 可行性验证的中心论旨

    Raises:
        ValueError: persona_id 为空或 persona 未就绪
        ExternalServiceError: LLM 调用失败
    """
    if not context.persona_id:
        raise ValueError("persona_id 不能为空：选题阶段必须指定 persona")

    # ── 1. 加载 persona ──────────────────────────────────────────────
    progress(5, "加载 persona 档案")
    check_cancelled()

    persona_spec = persona_repository.load_runtime(context.persona_id)
    if persona_spec is None:
        raise ValueError(f"persona '{context.persona_id}' 未就绪或不存在，请先完成蒸馏再选题")
    persona_json = persona_context_for_genre(persona_spec, context.generation_options.genre)
    logger.info("选题阶段加载运行时 persona: %s", persona_spec.name)

    progress(15, "准备创作依据")
    check_cancelled()

    # ── 2. KB 预检索 ─────────────────────────────────────────────────
    if context.generation_options.evidence_mode == "conceptual_only":
        kb_retrieval_summary = (
            "当前为无事实构思模式：未检索知识库。只能形成观点、问题、框架、建议"
            "和条件性假设，不得引入具体外部事实。"
        )
        progress(40, "已跳过知识库检索")
    else:
        retrieval_request = RetrievalRequest(
            kb_id=context.kb_id,
            query=context.task_description,
            top_k=8,
            filters=task_document_filter(context),
            use_rewrite=context.generation_options.use_query_rewrite,
            use_hyde=context.generation_options.use_hyde,
            use_rerank=True,
        )
        retrieval_result = retriever.search(
            retrieval_request,
            progress=progress,
            check_cancelled=check_cancelled,
        )
        enforce_retrieval_safety(retrieval_result, siliconflow)

        progress(40, "汇总检索结果")
        check_cancelled()
        kb_retrieval_summary = _format_retrieval_summary(retrieval_result)
        logger.info(
            "选题检索完成: %d hits, 摘要长度 %d",
            len(retrieval_result.hits),
            len(kb_retrieval_summary),
        )

    check_cancelled()

    progress(50, "构造选题提示词")
    check_cancelled()

    # ── 4. 构造消息 → LLM 调用 ───────────────────────────────────────
    messages = topic_selection_messages(
        context=context,
        persona_spec_json=persona_json,
        kb_retrieval_summary=kb_retrieval_summary,
    )

    progress(60, "调用 LLM 锐化选题")
    check_cancelled()

    result = siliconflow.chat(
        messages,
        thinking=True,
        reasoning_effort="high",
        temperature=0.3,
        max_tokens=8192,
        response_format="json_object",
        seed=42,
        stream=True,
        step_id="writing.topic",
    )

    progress(85, "解析选题结果")
    check_cancelled()

    # ── 5. 解析为 ThesisStatement ────────────────────────────────────
    try:
        thesis = ThesisStatement.model_validate_json(result.content)
    except Exception as exc:
        logger.error("选题解析失败，原始响应: %s", result.content[:500])
        raise ValueError(f"LLM 返回的选题结果无法解析为 ThesisStatement: {exc}") from exc
    if thesis.genre != context.generation_options.genre:
        thesis = thesis.model_copy(update={"genre": context.generation_options.genre})

    progress(100, "选题完成")
    logger.info(
        "选题完成: thesis='%s', angle='%s'",
        thesis.thesis_text[:80],
        thesis.angle[:80],
    )
    return thesis


def _format_retrieval_summary(retrieval_result) -> str:
    """将检索结果格式化为 LLM 可读的摘要文本。

    每个 hit 包含：文本片段、来源、页码、章节标题。
    """
    lines: list[str] = []
    lines.append(f"检索查询: {retrieval_result.query}")
    if retrieval_result.expanded_queries:
        lines.append(f"扩展查询: {', '.join(retrieval_result.expanded_queries)}")
    lines.append(f"命中数量: {len(retrieval_result.hits)}")
    lines.append("")

    for i, hit in enumerate(retrieval_result.hits, 1):
        source_label = _source_label(hit.source)
        page_info = ""
        if hit.page_start is not None and hit.page_end is not None:
            page_info = f"第{hit.page_start}–{hit.page_end}页"
        elif hit.page_start is not None:
            page_info = f"第{hit.page_start}页"
        section_info = f" | 章节: {hit.section_heading}" if hit.section_heading else ""

        lines.append(
            f"[H{i}] {source_label} | {page_info}{section_info} | "
            f"score={hit.rerank_score or hit.rrf_score:.3f}"
        )
        lines.append(f"    文档: {hit.doc_id}")
        lines.append(f"    片段: {hit.text[:300]}")
        lines.append("")

    return "\n".join(lines)


def _source_label(source: str) -> str:
    """将 source 值转为中文标签。"""
    labels = {
        "dense": "稠密检索",
        "bm25": "BM25检索",
        "hybrid": "混合检索",
        "web": "联网检索",
    }
    return labels.get(source, source)
