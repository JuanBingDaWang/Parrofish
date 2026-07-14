"""论证骨架构建：persona + 论点 + KB 检索 → 带注释提纲。

这是生成流水线（阶段 4）的第二步，产出 AnnotatedOutline。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from writing_factory.generate.models import (
    AnnotatedOutline,
    GenerationContext,
    OutlineEvidence,
    OutlineNode,
    ThesisStatement,
)
from writing_factory.generate.prompts import framework_messages
from writing_factory.generate.source_policy import (
    enforce_retrieval_safety,
    task_document_filter,
)
from writing_factory.kb.models import RetrievalRequest
from writing_factory.llm.models import ChatResult

if TYPE_CHECKING:
    from writing_factory.kb.retrieval import HybridRetriever
    from writing_factory.llm.siliconflow import SiliconFlowClient
    from writing_factory.store.persona_repository import PersonaRepository

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], None]
FRAMEWORK_OUTPUT_TOKEN_LIMITS = (8192, 16384, 32768)


class FrameworkOutputError(ValueError):
    """The provider completed a request without a usable full outline."""


def _no_progress(_percent: int, _message: str) -> None:
    pass


def _no_cancellation() -> None:
    pass


def build_framework(
    *,
    context: GenerationContext,
    thesis: ThesisStatement,
    persona_repository: PersonaRepository,
    retriever: HybridRetriever,
    siliconflow: SiliconFlowClient,
    request_timeout_seconds: float = 900.0,
    progress: ProgressCallback = _no_progress,
    check_cancelled: CancellationCheck = _no_cancellation,
) -> AnnotatedOutline:
    """构建论证骨架：persona + 论点 + KB 检索 → 带注释提纲。

    流水线步骤：
        1. 加载 persona 档案
        2. 基于论点 + 任务描述进行广域检索
        3. 构造 persona + 论点 + 检索结果 → LLM 框架消息
        4. 调用 LLM（thinking 模式，json_object 输出）
        5. 解析为 AnnotatedOutline

    LLM 在一次调用中完成：提纲结构设计 + 修辞目的标注 + 候选证据分配 + 术语登记。

    Args:
        context: 生成上下文
        thesis: 选题阶段产出的锚定论点
        persona_repository: persona 档案仓库
        retriever: 混合检索器
        siliconflow: SiliconFlow 客户端
        progress: 进度回调
        check_cancelled: 取消检查回调

    Returns:
        AnnotatedOutline: 带注释的完整提纲

    Raises:
        ValueError: persona 未就绪
        ExternalServiceError: LLM 调用失败
    """
    if not context.persona_id:
        raise ValueError("persona_id 不能为空")

    # ── 1. 加载 persona ──────────────────────────────────────────────
    progress(5, "加载 persona 档案")
    check_cancelled()

    persona_spec = persona_repository.load_runtime(context.persona_id)
    if persona_spec is None:
        raise ValueError(f"persona '{context.persona_id}' 未就绪")
    persona_json = persona_spec.model_dump(mode="json")

    progress(15, "广域检索证据")
    check_cancelled()

    # ── 2. 广域检索 ──────────────────────────────────────────────────
    # 用论点 + 任务描述拼接检索查询，扩大覆盖面
    framework_query = f"{thesis.thesis_text}\n{thesis.angle}\n{context.task_description}"
    retrieval_request = RetrievalRequest(
        kb_id=context.kb_id,
        query=framework_query,
        top_k=12,
        filters=task_document_filter(context),
        use_rerank=True,
    )
    retrieval_result = retriever.search(
        retrieval_request,
        progress=progress,
        check_cancelled=check_cancelled,
    )
    enforce_retrieval_safety(retrieval_result, siliconflow)

    progress(30, "汇总检索结果")
    check_cancelled()

    # ── 3. 格式化检索结果 ────────────────────────────────────────────
    node_retrieval_results = _format_broad_retrieval(retrieval_result)
    logger.info(
        "框架检索完成: %d hits → %d 节点检索块",
        len(retrieval_result.hits),
        len(node_retrieval_results),
    )

    progress(40, "构造框架提示词")
    check_cancelled()

    # ── 4. 构造消息 → LLM 调用 ───────────────────────────────────────
    messages = framework_messages(
        context=context,
        persona_spec_json=persona_json,
        thesis=thesis,
        node_retrieval_results=node_retrieval_results,
    )

    # ── 5. 校验并解析为 AnnotatedOutline ─────────────────────────────
    outline: AnnotatedOutline | None = None
    last_error: FrameworkOutputError | None = None
    for attempt, max_tokens in enumerate(FRAMEWORK_OUTPUT_TOKEN_LIMITS, start=1):
        check_cancelled()
        progress(
            50,
            f"调用 LLM 构建提纲（第 {attempt}/3 次，最多 {max_tokens} tokens）",
        )
        active_messages = messages
        if last_error is not None:
            active_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "上一次框架输出不完整或不符合 JSON Schema。请从头重新生成完整的 "
                        "AnnotatedOutline JSON 对象，不要续写残片，不要添加解释或 Markdown。"
                        f"上一次校验错误：{str(last_error)[:600]}"
                    ),
                },
            ]
        try:
            result = siliconflow.chat(
                active_messages,
                thinking=True,
                reasoning_effort="high",
                temperature=0.3,
                max_tokens=max_tokens,
                response_format="json_object",
                seed=42,
                request_timeout_seconds=request_timeout_seconds,
                request_total_timeout_seconds=request_timeout_seconds,
                stream=True,
                result_validator=_validate_framework_result,
            )
            outline = _parse_framework_result(result)
            break
        except FrameworkOutputError as exc:
            last_error = exc
            logger.warning(
                "框架输出校验失败，将重新生成: attempt=%d max_tokens=%d error=%s",
                attempt,
                max_tokens,
                str(exc)[:600],
            )
            if attempt == len(FRAMEWORK_OUTPUT_TOKEN_LIMITS):
                raise ValueError(
                    "LLM 连续三次未返回完整有效的 AnnotatedOutline JSON："
                    f"{exc}"
                ) from exc

    if outline is None:
        raise ValueError("LLM 未返回可用的 AnnotatedOutline")

    progress(85, "提纲 JSON 校验完成")
    check_cancelled()

    progress(88, "按提纲节点检索候选证据")
    outline = _attach_node_evidence(
        outline=outline,
        context=context,
        retriever=retriever,
        siliconflow=siliconflow,
        check_cancelled=check_cancelled,
    )

    progress(100, "提纲构建完成")
    node_count = len(outline.root_nodes)
    total_nodes = _count_all_nodes(outline.root_nodes)
    logger.info(
        "提纲构建完成: %d 个一级节点, 共 %d 个节点, %d 个术语",
        node_count,
        total_nodes,
        len(outline.term_registry),
    )
    return outline


def _validate_framework_result(result: ChatResult) -> None:
    """Validate a chat result before the transport is allowed to cache it."""

    _parse_framework_result(result)


def _parse_framework_result(result: ChatResult) -> AnnotatedOutline:
    if result.finish_reason == "length":
        raise FrameworkOutputError("输出达到 max_tokens 上限，JSON 被截断")
    if result.finish_reason != "stop":
        reason = result.finish_reason or "missing"
        raise FrameworkOutputError(f"流结束时缺少正常 stop 标记（finish_reason={reason}）")
    try:
        return AnnotatedOutline.model_validate_json(result.content)
    except Exception as exc:
        detail = str(exc)
        if "EOF while parsing" in detail or (
            "EOF" in detail and "json" in detail.lower()
        ):
            raise FrameworkOutputError("JSON 在输出结束前被截断（EOF）") from exc
        raise FrameworkOutputError(
            f"JSON 无法通过 AnnotatedOutline 校验：{detail[:1200]}"
        ) from exc


def _format_broad_retrieval(retrieval_result) -> list[dict[str, object]]:
    """将广域检索结果格式化为 LLM 可用的节点检索块。

    以单一检索块的形式传入所有命中，让 LLM 自行决定提纲结构
    以及每个节点应引用哪些 source_key。
    """
    chunks: list[dict[str, object]] = []
    for i, hit in enumerate(retrieval_result.hits, 1):
        chunks.append(
            {
                "source_key": f"S{i}",
                "chunk_id": hit.chunk_id,
                "doc_id": hit.doc_id,
                "text": hit.text,
                "page_start": hit.page_start,
                "page_end": hit.page_end,
                "section_heading": hit.section_heading,
                "rerank_score": hit.rerank_score,
            }
        )

    return [
        {
            "node_id": "broad",
            "heading_hint": "全篇 — LLM 自行划分节点",
            "retrieved_chunks": chunks,
        }
    ]


def _count_all_nodes(nodes: list) -> int:
    """递归统计节点总数（含子节点）。"""
    total = len(nodes)
    for node in nodes:
        if hasattr(node, "children") and node.children:
            total += _count_all_nodes(node.children)
    return total


def _attach_node_evidence(
    *,
    outline: AnnotatedOutline,
    context: GenerationContext,
    retriever: HybridRetriever,
    siliconflow: SiliconFlowClient,
    check_cancelled: CancellationCheck,
) -> AnnotatedOutline:
    """Retrieve each outline node independently and attach exact child evidence."""

    nodes = _flatten_nodes(outline.root_nodes)
    if not nodes:
        return outline

    gate = getattr(getattr(siliconflow, "transport", None), "concurrency_gate", None)
    worker_limit = max(1, min(len(nodes), getattr(gate, "limit", 3)))
    results: dict[str, object] = {}

    def retrieve(node: OutlineNode):
        check_cancelled()
        request = RetrievalRequest(
            kb_id=context.kb_id,
            query=(f"{outline.thesis.thesis_text}\n{node.heading}\n{node.rhetorical_purpose}"),
            top_k=6,
            filters=task_document_filter(context),
            use_rerank=True,
        )
        retrieval_result = retriever.search(
            request,
            check_cancelled=check_cancelled,
        )
        enforce_retrieval_safety(retrieval_result, siliconflow)
        return retrieval_result

    with ThreadPoolExecutor(max_workers=worker_limit) as executor:
        futures = {executor.submit(retrieve, node): node.node_id for node in nodes}
        for future in as_completed(futures):
            check_cancelled()
            results[futures[future]] = future.result()

    next_key = 1
    evidence_by_node: dict[str, list[OutlineEvidence]] = {}
    for node in nodes:
        candidates = _exact_node_candidates(
            results[node.node_id],
            repository=retriever.repository,
            kb_id=context.kb_id,
        )
        attached: list[OutlineEvidence] = []
        for chunk in candidates[:4]:
            attached.append(
                OutlineEvidence(
                    source_key=f"S{next_key}",
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    verbatim_excerpt=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_heading=chunk.section_heading,
                )
            )
            next_key += 1
        evidence_by_node[node.node_id] = attached

    return outline.model_copy(
        update={
            "root_nodes": _copy_nodes_with_evidence(
                outline.root_nodes,
                evidence_by_node,
            )
        }
    )


def _flatten_nodes(nodes: list[OutlineNode]) -> list[OutlineNode]:
    flattened: list[OutlineNode] = []
    for node in nodes:
        flattened.append(node)
        flattened.extend(_flatten_nodes(node.children))
    return flattened


def _copy_nodes_with_evidence(
    nodes: list[OutlineNode],
    evidence_by_node: dict[str, list[OutlineEvidence]],
) -> list[OutlineNode]:
    return [
        node.model_copy(
            update={
                "candidate_source_keys": [
                    item.source_key for item in evidence_by_node[node.node_id]
                ],
                "candidate_evidence": evidence_by_node[node.node_id],
                "children": _copy_nodes_with_evidence(node.children, evidence_by_node),
            }
        )
        for node in nodes
    ]


def _exact_node_candidates(retrieval_result, *, repository, kb_id: str):
    candidates = []
    seen: set[str] = set()
    for hit in retrieval_result.hits:
        child_ids = tuple(hit.matched_child_ids)
        children = repository.ready_child_chunks_by_ids(kb_id, set(child_ids)) if child_ids else []
        by_id = {child.chunk_id: child for child in children}
        exact = [by_id[chunk_id] for chunk_id in child_ids if chunk_id in by_id]
        if not exact:
            exact = [hit]
        for chunk in exact:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                candidates.append(chunk)
    return candidates
