"""论证骨架构建：persona + 论点 + KB 检索 → 带注释提纲。

这是生成流水线（阶段 4）的第二步，产出 AnnotatedOutline。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from contextvars import copy_context
from typing import TYPE_CHECKING

from writing_factory.generate.models import (
    AnnotatedOutline,
    DocumentForm,
    GenerationContext,
    OutlineEvidence,
    OutlineNode,
    ThesisStatement,
    drafting_unit_range,
)
from writing_factory.generate.persona_context import persona_context_for_genre
from writing_factory.generate.prompts import framework_messages
from writing_factory.generate.source_policy import (
    enforce_retrieval_safety,
    task_document_filter,
)
from writing_factory.kb.models import RetrievalRequest
from writing_factory.llm.models import ChatResult
from writing_factory.nonfiction import NonfictionGenre

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


def build_template_framework(
    *,
    context: GenerationContext,
    thesis: ThesisStatement,
) -> AnnotatedOutline:
    """Build a deterministic minimum outline when LLM framework generation is disabled."""

    document_form = context.generation_options.document_form
    if document_form == "paragraph":
        nodes = [
            OutlineNode(
                node_id="1",
                heading="",
                rhetorical_purpose="围绕中心论旨直接完成用户指定的单个段落",
            )
        ]
    elif document_form == "short_text":
        nodes = [
            OutlineNode(
                node_id="1",
                heading="正文",
                rhetorical_purpose="围绕创作意图完成符合目标文体的紧凑短篇文本",
            )
        ]
    else:
        minimum_units, _maximum_units = drafting_unit_range(
            document_form,
            context.generation_options.target_length_chars,
        )
        nodes = _genre_template_nodes(context.generation_options.genre, minimum_units)
    return AnnotatedOutline(
        thesis=thesis,
        root_nodes=nodes,
        term_registry={},
        kb_id=context.kb_id,
    )


def _genre_template_nodes(genre: NonfictionGenre, unit_count: int) -> list[OutlineNode]:
    """Return a conservative long-form fallback shaped by nonfiction genre."""

    blueprints: dict[NonfictionGenre, list[tuple[str, str, str]]] = {
        "academic_paper": [
            ("问题提出", "界定研究问题、范围与中心论点", ""),
            ("概念与视角", "说明关键概念和分析路径", "递进"),
            ("证据与分析", "使用冻结证据展开核心论证", "递进"),
            ("辨析与边界", "处理反例、异议和适用边界", "转折与限定"),
            ("结论", "汇总论证并回扣中心论点", "归纳与收束"),
        ],
        "research_report": [
            ("问题与范围", "交代调研目的、对象和范围", ""),
            ("资料与方法", "说明材料来源和分析方法", "递进"),
            ("主要发现", "分层呈现受证据支持的发现", "从方法到结果"),
            ("分析与建议", "解释发现并提出边界明确的建议", "从事实到解释"),
            ("结论", "概括发现、限制和后续问题", "归纳与收束"),
        ],
        "policy_brief": [
            ("核心结论", "先给出决策者最需要知道的判断", ""),
            ("问题与影响", "界定问题、受影响对象和紧迫性", "从结论回到问题"),
            ("证据与选项", "呈现证据、可选方案及其权衡", "分析与比较"),
            ("行动建议", "提出责任明确、可执行的建议", "从比较到选择"),
            ("实施要点", "说明条件、风险和检查点", "细化与限定"),
        ],
        "commentary": [
            ("判断与切口", "明确观点及其现实切口", ""),
            ("背景与依据", "提供理解判断所需的事实和背景", "从判断到依据"),
            ("分析展开", "解释因果、机制或价值冲突", "递进"),
            ("异议与边界", "回应可能的反方意见并限定结论", "转折与限定"),
            ("结语", "回扣判断并说明其意义", "归纳与收束"),
        ],
        "review": [
            ("对象与尺度", "介绍评价对象并明确评价标准", ""),
            ("核心特征", "分析对象最重要的内容与形式特征", "从标准到分析"),
            ("贡献与意义", "说明其价值及所在脉络", "递进"),
            ("局限与争议", "给出有依据的批评和边界", "转折与限定"),
            ("总体评价", "形成克制、完整的综合判断", "综合与收束"),
        ],
        "popular_science": [
            ("问题切入", "从读者可感知的问题或误区切入", ""),
            ("概念解释", "用准确而易懂的方式建立基础概念", "从问题到解释"),
            ("机制与证据", "说明现象如何发生及证据边界", "递进"),
            ("常见误解", "澄清容易混淆的判断", "对照与纠偏"),
            ("意义与延伸", "说明知识的现实意义和适用边界", "拓展与收束"),
        ],
        "speech": [
            ("开场", "建立听众连接并提出主线", ""),
            ("核心信息", "用清晰、可记忆的层次展开主要观点", "递进"),
            ("例证与解释", "用事实或例子增强理解和可信度", "具体化"),
            ("回应与转折", "处理听众疑问并推进主线", "转折与递进"),
            ("收束", "回扣主线并形成清晰结束或行动指向", "总结与号召"),
        ],
        "public_article": [
            ("切口", "用明确问题或现象建立阅读动机", ""),
            ("背景", "补充理解问题所需的事实和概念", "从现象到背景"),
            ("解释", "分层说明机制、影响或争议", "递进"),
            ("辨析", "回应误解并给出必要限定", "转折与限定"),
            ("结语", "回到读者关切并完成收束", "回扣与收束"),
        ],
        "news_analysis": [
            ("核心事实", "先交代最重要且已经确认的信息", ""),
            ("背景脉络", "补充事件发生的历史和制度背景", "从事实到背景"),
            ("多方视角", "区分不同主体的立场和证据", "并列与比较"),
            ("影响分析", "解释可能影响并区分事实与推测", "从事实到分析"),
            ("不确定性", "说明尚不能确定之处和后续观察点", "限定与收束"),
        ],
        "instructional": [
            ("目标与前提", "说明读者将完成什么以及需要哪些条件", ""),
            ("准备", "列出必要材料、概念和风险", "从目标到准备"),
            ("执行步骤", "按依赖关系给出可操作步骤", "顺序与递进"),
            ("检查与排错", "提供验收标准和常见问题处理", "反馈与纠偏"),
            ("完成与延伸", "确认结果并给出下一步方向", "总结与延伸"),
        ],
        "summary": [
            ("目的与范围", "交代原材料的主题、目的和边界", ""),
            ("核心信息", "忠实压缩最重要的信息", "从范围到要点"),
            ("关键依据", "保留支撑核心信息的必要依据", "解释与支撑"),
            ("限定条件", "保留原材料中的重要限制", "限定"),
            ("结论", "忠实呈现原材料的结论或意义", "归纳与收束"),
        ],
    }
    fallback = [
        ("目的与问题", "明确文本目的、受众和中心信息", ""),
        ("必要背景", "提供理解中心信息所需的事实和概念", "递进"),
        ("核心展开", "按用户要求组织证据、解释或建议", "递进"),
        ("边界与补充", "处理限制、异议或注意事项", "转折与限定"),
        ("收束", "回扣中心信息并完成沟通目的", "归纳与收束"),
    ]
    blueprint = blueprints.get(genre, fallback)
    fitted = _fit_blueprint(blueprint, unit_count)
    return [
        OutlineNode(
            node_id=str(index),
            heading=heading,
            rhetorical_purpose=purpose,
            relation_to_previous=relation,
        )
        for index, (heading, purpose, relation) in enumerate(fitted, start=1)
    ]


def _fit_blueprint(
    blueprint: list[tuple[str, str, str]],
    unit_count: int,
) -> list[tuple[str, str, str]]:
    if unit_count <= len(blueprint):
        if unit_count == 1:
            return [blueprint[0]]
        return [*blueprint[: unit_count - 1], blueprint[-1]]
    middle = list(blueprint[1:-1])
    while len(middle) < unit_count - 2:
        index = len(middle) + 1
        middle.append(
            (
                f"深入展开（{index}）",
                "使用冻结证据展开一个服务于中心信息的内容单元",
                "递进",
            )
        )
    return [blueprint[0], *middle[: unit_count - 2], blueprint[-1]]


def build_framework(
    *,
    context: GenerationContext,
    thesis: ThesisStatement,
    persona_repository: PersonaRepository,
    retriever: HybridRetriever,
    siliconflow: SiliconFlowClient,
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
    persona_json = persona_context_for_genre(persona_spec, context.generation_options.genre)

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
    profile_getter = getattr(siliconflow, "step_config", None)
    base_tokens = (
        profile_getter("writing.framework").max_tokens
        if callable(profile_getter)
        else FRAMEWORK_OUTPUT_TOKEN_LIMITS[0]
    )
    token_limits = tuple(min(131072, base_tokens * multiplier) for multiplier in (1, 2, 4))
    for attempt, max_tokens in enumerate(token_limits, start=1):
        check_cancelled()
        progress(
            50,
            f"调用 LLM 构建内容规划（第 {attempt}/3 次，最多 {max_tokens} tokens）",
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
                stream=True,
                step_id="writing.framework",
                step_max_tokens_multiplier=2 ** (attempt - 1),
                result_validator=lambda candidate: _validate_framework_result(
                    candidate,
                    target_length_chars=context.generation_options.target_length_chars,
                    document_form=context.generation_options.document_form,
                ),
            )
            outline = _parse_framework_result(result)
            outline = outline.model_copy(update={"thesis": thesis, "kb_id": context.kb_id})
            _validate_outline_budget(
                outline,
                target_length_chars=context.generation_options.target_length_chars,
                document_form=context.generation_options.document_form,
            )
            break
        except FrameworkOutputError as exc:
            last_error = exc
            logger.warning(
                "框架输出校验失败，将重新生成: attempt=%d max_tokens=%d error=%s",
                attempt,
                max_tokens,
                str(exc)[:600],
            )
            if attempt == len(token_limits):
                raise ValueError(
                    "LLM 连续三次未返回完整有效的 AnnotatedOutline JSON："
                    f"{exc}"
                ) from exc

    if outline is None:
        raise ValueError("LLM 未返回可用的 AnnotatedOutline")

    progress(85, "内容规划 JSON 校验完成")
    check_cancelled()

    progress(88, "按内容单元检索候选证据")
    outline = _attach_node_evidence(
        outline=outline,
        context=context,
        retriever=retriever,
        siliconflow=siliconflow,
        check_cancelled=check_cancelled,
    )

    progress(100, "内容规划完成")
    node_count = len(outline.root_nodes)
    total_nodes = _count_all_nodes(outline.root_nodes)
    logger.info(
        "提纲构建完成: %d 个一级节点, 共 %d 个节点, %d 个术语",
        node_count,
        total_nodes,
        len(outline.term_registry),
    )
    return outline


def _validate_framework_result(
    result: ChatResult,
    *,
    target_length_chars: int | None = None,
    document_form: DocumentForm = "paper",
) -> None:
    """Validate a chat result before the transport is allowed to cache it."""

    outline = _parse_framework_result(result)
    if target_length_chars is not None:
        _validate_outline_budget(
            outline,
            target_length_chars=target_length_chars,
            document_form=document_form,
        )


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


def _validate_outline_budget(
    outline: AnnotatedOutline,
    *,
    target_length_chars: int,
    document_form: DocumentForm = "paper",
) -> None:
    leaves = _draftable_nodes(outline.root_nodes)
    minimum, maximum = drafting_unit_range(document_form, target_length_chars)
    if not leaves:
        raise FrameworkOutputError("内容规划没有可起草的叶子正文单元")
    if len(leaves) > maximum:
        recommended = str(minimum) if minimum == maximum else f"{minimum}-{maximum}"
        raise FrameworkOutputError(
            f"目标篇幅约 {target_length_chars} 字，建议安排 {recommended} 个正文单元，"
            f"只允许最多 {maximum} 个；"
            f"当前内容规划有 {len(leaves)} 个叶子正文单元"
        )


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

    nodes = _draftable_nodes(outline.root_nodes)
    if not nodes:
        return outline

    gate = getattr(getattr(siliconflow, "transport", None), "concurrency_gate", None)
    worker_limit = max(1, min(len(nodes), getattr(gate, "limit", 3)))
    results: dict[str, object] = {}

    def retrieve(node: OutlineNode):
        check_cancelled()
        stage = getattr(siliconflow, "stream_stage", None)
        with stage(f"框架证据 · {node.heading}") if stage else nullcontext():
            request = RetrievalRequest(
                kb_id=context.kb_id,
                query=(f"{outline.thesis.thesis_text}\n{node.heading}\n{node.rhetorical_purpose}"),
                top_k=6,
                filters=task_document_filter(context),
                use_rewrite=context.generation_options.use_query_rewrite,
                use_hyde=context.generation_options.use_hyde,
                use_rerank=True,
            )
            retrieval_result = retriever.search(
                request,
                check_cancelled=check_cancelled,
            )
            enforce_retrieval_safety(retrieval_result, siliconflow)
            return retrieval_result

    with ThreadPoolExecutor(max_workers=worker_limit) as executor:
        futures = {
            executor.submit(copy_context().run, retrieve, node): node.node_id
            for node in nodes
        }
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


def _draftable_nodes(nodes: list[OutlineNode]) -> list[OutlineNode]:
    leaves: list[OutlineNode] = []
    for node in nodes:
        if node.children:
            leaves.extend(_draftable_nodes(node.children))
        else:
            leaves.append(node)
    return leaves


def _copy_nodes_with_evidence(
    nodes: list[OutlineNode],
    evidence_by_node: dict[str, list[OutlineEvidence]],
) -> list[OutlineNode]:
    return [
        node.model_copy(
            update={
                "candidate_source_keys": [
                    item.source_key for item in evidence_by_node.get(node.node_id, [])
                ],
                "candidate_evidence": evidence_by_node.get(node.node_id, []),
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
