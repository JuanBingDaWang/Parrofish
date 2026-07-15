"""逐节起草：persona + 提纲节点 + 证据包 → 结构化草稿。

这是生成流水线（阶段 4）的第三步，产出 SectionDraft。
逐节调用，由上游按提纲顺序遍历节点。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from writing_factory.generate.models import (
    EvidenceItem,
    EvidencePack,
    GenerationContext,
    OutlineNode,
    SectionDraft,
    SectionDraftOutput,
    ThesisStatement,
)
from writing_factory.generate.persona_context import persona_context_for_genre
from writing_factory.generate.prompts import drafting_messages
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


def draft_section(
    *,
    context: GenerationContext,
    thesis: ThesisStatement,
    outline_node: OutlineNode,
    term_registry: dict[str, str],
    persona_repository: PersonaRepository,
    retriever: HybridRetriever,
    siliconflow: SiliconFlowClient,
    previous_section_conclusion: str | None = None,
    next_section_purpose: str | None = None,
    revision_feedback: list[dict[str, object]] | None = None,
    prior_claims: list[str] | None = None,
    target_length_chars: int | None = None,
    evidence_pack: EvidencePack | None = None,
    source_key_offset: int = 0,
    progress: ProgressCallback = _no_progress,
    check_cancelled: CancellationCheck = _no_cancellation,
) -> SectionDraft:
    """起草单节：检索证据 → 锁定证据包 → LLM 起草 → 结构化草稿。

    遵循 Iron Law #2：Retrieve → lock EvidencePack → draft。
    遵循 Iron Law #7：thesis + outline + term_registry 在每个起草胶囊中。

    流水线步骤：
        1. 加载 persona 档案
        2. 基于本节标题 + 修辞目的 + 论点检索证据
        3. 锁定 EvidencePack（逐字摘录 + source_key）
        4. 构造 persona + 论点 + 证据包 → LLM 起草消息
        5. 调用 LLM（thinking 模式，json_object 输出）
        6. 解析为 SectionDraft

    Args:
        context: 生成上下文
        thesis: 锚定论点
        outline_node: 本节提纲节点
        term_registry: 术语登记表
        persona_repository: persona 档案仓库
        retriever: 混合检索器
        siliconflow: SiliconFlow 客户端
        previous_section_conclusion: 上一节结论（用于衔接）
        next_section_purpose: 下一节目的（用于铺垫）
        evidence_pack: 上游已冻结的证据包；传入后不再重复检索
        source_key_offset: source_key 起始偏移量（跨节全局递增）
        target_length_chars: 本正文单元的目标中文字数
        progress: 进度回调
        check_cancelled: 取消检查回调

    Returns:
        SectionDraft: 结构化草稿（段落 + 论断 + 证据包）

    Raises:
        ValueError: persona 未就绪
        ExternalServiceError: LLM 调用失败
    """
    if not context.persona_id:
        raise ValueError("persona_id 不能为空")

    # ── 1. 加载 persona ──────────────────────────────────────────────
    progress(5, f"加载 persona — {outline_node.heading}")
    check_cancelled()

    persona_spec = persona_repository.load_runtime(context.persona_id)
    if persona_spec is None:
        raise ValueError(f"persona '{context.persona_id}' 未就绪")
    persona_json = persona_context_for_genre(persona_spec, context.generation_options.genre)

    # ── 2–3. 检索并冻结证据，或复用上游并发预取的冻结结果 ────────────
    if evidence_pack is None:
        progress(10, f"检索证据 — {outline_node.heading}")
        check_cancelled()
        evidence_pack = build_evidence_pack_for_section(
            context=context,
            thesis=thesis,
            outline_node=outline_node,
            retriever=retriever,
            siliconflow=siliconflow,
            source_key_offset=source_key_offset,
            progress=progress,
            check_cancelled=check_cancelled,
        )
    else:
        progress(30, f"使用已冻结证据包 — {outline_node.heading}")
        check_cancelled()
        if evidence_pack.section_id != outline_node.node_id:
            raise ValueError("冻结证据包与当前内容单元不匹配")
    logger.info(
        "证据包锁定: node=%s, %d items",
        outline_node.node_id,
        len(evidence_pack.items),
    )

    # ── 4. 构造消息 → LLM 调用 ───────────────────────────────────────
    progress(40, f"起草 — {outline_node.heading}")
    check_cancelled()

    messages = drafting_messages(
        persona_spec_json=persona_json,
        thesis=thesis,
        outline_node=outline_node,
        evidence_pack=evidence_pack,
        term_registry=term_registry,
        previous_section_conclusion=previous_section_conclusion,
        next_section_purpose=next_section_purpose,
        revision_feedback=revision_feedback,
        prior_claims=prior_claims,
        target_length_chars=target_length_chars,
        document_form=context.generation_options.document_form,
        genre=context.generation_options.genre,
    )

    progress(50, f"调用 LLM 起草 — {outline_node.heading}")
    check_cancelled()

    def assemble_draft(content: str) -> SectionDraft:
        generated = SectionDraftOutput.model_validate_json(content)
        return SectionDraft(
            **generated.model_dump(mode="python"),
            evidence_pack=evidence_pack,
        )

    result = None
    last_error: Exception | None = None
    for attempt in range(1, 3):
        active_messages = messages
        if last_error is not None:
            allowed_keys = [item.source_key for item in evidence_pack.items]
            active_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "上一次草稿未通过结构与证据边界校验。请从头返回完整 JSON，"
                        f"只能使用这些 source_key：{allowed_keys}。"
                        "不得沿用其他内容单元的引用键，不要解释或续写残片。"
                        f"上一次校验错误：{str(last_error)[:600]}"
                    ),
                },
            ]
            progress(50, f"按校验反馈重新起草 — {outline_node.heading}")
        try:
            result = siliconflow.chat(
                active_messages,
                thinking=False,
                temperature=0.5,
                max_tokens=8192,
                response_format="json_object",
                seed=42,
                use_cache=not revision_feedback and attempt == 1,
                stream=True,
                request_attempts=2,
                step_id="writing.draft",
                result_validator=lambda candidate: assemble_draft(candidate.content),
            )
            break
        except Exception as exc:
            check_cancelled()
            last_error = exc
            if attempt == 2:
                raise

    if result is None:
        raise ValueError("LLM 未返回可用的内容单元草稿")

    progress(85, f"解析草稿 — {outline_node.heading}")
    check_cancelled()

    # ── 5. 解析为 SectionDraft ───────────────────────────────────────
    try:
        section_draft = assemble_draft(result.content)
    except Exception as exc:
        logger.error(
            "草稿解析失败: response_chars=%d error_type=%s",
            len(result.content),
            type(exc).__name__,
        )
        raise ValueError(f"LLM 返回的草稿无法解析为 SectionDraft: {exc}") from exc

    progress(100, f"草稿完成 — {outline_node.heading}")
    logger.info(
        "草稿完成: section=%s, %d paragraphs, %d claims",
        section_draft.section_id,
        len(section_draft.paragraphs),
        len(section_draft.claims),
    )
    return section_draft


def build_evidence_pack_for_section(
    *,
    context: GenerationContext,
    thesis: ThesisStatement,
    outline_node: OutlineNode,
    retriever: HybridRetriever,
    siliconflow: SiliconFlowClient,
    source_key_offset: int = 0,
    progress: ProgressCallback = _no_progress,
    check_cancelled: CancellationCheck = _no_cancellation,
) -> EvidencePack:
    """仅为单节构建证据包（不调用 LLM）。

    用于需要预检证据覆盖度的场景，或外部编排时分离检索与起草。
    """
    section_query = _build_section_query(thesis, outline_node)
    retrieval_request = RetrievalRequest(
        kb_id=context.kb_id,
        query=section_query,
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
    return _build_evidence_pack(
        outline_node=outline_node,
        retrieval_result=retrieval_result,
        source_key_offset=source_key_offset,
        repository=retriever.repository,
        kb_id=context.kb_id,
        seed_items=_outline_evidence_items(outline_node),
    )


def _build_section_query(thesis: ThesisStatement, node: OutlineNode) -> str:
    """拼接本节检索查询：论点 + 标题 + 修辞目的。"""
    parts = [thesis.thesis_text, node.heading, node.rhetorical_purpose]
    return "\n".join(parts)


def _build_evidence_pack(
    *,
    outline_node: OutlineNode,
    retrieval_result,
    source_key_offset: int = 0,
    repository=None,
    kb_id: str | None = None,
    seed_items: list[EvidenceItem] | None = None,
) -> EvidencePack:
    """从检索结果构建 EvidencePack。

    每个 hit 成为一条 EvidenceItem，source_key 从 S1 开始递增。
    """
    candidates: list[tuple[str, str, str, int | None, int | None, str | None]] = []
    items = list(seed_items or [])
    seen_chunk_ids: set[str] = {item.chunk_id for item in items}
    for hit in retrieval_result.hits:
        exact_chunks = []
        if repository is not None and kb_id and hit.matched_child_ids:
            exact_chunks = repository.ready_child_chunks_by_ids(kb_id, set(hit.matched_child_ids))
            by_id = {chunk.chunk_id: chunk for chunk in exact_chunks}
            exact_chunks = [by_id[item] for item in hit.matched_child_ids if item in by_id]
        if exact_chunks:
            for chunk in exact_chunks:
                if chunk.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk.chunk_id)
                candidates.append(
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.text,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.section_heading,
                    )
                )
        elif hit.chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(hit.chunk_id)
            candidates.append(
                (
                    hit.chunk_id,
                    hit.doc_id,
                    hit.text,
                    hit.page_start,
                    hit.page_end,
                    hit.section_heading,
                )
            )

    for i, (chunk_id, doc_id, text, page_start, page_end, heading) in enumerate(
        candidates[: max(0, 8 - len(items))], 1
    ):
        source_key = f"S{source_key_offset + i}"
        items.append(
            EvidenceItem(
                source_key=source_key,
                chunk_id=chunk_id,
                doc_id=doc_id,
                verbatim_excerpt=text,
                page_start=page_start,
                page_end=page_end,
                section_heading=heading,
            )
        )

    return EvidencePack(
        section_id=outline_node.node_id,
        items=items,
    )


def _outline_evidence_items(outline_node: OutlineNode) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            source_key=item.source_key,
            chunk_id=item.chunk_id,
            doc_id=item.doc_id,
            verbatim_excerpt=item.verbatim_excerpt,
            page_start=item.page_start,
            page_end=item.page_end,
            section_heading=item.section_heading,
        )
        for item in outline_node.candidate_evidence
    ]
