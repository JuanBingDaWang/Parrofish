"""LangGraph 写作流水线节点。

每个节点 = 一个 LangGraph 状态转换函数，封装一个流水线步骤。
通过 WritingPipeline 类实现依赖注入（persona_repository / retriever / siliconflow）。

设计铁律检查点（每个节点注释中说明如何遵守八条铁律）。
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from contextvars import copy_context
from functools import partial, wraps
from typing import TYPE_CHECKING

from writing_factory.generate.models import (
    AnnotatedOutline,
    EvidenceItem,
    EvidencePack,
    GenerationContext,
    GenerationOptions,
    GlobalPolishResult,
    OutlineNode,
    PolishedDraft,
    PolishedSection,
    SectionDraft,
    StructureReview,
    TermConsistencyReport,
    ThesisStatement,
    VerifiedClaim,
    VerifiedDraft,
)
from writing_factory.orchestration.consistency import (
    review_structure,
    review_term_consistency,
    run_global_polish,
)
from writing_factory.orchestration.errors import PipelineNodeError
from writing_factory.orchestration.reference_assembler import (
    assemble_reference_list,
    render_final_citation_markers,
)
from writing_factory.orchestration.state import (
    MAX_RECOVERY_REVISIONS_PER_SECTION,
    MAX_REVISIONS_PER_SECTION,
    PIPELINE_STATUS_ASSEMBLING,
    PIPELINE_STATUS_DONE,
    PIPELINE_STATUS_DRAFTING,
    PIPELINE_STATUS_EVIDENCE_PREFETCH,
    PIPELINE_STATUS_FRAMEWORK,
    PIPELINE_STATUS_GLOBAL_POLISH,
    PIPELINE_STATUS_POLISHING,
    PIPELINE_STATUS_STRUCTURE_REVIEW,
    PIPELINE_STATUS_VERIFYING,
    SECTION_STATUS_DRAFTED,
    SECTION_STATUS_PENDING,
    SECTION_STATUS_POLISHED,
    SECTION_STATUS_REVISING,
    SECTION_STATUS_VERIFIED,
    SectionState,
    WritingState,
)

if TYPE_CHECKING:
    from writing_factory.kb.retrieval import HybridRetriever
    from writing_factory.llm.siliconflow import SiliconFlowClient
    from writing_factory.store.kb_repository import KnowledgeBaseRepository
    from writing_factory.store.persona_repository import PersonaRepository

logger = logging.getLogger(__name__)

# ── 辅助函数 ──────────────────────────────────────────────────


def _ctx(state: WritingState) -> GenerationContext:
    """从 state 反序列化 GenerationContext。"""
    return GenerationContext.model_validate_json(state["context_json"])


def _options(state: WritingState) -> GenerationOptions:
    raw = state.get("context_json")
    if raw:
        return GenerationContext.model_validate_json(raw).generation_options
    return GenerationOptions()


def _thesis(state: WritingState) -> ThesisStatement | None:
    t = state.get("thesis_json")
    return ThesisStatement.model_validate_json(t) if t else None


def _outline(state: WritingState) -> AnnotatedOutline | None:
    o = state.get("outline_json")
    return AnnotatedOutline.model_validate_json(o) if o else None


def _current_section(state: WritingState) -> dict | None:
    """获取当前节的 SectionState dict。"""
    idx = state.get("current_section_index", 0)
    sections = state.get("sections", [])
    if 0 <= idx < len(sections):
        return sections[idx]
    return None


def _update_section(state: WritingState, updates: dict) -> list[dict]:
    """更新当前节并返回新的 sections 列表。"""
    idx = state["current_section_index"]
    sections = list(state["sections"])
    sections[idx] = {**sections[idx], **updates}
    return sections


def _claim_summary(text: str) -> str:
    """Remove section-local citation markers before sharing a claim downstream."""

    return re.sub(r"\[S\d+\]", "", text).strip()


def _quality_notes(options: GenerationOptions) -> list[str]:
    if options.evidence_mode == "conceptual_only":
        return [
            "无事实构思稿：未检索知识库，不含参考文献；已强制执行外部事实混入检查。"
        ]
    checks = (
        (options.use_hyde, "写作检索未使用 HyDE 假设文档"),
        (options.use_query_rewrite, "写作检索未使用查询改写"),
        (options.topic_refinement, "未执行 LLM 选题锐化，沿用用户任务作为论旨"),
        (options.framework_generation, "未执行 LLM 内容规划，使用代码生成的基础规划"),
        (options.fact_verification, "未执行 LLM 事实语义核验"),
        (options.section_polish, "未执行内容单元文风打磨"),
        (not options.section_polish or options.section_drift_check, "未执行内容单元防漂移核对"),
        (options.term_review, "未执行全文术语审查"),
        (options.structure_review, "未执行全文结构审查"),
        (options.global_polish, "未执行全局一致性打磨"),
        (not options.global_polish or options.global_drift_check, "未执行全局防漂移核对"),
    )
    return [message for enabled, message in checks if not enabled]


def _draftable_outline_nodes(nodes: list[OutlineNode]) -> list[OutlineNode]:
    """Treat hierarchy parents as containers and return only body-producing leaves."""

    result: list[OutlineNode] = []
    for node in nodes:
        if node.children:
            result.extend(_draftable_outline_nodes(node.children))
        else:
            result.append(node)
    return result


def _state_outline_nodes(state: WritingState, outline: AnnotatedOutline) -> list[OutlineNode]:
    """Resolve nodes in persisted section order, including legacy all-node checkpoints."""

    by_id = {node.node_id: node for node in _flatten_outline_nodes(outline.root_nodes)}
    resolved = [by_id[section["section_id"]] for section in state.get("sections", [])]
    return resolved or _draftable_outline_nodes(outline.root_nodes)


def _pipeline_node(label: str):
    """Wrap a node so exceptions abort before LangGraph commits its update."""

    def decorate(function):
        @wraps(function)
        def wrapped(self: WritingPipeline, state: WritingState) -> dict:
            try:
                return function(self, state)
            except PipelineNodeError:
                raise
            except Exception as exc:
                self.check_cancelled()
                logger.exception("%s failed", function.__name__)
                raise PipelineNodeError(label, str(exc)) from exc

        return wrapped

    return decorate


# ── 节点类 ────────────────────────────────────────────────────


class WritingPipeline:
    """LangGraph 写作流水线节点集合。

    通过构造函数注入外部依赖，每个方法可作为 LangGraph add_node 的回调。
    """

    def __init__(
        self,
        persona_repository: PersonaRepository,
        retriever: HybridRetriever,
        siliconflow: SiliconFlowClient,
        kb_repository: KnowledgeBaseRepository,
        progress: Callable[[int, str], None] | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ) -> None:
        self.persona_repository = persona_repository
        self.retriever = retriever
        self.siliconflow = siliconflow
        self.kb_repository = kb_repository
        self.progress = progress or (lambda _percent, _message: None)
        self.check_cancelled = check_cancelled or (lambda: None)

    # ── 节点：选题 ────────────────────────────────────────────
    # 铁律遵守：
    #   #1 persona 控文风 ✓ — 选题阶段用 persona 锐化角度，不是事实
    #   #7 锚定论点 ✓ — 产出 ThesisStatement 即全篇锚
    #   #8 生成阶段只读 ✓ — 不调用 write tool

    @_pipeline_node("选题")
    def select_topic_node(self, state: WritingState) -> dict:
        """选题节点：persona + KB 检索 → ThesisStatement。"""
        from writing_factory.generate.topic_selection import build_direct_thesis, select_topic

        logger.info("节点: select_topic")
        context = _ctx(state)

        try:
            if context.generation_options.topic_refinement:
                thesis = select_topic(
                    context=context,
                    persona_repository=self.persona_repository,
                    retriever=self.retriever,
                    siliconflow=self.siliconflow,
                    progress=self.progress,
                    check_cancelled=self.check_cancelled,
                )
            else:
                self.progress(100, "已按用户要求直接建立论旨锚点")
                thesis = build_direct_thesis(context)
            return {
                "thesis_json": thesis.model_dump_json(),
                "status": PIPELINE_STATUS_FRAMEWORK,
            }
        except Exception:
            self.check_cancelled()
            logger.exception("select_topic 失败")
            raise

    # ── 节点：框架 ────────────────────────────────────────────
    # 铁律遵守：
    #   #1 persona 控文风 ✓ — 框架用 persona 定论证骨架
    #   #2 事实先冻结 ✓ — 只在候选证据层面做映射，不写正文
    #   #7 锚定论点 ✓ — 每个节点 rhetorical_purpose 指向 thesis

    @_pipeline_node("框架生成")
    def build_framework_node(self, state: WritingState) -> dict:
        """框架节点：thesis + persona + KB 检索 → AnnotatedOutline。"""
        from writing_factory.generate.framework import build_framework, build_template_framework

        logger.info("节点: build_framework")
        context = _ctx(state)
        thesis = _thesis(state)
        if thesis is None:
            raise ValueError("缺少已冻结的中心论旨")

        try:
            if context.generation_options.framework_generation:
                outline = build_framework(
                    context=context,
                    thesis=thesis,
                    persona_repository=self.persona_repository,
                    retriever=self.retriever,
                    siliconflow=self.siliconflow,
                    progress=self.progress,
                    check_cancelled=self.check_cancelled,
                )
            else:
                self.progress(100, "已生成基础内容规划，跳过 LLM 内容规划")
                outline = build_template_framework(context=context, thesis=thesis)

            # 初始化 sections 列表
            sections: list[dict] = []
            all_nodes = _draftable_outline_nodes(outline.root_nodes)
            if not all_nodes:
                raise ValueError("内容规划没有可起草的叶子正文单元")
            section_budget = max(
                200,
                context.generation_options.target_length_chars // len(all_nodes),
            )
            for node in all_nodes:
                sections.append(
                    SectionState(
                        section_id=node.node_id,
                        heading=node.heading,
                        status=SECTION_STATUS_PENDING,
                        evidence_pack_json=None,
                        draft_json=None,
                        verified_draft_json=None,
                        polished_section_json=None,
                        revision_count=0,
                        recovery_revision_count=0,
                        source_key_offset=0,
                        target_length_chars=section_budget,
                        elapsed_seconds=0.0,
                        previous_conclusion=None,
                        next_purpose=None,
                        error=None,
                    )
                )

            return {
                "outline_json": outline.model_dump_json(),
                "term_registry_json": json.dumps(outline.term_registry, ensure_ascii=False),
                "sections": sections,
                "current_section_index": 0,
                "source_key_counter": _largest_source_key(outline),
                "status": PIPELINE_STATUS_DRAFTING,
            }
        except Exception:
            self.check_cancelled()
            logger.exception("build_framework 失败")
            raise

    # ── 节点：并发预取并冻结逐节证据 ───────────────────────────
    # 铁律遵守：
    #   #2 事实先冻结 ✓ — 先完成所有 EvidencePack，再开始正文起草
    #   #3 事实论断绑定 chunk ✓ — 每条证据保留 chunk 与页码锚点
    #   #4 引用由代码拼装 ✓ — 每节预留固定、互不重叠的 source_key 区间

    @_pipeline_node("内容单元证据预取")
    def prefetch_evidence_node(self, state: WritingState) -> dict:
        """并发检索各节证据，并把冻结证据包写入可恢复状态。"""

        from writing_factory.generate.drafting import build_evidence_pack_for_section

        logger.info("节点: prefetch_evidence")
        context = _ctx(state)
        thesis = _thesis(state)
        outline = _outline(state)
        if thesis is None or outline is None:
            raise ValueError("缺少中心信息或内容规划，无法预取内容单元证据")

        all_nodes = _state_outline_nodes(state, outline)
        if not all_nodes:
            raise ValueError("内容规划没有可起草的内容单元")
        self.progress(10, f"并发预取内容单元证据（0/{len(all_nodes)}）")
        sections = list(state.get("sections", []))
        if len(sections) != len(all_nodes):
            raise ValueError("内容单元状态与规划节点数量不一致")

        if context.generation_options.evidence_mode == "conceptual_only":
            for index, section in enumerate(sections):
                pack = EvidencePack(section_id=all_nodes[index].node_id, items=[])
                sections[index] = {
                    **section,
                    "evidence_pack_json": pack.model_dump_json(),
                    "source_key_offset": 0,
                }
            self.progress(10, "无事实构思模式已冻结全部空证据包")
            return {
                "sections": sections,
                "source_key_counter": 0,
                "status": PIPELINE_STATUS_EVIDENCE_PREFETCH,
            }

        base_offset = state.get("source_key_counter", 0)
        key_span = 8
        gate = getattr(getattr(self.siliconflow, "transport", None), "concurrency_gate", None)
        worker_limit = max(1, min(len(all_nodes), getattr(gate, "limit", 3)))

        def retrieve(index: int) -> tuple[int, EvidencePack]:
            self.check_cancelled()
            node = all_nodes[index]
            stage = getattr(self.siliconflow, "stream_stage", None)
            with stage(f"证据预取 · {node.heading}") if stage else nullcontext():
                pack = build_evidence_pack_for_section(
                    context=context,
                    thesis=thesis,
                    outline_node=node,
                    retriever=self.retriever,
                    siliconflow=self.siliconflow,
                    source_key_offset=base_offset + index * key_span,
                    check_cancelled=self.check_cancelled,
                )
            return index, pack

        packs: dict[int, EvidencePack] = {}
        with ThreadPoolExecutor(
            max_workers=worker_limit,
            thread_name_prefix="evidence-prefetch",
        ) as executor:
            futures = []
            for index in range(len(all_nodes)):
                task_context = copy_context()
                futures.append(executor.submit(task_context.run, retrieve, index))
            for completed, future in enumerate(as_completed(futures), start=1):
                self.check_cancelled()
                index, pack = future.result()
                packs[index] = pack
                self.progress(10, f"并发预取内容单元证据（{completed}/{len(all_nodes)}）")

        for index, section in enumerate(sections):
            sections[index] = {
                **section,
                "evidence_pack_json": packs[index].model_dump_json(),
                "source_key_offset": base_offset + index * key_span,
            }

        return {
            "sections": sections,
            "source_key_counter": base_offset + len(all_nodes) * key_span,
            "status": PIPELINE_STATUS_EVIDENCE_PREFETCH,
        }

    # ── 节点：起草 ────────────────────────────────────────────
    # 铁律遵守：
    #   #1 persona 控文风 ✓ — 起草用 persona 表达 DNA 写初稿
    #   #2 事实先冻结 ✓ — 先锁定 EvidencePack 再写
    #   #3 事实论断绑定 chunk ✓ — 每个 fact claim 必须有 source_key
    #   #4 引用由代码拼装 ✓ — source_key 由代码分配，不由模型生成
    #   #7 锚定论点 ✓ — thesis + outline 在每个起草胶囊中

    @_pipeline_node("内容单元起草")
    def draft_section_node(self, state: WritingState) -> dict:
        """起草节点：逐节检索证据 → 锁定 EvidencePack → LLM 起草。"""
        from writing_factory.generate.drafting import draft_section

        logger.info("节点: draft_section")
        context = _ctx(state)
        thesis = _thesis(state)
        outline = _outline(state)
        cur = _current_section(state)
        if thesis is None or outline is None or cur is None:
            raise ValueError("缺少中心信息、内容规划或当前内容单元")

        term_registry = json.loads(state.get("term_registry_json", "{}"))
        recovery_revision_count = cur.get("recovery_revision_count", 0)
        is_revision = cur.get("revision_count", 0) > 0 or recovery_revision_count > 0
        frozen_evidence_json = cur.get("evidence_pack_json")
        if not frozen_evidence_json and is_revision and cur.get("draft_json"):
            legacy_draft = SectionDraft.model_validate_json(cur["draft_json"])
            frozen_evidence_json = legacy_draft.evidence_pack.model_dump_json()
        frozen_evidence = (
            EvidencePack.model_validate_json(frozen_evidence_json)
            if frozen_evidence_json
            else None
        )
        source_key_offset = (
            cur.get("source_key_offset", 0)
            if frozen_evidence is not None or is_revision
            else state.get("source_key_counter", 0)
        )
        revision_feedback: list[dict[str, str]] = []
        if is_revision and cur.get("verified_draft_json"):
            previous_verification = VerifiedDraft.model_validate_json(cur["verified_draft_json"])
            revision_feedback = [
                {
                    "claim_id": item.claim.claim_id,
                    "claim_text": item.claim.text,
                    "claim_type": item.claim.claim_type,
                    "source_keys": list(item.claim.source_keys),
                    "verdict": item.verdict,
                    "rationale": item.verifier_rationale,
                    "required_action": (
                        (
                            "删除需要外部来源支持的内容，或改写为观点、建议、方法或"
                            "明确标注的条件性假设；不得换一个类型标签来绕过检查。"
                            if context.generation_options.evidence_mode == "conceptual_only"
                            else "删除该事实论断，或缩写到冻结证据明确支持的范围；"
                            "不得只更换引用键。"
                        )
                        if item.verdict == "unsupported"
                        else "把论断收缩到核验理由指出的受支持范围。"
                    ),
                }
                for item in previous_verification.verified_claims
                if item.verdict != "supported"
            ]

        # 找到当前提纲节点
        all_nodes = _state_outline_nodes(state, outline)
        idx = state["current_section_index"]
        if idx >= len(all_nodes):
            raise IndexError(f"内容单元索引 {idx} 超出规划范围")

        outline_node = all_nodes[idx]

        # 衔接上下文
        previous_conclusion = None
        next_purpose = None
        if idx > 0:
            prev_sec = state["sections"][idx - 1]
            previous_conclusion = prev_sec.get("previous_conclusion")
        if idx + 1 < len(all_nodes):
            next_purpose = all_nodes[idx + 1].rhetorical_purpose

        try:
            started_at = time.perf_counter()
            if recovery_revision_count:
                version_label = (
                    "恢复修订 "
                    f"{recovery_revision_count}/{MAX_RECOVERY_REVISIONS_PER_SECTION}"
                )
            elif is_revision:
                version_label = f"修订 {cur.get('revision_count', 0)}"
            else:
                version_label = "初稿"
            stage = getattr(self.siliconflow, "stream_stage", None)
            with (
                stage(f"内容单元起草 · {outline_node.heading} · {version_label}")
                if stage
                else nullcontext()
            ):
                section_draft = draft_section(
                    context=context,
                    thesis=thesis,
                    outline_node=outline_node,
                    term_registry=term_registry,
                    persona_repository=self.persona_repository,
                    retriever=self.retriever,
                    siliconflow=self.siliconflow,
                    previous_section_conclusion=previous_conclusion,
                    next_section_purpose=next_purpose,
                    source_key_offset=source_key_offset,
                    revision_feedback=revision_feedback,
                    prior_claims=[
                        _claim_summary(text)
                        for text in json.loads(state.get("claims_made_json", "[]"))
                    ],
                    target_length_chars=cur.get("target_length_chars"),
                    evidence_pack=frozen_evidence,
                    progress=self.progress,
                    check_cancelled=self.check_cancelled,
                )

            # 更新 source_key_counter
            key_numbers = [
                int(item.source_key.removeprefix("S"))
                for item in section_draft.evidence_pack.items
                if item.source_key.removeprefix("S").isdigit()
            ]
            new_counter = max([state.get("source_key_counter", 0), *key_numbers])

            # 提取本节结论（最后一段最后一句，用于下一节衔接）
            conclusion = ""
            if section_draft.paragraphs:
                conclusion = section_draft.paragraphs[-1]

            sections_update = _update_section(
                state,
                {
                    "status": SECTION_STATUS_DRAFTED,
                    "draft_json": section_draft.model_dump_json(),
                    "evidence_pack_json": section_draft.evidence_pack.model_dump_json(),
                    "verified_draft_json": None,
                    "source_key_offset": source_key_offset,
                    "previous_conclusion": conclusion,
                    "next_purpose": next_purpose,
                    "elapsed_seconds": cur.get("elapsed_seconds", 0.0)
                    + (time.perf_counter() - started_at),
                },
            )

            return {
                "sections": sections_update,
                "source_key_counter": new_counter,
                "status": PIPELINE_STATUS_VERIFYING,
            }
        except Exception:
            self.check_cancelled()
            logger.exception("draft_section 失败")
            raise

    # ── 节点：核对 ────────────────────────────────────────────
    # 铁律遵守：
    #   #5 不让作者校验自己 ✓ — 核对使用中性角色，不加载 persona
    #   #2 事实先冻结 ✓ — 只检查 fact 类型 claim，不碰 interpretation/common
    #   #4 引用由代码拼装 ✓ — 不产生新引用，只比对已有 source_key → chunk

    @_pipeline_node("事实核对")
    def verify_section_node(self, state: WritingState) -> dict:
        """核对节点：逐 claim 比对 chunk 原文 → 判定 supported/partial/unsupported。"""
        from writing_factory.generate.verification import verify_section

        logger.info("节点: verify_section")
        cur = _current_section(state)
        if cur is None:
            raise ValueError("缺少当前内容单元")

        draft_json = cur.get("draft_json")
        if draft_json is None:
            raise ValueError("当前内容单元缺少结构化草稿")

        section_draft = SectionDraft.model_validate_json(draft_json)
        options = _options(state)

        try:
            started_at = time.perf_counter()
            if options.evidence_mode == "conceptual_only":
                from writing_factory.generate.conceptual_safety import (
                    verify_conceptual_section,
                )

                stage = getattr(self.siliconflow, "stream_stage", None)
                with (
                    stage(f"外部事实混入检查 · {cur.get('heading', cur.get('section_id', ''))}")
                    if stage
                    else nullcontext()
                ):
                    verified = verify_conceptual_section(
                        section_draft=section_draft,
                        task_description=_ctx(state).task_description,
                        siliconflow=self.siliconflow,
                        check_cancelled=self.check_cancelled,
                    )
            elif options.fact_verification:
                stage = getattr(self.siliconflow, "stream_stage", None)
                with (
                    stage(f"事实核验 · {cur.get('heading', cur.get('section_id', ''))}")
                    if stage
                    else nullcontext()
                ):
                    verified = verify_section(
                        section_draft=section_draft,
                        siliconflow=self.siliconflow,
                        progress=self.progress,
                        check_cancelled=self.check_cancelled,
                    )
            else:
                unsupported = sum(
                    claim.claim_type == "common" for claim in section_draft.claims
                )
                verified = VerifiedDraft(
                    section_id=section_draft.section_id,
                    verified_claims=[
                        VerifiedClaim(
                            claim=claim,
                            verdict=(
                                "unsupported"
                                if claim.claim_type == "common"
                                else "supported"
                            ),
                            verifier_rationale=(
                                "common 仅为兼容旧断点保留；请改为 fact 或 interpretation。"
                                if claim.claim_type == "common"
                                else "快速草稿模式：仅通过 source key 与引用标记结构安全门，"
                                "未执行 LLM 语义核验。"
                            ),
                        )
                        for claim in section_draft.claims
                    ],
                    unsupported_count=unsupported,
                    supported_count=len(section_draft.claims) - unsupported,
                    semantic_verification_performed=False,
                )

            sections_update = _update_section(
                state,
                {
                    "status": SECTION_STATUS_VERIFIED,
                    "verified_draft_json": verified.model_dump_json(),
                    "elapsed_seconds": cur.get("elapsed_seconds", 0.0)
                    + (time.perf_counter() - started_at),
                },
            )

            return {
                "sections": sections_update,
                "status": PIPELINE_STATUS_POLISHING,
            }
        except Exception:
            self.check_cancelled()
            logger.exception("verify_section 失败")
            raise

    @_pipeline_node("事实核对安全门")
    def fail_verification_node(self, state: WritingState) -> dict:
        """Stop the pipeline rather than publishing claims that failed verification."""

        cur = _current_section(state)
        section_id = cur.get("section_id", "?") if cur else "?"
        if cur is None or not cur.get("verified_draft_json"):
            raise PipelineNodeError("事实核对安全门", f"节 {section_id} 缺少核对结果")
        verified = VerifiedDraft.model_validate_json(cur["verified_draft_json"])
        failed = [item for item in verified.verified_claims if item.verdict != "supported"]
        details = "；".join(
            f"{item.claim.claim_id}={item.verdict}"
            f"({','.join(item.claim.source_keys) or '无来源键'})："
            f"{item.verifier_rationale[:180]}"
            for item in failed[:5]
        )
        recovery_count = cur.get("recovery_revision_count", 0)
        recovery_note = (
            f"，恢复修订 {recovery_count}/{MAX_RECOVERY_REVISIONS_PER_SECTION}"
            if recovery_count
            else ""
        )
        message = (
            f"节 {section_id} 在普通修订 {cur.get('revision_count', 0)}/"
            f"{MAX_REVISIONS_PER_SECTION}{recovery_note} 后仍有未通过核对的事实论断"
        )
        if details:
            message = f"{message}：{details}"
        raise PipelineNodeError("事实核对安全门", message)

    # ── 节点：打磨 ────────────────────────────────────────────
    # 铁律遵守：
    #   #1 persona 控文风 ✓ — 打磨用 persona 表达 DNA 润色文风
    #   #2 事实先冻结 ✓ — 此时 fact claim 已全部核对，打磨只做纯文风
    #   #5 不让作者校验自己 ✓ — 打磨后轻量核对使用中性角色
    #   #4 引用由代码拼装 ✓ — 打磨不修改 source_key 引用标记

    @_pipeline_node("内容单元打磨")
    def polish_section_node(self, state: WritingState) -> dict:
        """打磨节点：persona 表达 DNA + 已核对草稿 → 成稿 + 防漂移检查。"""
        from writing_factory.generate.polishing import polish_section

        logger.info("节点: polish_section")
        thesis = _thesis(state)
        cur = _current_section(state)
        if thesis is None or cur is None:
            raise ValueError("缺少中心信息或当前内容单元")

        verified_json = cur.get("verified_draft_json")
        if verified_json is None:
            raise ValueError("当前内容单元缺少核对结果")

        verified_draft = VerifiedDraft.model_validate_json(verified_json)
        options = _options(state)

        # 需要从 draft 中取段落文本
        draft_json = cur.get("draft_json")
        if draft_json is None:
            raise ValueError("当前内容单元缺少原始段落")

        section_draft = SectionDraft.model_validate_json(draft_json)
        started_at = time.perf_counter()

        if not options.section_polish:
            passthrough = PolishedSection(
                section_id=section_draft.section_id,
                heading=cur["heading"],
                polished_text="\n\n".join(section_draft.paragraphs).strip(),
                safety_note="已按任务选项跳过内容单元文风打磨。",
                style_polish_performed=False,
                drift_check_performed=False,
            )
            return {
                "sections": _update_section(
                    state,
                    {
                        "status": SECTION_STATUS_POLISHED,
                        "polished_section_json": passthrough.model_dump_json(),
                        "elapsed_seconds": cur.get("elapsed_seconds", 0.0)
                        + (time.perf_counter() - started_at),
                    },
                ),
                "claims_made_json": json.dumps(
                    list(
                        dict.fromkeys(
                            [
                                *json.loads(state.get("claims_made_json", "[]")),
                                *[
                                    _claim_summary(item.claim.text)
                                    for item in verified_draft.verified_claims
                                    if item.verdict == "supported"
                                ],
                            ]
                        )
                    ),
                    ensure_ascii=False,
                ),
                "status": PIPELINE_STATUS_POLISHING,
            }

        # 加载 persona spec
        context = _ctx(state)
        persona_spec = self.persona_repository.load_runtime(context.persona_id or "")
        if persona_spec is None:
            raise ValueError(f"persona '{context.persona_id}' 未就绪")
        persona_spec_json = persona_spec.model_dump(mode="json")

        try:
            stage = getattr(self.siliconflow, "stream_stage", None)
            with (
                stage(f"内容单元打磨 · {cur.get('heading', cur.get('section_id', ''))}")
                if stage
                else nullcontext()
            ):
                    polished = polish_section(
                    verified_draft=verified_draft,
                    persona_spec_json=persona_spec_json,
                    thesis=thesis,
                    section_heading=cur["heading"],
                    section_paragraphs=section_draft.paragraphs,
                        siliconflow=self.siliconflow,
                        document_form=context.generation_options.document_form,
                        genre=context.generation_options.genre,
                        check_drift=options.section_drift_check,
                    progress=self.progress,
                    check_cancelled=self.check_cancelled,
                )

            if options.evidence_mode == "conceptual_only":
                from writing_factory.generate.conceptual_safety import audit_conceptual_text

                safety = audit_conceptual_text(
                    section_id=section_draft.section_id,
                    paragraphs=[polished.polished_text],
                    task_description=context.task_description,
                    siliconflow=self.siliconflow,
                    check_cancelled=self.check_cancelled,
                )
                if not safety.safe:
                    polished = PolishedSection(
                        section_id=section_draft.section_id,
                        heading=cur["heading"],
                        polished_text="\n\n".join(section_draft.paragraphs).strip(),
                        reverted_to_verified=True,
                        safety_note="文风打磨混入外部事实，已回退到安全检查通过的正文。",
                        drift_check_performed=True,
                    )

            sections_update = _update_section(
                state,
                {
                    "status": SECTION_STATUS_POLISHED,
                    "polished_section_json": polished.model_dump_json(),
                    "elapsed_seconds": cur.get("elapsed_seconds", 0.0)
                    + (time.perf_counter() - started_at),
                },
            )
            existing_claims = json.loads(state.get("claims_made_json", "[]"))
            accepted_claims = [
                _claim_summary(item.claim.text)
                for item in verified_draft.verified_claims
                if item.verdict == "supported"
            ]
            claims_made = list(dict.fromkeys([*existing_claims, *accepted_claims]))

            return {
                "sections": sections_update,
                "claims_made_json": json.dumps(claims_made, ensure_ascii=False),
                "status": PIPELINE_STATUS_POLISHING,  # 下一节或组装
            }
        except Exception:
            self.check_cancelled()
            logger.exception("polish_section 失败")
            raise

    # ── 节点：组装 ────────────────────────────────────────────
    # 铁律遵守：
    #   #4 引用由代码拼装不由模型敲 ✓ — 引用列表由 reference_assembler 纯代码生成

    @_pipeline_node("稿件与引用组装")
    def assemble_node(self, state: WritingState) -> dict:
        """组装节点：拼接各节 polished_text + 代码生成参考文献列表。"""
        logger.info("节点: assemble")

        thesis = _thesis(state)
        if thesis is None:
            raise ValueError("缺少已冻结的中心论旨")

        # 收集所有 polished sections
        polished_sections: list[PolishedSection] = []
        all_evidence_items: list[EvidenceItem] = []

        for sec in state.get("sections", []):
            ps_json = sec.get("polished_section_json")
            if ps_json:
                ps = PolishedSection.model_validate_json(ps_json)
                if ps.fact_drift_detected:
                    raise ValueError(f"节 {ps.section_id} 的打磨结果未通过事实冻结安全门")
                polished_sections.append(ps)
            else:
                raise ValueError(f"节 {sec.get('section_id', '?')} 缺少打磨结果")

            draft_json = sec.get("draft_json")
            verified_json = sec.get("verified_draft_json")
            if not draft_json or not verified_json:
                raise ValueError(f"节 {sec.get('section_id', '?')} 缺少草稿或核对结果")
            sd = SectionDraft.model_validate_json(draft_json)
            verified = VerifiedDraft.model_validate_json(verified_json)
            if verified.unsupported_count or verified.partial_count:
                raise ValueError(f"节 {verified.section_id} 仍有未通过核对的事实论断")
            cited_keys = {
                key
                for item in verified.verified_claims
                if item.claim.claim_type == "fact" and item.verdict == "supported"
                for key in item.claim.source_keys
            }
            all_evidence_items.extend(
                item for item in sd.evidence_pack.items if item.source_key in cited_keys
            )

        # 代码拼装参考文献列表（铁律 #4）
        context = _ctx(state)
        try:
            reference_list = assemble_reference_list(
                evidence_items=all_evidence_items,
                citation_style=context.citation_style,
                kb_repository=self.kb_repository,
                kb_id=context.kb_id,
            )
            citation_display = context.generation_options.resolved_citation_display
            if citation_display == "bibliography":
                rendered_sections = render_final_citation_markers(
                    polished_sections,
                    reference_list,
                )
            else:
                rendered_sections = [
                    section.model_copy(
                        update={
                            "polished_text": _strip_internal_source_markers(
                                section.polished_text
                            )
                        }
                    )
                    for section in polished_sections
                ]
        except Exception as exc:
            self.check_cancelled()
            logger.exception("assemble_reference_list 失败")
            raise RuntimeError(f"引用拼装失败: {exc}") from exc

        # 组装最终稿
        if not _should_show_headings(context):
            rendered_sections = [
                section.model_copy(update={"heading": ""}) for section in rendered_sections
            ]
        final_draft = PolishedDraft(
            title=(
                ""
                if context.generation_options.document_form == "paragraph"
                else thesis.suggested_title or context.task_description.splitlines()[0][:100]
            ),
            sections=rendered_sections,
            reference_list=reference_list,
            citation_display=citation_display,
            thesis=thesis,
            fact_drift_free=(
                context.generation_options.quality_status
                in {"verified_final", "conceptual_draft"}
                and all(not ps.fact_drift_detected for ps in rendered_sections)
            ),
            quality_status=context.generation_options.quality_status,
            quality_notes=_quality_notes(context.generation_options),
        )

        return {
            "reference_list_json": reference_list.model_dump_json(),
            "final_draft_json": final_draft.model_dump_json(),
            "status": PIPELINE_STATUS_DONE,
        }

    # ── 节点：术语一致性审查 ────────────────────────────────────
    # 铁律遵守：
    #   #7 锚定论点 ✓ — 检查术语一致性，确保全文概念统一
    #   #5 不让作者校验自己 ✓ — 使用中性角色

    @_pipeline_node("全文并行审查")
    def parallel_reviews_node(self, state: WritingState) -> dict:
        """并发执行互不依赖的术语审查与结构审查。"""

        logger.info("节点: parallel_reviews")
        self.progress(78, "并行审查术语与全文结构")
        thesis = _thesis(state)
        outline = _outline(state)
        if thesis is None or outline is None:
            raise ValueError("缺少中心信息或内容规划，无法执行全文审查")

        term_registry = json.loads(state.get("term_registry_json", "{}"))
        options = _options(state)
        outline_nodes = [
            {
                "node_id": node.node_id,
                "heading": node.heading,
                "rhetorical_purpose": node.rhetorical_purpose,
            }
            for node in _state_outline_nodes(state, outline)
        ]
        sections = state.get("sections", [])

        def staged_call(label: str, call):
            stage = getattr(self.siliconflow, "stream_stage", None)
            with stage(label) if stage else nullcontext():
                return call()

        if options.term_review and term_registry:
            term_call = partial(
                staged_call,
                "术语一致性审查",
                partial(
                    review_term_consistency,
                    term_registry=term_registry,
                    sections=sections,
                    siliconflow=self.siliconflow,
                    check_cancelled=self.check_cancelled,
                ),
            )
        else:
            term_call = partial(
                TermConsistencyReport,
                reviewer_note="已按任务选项跳过术语一致性审查。",
            )
        if options.structure_review:
            structure_call = partial(
                staged_call,
                "全文结构审查",
                partial(
                    review_structure,
                    thesis_text=thesis.thesis_text,
                    outline_nodes=outline_nodes,
                    sections=sections,
                    siliconflow=self.siliconflow,
                    check_cancelled=self.check_cancelled,
                ),
            )
        else:
            structure_call = partial(
                StructureReview,
                overall_assessment="已按任务选项跳过全文结构审查。",
            )

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="writing-review") as executor:
            term_future = executor.submit(copy_context().run, term_call)
            structure_future = executor.submit(copy_context().run, structure_call)
            term_report = term_future.result()
            structure_report = structure_future.result()
        self.check_cancelled()
        return {
            "term_consistency_json": term_report.model_dump_json(),
            "structure_review_json": structure_report.model_dump_json(),
            "status": PIPELINE_STATUS_GLOBAL_POLISH,
        }

    @_pipeline_node("术语一致性审查")
    def term_consistency_node(self, state: WritingState) -> dict:
        """术语一致性审查节点：检查全篇术语使用是否一致。"""
        logger.info("节点: term_consistency")

        term_registry = json.loads(state.get("term_registry_json", "{}"))
        if not term_registry:
            logger.warning("term_registry 为空，跳过术语审查")
            return {
                "term_consistency_json": TermConsistencyReport().model_dump_json(),
                "status": PIPELINE_STATUS_STRUCTURE_REVIEW,
            }

        try:
            report = review_term_consistency(
                term_registry=term_registry,
                sections=state.get("sections", []),
                siliconflow=self.siliconflow,
                check_cancelled=self.check_cancelled,
            )
            return {
                "term_consistency_json": report.model_dump_json(),
                "status": PIPELINE_STATUS_STRUCTURE_REVIEW,
            }
        except Exception:
            self.check_cancelled()
            logger.exception("term_consistency 失败")
            raise

    # ── 节点：结构审查 ──────────────────────────────────────────
    # 铁律遵守：
    #   #7 锚定论点 ✓ — 评估论证结构是否与 thesis 一致

    @_pipeline_node("全文结构审查")
    def structure_review_node(self, state: WritingState) -> dict:
        """结构审查节点：检查全文结构平衡、逻辑推进、过渡衔接。"""
        logger.info("节点: structure_review")

        thesis = _thesis(state)
        if thesis is None:
            raise ValueError("缺少已冻结的中心论旨")

        outline = _outline(state)
        if outline is None:
            raise ValueError("缺少带证据映射的内容规划")

        # 扁平化提纲节点
        options = _options(state)
        all_nodes = _state_outline_nodes(state, outline)
        outline_nodes = [
            {
                "node_id": n.node_id,
                "heading": n.heading,
                "rhetorical_purpose": n.rhetorical_purpose,
                "relation_to_previous": n.relation_to_previous,
            }
            for n in all_nodes
        ]

        try:
            review = review_structure(
                thesis_text=thesis.thesis_text,
                outline_nodes=outline_nodes,
                sections=state.get("sections", []),
                siliconflow=self.siliconflow,
                document_form=options.document_form,
                genre=options.genre,
                check_cancelled=self.check_cancelled,
            )
            return {
                "structure_review_json": review.model_dump_json(),
                "status": PIPELINE_STATUS_GLOBAL_POLISH,
            }
        except Exception:
            self.check_cancelled()
            logger.exception("structure_review 失败")
            raise

    # ── 节点：全局一致性打磨（1M 上下文） ────────────────────────
    # 铁律遵守：
    #   #2 事实先冻结 ✓ — 只做术语/过渡，不改事实
    #   #4 引用由代码拼装 ✓ — 不修改 source_key
    #   #7 锚定论点 ✓ — 对照 thesis 检查整体一致性

    @_pipeline_node("全文一致性打磨")
    def global_polish_node(self, state: WritingState) -> dict:
        """全局打磨节点：利用 1M 上下文做全篇一致性审查与过渡润色。"""
        logger.info("节点: global_polish")
        self.progress(88, "全局一致性打磨")

        thesis = _thesis(state)
        if thesis is None:
            raise ValueError("缺少已冻结的中心论旨")

        options = _options(state)
        if not options.global_polish:
            sections = [
                PolishedSection.model_validate_json(section["polished_section_json"])
                for section in state.get("sections", [])
            ]
            result = GlobalPolishResult(
                sections=sections,
                global_consistency_notes="已按任务选项跳过全局一致性打磨。",
            )
            return {
                "global_polish_json": result.model_dump_json(),
                "status": PIPELINE_STATUS_ASSEMBLING,
            }

        # 加载已有审查报告（可选）
        term_report: TermConsistencyReport | None = None
        tcr = state.get("term_consistency_json")
        if tcr:
            try:
                term_report = TermConsistencyReport.model_validate_json(tcr)
            except Exception:
                pass

        struct_review: StructureReview | None = None
        sr = state.get("structure_review_json")
        if sr:
            try:
                struct_review = StructureReview.model_validate_json(sr)
            except Exception:
                pass

        try:
            result = run_global_polish(
                thesis_text=thesis.thesis_text,
                sections=state.get("sections", []),
                siliconflow=self.siliconflow,
                term_consistency_report=term_report,
                structure_review=struct_review,
                document_form=options.document_form,
                genre=options.genre,
                check_drift=options.global_drift_check,
                check_cancelled=self.check_cancelled,
            )

            if options.evidence_mode == "conceptual_only":
                from writing_factory.generate.conceptual_safety import audit_conceptual_text

                previous = {
                    PolishedSection.model_validate_json(section["polished_section_json"]).section_id:
                    PolishedSection.model_validate_json(section["polished_section_json"])
                    for section in state.get("sections", [])
                }
                checked_sections: list[PolishedSection] = []
                for candidate in result.sections:
                    safety = audit_conceptual_text(
                        section_id=candidate.section_id,
                        paragraphs=[candidate.polished_text],
                        task_description=_ctx(state).task_description,
                        siliconflow=self.siliconflow,
                        check_cancelled=self.check_cancelled,
                    )
                    if safety.safe:
                        checked_sections.append(candidate)
                    else:
                        fallback = previous[candidate.section_id]
                        checked_sections.append(
                            fallback.model_copy(
                                update={
                                    "reverted_to_verified": True,
                                    "safety_note": (
                                        "全文打磨混入外部事实，已回退到此前安全版本。"
                                    ),
                                }
                            )
                        )
                result = result.model_copy(update={"sections": checked_sections})

            # 将全局打磨后的各节文本写回 sections 状态
            sections_update = _apply_global_polish_sections(
                sections=state.get("sections", []),
                global_result=result,
            )

            return {
                "sections": sections_update,
                "global_polish_json": result.model_dump_json(),
                "status": PIPELINE_STATUS_ASSEMBLING,
            }
        except Exception:
            self.check_cancelled()
            logger.exception("global_polish 失败")
            raise


# ── 路由函数 ──────────────────────────────────────────────────


def should_continue_after_verify(state: WritingState) -> str:
    """核对后决定：继续打磨还是回退修订。

    Returns:
        "polish": 无 unsupported claim，进入打磨
        "revise": 存在 unsupported 且未达最大修订次数，回退起草
        "error": 达到最大修订次数
    """
    cur = _current_section(state)
    if cur is None:
        raise PipelineNodeError("事实核对路由", "缺少当前内容单元")

    verified_json = cur.get("verified_draft_json")
    if verified_json is None:
        raise PipelineNodeError("事实核对路由", "当前内容单元缺少核对结果")

    verified = VerifiedDraft.model_validate_json(verified_json)
    revision_count = cur.get("revision_count", 0)
    recovery_revision_count = cur.get("recovery_revision_count", 0)

    if verified.unsupported_count == 0 and verified.partial_count == 0:
        return "polish"

    if revision_count < MAX_REVISIONS_PER_SECTION:
        return "revise"

    if 0 < recovery_revision_count < MAX_RECOVERY_REVISIONS_PER_SECTION:
        return "revise"

    logger.warning(
        "节 %s 达到修订上限 %d + 恢复修订 %d/%d，仍有 %d unsupported / %d partial",
        cur.get("section_id"),
        MAX_REVISIONS_PER_SECTION,
        recovery_revision_count,
        MAX_RECOVERY_REVISIONS_PER_SECTION,
        verified.unsupported_count,
        verified.partial_count,
    )
    return "error"


def should_continue_after_polish(state: WritingState) -> str:
    """打磨后决定：下一节还是组装。

    Returns:
        "next_section": 还有未处理的节
        "assemble": 所有节已完成
    """
    sections = state.get("sections", [])
    current_idx = state.get("current_section_index", 0)

    if current_idx + 1 < len(sections):
        return "next_section"

    return "assemble"


def prepare_next_section(state: WritingState) -> dict:
    """将 current_section_index 推进到下一节。"""
    return {
        "current_section_index": state["current_section_index"] + 1,
        "status": PIPELINE_STATUS_DRAFTING,
    }


def prepare_revise_section(state: WritingState) -> dict:
    """增加修订计数，回到起草状态。"""
    cur = _current_section(state)
    if cur is None:
        raise PipelineNodeError("内容单元修订准备", "缺少当前内容单元")
    revision_count = cur.get("revision_count", 0)
    updates = {"status": SECTION_STATUS_REVISING}
    if revision_count < MAX_REVISIONS_PER_SECTION:
        updates["revision_count"] = revision_count + 1
    else:
        updates["recovery_revision_count"] = (
            cur.get("recovery_revision_count", 0) + 1
        )
    sections_update = _update_section(state, updates)
    return {"sections": sections_update, "status": PIPELINE_STATUS_DRAFTING}


# ── 内部工具 ──────────────────────────────────────────────────


def _flatten_outline_nodes(nodes: list[OutlineNode]) -> list[OutlineNode]:
    """DFS 展开提纲树为线性列表（先根遍历）。"""
    result: list[OutlineNode] = []
    for node in nodes:
        result.append(node)
        result.extend(_flatten_outline_nodes(node.children))
    return result


def _largest_source_key(outline: AnnotatedOutline) -> int:
    """Return the largest numeric source key already assigned by the framework."""

    numbers = [
        int(item.source_key.removeprefix("S"))
        for node in _flatten_outline_nodes(outline.root_nodes)
        for item in node.candidate_evidence
        if item.source_key.removeprefix("S").isdigit()
    ]
    return max(numbers, default=0)


def _strip_internal_source_markers(text: str) -> str:
    """Remove internal source keys only after all verification and drift checks finish."""

    cleaned = re.sub(r"\s*\[S\d+\]", "", text)
    return re.sub(r"[ \t]+([，。；：！？,.!?;:])", r"\1", cleaned)


def _should_show_headings(context: GenerationContext) -> bool:
    options = context.generation_options
    if options.document_form != "paper":
        return False
    task = context.task_description
    if re.search(r"不要(?:小)?标题|不设(?:小)?标题|无标题", task):
        return False
    if re.search(r"保留(?:小)?标题|使用(?:小)?标题|分节|分章", task):
        return True
    return options.genre in {
        "general_nonfiction",
        "academic_paper",
        "research_report",
        "policy_brief",
        "instructional",
        "other_nonfiction",
    }


def _apply_global_polish_sections(
    sections: list[dict],
    global_result: GlobalPolishResult,
) -> list[dict]:
    """将全局打磨后的各节文本写回 SectionState 列表。

    GlobalPolishResult 中的 sections 按顺序对应原始 sections。
    只更新 polished_section_json，保留其他字段不变。

    Args:
        sections: 原始 SectionState 列表。
        global_result: 全局打磨结果。

    Returns:
        更新后的 sections 列表。
    """
    result_sections = list(sections)
    polished_map: dict[str, str] = {}
    for ps in global_result.sections:
        polished_map[ps.section_id] = ps.model_dump_json()

    for i, sec in enumerate(result_sections):
        sid = sec.get("section_id", "")
        if sid in polished_map:
            result_sections[i] = {
                **sec,
                "polished_section_json": polished_map[sid],
            }
    return result_sections
