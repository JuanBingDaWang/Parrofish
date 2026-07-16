"""阶段 6 一致性与全局打磨：术语审查、结构审查、1M 上下文全局打磨。

这些步骤在逐节循环完成后运行，做最后一次全篇一致性检查与加工。

设计铁律遵守:
    #2 事实先冻结 ✓ — 全局打磨不修改事实内容，只做术语和过渡
    #4 引用由代码拼装 ✓ — 不修改 source_key 引用标记
    #7 锚定论点 ✓ — 全局审查对照 thesis 检查是否偏离
    #5 不让作者校验自己 ✓ — 术语/结构审查使用中性角色
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from writing_factory.generate.models import (
    DocumentForm,
    GlobalPolishResult,
    PolishedSection,
    StructureReview,
    TermConsistencyReport,
    VerifiedDraft,
)
from writing_factory.generate.polishing import protected_tokens
from writing_factory.generate.prompts import (
    global_polish_messages,
    polish_fact_check_messages,
    structure_review_messages,
    term_consistency_messages,
)
from writing_factory.nonfiction import NonfictionGenre

if TYPE_CHECKING:
    from writing_factory.llm.siliconflow import SiliconFlowClient

logger = logging.getLogger(__name__)


def _no_cancellation() -> None:
    return None


# ── 辅助函数 ──────────────────────────────────────────────────


def _collect_sections_text(
    sections: list[dict],
) -> list[dict[str, str]]:
    """从 SectionState 列表中提取各节标题与正文。

    Returns:
        [{section_id, heading, text}]
    """
    result: list[dict[str, str]] = []
    for sec in sections:
        ps_json = sec.get("polished_section_json")
        if not ps_json:
            continue
        try:
            ps = PolishedSection.model_validate_json(ps_json)
            result.append(
                {
                    "section_id": sec.get("section_id", ""),
                    "heading": sec.get("heading", ""),
                    "text": ps.polished_text,
                }
            )
        except Exception:
            logger.warning("跳过无法解析的节 %s", sec.get("section_id"))
    return result


# ── 术语一致性审查 ────────────────────────────────────────────


def review_term_consistency(
    *,
    term_registry: dict[str, str],
    sections: list[dict],
    siliconflow: SiliconFlowClient,
    check_cancelled: Callable[[], None] = _no_cancellation,
) -> TermConsistencyReport:
    """检查全篇术语一致性。

    Args:
        term_registry: 术语登记表 {术语名: 定义}。
        sections: SectionState 列表（含 polished_section_json）。
        siliconflow: SiliconFlow 客户端。

    Returns:
        TermConsistencyReport：术语不一致问题列表 + 一致术语 + 总体评价。
    """
    sections_text = _collect_sections_text(sections)
    if not sections_text:
        logger.warning("review_term_consistency: 无已打磨的节，返回空报告")
        return TermConsistencyReport()

    messages = term_consistency_messages(
        term_registry=term_registry,
        sections_text=sections_text,
    )

    check_cancelled()
    result = siliconflow.chat(
        messages=messages,
        thinking=False,  # 中性角色，不开思考
        temperature=0.0,  # 最低温度，确保一致性
        max_tokens=8192,
        response_format="json_object",
        seed=42,
        stream=True,
        step_id="writing.term_review",
    )

    check_cancelled()
    try:
        data = json.loads(result.content)
        report = TermConsistencyReport.model_validate(data)
        logger.info(
            "术语审查完成: %d 个不一致, %d 个一致术语",
            len(report.issues),
            len(report.consistent_terms),
        )
        return report
    except Exception as exc:
        logger.error("术语一致性报告解析失败: %s", exc)
        return TermConsistencyReport(
            reviewer_note=f"术语一致性审查时解析失败: {exc}",
        )


# ── 结构审查 ──────────────────────────────────────────────────


def review_structure(
    *,
    thesis_text: str,
    outline_nodes: list[dict[str, Any]],
    sections: list[dict],
    siliconflow: SiliconFlowClient,
    document_form: DocumentForm = "paper",
    genre: NonfictionGenre = "general_nonfiction",
    check_cancelled: Callable[[], None] = _no_cancellation,
) -> StructureReview:
    """审查全文结构：节篇幅平衡、逻辑推进、过渡衔接。

    Args:
        thesis_text: 核心论点文本。
        outline_nodes: 提纲节点列表 [{node_id, heading, rhetorical_purpose}]。
        sections: SectionState 列表（含 polished_section_json）。
        siliconflow: SiliconFlow 客户端。

    Returns:
        StructureReview：结构问题列表 + 总体评价。
    """
    sections_text = _collect_sections_text(sections)
    if not sections_text:
        logger.warning("review_structure: 无已打磨的节，返回空报告")
        return StructureReview(overall_assessment="全文为空，无法审查。")

    messages = structure_review_messages(
        thesis_text=thesis_text,
        outline_nodes=outline_nodes,
        sections_text=sections_text,
        document_form=document_form,
        genre=genre,
    )

    check_cancelled()
    result = siliconflow.chat(
        messages=messages,
        thinking=False,  # 中性角色，不开思考
        temperature=0.0,
        max_tokens=8192,
        response_format="json_object",
        seed=42,
        stream=True,
        step_id="writing.structure_review",
    )

    check_cancelled()
    try:
        data = json.loads(result.content)
        review = StructureReview.model_validate(data)
        logger.info(
            "结构审查完成: %d 个问题",
            len(review.issues),
        )
        return review
    except Exception as exc:
        logger.error("结构审查报告解析失败: %s", exc)
        return StructureReview(
            overall_assessment=f"结构审查时解析失败: {exc}",
        )


# ── 全局一致性打磨（1M 上下文） ───────────────────────────────


def run_global_polish(
    *,
    thesis_text: str,
    sections: list[dict],
    siliconflow: SiliconFlowClient,
    term_consistency_report: TermConsistencyReport | None = None,
    structure_review: StructureReview | None = None,
    document_form: DocumentForm = "paper",
    genre: NonfictionGenre = "general_nonfiction",
    check_drift: bool = True,
    check_cancelled: Callable[[], None] = _no_cancellation,
) -> GlobalPolishResult:
    """利用 1M 上下文做全篇全局一致性打磨。

    此步骤是最后一次全篇通读，做以下工作：
    1. 添加/完善节间过渡
    2. 修正术语不一致
    3. 小幅调整结构问题
    4. 确保与 thesis 锚定一致

    Args:
        thesis_text: 核心论点文本。
        sections: SectionState 列表（含 polished_section_json）。
        siliconflow: SiliconFlow 客户端。
        term_consistency_report: 术语一致性报告（可选，用于提供修正指导）。
        structure_review: 结构审查报告（可选，用于提供修正指导）。

    Returns:
        GlobalPolishResult：全局打磨后的各节正文 + 变更说明。
    """
    sections_text = _collect_sections_text(sections)
    if not sections_text:
        logger.warning("run_global_polish: 无已打磨的节，返回空结果")
        return GlobalPolishResult(sections=[])

    term_json: str | None = None
    struct_json: str | None = None
    if term_consistency_report is not None:
        try:
            term_json = term_consistency_report.model_dump_json()
        except Exception:
            pass
    if structure_review is not None:
        try:
            struct_json = structure_review.model_dump_json()
        except Exception:
            pass

    messages = global_polish_messages(
        thesis_text=thesis_text,
        sections_text=sections_text,
        term_consistency_json=term_json,
        structure_review_json=struct_json,
        document_form=document_form,
        genre=genre,
    )

    # 全局打磨使用思考模式（低）— 这是最后一次全篇审查，值得投入推理
    check_cancelled()
    result = siliconflow.chat(
        messages=messages,
        thinking=True,
        reasoning_effort="high",
        temperature=0.1,
        max_tokens=8192,  # 1M 上下文的输出可能较大
        response_format="json_object",
        seed=42,
        stream=True,
        step_id="writing.global_polish",
    )

    check_cancelled()
    try:
        data = json.loads(result.content)
        global_result = GlobalPolishResult.model_validate(data)
        original_sections = _original_polished_sections(sections)
        if [item.section_id for item in global_result.sections] != [
            item.section_id for item in original_sections
        ]:
            raise ValueError("全局打磨返回的内容单元集合或顺序发生变化")
        global_result = global_result.model_copy(
            update={
                "sections": [
                    candidate.model_copy(update={"heading": original.heading})
                    for candidate, original in zip(
                        global_result.sections,
                        original_sections,
                        strict=True,
                    )
                ]
            }
        )
        if any(item.fact_drift_detected for item in global_result.sections):
            return _reverted_global_result(
                original_sections,
                "全局打磨结果自报事实漂移，已回退。",
            )
        original_text = "\n\n".join(item.polished_text for item in original_sections)
        candidate_text = "\n\n".join(item.polished_text for item in global_result.sections)
        if protected_tokens(original_text) != protected_tokens(candidate_text):
            return _reverted_global_result(
                original_sections,
                "代码安全门检测到数字或引用标记变化，已回退全局打磨。",
            )

        if not check_drift:
            return global_result.model_copy(
                update={
                    "sections": [
                        section.model_copy(update={"drift_check_performed": False})
                        for section in global_result.sections
                    ],
                    "global_consistency_notes": (
                        f"{global_result.global_consistency_notes} "
                        "已按任务选项跳过全局 LLM 防漂移检查。"
                    ).strip(),
                }
            )

        verified_claims = []
        for section in sections:
            raw = section.get("verified_draft_json")
            if raw:
                verified_claims.extend(VerifiedDraft.model_validate_json(raw).verified_claims)
        check_cancelled()
        check_result = siliconflow.chat(
            messages=polish_fact_check_messages(
                original_paragraphs=[item.polished_text for item in original_sections],
                polished_paragraphs=[item.polished_text for item in global_result.sections],
                verified_claims=verified_claims,
            ),
            thinking=False,
            temperature=0.0,
            max_tokens=8192,
            response_format="json_object",
            seed=42,
            stream=True,
            step_id="writing.global_drift",
        )
        check_cancelled()
        check_data = json.loads(check_result.content)
        if check_data.get("fact_drift_detected", True):
            return _reverted_global_result(
                original_sections,
                "中性核对检测到事实漂移，已回退全局打磨。",
            )
        logger.info(
            "全局打磨完成: %d 节, %d 处过渡新增",
            len(global_result.sections),
            len(global_result.transitions_added),
        )
        return global_result
    except Exception as exc:
        logger.error("全局打磨结果解析失败: %s", exc)
        # 解析失败时返回原始 sections，不做修改
        return _reverted_global_result(
            _original_polished_sections(sections),
            f"全局打磨或安全核对失败: {exc}，已保留原始底稿。",
        )


def _original_polished_sections(sections: list[dict]) -> list[PolishedSection]:
    return [
        PolishedSection.model_validate_json(section["polished_section_json"])
        for section in sections
        if section.get("polished_section_json")
    ]


def _reverted_global_result(
    sections: list[PolishedSection],
    note: str,
) -> GlobalPolishResult:
    safe_sections = [
        item.model_copy(
            update={
                "fact_drift_detected": False,
                "reverted_to_verified": True,
                "safety_note": note,
            }
        )
        for item in sections
    ]
    return GlobalPolishResult(sections=safe_sections, global_consistency_notes=note)
