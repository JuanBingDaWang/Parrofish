"""文风打磨：persona 表达 DNA + 已核对草稿 → 成稿 + 轻量防漂移检查。

这是生成流水线（阶段 4）的第五步，产出 PolishedSection。

设计铁律（§3）本步骤遵守情况：
- #1 "persona 用来控文风不是事实"：打磨使用 persona 表达 DNA
  （句式指纹、风格标签、禁忌词），不动事实。
- #2 "事实先冻结、文风最后加"：此时 fact claim 已全部核对，打磨只做纯文风操作。
- #5 "不让作者校验自己"：打磨后轻量核对使用中性角色，不加载 persona。
- #4 "引用由代码拼装不由模型敲"：打磨不修改 source_key 引用标记。
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from writing_factory.generate.models import (
    PolishedDraft,
    PolishedSection,
    ReferenceList,
    ThesisStatement,
    VerifiedDraft,
)
from writing_factory.generate.prompts import (
    polish_fact_check_messages,
    polishing_messages,
)

if TYPE_CHECKING:
    from writing_factory.llm.siliconflow import SiliconFlowClient

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], None]


def _no_progress(_percent: int, _message: str) -> None:
    pass


def _no_cancellation() -> None:
    pass


def polish_section(
    *,
    verified_draft: VerifiedDraft,
    persona_spec_json: dict[str, Any],
    thesis: ThesisStatement,
    section_heading: str,
    section_paragraphs: list[str],
    siliconflow: SiliconFlowClient,
    progress: ProgressCallback = _no_progress,
    check_cancelled: CancellationCheck = _no_cancellation,
) -> PolishedSection:
    """对单节已核对草稿进行文风打磨 + 防漂移检查。

    两步流水线：
        1. 文风打磨（铁律 #1）：加载 persona 表达 DNA，纯文风操作
        2. 轻量防漂移核对（铁律 #5）：中性角色比对打磨前后事实一致性

    Args:
        verified_draft: 核对后的草稿（所有 fact claims 已验证通过）
        persona_spec_json: 序列化后的 PersonaSpec（主要使用 expression_dna 部分）
        thesis: 锚定论点
        section_heading: 本节标题
        section_paragraphs: 本节段落正文（已核对）
        siliconflow: SiliconFlow 客户端
        progress: 进度回调 (percent, message)
        check_cancelled: 取消检查回调，被取消时抛出异常

    Returns:
        PolishedSection: 打磨后的单节成稿，含事实漂移检测结果

    Raises:
        ExternalServiceError: LLM 调用失败
    """
    # ── 第 1 步：文风打磨（铁律 #1：persona 控文风，不动事实） ───
    progress(10, "构造文风打磨请求")
    check_cancelled()

    polish_msgs = polishing_messages(
        persona_spec_json=persona_spec_json,
        verified_draft=verified_draft,
        section_paragraphs=section_paragraphs,
        section_heading=section_heading,
        thesis=thesis,
    )

    progress(30, "LLM 文风打磨中")
    check_cancelled()

    polish_result = siliconflow.chat(
        messages=polish_msgs,
        thinking=True,  # 铁律 #1：使用 persona 思考模式
        reasoning_effort="high",
        temperature=0.7,  # 文风打磨允许一定随机性
        max_tokens=8192,
        seed=42,
        stream=True,
    )

    polished_text = polish_result.content.strip()
    verified_text = "\n\n".join(section_paragraphs).strip()

    # 清理可能的 Markdown 代码围栏
    if polished_text.startswith("```"):
        lines = polished_text.split("\n")
        if len(lines) > 2:
            polished_text = "\n".join(lines[1:-1]).strip()

    if protected_tokens(polished_text) != protected_tokens(verified_text):
        logger.warning("节 '%s' 的打磨候选改动了数字或引用标记，已回退", verified_draft.section_id)
        return PolishedSection(
            section_id=verified_draft.section_id,
            heading=section_heading,
            polished_text=verified_text,
            fact_drift_detected=False,
            reverted_to_verified=True,
            safety_note="代码安全门检测到数字或引用标记变化，已回退到核对通过的正文。",
        )

    progress(60, "打磨后轻量防漂移核对")

    # ── 第 2 步：轻量防漂移核对（铁律 #5：中性角色，不加载 persona） ─
    check_cancelled()

    fact_check_msgs = polish_fact_check_messages(
        original_paragraphs=section_paragraphs,
        polished_paragraphs=[polished_text],
        verified_claims=verified_draft.verified_claims,
    )

    progress(75, "LLM 防漂移检查中")
    check_cancelled()

    fact_check_result = siliconflow.chat(
        messages=fact_check_msgs,
        thinking=False,  # 铁律 #5：中性角色，不开思考模式
        temperature=0.0,
        max_tokens=8192,
        response_format="json_object",
        seed=42,
        stream=True,
    )

    progress(90, "解析打磨结果")
    check_cancelled()

    # ── 解析防漂移结果 ──────────────────────────────────────────────
    try:
        data = json.loads(fact_check_result.content)
    except json.JSONDecodeError as e:
        logger.error(
            "防漂移核对 JSON 解析失败: %s，原始输出前 500 字符: %s",
            e,
            fact_check_result.content[:500],
        )
        return PolishedSection(
            section_id=verified_draft.section_id,
            heading=section_heading,
            polished_text=verified_text,
            fact_drift_detected=False,
            reverted_to_verified=True,
            safety_note=f"防漂移核对结果无法解析，已按失败闭合原则回退：{e}",
        )

    fact_drift_detected = data.get("fact_drift_detected", False)

    if fact_drift_detected:
        logger.warning("节 '%s' 检测到事实漂移，已回退到冻结事实版本", verified_draft.section_id)
        return PolishedSection(
            section_id=verified_draft.section_id,
            heading=section_heading,
            polished_text=verified_text,
            fact_drift_detected=False,
            reverted_to_verified=True,
            safety_note="中性核对检测到事实漂移，已回退到核对通过的正文。",
        )

    progress(100, "打磨完成")

    return PolishedSection(
        section_id=verified_draft.section_id,
        heading=section_heading,
        polished_text=polished_text,
        fact_drift_detected=fact_drift_detected,
    )


def protected_tokens(text: str) -> Counter[str]:
    """Extract source markers and number-like tokens that style passes must preserve."""

    return Counter(re.findall(r"\[S\d+\]|\d+(?:\.\d+)?%?|[一二三四五六七八九十]+年", text))


def assemble_polished_draft(
    *,
    polished_sections: list[PolishedSection],
    reference_list: ReferenceList,
    thesis: ThesisStatement,
) -> PolishedDraft:
    """将逐节打磨结果组装为全篇成稿。

    Args:
        polished_sections: 逐节打磨后的正文列表
        reference_list: 代码拼装的参考文献列表
        thesis: 锚定论点

    Returns:
        PolishedDraft: 全篇成稿
    """
    fact_drift_free = not any(s.fact_drift_detected for s in polished_sections)

    if not fact_drift_free:
        drift_sections = [s.section_id for s in polished_sections if s.fact_drift_detected]
        logger.warning(
            "全篇存在事实漂移的节: %s，建议人工复核",
            ", ".join(drift_sections),
        )

    return PolishedDraft(
        sections=polished_sections,
        reference_list=reference_list,
        thesis=thesis,
        fact_drift_free=fact_drift_free,
    )
