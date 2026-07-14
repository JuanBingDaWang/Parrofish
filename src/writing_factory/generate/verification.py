"""事实论断核对：逐 claim 比对 chunk 原文 → 判定 supported/partial/unsupported。

这是生成流水线（阶段 4）的第四步，产出 VerifiedDraft。

设计铁律（§3）本步骤遵守情况：
- #5 "不让作者校验自己"：核对使用中性角色，不加载 persona，不传入 persona 上下文。
- #2 "事实先冻结、文风最后加"：核对只检查事实性论断（fact 类型），不碰 interpretation/common。
- #4 "引用由代码拼装不由模型敲"：核对不产生新引用，只比对已有的 source_key → chunk 原文。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from writing_factory.generate.models import (
    Claim,
    SectionDraft,
    VerificationResponse,
    VerifiedClaim,
    VerifiedDraft,
)
from writing_factory.generate.prompts import verification_messages

if TYPE_CHECKING:
    from writing_factory.llm.siliconflow import SiliconFlowClient

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], None]


def _no_progress(_percent: int, _message: str) -> None:
    pass


def _no_cancellation() -> None:
    pass


def verify_section(
    *,
    section_draft: SectionDraft,
    siliconflow: SiliconFlowClient,
    progress: ProgressCallback = _no_progress,
    check_cancelled: CancellationCheck = _no_cancellation,
) -> VerifiedDraft:
    """核对单节草稿中的事实论断：逐 claim 比对 chunk 原文。

    流水线步骤：
        1. 从 section_draft 提取 fact 类型 claim（interpretation/common 跳过）
        2. 为每个 fact claim 匹配对应的 chunk 原文
        3. 构造核对消息（不加载 persona —— 铁律 #5）
        4. 调用 LLM（thinking=False，中性角色，低温度）
        5. 解析为 VerifiedDraft

    Args:
        section_draft: 起草阶段产出的单节结构化草稿
        siliconflow: SiliconFlow 客户端
        progress: 进度回调 (percent, message)
        check_cancelled: 取消检查回调，被取消时抛出异常

    Returns:
        VerifiedDraft: 逐 claim 核对结果，含 verdict 与 verifier_rationale

    Raises:
        ExternalServiceError: LLM 调用失败
    """
    # ── 1. 快速检查：是否有 fact 论断需要核对 ──────────────────────
    fact_claims = [c for c in section_draft.claims if c.claim_type == "fact"]
    non_fact_claims = [c for c in section_draft.claims if c.claim_type != "fact"]

    if not fact_claims:
        logger.info("本节无 fact 类型论断，跳过核对，所有 claim 原样保留。")
        return VerifiedDraft(
            section_id=section_draft.section_id,
            verified_claims=[
                VerifiedClaim(
                    claim=c,
                    verdict="supported",
                    verifier_rationale=f"{c.claim_type} 类型论断无需核对，原样保留。",
                    matched_chunk_text=None,
                )
                for c in non_fact_claims
            ],
            unsupported_count=0,
            partial_count=0,
            supported_count=len(non_fact_claims),
        )

    progress(10, f"核对 {len(fact_claims)} 条 fact 论断")

    # ── 2. 构造核对消息（不加载 persona —— 铁律 #5） ──────────────
    progress(30, "构造核对请求")
    check_cancelled()

    messages = verification_messages(section_draft=section_draft)

    def parse_response(content: str) -> dict:
        normalized = _normalize_verification_response(json.loads(content))
        parsed = VerificationResponse.model_validate(normalized)
        if parsed.section_id != section_draft.section_id:
            raise ValueError("核对结果与当前章节不匹配")
        return parsed.model_dump(mode="python")

    # ── 3. 调用 LLM 进行核对 ────────────────────────────────────────
    progress(50, "LLM 逐 claim 核对中")
    check_cancelled()

    result = siliconflow.chat(
        messages=messages,
        thinking=False,  # 铁律 #5：中性角色，不开思考模式
        temperature=0.0,  # 最低温度，确保一致性
        max_tokens=8192,
        response_format="json_object",
        seed=42,
        stream=True,
        result_validator=lambda candidate: parse_response(candidate.content),
    )

    progress(80, "解析核对结果")
    check_cancelled()

    # ── 4. 解析 LLM 输出 ────────────────────────────────────────────
    try:
        data = parse_response(result.content)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error(
            "核对结果无法解析，响应长度=%d，错误类型=%s",
            len(result.content),
            type(exc).__name__,
        )
        raise ValueError(f"核对结果无法解析: {exc}") from exc

    # ── 5. 构造 VerifiedDraft ────────────────────────────────────────
    verified_claims: list[VerifiedClaim] = _parse_verified_claims(
        data=data,
        section_draft=section_draft,
    )

    unsupported_count = sum(1 for vc in verified_claims if vc.verdict == "unsupported")
    partial_count = sum(1 for vc in verified_claims if vc.verdict == "partial")
    supported_count = sum(1 for vc in verified_claims if vc.verdict == "supported")

    progress(
        100,
        f"核对完成：{supported_count} supported, {partial_count} partial, "
        f"{unsupported_count} unsupported",
    )

    return VerifiedDraft(
        section_id=section_draft.section_id,
        verified_claims=verified_claims,
        unsupported_count=unsupported_count,
        partial_count=partial_count,
        supported_count=supported_count,
    )


def _parse_verified_claims(
    *,
    data: dict,
    section_draft: SectionDraft,
) -> list[VerifiedClaim]:
    """从 LLM 返回的 JSON 中解析 VerifiedClaim 列表。

    以 LLM 返回的 verified_claims 为准，但会用原始 claim 补充缺失字段。
    """
    raw_claims: list[dict] = data.get("verified_claims", [])

    # 建立原始 claim 索引
    claim_by_id: dict[str, Claim] = {
        claim.claim_id: claim for claim in section_draft.claims if claim.claim_type == "fact"
    }
    evidence_by_key = {
        item.source_key: item.verbatim_excerpt for item in section_draft.evidence_pack.items
    }

    result: list[VerifiedClaim] = []
    returned_ids: set[str] = set()
    for rc in raw_claims:
        claim_id = rc.get("claim_id", "")
        original = claim_by_id.get(claim_id)

        if original is None:
            logger.warning("LLM 返回了未知 claim_id '%s'，跳过", claim_id)
            continue
        if claim_id in returned_ids:
            logger.warning("LLM 重复返回 claim_id '%s'，忽略后续结果", claim_id)
            continue
        returned_ids.add(claim_id)

        verdict_raw = rc.get("verdict", "supported")
        # 验证 verdict 值
        if verdict_raw not in ("supported", "partial", "unsupported"):
            logger.warning(
                "claim '%s' 的 verdict 值无效 '%s'，回退为 'unsupported'",
                claim_id,
                verdict_raw,
            )
            verdict_raw = "unsupported"

        result.append(
            VerifiedClaim(
                claim=original,
                verdict=verdict_raw,
                verifier_rationale=rc.get("verifier_rationale", "") or "核对者未提供理由",
                matched_chunk_text="\n---\n".join(
                    evidence_by_key[key] for key in original.source_keys if key in evidence_by_key
                ),
            )
        )

    # 补全 LLM 未返回的 claim（视为 unsupported）
    for c in section_draft.claims:
        if c.claim_id not in returned_ids:
            if c.claim_type == "fact":
                logger.warning("fact claim '%s' 未被 LLM 返回，标记为 unsupported", c.claim_id)
                result.append(
                    VerifiedClaim(
                        claim=c,
                        verdict="unsupported",
                        verifier_rationale="LLM 未返回该 claim 的核对结果，标记为 unsupported。",
                        matched_chunk_text=None,
                    )
                )
            else:
                result.append(
                    VerifiedClaim(
                        claim=c,
                        verdict="supported",
                        verifier_rationale=f"{c.claim_type} 类型论断无需核对，原样保留。",
                        matched_chunk_text=None,
                    )
                )

    return result


def _normalize_verification_response(data: dict) -> dict:
    """Accept old cached nested decisions while emitting the new flat contract."""

    normalized: list[dict] = []
    for raw in data.get("verified_claims", []):
        item = dict(raw)
        nested_claim = item.pop("claim", None)
        if not item.get("claim_id") and isinstance(nested_claim, dict):
            item["claim_id"] = nested_claim.get("claim_id", "")
        normalized.append(item)
    return {
        "section_id": data.get("section_id", ""),
        "verified_claims": normalized,
    }
