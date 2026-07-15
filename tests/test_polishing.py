"""Stage 4 打磨流水线 smoke tests —— polish_section + assemble_polished_draft."""

from __future__ import annotations

import json

import pytest

from writing_factory.generate.models import (
    Claim,
    EvidenceItem,
    EvidencePack,
    PolishedDraft,
    PolishedSection,
    ReferenceItem,
    ReferenceList,
    ThesisStatement,
    VerifiedClaim,
    VerifiedDraft,
)
from writing_factory.generate.polishing import (
    assemble_polished_draft,
    polish_section,
)
from writing_factory.llm.models import ChatResult

# ── Fake client ────────────────────────────────────────────────────────────


class FakeSiliconFlow:
    """按顺序返回预设 ChatResult；测试 polish_section 的两步调用。"""

    def __init__(self, responses: list[ChatResult]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if not self.responses:
            raise RuntimeError("FakeSiliconFlow 预设响应已耗尽")
        return self.responses.pop(0)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def thesis() -> ThesisStatement:
    return ThesisStatement(
        thesis_text="数字人文方法为传统文献学研究提供了可验证的新路径。",
        angle="从方法论融合的角度切入",
        kb_support_assessment="KB 中有 3 篇高被引论文直接讨论数字人文与文献学交叉",
        persona_id="persona_dh_001",
    )


@pytest.fixture
def evidence_pack() -> EvidencePack:
    return EvidencePack(
        section_id="2.1",
        items=[
            EvidenceItem(
                source_key="S1",
                chunk_id="chunk_001",
                doc_id="doc_a",
                verbatim_excerpt="数字人文方法在古籍文本分析中展现出独特优势",
            ),
            EvidenceItem(
                source_key="S2",
                chunk_id="chunk_002",
                doc_id="doc_b",
                verbatim_excerpt="传统文献学注重版本校勘与目录之学",
            ),
        ],
    )


@pytest.fixture
def verified_draft(thesis: ThesisStatement, evidence_pack: EvidencePack) -> VerifiedDraft:
    claim1 = Claim(
        claim_id="sec2.1_clm1",
        text="数字人文方法在古籍文本分析中展现出独特优势。",
        claim_type="fact",
        source_keys=["S1"],
        paragraph_index=0,
    )
    claim2 = Claim(
        claim_id="sec2.1_clm2",
        text="这一趋势为文献学方法论创新提供了契机。",
        claim_type="interpretation",
        source_keys=[],
        paragraph_index=0,
    )
    claim3 = Claim(
        claim_id="sec2.1_clm3",
        text="传统文献学注重版本校勘与目录之学。",
        claim_type="fact",
        source_keys=["S2"],
        paragraph_index=1,
    )

    return VerifiedDraft(
        section_id="2.1",
        verified_claims=[
            VerifiedClaim(
                claim=claim1,
                verdict="supported",
                verifier_rationale="chunk 原文完全支持",
                matched_chunk_text="数字人文方法在古籍文本分析中展现出独特优势",
            ),
            VerifiedClaim(
                claim=claim2,
                verdict="supported",
                verifier_rationale="interpretation 类型，无需核对",
                matched_chunk_text=None,
            ),
            VerifiedClaim(
                claim=claim3,
                verdict="supported",
                verifier_rationale="chunk 原文完全支持",
                matched_chunk_text="传统文献学注重版本校勘与目录之学",
            ),
        ],
        unsupported_count=0,
        partial_count=0,
        supported_count=3,
    )


@pytest.fixture
def persona_spec_json() -> dict[str, object]:
    return {
        "persona_id": "persona_dh_001",
        "author_name": "学者型研究者",
        "expression_dna": {
            "style_tags": {
                "sentence_length": "中长句为主，15-25字",
                "transition": "逻辑递进，多用'因此''然而''进一步而言'",
                "voice": "冷静分析型第三人称",
            },
            "taboo_words": ["众所周知", "显而易见", "毋庸置疑"],
            "tics": ["需要注意的是", "值得关注的是"],
            "sentence_fingerprint": {
                "avg_sentence_length": 22,
                "preferred_openings": ["从……角度来看", "就……而言"],
            },
        },
        "mental_models": [],
    }


@pytest.fixture
def section_paragraphs() -> list[str]:
    return [
        "数字人文方法在古籍文本分析中展现出独特优势。这一趋势为文献学方法论创新提供了契机。",
        "传统文献学注重版本校勘与目录之学。",
    ]


# ── Test: polish_section 正常流程（无漂移） ──────────────────────────────


def test_polish_section_no_drift(
    thesis: ThesisStatement,
    verified_draft: VerifiedDraft,
    persona_spec_json: dict[str, object],
    section_paragraphs: list[str],
) -> None:
    """打磨 + 防漂移检查：正常路径，无事实漂移。"""
    fake = FakeSiliconFlow(
        [
            # 第 1 步：文风打磨（纯文本输出）
            ChatResult(
                content=(
                    "从数字人文的视角来看，古籍文本分析方法展现出独特的方法论优势。"
                    "需要注意的是，这一趋势为文献学的范式创新打开了新的空间。\n\n"
                    "传统文献学素来重视版本校勘与目录之学，这构成了其学术根基。"
                ),
                model="fixture",
            ),
            # 第 2 步：防漂移核对（JSON 输出）
            ChatResult(
                content=json.dumps(
                    {
                        "section_id": "2.1",
                        "polished_text": "已忽略（由第 1 步覆盖）",
                        "fact_drift_detected": False,
                    }
                ),
                model="fixture",
            ),
        ]
    )

    result = polish_section(
        verified_draft=verified_draft,
        persona_spec_json=persona_spec_json,
        thesis=thesis,
        section_heading="数字人文与文献学融合",
        section_paragraphs=section_paragraphs,
        siliconflow=fake,
    )

    assert isinstance(result, PolishedSection)
    assert result.section_id == "2.1"
    assert result.fact_drift_detected is False
    assert "数字人文" in result.polished_text
    assert len(fake.calls) == 2

    # 验证第 1 步：应使用 thinking 模式
    call1 = fake.calls[0]
    assert call1.get("thinking") is True
    assert call1.get("reasoning_effort") == "high"

    # 验证第 2 步：应使用中性模式
    call2 = fake.calls[1]
    assert call2.get("thinking") is False
    assert call2.get("temperature") == 0.0
    assert call2.get("response_format") == "json_object"


# ── Test: polish_section 检测到事实漂移 ──────────────────────────────────


def test_polish_section_drift_detected(
    thesis: ThesisStatement,
    verified_draft: VerifiedDraft,
    persona_spec_json: dict[str, object],
    section_paragraphs: list[str],
) -> None:
    """防漂移核对检测到漂移时，必须回退到已核对正文。"""
    fake = FakeSiliconFlow(
        [
            ChatResult(content="打磨后的文本（无所谓内容）", model="fixture"),
            ChatResult(
                content=json.dumps(
                    {
                        "section_id": "2.1",
                        "polished_text": "打磨后",
                        "fact_drift_detected": True,
                    }
                ),
                model="fixture",
            ),
        ]
    )

    result = polish_section(
        verified_draft=verified_draft,
        persona_spec_json=persona_spec_json,
        thesis=thesis,
        section_heading="测试节",
        section_paragraphs=section_paragraphs,
        siliconflow=fake,
    )

    assert result.fact_drift_detected is False
    assert result.reverted_to_verified is True
    assert result.polished_text == "\n\n".join(section_paragraphs)


# ── Test: polish_section Markdown 代码围栏清理 ───────────────────────────


def test_polish_section_cleans_markdown_fences(
    thesis: ThesisStatement,
    verified_draft: VerifiedDraft,
    persona_spec_json: dict[str, object],
    section_paragraphs: list[str],
) -> None:
    """打磨输出被 Markdown 代码围栏包裹时，应正确清理。"""
    fake = FakeSiliconFlow(
        [
            ChatResult(
                content="```\n打磨后的干净正文，没有围栏。\n```",
                model="fixture",
            ),
            ChatResult(
                content=json.dumps({"section_id": "2.1", "fact_drift_detected": False}),
                model="fixture",
            ),
        ]
    )

    result = polish_section(
        verified_draft=verified_draft,
        persona_spec_json=persona_spec_json,
        thesis=thesis,
        section_heading="测试节",
        section_paragraphs=section_paragraphs,
        siliconflow=fake,
    )

    assert "```" not in result.polished_text
    assert "打磨后的干净正文，没有围栏。" in result.polished_text


# ── Test: polish_section JSON 解析失败 ───────────────────────────────────


def test_polish_section_json_parse_error(
    thesis: ThesisStatement,
    verified_draft: VerifiedDraft,
    persona_spec_json: dict[str, object],
    section_paragraphs: list[str],
) -> None:
    """防漂移核对返回非法 JSON 时，应失败闭合并回退。"""
    fake = FakeSiliconFlow(
        [
            ChatResult(content="打磨文本", model="fixture"),
            ChatResult(content="这不是合法的 JSON", model="fixture"),
        ]
    )

    result = polish_section(
        verified_draft=verified_draft,
        persona_spec_json=persona_spec_json,
        thesis=thesis,
        section_heading="测试节",
        section_paragraphs=section_paragraphs,
        siliconflow=fake,
    )
    assert result.reverted_to_verified is True
    assert result.fact_drift_detected is False


# ── Test: assemble_polished_draft 全部无漂移 ─────────────────────────────


def test_assemble_polished_draft_all_clean(thesis: ThesisStatement) -> None:
    """所有节都无漂移时，fact_drift_free=True。"""
    sections = [
        PolishedSection(
            section_id="1.1",
            polished_text="第一节正文",
            fact_drift_detected=False,
        ),
        PolishedSection(
            section_id="1.2",
            polished_text="第二节正文",
            fact_drift_detected=False,
        ),
    ]
    ref_list = ReferenceList(
        items=[
            ReferenceItem(
                source_key="S1",
                citation_text="张三. 数字人文导论. 2020.",
                doc_id="doc_a",
                chunk_id="chunk_001",
            ),
        ],
        style="gb-t-7714",
    )

    result = assemble_polished_draft(
        polished_sections=sections,
        reference_list=ref_list,
        thesis=thesis,
    )

    assert isinstance(result, PolishedDraft)
    assert result.fact_drift_free is True
    assert len(result.sections) == 2
    assert result.thesis is thesis
    assert result.reference_list is ref_list


# ── Test: assemble_polished_draft 部分漂移 ───────────────────────────────


def test_assemble_polished_draft_partial_drift(thesis: ThesisStatement) -> None:
    """存在漂移节时，fact_drift_free=False。"""
    sections = [
        PolishedSection(
            section_id="1.1",
            polished_text="干净的节",
            fact_drift_detected=False,
        ),
        PolishedSection(
            section_id="1.2",
            polished_text="有漂移的节",
            fact_drift_detected=True,
        ),
        PolishedSection(
            section_id="1.3",
            polished_text="另一个干净的节",
            fact_drift_detected=False,
        ),
    ]
    ref_list = ReferenceList(items=[], style="gb-t-7714")

    result = assemble_polished_draft(
        polished_sections=sections,
        reference_list=ref_list,
        thesis=thesis,
    )

    assert result.fact_drift_free is False
    assert len(result.sections) == 3


# ── Test: assemble_polished_draft 空列表 ─────────────────────────────────


def test_assemble_polished_draft_empty(thesis: ThesisStatement) -> None:
    """空节列表，fact_drift_free=True。"""
    result = assemble_polished_draft(
        polished_sections=[],
        reference_list=ReferenceList(items=[], style="gb-t-7714"),
        thesis=thesis,
    )

    assert result.fact_drift_free is True
    assert len(result.sections) == 0


# ── Test: progress / cancellation 回调 ────────────────────────────────────


def test_polish_section_progress_callback(
    thesis: ThesisStatement,
    verified_draft: VerifiedDraft,
    persona_spec_json: dict[str, object],
    section_paragraphs: list[str],
) -> None:
    """验证进度回调被正确调用。"""
    progress_log: list[tuple[int, str]] = []

    fake = FakeSiliconFlow(
        [
            ChatResult(content="打磨文本", model="fixture"),
            ChatResult(
                content=json.dumps({"section_id": "2.1", "fact_drift_detected": False}),
                model="fixture",
            ),
        ]
    )

    polish_section(
        verified_draft=verified_draft,
        persona_spec_json=persona_spec_json,
        thesis=thesis,
        section_heading="测试节",
        section_paragraphs=section_paragraphs,
        siliconflow=fake,
        progress=lambda p, m: progress_log.append((p, m)),
    )

    assert len(progress_log) >= 2
    percentages = [p for p, _ in progress_log]
    assert percentages == sorted(percentages)
    assert percentages[-1] == 100


def test_polish_section_cancellation(
    thesis: ThesisStatement,
    verified_draft: VerifiedDraft,
    persona_spec_json: dict[str, object],
    section_paragraphs: list[str],
) -> None:
    """取消检查应在第一次回调时截停。"""
    fake = FakeSiliconFlow(
        [
            ChatResult(content="打磨文本", model="fixture"),
            ChatResult(
                content=json.dumps({"section_id": "2.1", "fact_drift_detected": False}),
                model="fixture",
            ),
        ]
    )

    with pytest.raises(RuntimeError, match="用户取消"):
        polish_section(
            verified_draft=verified_draft,
            persona_spec_json=persona_spec_json,
            thesis=thesis,
            section_heading="测试节",
            section_paragraphs=section_paragraphs,
            siliconflow=fake,
            check_cancelled=lambda: (_ for _ in ()).throw(RuntimeError("用户取消")),
        )
