"""Evaluation-specific data models for faithfulness, LLM-judge, and golden regression."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Faithfulness (RAGAS-style)
# ---------------------------------------------------------------------------

AtomicVerdict = Literal["supported", "unsupported"]


class AtomicClaim(BaseModel):
    """A single atomic claim decomposed from generated text."""

    model_config = ConfigDict(frozen=True)

    claim_text: str = Field(description="原子化论断原文，不可改写")
    verdict: AtomicVerdict = Field(description="该论断是否被检索到的上下文支持")
    evidence: str | None = Field(
        default=None,
        description="支持或否定该论断的原文片段",
    )


class FaithfulnessResult(BaseModel):
    """Faithfulness evaluation result for one QA pair."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=1.0, description="忠实度分数")
    atomic_claims: list[AtomicClaim] = Field(description="原子论断逐条判定")
    supported_count: int = Field(ge=0, description="被支持的论断数")
    unsupported_count: int = Field(ge=0, description="不被支持的论断数")


class CitationTraceabilityResult(BaseModel):
    """Hard, code-derived generation safety metrics."""

    model_config = ConfigDict(frozen=True)

    fact_claim_count: int = Field(ge=0)
    valid_citation_count: int = Field(ge=0)
    supported_fact_count: int = Field(ge=0)
    valid_citation_ratio: float = Field(ge=0.0, le=1.0)
    verified_support_ratio: float = Field(ge=0.0, le=1.0)
    hallucination_rate: float = Field(ge=0.0, le=1.0)
    passed: bool


# ---------------------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------------------


class JudgeDimension(BaseModel):
    """Single dimension score from LLM-as-judge."""

    model_config = ConfigDict(frozen=True)

    dimension: str = Field(description="评估维度名称，如论点清晰度")
    score: int = Field(ge=1, le=5, description="1-5 评分")
    rationale: str = Field(description="该维度评分的推理依据")


class JudgeResult(BaseModel):
    """LLM-as-judge evaluation result."""

    model_config = ConfigDict(frozen=True)

    dimensions: list[JudgeDimension] = Field(description="各维度评分")
    overall_score: float = Field(ge=1.0, le=5.0, description="综合评分（各维度算术平均）")
    judge_rationale: str = Field(description="裁判的综合评语")
    evaluation_error: str | None = Field(
        default=None, description="评估调用失败时的错误；非空结果不得视为正常评分"
    )


# ---------------------------------------------------------------------------
# Golden Regression
# ---------------------------------------------------------------------------


class GoldenCase(BaseModel):
    """One golden regression test case."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(description="用例标识")
    task_description: str = Field(description="写作任务描述")
    expected_min_scores: dict[str, float] = Field(
        description="维度 → 最低通过分数，如 {'faithfulness': 0.8, 'argument_quality': 3.5}",
    )


class GoldenSuite(BaseModel):
    """A named suite of golden regression cases."""

    model_config = ConfigDict(frozen=True)

    suite_name: str = Field(description="测试套件名称")
    cases: list[GoldenCase] = Field(description="用例列表")


class GoldenRunResult(BaseModel):
    """Result of evaluating one golden case."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(description="用例标识")
    actual_scores: dict[str, float] = Field(description="各维度实际得分")
    all_passed: bool = Field(description="是否全部通过阈值")
    failures: list[str] = Field(description="未通过的维度描述列表")


class GoldenSuiteResult(BaseModel):
    """Aggregated result of a full golden suite run."""

    model_config = ConfigDict(frozen=True)

    suite_name: str = Field(description="套件名称")
    case_results: list[GoldenRunResult] = Field(description="各用例结果")
    pass_rate: float = Field(ge=0.0, le=1.0, description="通过率")
    overall_pass: bool = Field(description="是否全部通过")


# ---------------------------------------------------------------------------
# Injection Detection
# ---------------------------------------------------------------------------

InjectionRiskLevel = Literal["none", "low", "medium", "high"]


class InjectionVerdict(BaseModel):
    """Result of an injection detection check."""

    model_config = ConfigDict(frozen=True)

    detected: bool = Field(description="是否检测到注入")
    risk_level: InjectionRiskLevel = Field(description="风险等级")
    matched_patterns: list[str] = Field(description="匹配到的注入模式")
    description: str = Field(description="检测结果描述")
