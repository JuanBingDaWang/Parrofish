"""Typed contracts for independent Persona fidelity design and judging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FIDELITY_PIPELINE_VERSION = 2
FIDELITY_PROGRESS_PREFIX = "__fidelity_stage__"
FidelityStage = Literal["design", "answer", "judge"]
FidelityStageState = Literal["started", "restored", "completed", "failed"]


@dataclass(frozen=True, slots=True)
class FidelityStageProgress:
    """One typed progress event emitted at a recoverable self-check boundary."""

    stage: FidelityStage
    state: FidelityStageState
    duration_ms: int = 0


def encode_fidelity_progress(event: FidelityStageProgress) -> str:
    """Encode a typed stage event through the existing Qt string signal."""

    return "|".join(
        (
            FIDELITY_PROGRESS_PREFIX,
            event.stage,
            event.state,
            str(max(0, event.duration_ms)),
        )
    )


def parse_fidelity_progress(message: str) -> FidelityStageProgress | None:
    """Decode a stage event without treating ordinary progress text as control data."""

    parts = message.split("|")
    if len(parts) != 4 or parts[0] != FIDELITY_PROGRESS_PREFIX:
        return None
    stage, state = parts[1], parts[2]
    if stage not in {"design", "answer", "judge"}:
        return None
    if state not in {"started", "restored", "completed", "failed"}:
        return None
    try:
        duration_ms = max(0, int(parts[3]))
    except ValueError:
        return None
    return FidelityStageProgress(
        stage=stage,  # type: ignore[arg-type]
        state=state,  # type: ignore[arg-type]
        duration_ms=duration_ms,
    )


class FidelityResult(BaseModel):
    """Independent evaluator output following Nüwa's 100-point scorecard."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stance_consistency: int = Field(ge=0, le=30, description="立场一致性得分，满分30")
    style_distinctiveness: int = Field(ge=0, le=20, description="风格辨识度得分，满分20")
    edge_honesty: int = Field(ge=0, le=20, description="边界诚实度得分，满分20")
    source_transparency: int = Field(ge=0, le=15, description="来源透明度得分，满分15")
    structural_completeness: int = Field(ge=0, le=15, description="结构完整性得分，满分15")
    rationale: dict[str, Any] = Field(default_factory=dict, description="各评分维度的中文理由")

    @property
    def total(self) -> int:
        """Return the documented 100-point aggregate."""

        return (
            self.stance_consistency
            + self.style_distinctiveness
            + self.edge_honesty
            + self.source_transparency
            + self.structural_completeness
        )


class FidelityCase(BaseModel):
    """One known-position, edge-honesty, or blind-style evaluation prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(description="本套题内唯一、简短稳定的题目标识")
    case_type: Literal["known", "edge", "style"] = Field(
        description="known为已知立场题，edge为知识边界题，style为盲测文风题"
    )
    question: str = Field(description="简体中文测试问题")
    expected_evidence_ids: list[str] = Field(
        default_factory=list,
        description="仅复制输入中代表性证据的 evidence_id；边界题和文风题可为空",
    )


class FidelitySuite(BaseModel):
    """Nüwa's fixed three-known, one-edge, one-style test design."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: list[FidelityCase] = Field(
        min_length=5,
        max_length=5,
        description="恰好5题：3道已知立场题、1道边界题、1道盲测文风题",
    )

    @model_validator(mode="after")
    def require_case_mix(self) -> FidelitySuite:
        counts = {
            kind: sum(case.case_type == kind for case in self.cases)
            for kind in ("known", "edge", "style")
        }
        if counts != {"known": 3, "edge": 1, "style": 1}:
            raise ValueError("Fidelity suite must contain 3 known, 1 edge, and 1 style case")
        return self


class FidelityAnswer(BaseModel):
    """One answer produced in a Persona-only call with no judge context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(description="逐字复制测试题的 case_id")
    answer: str = Field(description="仅依据作者档案作答的简体中文回答")


class FidelityAnswers(BaseModel):
    """Answer-agent output for every case in a fidelity suite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    answers: list[FidelityAnswer] = Field(
        min_length=5,
        max_length=5,
        description="与5道测试题一一对应的回答",
    )
