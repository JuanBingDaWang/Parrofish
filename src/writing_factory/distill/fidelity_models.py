"""Typed contracts for independent Persona fidelity design and judging."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FidelityResult(BaseModel):
    """Independent evaluator output following Nüwa's 100-point scorecard."""

    model_config = ConfigDict(frozen=True)

    stance_consistency: int = Field(ge=0, le=30)
    style_distinctiveness: int = Field(ge=0, le=20)
    edge_honesty: int = Field(ge=0, le=20)
    source_transparency: int = Field(ge=0, le=15)
    structural_completeness: int = Field(ge=0, le=15)
    rationale: dict[str, Any] = Field(default_factory=dict)

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

    model_config = ConfigDict(frozen=True)

    case_id: str
    case_type: Literal["known", "edge", "style"]
    question: str
    expected_evidence_ids: list[str] = Field(default_factory=list)


class FidelitySuite(BaseModel):
    """Nüwa's fixed three-known, one-edge, one-style test design."""

    model_config = ConfigDict(frozen=True)

    cases: list[FidelityCase] = Field(min_length=5, max_length=5)

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

    model_config = ConfigDict(frozen=True)

    case_id: str
    answer: str


class FidelityAnswers(BaseModel):
    """Answer-agent output for every case in a fidelity suite."""

    model_config = ConfigDict(frozen=True)

    answers: list[FidelityAnswer] = Field(min_length=5, max_length=5)
