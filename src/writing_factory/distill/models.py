"""Typed contracts for map evidence, PersonaSpec reduction, and fidelity."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

UnitScore = Annotated[float, Field(ge=-1.0, le=1.0)]
Confidence = Literal["high", "medium", "low", "inferred"]
PersonaMode = Literal["person", "topic"]


class ExtractedEvidence(BaseModel):
    """Map-stage evidence anchored to one immutable source chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    domain: str
    summary: str
    confidence: Confidence = "medium"


class MapMentalCandidate(BaseModel):
    """A map-stage candidate not yet eligible as a mental model."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    evidence: list[ExtractedEvidence] = Field(min_length=1)
    generative_rationale: str
    exclusivity_rationale: str


class MapHeuristicCandidate(BaseModel):
    """A source-backed conditional decision rule."""

    model_config = ConfigDict(frozen=True)

    rule: str
    trigger: str
    example: str
    evidence: list[ExtractedEvidence] = Field(min_length=1)


class MapTensionCandidate(BaseModel):
    """Two source-backed positions that must not be silently reconciled."""

    model_config = ConfigDict(frozen=True)

    side_a: str
    side_b: str
    tension_type: Literal["temporal", "domain", "essential", "school"]
    evidence: list[ExtractedEvidence] = Field(min_length=2)


class MapResult(BaseModel):
    """Structured output from one independent source-unit extraction."""

    model_config = ConfigDict(frozen=True)

    unit_id: str
    mental_candidates: list[MapMentalCandidate] = Field(default_factory=list)
    heuristic_candidates: list[MapHeuristicCandidate] = Field(default_factory=list)
    tensions: list[MapTensionCandidate] = Field(default_factory=list)
    value_signals: list[str] = Field(default_factory=list)
    anti_pattern_signals: list[str] = Field(default_factory=list)
    style_observations: list[str] = Field(default_factory=list)
    insufficient_dimensions: list[str] = Field(default_factory=list)


class SourceSegment(BaseModel):
    """Immutable source data sent to one map call as data, never instructions."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    document_title: str
    filename: str
    text: str = Field(repr=False)
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None


class SourceUnit(BaseModel):
    """Bounded group of source segments processed independently by map."""

    model_config = ConfigDict(frozen=True)

    unit_id: str
    segments: list[SourceSegment] = Field(min_length=1)


class SentenceFingerprint(BaseModel):
    """Deterministic corpus statistics; the LLM never invents these values."""

    model_config = ConfigDict(frozen=True)

    character_count: int = Field(ge=0)
    sentence_count: int = Field(ge=0)
    paragraph_count: int = Field(ge=0)
    average_sentence_length: float = Field(ge=0)
    question_ratio: float = Field(ge=0, le=1)
    analogy_per_1000_chars: float = Field(ge=0)
    first_person_per_1000_chars: float = Field(ge=0)
    certainty_ratio: float = Field(ge=0, le=1)
    transition_per_1000_chars: float = Field(ge=0)


class StyleTags(BaseModel):
    """Seven Nüwa style axes, from -1 (left label) to +1 (right label)."""

    model_config = ConfigDict(frozen=True)

    formal_to_colloquial: UnitScore = 0.0
    abstract_to_concrete: UnitScore = 0.0
    cautious_to_assertive: UnitScore = 0.0
    academic_to_popular: UnitScore = 0.0
    long_to_short: UnitScore = 0.0
    setup_to_conclusion_first: UnitScore = 0.0
    data_to_narrative: UnitScore = 0.0


class PersonaEvidence(BaseModel):
    """Stable evidence copied from map results into the final PersonaSpec."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    chunk_id: str
    doc_id: str
    domain: str
    summary: str
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None
    confidence: Confidence


class TripleValidation(BaseModel):
    """Recorded decision for Nüwa's three mental-model tests."""

    model_config = ConfigDict(frozen=True)

    cross_domain: bool
    generative: bool
    exclusive: bool
    generative_rationale: str
    exclusivity_rationale: str

    @property
    def passed(self) -> bool:
        """A mental model must pass all three checks."""

        return self.cross_domain and self.generative and self.exclusive


class MentalModel(BaseModel):
    """One runnable lens with evidence, applicability, and failure conditions."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    cross_domain_evidence: list[PersonaEvidence] = Field(min_length=2)
    applicability: str
    limits: str
    validation: TripleValidation

    @model_validator(mode="after")
    def require_distinct_domains(self) -> MentalModel:
        domains = {item.domain.strip().casefold() for item in self.cross_domain_evidence}
        if len(domains) < 2:
            raise ValueError("Mental model evidence must cover at least two domains")
        if not self.validation.passed:
            raise ValueError("Mental model must pass all three Nüwa validations")
        return self


class DecisionHeuristic(BaseModel):
    """A reusable if-then rule downgraded from or adjacent to mental models."""

    model_config = ConfigDict(frozen=True)

    rule: str
    trigger: str
    example: str
    evidence: list[PersonaEvidence] = Field(min_length=1)


class LexicalMarker(BaseModel):
    """A tic or inferred taboo with explicit confidence and evidence."""

    model_config = ConfigDict(frozen=True)

    text: str
    confidence: Confidence
    evidence_ids: list[str] = Field(default_factory=list)


class ExpressionDNA(BaseModel):
    """Quantitative fingerprint plus reducer-classified style constraints."""

    model_config = ConfigDict(frozen=True)

    sentence_fingerprint: SentenceFingerprint
    style_tags: StyleTags
    taboo_words: list[LexicalMarker] = Field(default_factory=list)
    tics: list[LexicalMarker] = Field(default_factory=list)
    style_rules: list[str] = Field(default_factory=list)


class CoreTension(BaseModel):
    """A temporal, domain, essential, or school-level unresolved tension."""

    model_config = ConfigDict(frozen=True)

    side_a: str
    side_b: str
    tension_type: Literal["temporal", "domain", "essential", "school"]
    evidence: list[PersonaEvidence] = Field(min_length=2)
    interpretation: str


class SchoolPosition(BaseModel):
    """One side of a disagreement in topic mode."""

    model_config = ConfigDict(frozen=True)

    label: str
    position: str
    evidence_ids: list[str] = Field(min_length=1)


class SchoolDivergence(BaseModel):
    """A topic-mode disagreement kept visible rather than averaged away."""

    model_config = ConfigDict(frozen=True)

    question: str
    positions: list[SchoolPosition] = Field(min_length=2)


class SourceInfo(BaseModel):
    """Document-level provenance included in the serialized profile."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    filename: str
    source_type: Literal["primary", "secondary", "unknown"] = "primary"
    chunk_count: int = Field(ge=1)


class PersonaSpec(BaseModel):
    """Authoritative serializable output consumed by later writing stages."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    mode: PersonaMode
    mental_models: list[MentalModel] = Field(min_length=3, max_length=7)
    decision_heuristics: list[DecisionHeuristic] = Field(default_factory=list)
    expression_dna: ExpressionDNA
    core_tensions: list[CoreTension] = Field(default_factory=list)
    school_divergences: list[SchoolDivergence] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    anti_patterns: list[str] = Field(default_factory=list)
    evidence_registry: list[PersonaEvidence] = Field(min_length=1)
    source_info: list[SourceInfo] = Field(min_length=1)
    research_date: date
    declared_limits: list[str] = Field(min_length=3)
    information_gaps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_mode_contract(self) -> PersonaSpec:
        if self.mode == "topic" and not self.school_divergences:
            raise ValueError("Topic mode must preserve at least one school divergence")
        identifiers = [model.name.strip().casefold() for model in self.mental_models]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Mental model names must be unique")
        return self


class ReduceMentalModel(BaseModel):
    """Reducer proposal referencing only registered map evidence identifiers."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    evidence_ids: list[str] = Field(min_length=2)
    applicability: str
    limits: str
    generative: bool
    exclusive: bool
    generative_rationale: str
    exclusivity_rationale: str


class ReduceHeuristic(BaseModel):
    """Reducer proposal for a decision heuristic."""

    model_config = ConfigDict(frozen=True)

    rule: str
    trigger: str
    example: str
    evidence_ids: list[str] = Field(min_length=1)


class ReduceTension(BaseModel):
    """Reducer proposal for an unresolved source tension."""

    model_config = ConfigDict(frozen=True)

    side_a: str
    side_b: str
    tension_type: Literal["temporal", "domain", "essential", "school"]
    evidence_ids: list[str] = Field(min_length=2)
    interpretation: str


class ReduceResult(BaseModel):
    """Validated JSON shape requested from the reduce LLM call."""

    model_config = ConfigDict(frozen=True)

    mental_models: list[ReduceMentalModel] = Field(min_length=3, max_length=7)
    decision_heuristics: list[ReduceHeuristic] = Field(default_factory=list)
    style_tags: StyleTags
    taboo_words: list[LexicalMarker] = Field(default_factory=list)
    tics: list[LexicalMarker] = Field(default_factory=list)
    style_rules: list[str] = Field(default_factory=list)
    core_tensions: list[ReduceTension] = Field(default_factory=list)
    school_divergences: list[SchoolDivergence] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    anti_patterns: list[str] = Field(default_factory=list)
    declared_limits: list[str] = Field(min_length=3)
    information_gaps: list[str] = Field(default_factory=list)


class DistillationOutcome(BaseModel):
    """Result returned to the UI with explicit reuse and run identifiers."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    persona: PersonaSpec
    markdown: str
    reused: bool = False
