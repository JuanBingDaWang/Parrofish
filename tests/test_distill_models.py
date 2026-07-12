"""PersonaSpec invariants and deterministic expression statistics tests."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from writing_factory.distill.expression import ExpressionAnalyzer
from writing_factory.distill.models import (
    ExpressionDNA,
    MentalModel,
    PersonaEvidence,
    PersonaSpec,
    SourceInfo,
    StyleTags,
    TripleValidation,
)


def _evidence(identifier: str, domain: str) -> PersonaEvidence:
    return PersonaEvidence(
        evidence_id=identifier,
        chunk_id=f"chunk_{identifier}",
        doc_id=f"doc_{domain}",
        domain=domain,
        summary="source-backed summary",
        confidence="high",
    )


def _mental_model(name: str) -> MentalModel:
    return MentalModel(
        name=name,
        description="a runnable lens",
        cross_domain_evidence=[
            _evidence(f"{name}_1", "出版产业"),
            _evidence(f"{name}_2", "公共文化"),
        ],
        applicability="new publishing questions",
        limits="does not replace current evidence",
        validation=TripleValidation(
            cross_domain=True,
            generative=True,
            exclusive=True,
            generative_rationale="predicts a new stance",
            exclusivity_rationale="not a generic principle",
        ),
    )


def test_mental_model_requires_two_domains_and_all_validations() -> None:
    with pytest.raises(ValidationError, match="at least two domains"):
        _mental_model("model").model_copy(
            update={
                "cross_domain_evidence": [
                    _evidence("one", "出版"),
                    _evidence("two", "出版"),
                ]
            }
        ).model_validate(
            _mental_model("model").model_dump()
            | {
                "cross_domain_evidence": [
                    _evidence("one", "出版").model_dump(),
                    _evidence("two", "出版").model_dump(),
                ]
            }
        )


def test_persona_requires_three_unique_models() -> None:
    statistics = ExpressionAnalyzer().analyze(["我认为一定要研究问题。为什么？然而也许会变化。"])
    with pytest.raises(ValidationError, match="at least 3"):
        PersonaSpec(
            id="persona",
            name="test",
            mode="person",
            mental_models=[_mental_model("one"), _mental_model("two")],
            expression_dna=ExpressionDNA(
                sentence_fingerprint=statistics.fingerprint,
                style_tags=StyleTags(),
            ),
            evidence_registry=[_evidence("registry", "出版")],
            source_info=[
                SourceInfo(
                    doc_id="doc",
                    title="title",
                    filename="source.txt",
                    chunk_count=1,
                )
            ],
            research_date=date.today(),
            declared_limits=["one", "two", "three"],
        )


def test_expression_fingerprint_is_local_and_reproducible() -> None:
    analyzer = ExpressionAnalyzer()
    texts = [
        "我认为一定要研究现实问题。为什么？然而，理论也许会变化。",
        "我们把出版系统比作网络，但是不能忽略历史。",
    ]

    first = analyzer.analyze(texts)
    second = analyzer.analyze(texts)

    assert first == second
    assert first.fingerprint.character_count > 0
    assert first.fingerprint.question_ratio > 0
    assert first.fingerprint.analogy_per_1000_chars > 0
    assert 0 < first.fingerprint.certainty_ratio < 1
