"""Hard citation traceability metric tests."""

from __future__ import annotations

from writing_factory.eval.traceability import evaluate_citation_traceability
from writing_factory.generate.models import (
    Claim,
    EvidenceItem,
    EvidencePack,
    SectionDraft,
    VerifiedClaim,
    VerifiedDraft,
)


def test_traceability_passes_only_supported_fact_with_valid_key() -> None:
    claim = Claim(
        claim_id="c1",
        text="有依据的事实",
        claim_type="fact",
        source_keys=["S1"],
        paragraph_index=0,
    )
    draft = SectionDraft(
        section_id="1",
        heading="标题",
        paragraphs=["有依据的事实。[S1]"],
        claims=[claim],
        evidence_pack=EvidencePack(
            section_id="1",
            items=[
                EvidenceItem(
                    source_key="S1",
                    chunk_id="chunk",
                    doc_id="doc",
                    verbatim_excerpt="有依据的事实",
                )
            ],
        ),
    )
    verified = VerifiedDraft(
        section_id="1",
        verified_claims=[
            VerifiedClaim(
                claim=claim,
                verdict="supported",
                verifier_rationale="原文支持",
                matched_chunk_text="有依据的事实",
            )
        ],
        supported_count=1,
    )
    result = evaluate_citation_traceability(
        {
            "sections": [
                {
                    "draft_json": draft.model_dump_json(),
                    "verified_draft_json": verified.model_dump_json(),
                }
            ]
        }
    )
    assert result.passed
    assert result.valid_citation_ratio == 1.0
    assert result.hallucination_rate == 0.0
