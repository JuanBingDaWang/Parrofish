"""Code-only citation traceability metrics derived from structured drafts."""

from __future__ import annotations

import json
from typing import Any

from writing_factory.eval.models import CitationTraceabilityResult
from writing_factory.generate.models import SectionDraft, VerifiedDraft


def evaluate_citation_traceability(state: dict[str, Any]) -> CitationTraceabilityResult:
    """Measure citation validity and support without asking an LLM to grade itself."""

    fact_count = 0
    valid_citation_count = 0
    supported_count = 0
    failed_count = 0

    for section in state.get("sections", []):
        draft_raw = section.get("draft_json")
        verified_raw = section.get("verified_draft_json")
        if not draft_raw or not verified_raw:
            continue
        draft = SectionDraft.model_validate_json(draft_raw)
        verified = VerifiedDraft.model_validate_json(verified_raw)
        valid_keys = {item.source_key for item in draft.evidence_pack.items}
        verdict_by_id = {item.claim.claim_id: item.verdict for item in verified.verified_claims}
        for claim in draft.claims:
            if claim.claim_type != "fact":
                continue
            fact_count += 1
            if claim.source_keys and set(claim.source_keys) <= valid_keys:
                valid_citation_count += 1
            if verdict_by_id.get(claim.claim_id) == "supported":
                supported_count += 1
            else:
                failed_count += 1

    denominator = fact_count or 1
    return CitationTraceabilityResult(
        fact_claim_count=fact_count,
        valid_citation_count=valid_citation_count,
        supported_fact_count=supported_count,
        valid_citation_ratio=valid_citation_count / denominator if fact_count else 1.0,
        verified_support_ratio=supported_count / denominator if fact_count else 1.0,
        hallucination_rate=failed_count / denominator if fact_count else 0.0,
        passed=(
            valid_citation_count == fact_count
            and supported_count == fact_count
            and failed_count == 0
        ),
    )


def evidence_context_from_state(state: dict[str, Any]) -> list[str]:
    """Return each exact evidence chunk once for faithfulness evaluation."""

    excerpts: list[str] = []
    seen: set[str] = set()
    for section in state.get("sections", []):
        raw = section.get("draft_json")
        if not raw:
            continue
        data = json.loads(raw)
        for item in data.get("evidence_pack", {}).get("items", []):
            chunk_id = str(item.get("chunk_id", ""))
            excerpt = str(item.get("verbatim_excerpt", ""))
            if excerpt and chunk_id not in seen:
                seen.add(chunk_id)
                excerpts.append(excerpt)
    return excerpts
