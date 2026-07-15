"""RAGAS-style faithfulness evaluator for generation quality.

Methodology (following RAGAS faithfulness):
1. Decompose generated answer into atomic claims
2. Check each claim against the retrieved context chunks
3. Score = supported_claims / total_claims
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from writing_factory.eval.models import AtomicClaim, FaithfulnessResult
from writing_factory.eval.prompts import (
    CHECK_CLAIM_SYSTEM,
    CHECK_CLAIM_USER,
    DECOMPOSE_CLAIMS_SYSTEM,
    DECOMPOSE_CLAIMS_USER,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from writing_factory.llm.siliconflow import SiliconFlowClient


class FaithfulnessEvaluator:
    """Evaluator that measures factual faithfulness of generated text to context.

    Usage:
        evaluator = FaithfulnessEvaluator(siliconflow)
        result = evaluator.evaluate(
            question="...",
            answer="...",
            context=["chunk1 text", "chunk2 text", ...],
        )
        print(result.score)
    """

    def __init__(self, client: SiliconFlowClient) -> None:
        self._client = client

    def evaluate(
        self,
        question: str,
        answer: str,
        context: Sequence[str],
    ) -> FaithfulnessResult:
        """Run the full faithfulness evaluation pipeline.

        Args:
            question: The original question / writing task.
            answer: The generated answer / draft text.
            context: Retrieved context chunks used during generation.

        Returns:
            FaithfulnessResult with score, per-claim verdicts, and counts.
        """

        # Step 1 — decompose answer into atomic claims
        atomic_claims = self._decompose_claims(answer)

        # Step 2 — check each claim against context
        checked: list[AtomicClaim] = []
        joined_context = "\n---\n".join(context)

        for claim_text in atomic_claims:
            verdict = self._check_claim(claim_text, joined_context)
            checked.append(verdict)

        # Step 3 — aggregate
        supported = sum(1 for c in checked if c.verdict == "supported")
        unsupported = sum(1 for c in checked if c.verdict == "unsupported")
        total = len(checked)
        score = supported / total if total > 0 else 1.0

        return FaithfulnessResult(
            score=score,
            atomic_claims=checked,
            supported_count=supported,
            unsupported_count=unsupported,
        )

    # ── internal helpers ──

    def _decompose_claims(self, text: str) -> list[str]:
        """Decompose text into atomic claims via LLM.

        Falls back to sentence splitting on LLM failure.
        """

        try:
            result = self._client.chat(
                [
                    {"role": "system", "content": DECOMPOSE_CLAIMS_SYSTEM},
                    {"role": "user", "content": DECOMPOSE_CLAIMS_USER.format(text=text)},
                ],
                thinking=False,
                temperature=0.0,
                max_tokens=8192,
                seed=42,
                stream=True,
                step_id="evaluation.claim_decomposition",
            )
            raw = (result.content or "").strip()
            # Parse: each line is one claim (skip empty lines)
            claims = [line.strip() for line in raw.split("\n") if line.strip()]
            return claims if claims else self._fallback_decompose(text)
        except Exception:
            return self._fallback_decompose(text)

    def _fallback_decompose(self, text: str) -> list[str]:
        """Simple sentence-based fallback when LLM decomposition fails."""

        import re

        sentences = re.split(r"(?<=[。！？；])", text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _check_claim(
        self,
        claim: str,
        joined_context: str,
    ) -> AtomicClaim:
        """Check a single atomic claim against context."""

        if not joined_context.strip():
            return AtomicClaim(
                claim_text=claim,
                verdict="unsupported",
                evidence=None,
            )

        try:
            result = self._client.chat(
                [
                    {"role": "system", "content": CHECK_CLAIM_SYSTEM},
                    {
                        "role": "user",
                        "content": CHECK_CLAIM_USER.format(
                            claim=claim,
                            context=joined_context,
                        ),
                    },
                ],
                thinking=False,
                temperature=0.0,
                max_tokens=8192,
                seed=42,
                stream=True,
                step_id="evaluation.claim_support",
            )
            return self._parse_check_result(claim, result.content or "")
        except Exception:
            return AtomicClaim(claim_text=claim, verdict="unsupported", evidence=None)

    @staticmethod
    def _parse_check_result(claim_text: str, raw: str) -> AtomicClaim:
        """Parse LLM's verdict from the check response."""

        stripped = raw.strip()
        if stripped.startswith("[SUPPORTED]"):
            return AtomicClaim(claim_text=claim_text, verdict="supported", evidence=stripped)
        return AtomicClaim(claim_text=claim_text, verdict="unsupported", evidence=stripped)


def faithfulness(
    client: SiliconFlowClient,
    question: str,
    answer: str,
    context: Sequence[str],
) -> FaithfulnessResult:
    """Convenience function for one-shot faithfulness evaluation."""

    return FaithfulnessEvaluator(client).evaluate(question, answer, context)
