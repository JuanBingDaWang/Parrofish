"""Deterministic Chinese/English expression-DNA corpus statistics."""

from __future__ import annotations

import re
from dataclasses import dataclass

from writing_factory.distill.models import SentenceFingerprint

_SENTENCE_SPLIT = re.compile(r"[。！？!?；;]+|\n+")
_ANALOGY = re.compile(
    r"像|如同|犹如|仿佛|好比|就像|比作|比喻为|例如|比如|\blike\b|\bas if\b",
    re.I,
)
_FIRST_PERSON = re.compile(r"我们|我|\bI\b|\bwe\b", re.I)
_CERTAINTY = re.compile(
    r"一定|显然|必然|毫无疑问|必须|肯定|无疑|\bclearly\b|\bobviously\b|\bmust\b|\bcertainly\b",
    re.I,
)
_CAUTION = re.compile(
    r"可能|也许|或许|似乎|大概|不确定|\bperhaps\b|\bmaybe\b|\bmight\b|\buncertain\b",
    re.I,
)
_TRANSITION = re.compile(
    r"但是|然而|不过|可是|相反|另一方面|\bbut\b|\bhowever\b|\byet\b",
    re.I,
)
_TOKEN = re.compile(r"[\u4e00-\u9fff]{2,6}|[A-Za-z][A-Za-z'-]{2,}")


@dataclass(frozen=True, slots=True)
class ExpressionStatistics:
    """Numeric fingerprint plus repeated lexical candidates for the reducer."""

    fingerprint: SentenceFingerprint
    frequent_phrases: tuple[str, ...]


class ExpressionAnalyzer:
    """Measure Nüwa's sentence fingerprint without an LLM call."""

    def analyze(self, texts: list[str]) -> ExpressionStatistics:
        """Analyze non-overlapping canonical source texts."""

        corpus = "\n\n".join(text.strip() for text in texts if text.strip())
        character_count = len(re.sub(r"\s+", "", corpus))
        sentences = [item.strip() for item in _SENTENCE_SPLIT.split(corpus) if item.strip()]
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", corpus) if item.strip()]
        questions = len(re.findall(r"[？?]", corpus))
        certainty = len(_CERTAINTY.findall(corpus))
        caution = len(_CAUTION.findall(corpus))
        marker_total = certainty + caution
        per_1000 = 1000 / character_count if character_count else 0.0
        fingerprint = SentenceFingerprint(
            character_count=character_count,
            sentence_count=len(sentences),
            paragraph_count=len(paragraphs),
            average_sentence_length=(
                sum(len(re.sub(r"\s+", "", sentence)) for sentence in sentences) / len(sentences)
                if sentences
                else 0.0
            ),
            question_ratio=questions / len(sentences) if sentences else 0.0,
            analogy_per_1000_chars=len(_ANALOGY.findall(corpus)) * per_1000,
            first_person_per_1000_chars=len(_FIRST_PERSON.findall(corpus)) * per_1000,
            certainty_ratio=certainty / marker_total if marker_total else 0.5,
            transition_per_1000_chars=len(_TRANSITION.findall(corpus)) * per_1000,
        )
        return ExpressionStatistics(
            fingerprint=fingerprint,
            frequent_phrases=self._frequent_phrases(corpus),
        )

    @staticmethod
    def _frequent_phrases(corpus: str, *, limit: int = 20) -> tuple[str, ...]:
        counts: dict[str, int] = {}
        for token in _TOKEN.findall(corpus):
            normalized = token.casefold()
            counts[normalized] = counts.get(normalized, 0) + 1
        repeated = [
            (token, count)
            for token, count in counts.items()
            if count >= 3 and token not in {"我们", "一个", "the", "and", "that", "this"}
        ]
        repeated.sort(key=lambda item: (-item[1], item[0]))
        return tuple(token for token, _count in repeated[:limit])
