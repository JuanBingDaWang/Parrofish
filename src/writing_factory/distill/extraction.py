"""Structured SiliconFlow map extraction with source-ID validation and repair."""

from __future__ import annotations

import json

from pydantic import ValidationError

from writing_factory.distill.models import MapResult, PersonaMode, SourceUnit
from writing_factory.distill.prompts import map_messages
from writing_factory.llm import SiliconFlowClient


class StructuredDistillationError(ValueError):
    """Raised when a model response cannot satisfy a source-backed contract."""


class PersonaMapExtractor:
    """Run independent, deterministic extraction for each bounded source unit."""

    def __init__(self, siliconflow: SiliconFlowClient, *, max_attempts: int = 2) -> None:
        self.siliconflow = siliconflow
        self.max_attempts = max_attempts

    def extract(self, name: str, mode: PersonaMode, unit: SourceUnit) -> MapResult:
        """Extract and validate one map result, repairing at most once."""

        messages = map_messages(name, mode, unit)
        last_error = "unknown validation error"
        for attempt in range(self.max_attempts):
            active_messages = messages
            if attempt:
                active_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "The previous JSON violated the contract. "
                            "Return a corrected full object. "
                            f"Validation error: {last_error}"
                        ),
                    },
                ]
            result = self.siliconflow.chat(
                active_messages,
                thinking=True,
                reasoning_effort="high",
                temperature=0.0,
                max_tokens=8192,
                seed=11,
                response_format="json_object",
                use_cache=True,
                request_timeout_seconds=600.0,
                request_attempts=1,
                stream=True,
            )
            try:
                payload = json.loads(result.content)
                self._discard_unsubstantiated_tensions(payload)
                parsed = MapResult.model_validate(payload)
                self._validate_sources(parsed, unit)
                return parsed
            except (json.JSONDecodeError, ValidationError, StructuredDistillationError) as exc:
                last_error = str(exc)[:500]
        raise StructuredDistillationError(
            f"Map extraction failed after {self.max_attempts} attempts: {last_error}"
        )

    @staticmethod
    def _discard_unsubstantiated_tensions(payload: object) -> None:
        """Drop optional tensions that lack two anchors and record the evidence gap."""

        if not isinstance(payload, dict):
            return
        tensions = payload.get("tensions")
        if not isinstance(tensions, list):
            return
        supported = [
            item
            for item in tensions
            if isinstance(item, dict)
            and isinstance(item.get("evidence"), list)
            and len(item["evidence"]) >= 2
        ]
        if len(supported) == len(tensions):
            return
        payload["tensions"] = supported
        gaps = payload.setdefault("insufficient_dimensions", [])
        if isinstance(gaps, list):
            gaps.append("部分候选张力只有单一证据，未纳入档案")

    @staticmethod
    def _validate_sources(result: MapResult, unit: SourceUnit) -> None:
        if result.unit_id != unit.unit_id:
            raise StructuredDistillationError("Map result unit_id does not match input")
        allowed = {segment.chunk_id for segment in unit.segments}
        evidence_items = []
        for candidate in result.mental_candidates:
            evidence_items.extend(candidate.evidence)
        for candidate in result.heuristic_candidates:
            evidence_items.extend(candidate.evidence)
        for tension in result.tensions:
            evidence_items.extend(tension.evidence)
        unknown = {item.chunk_id for item in evidence_items} - allowed
        if unknown:
            raise StructuredDistillationError("Map result cited unknown chunk_id values")
