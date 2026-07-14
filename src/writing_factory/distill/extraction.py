"""Structured SiliconFlow map extraction with source-ID validation and repair."""

from __future__ import annotations

import json

from pydantic import ValidationError

from writing_factory.distill.language import (
    DEFAULT_OUTPUT_LANGUAGE,
    OutputLanguage,
    OutputLanguageError,
    validate_map_language,
)
from writing_factory.distill.models import MapResult, PersonaMode, SourceUnit
from writing_factory.distill.prompts import map_messages
from writing_factory.llm import SiliconFlowClient


class StructuredDistillationError(ValueError):
    """Raised when a model response cannot satisfy a source-backed contract."""


class PersonaMapExtractor:
    """Run independent, deterministic extraction for each bounded source unit."""

    def __init__(
        self,
        siliconflow: SiliconFlowClient,
        *,
        output_language: OutputLanguage = DEFAULT_OUTPUT_LANGUAGE,
        max_attempts: int = 2,
    ) -> None:
        self.siliconflow = siliconflow
        self.output_language = output_language
        self.max_attempts = max_attempts

    def extract(
        self,
        name: str,
        mode: PersonaMode,
        unit: SourceUnit,
        *,
        corpus_role: str = "target",
        domain: str = "",
    ) -> MapResult:
        """Extract and validate one map result, repairing at most once."""

        messages = map_messages(
            name,
            mode,
            unit,
            output_language=self.output_language,
            corpus_role=corpus_role,
            domain=domain,
        )
        last_error = "未知校验错误"
        for attempt in range(self.max_attempts):
            active_messages = messages
            if attempt:
                active_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "上一次 JSON 违反了契约。请返回修正后的完整对象，不要解释。"
                            f"校验错误：{last_error}"
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
                request_attempts=1,
                stream=True,
            )
            try:
                payload = json.loads(result.content)
                self._discard_unsubstantiated_tensions(payload)
                parsed = MapResult.model_validate(payload)
                self._validate_sources(parsed, unit)
                validate_map_language(parsed, self.output_language)
                return parsed
            except (
                json.JSONDecodeError,
                ValidationError,
                StructuredDistillationError,
                OutputLanguageError,
            ) as exc:
                last_error = str(exc)[:2000]
        raise StructuredDistillationError(
            f"Map 提取在 {self.max_attempts} 次尝试后失败：{last_error}"
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
        gaps = payload.setdefault("information_gaps", [])
        if isinstance(gaps, list):
            gaps.append(
                {
                    "dimension": "核心张力",
                    "description": "部分候选张力只有单一证据，不能纳入全局档案",
                    "reason": "当前单元缺少分别支持张力两侧的至少两条证据",
                    "resolvable_by_more_sources": True,
                    "confidence": "high",
                }
            )

    @staticmethod
    def _validate_sources(result: MapResult, unit: SourceUnit) -> None:
        if result.unit_id != unit.unit_id:
            raise StructuredDistillationError("Map 结果的 unit_id 与输入不一致")
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
            allowed_values = ", ".join(sorted(allowed))
            unknown_values = ", ".join(sorted(unknown))
            raise StructuredDistillationError(
                f"Map 结果引用了未知 chunk_id：{unknown_values}。"
                f"只能从以下合法值中逐字复制：{allowed_values}"
            )
