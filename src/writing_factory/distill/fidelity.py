"""Independent Nüwa fidelity design, answer, and neutral judging calls."""

from __future__ import annotations

import json

from pydantic import ValidationError

from writing_factory.distill.extraction import StructuredDistillationError
from writing_factory.distill.fidelity_models import (
    FidelityAnswers,
    FidelityResult,
    FidelitySuite,
)
from writing_factory.distill.models import PersonaSpec
from writing_factory.llm import SiliconFlowClient
from writing_factory.store.persona_repository import PersonaRepository


class PersonaFidelityEvaluator:
    """Use separate API calls so the Persona never judges its own answers."""

    def __init__(self, siliconflow: SiliconFlowClient) -> None:
        self.siliconflow = siliconflow

    def evaluate(self, persona: PersonaSpec, markdown: str) -> FidelityResult:
        """Design cases, answer under Persona, then judge neutrally."""

        suite = self._design_suite(persona)
        answers = self._answer_suite(markdown, suite)
        return self._judge(persona, suite, answers)

    def _design_suite(self, persona: PersonaSpec) -> FidelitySuite:
        evidence = [item.model_dump(mode="json") for item in persona.evidence_registry]
        payload = {
            "profile_name": persona.name,
            "mode": persona.mode,
            "mental_models": [
                {
                    "name": model.name,
                    "description": model.description,
                    "evidence_ids": [item.evidence_id for item in model.cross_domain_evidence],
                }
                for model in persona.mental_models
            ],
            "evidence_registry": evidence,
            "schema": FidelitySuite.model_json_schema(),
        }
        result = self.siliconflow.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a neutral test designer. Create exactly three "
                        "source-backed known position questions, one genuinely out-of-scope "
                        "edge question, and one blind style prompt. Use only supplied "
                        "evidence IDs. Return JSON only."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            thinking=True,
            reasoning_effort="high",
            temperature=0.0,
            max_tokens=8192,
            seed=23,
            response_format="json_object",
            use_cache=True,
            request_attempts=1,
            stream=True,
        )
        try:
            suite = FidelitySuite.model_validate(json.loads(result.content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise StructuredDistillationError("Invalid fidelity suite JSON") from exc
        known = {item.evidence_id for item in persona.evidence_registry}
        for case in suite.cases:
            if set(case.expected_evidence_ids) - known:
                raise StructuredDistillationError("Fidelity suite cited unknown evidence")
            if case.case_type == "known" and not case.expected_evidence_ids:
                raise StructuredDistillationError("Known fidelity cases require evidence")
        return suite

    def _answer_suite(self, markdown: str, suite: FidelitySuite) -> FidelityAnswers:
        payload = {
            "persona_markdown": markdown,
            "cases": [case.model_dump(mode="json") for case in suite.cases],
            "schema": FidelityAnswers.model_json_schema(),
        }
        result = self.siliconflow.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer using only the supplied Persona profile. Do not judge your "
                        "answers. For edge cases, explicitly distinguish inference from "
                        "known positions. Return JSON only."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            thinking=False,
            temperature=0.2,
            max_tokens=8192,
            seed=29,
            response_format="json_object",
            use_cache=True,
            request_attempts=1,
            stream=True,
        )
        try:
            answers = FidelityAnswers.model_validate(json.loads(result.content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise StructuredDistillationError("Invalid fidelity answer JSON") from exc
        expected = {case.case_id for case in suite.cases}
        received = {answer.case_id for answer in answers.answers}
        if expected != received:
            raise StructuredDistillationError("Fidelity answers do not match suite case IDs")
        return answers

    def _judge(
        self,
        persona: PersonaSpec,
        suite: FidelitySuite,
        answers: FidelityAnswers,
    ) -> FidelityResult:
        payload = {
            "rubric": {
                "stance_consistency": 30,
                "style_distinctiveness": 20,
                "edge_honesty": 20,
                "source_transparency": 15,
                "structural_completeness": 15,
            },
            "profile": persona.model_dump(mode="json"),
            "suite": suite.model_dump(mode="json"),
            "answers": answers.model_dump(mode="json"),
            "schema": FidelityResult.model_json_schema(),
        }
        result = self.siliconflow.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an independent neutral evaluator. You did not produce the "
                        "answers. Score only against supplied evidence and rubric. Penalize "
                        "unsupported certainty and generic AI style. Return JSON only."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            thinking=True,
            reasoning_effort="high",
            temperature=0.0,
            max_tokens=8192,
            seed=31,
            response_format="json_object",
            use_cache=True,
            request_attempts=1,
            stream=True,
        )
        try:
            return FidelityResult.model_validate(json.loads(result.content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise StructuredDistillationError("Invalid fidelity judge JSON") from exc


class FidelityService:
    """Load, independently evaluate, and persist one ready PersonaSpec."""

    def __init__(self, repository: PersonaRepository, evaluator: PersonaFidelityEvaluator) -> None:
        self.repository = repository
        self.evaluator = evaluator

    def evaluate(self, persona_id: str) -> FidelityResult:
        """Run one explicitly requested paid fidelity evaluation."""

        loaded = self.repository.load_ready(persona_id)
        if loaded is None:
            raise ValueError("PersonaSpec is not ready")
        result = self.evaluator.evaluate(*loaded)
        self.repository.save_evaluation(
            persona_id=persona_id,
            evaluation_type="nuwa_fidelity",
            score=result.total,
            result_json=result.model_dump_json(),
        )
        return result
