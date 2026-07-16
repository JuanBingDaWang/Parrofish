"""Recoverable, independently judged Persona fidelity self-checks."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from writing_factory.distill.extraction import StructuredDistillationError
from writing_factory.distill.fidelity_models import (
    FIDELITY_PIPELINE_VERSION,
    FidelityAnswers,
    FidelityResult,
    FidelityStage,
    FidelityStageProgress,
    FidelityStageState,
    FidelitySuite,
)
from writing_factory.distill.models import PersonaEvidence, PersonaSpec
from writing_factory.distill.runtime import build_runtime_persona
from writing_factory.llm import SiliconFlowClient
from writing_factory.llm.common import IncompleteStreamError
from writing_factory.llm.models import ChatResult
from writing_factory.store.persona_repository import PersonaRepository

StageModel = TypeVar("StageModel", bound=BaseModel)
ProgressCallback = Callable[[FidelityStageProgress], None]
CancellationCheck = Callable[[], None]


class PersonaFidelityEvaluator:
    """Use separate API calls so the Persona never judges its own answers."""

    def __init__(self, siliconflow: SiliconFlowClient) -> None:
        self.siliconflow = siliconflow

    def evaluate(self, persona: PersonaSpec, markdown: str) -> FidelityResult:
        """Run all three roles directly for callers that do not need persistence."""

        _ = markdown  # Kept for compatibility with the original public evaluator contract.
        suite = self.design_suite(persona)
        answers = self.answer_suite(persona, suite)
        return self.judge(persona, suite, answers)

    def design_input_hash(self, persona: PersonaSpec) -> str:
        """Return a stable hash for the exact test-design projection."""

        return self._payload_hash("design", self._design_payload(persona))

    def answer_input_hash(self, persona: PersonaSpec, suite: FidelitySuite) -> str:
        """Return a stable hash for the exact blind-answer input."""

        return self._payload_hash("answer", self._answer_payload(persona, suite))

    def judge_input_hash(
        self,
        persona: PersonaSpec,
        suite: FidelitySuite,
        answers: FidelityAnswers,
    ) -> str:
        """Return a stable hash for the exact neutral-judge input."""

        return self._payload_hash("judge", self._judge_payload(persona, suite, answers))

    def design_suite(self, persona: PersonaSpec) -> FidelitySuite:
        """Design five concise cases from representative evidence only."""

        payload = self._design_payload(persona)
        known = self._representative_evidence_ids(persona)

        def validate(result: ChatResult) -> None:
            self._parse_suite(result.content, known)

        result = self.siliconflow.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是独立、中立的作者档案测试题设计者。严格依据输入中的心智模型和"
                        "代表性证据，恰好生成3道有证据支持的已知立场题、1道确实超出档案"
                        "范围的边界题、1道不透露作者身份的文风盲测题。问题必须简短、使用"
                        "简体中文。expected_evidence_ids只能逐字复制输入中出现的evidence_id。"
                        "严格按所给JSON Schema直接返回对象，不要解释，不要添加Markdown。"
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
            result_validator=validate,
            step_id="distill.fidelity_design",
        )
        return self._parse_suite(result.content, known)

    def answer_suite(self, persona: PersonaSpec, suite: FidelitySuite) -> FidelityAnswers:
        """Answer a fixed suite using only the audit Persona profile."""

        payload = self._answer_payload(persona, suite)
        expected = {case.case_id for case in suite.cases}

        def validate(result: ChatResult) -> None:
            self._parse_answers(result.content, expected)

        result = self.siliconflow.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你只使用输入中的作者档案回答测试题，不评价自己的回答。遇到边界题，"
                        "必须明确区分档案中的已知立场和基于框架的推断。全部使用简体中文。"
                        "严格按所给JSON Schema直接返回对象，不要解释，不要添加Markdown。"
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
            result_validator=validate,
            step_id="distill.fidelity_answer",
        )
        return self._parse_answers(result.content, expected)

    def judge(
        self,
        persona: PersonaSpec,
        suite: FidelitySuite,
        answers: FidelityAnswers,
    ) -> FidelityResult:
        """Score the blind answers under a strict top-level business Schema."""

        payload = self._judge_payload(persona, suite, answers)

        def validate(result: ChatResult) -> None:
            self._parse_result(result.content)

        try:
            result = self.siliconflow.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是独立中性评判者，没有参与测试题设计或档案回答。只按输入中的"
                            "代表性证据和评分量表评分；对无依据的确定表达、通用AI腔和越界回答"
                            "扣分。返回对象顶层必须直接包含stance_consistency、"
                            "style_distinctiveness、edge_honesty、source_transparency、"
                            "structural_completeness和rationale，禁止套在results或其他字段中。"
                            "严格按所给JSON Schema返回JSON，不要解释，不要添加Markdown。"
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
                stream=True,
                result_validator=validate,
                step_id="distill.fidelity_judge",
            )
        except StructuredDistillationError as exc:
            raise StructuredDistillationError(
                "档案中性评判在定向重试后仍未返回有效评分结构"
            ) from exc
        except IncompleteStreamError as exc:
            if self._caused_by_schema_failure(exc):
                raise StructuredDistillationError(
                    "档案中性评判在定向重试后仍未返回有效评分结构"
                ) from exc
            raise
        return self._parse_result(result.content)

    def _design_payload(self, persona: PersonaSpec) -> dict[str, object]:
        models = []
        for model in persona.mental_models:
            evidence = self._representative_evidence(model.cross_domain_evidence)
            models.append(
                {
                    "name": model.name,
                    "description": model.description,
                    "applicability": model.applicability,
                    "limits": model.limits,
                    "representative_evidence": [item.model_dump(mode="json") for item in evidence],
                }
            )
        return {
            "profile_name": persona.name,
            "mode": persona.mode,
            "mental_models": models,
            "schema": FidelitySuite.model_json_schema(),
        }

    def _answer_payload(
        self,
        persona: PersonaSpec,
        suite: FidelitySuite,
    ) -> dict[str, object]:
        return {
            "profile": build_runtime_persona(persona).model_dump(mode="json"),
            "representative_evidence": self._representative_evidence_by_model(persona),
            "cases": [case.model_dump(mode="json") for case in suite.cases],
            "schema": FidelityAnswers.model_json_schema(),
        }

    def _judge_payload(
        self,
        persona: PersonaSpec,
        suite: FidelitySuite,
        answers: FidelityAnswers,
    ) -> dict[str, object]:
        return {
            "rubric": {
                "stance_consistency": 30,
                "style_distinctiveness": 20,
                "edge_honesty": 20,
                "source_transparency": 15,
                "structural_completeness": 15,
            },
            "profile": build_runtime_persona(persona).model_dump(mode="json"),
            "representative_evidence": self._representative_evidence_by_model(persona),
            "suite": suite.model_dump(mode="json"),
            "answers": answers.model_dump(mode="json"),
            "schema": FidelityResult.model_json_schema(),
        }

    def _representative_evidence_ids(self, persona: PersonaSpec) -> set[str]:
        return {
            item.evidence_id
            for model in persona.mental_models
            for item in self._representative_evidence(model.cross_domain_evidence)
        }

    def _representative_evidence_by_model(
        self,
        persona: PersonaSpec,
    ) -> dict[str, list[dict[str, object]]]:
        return {
            model.name: [
                item.model_dump(mode="json")
                for item in self._representative_evidence(model.cross_domain_evidence)
            ]
            for model in persona.mental_models
        }

    @staticmethod
    def _representative_evidence(
        evidence: list[PersonaEvidence],
        *,
        limit: int = 2,
    ) -> list[PersonaEvidence]:
        """Prefer high-confidence anchors from different documents and domains."""

        if len(evidence) <= limit:
            return list(evidence)
        confidence_rank = {"high": 0, "medium": 1, "low": 2}
        ranked = sorted(
            enumerate(evidence),
            key=lambda pair: (confidence_rank.get(pair[1].confidence, 3), pair[0]),
        )
        selected = [ranked[0][1]]
        remaining = [item for _, item in ranked[1:]]
        while remaining and len(selected) < limit:
            documents = {item.doc_id for item in selected}
            domains = {item.domain for item in selected}
            best = max(
                remaining,
                key=lambda item: (
                    item.doc_id not in documents,
                    item.domain not in domains,
                    -confidence_rank.get(item.confidence, 3),
                ),
            )
            selected.append(best)
            remaining.remove(best)
        return selected

    @staticmethod
    def _parse_suite(content: str, known: set[str]) -> FidelitySuite:
        try:
            suite = FidelitySuite.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise StructuredDistillationError("档案测试题不符合 FidelitySuite Schema") from exc
        for case in suite.cases:
            if set(case.expected_evidence_ids) - known:
                raise StructuredDistillationError("档案测试题引用了未知代表性证据")
            if case.case_type == "known" and not case.expected_evidence_ids:
                raise StructuredDistillationError("已知立场题必须引用代表性证据")
        return suite

    @staticmethod
    def _parse_answers(content: str, expected: set[str]) -> FidelityAnswers:
        try:
            answers = FidelityAnswers.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise StructuredDistillationError("档案盲测回答不符合 FidelityAnswers Schema") from exc
        received = {answer.case_id for answer in answers.answers}
        if expected != received:
            raise StructuredDistillationError("档案盲测回答与测试题 case_id 不一致")
        return answers

    @staticmethod
    def _parse_result(content: str) -> FidelityResult:
        try:
            return FidelityResult.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise StructuredDistillationError("档案中性评判不符合 FidelityResult Schema") from exc

    @staticmethod
    def _payload_hash(stage: FidelityStage, payload: dict[str, object]) -> str:
        canonical = json.dumps(
            {
                "pipeline_version": FIDELITY_PIPELINE_VERSION,
                "stage": stage,
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _caused_by_schema_failure(error: BaseException) -> bool:
        current: BaseException | None = error
        while current is not None:
            if isinstance(current, StructuredDistillationError):
                return True
            current = current.__cause__ or current.__context__
        return False


class FidelityService:
    """Resume, independently evaluate, and publish one ready PersonaSpec."""

    def __init__(self, repository: PersonaRepository, evaluator: PersonaFidelityEvaluator) -> None:
        self.repository = repository
        self.evaluator = evaluator

    def evaluate(
        self,
        persona_id: str,
        *,
        progress: ProgressCallback | None = None,
        check_cancelled: CancellationCheck | None = None,
    ) -> FidelityResult:
        """Resume at the first missing validated stage, then publish the final score."""

        loaded = self.repository.load_ready(persona_id)
        if loaded is None:
            raise ValueError("作者档案尚未完成，不能运行自检")
        persona, _markdown = loaded

        suite = self._load_or_run(
            persona_id=persona_id,
            stage="design",
            input_hash=self.evaluator.design_input_hash(persona),
            model=FidelitySuite,
            runner=lambda: self.evaluator.design_suite(persona),
            progress=progress,
            check_cancelled=check_cancelled,
        )
        answers = self._load_or_run(
            persona_id=persona_id,
            stage="answer",
            input_hash=self.evaluator.answer_input_hash(persona, suite),
            model=FidelityAnswers,
            runner=lambda: self.evaluator.answer_suite(persona, suite),
            progress=progress,
            check_cancelled=check_cancelled,
        )
        result = self._load_or_run(
            persona_id=persona_id,
            stage="judge",
            input_hash=self.evaluator.judge_input_hash(persona, suite, answers),
            model=FidelityResult,
            runner=lambda: self.evaluator.judge(persona, suite, answers),
            progress=progress,
            check_cancelled=check_cancelled,
        )
        if check_cancelled is not None:
            check_cancelled()
        self.repository.complete_fidelity_evaluation(
            persona_id=persona_id,
            score=result.total,
            result_json=result.model_dump_json(),
        )
        return result

    def _load_or_run(
        self,
        *,
        persona_id: str,
        stage: FidelityStage,
        input_hash: str,
        model: type[StageModel],
        runner: Callable[[], StageModel],
        progress: ProgressCallback | None,
        check_cancelled: CancellationCheck | None,
    ) -> StageModel:
        if check_cancelled is not None:
            check_cancelled()
        checkpoint = self.repository.load_fidelity_stage(
            persona_id=persona_id,
            stage=stage,
            input_hash=input_hash,
            model=model,
        )
        if checkpoint is not None:
            result, duration_ms = checkpoint
            self._report(progress, stage, "restored", duration_ms)
            return result

        self._report(progress, stage, "started", 0)
        started = time.perf_counter()
        try:
            result = runner()
            duration_ms = max(0, round((time.perf_counter() - started) * 1000))
            self.repository.save_fidelity_stage(
                persona_id=persona_id,
                stage=stage,
                input_hash=input_hash,
                result=result,
                duration_ms=duration_ms,
            )
        except Exception:
            duration_ms = max(0, round((time.perf_counter() - started) * 1000))
            self._report(progress, stage, "failed", duration_ms)
            raise
        self._report(progress, stage, "completed", duration_ms)
        return result

    @staticmethod
    def _report(
        callback: ProgressCallback | None,
        stage: FidelityStage,
        state: FidelityStageState,
        duration_ms: int,
    ) -> None:
        if callback is not None:
            callback(
                FidelityStageProgress(
                    stage=stage,
                    state=state,
                    duration_ms=max(0, duration_ms),
                )
            )
