"""Independent Persona fidelity evaluation and persistence tests."""

from __future__ import annotations

import json

import pytest

from tests.test_distill_pipeline import FakeChatClient, _persona
from writing_factory.distill.extraction import StructuredDistillationError
from writing_factory.distill.fidelity import FidelityService, PersonaFidelityEvaluator
from writing_factory.distill.fidelity_models import FidelityAnswers, FidelityResult, FidelitySuite
from writing_factory.distill.models import PersonaEvidence
from writing_factory.distill.serialization import render_persona_markdown
from writing_factory.llm.models import ChatResult
from writing_factory.store import Database
from writing_factory.store.kb_repository import KnowledgeBaseRepository
from writing_factory.store.persona_repository import PersonaRepository
from writing_factory.ui.workers import TaskCancelled


def _suite() -> FidelitySuite:
    return FidelitySuite(
        cases=[
            {
                "case_id": f"known_{index}",
                "case_type": "known",
                "question": f"Known position {index}?",
                "expected_evidence_ids": ["ev_a"],
            }
            for index in range(3)
        ]
        + [
            {
                "case_id": "edge_1",
                "case_type": "edge",
                "question": "Unknown position?",
            },
            {
                "case_id": "style_1",
                "case_type": "style",
                "question": "Blind style prompt",
            },
        ]
    )


def _answers(suite: FidelitySuite) -> FidelityAnswers:
    return FidelityAnswers(
        answers=[{"case_id": case.case_id, "answer": "Answer"} for case in suite.cases]
    )


def _result() -> FidelityResult:
    return FidelityResult(
        stance_consistency=27,
        style_distinctiveness=17,
        edge_honesty=18,
        source_transparency=14,
        structural_completeness=13,
    )


def test_fidelity_uses_three_separate_roles_and_persists_score(tmp_path) -> None:
    persona = _persona("persona_fidelity")
    suite = _suite()
    client = FakeChatClient(
        [suite.model_dump_json(), _answers(suite).model_dump_json(), _result().model_dump_json()]
    )
    database = Database(tmp_path / "fidelity.db")
    database.initialize()
    kb_id = KnowledgeBaseRepository(database).ensure_default()
    repository = PersonaRepository(database)
    run = repository.begin_or_resume(
        name=persona.name,
        mode=persona.mode,
        kb_id=kb_id,
        source_hash="source",
        input_hash="input",
        source_doc_ids=["doc_a"],
        map_total=1,
    )
    persona = persona.model_copy(update={"id": run.persona_id})
    repository.save_ready(
        run_id=run.run_id,
        persona=persona,
        markdown=render_persona_markdown(persona),
    )

    result = FidelityService(
        repository,
        PersonaFidelityEvaluator(client),  # type: ignore[arg-type]
    ).evaluate(persona.id)

    assert result.total == 89
    assert len(client.calls) == 3
    system_prompts = [call["messages"][0]["content"] for call in client.calls]
    assert "测试题设计者" in system_prompts[0]
    assert "不评价自己的回答" in system_prompts[1]
    assert "独立中性评判者" in system_prompts[2]
    assert [call["seed"] for call in client.calls] == [23, 29, 31]
    design_payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert "evidence_registry" not in design_payload
    assert all(
        len(model["representative_evidence"]) <= 2 for model in design_payload["mental_models"]
    )
    judge_payload = json.loads(client.calls[2]["messages"][1]["content"])
    assert "evidence_registry" not in judge_payload["profile"]
    assert "representative_evidence" in judge_payload
    answer_payload = json.loads(client.calls[1]["messages"][1]["content"])
    assert "persona_markdown" not in answer_payload
    assert "evidence_registry" not in answer_payload["profile"]
    assert "representative_evidence" in answer_payload
    validator = client.calls[2]["result_validator"]
    assert callable(validator)
    with pytest.raises(StructuredDistillationError, match="FidelityResult Schema"):
        validator(ChatResult(content='{"results": []}', model="fixture"))
    assert repository.list_personas(kb_id)[0]["fidelity_score"] == 89
    with database.connection() as connection:
        saved = connection.execute(
            "SELECT result_json FROM persona_evaluations "
            "WHERE persona_id = ? AND evaluation_type = 'nuwa_fidelity'",
            (persona.id,),
        ).fetchone()
    assert json.loads(saved["result_json"])["edge_honesty"] == 18


def test_fidelity_rejects_answers_with_wrong_case_ids() -> None:
    suite = _suite()
    invalid_answers = _answers(suite).model_dump(mode="json")
    invalid_answers["answers"][0]["case_id"] = "invented"
    client = FakeChatClient(
        [suite.model_dump_json(), json.dumps(invalid_answers), _result().model_dump_json()]
    )

    with pytest.raises(StructuredDistillationError, match="case_id"):
        PersonaFidelityEvaluator(client).evaluate(  # type: ignore[arg-type]
            _persona("persona"), "profile"
        )

    assert len(client.calls) == 2


def test_fidelity_selects_two_diverse_representative_anchors_per_model() -> None:
    persona = _persona("persona_representative")
    extra = [
        PersonaEvidence(
            evidence_id=f"ev_extra_{index}",
            chunk_id=f"chunk_extra_{index}",
            doc_id=f"doc_extra_{index}",
            domain=f"领域{index}",
            summary=f"不应全部进入自检输入的额外证据{index}",
            confidence="medium",
        )
        for index in range(4)
    ]
    persona = persona.model_copy(
        update={
            "mental_models": [
                model.model_copy(
                    update={"cross_domain_evidence": [*model.cross_domain_evidence, *extra]}
                )
                for model in persona.mental_models
            ],
            "evidence_registry": [*persona.evidence_registry, *extra],
        }
    )
    client = FakeChatClient([_suite().model_dump_json()])

    PersonaFidelityEvaluator(client).design_suite(persona)  # type: ignore[arg-type]

    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert len(persona.evidence_registry) == 6
    assert all(len(item["representative_evidence"]) == 2 for item in payload["mental_models"])
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "ev_extra_3" not in serialized


def test_fidelity_resumes_only_the_failed_stage(tmp_path) -> None:
    persona = _persona("persona_resume_fidelity")
    suite = _suite()
    database = Database(tmp_path / "fidelity-resume.db")
    database.initialize()
    kb_id = KnowledgeBaseRepository(database).ensure_default()
    repository = PersonaRepository(database)
    run = repository.begin_or_resume(
        name=persona.name,
        mode=persona.mode,
        kb_id=kb_id,
        source_hash="source",
        input_hash="input",
        source_doc_ids=["doc_a"],
        map_total=1,
    )
    persona = persona.model_copy(update={"id": run.persona_id})
    repository.save_ready(
        run_id=run.run_id,
        persona=persona,
        markdown=render_persona_markdown(persona),
    )
    first_client = FakeChatClient(
        [suite.model_dump_json(), _answers(suite).model_dump_json(), '{"results": []}']
    )

    with pytest.raises(StructuredDistillationError, match="FidelityResult Schema"):
        FidelityService(
            repository,
            PersonaFidelityEvaluator(first_client),  # type: ignore[arg-type]
        ).evaluate(persona.id)

    interrupted = repository.list_personas(kb_id)[0]
    assert interrupted["fidelity_score"] is None
    assert interrupted["fidelity_checkpoint_count"] == 2

    events = []
    second_client = FakeChatClient([_result().model_dump_json()])
    result = FidelityService(
        repository,
        PersonaFidelityEvaluator(second_client),  # type: ignore[arg-type]
    ).evaluate(persona.id, progress=events.append)

    assert result.total == 89
    assert len(second_client.calls) == 1
    assert [(event.stage, event.state) for event in events] == [
        ("design", "restored"),
        ("answer", "restored"),
        ("judge", "started"),
        ("judge", "completed"),
    ]
    completed = repository.list_personas(kb_id)[0]
    assert completed["fidelity_score"] == 89
    assert completed["fidelity_checkpoint_count"] == 0


def test_fidelity_does_not_wrap_user_cancellation_as_schema_failure() -> None:
    suite = _suite()

    class CancellingClient(FakeChatClient):
        def chat(self, messages, **kwargs):
            if len(self.calls) == 2:
                raise TaskCancelled("Task cancelled")
            return super().chat(messages, **kwargs)

    client = CancellingClient([suite.model_dump_json(), _answers(suite).model_dump_json()])

    with pytest.raises(TaskCancelled):
        PersonaFidelityEvaluator(client).evaluate(  # type: ignore[arg-type]
            _persona("persona_cancel_fidelity"),
            "legacy markdown argument",
        )
