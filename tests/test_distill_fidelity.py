"""Independent Persona fidelity evaluation and persistence tests."""

from __future__ import annotations

import json

import pytest

from tests.test_distill_pipeline import FakeChatClient, _persona
from writing_factory.distill.extraction import StructuredDistillationError
from writing_factory.distill.fidelity import FidelityService, PersonaFidelityEvaluator
from writing_factory.distill.fidelity_models import FidelityAnswers, FidelityResult, FidelitySuite
from writing_factory.distill.serialization import render_persona_markdown
from writing_factory.store import Database
from writing_factory.store.kb_repository import KnowledgeBaseRepository
from writing_factory.store.persona_repository import PersonaRepository


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
    assert "test designer" in system_prompts[0]
    assert "Do not judge" in system_prompts[1]
    assert "independent neutral evaluator" in system_prompts[2]
    assert [call["seed"] for call in client.calls] == [23, 29, 31]
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

    with pytest.raises(StructuredDistillationError, match="case IDs"):
        PersonaFidelityEvaluator(client).evaluate(  # type: ignore[arg-type]
            _persona("persona"), "profile"
        )

    assert len(client.calls) == 2
