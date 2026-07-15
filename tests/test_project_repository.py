"""Project and writing-task persistence tests."""

from pathlib import Path

from writing_factory.store import Database, ProjectRepository


def test_project_task_and_edited_draft_round_trip(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.initialize()
    repository = ProjectRepository(database)
    project_id = repository.create_project(kb_id="kb", title="论文项目")
    task_id = repository.create_task(
        project_id=project_id,
        kb_id="kb",
        persona_id="persona",
        title="测试任务",
        task_description="讨论数字人文",
        domain="出版学",
        citation_style="gb-t-7714",
        selected_doc_ids={"doc1", "doc2"},
        generation_options={
            "preset": "balanced",
            "document_form": "short_text",
            "target_length_chars": 1500,
            "use_hyde": False,
            "use_query_rewrite": True,
            "topic_refinement": False,
            "framework_generation": True,
        },
    )
    repository.update_task_state(task_id, {"status": "done", "task_id": task_id})
    repository.save_edited_draft(task_id, "人工编辑后的稿件")
    repository.save_evaluation(task_id, {"traceability": 1.0})

    task = repository.get_task(task_id)
    assert task is not None
    assert task["selected_doc_ids"] == {"doc1", "doc2"}
    assert task["generation_options"] == {
        "preset": "balanced",
        "document_form": "short_text",
        "target_length_chars": 1500,
        "use_hyde": False,
        "use_query_rewrite": True,
        "topic_refinement": False,
        "framework_generation": True,
    }
    assert task["state"]["status"] == "done"
    assert task["edited_draft_text"] == "人工编辑后的稿件"
    assert task["evaluation"] == {"traceability": 1.0}
    assert repository.list_projects()[0]["task_count"] == 1


def test_partial_pipeline_state_keeps_task_running(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.initialize()
    repository = ProjectRepository(database)
    project_id = repository.create_project(kb_id="kb", title="论文项目")
    task_id = repository.create_task(
        project_id=project_id,
        kb_id="kb",
        persona_id="persona",
        title="测试任务",
        task_description="讨论数字人文",
        domain="",
        citation_style="gb-t-7714",
        selected_doc_ids={"doc1"},
    )

    repository.update_task_state(task_id, {"status": "verifying", "sections": []})

    task = repository.get_task(task_id)
    assert task is not None
    assert task["status"] == "running"
    assert task["state"]["status"] == "verifying"
