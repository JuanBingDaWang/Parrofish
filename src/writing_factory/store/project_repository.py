"""Local project and recoverable writing-task persistence."""

from __future__ import annotations

import json
from uuid import uuid4

from writing_factory.store.database import Database, utc_now


class ProjectRepository:
    """Persist user projects, task configurations, outputs, and edited drafts."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def ensure_default(self, kb_id: str) -> str:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT project_id FROM projects ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is not None:
                return str(row["project_id"])
        return self.create_project(kb_id=kb_id, title="默认项目")

    def create_project(self, *, kb_id: str, title: str, description: str = "") -> str:
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("项目名称不能为空")
        project_id = f"project_{uuid4().hex}"
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO projects(project_id, kb_id, title, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (project_id, kb_id, clean_title, description.strip(), now, now),
            )
        return project_id

    def update_project(self, project_id: str, *, title: str, description: str) -> None:
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("项目名称不能为空")
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE projects SET title = ?, description = ?, updated_at = ?
                WHERE project_id = ?
                """,
                (clean_title, description.strip(), utc_now(), project_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"项目不存在: {project_id}")

    def list_projects(self) -> list[dict[str, object]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT p.*, count(t.task_id) AS task_count
                FROM projects p
                LEFT JOIN writing_tasks t ON t.project_id = p.project_id
                GROUP BY p.project_id
                ORDER BY p.updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_projects(self, project_ids: set[str]) -> int:
        if not project_ids:
            return 0
        placeholders = ",".join("?" for _ in project_ids)
        with self.database.connection() as connection:
            cursor = connection.execute(
                f"DELETE FROM projects WHERE project_id IN ({placeholders})",
                tuple(sorted(project_ids)),
            )
        return cursor.rowcount

    def create_task(
        self,
        *,
        project_id: str,
        kb_id: str,
        persona_id: str,
        title: str,
        task_description: str,
        domain: str,
        citation_style: str,
        selected_doc_ids: set[str],
        allowed_persona_doc_ids: set[str] | None = None,
        generation_options: dict | None = None,
    ) -> str:
        if not selected_doc_ids:
            raise ValueError("写作任务至少需要选择一篇事实语料")
        task_id = f"task_{uuid4().hex}"
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO writing_tasks(
                    task_id, project_id, kb_id, persona_id, title, task_description,
                    domain, citation_style, selected_doc_ids_json,
                    allowed_persona_doc_ids_json, generation_options_json,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    task_id,
                    project_id,
                    kb_id,
                    persona_id,
                    title.strip() or task_description.strip()[:60],
                    task_description.strip(),
                    domain.strip(),
                    citation_style,
                    json.dumps(sorted(selected_doc_ids), ensure_ascii=False),
                    json.dumps(sorted(allowed_persona_doc_ids or set()), ensure_ascii=False),
                    json.dumps(generation_options or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return task_id

    def update_task_state(self, task_id: str, state: dict) -> None:
        state_status = str(state.get("status", "unknown"))
        status = state_status if state_status in {"done", "error", "cancelled"} else "running"
        now = utc_now()
        completed_at = now if state_status == "done" else None
        with self.database.connection() as connection:
            connection.execute(
                """
                UPDATE writing_tasks
                SET status = ?, state_json = ?, error = ?, updated_at = ?,
                    completed_at = coalesce(?, completed_at)
                WHERE task_id = ?
                """,
                (
                    status,
                    json.dumps(state, ensure_ascii=False),
                    state.get("error"),
                    now,
                    completed_at,
                    task_id,
                ),
            )

    def mark_task_status(self, task_id: str, status: str, error: str | None = None) -> None:
        with self.database.connection() as connection:
            connection.execute(
                "UPDATE writing_tasks SET status = ?, error = ?, updated_at = ? WHERE task_id = ?",
                (status, error, utc_now(), task_id),
            )

    def save_edited_draft(
        self,
        task_id: str,
        text: str,
        outline_text: str | None = None,
    ) -> None:
        with self.database.connection() as connection:
            connection.execute(
                """
                UPDATE writing_tasks
                SET edited_draft_text = ?, edited_outline_text = coalesce(?, edited_outline_text),
                    updated_at = ?
                WHERE task_id = ?
                """,
                (text, outline_text, utc_now(), task_id),
            )

    def save_evaluation(self, task_id: str, result: dict) -> None:
        with self.database.connection() as connection:
            connection.execute(
                "UPDATE writing_tasks SET evaluation_json = ?, updated_at = ? WHERE task_id = ?",
                (json.dumps(result, ensure_ascii=False), utc_now(), task_id),
            )

    def list_tasks(self, project_id: str) -> list[dict[str, object]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT task_id, project_id, persona_id, title, task_description, domain,
                       status, error, created_at, updated_at, completed_at
                FROM writing_tasks WHERE project_id = ? ORDER BY updated_at DESC
                """,
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_task(self, task_id: str) -> dict[str, object] | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM writing_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["selected_doc_ids"] = set(json.loads(result.pop("selected_doc_ids_json")))
        result["allowed_persona_doc_ids"] = set(
            json.loads(result.pop("allowed_persona_doc_ids_json"))
        )
        result["generation_options"] = json.loads(
            result.pop("generation_options_json", "{}") or "{}"
        )
        result["state"] = json.loads(result["state_json"]) if result.get("state_json") else None
        result["evaluation"] = (
            json.loads(result["evaluation_json"]) if result.get("evaluation_json") else None
        )
        return result

    def delete_tasks(self, task_ids: set[str]) -> int:
        if not task_ids:
            return 0
        placeholders = ",".join("?" for _ in task_ids)
        with self.database.connection() as connection:
            cursor = connection.execute(
                f"DELETE FROM writing_tasks WHERE task_id IN ({placeholders})",
                tuple(sorted(task_ids)),
            )
        return cursor.rowcount
