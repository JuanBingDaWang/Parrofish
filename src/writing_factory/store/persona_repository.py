"""SQLite checkpoints and final storage for recoverable PersonaSpec distillation."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from writing_factory.distill.models import MapResult, PersonaMode, PersonaSpec
from writing_factory.store.database import Database, utc_now


@dataclass(frozen=True, slots=True)
class DistillationRunRecord:
    """Identifiers and completed map count for an active or resumed run."""

    run_id: str
    persona_id: str
    map_total: int
    map_completed: int


class PersonaRepository:
    """Persist every expensive map result before proceeding to reduction."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def find_ready(
        self,
        *,
        name: str,
        mode: PersonaMode,
        kb_id: str,
        source_hash: str,
        input_hash: str,
    ) -> tuple[str, PersonaSpec, str] | None:
        """Return a completed identical profile without spending another API call."""

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT r.run_id, p.spec_json, p.markdown
                FROM persona_specs p
                JOIN distillation_runs r ON r.persona_id = p.persona_id
                WHERE p.name = ? AND p.mode = ? AND p.kb_id = ?
                  AND p.source_hash = ? AND r.input_hash = ?
                  AND p.status = 'ready' AND r.status = 'ready'
                ORDER BY p.updated_at DESC LIMIT 1
                """,
                (name, mode, kb_id, source_hash, input_hash),
            ).fetchone()
        if row is None:
            return None
        return row["run_id"], PersonaSpec.model_validate_json(row["spec_json"]), row["markdown"]

    def begin_or_resume(
        self,
        *,
        name: str,
        mode: PersonaMode,
        kb_id: str,
        source_hash: str,
        input_hash: str,
        source_doc_ids: list[str],
        map_total: int,
    ) -> DistillationRunRecord:
        """Resume the latest incomplete identical run or create a new checkpoint tree."""

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT r.run_id, r.persona_id, r.map_total, r.map_completed
                FROM distillation_runs r
                JOIN persona_specs p ON p.persona_id = r.persona_id
                WHERE p.name = ? AND p.mode = ? AND p.kb_id = ?
                  AND p.source_hash = ? AND r.input_hash = ?
                  AND r.status != 'ready'
                ORDER BY r.updated_at DESC LIMIT 1
                """,
                (name, mode, kb_id, source_hash, input_hash),
            ).fetchone()
            now = utc_now()
            if row is not None:
                connection.execute(
                    """
                    UPDATE distillation_runs
                    SET status = 'mapping', error_type = NULL, map_total = ?, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (map_total, now, row["run_id"]),
                )
                connection.execute(
                    """
                    UPDATE persona_specs
                    SET status = 'mapping', error_type = NULL, updated_at = ?
                    WHERE persona_id = ?
                    """,
                    (now, row["persona_id"]),
                )
                return DistillationRunRecord(
                    run_id=row["run_id"],
                    persona_id=row["persona_id"],
                    map_total=map_total,
                    map_completed=row["map_completed"],
                )

            persona_id = f"persona_{uuid.uuid4().hex}"
            run_id = f"distill_{uuid.uuid4().hex}"
            connection.execute(
                """
                INSERT INTO persona_specs(
                    persona_id, name, mode, kb_id, status, source_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'mapping', ?, ?, ?)
                """,
                (persona_id, name, mode, kb_id, source_hash, now, now),
            )
            connection.execute(
                """
                INSERT INTO distillation_runs(
                    run_id, persona_id, input_hash, status, source_doc_ids_json,
                    map_total, map_completed, created_at, updated_at
                ) VALUES (?, ?, ?, 'mapping', ?, ?, 0, ?, ?)
                """,
                (
                    run_id,
                    persona_id,
                    input_hash,
                    json.dumps(source_doc_ids, ensure_ascii=False),
                    map_total,
                    now,
                    now,
                ),
            )
        return DistillationRunRecord(run_id, persona_id, map_total, 0)

    def get_map_result(self, run_id: str, unit_id: str) -> MapResult | None:
        """Load one completed map checkpoint."""

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT result_json FROM distillation_map_results
                WHERE run_id = ? AND unit_id = ?
                """,
                (run_id, unit_id),
            ).fetchone()
        return None if row is None else MapResult.model_validate_json(row["result_json"])

    def save_map_result(
        self,
        *,
        run_id: str,
        unit_id: str,
        input_hash: str,
        chunk_ids: list[str],
        result: MapResult,
    ) -> int:
        """Upsert a map checkpoint and return the authoritative completed count."""

        now = utc_now()
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO distillation_map_results(
                    run_id, unit_id, input_hash, chunk_ids_json, result_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, unit_id) DO UPDATE SET
                    input_hash = excluded.input_hash,
                    chunk_ids_json = excluded.chunk_ids_json,
                    result_json = excluded.result_json,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    unit_id,
                    input_hash,
                    json.dumps(chunk_ids),
                    result.model_dump_json(),
                    now,
                    now,
                ),
            )
            completed = connection.execute(
                "SELECT COUNT(*) FROM distillation_map_results WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            connection.execute(
                """
                UPDATE distillation_runs SET map_completed = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (completed, now, run_id),
            )
        return completed

    def update_stage(self, run_id: str, persona_id: str, status: str) -> None:
        """Move run and profile through mapping, reducing, and validating together."""

        now = utc_now()
        with self.database.connection() as connection:
            connection.execute(
                "UPDATE distillation_runs SET status = ?, updated_at = ? WHERE run_id = ?",
                (status, now, run_id),
            )
            connection.execute(
                "UPDATE persona_specs SET status = ?, updated_at = ? WHERE persona_id = ?",
                (status, now, persona_id),
            )

    def save_ready(
        self,
        *,
        run_id: str,
        persona: PersonaSpec,
        markdown: str,
    ) -> None:
        """Atomically publish final JSON and deterministic Markdown."""

        now = utc_now()
        with self.database.connection() as connection:
            connection.execute(
                """
                UPDATE persona_specs
                SET status = 'ready', spec_json = ?, markdown = ?, research_date = ?,
                    error_type = NULL, updated_at = ?
                WHERE persona_id = ?
                """,
                (
                    persona.model_dump_json(),
                    markdown,
                    persona.research_date.isoformat(),
                    now,
                    persona.id,
                ),
            )
            connection.execute(
                """
                UPDATE distillation_runs
                SET status = 'ready', error_type = NULL, updated_at = ?
                WHERE run_id = ?
                """,
                (now, run_id),
            )

    def mark_failed(self, run_id: str, persona_id: str, error_type: str) -> None:
        """Persist only the sanitized error type and retain completed map results."""

        now = utc_now()
        with self.database.connection() as connection:
            connection.execute(
                """
                UPDATE distillation_runs
                SET status = 'failed', error_type = ?, updated_at = ? WHERE run_id = ?
                """,
                (error_type, now, run_id),
            )
            connection.execute(
                """
                UPDATE persona_specs
                SET status = 'failed', error_type = ?, updated_at = ? WHERE persona_id = ?
                """,
                (error_type, now, persona_id),
            )

    def save_evaluation(
        self,
        *,
        persona_id: str,
        evaluation_type: str,
        result_json: str,
        score: int | None = None,
    ) -> str:
        """Persist an independent static or LLM-based quality report."""

        evaluation_id = f"eval_{uuid.uuid4().hex}"
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO persona_evaluations(
                    evaluation_id, persona_id, evaluation_type, score,
                    result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    persona_id,
                    evaluation_type,
                    score,
                    result_json,
                    utc_now(),
                ),
            )
        return evaluation_id

    def load_ready(self, persona_id: str) -> tuple[PersonaSpec, str] | None:
        """Load authoritative JSON and deterministic Markdown for one ready profile."""

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT spec_json, markdown FROM persona_specs
                WHERE persona_id = ? AND status = 'ready'
                """,
                (persona_id,),
            ).fetchone()
        if row is None:
            return None
        return PersonaSpec.model_validate_json(row["spec_json"]), row["markdown"]

    def list_personas(self, kb_id: str) -> list[dict[str, object]]:
        """Return compact profile metadata for the desktop table."""

        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT p.persona_id, p.name, p.mode, p.status, p.research_date,
                       p.spec_json, p.updated_at,
                       (
                           SELECT e.score FROM persona_evaluations e
                           WHERE e.persona_id = p.persona_id
                             AND e.evaluation_type = 'nuwa_fidelity'
                           ORDER BY e.created_at DESC LIMIT 1
                       ) AS fidelity_score
                FROM persona_specs p WHERE p.kb_id = ? ORDER BY p.updated_at DESC
                """,
                (kb_id,),
            ).fetchall()
        profiles: list[dict[str, object]] = []
        for row in rows:
            model_count = 0
            if row["spec_json"]:
                model_count = len(json.loads(row["spec_json"]).get("mental_models", []))
            profiles.append(
                {
                    "persona_id": row["persona_id"],
                    "name": row["name"],
                    "mode": row["mode"],
                    "status": row["status"],
                    "model_count": model_count,
                    "fidelity_score": row["fidelity_score"],
                    "research_date": row["research_date"] or row["updated_at"],
                }
            )
        return profiles
