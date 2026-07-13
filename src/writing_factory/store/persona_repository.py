"""SQLite checkpoints and final storage for recoverable PersonaSpec distillation."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from writing_factory.distill.models import MapResult, PersonaMode, PersonaSpec
from writing_factory.distill.runtime import RuntimePersonaSpec, build_runtime_persona
from writing_factory.store.database import Database, utc_now

CheckpointModel = TypeVar("CheckpointModel", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class DistillationRunRecord:
    """Identifiers and completed map count for an active or resumed run."""

    run_id: str
    persona_id: str
    map_total: int
    map_completed: int
    profile_id: str = ""
    version_number: int = 1


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
        control_doc_ids: list[str] | None = None,
        domain: str = "",
    ) -> DistillationRunRecord:
        """Resume the latest incomplete identical run or create a new checkpoint tree."""

        with self.database.connection() as connection:
            now = utc_now()
            name_key = name.strip().casefold()
            profile = connection.execute(
                """
                SELECT profile_id FROM persona_profiles
                WHERE kb_id = ? AND name_key = ? AND mode = ?
                """,
                (kb_id, name_key, mode),
            ).fetchone()
            if profile is None:
                profile_id = f"profile_{uuid.uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO persona_profiles(
                        profile_id, kb_id, name, name_key, mode, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (profile_id, kb_id, name, name_key, mode, now, now),
                )
            else:
                profile_id = str(profile["profile_id"])
                connection.execute(
                    """
                    UPDATE persona_profiles SET name = ?, updated_at = ? WHERE profile_id = ?
                    """,
                    (name, now, profile_id),
                )
            row = connection.execute(
                """
                SELECT r.run_id, r.persona_id, r.map_total, r.map_completed,
                       p.version_number
                FROM distillation_runs r
                JOIN persona_specs p ON p.persona_id = r.persona_id
                WHERE p.profile_id = ?
                  AND p.source_hash = ? AND r.input_hash = ?
                  AND r.status != 'ready'
                ORDER BY r.updated_at DESC LIMIT 1
                """,
                (profile_id, source_hash, input_hash),
            ).fetchone()
            if row is not None:
                connection.execute(
                    """
                    UPDATE distillation_runs
                    SET status = 'mapping', error_type = NULL, map_total = ?,
                        target_doc_ids_json = ?, control_doc_ids_json = ?, domain = ?,
                        updated_at = ?
                    WHERE run_id = ?
                    """,
                    (
                        map_total,
                        json.dumps(source_doc_ids, ensure_ascii=False),
                        json.dumps(control_doc_ids or [], ensure_ascii=False),
                        domain,
                        now,
                        row["run_id"],
                    ),
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
                    profile_id=profile_id,
                    version_number=row["version_number"],
                )

            persona_id = f"persona_{uuid.uuid4().hex}"
            run_id = f"distill_{uuid.uuid4().hex}"
            version_number = connection.execute(
                """
                SELECT coalesce(max(version_number), 0) + 1
                FROM persona_specs WHERE profile_id = ?
                """,
                (profile_id,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO persona_specs(
                    persona_id, name, mode, kb_id, status, source_hash,
                    profile_id, version_number, schema_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'mapping', ?, ?, ?, 2, ?, ?)
                """,
                (
                    persona_id,
                    name,
                    mode,
                    kb_id,
                    source_hash,
                    profile_id,
                    version_number,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO distillation_runs(
                    run_id, persona_id, input_hash, status, source_doc_ids_json,
                    map_total, map_completed, target_doc_ids_json, control_doc_ids_json,
                    domain, created_at, updated_at
                ) VALUES (?, ?, ?, 'mapping', ?, ?, 0, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    persona_id,
                    input_hash,
                    json.dumps(source_doc_ids, ensure_ascii=False),
                    map_total,
                    json.dumps(source_doc_ids, ensure_ascii=False),
                    json.dumps(control_doc_ids or [], ensure_ascii=False),
                    domain,
                    now,
                    now,
                ),
            )
        return DistillationRunRecord(
            run_id,
            persona_id,
            map_total,
            0,
            profile_id,
            version_number,
        )

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

    def find_compatible_map_result(
        self,
        *,
        input_hash: str,
        unit_id: str,
    ) -> MapResult | None:
        """复用其他运行中语言、版本和语料单元完全一致的 Map 结果。"""

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT result_json FROM distillation_map_results
                WHERE input_hash = ? AND unit_id = ?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (input_hash, unit_id),
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
        runtime = build_runtime_persona(persona)
        with self.database.connection() as connection:
            connection.execute(
                """
                UPDATE persona_specs
                SET status = 'ready', spec_json = ?, runtime_spec_json = ?, markdown = ?,
                    schema_version = ?, research_date = ?, error_type = NULL, updated_at = ?
                WHERE persona_id = ?
                """,
                (
                    persona.model_dump_json(),
                    runtime.model_dump_json(),
                    markdown,
                    persona.schema_version,
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
            connection.execute(
                """
                UPDATE persona_profiles
                SET current_persona_id = ?, name = ?, updated_at = ?
                WHERE profile_id = (
                    SELECT profile_id FROM persona_specs WHERE persona_id = ?
                )
                """,
                (persona.id, persona.name, now, persona.id),
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

    def load_runtime(self, persona_id: str) -> RuntimePersonaSpec | None:
        """读取不包含蒸馏证据和旧论文事实的运行时档案。"""

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT spec_json, runtime_spec_json FROM persona_specs
                WHERE persona_id = ? AND status = 'ready'
                """,
                (persona_id,),
            ).fetchone()
        if row is None:
            return None
        if row["runtime_spec_json"]:
            return RuntimePersonaSpec.model_validate_json(row["runtime_spec_json"])
        return build_runtime_persona(PersonaSpec.model_validate_json(row["spec_json"]))

    def update_ready(
        self,
        *,
        persona_id: str,
        persona: PersonaSpec,
        markdown: str,
    ) -> None:
        """保存人工编辑后的有效档案，并使基于旧内容的评估失效。"""

        if persona.id != persona_id:
            raise ValueError("档案 id 不允许修改")
        runtime = build_runtime_persona(persona)
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE persona_specs
                SET name = ?, mode = ?, spec_json = ?, runtime_spec_json = ?, markdown = ?,
                    schema_version = ?, research_date = ?, error_type = NULL, updated_at = ?
                WHERE persona_id = ? AND status = 'ready'
                """,
                (
                    persona.name,
                    persona.mode,
                    persona.model_dump_json(),
                    runtime.model_dump_json(),
                    markdown,
                    persona.schema_version,
                    persona.research_date.isoformat(),
                    utc_now(),
                    persona_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("只能编辑已经完成的档案")
            connection.execute(
                """
                UPDATE persona_profiles
                SET name = ?, name_key = ?, mode = ?, updated_at = ?
                WHERE profile_id = (
                    SELECT profile_id FROM persona_specs WHERE persona_id = ?
                )
                """,
                (
                    persona.name,
                    persona.name.strip().casefold(),
                    persona.mode,
                    utc_now(),
                    persona_id,
                ),
            )
            connection.execute(
                "DELETE FROM persona_evaluations WHERE persona_id = ?",
                (persona_id,),
            )

    def delete_personas(self, kb_id: str, persona_ids: set[str]) -> int:
        """批量删除顶层档案及其全部版本、断点和评估。"""

        identifiers = sorted(persona_ids)
        if not identifiers:
            return 0
        placeholders = ",".join("?" for _ in identifiers)
        with self.database.connection() as connection:
            profiles = connection.execute(
                f"""
                SELECT DISTINCT profile_id FROM persona_specs
                WHERE kb_id = ? AND persona_id IN ({placeholders})
                """,
                [kb_id, *identifiers],
            ).fetchall()
            profile_ids = [str(row["profile_id"]) for row in profiles if row["profile_id"]]
            if not profile_ids:
                return 0
            profile_placeholders = ",".join("?" for _ in profile_ids)
            connection.execute(
                f"DELETE FROM persona_specs WHERE profile_id IN ({profile_placeholders})",
                profile_ids,
            )
            cursor = connection.execute(
                f"DELETE FROM persona_profiles WHERE profile_id IN ({profile_placeholders})",
                profile_ids,
            )
        return max(0, cursor.rowcount)

    def list_personas(self, kb_id: str) -> list[dict[str, object]]:
        """每个顶层档案只返回最新版本，历史版本在详情中查看。"""

        with self.database.connection() as connection:
            rows = connection.execute(
                """
                WITH ranked AS (
                    SELECT p.*,
                           row_number() OVER (
                               PARTITION BY p.profile_id
                               ORDER BY p.version_number DESC, p.updated_at DESC
                           ) AS version_rank
                    FROM persona_specs p
                    WHERE p.kb_id = ?
                )
                SELECT p.persona_id, p.profile_id, p.version_number, p.name, p.mode,
                       p.status, p.research_date, p.spec_json, p.updated_at,
                       (
                           SELECT count(*) FROM persona_specs versions
                           WHERE versions.profile_id = p.profile_id
                       ) AS version_count,
                       (
                           SELECT e.score FROM persona_evaluations e
                           WHERE e.persona_id = p.persona_id
                             AND e.evaluation_type = 'nuwa_fidelity'
                           ORDER BY e.created_at DESC LIMIT 1
                       ) AS fidelity_score
                FROM ranked p WHERE p.version_rank = 1
                ORDER BY p.updated_at DESC
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
                    "profile_id": row["profile_id"],
                    "version_number": row["version_number"],
                    "version_count": row["version_count"],
                    "name": row["name"],
                    "mode": row["mode"],
                    "status": row["status"],
                    "model_count": model_count,
                    "fidelity_score": row["fidelity_score"],
                    "research_date": row["research_date"] or row["updated_at"],
                }
            )
        return profiles

    def list_versions(self, persona_id: str) -> list[dict[str, object]]:
        """返回与指定版本同属一个顶层档案的版本历史。"""

        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT version.persona_id, version.version_number, version.status,
                       version.research_date, version.source_hash, version.updated_at
                FROM persona_specs version
                WHERE version.profile_id = (
                    SELECT profile_id FROM persona_specs WHERE persona_id = ?
                )
                ORDER BY version.version_number DESC
                """,
                (persona_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_stage_result(
        self,
        *,
        run_id: str,
        stage: str,
        item_id: str,
        input_hash: str,
        result: BaseModel,
    ) -> None:
        """保存论文归并、候选登记或验证结果检查点。"""

        now = utc_now()
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO distillation_stage_results(
                    run_id, stage, item_id, input_hash, result_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, stage, item_id) DO UPDATE SET
                    input_hash = excluded.input_hash,
                    result_json = excluded.result_json,
                    updated_at = excluded.updated_at
                """,
                (run_id, stage, item_id, input_hash, result.model_dump_json(), now, now),
            )

    def load_stage_result(
        self,
        *,
        run_id: str,
        stage: str,
        item_id: str,
        model: type[CheckpointModel],
    ) -> CheckpointModel | None:
        """读取当前运行的一个结构化阶段检查点。"""

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT result_json FROM distillation_stage_results
                WHERE run_id = ? AND stage = ? AND item_id = ?
                """,
                (run_id, stage, item_id),
            ).fetchone()
        return None if row is None else model.model_validate_json(row["result_json"])

    def find_compatible_stage_result(
        self,
        *,
        stage: str,
        item_id: str,
        input_hash: str,
        model: type[CheckpointModel],
    ) -> CheckpointModel | None:
        """跨运行复用输入、Schema 和提示词版本完全相同的阶段结果。"""

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT result_json FROM distillation_stage_results
                WHERE stage = ? AND item_id = ? AND input_hash = ?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (stage, item_id, input_hash),
            ).fetchone()
        return None if row is None else model.model_validate_json(row["result_json"])
