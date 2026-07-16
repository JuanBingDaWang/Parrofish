"""SQLite checkpoints and final storage for recoverable PersonaSpec distillation."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Literal, TypeVar

from pydantic import BaseModel

from writing_factory.distill.fidelity_models import FIDELITY_PIPELINE_VERSION
from writing_factory.distill.models import MapResult, PersonaMode, PersonaSpec
from writing_factory.distill.options import LEGACY_DISTILLATION_OPTIONS, DistillationOptions
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


@dataclass(frozen=True, slots=True)
class PersonaSourceRoles:
    """Distinguish author target texts from optional comparison texts."""

    target_doc_ids: frozenset[str]
    control_doc_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class DistillationRunContext:
    """Persisted inputs needed to resume one exact interrupted run."""

    run_id: str
    persona_id: str
    profile_id: str
    version_number: int
    name: str
    mode: PersonaMode
    kb_id: str
    status: str
    source_hash: str
    input_hash: str
    target_doc_ids: frozenset[str]
    control_doc_ids: frozenset[str]
    domain: str
    map_total: int
    map_completed: int
    options: DistillationOptions


RunStrategy = Literal["auto", "new", "upgrade"]


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
        quality_options: DistillationOptions = LEGACY_DISTILLATION_OPTIONS,
        strategy: RunStrategy = "auto",
        base_persona_id: str | None = None,
    ) -> DistillationRunRecord:
        """Create or resume a run according to an explicit caller-selected strategy."""

        with self.database.connection() as connection:
            now = utc_now()
            name_key = name.strip().casefold()
            if strategy == "upgrade":
                if not base_persona_id:
                    raise ValueError("升级档案必须指定一个已完成的基础版本")
                base = connection.execute(
                    """
                    SELECT p.profile_id, p.name, p.mode, p.kb_id
                    FROM persona_specs p
                    WHERE p.persona_id = ? AND p.kb_id = ? AND p.status = 'ready'
                    """,
                    (base_persona_id, kb_id),
                ).fetchone()
                if base is None:
                    raise ValueError("只能升级已经完成的作者档案")
                if str(base["name"]).strip().casefold() != name_key or base["mode"] != mode:
                    raise ValueError("升级任务的名称和模式必须与基础档案一致")
                profile = connection.execute(
                    "SELECT profile_id FROM persona_profiles WHERE profile_id = ?",
                    (base["profile_id"],),
                ).fetchone()
            else:
                profile = connection.execute(
                    """
                    SELECT profile_id FROM persona_profiles
                    WHERE kb_id = ? AND name_key = ? AND mode = ?
                    """,
                    (kb_id, name_key, mode),
                ).fetchone()
            if strategy == "new" and profile is not None:
                existing = connection.execute(
                    """
                    SELECT p.persona_id, p.status
                    FROM persona_specs p
                    WHERE p.profile_id = ?
                    ORDER BY p.version_number DESC LIMIT 1
                    """,
                    (profile["profile_id"],),
                ).fetchone()
                action = (
                    "继续" if existing is not None and existing["status"] != "ready" else "升级"
                )
                raise ValueError(f"同名档案已经存在，请选中该档案并点击“{action}”")
            if profile is None:
                if strategy == "upgrade":
                    raise ValueError("升级所选档案不存在")
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
            row = None
            if strategy == "auto":
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
                        quality_options_json = ?,
                        updated_at = ?
                    WHERE run_id = ?
                    """,
                    (
                        map_total,
                        json.dumps(source_doc_ids, ensure_ascii=False),
                        json.dumps(control_doc_ids or [], ensure_ascii=False),
                        domain,
                        quality_options.model_dump_json(),
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

            if strategy == "upgrade":
                unfinished = connection.execute(
                    """
                    SELECT p.persona_id FROM persona_specs p
                    WHERE p.profile_id = ? AND p.status != 'ready'
                    ORDER BY p.version_number DESC LIMIT 1
                    """,
                    (profile_id,),
                ).fetchone()
                if unfinished is not None:
                    raise ValueError("该档案已有未完成版本，请先选中它并点击“继续”")

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
                    domain, quality_options_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'mapping', ?, ?, 0, ?, ?, ?, ?, ?, ?)
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
                    quality_options.model_dump_json(),
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

    def load_run_context(self, kb_id: str, persona_id: str) -> DistillationRunContext | None:
        """Load the exact persisted inputs for one persona version."""

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT r.run_id, r.persona_id, r.status, r.input_hash,
                       r.target_doc_ids_json, r.control_doc_ids_json, r.domain,
                       r.map_total, r.map_completed, r.quality_options_json,
                       p.profile_id, p.version_number, p.name, p.mode, p.kb_id,
                       p.source_hash
                FROM distillation_runs r
                JOIN persona_specs p ON p.persona_id = r.persona_id
                WHERE r.persona_id = ? AND p.kb_id = ?
                ORDER BY r.updated_at DESC LIMIT 1
                """,
                (persona_id, kb_id),
            ).fetchone()
        if row is None:
            return None
        try:
            target_doc_ids = frozenset(
                str(item) for item in json.loads(row["target_doc_ids_json"] or "[]")
            )
            control_doc_ids = frozenset(
                str(item) for item in json.loads(row["control_doc_ids_json"] or "[]")
            )
            raw_options = json.loads(row["quality_options_json"] or "{}")
            options = (
                DistillationOptions.model_validate(raw_options)
                if raw_options
                else LEGACY_DISTILLATION_OPTIONS
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return DistillationRunContext(
            run_id=str(row["run_id"]),
            persona_id=str(row["persona_id"]),
            profile_id=str(row["profile_id"]),
            version_number=int(row["version_number"]),
            name=str(row["name"]),
            mode=row["mode"],
            kb_id=str(row["kb_id"]),
            status=str(row["status"]),
            source_hash=str(row["source_hash"]),
            input_hash=str(row["input_hash"]),
            target_doc_ids=target_doc_ids,
            control_doc_ids=control_doc_ids,
            domain=str(row["domain"] or ""),
            map_total=int(row["map_total"]),
            map_completed=int(row["map_completed"]),
            options=options,
        )

    def prepare_exact_resume(
        self,
        context: DistillationRunContext,
        *,
        map_total: int,
    ) -> DistillationRunRecord:
        """Reopen one selected non-ready run without searching by display name."""

        if context.status == "ready":
            raise ValueError("已经完成的档案不能继续；如需加入语料请使用“升级”")
        now = utc_now()
        with self.database.connection() as connection:
            run_cursor = connection.execute(
                """
                UPDATE distillation_runs
                SET status = 'mapping', error_type = NULL, map_total = ?, updated_at = ?
                WHERE run_id = ? AND persona_id = ? AND status != 'ready'
                """,
                (map_total, now, context.run_id, context.persona_id),
            )
            if run_cursor.rowcount != 1:
                raise ValueError("所选蒸馏断点已经变化，请刷新档案列表")
            connection.execute(
                """
                UPDATE persona_specs
                SET status = 'mapping', error_type = NULL, updated_at = ?
                WHERE persona_id = ?
                """,
                (now, context.persona_id),
            )
        return DistillationRunRecord(
            run_id=context.run_id,
            persona_id=context.persona_id,
            map_total=map_total,
            map_completed=context.map_completed,
            profile_id=context.profile_id,
            version_number=context.version_number,
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
        persona_id: str | None = None,
    ) -> MapResult | None:
        """Reuse a compatible Map only from an explicitly selected ready persona."""

        if not persona_id:
            return None
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT m.result_json FROM distillation_map_results m
                JOIN distillation_runs r ON r.run_id = m.run_id
                JOIN persona_specs p ON p.persona_id = r.persona_id
                WHERE m.input_hash = ? AND m.unit_id = ?
                  AND p.persona_id = ? AND p.status = 'ready' AND r.status = 'ready'
                ORDER BY m.updated_at DESC LIMIT 1
                """,
                (input_hash, unit_id, persona_id),
            ).fetchone()
        return None if row is None else MapResult.model_validate_json(row["result_json"])

    def load_persona_map_result(
        self,
        *,
        persona_id: str,
        unit_id: str,
        chunk_ids: list[str],
    ) -> MapResult | None:
        """Load an older Map after verifying its exact immutable chunk membership."""

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT m.chunk_ids_json, m.result_json
                FROM distillation_map_results m
                JOIN distillation_runs r ON r.run_id = m.run_id
                JOIN persona_specs p ON p.persona_id = r.persona_id
                WHERE p.persona_id = ? AND p.status = 'ready' AND r.status = 'ready'
                  AND m.unit_id = ?
                ORDER BY m.updated_at DESC LIMIT 1
                """,
                (persona_id, unit_id),
            ).fetchone()
        if row is None:
            return None
        try:
            stored_chunks = [str(item) for item in json.loads(row["chunk_ids_json"])]
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if stored_chunks != chunk_ids:
            return None
        return MapResult.model_validate_json(row["result_json"])

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

    def save_fidelity_stage(
        self,
        *,
        persona_id: str,
        stage: Literal["design", "answer", "judge"],
        input_hash: str,
        result: BaseModel,
        duration_ms: int,
    ) -> None:
        """Persist one validated self-check stage and invalidate its dependants."""

        downstream = {
            "design": ("answer", "judge"),
            "answer": ("judge",),
            "judge": (),
        }[stage]
        now = utc_now()
        with self.database.connection() as connection:
            if downstream:
                placeholders = ",".join("?" for _ in downstream)
                connection.execute(
                    f"""
                    DELETE FROM persona_fidelity_stages
                    WHERE persona_id = ? AND stage IN ({placeholders})
                    """,
                    (persona_id, *downstream),
                )
            connection.execute(
                """
                INSERT INTO persona_fidelity_stages(
                    persona_id, stage, pipeline_version, input_hash, result_json,
                    duration_ms, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(persona_id, stage) DO UPDATE SET
                    pipeline_version = excluded.pipeline_version,
                    input_hash = excluded.input_hash,
                    result_json = excluded.result_json,
                    duration_ms = excluded.duration_ms,
                    updated_at = excluded.updated_at
                """,
                (
                    persona_id,
                    stage,
                    FIDELITY_PIPELINE_VERSION,
                    input_hash,
                    result.model_dump_json(),
                    max(0, duration_ms),
                    now,
                    now,
                ),
            )

    def load_fidelity_stage(
        self,
        *,
        persona_id: str,
        stage: Literal["design", "answer", "judge"],
        input_hash: str,
        model: type[CheckpointModel],
    ) -> tuple[CheckpointModel, int] | None:
        """Load a stage only when its pipeline version and exact input still match."""

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT result_json, duration_ms
                FROM persona_fidelity_stages
                WHERE persona_id = ? AND stage = ? AND pipeline_version = ?
                  AND input_hash = ?
                """,
                (persona_id, stage, FIDELITY_PIPELINE_VERSION, input_hash),
            ).fetchone()
        if row is None:
            return None
        return model.model_validate_json(row["result_json"]), int(row["duration_ms"])

    def complete_fidelity_evaluation(
        self,
        *,
        persona_id: str,
        result_json: str,
        score: int,
    ) -> str:
        """Atomically publish a valid score and clear its temporary stage checkpoints."""

        evaluation_id = f"eval_{uuid.uuid4().hex}"
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO persona_evaluations(
                    evaluation_id, persona_id, evaluation_type, score,
                    result_json, created_at
                ) VALUES (?, ?, 'nuwa_fidelity', ?, ?, ?)
                """,
                (evaluation_id, persona_id, score, result_json, utc_now()),
            )
            connection.execute(
                "DELETE FROM persona_fidelity_stages WHERE persona_id = ?",
                (persona_id,),
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

    def load_source_roles(self, persona_id: str) -> PersonaSourceRoles | None:
        """Load target/control roles from the latest completed distillation run.

        Older profiles may predate role columns or have no completed run metadata;
        callers must treat ``None`` as a signal to use the conservative legacy policy.
        """

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT target_doc_ids_json, control_doc_ids_json
                FROM distillation_runs
                WHERE persona_id = ? AND status = 'ready'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (persona_id,),
            ).fetchone()
        if row is None or not row["target_doc_ids_json"]:
            return None
        try:
            target = frozenset(str(item) for item in json.loads(row["target_doc_ids_json"]))
            control = frozenset(
                str(item) for item in json.loads(row["control_doc_ids_json"] or "[]")
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not target:
            return None
        return PersonaSourceRoles(target_doc_ids=target, control_doc_ids=control)

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
            connection.execute(
                "DELETE FROM persona_fidelity_stages WHERE persona_id = ?",
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
                       p.status, p.error_type, p.research_date, p.spec_json, p.updated_at,
                       (
                           SELECT r.quality_options_json FROM distillation_runs r
                           WHERE r.persona_id = p.persona_id
                           ORDER BY r.updated_at DESC LIMIT 1
                       ) AS quality_options_json,
                       (
                           SELECT count(*) FROM persona_specs versions
                           WHERE versions.profile_id = p.profile_id
                       ) AS version_count,
                       (
                           SELECT e.score FROM persona_evaluations e
                           WHERE e.persona_id = p.persona_id
                             AND e.evaluation_type = 'nuwa_fidelity'
                           ORDER BY e.created_at DESC LIMIT 1
                       ) AS fidelity_score,
                       (
                           SELECT count(*) FROM persona_fidelity_stages f
                           WHERE f.persona_id = p.persona_id
                             AND f.pipeline_version = ?
                       ) AS fidelity_checkpoint_count
                FROM ranked p WHERE p.version_rank = 1
                ORDER BY p.updated_at DESC
                """,
                (kb_id, FIDELITY_PIPELINE_VERSION),
            ).fetchall()
        profiles: list[dict[str, object]] = []
        for row in rows:
            model_count = 0
            quality_options = LEGACY_DISTILLATION_OPTIONS
            if row["spec_json"]:
                spec_payload = json.loads(row["spec_json"])
                model_count = len(spec_payload.get("mental_models", []))
                quality_options = DistillationOptions.model_validate(
                    spec_payload.get("distillation_options") or {}
                )
            elif row["quality_options_json"]:
                payload = json.loads(row["quality_options_json"] or "{}")
                if payload:
                    quality_options = DistillationOptions.model_validate(payload)
            profiles.append(
                {
                    "persona_id": row["persona_id"],
                    "profile_id": row["profile_id"],
                    "version_number": row["version_number"],
                    "version_count": row["version_count"],
                    "name": row["name"],
                    "mode": row["mode"],
                    "status": row["status"],
                    "error_type": row["error_type"],
                    "model_count": model_count,
                    "quality_preset": quality_options.preset,
                    "quality_label": quality_options.label,
                    "fidelity_score": row["fidelity_score"],
                    "fidelity_checkpoint_count": row["fidelity_checkpoint_count"],
                    "research_date": row["research_date"] or row["updated_at"],
                }
            )
        return profiles

    def list_ready_personas(self, kb_id: str) -> list[dict[str, object]]:
        """Return only each profile's last successfully published version."""

        with self.database.connection() as connection:
            rows = connection.execute(
                """
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
                       ) AS fidelity_score,
                       (
                           SELECT count(*) FROM persona_fidelity_stages f
                           WHERE f.persona_id = p.persona_id
                             AND f.pipeline_version = ?
                       ) AS fidelity_checkpoint_count
                FROM persona_profiles profile
                JOIN persona_specs p ON p.persona_id = profile.current_persona_id
                WHERE profile.kb_id = ? AND p.status = 'ready'
                ORDER BY profile.updated_at DESC
                """,
                (FIDELITY_PIPELINE_VERSION, kb_id),
            ).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            payload = json.loads(row["spec_json"])
            options = DistillationOptions.model_validate(payload.get("distillation_options") or {})
            result.append(
                {
                    "persona_id": row["persona_id"],
                    "profile_id": row["profile_id"],
                    "version_number": row["version_number"],
                    "version_count": row["version_count"],
                    "name": row["name"],
                    "mode": row["mode"],
                    "status": row["status"],
                    "model_count": len(payload.get("mental_models", [])),
                    "quality_preset": options.preset,
                    "quality_label": options.label,
                    "fidelity_score": row["fidelity_score"],
                    "fidelity_checkpoint_count": row["fidelity_checkpoint_count"],
                    "research_date": row["research_date"] or row["updated_at"],
                }
            )
        return result

    def list_versions(self, persona_id: str) -> list[dict[str, object]]:
        """返回与指定版本同属一个顶层档案的版本历史。"""

        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT version.persona_id, version.version_number, version.status,
                       version.error_type,
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
        persona_id: str | None = None,
    ) -> CheckpointModel | None:
        """Reuse a stage only from an explicitly selected ready persona."""

        if not persona_id:
            return None
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT s.result_json FROM distillation_stage_results s
                JOIN distillation_runs r ON r.run_id = s.run_id
                JOIN persona_specs p ON p.persona_id = r.persona_id
                WHERE s.stage = ? AND s.item_id = ? AND s.input_hash = ?
                  AND p.persona_id = ? AND p.status = 'ready' AND r.status = 'ready'
                ORDER BY s.updated_at DESC LIMIT 1
                """,
                (stage, item_id, input_hash, persona_id),
            ).fetchone()
        return None if row is None else model.model_validate_json(row["result_json"])

    def load_persona_stage_result(
        self,
        *,
        persona_id: str,
        stage: str,
        item_id: str,
        model: type[CheckpointModel],
    ) -> CheckpointModel | None:
        """Load a document-scoped stage from one explicit completed base version."""

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT s.result_json FROM distillation_stage_results s
                JOIN distillation_runs r ON r.run_id = s.run_id
                JOIN persona_specs p ON p.persona_id = r.persona_id
                WHERE p.persona_id = ? AND p.status = 'ready' AND r.status = 'ready'
                  AND s.stage = ? AND s.item_id = ?
                ORDER BY s.updated_at DESC LIMIT 1
                """,
                (persona_id, stage, item_id),
            ).fetchone()
        return None if row is None else model.model_validate_json(row["result_json"])
