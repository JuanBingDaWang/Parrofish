"""Versioned SQLite schema for local, recoverable application state."""

from __future__ import annotations

import sqlite3

MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS knowledge_bases (
        kb_id TEXT PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        description TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        sha256 TEXT NOT NULL UNIQUE,
        filename TEXT NOT NULL,
        format TEXT NOT NULL,
        source_path TEXT NOT NULL,
        managed_path TEXT NOT NULL,
        bib_json TEXT NOT NULL DEFAULT '{}',
        parser_name TEXT,
        parser_version TEXT,
        canonical_text TEXT NOT NULL DEFAULT '',
        ingest_date TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS knowledge_base_documents (
        kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
        added_at TEXT NOT NULL,
        PRIMARY KEY (kb_id, doc_id)
    );

    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
        text TEXT NOT NULL,
        page_start INTEGER,
        page_end INTEGER,
        section_heading TEXT,
        chunk_index INTEGER NOT NULL,
        char_start INTEGER NOT NULL,
        char_end INTEGER NOT NULL,
        parent_id TEXT REFERENCES chunks(chunk_id) ON DELETE CASCADE,
        chunk_kind TEXT NOT NULL CHECK (chunk_kind IN ('parent', 'child')),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        CHECK (char_start >= 0 AND char_end >= char_start),
        UNIQUE (doc_id, chunk_kind, chunk_index)
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);

    CREATE TABLE IF NOT EXISTS ingest_jobs (
        job_id TEXT PRIMARY KEY,
        kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
        source_path TEXT NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('pending', 'parsing', 'chunking', 'indexing', 'ready', 'failed')
        ),
        document_id TEXT REFERENCES documents(doc_id) ON DELETE SET NULL,
        error_message TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_ingest_jobs_kb_status
        ON ingest_jobs(kb_id, status);

    CREATE TABLE IF NOT EXISTS api_cache (
        cache_key TEXT PRIMARY KEY,
        provider TEXT NOT NULL,
        operation TEXT NOT NULL,
        response_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT
    );

    CREATE TABLE IF NOT EXISTS api_calls (
        call_id TEXT PRIMARY KEY,
        request_hash TEXT NOT NULL,
        provider TEXT NOT NULL,
        operation TEXT NOT NULL,
        model TEXT,
        reasoning_effort TEXT,
        prompt_summary TEXT NOT NULL,
        cache_hit INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        input_tokens INTEGER,
        output_tokens INTEGER,
        total_tokens INTEGER,
        duration_ms INTEGER NOT NULL,
        result_summary TEXT,
        error_type TEXT,
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_api_calls_created ON api_calls(created_at);
    CREATE INDEX IF NOT EXISTS idx_api_calls_request ON api_calls(request_hash);
    """,
    """
    ALTER TABLE knowledge_base_documents
        ADD COLUMN status TEXT NOT NULL DEFAULT 'indexing'
        CHECK (status IN ('indexing', 'ready', 'failed'));
    ALTER TABLE knowledge_base_documents
        ADD COLUMN last_job_id TEXT;
    CREATE INDEX IF NOT EXISTS idx_kb_documents_status
        ON knowledge_base_documents(kb_id, status);
    """,
    """
    CREATE TABLE IF NOT EXISTS persona_specs (
        persona_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        mode TEXT NOT NULL CHECK (mode IN ('person', 'topic')),
        kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
        status TEXT NOT NULL CHECK (
            status IN ('pending', 'mapping', 'reducing', 'validating', 'ready', 'failed')
        ),
        spec_json TEXT,
        markdown TEXT,
        source_hash TEXT NOT NULL,
        research_date TEXT,
        error_type TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_persona_specs_kb_status
        ON persona_specs(kb_id, status);

    CREATE TABLE IF NOT EXISTS distillation_runs (
        run_id TEXT PRIMARY KEY,
        persona_id TEXT NOT NULL REFERENCES persona_specs(persona_id) ON DELETE CASCADE,
        input_hash TEXT NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('pending', 'mapping', 'reducing', 'validating', 'ready', 'failed')
        ),
        source_doc_ids_json TEXT NOT NULL,
        map_total INTEGER NOT NULL DEFAULT 0,
        map_completed INTEGER NOT NULL DEFAULT 0,
        error_type TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_distillation_runs_persona
        ON distillation_runs(persona_id, status);
    CREATE INDEX IF NOT EXISTS idx_distillation_runs_input
        ON distillation_runs(input_hash, status);

    CREATE TABLE IF NOT EXISTS distillation_map_results (
        run_id TEXT NOT NULL REFERENCES distillation_runs(run_id) ON DELETE CASCADE,
        unit_id TEXT NOT NULL,
        input_hash TEXT NOT NULL,
        chunk_ids_json TEXT NOT NULL,
        result_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (run_id, unit_id)
    );

    CREATE TABLE IF NOT EXISTS persona_evaluations (
        evaluation_id TEXT PRIMARY KEY,
        persona_id TEXT NOT NULL REFERENCES persona_specs(persona_id) ON DELETE CASCADE,
        evaluation_type TEXT NOT NULL,
        score INTEGER,
        result_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS persona_profiles (
        profile_id TEXT PRIMARY KEY,
        kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        name_key TEXT NOT NULL,
        mode TEXT NOT NULL CHECK (mode IN ('person', 'topic')),
        current_persona_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (kb_id, name_key, mode)
    );

    ALTER TABLE persona_specs ADD COLUMN profile_id TEXT;
    ALTER TABLE persona_specs ADD COLUMN version_number INTEGER NOT NULL DEFAULT 1;
    ALTER TABLE persona_specs ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 1;
    ALTER TABLE persona_specs ADD COLUMN runtime_spec_json TEXT;

    ALTER TABLE distillation_runs ADD COLUMN target_doc_ids_json TEXT NOT NULL DEFAULT '[]';
    ALTER TABLE distillation_runs ADD COLUMN control_doc_ids_json TEXT NOT NULL DEFAULT '[]';
    ALTER TABLE distillation_runs ADD COLUMN domain TEXT NOT NULL DEFAULT '';

    CREATE TABLE IF NOT EXISTS distillation_stage_results (
        run_id TEXT NOT NULL REFERENCES distillation_runs(run_id) ON DELETE CASCADE,
        stage TEXT NOT NULL,
        item_id TEXT NOT NULL,
        input_hash TEXT NOT NULL,
        result_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (run_id, stage, item_id)
    );

    CREATE INDEX IF NOT EXISTS idx_distillation_stage_compatible
        ON distillation_stage_results(stage, input_hash, item_id, updated_at);

    CREATE TABLE IF NOT EXISTS app_settings (
        setting_key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    INSERT OR IGNORE INTO persona_profiles(
        profile_id, kb_id, name, name_key, mode, current_persona_id, created_at, updated_at
    )
    SELECT
        'profile_' || lower(hex(randomblob(16))),
        kb_id,
        name,
        lower(trim(name)),
        mode,
        NULL,
        min(created_at),
        max(updated_at)
    FROM persona_specs
    GROUP BY kb_id, lower(trim(name)), mode;

    UPDATE persona_specs
    SET profile_id = (
        SELECT profile_id FROM persona_profiles f
        WHERE f.kb_id = persona_specs.kb_id
          AND f.name_key = lower(trim(persona_specs.name))
          AND f.mode = persona_specs.mode
    )
    WHERE profile_id IS NULL;

    UPDATE persona_specs AS target
    SET version_number = (
        SELECT count(*) FROM persona_specs AS earlier
        WHERE earlier.profile_id = target.profile_id
          AND (
              earlier.created_at < target.created_at
              OR (
                  earlier.created_at = target.created_at
                  AND earlier.persona_id <= target.persona_id
              )
          )
    );

    UPDATE persona_profiles
    SET current_persona_id = (
        SELECT persona_id FROM persona_specs p
        WHERE p.profile_id = persona_profiles.profile_id AND p.status = 'ready'
        ORDER BY p.version_number DESC LIMIT 1
    );

    CREATE INDEX IF NOT EXISTS idx_persona_specs_profile_version
        ON persona_specs(profile_id, version_number DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS generation_evaluations (
        evaluation_id TEXT PRIMARY KEY,
        kb_id TEXT,
        task_description TEXT,
        pipeline_run_id TEXT,
        evaluation_type TEXT NOT NULL,
        score REAL NOT NULL,
        pass_threshold REAL,
        passed INTEGER NOT NULL DEFAULT 0,
        result_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_gen_eval_type
        ON generation_evaluations(evaluation_type, created_at DESC);
    """,
    """
    CREATE TABLE IF NOT EXISTS projects (
        project_id TEXT PRIMARY KEY,
        kb_id TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS writing_tasks (
        task_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
        kb_id TEXT NOT NULL,
        persona_id TEXT NOT NULL,
        title TEXT NOT NULL,
        task_description TEXT NOT NULL,
        domain TEXT NOT NULL DEFAULT '',
        citation_style TEXT NOT NULL DEFAULT 'gb-t-7714',
        selected_doc_ids_json TEXT NOT NULL,
        allowed_persona_doc_ids_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'pending',
        state_json TEXT,
        edited_draft_text TEXT,
        edited_outline_text TEXT,
        evaluation_json TEXT,
        error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_writing_tasks_project_updated
        ON writing_tasks(project_id, updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_writing_tasks_status
        ON writing_tasks(status, updated_at DESC);
    """,
    """
    ALTER TABLE writing_tasks
        ADD COLUMN generation_options_json TEXT NOT NULL DEFAULT '{}';
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_conversations (
        conversation_id TEXT PRIMARY KEY,
        kb_id TEXT NOT NULL,
        persona_id TEXT,
        persona_name TEXT NOT NULL,
        persona_version INTEGER NOT NULL,
        title TEXT NOT NULL,
        knowledge_mode TEXT NOT NULL DEFAULT 'none'
            CHECK (knowledge_mode IN ('none', 'all', 'selected')),
        selected_doc_ids_json TEXT NOT NULL DEFAULT '[]',
        allowed_persona_doc_ids_json TEXT NOT NULL DEFAULT '[]',
        target_persona_doc_ids_json TEXT NOT NULL DEFAULT '[]',
        runtime_persona_json TEXT NOT NULL,
        summary_text TEXT NOT NULL DEFAULT '',
        summary_through_sequence INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_chat_conversations_updated
        ON chat_conversations(updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_chat_conversations_persona
        ON chat_conversations(persona_id, updated_at DESC);

    CREATE TABLE IF NOT EXISTS chat_messages (
        message_id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL
            REFERENCES chat_conversations(conversation_id) ON DELETE CASCADE,
        sequence INTEGER NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'complete'
            CHECK (status IN ('complete', 'interrupted', 'error')),
        sources_json TEXT NOT NULL DEFAULT '[]',
        verification_json TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (conversation_id, sequence)
    );

    CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation
        ON chat_messages(conversation_id, sequence);
    """,
)


def apply_migrations(connection: sqlite3.Connection) -> None:
    """Apply each migration exactly once inside a transaction."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    applied = {
        row[0] for row in connection.execute("SELECT version FROM schema_migrations").fetchall()
    }
    for version, script in enumerate(MIGRATIONS, start=1):
        if version in applied:
            continue
        connection.executescript(script)
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, datetime('now'))",
            (version,),
        )
    connection.commit()
