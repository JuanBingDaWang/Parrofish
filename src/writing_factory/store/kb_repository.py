"""SQLite repositories for knowledge bases, documents, chunks, and ingest jobs."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from writing_factory.kb.models import (
    Bibliography,
    Chunk,
    ChunkedDocument,
    ManagedDocument,
    MetadataFilter,
    ParsedDocument,
)
from writing_factory.store.database import Database, utc_now


@dataclass(frozen=True, slots=True)
class RemovedDocument:
    """本次从知识库移除的文档及其本地清理信息。"""

    doc_id: str
    sha256: str
    managed_path: Path
    orphaned: bool


class KnowledgeBaseRepository:
    """Persist KB membership and ingestion state with SQLite as source of truth."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def ensure_default(self) -> str:
        """Create and return the stable default knowledge base."""

        return self.create("默认知识库", kb_id="kb_default")

    def create(self, name: str, *, kb_id: str | None = None) -> str:
        """Create a named KB or return the existing record with that identifier."""

        identifier = kb_id or f"kb_{uuid.uuid4().hex}"
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO knowledge_bases(kb_id, name, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(kb_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (identifier, name, now, now),
            )
        return identifier

    def list_knowledge_bases(self) -> list[dict[str, object]]:
        """Return KBs with ready document counts for the desktop UI."""

        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT kb.kb_id, kb.name, kb.description,
                       COUNT(CASE WHEN kbd.status = 'ready' THEN 1 END) AS document_count
                FROM knowledge_bases kb
                LEFT JOIN knowledge_base_documents kbd ON kbd.kb_id = kb.kb_id
                GROUP BY kb.kb_id
                ORDER BY kb.created_at
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def create_job(self, kb_id: str, source_path: Path) -> str:
        """Persist a pending job before any file or network operation starts."""

        job_id = f"job_{uuid.uuid4().hex}"
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO ingest_jobs(
                    job_id, kb_id, source_path, status, created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', ?, ?)
                """,
                (job_id, kb_id, str(source_path), now, now),
            )
        return job_id

    def update_job(
        self,
        job_id: str,
        status: str,
        *,
        document_id: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Move a job forward or mark a sanitized terminal failure."""

        with self.database.connection() as connection:
            connection.execute(
                """
                UPDATE ingest_jobs
                SET status = ?, document_id = COALESCE(?, document_id),
                    error_message = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (status, document_id, error_message, utc_now(), job_id),
            )

    def ready_document(self, kb_id: str, sha256: str) -> tuple[str, int] | None:
        """Return an already indexed document and child count for idempotent import."""

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT d.doc_id, COUNT(c.chunk_id) AS child_count
                FROM documents d
                JOIN knowledge_base_documents kbd ON kbd.doc_id = d.doc_id
                LEFT JOIN chunks c ON c.doc_id = d.doc_id AND c.chunk_kind = 'child'
                WHERE kbd.kb_id = ? AND d.sha256 = ? AND kbd.status = 'ready'
                GROUP BY d.doc_id
                """,
                (kb_id, sha256),
            ).fetchone()
        if row is None:
            return None
        return row["doc_id"], row["child_count"]

    def save_document_and_chunks(
        self,
        *,
        kb_id: str,
        job_id: str,
        managed: ManagedDocument,
        bibliography: Bibliography,
        parsed: ParsedDocument,
        chunked: ChunkedDocument,
    ) -> None:
        """Atomically replace canonical metadata and chunks before index publication."""

        now = utc_now()
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO documents(
                    doc_id, sha256, filename, format, source_path, managed_path,
                    bib_json, parser_name, parser_version, canonical_text,
                    ingest_date, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    managed_path = excluded.managed_path,
                    bib_json = excluded.bib_json,
                    parser_name = excluded.parser_name,
                    parser_version = excluded.parser_version,
                    canonical_text = excluded.canonical_text,
                    ingest_date = excluded.ingest_date
                """,
                (
                    managed.doc_id,
                    managed.sha256,
                    managed.filename,
                    managed.format,
                    str(managed.source_path),
                    str(managed.managed_path),
                    bibliography.model_dump_json(),
                    parsed.parser_name,
                    parsed.parser_version,
                    chunked.canonical_text,
                    now,
                    now,
                ),
            )
            connection.execute("DELETE FROM chunks WHERE doc_id = ?", (managed.doc_id,))
            for chunk in sorted(
                chunked.chunks, key=lambda item: (item.chunk_kind == "child", item.chunk_index)
            ):
                connection.execute(
                    """
                    INSERT INTO chunks(
                        chunk_id, doc_id, text, page_start, page_end, section_heading,
                        chunk_index, char_start, char_end, parent_id, chunk_kind, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.text,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.section_heading,
                        chunk.chunk_index,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.parent_id,
                        chunk.chunk_kind,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                    ),
                )
            connection.execute(
                """
                INSERT INTO knowledge_base_documents(
                    kb_id, doc_id, added_at, status, last_job_id
                ) VALUES (?, ?, ?, 'indexing', ?)
                ON CONFLICT(kb_id, doc_id) DO UPDATE SET
                    status = 'indexing', last_job_id = excluded.last_job_id
                """,
                (kb_id, managed.doc_id, now, job_id),
            )
        self.update_job(job_id, "indexing", document_id=managed.doc_id)

    def mark_ready(self, kb_id: str, doc_id: str, job_id: str) -> None:
        """Publish a fully indexed document to retrieval."""

        with self.database.connection() as connection:
            connection.execute(
                """
                UPDATE knowledge_base_documents
                SET status = 'ready', last_job_id = ?
                WHERE kb_id = ? AND doc_id = ?
                """,
                (job_id, kb_id, doc_id),
            )
        self.update_job(job_id, "ready", document_id=doc_id)

    def mark_failed(self, job_id: str, error_type: str) -> None:
        """Record a recoverable failure without storing provider or document payloads."""

        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT kb_id, document_id FROM ingest_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is not None and row["document_id"]:
                connection.execute(
                    """
                    UPDATE knowledge_base_documents SET status = 'failed'
                    WHERE kb_id = ? AND doc_id = ? AND last_job_id = ?
                    """,
                    (row["kb_id"], row["document_id"], job_id),
                )
        self.update_job(job_id, "failed", error_message=error_type)

    def ready_child_chunks(
        self, kb_id: str, *, filters: MetadataFilter | None = None
    ) -> list[Chunk]:
        """Load the authoritative child corpus after applying every metadata filter."""

        if filters is not None:
            constrained_sets = (
                filters.doc_ids,
                filters.authors,
                filters.years,
                filters.formats,
                filters.section_headings,
                filters.chunk_kinds,
            )
            if any(values == set() for values in constrained_sets):
                return []
            if filters.chunk_kinds is not None and "child" not in filters.chunk_kinds:
                return []

        parameters: list[object] = [kb_id]
        extra: list[str] = []
        if filters is not None:
            if filters.doc_ids:
                placeholders = ",".join("?" for _ in filters.doc_ids)
                extra.append(f"AND c.doc_id IN ({placeholders})")
                parameters.extend(sorted(filters.doc_ids))
            if filters.formats:
                placeholders = ",".join("?" for _ in filters.formats)
                extra.append(f"AND d.format IN ({placeholders})")
                parameters.extend(sorted(filters.formats))
            if filters.section_headings:
                placeholders = ",".join("?" for _ in filters.section_headings)
                extra.append(f"AND c.section_heading IN ({placeholders})")
                parameters.extend(sorted(filters.section_headings))
            if filters.authors:
                placeholders = ",".join("?" for _ in filters.authors)
                extra.append(f"AND json_extract(d.bib_json, '$.author') IN ({placeholders})")
                parameters.extend(sorted(filters.authors))
            if filters.years:
                placeholders = ",".join("?" for _ in filters.years)
                extra.append(
                    f"AND CAST(json_extract(d.bib_json, '$.year') AS INTEGER) IN ({placeholders})"
                )
                parameters.extend(sorted(filters.years))
        with self.database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT c.* FROM chunks c
                JOIN knowledge_base_documents kbd ON kbd.doc_id = c.doc_id
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE kbd.kb_id = ? AND kbd.status = 'ready' AND c.chunk_kind = 'child'
                {" ".join(extra)}
                ORDER BY c.doc_id, c.chunk_index
                """,
                parameters,
            ).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def retrieval_fingerprint(self, kb_id: str) -> str:
        """Hash the published corpus and filterable metadata for cache versioning."""

        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT c.chunk_id, c.parent_id, c.section_heading,
                       d.doc_id, d.format, d.bib_json, d.ingest_date
                FROM chunks c
                JOIN knowledge_base_documents kbd ON kbd.doc_id = c.doc_id
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE kbd.kb_id = ? AND kbd.status = 'ready' AND c.chunk_kind = 'child'
                ORDER BY c.doc_id, c.chunk_index
                """,
                (kb_id,),
            ).fetchall()
        digest = hashlib.sha256()
        for row in rows:
            for key in (
                "chunk_id",
                "parent_id",
                "section_heading",
                "doc_id",
                "format",
                "bib_json",
                "ingest_date",
            ):
                digest.update(str(row[key] or "").encode("utf-8"))
                digest.update(b"\x00")
        return digest.hexdigest()

    def ready_child_chunks_by_ids(self, kb_id: str, chunk_ids: set[str]) -> list[Chunk]:
        """Resolve evidence anchors only within one published knowledge base."""

        identifiers = sorted(chunk_ids)
        if not identifiers:
            return []
        placeholders = ",".join("?" for _identifier in identifiers)
        with self.database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT c.* FROM chunks c
                JOIN knowledge_base_documents kbd ON kbd.doc_id = c.doc_id
                WHERE kbd.kb_id = ? AND kbd.status = 'ready'
                  AND c.chunk_kind = 'child' AND c.chunk_id IN ({placeholders})
                ORDER BY c.doc_id, c.chunk_index
                """,
                [kb_id, *identifiers],
            ).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def parent_chunk(self, parent_id: str) -> Chunk | None:
        """Return one parent chunk by identifier for parent-document expansion."""

        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT c.* FROM chunks c WHERE c.chunk_id = ? AND c.chunk_kind = 'parent'",
                (parent_id,),
            ).fetchone()
        return None if row is None else self._chunk_from_row(row)

    def ready_parent_chunks(self, kb_id: str, *, doc_ids: set[str] | None = None) -> list[Chunk]:
        """Load non-overlapping parent chunks used as distillation source units."""

        parameters: list[object] = [kb_id]
        document_filter = ""
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            document_filter = f" AND c.doc_id IN ({placeholders})"
            parameters.extend(sorted(doc_ids))
        with self.database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT c.* FROM chunks c
                JOIN knowledge_base_documents kbd ON kbd.doc_id = c.doc_id
                WHERE kbd.kb_id = ? AND kbd.status = 'ready'
                  AND c.chunk_kind = 'parent' {document_filter}
                ORDER BY c.doc_id, c.chunk_index
                """,
                parameters,
            ).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def source_documents(
        self, kb_id: str, *, doc_ids: set[str] | None = None
    ) -> list[dict[str, object]]:
        """Return ready document metadata for PersonaSpec provenance."""

        parameters: list[object] = [kb_id]
        document_filter = ""
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            document_filter = f" AND d.doc_id IN ({placeholders})"
            parameters.extend(sorted(doc_ids))
        with self.database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT d.doc_id, d.filename, d.bib_json,
                       COUNT(c.chunk_id) AS chunk_count
                FROM documents d
                JOIN knowledge_base_documents kbd ON kbd.doc_id = d.doc_id
                JOIN chunks c ON c.doc_id = d.doc_id AND c.chunk_kind = 'parent'
                WHERE kbd.kb_id = ? AND kbd.status = 'ready' {document_filter}
                GROUP BY d.doc_id
                ORDER BY d.ingest_date, d.doc_id
                """,
                parameters,
            ).fetchall()
        documents: list[dict[str, object]] = []
        for row in rows:
            bibliography = json.loads(row["bib_json"])
            documents.append(
                {
                    "doc_id": row["doc_id"],
                    "filename": row["filename"],
                    "title": bibliography.get("title") or Path(row["filename"]).stem,
                    "chunk_count": row["chunk_count"],
                }
            )
        return documents

    def list_documents(self, kb_id: str) -> list[dict[str, object]]:
        """Return document metadata and status for the minimal KB interface."""

        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT d.doc_id, d.filename, d.format, d.bib_json, d.ingest_date,
                       kbd.status,
                       COUNT(CASE WHEN c.chunk_kind = 'child' THEN 1 END) AS chunk_count
                FROM knowledge_base_documents kbd
                JOIN documents d ON d.doc_id = kbd.doc_id
                LEFT JOIN chunks c ON c.doc_id = d.doc_id
                WHERE kbd.kb_id = ?
                GROUP BY d.doc_id
                ORDER BY d.ingest_date DESC
                """,
                (kb_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def remove_documents(self, kb_id: str, doc_ids: set[str]) -> list[RemovedDocument]:
        """移除知识库成员；没有其他知识库引用时一并删除规范文档和切片。"""

        identifiers = sorted(doc_ids)
        if not identifiers:
            return []
        placeholders = ",".join("?" for _ in identifiers)
        with self.database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT d.doc_id, d.sha256, d.managed_path
                FROM documents d
                JOIN knowledge_base_documents kbd ON kbd.doc_id = d.doc_id
                WHERE kbd.kb_id = ? AND d.doc_id IN ({placeholders})
                ORDER BY d.doc_id
                """,
                [kb_id, *identifiers],
            ).fetchall()
            if not rows:
                return []
            removed_ids = [str(row["doc_id"]) for row in rows]
            removed_placeholders = ",".join("?" for _ in removed_ids)
            connection.execute(
                f"""
                DELETE FROM knowledge_base_documents
                WHERE kb_id = ? AND doc_id IN ({removed_placeholders})
                """,
                [kb_id, *removed_ids],
            )
            remaining = {
                str(row["doc_id"])
                for row in connection.execute(
                    f"""
                    SELECT DISTINCT doc_id FROM knowledge_base_documents
                    WHERE doc_id IN ({removed_placeholders})
                    """,
                    removed_ids,
                ).fetchall()
            }
            orphaned_ids = [doc_id for doc_id in removed_ids if doc_id not in remaining]
            if orphaned_ids:
                orphaned_placeholders = ",".join("?" for _ in orphaned_ids)
                connection.execute(
                    f"DELETE FROM documents WHERE doc_id IN ({orphaned_placeholders})",
                    orphaned_ids,
                )
            connection.execute(
                "UPDATE knowledge_bases SET updated_at = ? WHERE kb_id = ?",
                (utc_now(), kb_id),
            )
        return [
            RemovedDocument(
                doc_id=str(row["doc_id"]),
                sha256=str(row["sha256"]),
                managed_path=Path(str(row["managed_path"])),
                orphaned=str(row["doc_id"]) not in remaining,
            )
            for row in rows
        ]

    @staticmethod
    def _chunk_from_row(row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            text=row["text"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            section_heading=row["section_heading"],
            chunk_index=row["chunk_index"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            parent_id=row["parent_id"],
            chunk_kind=row["chunk_kind"],
            metadata=json.loads(row["metadata_json"]),
        )
