"""Embedded LanceDB child-chunk index with document-scoped replacement."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import lancedb
import pyarrow as pa

from writing_factory.kb.models import Chunk, SearchHit

logger = logging.getLogger(__name__)


class LanceVectorIndex:
    """Persist dense vectors while retaining stable chunk/source identifiers."""

    TABLE_NAME = "chunk_vectors"

    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._active_pointer = path / ".active-vector-table"
        self._connection = lancedb.connect(path)
        self._lock = threading.Lock()
        self._table_name, self._embedding_model = self._load_active_table()

    def replace_document(
        self,
        doc_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Replace one document's vectors without disturbing other documents."""

        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts differ")
        if any(chunk.chunk_kind != "child" for chunk in chunks):
            raise ValueError("Only child chunks may enter the vector index")
        with self._lock:
            table = self._open_table()
            if table is not None:
                table.delete(f"doc_id = '{self._sql_string(doc_id)}'")
            if not chunks:
                return
            arrow = self._to_arrow(chunks, vectors)
            if table is None:
                self._connection.create_table(self.TABLE_NAME, data=arrow)
            else:
                table.add(arrow)

    def has_document(self, doc_id: str) -> bool:
        """Return whether at least one vector exists for a document."""

        table = self._open_table()
        if table is None:
            return False
        rows = table.search().where(f"doc_id = '{self._sql_string(doc_id)}'").limit(1).to_list()
        return bool(rows)

    def rebuild(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        model_id: str | None = None,
    ) -> None:
        """Build a complete replacement table and switch only after it succeeds."""

        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts differ")
        if any(chunk.chunk_kind != "child" for chunk in chunks):
            raise ValueError("Only child chunks may enter the vector index")
        replacement = f"{self.TABLE_NAME}_v_{uuid4().hex}"
        with self._lock:
            previous = self._table_name
            if chunks:
                self._connection.create_table(replacement, data=self._to_arrow(chunks, vectors))
                pointer_tmp = self._active_pointer.with_suffix(f".{uuid4().hex}.tmp")
                pointer_tmp.write_text(
                    json.dumps({"table": replacement, "model": model_id}, ensure_ascii=False),
                    encoding="utf-8",
                )
                os.replace(pointer_tmp, self._active_pointer)
                self._table_name = replacement
                self._embedding_model = model_id
            else:
                self._active_pointer.unlink(missing_ok=True)
                self._table_name = self.TABLE_NAME
                self._embedding_model = None
                if previous in set(self._connection.list_tables().tables):
                    self._connection.drop_table(previous)
                return
            if previous != replacement and previous in set(self._connection.list_tables().tables):
                try:
                    self._connection.drop_table(previous)
                except Exception:
                    logger.warning("旧向量表暂未清理: %s", previous, exc_info=True)

    def delete_document(self, doc_id: str) -> None:
        """删除一个文档的全部稠密向量；表不存在时视为已经清理。"""

        with self._lock:
            table = self._open_table()
            if table is not None:
                table.delete(f"doc_id = '{self._sql_string(doc_id)}'")

    def search(
        self,
        query_vector: Sequence[float],
        *,
        allowed_doc_ids: set[str] | None = None,
        allowed_chunk_ids: set[str] | None = None,
        limit: int,
    ) -> list[SearchHit]:
        """Search only the published document/chunk scope resolved by SQLite."""

        table = self._open_table()
        if (
            table is None
            or limit <= 0
            or allowed_doc_ids == set()
            or allowed_chunk_ids == set()
            or (allowed_doc_ids is None and allowed_chunk_ids is None)
        ):
            return []
        predicates: list[str] = []
        if allowed_doc_ids is not None:
            allowed = ",".join(
                f"'{self._sql_string(doc_id)}'" for doc_id in sorted(allowed_doc_ids)
            )
            predicates.append(f"doc_id IN ({allowed})")
        if allowed_chunk_ids is not None:
            allowed = ",".join(
                f"'{self._sql_string(chunk_id)}'" for chunk_id in sorted(allowed_chunk_ids)
            )
            predicates.append(f"chunk_id IN ({allowed})")
        rows = (
            table.search(list(query_vector), vector_column_name="vector")
            .where(" AND ".join(predicates), prefilter=True)
            .limit(limit)
            .to_list()
        )
        hits: list[SearchHit] = []
        for rank, row in enumerate(rows, start=1):
            distance = float(row.get("_distance", 0.0))
            hits.append(
                SearchHit(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    score=1.0 / (1.0 + max(0.0, distance)),
                    rank=rank,
                    source="dense",
                    page_start=row.get("page_start"),
                    page_end=row.get("page_end"),
                    section_heading=row.get("section_heading"),
                    parent_id=row.get("parent_id"),
                )
            )
        return hits

    def _open_table(self):
        if self._table_name not in self._connection.list_tables().tables:
            return None
        return self._connection.open_table(self._table_name)

    @property
    def embedding_model(self) -> str | None:
        """Return the model bound to the atomically selected vector table."""

        return self._embedding_model

    def _load_active_table(self) -> tuple[str, str | None]:
        if self._active_pointer.is_file():
            raw = self._active_pointer.read_text(encoding="utf-8").strip()
            try:
                payload = json.loads(raw)
                selected = str(payload["table"])
                model = str(payload["model"]) if payload.get("model") else None
            except (ValueError, KeyError, TypeError):
                selected = raw
                model = None
            if selected in self._connection.list_tables().tables:
                return selected, model
        return self.TABLE_NAME, None

    @staticmethod
    def _to_arrow(chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> pa.Table:
        dimension = len(vectors[0])
        if dimension == 0 or any(len(vector) != dimension for vector in vectors):
            raise ValueError("Vectors must have one non-zero dimension")
        schema = pa.schema(
            [
                pa.field("chunk_id", pa.string(), nullable=False),
                pa.field("doc_id", pa.string(), nullable=False),
                pa.field("vector", pa.list_(pa.float32(), dimension), nullable=False),
                pa.field("text", pa.string(), nullable=False),
                pa.field("page_start", pa.int32()),
                pa.field("page_end", pa.int32()),
                pa.field("section_heading", pa.string()),
                pa.field("parent_id", pa.string()),
            ]
        )
        records = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "vector": [float(value) for value in vector],
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_heading": chunk.section_heading,
                "parent_id": chunk.parent_id,
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        return pa.Table.from_pylist(records, schema=schema)

    @staticmethod
    def _sql_string(value: str) -> str:
        return value.replace("'", "''")
