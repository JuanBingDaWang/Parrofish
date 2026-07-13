"""Embedded LanceDB child-chunk index with document-scoped replacement."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import lancedb
import pyarrow as pa

from writing_factory.kb.models import Chunk, SearchHit


class LanceVectorIndex:
    """Persist dense vectors while retaining stable chunk/source identifiers."""

    TABLE_NAME = "chunk_vectors"

    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._connection = lancedb.connect(path)
        self._lock = threading.Lock()

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
        allowed_doc_ids: set[str],
        limit: int,
    ) -> list[SearchHit]:
        """Search only documents SQLite has published as ready for this KB."""

        table = self._open_table()
        if table is None or not allowed_doc_ids or limit <= 0:
            return []
        allowed = ",".join(f"'{self._sql_string(doc_id)}'" for doc_id in sorted(allowed_doc_ids))
        rows = (
            table.search(list(query_vector), vector_column_name="vector")
            .where(f"doc_id IN ({allowed})", prefilter=True)
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
        if self.TABLE_NAME not in self._connection.list_tables().tables:
            return None
        return self._connection.open_table(self.TABLE_NAME)

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
