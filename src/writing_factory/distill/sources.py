"""Build bounded, non-overlapping map units from ready KB parent chunks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from writing_factory.distill.models import SourceInfo, SourceSegment, SourceUnit
from writing_factory.store.kb_repository import KnowledgeBaseRepository


@dataclass(frozen=True, slots=True)
class SourceCorpus:
    """Stable source units, provenance, and input hash for one distillation."""

    units: tuple[SourceUnit, ...]
    source_info: tuple[SourceInfo, ...]
    source_hash: str


class SourceCorpusBuilder:
    """Group adjacent parent chunks per document without duplicating source text."""

    def __init__(
        self,
        repository: KnowledgeBaseRepository,
        *,
        max_unit_characters: int = 12_000,
    ) -> None:
        self.repository = repository
        self.max_unit_characters = max_unit_characters

    def build(self, kb_id: str, *, doc_ids: set[str] | None = None) -> SourceCorpus:
        """Build deterministic map units from currently ready documents."""

        documents = self.repository.source_documents(kb_id, doc_ids=doc_ids)
        chunks = self.repository.ready_parent_chunks(kb_id, doc_ids=doc_ids)
        if not documents or not chunks:
            raise ValueError("Distillation requires ready source documents and parent chunks")
        document_by_id = {str(item["doc_id"]): item for item in documents}
        segments = [
            SourceSegment(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                document_title=str(document_by_id[chunk.doc_id]["title"]),
                filename=str(document_by_id[chunk.doc_id]["filename"]),
                text=chunk.text,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_heading=chunk.section_heading,
            )
            for chunk in chunks
        ]
        units = self._group_units(segments)
        source_info = tuple(
            SourceInfo(
                doc_id=str(item["doc_id"]),
                title=str(item["title"]),
                filename=str(item["filename"]),
                chunk_count=int(item["chunk_count"]),
            )
            for item in documents
        )
        digest = hashlib.sha256()
        digest.update(f"source-corpus-v2|{self.max_unit_characters}".encode("ascii"))
        for segment in segments:
            digest.update(segment.chunk_id.encode("utf-8"))
            digest.update(hashlib.sha256(segment.text.encode("utf-8")).digest())
        return SourceCorpus(
            units=tuple(units),
            source_info=source_info,
            source_hash=digest.hexdigest(),
        )

    def _group_units(self, segments: list[SourceSegment]) -> list[SourceUnit]:
        units: list[SourceUnit] = []
        current: list[SourceSegment] = []
        current_characters = 0
        current_doc_id: str | None = None
        for segment in segments:
            starts_new = current and (
                segment.doc_id != current_doc_id
                or current_characters + len(segment.text) > self.max_unit_characters
            )
            if starts_new:
                units.append(self._make_unit(current))
                current = []
                current_characters = 0
            current.append(segment)
            current_characters += len(segment.text)
            current_doc_id = segment.doc_id
        if current:
            units.append(self._make_unit(current))
        return units

    @staticmethod
    def _make_unit(segments: list[SourceSegment]) -> SourceUnit:
        key = "|".join(segment.chunk_id for segment in segments)
        unit_id = f"unit_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:24]}"
        return SourceUnit(unit_id=unit_id, segments=segments)
