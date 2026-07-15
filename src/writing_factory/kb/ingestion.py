"""Recoverable document ingestion from managed source through both indexes."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from writing_factory.config import Settings
from writing_factory.kb.chunking import StructureChunker
from writing_factory.kb.files import ManagedFileStore
from writing_factory.kb.mineru_parser import DocumentParserRouter
from writing_factory.kb.models import Bibliography, IngestResult
from writing_factory.llm import SiliconFlowClient
from writing_factory.store.bm25_index import BM25Index
from writing_factory.store.kb_repository import KnowledgeBaseRepository
from writing_factory.store.vector_index import LanceVectorIndex

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], None]


def _no_progress(_percent: int, _message: str) -> None:
    pass


def _no_cancellation() -> None:
    pass


@dataclass(frozen=True, slots=True)
class DocumentDeletionResult:
    """知识库批量删除结果；清理警告不影响文档已经下线。"""

    removed_count: int
    orphaned_count: int
    cleanup_failures: int = 0


class IngestionService:
    """Coordinate one idempotent, observable, and recoverable KB import."""

    def __init__(
        self,
        settings: Settings,
        repository: KnowledgeBaseRepository,
        files: ManagedFileStore,
        parsers: DocumentParserRouter,
        chunker: StructureChunker,
        siliconflow: SiliconFlowClient,
        vectors: LanceVectorIndex,
        bm25: BM25Index,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.files = files
        self.parsers = parsers
        self.chunker = chunker
        self.siliconflow = siliconflow
        self.vectors = vectors
        self.bm25 = bm25

    def ingest(
        self,
        kb_id: str,
        source_path: Path,
        *,
        bibliography: Bibliography | None = None,
        progress: ProgressCallback = _no_progress,
        check_cancelled: CancellationCheck = _no_cancellation,
    ) -> IngestResult:
        """Ingest one file and publish it only after both indexes succeed."""

        source = source_path.expanduser().resolve()
        job_id = self.repository.create_job(kb_id, source)
        try:
            progress(2, "复制源文件")
            check_cancelled()
            managed = self.files.import_file(source)
            existing = self.repository.ready_document(kb_id, managed.sha256)
            if existing is not None and self.vectors.has_document(existing[0]):
                self.repository.update_job(job_id, "ready", document_id=existing[0])
                self.bm25.rebuild(kb_id)
                progress(100, "文档已存在")
                return IngestResult(
                    job_id=job_id,
                    kb_id=kb_id,
                    doc_id=existing[0],
                    child_chunk_count=existing[1],
                    reused_document=True,
                )

            self.repository.update_job(job_id, "parsing")
            parsed = self.parsers.parse(
                managed,
                progress=progress,
                check_cancelled=check_cancelled,
            )
            check_cancelled()
            self.repository.update_job(job_id, "chunking")
            progress(52, "结构化切片")
            chunked = self.chunker.chunk(managed.doc_id, parsed)
            children = [chunk for chunk in chunked.chunks if chunk.chunk_kind == "child"]
            if not children:
                raise ValueError("Document parsing produced no indexable chunks")

            self.repository.update_job(job_id, "indexing")
            vectors: list[list[float]] = []
            for start in range(0, len(children), self.settings.embedding_batch_size):
                check_cancelled()
                batch = children[start : start + self.settings.embedding_batch_size]
                embedded = self.siliconflow.embeddings([chunk.text for chunk in batch])
                if len(embedded.vectors) != len(batch):
                    raise ValueError("Embedding result count does not match chunk batch")
                vectors.extend(embedded.vectors)
                completed = start + len(batch)
                progress(58 + round(24 * completed / len(children)), "生成稠密向量")

            source_bib = bibliography or Bibliography(title=source.stem)
            self.repository.save_document_and_chunks(
                kb_id=kb_id,
                job_id=job_id,
                managed=managed,
                bibliography=source_bib,
                parsed=parsed,
                chunked=chunked,
            )
            check_cancelled()
            progress(86, "写入 LanceDB")
            self.vectors.replace_document(managed.doc_id, children, vectors)
            self.repository.mark_ready(kb_id, managed.doc_id, job_id)
            progress(94, "重建 BM25 索引")
            self.bm25.rebuild(kb_id)
            progress(100, "入库完成")
            return IngestResult(
                job_id=job_id,
                kb_id=kb_id,
                doc_id=managed.doc_id,
                child_chunk_count=len(children),
            )
        except Exception as exc:
            logger.exception("Ingestion failed: %s", type(exc).__name__)
            self.repository.mark_failed(job_id, type(exc).__name__)
            raise

    def delete_documents(
        self,
        kb_id: str,
        doc_ids: set[str],
        *,
        progress: ProgressCallback = _no_progress,
        check_cancelled: CancellationCheck = _no_cancellation,
    ) -> DocumentDeletionResult:
        """先让文档退出检索，再尽力清理只属于它的本地派生文件。"""

        check_cancelled()
        progress(10, "从知识库移除")
        removed = self.repository.remove_documents(kb_id, doc_ids)
        orphaned = [item for item in removed if item.orphaned]
        cleanup_failures = 0
        total = max(1, len(orphaned))
        for index, item in enumerate(orphaned, start=1):
            try:
                self.vectors.delete_document(item.doc_id)
                self.files.delete_file(item.managed_path)
                artifact_dir = (self.settings.mineru_artifacts_dir / item.sha256).resolve()
                if artifact_dir.parent != self.settings.mineru_artifacts_dir.resolve():
                    raise ValueError("MinerU artifact path escaped its configured directory")
                if artifact_dir.exists():
                    shutil.rmtree(artifact_dir)
            except Exception:
                cleanup_failures += 1
                logger.exception("Document derivative cleanup failed for %s", item.doc_id)
            progress(15 + round(70 * index / total), "清理本地索引与缓存")
        progress(90, "重建稀疏索引")
        self.bm25.rebuild(kb_id)
        progress(100, "删除完成")
        return DocumentDeletionResult(
            removed_count=len(removed),
            orphaned_count=len(orphaned),
            cleanup_failures=cleanup_failures,
        )
