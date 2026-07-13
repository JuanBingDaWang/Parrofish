"""Application dependency assembly and controlled resource shutdown."""

from __future__ import annotations

from dataclasses import dataclass

from writing_factory.config import Settings, load_settings
from writing_factory.config.logging import configure_logging
from writing_factory.distill.extraction import PersonaMapExtractor
from writing_factory.distill.fidelity import FidelityService, PersonaFidelityEvaluator
from writing_factory.distill.service import DistillationService
from writing_factory.distill.sources import SourceCorpusBuilder
from writing_factory.distill.synthesis import PersonaSynthesizer
from writing_factory.kb.chunking import StructureChunker
from writing_factory.kb.files import ManagedFileStore
from writing_factory.kb.ingestion import IngestionService
from writing_factory.kb.mineru_parser import DocumentParserRouter, MinerUDocumentParser
from writing_factory.kb.retrieval import DenseRetriever, SparseRetriever
from writing_factory.llm import MinerUClient, SiliconFlowClient
from writing_factory.store import Database
from writing_factory.store.bm25_index import BM25Index
from writing_factory.store.kb_repository import KnowledgeBaseRepository
from writing_factory.store.persona_repository import PersonaRepository
from writing_factory.store.vector_index import LanceVectorIndex


@dataclass(slots=True, weakref_slot=True)
class ApplicationContext:
    """Own long-lived services and expose one deterministic shutdown point."""

    settings: Settings
    database: Database
    siliconflow: SiliconFlowClient
    mineru: MinerUClient
    repository: KnowledgeBaseRepository
    ingestion: IngestionService
    dense_retriever: DenseRetriever
    sparse_retriever: SparseRetriever
    persona_repository: PersonaRepository
    distillation: DistillationService
    fidelity: FidelityService
    default_kb_id: str

    def close(self) -> None:
        """Close all external service connection pools."""

        self.siliconflow.close()
        self.mineru.close()


def build_application(settings: Settings | None = None) -> ApplicationContext:
    """Build the application from centralized settings."""

    resolved = settings or load_settings()
    resolved.ensure_runtime_directories()
    configure_logging(
        resolved.log_dir,
        (
            resolved.siliconflow_api_key.get_secret_value(),
            resolved.mineru_api_token.get_secret_value(),
        ),
    )
    database = Database(resolved.database_path)
    database.initialize()
    siliconflow = SiliconFlowClient(resolved, database)
    mineru = MinerUClient(resolved, database)
    repository = KnowledgeBaseRepository(database)
    default_kb_id = repository.ensure_default()
    vectors = LanceVectorIndex(resolved.lancedb_path)
    bm25 = BM25Index(repository)
    persona_repository = PersonaRepository(database)
    ingestion = IngestionService(
        resolved,
        repository,
        ManagedFileStore(resolved.managed_documents_dir),
        DocumentParserRouter(MinerUDocumentParser(resolved, mineru)),
        StructureChunker(),
        siliconflow,
        vectors,
        bm25,
    )
    return ApplicationContext(
        settings=resolved,
        database=database,
        siliconflow=siliconflow,
        mineru=mineru,
        repository=repository,
        ingestion=ingestion,
        dense_retriever=DenseRetriever(repository, vectors, siliconflow),
        sparse_retriever=SparseRetriever(bm25),
        persona_repository=persona_repository,
        distillation=DistillationService(
            persona_repository,
            SourceCorpusBuilder(repository),
            PersonaMapExtractor(
                siliconflow,
                output_language=resolved.distillation_output_language,
            ),
            PersonaSynthesizer(
                siliconflow,
                output_language=resolved.distillation_output_language,
            ),
            map_concurrency=resolved.distillation_map_concurrency,
            output_language=resolved.distillation_output_language,
        ),
        fidelity=FidelityService(
            persona_repository,
            PersonaFidelityEvaluator(siliconflow),
        ),
        default_kb_id=default_kb_id,
    )
