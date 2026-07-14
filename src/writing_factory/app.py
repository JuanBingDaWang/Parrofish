"""Application dependency assembly and controlled resource shutdown."""

from __future__ import annotations

from dataclasses import dataclass

from writing_factory.config import Settings, load_settings
from writing_factory.config.logging import configure_logging, shutdown_logging
from writing_factory.distill.academic_pipeline import AcademicDistillationEngine
from writing_factory.distill.extraction import PersonaMapExtractor
from writing_factory.distill.fidelity import FidelityService, PersonaFidelityEvaluator
from writing_factory.distill.service import DistillationService
from writing_factory.distill.sources import SourceCorpusBuilder
from writing_factory.distill.synthesis import PersonaSynthesizer
from writing_factory.kb.chunking import StructureChunker
from writing_factory.kb.files import ManagedFileStore
from writing_factory.kb.ingestion import IngestionService
from writing_factory.kb.mineru_parser import DocumentParserRouter, MinerUDocumentParser
from writing_factory.kb.retrieval import (
    DenseRetriever,
    HybridRetriever,
    SparseRetriever,
)
from writing_factory.llm import MinerUClient, SiliconFlowClient
from writing_factory.llm.common import DynamicConcurrencyGate
from writing_factory.store import Database, ProjectRepository, RuntimeSettingsRepository
from writing_factory.store.bm25_index import BM25Index
from writing_factory.store.kb_repository import KnowledgeBaseRepository
from writing_factory.store.persona_repository import PersonaRepository
from writing_factory.store.vector_index import LanceVectorIndex


@dataclass(slots=True, weakref_slot=True)
class ApplicationContext:
    """Own long-lived services and expose one deterministic shutdown point."""

    settings: Settings
    database: Database
    runtime_settings: RuntimeSettingsRepository
    siliconflow_gate: DynamicConcurrencyGate
    siliconflow: SiliconFlowClient
    mineru: MinerUClient
    repository: KnowledgeBaseRepository
    ingestion: IngestionService
    dense_retriever: DenseRetriever
    sparse_retriever: SparseRetriever
    hybrid_retriever: HybridRetriever
    persona_repository: PersonaRepository
    project_repository: ProjectRepository
    distillation: DistillationService
    fidelity: FidelityService
    default_kb_id: str

    def close(self) -> None:
        """Close all external service connection pools."""

        self.siliconflow.close()
        self.mineru.close()
        shutdown_logging()

    def set_siliconflow_concurrency(self, value: int) -> None:
        """持久化并立即应用全局 SiliconFlow 并发上限。"""

        self.siliconflow_gate.set_limit(value)
        self.runtime_settings.set("siliconflow_max_concurrency", value)
        self.distillation.set_max_parallel_tasks(value)

    def get_framework_generation_timeout(self) -> int:
        """读取每次框架生成尝试包含网络重试在内的超时秒数。"""

        value = self.runtime_settings.get(
            "framework_generation_timeout_seconds",
            self.settings.framework_generation_timeout_seconds,
        )
        if isinstance(value, int) and 60 <= value <= 3600:
            return value
        return self.settings.framework_generation_timeout_seconds

    def set_framework_generation_timeout(self, value: int) -> None:
        """校验并持久化单次框架生成尝试的超时秒数。"""

        if not 60 <= value <= 3600:
            raise ValueError("框架生成超时上限必须在 60 至 3600 秒之间")
        self.runtime_settings.set("framework_generation_timeout_seconds", value)

    def get_retrieval_option(self, key: str, default: bool = True) -> bool:
        """读取检索增强开关（HyDE / 查询改写），默认开启以优先写作质量。"""

        value = self.runtime_settings.get(f"retrieval_{key}", default)
        return bool(value)

    def set_retrieval_option(self, key: str, enabled: bool) -> None:
        """持久化一个检索增强开关。"""

        self.runtime_settings.set(f"retrieval_{key}", bool(enabled))


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
    runtime_settings = RuntimeSettingsRepository(database)
    stored_concurrency = runtime_settings.get(
        "siliconflow_max_concurrency", resolved.siliconflow_max_concurrency
    )
    concurrency = (
        stored_concurrency
        if isinstance(stored_concurrency, int) and 1 <= stored_concurrency <= 8
        else resolved.siliconflow_max_concurrency
    )
    siliconflow_gate = DynamicConcurrencyGate(concurrency)
    siliconflow = SiliconFlowClient(resolved, database, siliconflow_gate)
    mineru = MinerUClient(resolved, database)
    repository = KnowledgeBaseRepository(database)
    default_kb_id = repository.ensure_default()
    vectors = LanceVectorIndex(resolved.lancedb_path)
    bm25 = BM25Index(repository)
    persona_repository = PersonaRepository(database)
    project_repository = ProjectRepository(database)
    project_repository.ensure_default(default_kb_id)
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
    academic_engine = AcademicDistillationEngine(
        siliconflow,
        persona_repository,
        parallelism=lambda: siliconflow_gate.limit,
    )
    distillation = DistillationService(
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
        map_concurrency=concurrency,
        output_language=resolved.distillation_output_language,
        academic_engine=academic_engine,
    )
    return ApplicationContext(
        settings=resolved,
        database=database,
        runtime_settings=runtime_settings,
        siliconflow_gate=siliconflow_gate,
        siliconflow=siliconflow,
        mineru=mineru,
        repository=repository,
        ingestion=ingestion,
        dense_retriever=DenseRetriever(repository, vectors, siliconflow),
        sparse_retriever=SparseRetriever(bm25),
        hybrid_retriever=HybridRetriever(repository, vectors, bm25, siliconflow),
        persona_repository=persona_repository,
        project_repository=project_repository,
        distillation=distillation,
        fidelity=FidelityService(
            persona_repository,
            PersonaFidelityEvaluator(siliconflow),
        ),
        default_kb_id=default_kb_id,
    )
