"""Application dependency assembly and controlled resource shutdown."""

from __future__ import annotations

from dataclasses import dataclass

from writing_factory.chat import AuthorChatService, ChatRepository
from writing_factory.config import CredentialStore, KeyringCredentialStore, Settings, load_settings
from writing_factory.config.logging import configure_logging, shutdown_logging
from writing_factory.distill.academic_pipeline import AcademicDistillationEngine
from writing_factory.distill.composition import CompositionDistiller
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
from writing_factory.llm.settings_service import ApplicationSettingsService
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
    settings_service: ApplicationSettingsService
    repository: KnowledgeBaseRepository
    ingestion: IngestionService
    dense_retriever: DenseRetriever
    sparse_retriever: SparseRetriever
    hybrid_retriever: HybridRetriever
    persona_repository: PersonaRepository
    project_repository: ProjectRepository
    chat_repository: ChatRepository
    author_chat: AuthorChatService
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

    def get_siliconflow_request_timeout(self) -> int:
        """读取所有 SiliconFlow 逻辑请求共享的单次超时秒数。"""

        legacy = self.runtime_settings.get(
            "framework_generation_timeout_seconds",
            self.settings.siliconflow_request_timeout_seconds,
        )
        value = self.runtime_settings.get("siliconflow_request_timeout_seconds", legacy)
        if isinstance(value, int) and 60 <= value <= 3600:
            return value
        return self.settings.siliconflow_request_timeout_seconds

    def set_siliconflow_request_timeout(self, value: int) -> None:
        """校验、持久化并立即应用全局 SiliconFlow 请求超时。"""

        if not 60 <= value <= 3600:
            raise ValueError("SiliconFlow 单次请求超时上限必须在 60 至 3600 秒之间")
        self.runtime_settings.set("siliconflow_request_timeout_seconds", value)
        self.siliconflow.set_request_timeout(value)

    def get_author_chat_recent_rounds(self) -> int:
        """Return the persisted number of verbatim recent chat rounds."""

        value = self.runtime_settings.get("author_chat_recent_rounds", 6)
        return value if isinstance(value, int) and 1 <= value <= 20 else 6

    def set_author_chat_recent_rounds(self, value: int) -> None:
        """Persist the rolling chat memory window."""

        if not 1 <= value <= 20:
            raise ValueError("作者对话最近轮数必须在 1 至 20 之间")
        self.runtime_settings.set("author_chat_recent_rounds", value)

    def get_retrieval_option(self, key: str, default: bool = True) -> bool:
        """读取检索增强开关（HyDE / 查询改写），默认开启以优先写作质量。"""

        value = self.runtime_settings.get(f"retrieval_{key}", default)
        return bool(value)

    def set_retrieval_option(self, key: str, enabled: bool) -> None:
        """持久化一个检索增强开关。"""

        self.runtime_settings.set(f"retrieval_{key}", bool(enabled))


def build_application(
    settings: Settings | None = None,
    *,
    credential_store: CredentialStore | None = None,
) -> ApplicationContext:
    """Build the application from centralized settings."""

    secret_store = credential_store or KeyringCredentialStore()
    resolved = settings or load_settings(credential_store=secret_store)
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
    legacy_timeout = runtime_settings.get(
        "framework_generation_timeout_seconds",
        resolved.siliconflow_request_timeout_seconds,
    )
    stored_timeout = runtime_settings.get("siliconflow_request_timeout_seconds", legacy_timeout)
    request_timeout = (
        stored_timeout
        if isinstance(stored_timeout, int) and 60 <= stored_timeout <= 3600
        else resolved.siliconflow_request_timeout_seconds
    )
    siliconflow_gate = DynamicConcurrencyGate(concurrency)
    siliconflow = SiliconFlowClient(
        resolved,
        database,
        siliconflow_gate,
        request_timeout_seconds=request_timeout,
    )
    mineru = MinerUClient(resolved, database)
    repository = KnowledgeBaseRepository(database)
    default_kb_id = repository.ensure_default()
    vectors = LanceVectorIndex(resolved.lancedb_path)
    bm25 = BM25Index(repository)
    dense_retriever = DenseRetriever(repository, vectors, siliconflow)
    sparse_retriever = SparseRetriever(bm25)
    hybrid_retriever = HybridRetriever(repository, vectors, bm25, siliconflow)
    persona_repository = PersonaRepository(database)
    project_repository = ProjectRepository(database)
    project_repository.ensure_default(default_kb_id)
    settings_service = ApplicationSettingsService(
        settings=resolved,
        repository=runtime_settings,
        credential_store=secret_store,
        siliconflow=siliconflow,
        mineru=mineru,
        kb_repository=repository,
        vectors=vectors,
        kb_id=default_kb_id,
        on_retrieval_configuration_changed=hybrid_retriever.clear_cache,
    )
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
        composition_distiller=CompositionDistiller(siliconflow, persona_repository),
    )
    chat_repository = ChatRepository(database)
    author_chat = AuthorChatService(
        repository=chat_repository,
        persona_repository=persona_repository,
        kb_repository=repository,
        retriever=hybrid_retriever,
        siliconflow=siliconflow,
        kb_id=default_kb_id,
        recent_rounds=lambda: _recent_chat_rounds(runtime_settings),
    )
    return ApplicationContext(
        settings=resolved,
        database=database,
        runtime_settings=runtime_settings,
        siliconflow_gate=siliconflow_gate,
        siliconflow=siliconflow,
        mineru=mineru,
        settings_service=settings_service,
        repository=repository,
        ingestion=ingestion,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        hybrid_retriever=hybrid_retriever,
        persona_repository=persona_repository,
        project_repository=project_repository,
        chat_repository=chat_repository,
        author_chat=author_chat,
        distillation=distillation,
        fidelity=FidelityService(
            persona_repository,
            PersonaFidelityEvaluator(siliconflow),
        ),
        default_kb_id=default_kb_id,
    )


def _recent_chat_rounds(repository: RuntimeSettingsRepository) -> int:
    value = repository.get("author_chat_recent_rounds", 6)
    return value if isinstance(value, int) and 1 <= value <= 20 else 6
