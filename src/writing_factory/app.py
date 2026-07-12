"""Application dependency assembly and controlled resource shutdown."""

from __future__ import annotations

from dataclasses import dataclass

from writing_factory.config import Settings, load_settings
from writing_factory.config.logging import configure_logging
from writing_factory.llm import MinerUClient, SiliconFlowClient
from writing_factory.store import Database


@dataclass(slots=True, weakref_slot=True)
class ApplicationContext:
    """Own long-lived services and expose one deterministic shutdown point."""

    settings: Settings
    database: Database
    siliconflow: SiliconFlowClient
    mineru: MinerUClient

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
    return ApplicationContext(
        settings=resolved,
        database=database,
        siliconflow=SiliconFlowClient(resolved, database),
        mineru=MinerUClient(resolved, database),
    )
