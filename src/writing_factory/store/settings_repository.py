"""持久化不含密钥、可在运行时修改的桌面设置。"""

from __future__ import annotations

import json
from typing import Any

from writing_factory.store.database import Database, utc_now


class RuntimeSettingsRepository:
    """在 SQLite 中读写用户设置，密钥仍只由集中配置加载。"""

    def __init__(self, database: Database) -> None:
        self.database = database

    def get(self, key: str, default: Any = None) -> Any:
        """读取 JSON 设置值，不存在时返回默认值。"""

        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT value_json FROM app_settings WHERE setting_key = ?", (key,)
            ).fetchone()
        return default if row is None else json.loads(row["value_json"])

    def set(self, key: str, value: Any) -> None:
        """原子保存一个 JSON 设置值。"""

        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO app_settings(setting_key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(setting_key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, payload, utc_now()),
            )
