"""SQLite persistence for author conversations, messages, and rolling summaries."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable

from writing_factory.chat.models import (
    AnswerPolicy,
    ChatConversation,
    ChatMessage,
    ChatMessageStatus,
    ChatSource,
    KnowledgeMode,
)
from writing_factory.store.database import Database, utc_now


class ChatRepository:
    """Persist every completed chat turn independently of the UI process."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_conversation(
        self,
        *,
        kb_id: str,
        persona_id: str,
        persona_name: str,
        persona_version: int,
        runtime_persona: dict[str, object],
        knowledge_mode: KnowledgeMode,
        answer_policy: AnswerPolicy = "general_assisted",
        use_web_search: bool = False,
        selected_doc_ids: Iterable[str] = (),
        allowed_persona_doc_ids: Iterable[str] = (),
        target_persona_doc_ids: Iterable[str] = (),
        title: str = "新对话",
    ) -> str:
        conversation_id = f"chat_{uuid.uuid4().hex}"
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO chat_conversations(
                    conversation_id, kb_id, persona_id, persona_name, persona_version,
                    title, knowledge_mode, answer_policy, use_web_search,
                    selected_doc_ids_json,
                    allowed_persona_doc_ids_json, target_persona_doc_ids_json,
                    runtime_persona_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    kb_id,
                    persona_id,
                    persona_name,
                    persona_version,
                    title.strip() or "新对话",
                    knowledge_mode,
                    answer_policy,
                    int(use_web_search),
                    _json_list(selected_doc_ids),
                    _json_list(allowed_persona_doc_ids),
                    _json_list(target_persona_doc_ids),
                    json.dumps(runtime_persona, ensure_ascii=False, separators=(",", ":")),
                    now,
                    now,
                ),
            )
        return conversation_id

    def load_conversation(self, conversation_id: str) -> ChatConversation | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM chat_conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return None if row is None else _conversation(row)

    def list_conversations(self) -> list[dict[str, object]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT c.conversation_id, c.persona_id, c.persona_name, c.persona_version,
                       c.title, c.knowledge_mode, c.answer_policy, c.use_web_search,
                       c.updated_at,
                       count(m.message_id) AS message_count
                FROM chat_conversations c
                LEFT JOIN chat_messages m ON m.conversation_id = c.conversation_id
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        normalized = title.strip()[:80]
        if not normalized:
            return
        with self.database.connection() as connection:
            connection.execute(
                "UPDATE chat_conversations SET title = ?, updated_at = ? WHERE conversation_id = ?",
                (normalized, utc_now(), conversation_id),
            )

    def delete_conversations(self, conversation_ids: set[str]) -> int:
        identifiers = sorted(conversation_ids)
        if not identifiers:
            return 0
        placeholders = ",".join("?" for _ in identifiers)
        with self.database.connection() as connection:
            cursor = connection.execute(
                f"DELETE FROM chat_conversations WHERE conversation_id IN ({placeholders})",
                identifiers,
            )
        return max(0, cursor.rowcount)

    def append_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        status: ChatMessageStatus = "complete",
        sources: Iterable[ChatSource] = (),
    ) -> ChatMessage:
        message_id = f"chat_message_{uuid.uuid4().hex}"
        now = utc_now()
        source_values = list(sources)
        with self.database.connection() as connection:
            row = connection.execute(
                """SELECT coalesce(max(sequence), 0) + 1
                   FROM chat_messages WHERE conversation_id = ?""",
                (conversation_id,),
            ).fetchone()
            sequence = int(row[0])
            connection.execute(
                """
                INSERT INTO chat_messages(
                    message_id, conversation_id, sequence, role, content, status,
                    sources_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    sequence,
                    role,
                    content,
                    status,
                    json.dumps(
                        [item.model_dump(mode="json") for item in source_values],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    now,
                ),
            )
            connection.execute(
                "UPDATE chat_conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )
        return ChatMessage(
            message_id=message_id,
            conversation_id=conversation_id,
            sequence=sequence,
            role=role,
            content=content,
            status=status,
            sources=source_values,
            created_at=now,
        )

    def list_messages(self, conversation_id: str) -> list[ChatMessage]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM chat_messages
                WHERE conversation_id = ? ORDER BY sequence
                """,
                (conversation_id,),
            ).fetchall()
        return [_message(row) for row in rows]

    def load_message(self, message_id: str) -> ChatMessage | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM chat_messages WHERE message_id = ?",
                (message_id,),
            ).fetchone()
        return None if row is None else _message(row)

    def save_verification(self, message_id: str, result: dict[str, object]) -> None:
        with self.database.connection() as connection:
            cursor = connection.execute(
                "UPDATE chat_messages SET verification_json = ? WHERE message_id = ?",
                (json.dumps(result, ensure_ascii=False, separators=(",", ":")), message_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("找不到要核验的对话消息")

    def save_summary(self, conversation_id: str, text: str, through_sequence: int) -> None:
        with self.database.connection() as connection:
            connection.execute(
                """
                UPDATE chat_conversations
                SET summary_text = ?, summary_through_sequence = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (text, through_sequence, utc_now(), conversation_id),
            )


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))


def _conversation(row) -> ChatConversation:
    return ChatConversation(
        conversation_id=row["conversation_id"],
        kb_id=row["kb_id"],
        persona_id=row["persona_id"],
        persona_name=row["persona_name"],
        persona_version=row["persona_version"],
        title=row["title"],
        knowledge_mode=row["knowledge_mode"],
        answer_policy=row["answer_policy"],
        use_web_search=bool(row["use_web_search"]),
        selected_doc_ids=json.loads(row["selected_doc_ids_json"]),
        allowed_persona_doc_ids=json.loads(row["allowed_persona_doc_ids_json"]),
        target_persona_doc_ids=json.loads(row["target_persona_doc_ids_json"]),
        runtime_persona=json.loads(row["runtime_persona_json"]),
        summary_text=row["summary_text"],
        summary_through_sequence=row["summary_through_sequence"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _message(row) -> ChatMessage:
    verification = json.loads(row["verification_json"]) if row["verification_json"] else None
    return ChatMessage(
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        sequence=row["sequence"],
        role=row["role"],
        content=row["content"],
        status=row["status"],
        sources=[ChatSource.model_validate(item) for item in json.loads(row["sources_json"])],
        verification=verification,
        created_at=row["created_at"],
    )
