"""Deterministic per-KB BM25 index rebuilt from SQLite child chunks."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi

from writing_factory.kb.models import Chunk, SearchHit
from writing_factory.store.kb_repository import KnowledgeBaseRepository

jieba.setLogLevel(logging.ERROR)


@dataclass(slots=True)
class _IndexState:
    fingerprint: str
    chunks: list[Chunk]
    tokenized_corpus: list[list[str]]
    index: BM25Okapi


class BM25Index:
    """Tokenize Chinese with jieba and lazily refresh when the corpus changes."""

    def __init__(self, repository: KnowledgeBaseRepository) -> None:
        self.repository = repository
        self._states: dict[str, _IndexState] = {}

    def invalidate(self, kb_id: str) -> None:
        """Force a deterministic SQLite rebuild on the next query."""

        self._states.pop(kb_id, None)

    def rebuild(self, kb_id: str) -> int:
        """Rebuild now and return the number of indexed child chunks."""

        chunks = self.repository.ready_child_chunks(kb_id)
        self._states.pop(kb_id, None)
        if chunks:
            self._states[kb_id] = self._build_state(chunks)
        return len(chunks)

    def search(self, kb_id: str, query: str, *, limit: int) -> list[SearchHit]:
        """Return ranked sparse hits with complete source metadata."""

        if limit <= 0 or not query.strip():
            return []
        chunks = self.repository.ready_child_chunks(kb_id)
        if not chunks:
            self._states.pop(kb_id, None)
            return []
        fingerprint = self._fingerprint(chunks)
        state = self._states.get(kb_id)
        if state is None or state.fingerprint != fingerprint:
            state = self._build_state(chunks)
            self._states[kb_id] = state
        query_tokens = self._tokenize(query)
        scores = state.index.get_scores(query_tokens)
        adjusted_scores = [
            float(score) + 1e-6 * len(set(query_tokens).intersection(state.tokenized_corpus[index]))
            for index, score in enumerate(scores)
        ]
        ranked = sorted(
            range(len(adjusted_scores)),
            key=lambda index: (-adjusted_scores[index], index),
        )[:limit]
        return [
            SearchHit(
                chunk_id=state.chunks[index].chunk_id,
                doc_id=state.chunks[index].doc_id,
                text=state.chunks[index].text,
                score=adjusted_scores[index],
                rank=rank,
                source="bm25",
                page_start=state.chunks[index].page_start,
                page_end=state.chunks[index].page_end,
                section_heading=state.chunks[index].section_heading,
                parent_id=state.chunks[index].parent_id,
            )
            for rank, index in enumerate(ranked, start=1)
        ]

    def _build_state(self, chunks: list[Chunk]) -> _IndexState:
        corpus = [self._tokenize(chunk.text) for chunk in chunks]
        return _IndexState(
            fingerprint=self._fingerprint(chunks),
            chunks=chunks,
            tokenized_corpus=corpus,
            index=BM25Okapi(corpus),
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.casefold() for token in jieba.lcut(text) if token.strip()]

    @staticmethod
    def _fingerprint(chunks: list[Chunk]) -> str:
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(chunk.chunk_id.encode("utf-8"))
        return digest.hexdigest()
