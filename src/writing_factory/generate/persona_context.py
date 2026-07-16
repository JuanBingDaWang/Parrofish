"""Build a task-scoped runtime persona context for one nonfiction genre."""

from __future__ import annotations

from typing import Any

from writing_factory.distill.runtime import RuntimePersonaSpec
from writing_factory.nonfiction import NonfictionGenre


def persona_context_for_genre(
    persona: RuntimePersonaSpec,
    genre: NonfictionGenre,
) -> dict[str, Any]:
    """Keep only the matching genre overlay and source-free cross-genre rules."""

    payload = persona.model_dump(mode="json")
    composition = payload.get("composition_dna")
    if not isinstance(composition, dict):
        return payload
    profiles = composition.get("genre_profiles")
    if isinstance(profiles, list):
        composition["genre_profiles"] = [
            item for item in profiles if isinstance(item, dict) and item.get("genre") == genre
        ]
    composition.pop("information_gaps", None)
    return payload
