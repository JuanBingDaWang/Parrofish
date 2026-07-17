"""Validation and deterministic assembly for composition distillation outputs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from writing_factory.distill.composition_models import (
    CompositionDNA,
    CompositionEvidence,
    CompositionPattern,
    CompositionReduceResult,
    DocumentCompositionProfile,
    GenreCompositionProfile,
    ReducedCompositionPattern,
)
from writing_factory.distill.extraction import StructuredDistillationError
from writing_factory.distill.models import SourceSegment
from writing_factory.nonfiction import genre_label


def validate_document_profile(
    profile: DocumentCompositionProfile,
    doc_id: str,
    segments: list[SourceSegment],
) -> None:
    if profile.doc_id != doc_id:
        raise StructuredDistillationError("文档结构画像的 doc_id 与输入不一致")
    if profile.genre_label != genre_label(profile.genre):
        raise StructuredDistillationError("文档结构画像的文体标签与 genre 不一致")
    allowed = {item.chunk_id for item in segments}
    cited = {
        chunk_id
        for pattern in profile.patterns
        for chunk_id in pattern.evidence_chunk_ids
    }
    if cited - allowed:
        raise StructuredDistillationError("文档结构画像引用了未知 chunk_id")


def validate_composition_reduce(
    reduced: CompositionReduceResult,
    targets: list[DocumentCompositionProfile],
    controls: list[DocumentCompositionProfile],
) -> None:
    target_docs = {item.doc_id for item in targets}
    chunk_owners = {
        chunk_id: profile.doc_id
        for profile in targets
        for pattern in profile.patterns
        for chunk_id in pattern.evidence_chunk_ids
    }
    target_genres = {item.doc_id: item.genre for item in targets}
    expected_genres = set(target_genres.values())
    reduced_genres = [item.genre for item in reduced.genre_profiles]
    if len(reduced_genres) != len(set(reduced_genres)):
        raise StructuredDistillationError("每种目标文体只能有一个谋篇侧写")
    if set(reduced_genres) != expected_genres:
        raise StructuredDistillationError("谋篇侧写必须完整覆盖全部目标文体")
    pattern_ids: list[str] = []
    all_patterns: list[ReducedCompositionPattern] = []
    for profile in reduced.genre_profiles:
        if profile.genre_label != genre_label(profile.genre):
            raise StructuredDistillationError("文体侧写标签与 genre 不一致")
        expected_count = sum(item.genre == profile.genre for item in targets)
        if profile.source_document_count != expected_count or expected_count == 0:
            raise StructuredDistillationError("文体侧写的目标文档计数不正确")
        if any(
            target_genres.get(doc_id) != profile.genre
            for pattern in profile.patterns
            for doc_id in pattern.supporting_doc_ids
        ):
            raise StructuredDistillationError("文体侧写混入了其他文体的支持文档")
        all_patterns.extend(profile.patterns)
    all_patterns.extend(reduced.cross_genre_patterns)
    for pattern in all_patterns:
        pattern_ids.append(pattern.pattern_id)
        docs = set(pattern.supporting_doc_ids)
        if not docs <= target_docs:
            raise StructuredDistillationError("谋篇模式引用了非目标语料 doc_id")
        if pattern.recurrence_document_count == 1 and pattern.specificity != "provisional":
            raise StructuredDistillationError("单文档谋篇观察必须标记为 provisional")
        if (
            not controls
            and pattern.recurrence_document_count > 1
            and pattern.specificity
            in {"author_distinctive", "genre_conventional", "cross_genre_author"}
        ):
            raise StructuredDistillationError("没有对照语料时不得宣称结构具有排他性")
        for chunk_id in pattern.evidence_chunk_ids:
            owner = chunk_owners.get(chunk_id)
            if owner is None:
                raise StructuredDistillationError(
                    f"谋篇模式 {pattern.pattern_id} 引用了单篇画像中不存在的结构证据 "
                    f"{chunk_id}"
                )
            if owner not in docs:
                declared = ", ".join(sorted(docs))
                raise StructuredDistillationError(
                    f"谋篇模式 {pattern.pattern_id} 的结构证据 {chunk_id} 实际属于 "
                    f"{owner}，但 supporting_doc_ids 为：{declared}"
                )
    if len(pattern_ids) != len(set(pattern_ids)):
        raise StructuredDistillationError("谋篇模式 pattern_id 必须全局唯一")
    for pattern in reduced.cross_genre_patterns:
        genres = {target_genres[doc_id] for doc_id in pattern.supporting_doc_ids}
        if len(genres) < 2:
            raise StructuredDistillationError("跨文体谋篇模式必须覆盖至少两种文体")


def normalize_composition_evidence_ownership(
    reduced: CompositionReduceResult,
    targets: list[DocumentCompositionProfile],
    *,
    control_available: bool,
) -> CompositionReduceResult:
    """以单篇画像为准重建证据归属、复现数和派生置信度。"""

    chunk_owners = {
        chunk_id: profile.doc_id
        for profile in targets
        for pattern in profile.patterns
        for chunk_id in pattern.evidence_chunk_ids
    }

    def normalize(pattern: ReducedCompositionPattern) -> ReducedCompositionPattern:
        evidence_chunk_ids = list(dict.fromkeys(pattern.evidence_chunk_ids))
        unknown = [chunk_id for chunk_id in evidence_chunk_ids if chunk_id not in chunk_owners]
        if unknown:
            raise StructuredDistillationError(
                f"谋篇模式 {pattern.pattern_id} 引用了未知结构证据："
                f"{', '.join(unknown)}"
            )
        supporting_doc_ids = list(
            dict.fromkeys(chunk_owners[chunk_id] for chunk_id in evidence_chunk_ids)
        )
        count = len(supporting_doc_ids)
        specificity = pattern.specificity
        if count == 1:
            specificity = "provisional"
        elif not control_available and specificity in {
            "author_distinctive",
            "genre_conventional",
            "cross_genre_author",
        }:
            specificity = "unverified"
        confidence = "high" if count >= 3 else "medium" if count == 2 else "low"
        return pattern.model_copy(
            update={
                "evidence_chunk_ids": evidence_chunk_ids,
                "supporting_doc_ids": supporting_doc_ids,
                "recurrence_document_count": count,
                "specificity": specificity,
                "confidence": confidence,
            }
        )

    genre_profiles = [
        profile.model_copy(
            update={"patterns": [normalize(pattern) for pattern in profile.patterns]}
        )
        for profile in reduced.genre_profiles
    ]
    return reduced.model_copy(
        update={
            "genre_profiles": genre_profiles,
            "cross_genre_patterns": [
                normalize(pattern) for pattern in reduced.cross_genre_patterns
            ],
        }
    )


def assemble_composition_dna(
    reduced: CompositionReduceResult,
    segments: dict[str, SourceSegment],
    control_available: bool,
) -> CompositionDNA:
    def pattern(item: ReducedCompositionPattern) -> CompositionPattern:
        count = len(set(item.supporting_doc_ids))
        confidence = "high" if count >= 3 else "medium" if count == 2 else "low"
        specificity = item.specificity
        if count == 1:
            specificity = "provisional"
        elif not control_available and specificity in {
            "author_distinctive",
            "genre_conventional",
            "cross_genre_author",
        }:
            specificity = "unverified"
        evidence = []
        for chunk_id in dict.fromkeys(item.evidence_chunk_ids):
            segment = segments[chunk_id]
            digest = hashlib.sha256(f"{item.pattern_id}|{chunk_id}".encode()).hexdigest()[:20]
            evidence.append(
                CompositionEvidence(
                    evidence_id=f"structure_ev_{digest}",
                    chunk_id=chunk_id,
                    doc_id=segment.doc_id,
                    summary=f"{item.scope} 层级的结构证据：{item.name}",
                    page_start=segment.page_start,
                    page_end=segment.page_end,
                    section_heading=segment.section_heading,
                )
            )
        return CompositionPattern(
            pattern_id=item.pattern_id,
            name=item.name,
            scope=item.scope,
            description=item.description,
            sequence=item.sequence,
            relations=item.relations,
            applicability=item.applicability,
            variability=item.variability,
            supporting_doc_ids=list(dict.fromkeys(item.supporting_doc_ids)),
            recurrence_document_count=count,
            specificity=specificity,
            confidence=confidence,
            evidence=evidence,
        )

    profiles = [
        GenreCompositionProfile(
            genre=item.genre,
            genre_label=item.genre_label,
            source_document_count=item.source_document_count,
            typical_purposes=item.typical_purposes,
            audience_tendencies=item.audience_tendencies,
            heading_strategy=item.heading_strategy,
            paragraph_strategy=item.paragraph_strategy,
            patterns=[pattern(value) for value in item.patterns],
            declared_limits=item.declared_limits,
        )
        for item in reduced.genre_profiles
    ]
    return CompositionDNA(
        genre_profiles=profiles,
        cross_genre_patterns=[pattern(item) for item in reduced.cross_genre_patterns],
        information_gaps=reduced.information_gaps,
    )


def ensure_composition_chinese(value: Any) -> None:
    payload = json.dumps(value.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    if sum("\u3400" <= character <= "\u9fff" for character in payload) < 20:
        raise StructuredDistillationError("谋篇分析的简体中文内容不足")
