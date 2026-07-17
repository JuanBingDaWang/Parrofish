"""以确定性规则汇总学术候选验证并选择核心模型。"""

from __future__ import annotations

from writing_factory.distill.academic import (
    AcademicModelValidation,
    CandidateAssessment,
    CandidateCluster,
    CandidateRecord,
    CandidateRegistry,
    ExclusivityAssessment,
    PaperProfile,
)
from writing_factory.distill.models import PersonaMode


def select_academic_candidates(
    *,
    mode: PersonaMode,
    clusters: list[CandidateCluster],
    target_profiles: list[PaperProfile],
    target_doc_ids: list[str],
    holdout_doc_ids: list[str],
    control_doc_ids: list[str],
    domain: str,
    generative: list[CandidateAssessment],
    exclusivity: list[ExclusivityAssessment],
) -> CandidateRegistry:
    """按人物个性或主题共性语义确定性选择 3–7 个核心模型。"""

    paper_candidates = {
        item.paper_candidate_id: (profile.doc_id, item)
        for profile in target_profiles
        for item in profile.candidates
    }
    generative_by_id = {item.candidate_id: item for item in generative}
    exclusivity_by_id = {item.candidate_id: item for item in exclusivity}
    records: list[CandidateRecord] = []
    for cluster in clusters:
        generation = generative_by_id.get(cluster.candidate_id)
        matched_ids = generation.matched_paper_candidate_ids if generation is not None else []
        member_ids = list(dict.fromkeys([*cluster.paper_candidate_ids, *matched_ids]))
        unknown = set(member_ids) - paper_candidates.keys()
        if unknown:
            raise ValueError(f"候选验证引用了未知单篇候选：{', '.join(sorted(unknown))}")
        supporting_doc_ids = sorted({paper_candidates[item_id][0] for item_id in member_ids})
        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for item_id in member_ids
                for evidence_id in paper_candidates[item_id][1].evidence_ids
            )
        )
        updated_cluster = cluster.model_copy(
            update={"paper_candidate_ids": member_ids, "evidence_ids": evidence_ids}
        )
        count = len(supporting_doc_ids)
        recurrence = "high" if count >= 3 else "basic" if count >= 2 else "provisional"
        if generation is None:
            generative_status = "not_tested"
            generative_rationale = "语料规模未触发留出验证。"
        else:
            generative_status = generation.status
            generative_rationale = generation.rationale
        exclusive = exclusivity_by_id.get(cluster.candidate_id)
        if exclusive is None:
            specificity = "unverified"
            exclusivity_rationale = "本次蒸馏没有提供可用的同领域对照语料。"
        else:
            specificity = exclusive.specificity
            exclusivity_rationale = exclusive.rationale
        validation = AcademicModelValidation(
            supporting_doc_ids=supporting_doc_ids,
            recurrence_document_count=count,
            recurrence_level=recurrence,
            generative_status=generative_status,
            generative_rationale=generative_rationale,
            specificity=specificity,
            exclusivity_rationale=exclusivity_rationale,
            control_corpus_used=bool(control_doc_ids),
        )
        records.append(CandidateRecord(candidate=updated_cluster, validation=validation))

    eligible = [record for record in records if record.validation.eligible]
    if mode == "topic":
        eligible.sort(key=_rank_key)
        core = eligible[:7]
        if len(core) < 3:
            raise ValueError("跨文档复现的有证据候选不足 3 个，不能发布主题档案")
        convention_ids: set[str] = set()
    else:
        personal = [
            record
            for record in eligible
            if record.validation.specificity in {"author_distinctive", "unverified"}
        ]
        generic = [
            record
            for record in eligible
            if record.validation.specificity
            in {"field_conventional", "general_academic", "general_nonfiction"}
        ]
        personal.sort(key=_rank_key)
        generic.sort(key=_rank_key)
        core = personal[:7]
        if len(core) < 3:
            core.extend(generic[: 3 - len(core)])
        if len(core) < 3:
            raise ValueError("全部有证据的候选仍不足 3 个，不能发布作者档案")
        core_ids = {record.candidate.candidate_id for record in core}
        remaining_generic = [
            record for record in generic if record.candidate.candidate_id not in core_ids
        ][:7]
        convention_ids = {
            record.candidate.candidate_id for record in remaining_generic
        }

    core_ids = {record.candidate.candidate_id for record in core}
    ranks = {record.candidate.candidate_id: index for index, record in enumerate(core, 1)}
    selected_records: list[CandidateRecord] = []
    for record in records:
        identifier = record.candidate.candidate_id
        if identifier in core_ids:
            selected_records.append(
                record.model_copy(
                    update={"selected_as": "core", "selection_rank": ranks[identifier]}
                )
            )
        elif identifier in convention_ids:
            selected_records.append(record.model_copy(update={"selected_as": "convention"}))
        elif _passed_validation_count(record, mode=mode) >= 1:
            selected_records.append(record.model_copy(update={"selected_as": "heuristic"}))
        else:
            selected_records.append(record)
    return CandidateRegistry(
        target_doc_ids=target_doc_ids,
        holdout_doc_ids=holdout_doc_ids,
        control_doc_ids=control_doc_ids,
        domain=domain,
        records=selected_records,
    )


def _rank_key(record: CandidateRecord) -> tuple[int, int, int, int, str]:
    validation = record.validation
    recurrence = {"high": 0, "basic": 1, "provisional": 2}[validation.recurrence_level]
    generation = {"passed": 0, "not_tested": 1, "failed": 2}[validation.generative_status]
    specificity = {
        "author_distinctive": 0,
        "unverified": 1,
        "field_conventional": 2,
        "general_academic": 3,
        "general_nonfiction": 3,
    }[validation.specificity]
    return (
        recurrence,
        generation,
        specificity,
        -len(record.candidate.evidence_ids),
        record.candidate.candidate_id,
    )


def _passed_validation_count(record: CandidateRecord, *, mode: PersonaMode) -> int:
    """Nüwa 三重验证通过一至两项的候选降级为启发式。"""

    validation = record.validation
    checks = [
        validation.recurrence_document_count >= 2,
        validation.generative_status == "passed",
    ]
    if mode == "person":
        checks.append(validation.specificity == "author_distinctive")
    return sum(checks)
