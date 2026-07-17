"""文档级归并、跨文档聚类与独立非虚构候选验证流水线。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from contextvars import copy_context
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from writing_factory.distill.academic import (
    CandidateCluster,
    CandidateClusterResult,
    CandidateRegistry,
    ExclusivityBatchResult,
    PaperMentalCandidate,
    PaperProfile,
    ValidationBatchResult,
)
from writing_factory.distill.academic_prompts import (
    cluster_messages,
    exclusivity_validation_messages,
    generative_validation_messages,
    paper_profile_messages,
)
from writing_factory.distill.extraction import StructuredDistillationError
from writing_factory.distill.language import OutputLanguageError, validate_academic_language
from writing_factory.distill.models import PersonaMode, SourceInfo
from writing_factory.distill.progress import WeightedProgress
from writing_factory.distill.selection import select_academic_candidates
from writing_factory.llm import SiliconFlowClient
from writing_factory.store.persona_repository import PersonaRepository

ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], None]
StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class AcademicDistillationEngine:
    """把人物或主题 Map 候选变成经过独立验证的登记表。"""

    PAPER_VERSION = "nonfiction-paper-v2-mode-aware"
    CLUSTER_VERSION = "nonfiction-cluster-v2-mode-aware"
    GENERATIVE_VERSION = "nonfiction-generative-v2-mode-aware"
    EXCLUSIVITY_VERSION = "academic-exclusivity-v1"
    REGISTRY_VERSION = "nonfiction-registry-v3-mode-aware-selection"

    def __init__(
        self,
        siliconflow: SiliconFlowClient,
        repository: PersonaRepository,
        parallelism: Callable[[], int],
    ) -> None:
        self.siliconflow = siliconflow
        self.repository = repository
        self.parallelism = parallelism

    def build_registry(
        self,
        *,
        run_id: str,
        mode: PersonaMode,
        target_label: str,
        domain: str,
        target_bundle: dict[str, object],
        target_source_info: tuple[SourceInfo, ...],
        target_hash: str,
        control_bundle: dict[str, object] | None,
        control_source_info: tuple[SourceInfo, ...],
        control_hash: str | None,
        run_generative_validation: bool = True,
        run_exclusivity_validation: bool = True,
        reuse_persona_id: str | None = None,
        progress: ProgressCallback,
        check_cancelled: CancellationCheck,
    ) -> CandidateRegistry:
        """运行或恢复全部文档级与全局候选验证阶段。"""

        def target_job(callback: ProgressCallback) -> list[PaperProfile]:
            return self._consolidate_corpus(
                run_id=run_id,
                stage="target_paper",
                mode=mode,
                bundle=target_bundle,
                source_info=target_source_info,
                corpus_hash=target_hash,
                reuse_persona_id=reuse_persona_id,
                progress=callback,
                progress_start=0,
                progress_end=100,
                check_cancelled=check_cancelled,
            )

        target_profiles: list[PaperProfile]
        control_profiles: list[PaperProfile] = []
        if control_bundle is not None and control_hash is not None:
            tracker = WeightedProgress(
                progress,
                start=66,
                end=82,
                weights={
                    "target": max(1, len(target_source_info)),
                    "control": max(1, len(control_source_info)),
                },
            )

            def control_job() -> list[PaperProfile]:
                return self._consolidate_corpus(
                    run_id=run_id,
                    stage="control_paper",
                    mode=mode,
                    bundle=control_bundle,
                    source_info=control_source_info,
                    corpus_hash=control_hash,
                    reuse_persona_id=reuse_persona_id,
                    progress=tracker.branch("control"),
                    progress_start=0,
                    progress_end=100,
                    check_cancelled=check_cancelled,
                )

            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="paper-corpus") as pool:
                target_future = pool.submit(
                    copy_context().run,
                    target_job,
                    tracker.branch("target"),
                )
                control_future = pool.submit(copy_context().run, control_job)
                target_profiles = target_future.result()
                tracker.complete("target")
                control_profiles = control_future.result()
                tracker.complete("control")
        else:
            def target_progress(percent: int, message: str) -> None:
                progress(66 + round(16 * percent / 100), message)

            target_profiles = target_job(target_progress)

        holdout_doc_ids = (
            choose_holdout_doc_ids(target_source_info) if run_generative_validation else []
        )
        training_profiles = [
            profile for profile in target_profiles if profile.doc_id not in holdout_doc_ids
        ]
        check_cancelled()
        progress(83, "跨文档聚类候选")
        clusters = self._cluster(
            run_id=run_id,
            mode=mode,
            target_label=target_label,
            domain=domain,
            profiles=training_profiles,
            target_hash=target_hash,
        )
        holdout_profiles = [
            profile for profile in target_profiles if profile.doc_id in holdout_doc_ids
        ]
        generative = None
        exclusivity = None
        validation_jobs: dict[str, Callable[[], BaseModel | None]] = {}
        if run_generative_validation:
            validation_jobs["generative"] = lambda: self._validate_generative(
                run_id=run_id,
                mode=mode,
                clusters=clusters,
                holdout_profiles=holdout_profiles,
                target_hash=target_hash,
            )
        if run_exclusivity_validation:
            validation_jobs["exclusivity"] = lambda: self._validate_exclusivity(
                run_id=run_id,
                domain=domain,
                clusters=clusters,
                control_profiles=control_profiles,
                control_hash=control_hash,
            )
        if validation_jobs:
            progress(87, "并发运行中性验证")
            with ThreadPoolExecutor(
                max_workers=len(validation_jobs), thread_name_prefix="candidate-validation"
            ) as pool:
                futures = {
                    pool.submit(copy_context().run, job): key
                    for key, job in validation_jobs.items()
                }
                for completed, future in enumerate(as_completed(futures), 1):
                    check_cancelled()
                    key = futures[future]
                    value = future.result()
                    if key == "generative":
                        generative = value
                    else:
                        exclusivity = value
                    progress(87 + round(3 * completed / len(futures)), "中性验证完成")
        registry = select_academic_candidates(
            mode=mode,
            clusters=clusters,
            target_profiles=target_profiles,
            target_doc_ids=[item.doc_id for item in target_source_info],
            holdout_doc_ids=holdout_doc_ids,
            control_doc_ids=[item.doc_id for item in control_source_info],
            domain=domain,
            generative=generative.assessments if generative is not None else [],
            exclusivity=exclusivity.assessments if exclusivity is not None else [],
        )
        input_hash = _hash_payload(
            self.REGISTRY_VERSION,
            target_hash,
            control_hash or "",
            domain,
            registry.model_dump(mode="json"),
        )
        self.repository.save_stage_result(
            run_id=run_id,
            stage="candidate_registry",
            item_id="global",
            input_hash=input_hash,
            result=registry,
        )
        return registry

    def _consolidate_corpus(
        self,
        *,
        run_id: str,
        stage: str,
        mode: PersonaMode,
        bundle: dict[str, object],
        source_info: tuple[SourceInfo, ...],
        corpus_hash: str,
        reuse_persona_id: str | None,
        progress: ProgressCallback,
        progress_start: int,
        progress_end: int,
        check_cancelled: CancellationCheck,
    ) -> list[PaperProfile]:
        candidates = list(bundle.get("mental_candidates", []))
        evidence = list(bundle.get("evidence_registry", []))
        profiles: dict[str, PaperProfile] = {}
        missing: list[tuple[SourceInfo, list[dict[str, object]], list[dict[str, object]], str]] = []
        for source in source_info:
            doc_candidates = [
                item for item in candidates if source.doc_id in item.get("source_doc_ids", [])
            ]
            evidence_ids = {
                identifier
                for item in doc_candidates
                for identifier in item.get("evidence_ids", [])
                if isinstance(identifier, str)
            }
            doc_evidence = [item for item in evidence if item.get("evidence_id") in evidence_ids]
            input_hash = _hash_payload(
                self.PAPER_VERSION,
                mode,
                source.doc_id,
                doc_candidates,
                doc_evidence,
            )
            cached = self.repository.load_stage_result(
                run_id=run_id,
                stage=stage,
                item_id=source.doc_id,
                model=PaperProfile,
            )
            if cached is None:
                cached = self.repository.find_compatible_stage_result(
                    stage=stage,
                    item_id=source.doc_id,
                    input_hash=input_hash,
                    model=PaperProfile,
                    persona_id=reuse_persona_id,
                )
            if cached is not None:
                cached = self._stabilize_paper_profile(cached, source.doc_id, doc_candidates)
                profiles[source.doc_id] = cached
                self.repository.save_stage_result(
                    run_id=run_id,
                    stage=stage,
                    item_id=source.doc_id,
                    input_hash=input_hash,
                    result=cached,
                )
            elif not doc_candidates:
                empty = PaperProfile(doc_id=source.doc_id)
                profiles[source.doc_id] = empty
                self.repository.save_stage_result(
                    run_id=run_id,
                    stage=stage,
                    item_id=source.doc_id,
                    input_hash=input_hash,
                    result=empty,
                )
            else:
                missing.append((source, doc_candidates, doc_evidence, input_hash))

        if missing:
            workers = min(max(1, self.parallelism()), len(missing))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=stage) as executor:
                futures: dict[Future[PaperProfile], tuple[SourceInfo, str]] = {}
                for source, doc_candidates, doc_evidence, input_hash in missing:
                    future = executor.submit(
                        copy_context().run,
                        self._consolidate_one,
                        mode,
                        source.doc_id,
                        doc_candidates,
                        doc_evidence,
                    )
                    futures[future] = (source, input_hash)
                for completed, future in enumerate(as_completed(futures), 1):
                    check_cancelled()
                    source, input_hash = futures[future]
                    profile = future.result()
                    profiles[source.doc_id] = profile
                    self.repository.save_stage_result(
                        run_id=run_id,
                        stage=stage,
                        item_id=source.doc_id,
                        input_hash=input_hash,
                        result=profile,
                    )
                    percent = progress_start + round(
                        (progress_end - progress_start) * completed / len(missing)
                    )
                    progress(percent, f"并发归并单篇文档（{completed}/{len(missing)}）")
        return [profiles[item.doc_id] for item in source_info]

    def _consolidate_one(
        self,
        mode: PersonaMode,
        doc_id: str,
        candidates: list[dict[str, object]],
        evidence: list[dict[str, object]],
    ) -> PaperProfile:
        stage = getattr(self.siliconflow, "stream_stage", None)
        stream_context = (
            stage(f"单篇文档画像 · {doc_id[-8:]}") if callable(stage) else nullcontext()
        )
        with stream_context:
            return self._structured_call(
                paper_profile_messages(
                    mode=mode,
                    doc_id=doc_id,
                    candidates=candidates,
                    evidence=evidence,
                ),
                PaperProfile,
                seed=23,
                step_id="distill.paper_profile",
                validator=lambda profile: self._stabilize_paper_profile(
                    profile,
                    doc_id,
                    candidates,
                ),
            )

    @staticmethod
    def _stabilize_paper_profile(
        profile: PaperProfile,
        doc_id: str,
        candidates: list[dict[str, object]],
    ) -> PaperProfile:
        if profile.doc_id != doc_id:
            raise StructuredDistillationError("单篇归并返回了错误的 doc_id")
        candidate_by_id = {str(item["map_candidate_id"]): item for item in candidates}
        used: set[str] = set()
        stable: list[PaperMentalCandidate] = []
        for item in profile.candidates:
            member_ids = list(dict.fromkeys(item.map_candidate_ids))
            unknown = set(member_ids) - candidate_by_id.keys()
            if unknown or used.intersection(member_ids):
                raise StructuredDistillationError("单篇归并引用了未知或重复的 Map 候选")
            allowed_evidence = {
                evidence_id
                for member_id in member_ids
                for evidence_id in candidate_by_id[member_id].get("evidence_ids", [])
            }
            used.update(member_ids)
            digest = hashlib.sha256(
                f"{doc_id}|{'|'.join(sorted(member_ids))}".encode()
            ).hexdigest()[:24]
            stable.append(
                item.model_copy(
                    update={
                        "paper_candidate_id": f"paper_candidate_{digest}",
                        "map_candidate_ids": member_ids,
                        "evidence_ids": sorted(allowed_evidence),
                    }
                )
            )
        return PaperProfile(doc_id=doc_id, candidates=stable)

    def _cluster(
        self,
        *,
        run_id: str,
        mode: PersonaMode,
        target_label: str,
        domain: str,
        profiles: list[PaperProfile],
        target_hash: str,
    ) -> list[CandidateCluster]:
        payload = [item.model_dump(mode="json") for item in profiles]
        input_hash = _hash_payload(
            self.CLUSTER_VERSION, mode, target_label, domain, target_hash, payload
        )
        cached = self.repository.find_compatible_stage_result(
            stage="candidate_clusters", item_id="global", input_hash=input_hash,
            model=CandidateClusterResult, persona_id=None,
        )
        current = self.repository.load_stage_result(
            run_id=run_id,
            stage="candidate_clusters",
            item_id="global",
            model=CandidateClusterResult,
        )
        cached = current or cached
        if cached is None:
            cached = self._structured_call(
                cluster_messages(
                    mode=mode,
                    target_label=target_label,
                    domain=domain,
                    paper_profiles=payload,
                ),
                CandidateClusterResult,
                seed=29,
                step_id="distill.cluster",
                validator=lambda result: self._stabilize_clusters(result, profiles),
            )
        self.repository.save_stage_result(
            run_id=run_id,
            stage="candidate_clusters",
            item_id="global",
            input_hash=input_hash,
            result=cached,
        )
        return cached.candidates

    @staticmethod
    def _stabilize_clusters(
        result: CandidateClusterResult, profiles: list[PaperProfile]
    ) -> CandidateClusterResult:
        members = {
            item.paper_candidate_id: item for profile in profiles for item in profile.candidates
        }
        used: set[str] = set()
        clusters: list[CandidateCluster] = []
        for item in result.candidates:
            member_ids = list(dict.fromkeys(item.paper_candidate_ids))
            if set(member_ids) - members.keys() or used.intersection(member_ids):
                raise StructuredDistillationError("跨文档聚类引用了未知或重复的单篇候选")
            allowed_evidence = {
                evidence_id
                for member_id in member_ids
                for evidence_id in members[member_id].evidence_ids
            }
            used.update(member_ids)
            digest = hashlib.sha256("|".join(sorted(member_ids)).encode("utf-8")).hexdigest()[:24]
            clusters.append(
                item.model_copy(
                    update={
                        "candidate_id": f"candidate_{digest}",
                        "paper_candidate_ids": member_ids,
                        "evidence_ids": sorted(allowed_evidence),
                    }
                )
            )
        return CandidateClusterResult(candidates=clusters)

    def _validate_generative(
        self,
        *,
        run_id: str,
        mode: PersonaMode,
        clusters: list[CandidateCluster],
        holdout_profiles: list[PaperProfile],
        target_hash: str,
    ) -> ValidationBatchResult | None:
        if not holdout_profiles:
            return None
        candidate_payload = [item.model_dump(mode="json") for item in clusters]
        holdout_payload = [item.model_dump(mode="json") for item in holdout_profiles]
        input_hash = _hash_payload(
            self.GENERATIVE_VERSION, mode, target_hash, candidate_payload, holdout_payload
        )
        result = self.repository.load_stage_result(
            run_id=run_id, stage="generative_validation", item_id="global",
            model=ValidationBatchResult,
        )
        if result is None:
            result = self._structured_call(
                generative_validation_messages(
                    mode=mode,
                    candidates=candidate_payload,
                    holdout_profiles=holdout_payload,
                ),
                ValidationBatchResult,
                seed=31,
                step_id="distill.generative_validation",
                validator=lambda value: self._validate_generative_result(
                    value,
                    clusters,
                    holdout_profiles,
                ),
            )
        self.repository.save_stage_result(
            run_id=run_id,
            stage="generative_validation",
            item_id="global",
            input_hash=input_hash,
            result=result,
        )
        return result

    @staticmethod
    def _validate_generative_result(
        result: ValidationBatchResult,
        clusters: list[CandidateCluster],
        holdout_profiles: list[PaperProfile],
    ) -> ValidationBatchResult:
        expected = {item.candidate_id for item in clusters}
        actual = {item.candidate_id for item in result.assessments}
        allowed_matches = {
            item.paper_candidate_id for profile in holdout_profiles for item in profile.candidates
        }
        if actual != expected or any(
            set(item.matched_paper_candidate_ids) - allowed_matches for item in result.assessments
        ):
            raise StructuredDistillationError("生成力验证没有覆盖准确的候选集合")
        return result

    def _validate_exclusivity(
        self,
        *,
        run_id: str,
        domain: str,
        clusters: list[CandidateCluster],
        control_profiles: list[PaperProfile],
        control_hash: str | None,
    ) -> ExclusivityBatchResult | None:
        if not control_profiles or control_hash is None:
            return None
        candidate_payload = [item.model_dump(mode="json") for item in clusters]
        control_payload = [item.model_dump(mode="json") for item in control_profiles]
        input_hash = _hash_payload(
            self.EXCLUSIVITY_VERSION, domain, control_hash, candidate_payload, control_payload
        )
        result = self.repository.load_stage_result(
            run_id=run_id, stage="exclusivity_validation", item_id="global",
            model=ExclusivityBatchResult,
        )
        if result is None:
            result = self._structured_call(
                exclusivity_validation_messages(
                    domain=domain,
                    candidates=candidate_payload,
                    control_profiles=control_payload,
                ),
                ExclusivityBatchResult,
                seed=37,
                step_id="distill.exclusivity_validation",
                validator=lambda value: self._validate_exclusivity_result(
                    value,
                    clusters,
                    control_profiles,
                ),
            )
        self.repository.save_stage_result(
            run_id=run_id,
            stage="exclusivity_validation",
            item_id="global",
            input_hash=input_hash,
            result=result,
        )
        return result

    @staticmethod
    def _validate_exclusivity_result(
        result: ExclusivityBatchResult,
        clusters: list[CandidateCluster],
        control_profiles: list[PaperProfile],
    ) -> ExclusivityBatchResult:
        expected = {item.candidate_id for item in clusters}
        actual = {item.candidate_id for item in result.assessments}
        allowed_matches = {
            item.paper_candidate_id for profile in control_profiles for item in profile.candidates
        }
        if actual != expected or any(
            set(item.matched_paper_candidate_ids) - allowed_matches for item in result.assessments
        ):
            raise StructuredDistillationError("排他性验证没有覆盖准确的候选集合")
        return result

    def _structured_call(
        self,
        messages: list[dict[str, str]],
        model: type[StructuredModel],
        *,
        seed: int,
        step_id: str,
        validator: Callable[[StructuredModel], StructuredModel] | None = None,
    ) -> StructuredModel:
        last_error = "未知校验错误"
        for attempt in range(2):
            active = messages
            if attempt:
                active = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "上一次输出违反结构或证据契约。请返回修正后的完整 JSON。"
                            f"校验错误：{last_error}"
                        ),
                    },
                ]
            response = self.siliconflow.chat(
                active,
                thinking=True,
                reasoning_effort="high",
                temperature=0.0,
                max_tokens=8192,
                seed=seed,
                response_format="json_object",
                use_cache=True,
                request_attempts=2,
                stream=True,
                step_id=step_id,
            )
            try:
                parsed = model.model_validate(json.loads(response.content))
                validate_academic_language(parsed)
                if validator is not None:
                    parsed = validator(parsed)
                return parsed
            except (
                json.JSONDecodeError,
                ValidationError,
                OutputLanguageError,
                StructuredDistillationError,
            ) as exc:
                last_error = str(exc)[:2000]
        raise StructuredDistillationError(f"非虚构蒸馏结构化调用失败：{last_error}")


def choose_holdout_doc_ids(source_info: tuple[SourceInfo, ...]) -> list[str]:
    """按语料规模确定最多两个稳定留出项，不固定使用最后一篇。"""

    count = len(source_info)
    if count <= 3:
        return []
    holdout_count = 1 if count <= 7 else 2
    ordered = sorted(source_info, key=lambda item: (item.title.casefold(), item.doc_id))
    if holdout_count == 1:
        return [ordered[count // 2].doc_id]
    return [ordered[count // 3].doc_id, ordered[(2 * count) // 3].doc_id]


def _hash_payload(*values: object) -> str:
    canonical = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
