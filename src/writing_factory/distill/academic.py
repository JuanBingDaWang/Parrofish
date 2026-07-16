"""非虚构作者蒸馏的文档级候选、验证与选择数据契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AcademicOperation = Literal[
    "problem_framing",
    "conceptualization",
    "evidence_selection",
    "argument_structure",
    "counterargument",
    "boundary_setting",
    "expression",
]
AttributionScope = Literal["author_specific", "coauthored_voice", "uncertain"]
Specificity = Literal[
    "author_distinctive",
    "field_conventional",
    "general_academic",
    "general_nonfiction",
    "unverified",
]
RecurrenceLevel = Literal["provisional", "basic", "high"]
ValidationStatus = Literal["passed", "failed", "not_tested"]


class PaperMentalCandidate(BaseModel):
    """一篇文档内部归并后的可复用非虚构写作操作。"""

    model_config = ConfigDict(frozen=True)

    paper_candidate_id: str = Field(description="程序提供或返回的稳定单篇候选标识")
    map_candidate_ids: list[str] = Field(
        min_length=1, description="被归并到该候选的局部 Map 候选标识"
    )
    operation: AcademicOperation = Field(description="候选所属的非虚构写作操作类别")
    name: str = Field(description="候选操作的简体中文名称")
    description: str = Field(description="候选操作如何运行的简体中文描述")
    evidence_ids: list[str] = Field(min_length=1, description="支持候选的登记证据标识")
    applicability: str = Field(description="该操作适用的问题和条件，使用简体中文")
    limits: str = Field(description="该操作的失效条件，使用简体中文")
    research_context: str = Field(description="该文档中出现此操作的写作情境")


class PaperProfile(BaseModel):
    """单篇文档画像；同一文档的多个 Map 单元在此只计一次。"""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(description="必须与输入文档标识完全一致")
    candidates: list[PaperMentalCandidate] = Field(
        default_factory=list, description="该文档中有证据支持的非虚构写作操作"
    )


class CandidateCluster(BaseModel):
    """跨文档聚类得到的候选模型，还没有通过中性验证。"""

    model_config = ConfigDict(frozen=True)

    candidate_id: str = Field(description="全局候选的稳定标识")
    operation: AcademicOperation = Field(description="候选所属的非虚构写作操作类别")
    name: str = Field(description="候选模型的简体中文名称")
    description: str = Field(description="候选模型如何运行的简体中文描述")
    paper_candidate_ids: list[str] = Field(min_length=1, description="被聚入该候选的单篇候选标识")
    evidence_ids: list[str] = Field(min_length=1, description="候选引用的登记证据标识")
    applicability: str = Field(description="适用问题和触发条件，使用简体中文")
    limits: str = Field(description="失效条件和适用边界，使用简体中文")
    attribution_scope: AttributionScope = Field(description="作者归属判断")
    attribution_rationale: str = Field(description="作者归属判断的简体中文理由")


class CandidateClusterResult(BaseModel):
    """中性聚类器输出的完整候选集合。"""

    model_config = ConfigDict(frozen=True)

    candidates: list[CandidateCluster] = Field(default_factory=list)


class CandidateAssessment(BaseModel):
    """对一个候选在留出或对照语料上的中性判断。"""

    model_config = ConfigDict(frozen=True)

    candidate_id: str = Field(description="必须复制输入中的候选标识")
    status: ValidationStatus = Field(description="通过、失败或没有条件测试")
    rationale: str = Field(description="只根据提供材料形成的简体中文理由")
    matched_paper_candidate_ids: list[str] = Field(
        default_factory=list, description="支持判断的单篇候选标识"
    )

    @model_validator(mode="after")
    def require_match_for_pass(self) -> CandidateAssessment:
        if self.status == "passed" and not self.matched_paper_candidate_ids:
            raise ValueError("生成力通过时必须提供留出文档中的匹配候选")
        return self


class ValidationBatchResult(BaseModel):
    """一次批量中性验证输出。"""

    model_config = ConfigDict(frozen=True)

    assessments: list[CandidateAssessment] = Field(default_factory=list)


class ExclusivityAssessment(BaseModel):
    """候选相对于可选同领域对照语料的区分度判断。"""

    model_config = ConfigDict(frozen=True)

    candidate_id: str = Field(description="必须复制输入中的候选标识")
    specificity: Specificity = Field(description="作者独特、领域惯例、通用规范或无法判断")
    rationale: str = Field(description="只根据对照材料形成的简体中文理由")
    matched_paper_candidate_ids: list[str] = Field(
        default_factory=list, description="对照语料中的相似单篇候选标识"
    )


class ExclusivityBatchResult(BaseModel):
    """一次批量排他性验证输出。"""

    model_config = ConfigDict(frozen=True)

    assessments: list[ExclusivityAssessment] = Field(default_factory=list)


class AcademicModelValidation(BaseModel):
    """由代码汇总复现计数和独立验证结果，不由最终 Reduce 自报。"""

    model_config = ConfigDict(frozen=True)

    supporting_doc_ids: list[str] = Field(min_length=1)
    recurrence_document_count: int = Field(ge=1)
    recurrence_level: RecurrenceLevel
    generative_status: ValidationStatus
    generative_rationale: str
    specificity: Specificity
    exclusivity_rationale: str
    control_corpus_used: bool
    counterexamples: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_recurrence(self) -> AcademicModelValidation:
        if self.recurrence_document_count != len(set(self.supporting_doc_ids)):
            raise ValueError("复现文档数必须等于去重后的 supporting_doc_ids 数量")
        expected = (
            "high"
            if self.recurrence_document_count >= 3
            else "basic"
            if self.recurrence_document_count >= 2
            else "provisional"
        )
        if self.recurrence_level != expected:
            raise ValueError("复现等级与覆盖文档数不一致")
        return self

    @property
    def eligible(self) -> bool:
        """至少跨两篇文档复现且没有生成力反证才可进入模型列表。"""

        return self.recurrence_document_count >= 2 and self.generative_status != "failed"


class CandidateRecord(BaseModel):
    """可持久化的候选登记项，包含证据、计数和验证结论。"""

    model_config = ConfigDict(frozen=True)

    candidate: CandidateCluster
    validation: AcademicModelValidation
    selected_as: Literal["core", "convention", "heuristic", "discarded"] = "discarded"
    selection_rank: int | None = Field(default=None, ge=1)


class CandidateRegistry(BaseModel):
    """一次蒸馏中全部候选的确定性登记表。"""

    model_config = ConfigDict(frozen=True)

    target_doc_ids: list[str] = Field(min_length=1)
    holdout_doc_ids: list[str] = Field(default_factory=list)
    control_doc_ids: list[str] = Field(default_factory=list)
    domain: str = ""
    records: list[CandidateRecord] = Field(default_factory=list)
