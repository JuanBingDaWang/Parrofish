"""Typed model selections and per-step SiliconFlow chat profiles."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

StepGroup = Literal["distill", "retrieval", "chat", "writing", "evaluation"]
ReasoningEffortSetting = Literal["auto", "high", "max"]


class ChatStepConfig(BaseModel):
    """Validated settings for one logical text-generation stage."""

    model_config = ConfigDict(frozen=True)

    temperature: float = Field(ge=0.0, le=2.0)
    thinking: bool | None = Field(description="None 表示沿用该步骤的流程默认值")
    reasoning_effort: ReasoningEffortSetting = "auto"
    max_tokens: int = Field(ge=256, le=65536)
    stream: bool
    retry_count: int = Field(ge=0, le=5)
    timeout_seconds: int | None = Field(default=None, ge=60, le=3600)


class ChatStepDefinition(BaseModel):
    """Stable UI and persistence identity for one logical LLM stage."""

    model_config = ConfigDict(frozen=True)

    step_id: str
    group: StepGroup
    name: str
    description: str
    default: ChatStepConfig
    framework_token_escalation: bool = False


class ModelSelections(BaseModel):
    """Active SiliconFlow model IDs and a pending embedding migration."""

    model_config = ConfigDict(frozen=True)

    chat_model: str
    embedding_model: str
    rerank_model: str
    pending_embedding_model: str | None = None


class ModelCatalogEntry(BaseModel):
    """One item returned by SiliconFlow's model list endpoint."""

    model_config = ConfigDict(frozen=True)

    id: str
    object: str = "model"
    created: int | None = None
    owned_by: str = ""


def _profile(
    temperature: float,
    thinking: bool | None,
    *,
    max_tokens: int = 8192,
    stream: bool = True,
    retries: int = 2,
    effort: ReasoningEffortSetting = "auto",
) -> ChatStepConfig:
    return ChatStepConfig(
        temperature=temperature,
        thinking=thinking,
        reasoning_effort=effort,
        max_tokens=max_tokens,
        stream=stream,
        retry_count=retries,
        timeout_seconds=None,
    )


STEP_DEFINITIONS: tuple[ChatStepDefinition, ...] = (
    ChatStepDefinition(
        step_id="distill.map",
        group="distill",
        name="Map 提取",
        description="从单个语料单元提取候选模型、启发式和表达特征。",
        default=_profile(0.0, True, retries=0, effort="high"),
    ),
    ChatStepDefinition(
        step_id="distill.structure_map",
        group="distill",
        name="完整文档谋篇分析",
        description="按原始顺序提取全文、章节、段落、句群和过渡结构。",
        default=_profile(0.0, True, retries=1, effort="high"),
    ),
    ChatStepDefinition(
        step_id="distill.structure_reduce",
        group="distill",
        name="谋篇 DNA 归并",
        description="按非虚构文体归并跨文档复现的结构模式与允许变体。",
        default=_profile(0.0, True, retries=1, effort="high"),
    ),
    ChatStepDefinition(
        step_id="distill.paper_profile",
        group="distill",
        name="单篇文档画像",
        description="把一篇文档的候选证据归并为稳定的非虚构写作画像。",
        default=_profile(0.0, True, retries=1, effort="high"),
    ),
    ChatStepDefinition(
        step_id="distill.cluster",
        group="distill",
        name="跨文档聚类",
        description="在多篇文档之间聚合同一写作机制的候选项。",
        default=_profile(0.0, True, retries=1, effort="high"),
    ),
    ChatStepDefinition(
        step_id="distill.generative_validation",
        group="distill",
        name="生成力验证",
        description="验证候选模型能否预测留出文档的信息选择与组织路径。",
        default=_profile(0.0, True, retries=1, effort="high"),
    ),
    ChatStepDefinition(
        step_id="distill.exclusivity_validation",
        group="distill",
        name="排他性验证",
        description="与控制语料比较，判断候选项是否具有作者个性。",
        default=_profile(0.0, True, retries=1, effort="high"),
    ),
    ChatStepDefinition(
        step_id="distill.reduce",
        group="distill",
        name="Reduce 汇总",
        description="汇总全部候选并生成 PersonaSpec；思考模式默认按蒸馏类型决定。",
        default=_profile(0.0, None, retries=1, effort="high"),
    ),
    ChatStepDefinition(
        step_id="distill.academic_supplement",
        group="distill",
        name="作者档案补充",
        description="补充非虚构写作模型的主题、证据和表达特征。",
        default=_profile(0.0, False, max_tokens=10000, stream=False, retries=1),
    ),
    ChatStepDefinition(
        step_id="distill.fidelity_design",
        group="distill",
        name="档案测试题生成",
        description="根据证据和档案设计忠实度测试题。",
        default=_profile(0.0, True, retries=0, effort="high"),
    ),
    ChatStepDefinition(
        step_id="distill.fidelity_answer",
        group="distill",
        name="档案盲测回答",
        description="只使用 Persona 档案回答测试题。",
        default=_profile(0.2, False, retries=0),
    ),
    ChatStepDefinition(
        step_id="distill.fidelity_judge",
        group="distill",
        name="档案中性评判",
        description="由独立中性角色根据证据评判档案回答。",
        default=_profile(0.0, True, retries=0, effort="high"),
    ),
    ChatStepDefinition(
        step_id="retrieval.query_rewrite",
        group="retrieval",
        name="查询改写",
        description="把抽象问题扩展为多个具体检索查询。",
        default=_profile(0.2, False),
    ),
    ChatStepDefinition(
        step_id="retrieval.hyde",
        group="retrieval",
        name="HyDE",
        description="生成假设性答案，用于提高向量检索召回率。",
        default=_profile(0.3, False),
    ),
    ChatStepDefinition(
        step_id="chat.reply",
        group="chat",
        name="作者对话回答",
        description="使用无证据作者档案、最近对话和可选检索证据生成流式回答。",
        default=_profile(0.6, False, retries=1),
    ),
    ChatStepDefinition(
        step_id="chat.summary",
        group="chat",
        name="对话历史摘要",
        description="中性压缩较早轮次，保留承诺、问题和对话上下文，不充当事实来源。",
        default=_profile(0.0, False, max_tokens=4096, stream=False, retries=1),
    ),
    ChatStepDefinition(
        step_id="chat.verify",
        group="chat",
        name="对话回答核验",
        description="按需由不带作者档案的中性角色核对回答中的事实性陈述。",
        default=_profile(0.0, False, stream=False, retries=1),
    ),
    ChatStepDefinition(
        step_id="writing.topic",
        group="writing",
        name="选题锐化",
        description="结合作者档案和知识库建立中心论旨。",
        default=_profile(0.3, True, effort="high"),
    ),
    ChatStepDefinition(
        step_id="writing.framework",
        group="writing",
        name="提纲构建",
        description="生成带修辞目的、术语和证据候选的论证提纲。",
        default=_profile(0.3, True, effort="high"),
        framework_token_escalation=True,
    ),
    ChatStepDefinition(
        step_id="writing.draft",
        group="writing",
        name="内容单元起草与修订",
        description="根据冻结证据包起草或修订结构化内容单元。",
        default=_profile(0.5, False, retries=1),
    ),
    ChatStepDefinition(
        step_id="writing.verify",
        group="writing",
        name="事实核验",
        description="由不带 Persona 的中性角色核对事实论断。",
        default=_profile(0.0, False),
    ),
    ChatStepDefinition(
        step_id="writing.section_polish",
        group="writing",
        name="内容单元文风打磨",
        description="在事实核验后按目标文体施加作者表达 DNA。",
        default=_profile(0.7, True, effort="high"),
    ),
    ChatStepDefinition(
        step_id="writing.section_drift",
        group="writing",
        name="内容单元防漂移",
        description="由中性角色检查内容单元打磨是否改变事实。",
        default=_profile(0.0, False),
    ),
    ChatStepDefinition(
        step_id="writing.term_review",
        group="writing",
        name="术语审查",
        description="检查全文术语定义和使用的一致性。",
        default=_profile(0.0, False),
    ),
    ChatStepDefinition(
        step_id="writing.structure_review",
        group="writing",
        name="结构审查",
        description="检查论证结构、衔接和论点支撑关系。",
        default=_profile(0.0, False),
    ),
    ChatStepDefinition(
        step_id="writing.global_polish",
        group="writing",
        name="全局打磨",
        description="统一全文过渡、语气和表达。",
        default=_profile(0.1, True, effort="high"),
    ),
    ChatStepDefinition(
        step_id="writing.global_drift",
        group="writing",
        name="全局防漂移",
        description="核对全局打磨是否改变已核验事实。",
        default=_profile(0.0, False),
    ),
    ChatStepDefinition(
        step_id="evaluation.claim_decomposition",
        group="evaluation",
        name="论断拆分",
        description="把待评估文本拆分为可独立核验的原子论断。",
        default=_profile(0.0, False),
    ),
    ChatStepDefinition(
        step_id="evaluation.claim_support",
        group="evaluation",
        name="论断支持度判断",
        description="根据提供的证据判断每条原子论断是否受支持。",
        default=_profile(0.0, False),
    ),
    ChatStepDefinition(
        step_id="evaluation.injection",
        group="evaluation",
        name="提示注入检测",
        description="判断文本是否含有试图改变系统行为的指令。",
        default=_profile(0.0, False),
    ),
    ChatStepDefinition(
        step_id="evaluation.llm_judge",
        group="evaluation",
        name="LLM 综合评判",
        description="按照目标非虚构文体的量规对成稿做中性综合评分。",
        default=_profile(0.0, False),
    ),
)

STEP_BY_ID = {definition.step_id: definition for definition in STEP_DEFINITIONS}


def get_step_definition(step_id: str) -> ChatStepDefinition:
    try:
        return STEP_BY_ID[step_id]
    except KeyError as exc:
        raise ValueError(f"未知的 SiliconFlow 步骤：{step_id}") from exc
