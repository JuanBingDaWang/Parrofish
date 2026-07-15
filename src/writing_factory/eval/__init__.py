"""Retrieval, traceability, generation, and style evaluation.

Modules:
    - retrieval:  Recall / precision metrics for retrieval evaluation
    - faithfulness: RAGAS-style faithfulness evaluator for generation
    - llm_judge:  LLM-as-judge for soft quality dimensions
    - golden:     Golden regression set framework
    - injection:  Prompt injection detection and defence
    - run_eval:   Unified evaluation runner with persistence
"""

from writing_factory.eval.faithfulness import FaithfulnessEvaluator, faithfulness
from writing_factory.eval.golden import GoldenRunner, GoldenSuite, load_golden_suite
from writing_factory.eval.injection import InjectionDetector, PromptHardening
from writing_factory.eval.llm_judge import LLMJudge, judge_draft
from writing_factory.eval.models import (
    AtomicClaim,
    CitationTraceabilityResult,
    FaithfulnessResult,
    GoldenCase,
    GoldenRunResult,
    GoldenSuiteResult,
    InjectionVerdict,
    JudgeDimension,
    JudgeResult,
)
from writing_factory.eval.retrieval import (
    RecallCase,
    evidence_recall_at_k,
    parent_hit_rate,
    precision_at_k,
    recall_at_k,
)
from writing_factory.eval.run_eval import EvaluationRunner
from writing_factory.eval.traceability import (
    evaluate_citation_traceability,
    evidence_context_from_state,
)

__all__ = [
    # retrieval
    "RecallCase",
    "evidence_recall_at_k",
    "parent_hit_rate",
    "precision_at_k",
    "recall_at_k",
    # faithfulness
    "AtomicClaim",
    "FaithfulnessResult",
    "CitationTraceabilityResult",
    "FaithfulnessEvaluator",
    "faithfulness",
    # llm-judge
    "JudgeDimension",
    "JudgeResult",
    "LLMJudge",
    "judge_draft",
    # golden
    "GoldenCase",
    "GoldenRunResult",
    "GoldenSuiteResult",
    "GoldenSuite",
    "GoldenRunner",
    "load_golden_suite",
    "evaluate_citation_traceability",
    "evidence_context_from_state",
    # injection
    "InjectionVerdict",
    "InjectionDetector",
    "PromptHardening",
    # runner
    "EvaluationRunner",
]
