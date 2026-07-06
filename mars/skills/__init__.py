"""Self-compressing hypothesis skills.

This package is intentionally benchmark-agnostic. It treats a skill as an
executable hypothesis constructor with a falsification contract, then compresses
many local skills into a smaller evolving hypothesis grammar.
"""

from .grammar import (
    AntiUnification,
    ResidualCase,
    SkillContract,
    SkillGrammar,
    SkillInductionReport,
    anti_unify,
    expr_to_text,
)
from .metric_compiler import (
    CompiledHypothesis,
    ContractScore,
    HypothesisCandidate,
    HypothesisWorkbenchReport,
    InterfaceCheck,
    InterfaceContract,
    InterfaceObservation,
    InterfaceResidual,
    InterfaceScore,
    MetricContract,
    SlotResidual,
    SlotScore,
    compile_interface_contract,
    compile_hypothesis_workbench,
    discoverybench_contracts,
    infer_temporal_event_hypothesis,
    residuals_from_score,
    score_contract,
    score_interface_contract,
)
from .slot_contract import (
    SlotCandidate,
    SlotContractResult,
    infer_slot_contract_hypothesis,
)
from .answer_slot_compiler import (
    infer_answer_slot_hypothesis,
)
from .answer_plan_inducer import (
    AnswerPlan,
    infer_answer_plan_hypothesis,
)
from .abstraction_posterior import (
    ContrastAbstraction,
    MeasuredGroup,
    induce_contrast_abstraction,
)
from .scope_abstraction import (
    MeasuredPairGroup,
    ScopeAbstraction,
    induce_scope_abstraction,
)
from .question_form_inducer import (
    QuestionForm,
    infer_question_form,
)
from .problem_frame_inducer import (
    ProblemFrame,
    infer_problem_frame_hypothesis,
)
from .contrastive_world_inducer import (
    CounterfactualWorld,
    infer_contrastive_world_hypothesis,
)
from .universal_slot_compiler import (
    UniversalSlotCompiler,
    UniversalSlotPlan,
    UniversalSlotResult,
    UniversalSlotTask,
    compile_universal_slot_plan,
    infer_universal_slot_hypothesis,
)
from .evidence_contract_compiler import (
    EvidenceContract,
    EvidenceContractResult,
    EvidenceCoverage,
    EvidenceSlot,
    compile_evidence_contract,
    infer_evidence_contract_hypothesis,
    score_evidence_coverage,
)
from .supermetrics import (
    SuperMetricReport,
    aggregate_supermetrics,
    compute_cpi_supermetrics,
    official_gap_metric,
)
from .interface_profiler import (
    InterfaceProfile,
    profile_observations,
    profile_prompt,
)
from .task_contract import (
    FileContract,
    TaskContract,
    compile_task_contract,
    task_contract_prompt,
)
from .contract_baselines import (
    ContractBaseline,
    generate_auto_baseline_program,
    generate_contract_baselines,
    source_target_tokens,
)
from .benchmark_theory import (
    ActionSpec,
    BenchmarkTheory,
    CheckSpec,
    MetricSpec,
    ResidualSpec,
    TheorySkillSchema,
    compile_benchmark_theory,
)
from .self_layer_registry import (
    SelfLayerRecord,
    SelfLayerRegistry,
    TrustedSelfLayerRecord,
    validate_self_layer_source,
)
from .self_module_registry import (
    SelfModuleRecord,
    SelfModuleRegistry,
    TrustedSelfModuleRecord,
    validate_self_module_source,
)
from .residual_kernel import (
    EvidencePlan,
    HypothesisSketch,
    InducedProgram,
    ResidualFingerprint,
    ResidualKernel,
    RewriteRule,
    SketchExpansion,
    TypedTerm,
    compress_rewrite_rules,
)

__all__ = [
    "AntiUnification",
    "ActionSpec",
    "BenchmarkTheory",
    "CheckSpec",
    "CompiledHypothesis",
    "ContractBaseline",
    "ContractScore",
    "HypothesisCandidate",
    "HypothesisWorkbenchReport",
    "InterfaceProfile",
    "QuestionForm",
    "ProblemFrame",
    "CounterfactualWorld",
    "InterfaceCheck",
    "InterfaceContract",
    "InterfaceObservation",
    "InterfaceResidual",
    "InterfaceScore",
    "EvidencePlan",
    "EvidenceContract",
    "EvidenceContractResult",
    "EvidenceCoverage",
    "EvidenceSlot",
    "FileContract",
    "HypothesisSketch",
    "InducedProgram",
    "MetricSpec",
    "MetricContract",
    "ResidualCase",
    "ResidualFingerprint",
    "ResidualSpec",
    "ResidualKernel",
    "RewriteRule",
    "SketchExpansion",
    "SelfLayerRecord",
    "SelfLayerRegistry",
    "SelfModuleRecord",
    "SelfModuleRegistry",
    "SlotResidual",
    "SlotScore",
    "SlotCandidate",
    "SlotContractResult",
    "SuperMetricReport",
    "TaskContract",
    "SkillContract",
    "SkillGrammar",
    "SkillInductionReport",
    "TrustedSelfLayerRecord",
    "TrustedSelfModuleRecord",
    "TypedTerm",
    "TheorySkillSchema",
    "UniversalSlotCompiler",
    "UniversalSlotPlan",
    "UniversalSlotResult",
    "UniversalSlotTask",
    "anti_unify",
    "aggregate_supermetrics",
    "compile_interface_contract",
    "compile_hypothesis_workbench",
    "compile_task_contract",
    "compile_evidence_contract",
    "compile_benchmark_theory",
    "compress_rewrite_rules",
    "discoverybench_contracts",
    "expr_to_text",
    "generate_auto_baseline_program",
    "generate_contract_baselines",
    "infer_temporal_event_hypothesis",
    "infer_answer_slot_hypothesis",
    "AnswerPlan",
    "infer_answer_plan_hypothesis",
    "ContrastAbstraction",
    "MeasuredGroup",
    "induce_contrast_abstraction",
    "MeasuredPairGroup",
    "ScopeAbstraction",
    "induce_scope_abstraction",
    "infer_question_form",
    "infer_problem_frame_hypothesis",
    "infer_contrastive_world_hypothesis",
    "infer_universal_slot_hypothesis",
    "infer_evidence_contract_hypothesis",
    "compile_universal_slot_plan",
    "infer_slot_contract_hypothesis",
    "compute_cpi_supermetrics",
    "official_gap_metric",
    "profile_observations",
    "profile_prompt",
    "residuals_from_score",
    "score_contract",
    "score_evidence_coverage",
    "score_interface_contract",
    "source_target_tokens",
    "task_contract_prompt",
    "validate_self_layer_source",
    "validate_self_module_source",
]
