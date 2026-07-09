"""Universal hypothesis kernel.

The kernel is the single search object that turns a task interface into
candidate artifacts with typed holes, executable evidence, and a posterior
score.  Domain-specific routines are operators inside the search, not layers
stacked outside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Mapping, Protocol

from mars.induction.estimand_synthesizer import infer_estimand_hypothesis
from mars.induction.metamorphic_estimand_kernel import evaluate_metamorphic_estimand
from mars.skills.evidence_contract_compiler import (
    EvidenceContract,
    compile_evidence_contract,
    infer_evidence_contract_hypothesis,
    score_evidence_coverage,
)
from mars.skills.slot_contract import SlotContractResult, infer_slot_contract_hypothesis
from mars.skills.answer_plan_inducer import infer_answer_plan_hypothesis
from mars.skills.universal_slot_compiler import infer_universal_slot_hypothesis
from mars.skills.problem_frame_inducer import infer_problem_frame_hypothesis
from mars.skills.contrastive_world_inducer import infer_contrastive_world_hypothesis


@dataclass(frozen=True)
class TypedHole:
    name: str
    type_name: str = "unknown"
    required: bool = True
    value: Any = None
    evidence: str = ""


@dataclass(frozen=True)
class KernelTask:
    task_text: str
    interface_kind: str
    data: Any = None
    domain_context: str = ""
    schema: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KernelCandidate:
    hypothesis: str
    workflow: str
    evidence: str
    answer_form: str
    holes: tuple[TypedHole, ...]
    operator: str
    source: str
    evidence_score: float
    complexity: float
    coverage: float
    loss: float
    posterior: float
    metamorphic_loss: float = 0.0
    metamorphic_signatures: tuple[str, ...] = ()
    metamorphic_trace: str = ""

    def report(self) -> str:
        covered = ",".join(h.name for h in self.holes if h.value not in (None, "", [], {}))
        missing = ",".join(h.name for h in self.holes if h.required and h.value in (None, "", [], {})) or "none"
        metamorphic = (
            f" metamorphic_loss={self.metamorphic_loss:.3f}, "
            f"metamorphic_signatures={','.join(self.metamorphic_signatures) or 'none'}."
            if self.metamorphic_trace
            else ""
        )
        return (
            f"HYPOTHESIS: {self.hypothesis}\n"
            f"WORKFLOW SUMMARY: UniversalHypothesisKernel operator={self.operator}, "
            f"answer_form={self.answer_form}, source={self.source}, posterior={self.posterior:.3f}, "
            f"coverage={self.coverage:.3f}, complexity={self.complexity:.3f}, "
            f"covered_holes={covered}, missing_holes={missing}.{metamorphic} {self.workflow} "
            f"Evidence: {self.evidence}."
        )


@dataclass(frozen=True)
class KernelResult:
    task: KernelTask
    contract: EvidenceContract
    candidates: tuple[KernelCandidate, ...]

    @property
    def best(self) -> KernelCandidate | None:
        return self.candidates[0] if self.candidates else None


class KernelOperator(Protocol):
    name: str
    complexity: float

    def propose(self, task: KernelTask, contract: EvidenceContract) -> list[KernelCandidate]:
        ...


class UniversalHypothesisKernel:
    """One posterior search over artifact-producing operators."""

    def __init__(self, operators: list[KernelOperator] | None = None):
        self.operators = operators or default_kernel_operators()

    def run(self, task: KernelTask) -> KernelResult:
        contract = compile_evidence_contract(
            task.task_text,
            interface_kind=task.interface_kind,
            schema_text=_schema_text(task.schema),
        )
        candidates: list[KernelCandidate] = []
        for operator in self.operators:
            try:
                candidates.extend(operator.propose(task, contract))
            except Exception:
                continue
        candidates = [c for c in candidates if c.hypothesis.strip()]
        candidates = [_apply_metamorphic_rescore(task, contract, c) for c in candidates]
        candidates.sort(key=lambda c: c.posterior, reverse=True)
        return KernelResult(task=task, contract=contract, candidates=tuple(candidates))


class EvidenceContractOperator:
    name = "evidence_contract"
    complexity = 1.0

    def propose(self, task: KernelTask, contract: EvidenceContract) -> list[KernelCandidate]:
        result = infer_evidence_contract_hypothesis(
            question=task.task_text,
            domain_context=task.domain_context,
            data=task.data,
            schema=task.schema,
            interface_kind=task.interface_kind,
        )
        if result is None or not result.coverage.accepted:
            return []
        return [
            _candidate_from_parts(
                contract=contract,
                hypothesis=result.hypothesis,
                workflow=result.workflow,
                evidence=result.evidence,
                slots={name: name for name in result.coverage.covered_slots},
                answer_form=result.contract.answer_form,
                operator=self.name,
                source=result.source,
                evidence_score=result.score,
                complexity=self.complexity,
            )
        ]


class UniversalSlotOperator:
    name = "universal_slot"
    complexity = 1.4

    def propose(self, task: KernelTask, contract: EvidenceContract) -> list[KernelCandidate]:
        result = infer_universal_slot_hypothesis(
            task_text=task.task_text,
            interface_kind=task.interface_kind,
            data=task.data,
            domain_context=task.domain_context,
            schema=task.schema,
            metadata=task.metadata,
        )
        if result is None:
            return []
        return [_candidate_from_slot_result(contract, result, self.name, task.interface_kind, self.complexity)]


class TabularAnswerPlanOperator:
    name = "answer_plan"
    complexity = 1.7

    def propose(self, task: KernelTask, contract: EvidenceContract) -> list[KernelCandidate]:
        return _map_tables(task, contract, self.name, self.complexity, infer_answer_plan_hypothesis)


class SlotContractOperator:
    name = "slot_contract"
    complexity = 1.5

    def propose(self, task: KernelTask, contract: EvidenceContract) -> list[KernelCandidate]:
        def call(*, question: str, domain_context: str, df: Any, column_descriptions: Mapping[str, str]) -> SlotContractResult | None:
            return infer_slot_contract_hypothesis(
                question=question,
                domain_context=domain_context,
                df=df,
                column_descriptions=column_descriptions,
            )

        return _map_tables(task, contract, self.name, self.complexity, call)


class ProblemFrameOperator:
    name = "problem_frame"
    complexity = 2.0

    def propose(self, task: KernelTask, contract: EvidenceContract) -> list[KernelCandidate]:
        result = infer_problem_frame_hypothesis(
            question=task.task_text,
            domain_context=task.domain_context,
            data=task.data,
            column_descriptions=task.schema,
        )
        if result is None:
            return []
        return [
            _candidate_from_parts(
                contract=contract,
                hypothesis=result.hypothesis,
                workflow=result.workflow,
                evidence=result.evidence,
                slots=result.slots,
                answer_form=str(result.slots.get("answer_form", contract.answer_form)),
                operator=self.name,
                source="interface",
                evidence_score=result.score,
                complexity=self.complexity,
            )
        ]


class ContrastiveWorldOperator:
    name = "contrastive_world"
    complexity = 2.1

    def propose(self, task: KernelTask, contract: EvidenceContract) -> list[KernelCandidate]:
        result = infer_contrastive_world_hypothesis(
            question=task.task_text,
            domain_context=task.domain_context,
            data=task.data,
            column_descriptions=task.schema,
        )
        if result is None:
            return []
        return [
            _candidate_from_parts(
                contract=contract,
                hypothesis=result.hypothesis,
                workflow=result.workflow,
                evidence=result.evidence,
                slots=result.slots,
                answer_form=str(result.slots.get("answer_form", contract.answer_form)),
                operator=self.name,
                source="interface",
                evidence_score=result.score,
                complexity=self.complexity,
            )
        ]


class EstimandSynthesizerOperator:
    name = "estimand_synthesizer"
    complexity = 2.2

    def propose(self, task: KernelTask, contract: EvidenceContract) -> list[KernelCandidate]:
        result = infer_estimand_hypothesis(
            question=task.task_text,
            domain_context=task.domain_context,
            data=task.data,
            schema=task.schema,
        )
        if result is None:
            return []
        return [_candidate_from_slot_result(contract, result, self.name, "interface", self.complexity)]


def default_kernel_operators() -> list[KernelOperator]:
    return [
        EvidenceContractOperator(),
        UniversalSlotOperator(),
        TabularAnswerPlanOperator(),
        SlotContractOperator(),
        ProblemFrameOperator(),
        EstimandSynthesizerOperator(),
        ContrastiveWorldOperator(),
    ]


def _map_tables(
    task: KernelTask,
    contract: EvidenceContract,
    operator: str,
    complexity: float,
    fn: Any,
) -> list[KernelCandidate]:
    rows: list[KernelCandidate] = []
    if isinstance(task.data, Mapping):
        schemas = task.schema if isinstance(task.schema, Mapping) else {}
        for name, df in task.data.items():
            schema = schemas.get(name, {}) if isinstance(schemas.get(name, {}), Mapping) else {}
            result = fn(
                question=task.task_text,
                domain_context=task.domain_context,
                df=df,
                column_descriptions={str(k): str(v) for k, v in schema.items()},
            )
            if result is not None:
                rows.append(_candidate_from_slot_result(contract, result, operator, str(name), complexity))
    else:
        result = fn(
            question=task.task_text,
            domain_context=task.domain_context,
            df=task.data,
            column_descriptions={str(k): str(v) for k, v in task.schema.items()},
        )
        if result is not None:
            rows.append(_candidate_from_slot_result(contract, result, operator, "table", complexity))
    return rows


def _candidate_from_slot_result(
    contract: EvidenceContract,
    result: SlotContractResult,
    operator: str,
    source: str,
    complexity: float,
) -> KernelCandidate:
    return _candidate_from_parts(
        contract=contract,
        hypothesis=result.hypothesis,
        workflow=result.workflow,
        evidence=result.evidence,
        slots=result.slots,
        answer_form=str(result.slots.get("answer_form", contract.answer_form)),
        operator=operator,
        source=source,
        evidence_score=result.score,
        complexity=complexity,
    )


def _candidate_from_parts(
    *,
    contract: EvidenceContract,
    hypothesis: str,
    workflow: str,
    evidence: str,
    slots: Mapping[str, Any],
    answer_form: str,
    operator: str,
    source: str,
    evidence_score: float,
    complexity: float,
) -> KernelCandidate:
    report = f"{hypothesis}\n{workflow}\n{evidence}"
    coverage = score_evidence_coverage(contract, report, slots)
    holes = tuple(
        TypedHole(
            name=slot.name,
            type_name=_hole_type(slot.name),
            required=slot.required,
            value=_slot_value(slot.name, slots, report),
            evidence=evidence,
        )
        for slot in contract.slots
    )
    loss = 1.0 - coverage.score
    answer_form_match = _answer_form_match(contract.answer_form, answer_form)
    posterior = _posterior(
        evidence_score=evidence_score,
        coverage=coverage.score,
        complexity=complexity,
        loss=loss,
        answer_form_match=answer_form_match,
    )
    posterior -= _invalid_evidence_penalty(report)
    return KernelCandidate(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        answer_form=answer_form,
        holes=holes,
        operator=operator,
        source=source,
        evidence_score=evidence_score,
        complexity=complexity,
        coverage=coverage.score,
        loss=loss,
        posterior=posterior,
    )


def _invalid_evidence_penalty(report: str) -> float:
    low = str(report or "").lower()
    penalty = 0.0
    empty_markers = (
        "not enough relevant numeric columns",
        "not enough relevant columns",
        "no stable evidence found",
        "no valid evidence",
        "insufficient evidence",
        "could not instantiate",
    )
    if any(marker in low for marker in empty_markers):
        penalty += 24.0
    if "available evidence does not support a simple direct effect" in low and any(marker in low for marker in empty_markers):
        penalty += 18.0
    return penalty


def _apply_metamorphic_rescore(
    task: KernelTask,
    contract: EvidenceContract,
    candidate: KernelCandidate,
) -> KernelCandidate:
    evaluation = evaluate_metamorphic_estimand(
        task_text=task.task_text,
        domain_context=task.domain_context,
        schema=task.schema,
        contract=contract,
        candidate=candidate,
    )
    if evaluation.loss <= 0:
        return replace(
            candidate,
            metamorphic_loss=0.0,
            metamorphic_signatures=(),
            metamorphic_trace=evaluation.trace(),
        )
    posterior = candidate.posterior - 32.0 * evaluation.loss
    return replace(
        candidate,
        posterior=posterior,
        metamorphic_loss=evaluation.loss,
        metamorphic_signatures=evaluation.signatures,
        metamorphic_trace=evaluation.trace(),
    )


def _posterior(*, evidence_score: float, coverage: float, complexity: float, loss: float, answer_form_match: float) -> float:
    if not math.isfinite(evidence_score):
        evidence_score = 0.0
    evidence_term = min(24.0, 0.45 * max(0.0, evidence_score))
    mismatch_penalty = 14.0 * (1.0 - answer_form_match)
    return float(
        evidence_term
        + 34.0 * coverage
        + 8.0 * answer_form_match
        - mismatch_penalty
        - 2.0 * complexity
        - 12.0 * loss
    )


def _answer_form_match(expected: str, actual: str) -> float:
    exp = str(expected or "").lower()
    act = str(actual or "").lower()
    if not exp or not act:
        return 0.0
    if exp == act:
        return 1.0
    aliases = {
        "period_category_crossover": {"period_crossover", "period_category_crossover"},
        "temporal_event": {"peak", "trend", "onset", "temporal_event"},
        "measured_relation": {"association", "mediation", "coefficient_relationship", "measured_relation"},
        "coefficient_or_group_difference": {
            "coefficient_or_group_difference",
            "coefficient_relationship",
            "measured_relation",
            "racial_difference",
            "group_mean_difference",
        },
        "grouped_original_replication_comparison": {"grouped_comparison", "grouped_original_replication_comparison"},
        "paired_group_mean_comparison": {"paired_group_mean_comparison", "grouped_original_replication_comparison"},
        "original_replication_design": {
            "original_replication_design",
            "grouped_original_replication_comparison",
            "paired_group_mean_comparison",
        },
        "measured_selection": {"generic_categorical_measurement", "top_category_proportion", "measured_selection"},
        "prompted_survey_item_proportion": {"prompted_survey_item_proportion", "top_role_proportion"},
    }
    if act in aliases.get(exp, set()):
        return 1.0
    if exp in aliases.get(act, set()):
        return 1.0
    exp_tokens = {tok for tok in exp.split("_") if tok}
    act_tokens = {tok for tok in act.split("_") if tok}
    if exp_tokens and act_tokens:
        overlap = len(exp_tokens & act_tokens) / max(len(exp_tokens), len(act_tokens))
        if overlap >= 0.5:
            return 0.5
    return 0.0


def _slot_value(slot: str, slots: Mapping[str, Any], report: str) -> Any:
    if slot in slots and slots[slot] not in (None, "", [], {}):
        return slots[slot]
    structured = {str(k).lower(): v for k, v in dict(slots or {}).items()}
    slot_l = slot.lower()
    if slot_l in structured and structured[slot_l] not in (None, "", [], {}):
        return structured[slot_l]
    aliases = {
        "event": ("event=", "peak", "onset", "trend", "surpass"),
        "variable": ("variable=", "target=", "cause_series", "effect_series"),
        "variables": ("variables=", "variable=", "between"),
        "time": ("time=", "century", "bce", "year", "period"),
        "relation": ("relation", "positive", "negative", "increase", "decrease"),
        "statistic": ("coef=", "coefficient", "corr", "mean=", "median", "value="),
        "target": ("target=", "target_column", "category", "variable="),
        "operator": ("operator", "argmax", "maximum", "highest", "mean", "median"),
        "measured_value": ("value=", "pct=", "mean=", "median", "coefficient", "score="),
        "coefficient": ("coefficient", "coef=", "target_coefficient"),
        "direction": ("positive", "negative", "direction"),
        "region": ("region", "groups", "countries"),
        "cause_series": ("cause_series", "education", "expenditure"),
        "effect_series": ("effect_series", "gdp", "gni", "income"),
        "temporal_association": ("temporal association", "lagged", "first_difference", "panel"),
        "evidence": ("evidence:", "answer_slot_", "slot_contract_", "estimand_"),
    }
    low = report.lower()
    if any(alias in low for alias in aliases.get(slot_l, (slot_l,))):
        return "text"
    return None


def _hole_type(name: str) -> str:
    low = name.lower()
    if low in {"time", "period", "year"}:
        return "time"
    if low in {"variable", "variables", "x", "y", "target", "cause_series", "effect_series"}:
        return "variable"
    if low in {"relation", "direction", "temporal_association"}:
        return "relation"
    if low in {"coefficient", "statistic", "measured_value", "left_mean", "right_mean"}:
        return "measurement"
    if low in {"group", "region", "context"}:
        return "context"
    return "slot"


def _schema_text(schema: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, value in schema.items():
        if isinstance(value, Mapping):
            parts.append(str(key))
            parts.extend(f"{k} {v}" for k, v in value.items())
        else:
            parts.append(f"{key} {value}")
    return "\n".join(parts)
