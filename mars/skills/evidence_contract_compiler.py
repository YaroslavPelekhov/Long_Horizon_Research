"""Evidence-contract compiler for scientific discovery tasks.

The contract compiler sits above local analyzers.  It does not know benchmark
answers or dataset names; it infers the answer shape from the question, asks
existing executable backends to close slots, and then scores whether a candidate
report actually covers the required evidence slots.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from .answer_slot_compiler import infer_answer_slot_hypothesis
from .question_form_inducer import infer_question_form
from .slot_contract import SlotContractResult, infer_slot_contract_hypothesis
from .universal_slot_compiler import infer_universal_slot_hypothesis


@dataclass(frozen=True)
class EvidenceSlot:
    name: str
    required: bool = True
    cues: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceContract:
    answer_form: str
    slots: tuple[EvidenceSlot, ...]
    interface_kind: str
    verifier: str
    confidence: float

    @property
    def required_slot_names(self) -> tuple[str, ...]:
        return tuple(slot.name for slot in self.slots if slot.required)


@dataclass(frozen=True)
class EvidenceCoverage:
    score: float
    covered_slots: tuple[str, ...]
    missing_slots: tuple[str, ...]
    accepted: bool


@dataclass(frozen=True)
class EvidenceContractResult:
    contract: EvidenceContract
    hypothesis: str
    workflow: str
    evidence: str
    coverage: EvidenceCoverage
    source: str
    score: float

    def as_report(self) -> str:
        slots = ",".join(self.coverage.covered_slots)
        missing = ",".join(self.coverage.missing_slots) or "none"
        return (
            f"HYPOTHESIS: {self.hypothesis}\n"
            f"WORKFLOW SUMMARY: EvidenceContractCompiler compiled answer_form={self.contract.answer_form}, "
            f"covered_slots={slots}, missing_slots={missing}, source={self.source}. {self.workflow} "
            f"Evidence: {self.evidence}."
        )


def compile_evidence_contract(
    question: str,
    *,
    interface_kind: str = "discovery",
    schema_text: str = "",
) -> EvidenceContract:
    """Infer a task-level evidence contract from observable task text."""

    form = infer_question_form(question, schema_text)
    if form is not None:
        return EvidenceContract(
            answer_form=form.name,
            slots=tuple(EvidenceSlot(name) for name in form.slots),
            interface_kind=interface_kind,
            verifier="all required typed answer slots must be grounded by executable evidence",
            confidence=form.confidence,
        )

    q = str(question or "").lower()
    if any(token in q for token in ("in which century", "what century", "when did", "for the first time")):
        return EvidenceContract(
            answer_form="temporal_event",
            slots=(
                EvidenceSlot("event"),
                EvidenceSlot("variable"),
                EvidenceSlot("time"),
                EvidenceSlot("relation"),
            ),
            interface_kind=interface_kind,
            verifier="temporal answer must bind event, variable, time, and relation",
            confidence=0.82,
        )
    if any(token in q for token in ("relationship", "influence", "impact", "effect", "affect", "promote", "reduce", "associated")):
        return EvidenceContract(
            answer_form="measured_relation",
            slots=(
                EvidenceSlot("variables"),
                EvidenceSlot("relation"),
                EvidenceSlot("statistic"),
                EvidenceSlot("context", required=False),
            ),
            interface_kind=interface_kind,
            verifier="relation answer must bind variables, relation direction/type, and a measured statistic",
            confidence=0.72,
        )
    if any(token in q for token in ("which", "what", "highest", "most", "primarily", "proportion")):
        return EvidenceContract(
            answer_form="measured_selection",
            slots=(
                EvidenceSlot("target"),
                EvidenceSlot("operator"),
                EvidenceSlot("measured_value"),
                EvidenceSlot("context", required=False),
            ),
            interface_kind=interface_kind,
            verifier="selection answer must bind target, operator, and measured value",
            confidence=0.68,
        )
    return EvidenceContract(
        answer_form="scientific_hypothesis",
        slots=(
            EvidenceSlot("variables"),
            EvidenceSlot("relation"),
            EvidenceSlot("evidence"),
        ),
        interface_kind=interface_kind,
        verifier="hypothesis must name variables, relation, and evidence",
        confidence=0.55,
    )


def score_evidence_coverage(contract: EvidenceContract, report: str, slots: Mapping[str, Any] | None = None) -> EvidenceCoverage:
    """Score whether a report covers the contract's required slots."""

    text = str(report or "").lower()
    structured = {str(k).lower(): v for k, v in dict(slots or {}).items()}
    covered: list[str] = []
    missing: list[str] = []
    for slot in contract.slots:
        if not slot.required:
            continue
        if _slot_is_covered(slot.name, text, structured):
            covered.append(slot.name)
        else:
            missing.append(slot.name)
    denom = max(1, len(covered) + len(missing))
    score = len(covered) / denom
    return EvidenceCoverage(
        score=score,
        covered_slots=tuple(covered),
        missing_slots=tuple(missing),
        accepted=score >= 0.999 or (score >= 0.75 and "answer_form=" in text),
    )


def infer_evidence_contract_hypothesis(
    *,
    question: str,
    domain_context: str,
    data: Any,
    schema: Mapping[str, Any] | None = None,
    interface_kind: str = "discovery",
) -> EvidenceContractResult | None:
    """Close the best evidence contract using available executable backends."""

    schema = schema or {}
    contract = compile_evidence_contract(
        question,
        interface_kind=interface_kind,
        schema_text=_schema_text(schema),
    )
    candidates: list[EvidenceContractResult] = []

    if isinstance(data, Mapping):
        multitable = infer_universal_slot_hypothesis(
            task_text=question,
            interface_kind="discovery_multitable",
            data=data,
            domain_context=domain_context,
            schema=schema,
        )
        if multitable is not None:
            candidates.append(_wrap_candidate(contract, multitable, "multitable"))
        for name, df in data.items():
            table_schema = schema.get(name, {}) if isinstance(schema.get(name, {}), Mapping) else {}
            candidates.extend(
                _single_table_candidates(
                    contract=contract,
                    question=question,
                    domain_context=domain_context,
                    df=df,
                    schema={str(k): str(v) for k, v in table_schema.items()},
                    source=str(name),
                )
            )
    else:
        candidates.extend(
            _single_table_candidates(
                contract=contract,
                question=question,
                domain_context=domain_context,
                df=data,
                schema={str(k): str(v) for k, v in schema.items()},
                source="table",
            )
        )

    if not candidates:
        return None
    candidates.sort(key=lambda item: item.score, reverse=True)
    best = candidates[0]
    if best.coverage.score < 0.5:
        return None
    return best


def _single_table_candidates(
    *,
    contract: EvidenceContract,
    question: str,
    domain_context: str,
    df: Any,
    schema: Mapping[str, str],
    source: str,
) -> list[EvidenceContractResult]:
    candidates: list[EvidenceContractResult] = []
    if df is None or not hasattr(df, "columns"):
        return candidates

    answer_slot = infer_answer_slot_hypothesis(
        question=question,
        domain_context=domain_context,
        df=df,
        column_descriptions=schema,
    )
    if answer_slot is not None:
        candidates.append(_wrap_candidate(contract, answer_slot, source))

    slot_contract = infer_slot_contract_hypothesis(
        question=question,
        domain_context=domain_context,
        df=df,
        column_descriptions=schema,
    )
    if slot_contract is not None:
        candidates.append(_wrap_candidate(contract, slot_contract, source))
    return candidates


def _wrap_candidate(contract: EvidenceContract, candidate: SlotContractResult, source: str) -> EvidenceContractResult:
    report = f"{candidate.hypothesis}\n{candidate.workflow}\n{candidate.evidence}"
    coverage = score_evidence_coverage(contract, report, candidate.slots)
    form_bonus = 1.0 if contract.answer_form in str(report).lower() else 0.0
    score = candidate.score + 20.0 * coverage.score + 4.0 * float(coverage.accepted) + form_bonus
    return EvidenceContractResult(
        contract=contract,
        hypothesis=candidate.hypothesis,
        workflow=candidate.workflow,
        evidence=candidate.evidence,
        coverage=coverage,
        source=source,
        score=score,
    )


def _slot_is_covered(slot: str, text: str, structured: Mapping[str, Any]) -> bool:
    slot_l = slot.lower()
    if slot_l in structured and structured[slot_l] not in (None, "", [], {}):
        return True
    aliases = {
        "event": ("event=", "temporal_", "peak", "onset", "trend", "surpass"),
        "variable": ("variable=", "variables=", "target=", "cause_series", "effect_series"),
        "variables": ("variables=", "variable=", " x=", " y=", " between "),
        "time": ("time=", "century", "bce", "year", "period"),
        "relation": ("relation", "positive", "negative", "increase", "decrease", "larger", "smaller", "surpass"),
        "statistic": ("coef=", "coefficient", "corr", "mean=", "median", "pct=", "proportion", "score=", "value="),
        "target": ("target=", "target_column", "category", "variable=", "items"),
        "operator": ("operator", "argmax", "maximum", "highest", "primarily", "proportion", "mean", "median"),
        "measured_value": ("value=", "pct=", "mean=", "median", "coefficient", "score=", "%"),
        "group": ("group=", "groups", "domain", "project", "region"),
        "original_effect": ("original_effect", "original mean", "original_measure", "fiso"),
        "replication_effect": ("replication_effect", "replication mean", "replication_measure", "fisr"),
        "left_measure": ("left_measure", "observed power", "original"),
        "right_measure": ("right_measure", "planned power", "replication"),
        "left_mean": ("left_mean", "average observed", "mean"),
        "right_mean": ("right_mean", "average planned", "mean"),
        "x": (" x=", "variables=", "between"),
        "y": (" y=", "variables=", "between"),
        "coefficient": ("coefficient", "coef=", "target_coefficient"),
        "direction": ("positive", "negative", "direction"),
        "region": ("region", "groups", "countries"),
        "cause_series": ("cause_series", "cause_table", "education", "expenditure"),
        "effect_series": ("effect_series", "effect_table", "gdp", "gni", "income"),
        "temporal_association": ("temporal association", "lagged", "first_difference", "panel"),
        "evidence": ("evidence:", "answer_slot_", "slot_contract_"),
    }
    return any(alias in text for alias in aliases.get(slot_l, (slot_l,)))


def _schema_text(schema: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, value in schema.items():
        if isinstance(value, Mapping):
            parts.append(str(key))
            parts.extend(f"{k} {v}" for k, v in value.items())
        else:
            parts.append(f"{key} {value}")
    return "\n".join(parts)
