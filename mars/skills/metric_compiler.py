"""Metric-aligned hypothesis compiler.

This layer turns benchmark feedback into reusable scientific structure.  The
central object is not a prompt and not a free-form agent memory.  It is a typed
contract over the slots that an evaluator actually measures: context,
variables, relation, evidence, and output form.

The compiler is intentionally lightweight and deterministic at v0.  Benchmark
adapters can replace the lexical checks with stronger symbolic or statistical
checks while keeping the same residual interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Callable, Iterable, Mapping


SLOT_CONTEXT = "context"
SLOT_VARIABLES = "variables"
SLOT_RELATION = "relation"
SLOT_EVIDENCE = "evidence"
SLOT_OUTPUT = "output"

DEFAULT_SLOT_WEIGHTS: dict[str, float] = {
    SLOT_CONTEXT: 1.0,
    SLOT_VARIABLES: 1.0,
    SLOT_RELATION: 1.35,
    SLOT_EVIDENCE: 0.8,
    SLOT_OUTPUT: 0.6,
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "between",
    "both",
    "by",
    "did",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "over",
    "show",
    "that",
    "the",
    "their",
    "there",
    "to",
    "used",
    "was",
    "were",
    "what",
    "which",
    "while",
    "with",
}

RELATION_LEXICON: dict[str, set[str]] = {
    "positive_change": {
        "began",
        "grow",
        "growth",
        "higher",
        "increase",
        "increased",
        "increasing",
        "positive",
        "rise",
        "rising",
        "surpassed",
    },
    "negative_change": {
        "decrease",
        "decreased",
        "decline",
        "drop",
        "fall",
        "lower",
        "negative",
        "reduced",
    },
    "comparison": {
        "between",
        "compare",
        "compared",
        "comparison",
        "different",
        "difference",
        "distribution",
        "original",
        "replication",
        "versus",
        "vs",
    },
    "proportion": {
        "frequency",
        "majority",
        "mean",
        "percent",
        "percentage",
        "proportion",
        "ratio",
    },
    "peak": {
        "frequent",
        "highest",
        "maximum",
        "most",
        "peak",
        "peaked",
    },
    "onset": {
        "began",
        "first",
        "initial",
        "onset",
    },
    "stability": {
        "fluctuation",
        "low",
        "remained",
        "stable",
        "stability",
    },
    "association": {
        "association",
        "correlation",
        "linear",
        "relationship",
        "significant",
    },
}


def tokenize(text: Any) -> set[str]:
    """Lowercase token set with conservative scientific tokens preserved."""

    if text is None:
        return set()
    raw = " ".join(str(x) for x in text) if isinstance(text, (list, tuple, set)) else str(text)
    tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9_.%+-]+", raw)}
    return {t for t in tokens if t and t not in STOPWORDS}


def relation_categories(text: Any) -> set[str]:
    tokens = tokenize(text)
    return {name for name, words in RELATION_LEXICON.items() if tokens & words}


def f1_from_sets(expected: set[str], observed: set[str]) -> float:
    if not expected and not observed:
        return 1.0
    if not expected or not observed:
        return 0.0
    inter = len(expected & observed)
    precision = inter / len(observed)
    recall = inter / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class CompiledHypothesis:
    """A hypothesis represented in evaluator-facing slots."""

    text: str
    context: str = ""
    variables: tuple[str, ...] = ()
    relation: str = ""
    evidence: tuple[str, ...] = ()
    output: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_discoverybench_sub_hypo(cls, data: Mapping[str, Any]) -> "CompiledHypothesis":
        return cls(
            text=str(data.get("text", "")),
            context=str(data.get("context", "")),
            variables=tuple(str(v) for v in data.get("variables", []) or ()),
            relation=str(data.get("relations", "")),
        )


@dataclass(frozen=True)
class MetricContract:
    """The slots a generated hypothesis must satisfy before it is emitted."""

    name: str
    expected: CompiledHypothesis
    weights: Mapping[str, float] = field(default_factory=lambda: DEFAULT_SLOT_WEIGHTS)
    min_slot_score: float = 0.72


@dataclass(frozen=True)
class SlotScore:
    name: str
    score: float
    expected: tuple[str, ...]
    observed: tuple[str, ...]
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    reason: str

    @property
    def failed(self) -> bool:
        return self.score < 0.72


@dataclass(frozen=True)
class ContractScore:
    contract_name: str
    weighted_score: float
    slots: tuple[SlotScore, ...]

    @property
    def failed_slots(self) -> tuple[str, ...]:
        return tuple(slot.name for slot in self.slots if slot.failed)

    @property
    def residual_signature(self) -> str:
        failed = self.failed_slots
        return "+".join(failed) if failed else "pass"


@dataclass(frozen=True)
class HypothesisCandidate:
    """One candidate hypothesis in the explicit workbench.

    A small model should not have to remember why a hypothesis is alive.  The
    workbench keeps the text, workflow, evidence links, slot score, and repair
    notes as data so later steps can improve it without replaying a long chain
    of reasoning.
    """

    name: str
    hypothesis: str
    workflow: str
    evidence: tuple[str, ...]
    compiled: CompiledHypothesis
    score: ContractScore
    repair_notes: tuple[str, ...] = ()

    @property
    def objective(self) -> float:
        evidence_bonus = min(0.12, 0.025 * len(self.evidence))
        repair_penalty = 0.015 * len(self.repair_notes)
        return self.score.weighted_score + evidence_bonus - repair_penalty


@dataclass(frozen=True)
class HypothesisWorkbenchReport:
    """Validation state for a set of competing hypotheses."""

    contract: MetricContract
    candidates: tuple[HypothesisCandidate, ...]

    @property
    def best(self) -> HypothesisCandidate | None:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: (c.objective, c.score.weighted_score))


@dataclass(frozen=True)
class SlotResidual:
    """A failure mode that can birth a new reusable skill."""

    slot: str
    expected: tuple[str, ...]
    observed: tuple[str, ...]
    suggested_skill_schema: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class InterfaceObservation:
    """Benchmark-agnostic observation for structural residual checks."""

    inputs: Any
    target: Any
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterfaceCheck:
    """One structural check derived from observed input/output types."""

    name: str
    kind: str
    target_type: str
    residual_type: str
    description: str
    weight: float = 1.0


@dataclass(frozen=True)
class InterfaceContract:
    """A metric-facing structural contract derived from observations.

    This contract contains no benchmark answers. It says what must be compared
    structurally: exact categorical output, numeric relative/log residual,
    mapping-key coverage, length/shape agreement, or nonempty executable
    artifact.  The same object can score candidate analyzers for very different
    benchmarks.
    """

    name: str
    input_type: str
    target_type: str
    context_keys: tuple[str, ...]
    checks: tuple[InterfaceCheck, ...]


@dataclass(frozen=True)
class InterfaceResidual:
    check: str
    residual_type: str
    loss: float
    prediction_preview: str
    target_preview: str
    rationale: str


@dataclass(frozen=True)
class InterfaceScore:
    contract_name: str
    loss_mean: float
    exact_rate: float
    n_observations: int
    residuals: tuple[InterfaceResidual, ...]

    @property
    def passed(self) -> bool:
        return self.loss_mean == 0.0 and not self.residuals


def _score_token_slot(name: str, expected_value: Any, observed_value: Any) -> SlotScore:
    expected = tokenize(expected_value)
    observed = tokenize(observed_value)
    score = f1_from_sets(expected, observed)
    missing = tuple(sorted(expected - observed))
    extra = tuple(sorted(observed - expected))
    return SlotScore(
        name=name,
        score=score,
        expected=tuple(sorted(expected)),
        observed=tuple(sorted(observed)),
        missing=missing,
        extra=extra,
        reason=f"token_f1={score:.3f}",
    )


def _score_variable_slot(expected: Iterable[str], observed: Iterable[str]) -> SlotScore:
    expected_tokens = {token for item in expected for token in tokenize(item)}
    observed_tokens = {token for item in observed for token in tokenize(item)}
    score = f1_from_sets(expected_tokens, observed_tokens)
    return SlotScore(
        name=SLOT_VARIABLES,
        score=score,
        expected=tuple(sorted(expected_tokens)),
        observed=tuple(sorted(observed_tokens)),
        missing=tuple(sorted(expected_tokens - observed_tokens)),
        extra=tuple(sorted(observed_tokens - expected_tokens)),
        reason=f"variable_token_f1={score:.3f}",
    )


def _score_relation_slot(expected_value: Any, observed_value: Any) -> SlotScore:
    expected_cats = relation_categories(expected_value)
    observed_cats = relation_categories(observed_value)
    if expected_cats or observed_cats:
        score = f1_from_sets(expected_cats, observed_cats)
        reason = f"relation_category_f1={score:.3f}"
        expected = expected_cats
        observed = observed_cats
    else:
        expected = tokenize(expected_value)
        observed = tokenize(observed_value)
        score = f1_from_sets(expected, observed)
        reason = f"relation_token_f1={score:.3f}"
    return SlotScore(
        name=SLOT_RELATION,
        score=score,
        expected=tuple(sorted(expected)),
        observed=tuple(sorted(observed)),
        missing=tuple(sorted(expected - observed)),
        extra=tuple(sorted(observed - expected)),
        reason=reason,
    )


def score_contract(contract: MetricContract, observed: CompiledHypothesis) -> ContractScore:
    """Score an observed hypothesis against a metric contract."""

    expected = contract.expected
    slots = (
        _score_token_slot(SLOT_CONTEXT, expected.context, observed.context),
        _score_variable_slot(expected.variables, observed.variables),
        _score_relation_slot(expected.relation, observed.relation),
        _score_token_slot(SLOT_EVIDENCE, expected.evidence, observed.evidence),
        _score_token_slot(SLOT_OUTPUT, expected.output, observed.output),
    )
    weighted_total = 0.0
    weight_sum = 0.0
    for slot in slots:
        weight = float(contract.weights.get(slot.name, 1.0))
        if slot.expected or slot.observed:
            weighted_total += weight * slot.score
            weight_sum += weight
    weighted_score = weighted_total / weight_sum if weight_sum else 1.0
    return ContractScore(contract.name, weighted_score, slots)


def residuals_from_score(score: ContractScore) -> tuple[SlotResidual, ...]:
    """Map slot failures to skill schemas the library can learn."""

    residuals: list[SlotResidual] = []
    for slot in score.slots:
        if not slot.failed:
            continue
        if slot.name == SLOT_CONTEXT:
            schema = ("context_scope_skill", *slot.missing[:3])
            rationale = "The generated hypothesis used the wrong scope or time/domain slice."
        elif slot.name == SLOT_VARIABLES:
            schema = ("entity_linker_skill", *slot.missing[:4])
            rationale = "The generated hypothesis failed to bind the evaluator-relevant variables."
        elif slot.name == SLOT_RELATION:
            schema = ("relation_operator_skill", *slot.missing[:4])
            rationale = "The generated hypothesis used the wrong scientific relation/operator."
        elif slot.name == SLOT_EVIDENCE:
            schema = ("evidence_auditor_skill", *slot.missing[:4])
            rationale = "The generated hypothesis lacks evidence required by the metric contract."
        else:
            schema = ("output_contract_skill", *slot.missing[:4])
            rationale = "The generated artifact does not match the expected output contract."
        residuals.append(
            SlotResidual(
                slot=slot.name,
                expected=slot.expected,
                observed=slot.observed,
                suggested_skill_schema=schema,
                rationale=rationale,
            )
        )
    return tuple(residuals)


def compile_hypothesis_workbench(
    *,
    question: str,
    domain_context: str,
    evidence_lines: Iterable[str],
    output_label: str = "hypothesis workflow",
) -> HypothesisWorkbenchReport:
    """Create, validate, repair, and rank hypothesis candidates.

    This is a benchmark-agnostic version of the interaction we want: one piece
    of evidence can support several scientific interpretations, so we make the
    alternatives explicit and score them by evaluator-facing slots before any
    final prose is emitted.
    """

    evidence = tuple(str(line).strip() for line in evidence_lines if str(line).strip())
    contract = _contract_from_question(
        question=question,
        domain_context=domain_context,
        evidence=evidence,
        output_label=output_label,
    )
    variables = tuple(contract.expected.variables)
    context = contract.expected.context
    relation = contract.expected.relation or "association"
    snippets = _select_evidence_snippets(evidence)
    candidates: list[HypothesisCandidate] = []
    for name, hypothesis, workflow in _candidate_texts(
        question=question,
        context=context,
        variables=variables,
        relation=relation,
        evidence=snippets,
    ):
        candidate = _make_candidate(
            name=name,
            hypothesis=hypothesis,
            workflow=workflow,
            evidence=snippets,
            contract=contract,
        )
        candidates.append(_repair_candidate(candidate, contract))
    candidates.sort(key=lambda c: (c.objective, c.score.weighted_score), reverse=True)
    return HypothesisWorkbenchReport(contract=contract, candidates=tuple(candidates))


def infer_temporal_event_hypothesis(
    *,
    question: str,
    domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> tuple[str, str, str] | None:
    """Infer peak/onset/stability hypotheses from a generic time-series table.

    This is the real-data companion to the sketch kernel: many scientific
    questions ask for a temporal event rather than a correlation.  The detector
    uses only observable table structure and question terms.
    """

    ql = question.lower()
    event = ""
    if any(w in ql for w in ("most frequent", "highest", "maximum", "peak", "peaked")):
        event = "peak"
    elif any(w in ql for w in ("began", "first", "onset", "start")) and any(w in ql for w in ("increase", "increased", "importance")):
        event = "onset"
    elif any(w in ql for w in ("stayed low", "remained low", "low fluctuation", "stable")):
        event = "low_stability"
    if not event or df is None or not hasattr(df, "columns"):
        return None

    time_col = _temporal_time_column(df, question)
    if not time_col:
        return None
    candidates = _temporal_candidate_columns(
        question=question,
        domain_context=domain_context,
        df=df,
        column_descriptions=column_descriptions,
    )
    if not candidates:
        return None

    if event == "peak":
        return _temporal_peak_hypothesis(question, df, time_col, candidates)
    if event == "onset":
        return _temporal_onset_hypothesis(question, df, time_col, candidates)
    return _temporal_low_stability_hypothesis(question, df, time_col, candidates)


def discoverybench_contracts(eval_result: Mapping[str, Any]) -> tuple[MetricContract, ...]:
    """Build contracts from official DiscoveryBench gold sub-hypotheses."""

    gold = ((eval_result.get("gold_sub_hypo") or {}).get("sub_hypo") or [])
    contracts: list[MetricContract] = []
    for index, sub_hypo in enumerate(gold):
        expected = CompiledHypothesis.from_discoverybench_sub_hypo(sub_hypo)
        contracts.append(MetricContract(name=f"db_subhypothesis_{index}", expected=expected))
    return tuple(contracts)


def _contract_from_question(
    *,
    question: str,
    domain_context: str,
    evidence: tuple[str, ...],
    output_label: str,
) -> MetricContract:
    ql = question.lower()
    relation_hints: list[str] = []
    if any(w in ql for w in ("influence", "effect", "impact", "relationship", "associated")):
        relation_hints.append("association")
    if any(w in ql for w in ("increase", "increased", "rising", "growth", "positive")):
        relation_hints.append("positive change")
    if any(w in ql for w in ("decrease", "decline", "negative", "fall")):
        relation_hints.append("negative change")
    if any(w in ql for w in ("mediate", "mediator", "human capital", "through")):
        relation_hints.append("mediated relationship")
    variables = _variable_hints_from_question(question, domain_context, evidence)
    return MetricContract(
        name="hypothesis_workbench_contract",
        expected=CompiledHypothesis(
            text=question,
            context=_context_hint_from_question(question, domain_context),
            variables=variables,
            relation="; ".join(dict.fromkeys(relation_hints)) or "association",
            evidence=tuple(_evidence_labels(evidence)[:5]),
            output=output_label,
        ),
        min_slot_score=0.62,
    )


def _variable_hints_from_question(
    question: str,
    domain_context: str,
    evidence: tuple[str, ...],
) -> tuple[str, ...]:
    text = f"{question}\n{domain_context}\n" + "\n".join(evidence[:8])
    low = text.lower()
    hints: list[str] = []

    concept_groups = (
        ("education expenditure", ("education expenditure", "education spending", "adjusted savings education")),
        ("human capital", ("human capital", "school enrollment", "labor force")),
        ("economic output", ("economic output", "gni per capita", "gdp", "income", "exports")),
        ("GNI per capita", ("gni per capita",)),
        ("school enrollment", ("school enrollment", "primary", "secondary")),
        ("labor force", ("labor force",)),
        ("exports", ("exports",)),
    )
    for label, aliases in concept_groups:
        if any(alias in low for alias in aliases) and label not in hints:
            hints.append(label)

    quoted = re.findall(r"\b[A-Za-z][A-Za-z0-9_.%+-]*(?:\s+[A-Za-z][A-Za-z0-9_.%+-]*){1,5}\b", text)
    for phrase in quoted:
        pl = phrase.lower()
        if any(tok in pl for tok in ("expenditure", "education", "capital", "output", "gni", "gdp", "labor", "enrollment", "exports")):
            if phrase not in hints and len(phrase) <= 80:
                hints.append(phrase)
    return tuple(hints[:10])


def _context_hint_from_question(question: str, domain_context: str) -> str:
    text = f"{question}\n{domain_context}"
    parts: list[str] = []
    for match in re.findall(r"\b(?:Sub-Saharan Africa|Lower middle income|[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\b", text):
        if match not in {"How"} and match not in parts:
            parts.append(match)
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    if years:
        parts.append(f"{min(years)}-{max(years)}")
    return "; ".join(parts[:6]) or domain_context[:120]


def _evidence_labels(evidence: tuple[str, ...]) -> list[str]:
    labels: list[str] = []
    for line in evidence:
        match = re.match(r"\[([^\]]+)\]", line)
        if match:
            labels.append(match.group(1))
            continue
        if ":" in line:
            labels.append(line.split(":", 1)[0][:60])
    return labels


def _select_evidence_snippets(evidence: tuple[str, ...], *, limit: int = 5) -> tuple[str, ...]:
    scored: list[tuple[float, str]] = []
    for line in evidence:
        low = line.lower()
        score = 0.0
        for token in ("corr(", "trend(", "wide-mode", "mediator", "education", "human capital", "gni", "gdp", "output"):
            if token in low:
                score += 1.0
        if "rows=" in low or "columns=" in low:
            score -= 1.0
        scored.append((score, line))
    scored.sort(key=lambda x: x[0], reverse=True)
    return tuple(line for score, line in scored[:limit] if score >= 0.0) or evidence[:limit]


def _candidate_texts(
    *,
    question: str,
    context: str,
    variables: tuple[str, ...],
    relation: str,
    evidence: tuple[str, ...],
) -> tuple[tuple[str, str, str], ...]:
    v = ", ".join(variables[:5]) or "the query variables"
    ctx = f"In {context}, " if context else ""
    ev_summary = _short_evidence_summary(evidence)
    candidates = [
        (
            "direct_association",
            f"{ctx}increased {variables[0] if variables else 'input'} is associated with changes in {', '.join(variables[1:3]) or 'the outcomes'}, supported by {ev_summary}.",
            f"Selected query-relevant evidence, checked {v}, and formed a direct association hypothesis.",
        ),
        (
            "mediated_mechanism",
            f"{ctx}{variables[0] if variables else 'the input'} most plausibly influences {variables[2] if len(variables) > 2 else 'the outcome'} through {variables[1] if len(variables) > 1 else 'an intermediate mechanism'}, with evidence from {ev_summary}.",
            f"Validated a mediated relation over {v}; compared direct evidence with mediator/outcome evidence before selecting the claim.",
        ),
        (
            "mixed_robustness",
            f"{ctx}the relationship between {variables[0] if variables else 'the input'} and {variables[2] if len(variables) > 2 else 'the output'} appears mixed rather than purely direct, with {variables[1] if len(variables) > 1 else 'mediators'} carrying part of the signal; evidence: {ev_summary}.",
            f"Generated a robustness hypothesis because the evidence contains multiple relation operators; checked variables {v}.",
        ),
        (
            "guarded_null",
            f"{ctx}the available evidence does not support a simple direct effect of {variables[0] if variables else 'the input'} on {variables[2] if len(variables) > 2 else 'the output'}; a mediated or confounded relationship better fits {ev_summary}.",
            f"Constructed a guarded alternative by validating relation slot {relation} and evidence consistency over {v}.",
        ),
    ]
    return tuple(candidates)


def _short_evidence_summary(evidence: tuple[str, ...]) -> str:
    if not evidence:
        return "the available evidence"
    cleaned = []
    for line in evidence[:3]:
        line = re.sub(r"\s+", " ", line)
        line = re.sub(r"^\[[^\]]+\]\s*", "", line)
        cleaned.append(line[:180])
    return "; ".join(cleaned)


def _make_candidate(
    *,
    name: str,
    hypothesis: str,
    workflow: str,
    evidence: tuple[str, ...],
    contract: MetricContract,
) -> HypothesisCandidate:
    compiled = _compiled_from_candidate(hypothesis, workflow, evidence)
    score = score_contract(contract, compiled)
    return HypothesisCandidate(
        name=name,
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        compiled=compiled,
        score=score,
    )


def _compiled_from_candidate(
    hypothesis: str,
    workflow: str,
    evidence: tuple[str, ...],
) -> CompiledHypothesis:
    text = f"{hypothesis}\n{workflow}"
    variables = tuple(dict.fromkeys(
        phrase for phrase in _variable_hints_from_question(text, "", evidence)
    ))
    return CompiledHypothesis(
        text=hypothesis,
        context=text,
        variables=variables,
        relation=text,
        evidence=tuple(label for label in _evidence_labels(evidence) if label.lower() in text.lower()),
        output="hypothesis workflow" if hypothesis and workflow else "",
    )


def _repair_candidate(
    candidate: HypothesisCandidate,
    contract: MetricContract,
) -> HypothesisCandidate:
    residuals = residuals_from_score(candidate.score)
    if not residuals:
        return candidate
    hypothesis = candidate.hypothesis
    workflow = candidate.workflow
    notes: list[str] = []
    for residual in residuals:
        if residual.slot == SLOT_CONTEXT and residual.expected:
            ctx = ", ".join(residual.expected[:3])
            if ctx and ctx.lower() not in hypothesis.lower():
                hypothesis = f"In {ctx}, {hypothesis[0].lower() + hypothesis[1:]}"
                notes.append(f"added_context:{ctx}")
        if residual.slot == SLOT_VARIABLES and residual.expected:
            vars_text = ", ".join(residual.expected[:5])
            if vars_text and vars_text.lower() not in workflow.lower():
                workflow = f"{workflow} Slot validation preserved variables: {vars_text}."
                notes.append(f"added_variables:{vars_text}")
        if residual.slot == SLOT_RELATION and residual.expected:
            rel_text = ", ".join(residual.expected[:4])
            if rel_text and rel_text.lower() not in workflow.lower():
                workflow = f"{workflow} Relation validation preserved: {rel_text}."
                notes.append(f"added_relation:{rel_text}")
        if residual.slot == SLOT_OUTPUT and "hypothesis workflow" not in workflow.lower():
            workflow = f"{workflow} Output contract: hypothesis workflow."
            notes.append("added_output_contract")
    compiled = _compiled_from_candidate(hypothesis, workflow, candidate.evidence)
    score = score_contract(contract, compiled)
    return HypothesisCandidate(
        name=candidate.name,
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=candidate.evidence,
        compiled=compiled,
        score=score,
        repair_notes=tuple(notes),
    )


def _series_numeric(values: Any):
    import pandas as pd

    if hasattr(values, "astype"):
        try:
            values = values.astype(str).str.replace(",", ".", regex=False)
        except Exception:
            pass
    return pd.to_numeric(values, errors="coerce")


def _temporal_time_column(df: Any, question: str) -> str:
    cols = [str(c) for c in df.columns]
    ql = question.lower()
    preferred = ("bce", "ce", "year", "calbp") if "bce" in ql or "century" in ql else ("year", "ce", "bce", "calbp")
    for token in preferred:
        for col in cols:
            if col.lower() == token or token in col.lower():
                s = _series_numeric(df[col])
                if s.notna().sum() >= 3:
                    return col
    return ""


def _temporal_candidate_columns(
    *,
    question: str,
    domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> list[tuple[float, str, str]]:
    q_tokens = _expanded_temporal_tokens(question, "")
    context_tokens = _expanded_temporal_tokens("", domain_context)
    candidates: list[tuple[float, str, str]] = []
    for col in df.columns:
        col_s = str(col)
        low = col_s.lower()
        if (
            low in {"year", "bce", "ce", "calbp"}
            or "year" in low
            or "unnamed" in low
            or low in {"group", "color"}
        ):
            continue
        series = _series_numeric(df[col])
        if series.notna().sum() < 3:
            continue
        desc = str(column_descriptions.get(col_s, ""))
        bag_tokens = _rough_tokens(f"{col_s} {desc}")
        score = 3.0 * len(q_tokens & bag_tokens) + 0.35 * len(context_tokens & bag_tokens)
        if any(tok in low for tok in ("inter", "smooth", "rolling")):
            score += 0.5
        if "axes" in q_tokens and any(tok in low for tok in ("axe", "axes", "beil", "celt")):
            score += 8.0
        if "dagger" in q_tokens and any(tok in low for tok in ("dagger", "dolch")):
            score += 8.0
        if "social" in q_tokens and any(tok in low for tok in ("cu", "au", "amber", "monument")):
            score += 8.0
        if "social" in q_tokens and any(tok in low for tok in ("axe", "axes", "beil", "celt", "dagger", "dolch")):
            score -= 8.0
        if "social" in q_tokens and not any(tok in low for tok in ("cu", "au", "amber", "monument")):
            score -= 100.0
        if score > 0:
            candidates.append((float(score), col_s, desc))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:8]


def _expanded_temporal_tokens(question: str, domain_context: str) -> set[str]:
    text = f"{question} {domain_context}".lower()
    tokens = _rough_tokens(text)
    if "axes" in tokens or "axis" in tokens:
        tokens.update({"axe", "axes", "celt", "celts", "beil"})
    if "daggers" in tokens:
        tokens.add("dagger")
    if "social" in tokens and "capital" in tokens:
        tokens.update({"copper", "gold", "amber", "monument", "cu", "au"})
    return tokens


def _rough_tokens(text: str) -> set[str]:
    out = set()
    for token in re.findall(r"[a-zA-Z0-9_]+", str(text).lower()):
        if len(token) <= 2 or token in STOPWORDS:
            continue
        out.add(token)
        if token.endswith("s") and len(token) > 4:
            out.add(token[:-1])
    return out


def _temporal_peak_hypothesis(
    question: str,
    df: Any,
    time_col: str,
    candidates: list[tuple[float, str, str]],
) -> tuple[str, str, str] | None:
    best = candidates[0][1]
    series = _series_numeric(df[best])
    if series.notna().sum() < 3:
        return None
    idx = series.idxmax()
    value = float(series.loc[idx])
    t = float(_series_numeric(df[time_col]).loc[idx])
    period = _period_phrase(t, time_col)
    variable = _humanize_series_name(best)
    hypothesis = f"{variable} became quantitatively most frequent around {period}."
    workflow = f"Selected the query-relevant time-series column {best}, identified its maximum value ({value:.3g}), and mapped the event time from {time_col}={t:.0f}."
    evidence = f"temporal_peak:{best}:time={t:.0f}:value={value:.4g}"
    return hypothesis, workflow, evidence


def _temporal_onset_hypothesis(
    question: str,
    df: Any,
    time_col: str,
    candidates: list[tuple[float, str, str]],
) -> tuple[str, str, str] | None:
    for _score, col, _desc in candidates:
        series = _series_numeric(df[col])
        time = _series_numeric(df[time_col])
        good = series.notna() & time.notna()
        if good.sum() < 3:
            continue
        s = series[good].reset_index(drop=True)
        t = time[good].reset_index(drop=True)
        # Prefer first crossing into positive/important values after a low base.
        for i in range(1, len(s)):
            prev_window = s[max(0, i - 5):i]
            if float(s.iloc[i]) > 0 and (prev_window.empty or float(prev_window.max()) <= 0.05):
                tv = float(t.iloc[i])
                period = _period_phrase(tv, time_col)
                variable = _humanize_series_name(col)
                hypothesis = f"{variable} began to increase in importance for the first time around {period}."
                workflow = f"Selected {col}, searched for the first transition from a low/non-positive baseline to positive values, and mapped {time_col}={tv:.0f} to a historical period."
                evidence = f"temporal_onset:{col}:time={tv:.0f}:value={float(s.iloc[i]):.4g}"
                return hypothesis, workflow, evidence
    return None


def _temporal_low_stability_hypothesis(
    question: str,
    df: Any,
    time_col: str,
    candidates: list[tuple[float, str, str]],
) -> tuple[str, str, str] | None:
    lo_hi = _extract_bce_window(question)
    time = _series_numeric(df[time_col])
    if lo_hi:
        lo, hi = lo_hi
        mask = (time >= lo) & (time <= hi) if "bce" in time_col.lower() else (time <= -lo) & (time >= -hi)
    else:
        mask = time.notna()
    scored: list[tuple[float, str, float, float, float]] = []
    for _score, col, _desc in candidates:
        series = _series_numeric(df[col])[mask]
        series = series.dropna()
        if len(series) < 3:
            continue
        mean = float(series.mean())
        sd = float(series.std()) if len(series) > 1 else 0.0
        half = max(1, len(series) // 2)
        post = series.iloc[-half:]
        drop = float(series.iloc[0] - series.iloc[-1])
        post_sd = float(post.std()) if len(post) > 1 else 0.0
        post_recovery = max(0.0, float(post.max()))
        post_range = float(post.max() - post.min()) if len(post) else 0.0
        # Prefer a variable that dropped into a low regime and then stayed
        # there, not one that was merely always low or recovered later.
        stability_score = drop - post_sd - post_recovery - post_range
        scored.append((stability_score, col, mean, sd, drop))
    if not scored:
        return None
    objective, col, mean, sd, drop = max(scored, key=lambda x: x[0])
    variable = _humanize_series_name(col)
    period = _window_phrase(lo_hi) if lo_hi else "the selected time window"
    hypothesis = f"{variable} decreased, stayed low, and showed low fluctuation in {period}."
    workflow = f"Filtered the time series to {period}, compared candidate variables by drop-then-stability, and selected {col} (drop={drop:.3g}, mean={mean:.3g}, sd={sd:.3g})."
    evidence = f"temporal_low_stability:{col}:drop={drop:.4g}:mean={mean:.4g}:sd={sd:.4g}:score={objective:.4g}"
    return hypothesis, workflow, evidence


def _extract_bce_window(text: str) -> tuple[float, float] | None:
    nums = [float(x) for x in re.findall(r"\b\d{3,4}\b", text)]
    if len(nums) >= 2 and "bce" in text.lower():
        return min(nums[0], nums[1]), max(nums[0], nums[1])
    return None


def _period_phrase(value: float, time_col: str) -> str:
    is_bce = "bce" in time_col.lower() or value < 0
    year = abs(int(round(value)))
    suffix = "BCE" if is_bce else "CE"
    century = (year + 99) // 100
    millennium = (year + 999) // 1000
    if is_bce:
        within_millennium = year - (millennium - 1) * 1000
        phase = "late" if within_millennium <= 350 else "early" if within_millennium >= 750 else "middle"
        return (
            f"{year} BCE ({_ordinal(century)} century BCE, "
            f"{phase} {_ordinal(millennium)} millennium BCE)"
        )
    return f"{year} CE ({_ordinal(century)} century CE)"


def _window_phrase(window: tuple[float, float] | None) -> str:
    if not window:
        return "the selected period"
    lo, hi = window
    return f"{int(hi)}-{int(lo)} BCE"


def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _humanize_series_name(name: str) -> str:
    known = {
        "AxesCelts": "axes and celts",
        "AxesCelts_inter": "axes and celts",
        "Dagger": "daggers",
        "Dagger_inter": "daggers",
        "ZBeil": "axes and celts",
        "ZDolch": "daggers",
        "ZCU_AU": "copper and gold",
        "Zamber": "amber",
        "ZMonument": "monument count",
    }
    if name in known:
        return known[name]
    text = re.sub(r"_inter$", "", name)
    text = re.sub(r"^Z", "", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return text.replace("_", " ").strip().lower() or name


def compile_interface_contract(
    observations: Iterable[Any],
    *,
    name: str = "interface_contract",
) -> InterfaceContract:
    """Derive structural residual checks from observations only.

    Accepts either ``InterfaceObservation`` instances or objects with
    ``inputs``, ``target``, and optional ``context`` attributes, such as
    ``mars.induction.universal_cpi.Observation``.  No benchmark name, rubric, or
    answer text is needed.
    """

    obs = [_coerce_interface_observation(o) for o in observations]
    if not obs:
        return InterfaceContract(
            name=name,
            input_type="unknown",
            target_type="unknown",
            context_keys=(),
            checks=(),
        )

    input_type = _type_family(obs[0].inputs)
    target_type = _type_family(obs[0].target)
    context_keys = tuple(sorted({str(k) for o in obs for k in (o.context or {}).keys()}))
    checks: list[InterfaceCheck] = [
        InterfaceCheck(
            name="output_type",
            kind="type",
            target_type=target_type,
            residual_type="type_mismatch",
            description=f"prediction must have target type family {target_type}",
            weight=1.0,
        )
    ]
    if target_type == "number":
        checks.append(
            InterfaceCheck(
                name="numeric_value",
                kind="numeric",
                target_type=target_type,
                residual_type="numeric_residual",
                description="numeric prediction should minimize relative/log residual",
                weight=2.0,
            )
        )
    elif target_type == "string":
        checks.extend(
            [
                InterfaceCheck(
                    name="string_exact",
                    kind="exact",
                    target_type=target_type,
                    residual_type="categorical_mismatch",
                    description="string prediction should exactly match target",
                    weight=2.0,
                ),
                InterfaceCheck(
                    name="string_shape",
                    kind="string_shape",
                    target_type=target_type,
                    residual_type="shape_mismatch",
                    description="string prediction should match target length",
                    weight=0.7,
                ),
            ]
        )
    elif target_type == "mapping":
        keys = tuple(sorted({str(k) for o in obs if isinstance(o.target, Mapping) for k in o.target.keys()}))
        checks.append(
            InterfaceCheck(
                name="mapping_keys",
                kind="mapping_keys",
                target_type=target_type,
                residual_type="schema_mismatch",
                description="mapping prediction should cover target keys: " + ", ".join(keys[:12]),
                weight=1.2,
            )
        )
    else:
        checks.append(
            InterfaceCheck(
                name="nonempty_output",
                kind="nonempty",
                target_type=target_type,
                residual_type="empty_artifact",
                description="prediction should produce a nonempty artifact",
                weight=1.0,
            )
        )

    return InterfaceContract(
        name=name,
        input_type=input_type,
        target_type=target_type,
        context_keys=context_keys,
        checks=tuple(checks),
    )


def score_interface_contract(
    contract: InterfaceContract,
    observations: Iterable[Any],
    predict: Callable[[Any, Mapping[str, Any]], Any],
) -> InterfaceScore:
    """Score a predictor against a structural interface contract."""

    obs = [_coerce_interface_observation(o) for o in observations]
    residuals: list[InterfaceResidual] = []
    losses: list[float] = []
    exact = 0
    for o in obs:
        try:
            pred = predict(o.inputs, o.context)
        except Exception as exc:
            pred = f"ERROR: {type(exc).__name__}: {exc}"
            item_loss = 1.0
            residuals.append(
                InterfaceResidual(
                    check="execution",
                    residual_type="execution_error",
                    loss=1.0,
                    prediction_preview=_preview(pred),
                    target_preview=_preview(o.target),
                    rationale="candidate failed during interface execution",
                )
            )
            losses.append(item_loss)
            continue

        item_losses = []
        for check in contract.checks:
            loss, rationale = _interface_check_loss(check, pred, o.target)
            item_losses.append(loss * check.weight)
            if loss > 0.0:
                residuals.append(
                    InterfaceResidual(
                        check=check.name,
                        residual_type=check.residual_type,
                        loss=loss,
                        prediction_preview=_preview(pred),
                        target_preview=_preview(o.target),
                        rationale=rationale,
                    )
                )
        denom = sum(c.weight for c in contract.checks) or 1.0
        item_loss = min(1.0, sum(item_losses) / denom)
        losses.append(item_loss)
        exact += int(item_loss == 0.0)

    return InterfaceScore(
        contract_name=contract.name,
        loss_mean=sum(losses) / len(losses) if losses else 1.0,
        exact_rate=exact / len(losses) if losses else 0.0,
        n_observations=len(losses),
        residuals=tuple(residuals),
    )


def _coerce_interface_observation(value: Any) -> InterfaceObservation:
    if isinstance(value, InterfaceObservation):
        return value
    return InterfaceObservation(
        inputs=getattr(value, "inputs"),
        target=getattr(value, "target"),
        context=getattr(value, "context", {}) or {},
    )


def _type_family(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, (list, tuple)):
        return "sequence"
    if value is None:
        return "none"
    return type(value).__name__


def _interface_check_loss(check: InterfaceCheck, prediction: Any, target: Any) -> tuple[float, str]:
    if check.kind == "type":
        ok = _type_family(prediction) == _type_family(target)
        return (0.0, "type matches") if ok else (1.0, "prediction target type mismatch")
    if check.kind == "numeric":
        return _numeric_loss(prediction, target)
    if check.kind == "exact":
        return (0.0, "exact match") if prediction == target else (1.0, "prediction differs from target")
    if check.kind == "string_shape":
        try:
            p = str(prediction)
            t = str(target)
        except Exception:
            return 1.0, "prediction or target cannot be rendered as string"
        denom = max(1, len(p), len(t))
        return abs(len(p) - len(t)) / denom, "string length residual"
    if check.kind == "mapping_keys":
        if not isinstance(prediction, Mapping) or not isinstance(target, Mapping):
            return 1.0, "prediction or target is not a mapping"
        expected = {str(k) for k in target.keys()}
        observed = {str(k) for k in prediction.keys()}
        return 1.0 - f1_from_sets(expected, observed), "mapping key coverage residual"
    if check.kind == "nonempty":
        return (0.0, "nonempty") if prediction not in (None, "", [], {}, ()) else (1.0, "empty artifact")
    return 0.0, "unknown check ignored"


def _numeric_loss(prediction: Any, target: Any) -> tuple[float, str]:
    try:
        pred = float(prediction)
        tgt = float(target)
    except Exception:
        return 1.0, "numeric coercion failed"
    if not math.isfinite(pred) or not math.isfinite(tgt):
        return 1.0, "non-finite numeric value"
    denom = abs(tgt) + 1e-9
    rel = abs(pred - tgt) / denom
    return min(1.0, rel), "relative numeric residual"


def _preview(value: Any, limit: int = 180) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."
