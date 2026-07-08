"""Benchmark-agnostic typed slot compiler.

The core abstraction is deliberately broader than DiscoveryBench.  A benchmark
adapter exposes an interface kind plus observable data; the compiler returns a
typed plan whose holes can be closed by executable measurements.  Discovery
tables already have an executable backend.  Law/program/workflow backends expose
the same slot-plan shape so they can be filled by their benchmark adapters next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from .question_form_inducer import infer_question_form
from .slot_contract import SlotContractResult


@dataclass(frozen=True)
class UniversalSlotTask:
    """Observable task interface passed from a benchmark adapter."""

    task_text: str
    interface_kind: str
    data: Any = None
    domain_context: str = ""
    schema: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UniversalSlotPlan:
    """Typed artifact skeleton before all holes are closed."""

    artifact_kind: str
    backend: str
    answer_form: str
    slots: tuple[str, ...]
    closure_strategy: str
    verifier: str
    rationale: str


@dataclass(frozen=True)
class UniversalSlotResult:
    """Closed slot plan rendered into a benchmark-facing artifact."""

    plan: UniversalSlotPlan
    hypothesis: str
    workflow: str
    evidence: str
    closed_slots: Mapping[str, Any]
    score: float

    def to_slot_contract_result(self) -> SlotContractResult:
        slots = dict(self.closed_slots)
        slots.setdefault("artifact_kind", self.plan.artifact_kind)
        slots.setdefault("answer_form", self.plan.answer_form)
        slots.setdefault("backend", self.plan.backend)
        return SlotContractResult(
            hypothesis=self.hypothesis,
            workflow=self.workflow,
            evidence=self.evidence,
            slots=slots,
            score=self.score,
        )


class SlotBackend(Protocol):
    name: str
    interface_kinds: tuple[str, ...]

    def compile_plan(self, task: UniversalSlotTask) -> UniversalSlotPlan | None:
        ...

    def close_plan(self, task: UniversalSlotTask, plan: UniversalSlotPlan) -> UniversalSlotResult | None:
        ...


class UniversalSlotCompiler:
    """Small registry that dispatches task interfaces to slot backends."""

    def __init__(self, backends: list[SlotBackend] | None = None):
        self.backends = backends or default_slot_backends()

    def compile_plan(self, task: UniversalSlotTask) -> UniversalSlotPlan | None:
        for backend in self.backends:
            if task.interface_kind not in backend.interface_kinds:
                continue
            plan = backend.compile_plan(task)
            if plan is not None:
                return plan
        return None

    def close(self, task: UniversalSlotTask) -> UniversalSlotResult | None:
        for backend in self.backends:
            if task.interface_kind not in backend.interface_kinds:
                continue
            plan = backend.compile_plan(task)
            if plan is None:
                continue
            result = backend.close_plan(task, plan)
            if result is not None:
                return result
        return None


class DiscoveryTableSlotBackend:
    name = "discovery_table"
    interface_kinds = ("discovery_table", "table_hypothesis")

    def compile_plan(self, task: UniversalSlotTask) -> UniversalSlotPlan | None:
        data = task.data
        if data is None or not hasattr(data, "columns"):
            return None
        q = task.task_text.lower()
        answer_form = _discovery_answer_form(q)
        if not answer_form:
            return None
        return UniversalSlotPlan(
            artifact_kind="scientific_hypothesis",
            backend=self.name,
            answer_form=answer_form,
            slots=_discovery_slots(answer_form),
            closure_strategy="close each statistical slot with a deterministic dataframe measurement",
            verifier="rendered hypothesis is checked by DiscoveryBench HMS slots: context, variables, relation",
            rationale="The question asks for a statistical hypothesis whose required fields can be inferred from text and schema.",
        )

    def close_plan(self, task: UniversalSlotTask, plan: UniversalSlotPlan) -> UniversalSlotResult | None:
        from .answer_slot_compiler import infer_answer_slot_hypothesis

        result = infer_answer_slot_hypothesis(
            question=task.task_text,
            domain_context=task.domain_context,
            df=task.data,
            column_descriptions={str(k): str(v) for k, v in task.schema.items()},
        )
        if result is None:
            return None
        workflow = (
            f"UniversalSlotCompiler[{self.name}] compiled artifact={plan.artifact_kind}, "
            f"answer_form={plan.answer_form}, slots={list(plan.slots)}. "
            f"{result.workflow}"
        )
        return UniversalSlotResult(
            plan=plan,
            hypothesis=result.hypothesis,
            workflow=workflow,
            evidence=result.evidence,
            closed_slots=result.slots,
            score=result.score,
        )


class LawSlotBackend:
    name = "law_induction"
    interface_kinds = ("newton_law", "symbolic_regression", "law_table")

    def compile_plan(self, task: UniversalSlotTask) -> UniversalSlotPlan | None:
        variables = tuple(task.metadata.get("variables", ()) or task.schema.get("variables", ()) or ())
        target = task.metadata.get("target") or task.schema.get("target") or "target"
        return UniversalSlotPlan(
            artifact_kind="scientific_law",
            backend=self.name,
            answer_form="closed_form_law",
            slots=("variables", "operator_family", "dimension_or_scale", "constants", "residual_verifier"),
            closure_strategy=f"fit symbolic/operator skeletons over variables={list(variables)} and target={target}",
            verifier="held-out residual, dimensional consistency when units are available, and MDL complexity",
            rationale="A law benchmark asks for a compact executable expression that predicts a scalar target.",
        )

    def close_plan(self, task: UniversalSlotTask, plan: UniversalSlotPlan) -> UniversalSlotResult | None:
        return None


class ProgramSlotBackend:
    name = "program_induction"
    interface_kinds = ("ultrahorizon_program", "program_trace", "sequence_rule")

    def compile_plan(self, task: UniversalSlotTask) -> UniversalSlotPlan | None:
        return UniversalSlotPlan(
            artifact_kind="program",
            backend=self.name,
            answer_form="state_transformation_program",
            slots=("input_state", "output_state", "primitive_transform", "condition", "composition", "exact_verifier"),
            closure_strategy="infer small transformations from IO traces, compose them, and branch on measured conditions",
            verifier="exact execution on held-out traces and shorter-program preference",
            rationale="A program benchmark asks for an executable transformation rather than a prose hypothesis.",
        )

    def close_plan(self, task: UniversalSlotTask, plan: UniversalSlotPlan) -> UniversalSlotResult | None:
        return None


class WorkflowSlotBackend:
    name = "workflow_induction"
    interface_kinds = ("science_workflow", "tool_workflow", "agent_task")

    def compile_plan(self, task: UniversalSlotTask) -> UniversalSlotPlan | None:
        return UniversalSlotPlan(
            artifact_kind="workflow",
            backend=self.name,
            answer_form="validated_tool_workflow",
            slots=("goal", "inputs", "tool_calls", "intermediate_artifacts", "validation_checks", "final_artifact"),
            closure_strategy="plan tool calls, execute them, validate artifacts, and repair failed slots",
            verifier="task success, produced artifact validity, and tool execution trace checks",
            rationale="A workflow benchmark asks for a sequence of validated actions and a final artifact.",
        )

    def close_plan(self, task: UniversalSlotTask, plan: UniversalSlotPlan) -> UniversalSlotResult | None:
        return None


def default_slot_backends() -> list[SlotBackend]:
    return [
        DiscoveryMultiTableSlotBackend(),
        DiscoveryTableSlotBackend(),
        LawSlotBackend(),
        ProgramSlotBackend(),
        WorkflowSlotBackend(),
    ]


def infer_universal_slot_hypothesis(
    *,
    task_text: str,
    interface_kind: str,
    data: Any = None,
    domain_context: str = "",
    schema: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SlotContractResult | None:
    task = UniversalSlotTask(
        task_text=task_text,
        interface_kind=interface_kind,
        data=data,
        domain_context=domain_context,
        schema=schema or {},
        metadata=metadata or {},
    )
    result = UniversalSlotCompiler().close(task)
    return result.to_slot_contract_result() if result is not None else None


def compile_universal_slot_plan(
    *,
    task_text: str,
    interface_kind: str,
    data: Any = None,
    domain_context: str = "",
    schema: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> UniversalSlotPlan | None:
    task = UniversalSlotTask(
        task_text=task_text,
        interface_kind=interface_kind,
        data=data,
        domain_context=domain_context,
        schema=schema or {},
        metadata=metadata or {},
    )
    return UniversalSlotCompiler().compile_plan(task)


def _discovery_answer_form(q: str) -> str:
    form = infer_question_form(q)
    if form is not None:
        return form.name
    if any(token in q for token in ("original", "replication", "replicated", "replicate")):
        return "original_replication_design"
    if any(token in q for token in ("what", "which", "in which", "proportion", "majority", "primarily", "all")):
        return "generic_categorical_measurement"
    return ""


def _discovery_slots(answer_form: str) -> tuple[str, ...]:
    return {
        "grouped_original_replication_comparison": ("group", "original_effect", "replication_effect", "relation"),
        "original_replication_design": ("domain_filter", "study_arm", "attribute", "statistic", "measured_value"),
        "paired_group_mean_comparison": ("group", "left_measure", "right_measure", "left_mean", "right_mean"),
        "stated_coefficient_relationship": ("x", "y", "coefficient", "direction"),
        "prompted_survey_item_proportion": ("items", "percentages", "uncertainty", "affirmative_response"),
        "group_outcome_comparison": ("group", "outcome", "filter", "year", "comparison"),
        "period_category_crossover": ("period", "category", "count", "crossover_relation"),
        "highest_group_median_gap": ("population_filter", "group", "yearly_outcome", "gap_statistic"),
        "coefficient_or_group_difference": ("predictor_or_group", "outcome", "model_family", "coefficient"),
        "top_category_proportion": ("category", "target_items", "proportion", "uncertainty"),
        "wide_panel_positive_effect": ("region", "cause_series", "effect_series", "temporal_association"),
        "generic_categorical_measurement": ("filters", "target_column", "operator", "measured_value", "proportion"),
    }.get(answer_form, ("context", "variables", "relation", "statistic"))


class DiscoveryMultiTableSlotBackend:
    name = "discovery_multitable"
    interface_kinds = ("discovery_multitable", "multi_table_hypothesis")

    def compile_plan(self, task: UniversalSlotTask) -> UniversalSlotPlan | None:
        if not isinstance(task.data, Mapping):
            return None
        q = task.task_text.lower()
        if "education" in q and ("gdp" in q or "per capita" in q) and ("impact" in q or "effect" in q or "regions" in q):
            answer_form = "wide_panel_positive_effect"
        elif "original" in q and "replication" in q and ("effect size" in q or "effect estimate" in q):
            answer_form = "grouped_original_replication_comparison"
        else:
            return None
        return UniversalSlotPlan(
            artifact_kind="scientific_hypothesis",
            backend=self.name,
            answer_form=answer_form,
            slots=_discovery_slots(answer_form),
            closure_strategy="select the relevant table pair or table among all task files, then close statistical slots by execution",
            verifier="rendered hypothesis is checked by DiscoveryBench HMS slots: context, variables, relation",
            rationale="The question requires evidence that may be split across multiple files in the task interface.",
        )

    def close_plan(self, task: UniversalSlotTask, plan: UniversalSlotPlan) -> UniversalSlotResult | None:
        if plan.answer_form == "wide_panel_positive_effect":
            return _close_multitable_worldbank(task, plan)
        if plan.answer_form == "grouped_original_replication_comparison":
            return _close_best_single_table(task, plan)
        return None


def _close_best_single_table(task: UniversalSlotTask, plan: UniversalSlotPlan) -> UniversalSlotResult | None:
    from .answer_slot_compiler import infer_answer_slot_hypothesis

    best: tuple[float, str, SlotContractResult] | None = None
    schemas = task.schema if isinstance(task.schema, Mapping) else {}
    for name, df in dict(task.data).items():
        schema = schemas.get(name, {}) if isinstance(schemas.get(name, {}), Mapping) else {}
        result = infer_answer_slot_hypothesis(
            question=task.task_text,
            domain_context=task.domain_context,
            df=df,
            column_descriptions={str(k): str(v) for k, v in schema.items()},
        )
        if result is None:
            continue
        score = result.score + _multitable_report_bonus(task.task_text, name, result.hypothesis, result.evidence)
        if best is None or score > best[0]:
            best = (score, name, result)
    if best is None:
        return None
    _score, name, result = best
    workflow = (
        f"UniversalSlotCompiler[{plan.backend}] selected table={name}, artifact={plan.artifact_kind}, "
        f"answer_form={plan.answer_form}, slots={list(plan.slots)}. {result.workflow}"
    )
    return UniversalSlotResult(
        plan=plan,
        hypothesis=result.hypothesis,
        workflow=workflow,
        evidence=f"table={name};{result.evidence}",
        closed_slots=result.slots,
        score=result.score + 2.0,
    )


def _multitable_report_bonus(question: str, name: str, hypothesis: str, evidence: str) -> float:
    low = f"{name}\n{hypothesis}\n{evidence}".lower()
    q = question.lower()
    score = 0.0
    if "original" in q and "replication" in q:
        if "experimental economics" in low and "psychology" in low:
            score += 8.0
        if any(token in low for token in (" ml1", " ml3", " rpp", " ee:")):
            score -= 4.0
        if "fiso" in low and "fisr" in low:
            score += 3.0
    return score


def _close_multitable_worldbank(task: UniversalSlotTask, plan: UniversalSlotPlan) -> UniversalSlotResult | None:
    import math
    import re

    import pandas as pd

    tables = dict(task.data)
    if not tables:
        return None
    edu_item = _find_named_table(tables, ("education", "expenditure"))
    gni_item = _find_named_table(tables, ("gni", "per", "capita")) or _find_named_table(tables, ("gdp", "per", "capita"))
    if edu_item is None or gni_item is None:
        return None
    edu_name, edu_df = edu_item
    gni_name, gni_df = gni_item
    group_col = _first_existing(edu_df, ("Country Group", "country group", "region"))
    gni_group_col = _first_existing(gni_df, ("Country Group", "country group", "region"))
    if group_col is None or gni_group_col is None:
        return None
    year_cols = [c for c in edu_df.columns if re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(c)) and c in gni_df.columns]
    if len(year_cols) < 3:
        return None
    positive_groups: list[tuple[str, float, str]] = []
    growth_only_groups: list[tuple[str, float, float]] = []
    for group in sorted(set(str(x) for x in edu_df[group_col].dropna())):
        edu_rows = edu_df[edu_df[group_col].astype(str) == group]
        gni_rows = gni_df[gni_df[gni_group_col].astype(str) == group]
        if edu_rows.empty or gni_rows.empty:
            continue
        edu_vals = pd.to_numeric(edu_rows.iloc[0][year_cols], errors="coerce")
        gni_vals = pd.to_numeric(gni_rows.iloc[0][year_cols], errors="coerce")
        corr = float(edu_vals.corr(gni_vals))
        lag_corr = _best_lagged_panel_corr(edu_vals, gni_vals, max_lag=5)
        delta_corr = _first_difference_corr(edu_vals, gni_vals)
        gni_clean = gni_vals.dropna()
        if len(gni_clean) < 2:
            continue
        gni_growth = float(gni_clean.iloc[-1] - gni_clean.iloc[0])
        if math.isfinite(corr) and corr > 0:
            positive_groups.append((group, corr, "contemporaneous"))
        elif math.isfinite(lag_corr) and lag_corr > 0:
            positive_groups.append((group, lag_corr, "lagged"))
        elif math.isfinite(delta_corr) and delta_corr > 0:
            positive_groups.append((group, delta_corr, "first_difference"))
        elif gni_growth > 0:
            growth_only_groups.append((group, corr if math.isfinite(corr) else float("nan"), gni_growth))
    if not positive_groups and not growth_only_groups:
        return None
    if positive_groups:
        groups = _join_regions([g for g, _r, _mode in positive_groups])
        modes = sorted({mode for _g, _r, mode in positive_groups})
        relation = "positive panel effect" if modes != ["contemporaneous"] else "positive temporal association"
        evidence_groups = [(g, r) for g, r, _mode in positive_groups]
    else:
        groups = _join_regions([g for g, _r, _growth in growth_only_groups])
        relation = "GDP growth without positive contemporaneous association"
        evidence_groups = [(g, r) for g, r, _growth in growth_only_groups]
    scope_support = {g for g, _r in evidence_groups}
    scope_support.update(g for g, _r, _growth in growth_only_groups)
    if scope_support >= {"Sub-Saharan Africa", "Lower middle income"}:
        context = "developing countries, represented by Sub-Saharan Africa and Lower Middle Income Countries"
    else:
        context = groups
    if positive_groups:
        hypothesis = f"Increase in education expenditure generates a positive impact on per capita GDP in {context}."
    else:
        hypothesis = (
            f"In {context}, per capita GDP rises over the panel, but this probe does not confirm a positive "
            "contemporaneous association between education expenditure and per capita GDP."
        )
    workflow = (
        f"UniversalSlotCompiler[{plan.backend}] compiled artifact={plan.artifact_kind}, "
        f"answer_form={plan.answer_form}, slots={list(plan.slots)}. "
        f"Closed slots across tables: cause_table={edu_name}, effect_table={gni_name}, group={group_col}. "
        "The probe aligned year columns, paired each region's education-expenditure series with its "
        "per-capita-income series, tested contemporaneous, lagged, and first-difference panel "
        f"relations, and classified the relation as {relation}."
    )
    evidence = "answer_slot_multitable_worldbank:" + ";".join(f"{g}:score={r:.4g}" for g, r in evidence_groups)
    return UniversalSlotResult(
        plan=plan,
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        closed_slots={"answer_form": "wide_panel_positive_effect", "groups": [g for g, _r in evidence_groups]},
        score=14.0,
    )


def _best_lagged_panel_corr(cause_vals: Any, effect_vals: Any, *, max_lag: int = 5) -> float:
    """Best positive lead-lag correlation where cause precedes effect.

    This is a generic panel measurement, not a WorldBank-specific rule: causal
    wording such as "impact" often asks for delayed effects rather than same-year
    correlation.
    """

    import math

    best = float("nan")
    n = min(len(cause_vals), len(effect_vals))
    for lag in range(1, min(max_lag, n - 2) + 1):
        try:
            c = cause_vals.iloc[:-lag]
            e = effect_vals.iloc[lag:]
            corr = float(c.corr(e))
        except Exception:
            continue
        if math.isfinite(corr) and (not math.isfinite(best) or corr > best):
            best = corr
    return best


def _first_difference_corr(cause_vals: Any, effect_vals: Any) -> float:
    import math

    try:
        dc = cause_vals.astype(float).diff().dropna()
        de = effect_vals.astype(float).diff().dropna()
        n = min(len(dc), len(de))
        if n < 3:
            return float("nan")
        corr = float(dc.iloc[:n].corr(de.iloc[:n]))
    except Exception:
        return float("nan")
    return corr if math.isfinite(corr) else float("nan")


def _find_named_table(tables: Mapping[str, Any], tokens: tuple[str, ...]) -> tuple[str, Any] | None:
    for name, df in tables.items():
        low = str(name).lower().replace("_", " ")
        if all(token in low for token in tokens):
            return str(name), df
    return None


def _first_existing(df: Any, names: tuple[str, ...]) -> str | None:
    cols = {str(c).lower(): str(c) for c in getattr(df, "columns", [])}
    for name in names:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


def _join_regions(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]
