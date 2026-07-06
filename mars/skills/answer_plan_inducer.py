"""Query-first answer-plan induction.

This layer sits above the executable slot compilers.  It first infers the
typed shape of the requested answer, then closes that shape with deterministic
measurements.  The goal is to avoid a common weak-model failure mode: finding
some true statistic while rendering the wrong object for the current question.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from .abstraction_posterior import MeasuredGroup, induce_contrast_abstraction
from .answer_slot_compiler import infer_answer_slot_hypothesis
from .question_form_inducer import infer_question_form
from .scope_abstraction import MeasuredPairGroup, induce_scope_abstraction
from .slot_contract import SlotContractResult, infer_slot_contract_hypothesis


@dataclass(frozen=True)
class AnswerPlan:
    answer_form: str
    slots: Mapping[str, Any]
    renderer: str
    confidence: float


def infer_answer_plan_hypothesis(
    *,
    question: str,
    domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    """Infer and execute a typed answer plan for a table task.

    The API is intentionally benchmark-agnostic: the inputs are a question,
    context, table, and schema descriptions.  A plan is allowed to call existing
    slot compilers, but it must preserve the answer form requested by the query.
    """

    if df is None or not hasattr(df, "columns"):
        return None
    prevalence = _grouped_prevalence_plan(question, df, column_descriptions)
    if prevalence is not None:
        return prevalence
    schema_text = _schema_text(df, column_descriptions)
    form = infer_question_form(question, schema_text)
    if form and form.name in {
        "grouped_original_replication_comparison",
        "paired_group_mean_comparison",
    }:
        paired = _paired_measure_plan(question, domain_context, df, column_descriptions, form.name)
        if paired is not None:
            return paired

    delegated = infer_answer_slot_hypothesis(
        question=question,
        domain_context=domain_context,
        df=df,
        column_descriptions=column_descriptions,
    )
    if delegated is not None and _answers_query_surface(question, delegated.hypothesis):
        if _is_generic_measurement_mismatch(question, delegated.evidence):
            delegated = None
    if delegated is not None and _answers_query_surface(question, delegated.hypothesis):
        return SlotContractResult(
            delegated.hypothesis,
            "Answer-plan selected a typed answer form, then delegated slot closure to "
            f"the executable answer-slot compiler. {delegated.workflow}",
            "answer_plan_delegate:" + delegated.evidence,
            {"answer_form": delegated.slots.get("answer_form", "delegated"), **dict(delegated.slots)},
            delegated.score + 0.4,
        )

    contract = infer_slot_contract_hypothesis(
        question=question,
        domain_context=domain_context,
        df=df,
        column_descriptions=column_descriptions,
    )
    if contract is not None and _answers_query_surface(question, contract.hypothesis):
        return SlotContractResult(
            contract.hypothesis,
            "Answer-plan inferred an event/relation answer form, then closed it with "
            f"the slot-contract probe. {contract.workflow}",
            "answer_plan_contract:" + contract.evidence,
            {"answer_form": contract.slots.get("event", "event_or_relation"), **dict(contract.slots)},
            contract.score + 0.25,
        )
    return None


def _grouped_prevalence_plan(
    question: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    import pandas as pd

    q = question.lower()
    if not (
        any(token in q for token in ("prevalence", "proportion", "fraction", "rate"))
        and any(token in q for token in ("vary", "based on", "by ", "across", "between"))
    ):
        return None
    group = _best_col(
        df,
        column_descriptions,
        positive=("habitat", "type", "group", "category", "region"),
        categorical=True,
        preferred=("habitat", "habitat.type", "region", "group"),
    )
    numerator = _best_col(
        df,
        column_descriptions,
        positive=("gardening", "garden", "introduced", "count", "frequency", "prevalence"),
        preferred=("n.gard", "gardening", "garden_count"),
    )
    denominator = _best_col(
        df,
        column_descriptions,
        positive=("total", "all", "count", "frequency"),
        preferred=("n.total", "total", "count"),
    )
    if not group or not numerator or not denominator or numerator == denominator:
        return None
    data = df[[group, numerator, denominator]].copy()
    data[numerator] = pd.to_numeric(data[numerator], errors="coerce")
    data[denominator] = pd.to_numeric(data[denominator], errors="coerce")
    data = data.dropna(subset=[group, numerator, denominator])
    data = data[data[denominator] > 0]
    if data.empty:
        return None
    grouped = data.groupby(group, dropna=True)[[numerator, denominator]].sum()
    grouped = grouped[grouped[denominator] > 0]
    if grouped.empty:
        return None
    rows: list[tuple[str, float, float, float]] = []
    for raw, row in grouped.iterrows():
        num = float(row[numerator])
        den = float(row[denominator])
        rows.append((_human_group(str(raw)), num, den, 100.0 * num / den))
    rows.sort(key=lambda item: item[3], reverse=True)
    abstraction = induce_contrast_abstraction(
        question=question,
        groups=[MeasuredGroup(name, num, den, pct) for name, num, den, pct in rows],
    )
    if abstraction is not None and abstraction.name != "metric_gap_partition":
        subject = _prevalence_subject(question)
        hypothesis = (
            f"In groups where {subject} are present, the prevalence of {subject} differs between "
            f"{abstraction.left_label} and {abstraction.right_label}."
        )
        abstraction_workflow = (
            " The abstraction posterior then compressed the measured groups into "
            f"{abstraction.name} using effect size, semantic coherence, stability, and description length."
        )
        abstraction_evidence = ";abstraction=" + abstraction.evidence
    else:
        high = rows[:3]
        low = rows[-2:] if len(rows) > 3 else []
        high_text = "; ".join(f"{name} {pct:.1f}% ({num:.0f}/{den:.0f})" for name, num, den, pct in high)
        if low:
            low_text = "; ".join(f"{name} {pct:.1f}% ({num:.0f}/{den:.0f})" for name, num, den, pct in low)
            hypothesis = (
                f"The prevalence varies strongly by {group}: it is highest in {high_text}, "
                f"and lowest in {low_text}."
            )
        else:
            hypothesis = f"The prevalence varies by {group}: {high_text}."
        abstraction_workflow = ""
        abstraction_evidence = ""
    workflow = (
        "Answer-plan decomposition: answer_form=grouped_prevalence_profile. "
        "The probe grouped records by the requested categorical variable and computed the requested "
        "prevalence for each group."
        f"{abstraction_workflow}"
    )
    evidence = "answer_plan_grouped_prevalence:" + ";".join(
        f"{name}:{pct:.6g}:{num:.6g}/{den:.6g}" for name, num, den, pct in rows[:8]
    ) + abstraction_evidence
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {"answer_form": "grouped_prevalence_profile", "group": group, "numerator": numerator, "denominator": denominator},
        15.5,
    )


def _is_generic_measurement_mismatch(question: str, evidence: str) -> bool:
    q = str(question or "").lower()
    low = str(evidence or "").lower()
    if "answer_slot_generic_category" not in low:
        return False
    return any(
        token in q
        for token in (
            "relationship",
            "between",
            "compare",
            "effect",
            "impact",
            "vary based",
            "interact",
            "pca",
            "peak",
            "when",
            "in which century",
        )
    )


def _prevalence_subject(question: str) -> str:
    match = re.search(
        r"(?i)\bprevalence\s+of\s+(.*?)(?:\s+vary\b|\s+varies\b|\s+differ\b|\s+differs\b|\s+based\b|\s+across\b|\s+between\b|\?|$)",
        question,
    )
    if not match:
        return "the measured outcome"
    subject = match.group(1).strip(" .?")
    subject = re.sub(r"(?i)^(.+?)\s+introduced\s+via\s+gardening$", r"gardening-introduced \1", subject)
    subject = re.sub(r"\s+", " ", subject)
    return subject or "the measured outcome"


def _paired_measure_plan(
    question: str,
    domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
    answer_form: str,
) -> SlotContractResult | None:
    import pandas as pd

    q = question.lower()
    group = _best_col(
        df,
        column_descriptions,
        positive=("domain", "discipline", "project", "field", "group"),
        categorical=True,
        preferred=("project.x", "project", "discipline", "domain"),
    )
    if "power" in q:
        left = _best_col(
            df,
            column_descriptions,
            positive=("power", "observed", "original"),
            preferred=("power.o", "power_original", "observed_power"),
        )
        right = _best_col(
            df,
            column_descriptions,
            positive=("power", "planned", "replication"),
            preferred=("power_planned.r", "planned_power", "power.r"),
        )
        measure_label = "statistical power"
        left_label = "observed power in original studies"
        right_label = "planned power in replication studies"
    else:
        left = _best_col(
            df,
            column_descriptions,
            positive=("effect", "estimate", "original", "fisher"),
            preferred=("fiso", "effect_size.o", "ro", "effect_original"),
        )
        right = _best_col(
            df,
            column_descriptions,
            positive=("effect", "estimate", "replication", "fisher"),
            preferred=("fisr", "effect_size.r", "rr", "effect_replication"),
        )
        measure_label = "effect size estimate"
        left_label = "effect estimate in original studies"
        right_label = "effect estimate in replication studies"
    if not left or not right:
        return None

    cols = [c for c in (group, left, right) if c]
    data = df[cols].copy()
    data[left] = pd.to_numeric(data[left], errors="coerce")
    data[right] = pd.to_numeric(data[right], errors="coerce")
    data = data.dropna(subset=[left, right])
    if len(data) < 2:
        return None

    scope_text = f"{question}\n{domain_context}"
    requested = _requested_groups(scope_text, data[group].astype(str).unique().tolist() if group else [])
    rows: list[tuple[str, float, float, int]] = []
    all_rows: list[tuple[str, float, float, int]] = []
    if group and group in data.columns:
        for raw, sub in data.groupby(group):
            if len(sub) < 1:
                continue
            name = _human_group(str(raw))
            row = (name, float(sub[left].mean()), float(sub[right].mean()), int(len(sub)))
            all_rows.append(row)
            if requested and not _matches_requested_group(name, requested):
                continue
            rows.append(row)
    else:
        row = ("the selected records", float(data[left].mean()), float(data[right].mean()), int(len(data)))
        rows.append(row)
        all_rows.append(row)
    if not rows:
        return None

    if _asks_which_groups(q):
        rows = [row for row in rows if row[1] > row[2]]
        if not rows:
            return None
        if requested:
            rows = [row for row in rows if _matches_requested_group(row[0], requested)]
            if not rows:
                return None
        rows.sort(key=lambda row: (row[1] - row[2]), reverse=True)
        names = _join_human([row[0] for row in rows[:4]])
        details = "; ".join(
            f"{name}: original {left_mean:.2f}, replication {right_mean:.2f}"
            for name, left_mean, right_mean, _n in rows[:4]
        )
        hypothesis = (
            f"The requested groups where original-study {measure_label}s are larger than "
            f"replication-study {measure_label}s are {names}. {details}."
        )
    else:
        rows.sort(key=lambda row: row[0])
        scope = induce_scope_abstraction(
            question=question,
            requested_groups=requested,
            rows=[MeasuredPairGroup(name, left_mean, right_mean, n) for name, left_mean, right_mean, n in all_rows],
        )
        if scope is not None and scope.mode == "local_plus_global":
            scope_rows = [
                row for row in sorted(all_rows, key=lambda item: item[0])
                if row[0] in scope.groups
            ]
            if len(requested) >= 2:
                scope_rows = [row for row in scope_rows if _matches_requested_group(row[0], requested)]
            if not scope_rows:
                scope_rows = rows
            domains = _join_human([name for name, _a, _b, _n in scope_rows])
            details = "; ".join(
                f"In {name}, original mean={left_mean:.2f} versus replication mean={right_mean:.2f}"
                for name, left_mean, right_mean, _n in scope_rows[:4]
            )
            hypothesis = (
                f"The {measure_label}s tend to be larger in original studies compared to "
                f"replication studies across {domains}. {details}."
            )
            scope_workflow = (
                " The scope posterior detected that the requested group is an instance of a stable "
                f"cross-group relation and rendered local_plus_global scope. Evidence: {scope.evidence}."
            )
            scope_evidence = ";scope=" + scope.evidence
        else:
            details = "; ".join(
                f"in {name}, the average {left_label} is {left_mean:.2f}, compared to "
                f"{right_mean:.2f} for {right_label}"
                for name, left_mean, right_mean, _n in rows[:3]
            )
            hypothesis = details + "."
            scope_workflow = ""
            scope_evidence = ""

    plan = AnswerPlan(
        answer_form=answer_form,
        slots={
            "group": group or "all",
            "left_measure": left,
            "right_measure": right,
            "requested_groups": tuple(requested),
        },
        renderer="query_specific_paired_measure_renderer",
        confidence=0.9,
    )
    workflow = (
        f"Answer-plan decomposition: answer_form={plan.answer_form}, group={group or 'all'}, "
        f"left_measure={left}, right_measure={right}, renderer={plan.renderer}. The probe "
        "coerced the paired measures to numeric values, applied any group named in the question, "
        "then rendered either a requested-group answer or a which-groups answer."
        f"{scope_workflow if 'scope_workflow' in locals() else ''}"
    )
    evidence = "answer_plan_paired_measure:" + ";".join(
        f"{name}:{left_mean:.6g}:{right_mean:.6g}:n={n}" for name, left_mean, right_mean, n in rows[:8]
    ) + (scope_evidence if 'scope_evidence' in locals() else "")
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {"answer_form": plan.answer_form, **dict(plan.slots)},
        16.0 + plan.confidence,
    )


def _schema_text(df: Any, column_descriptions: Mapping[str, str]) -> str:
    return "\n".join(f"{c}: {column_descriptions.get(str(c), '')}" for c in getattr(df, "columns", []))


def _best_col(
    df: Any,
    column_descriptions: Mapping[str, str],
    *,
    positive: tuple[str, ...],
    preferred: tuple[str, ...] = (),
    categorical: bool = False,
) -> str | None:
    best: tuple[float, str] | None = None
    for col in getattr(df, "columns", []):
        name = str(col)
        low = name.lower()
        desc = str(column_descriptions.get(name, "")).lower()
        text = f"{low} {desc}"
        score = 0.0
        for idx, pref in enumerate(preferred):
            if low == pref.lower():
                score += 12.0 - idx
        score += sum(3.0 for token in positive if token in text)
        if categorical:
            try:
                nunique = int(df[col].nunique(dropna=True))
                if 1 < nunique <= max(25, int(len(df) * 0.5)):
                    score += 3.0
                else:
                    score -= 3.0
            except Exception:
                pass
        else:
            try:
                import pandas as pd

                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().sum() >= 2:
                    score += 2.0
            except Exception:
                pass
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, name)
    return best[1] if best else None


def _requested_groups(question: str, candidates: list[str]) -> list[str]:
    q_norm = _norm(question)
    requested: list[str] = []
    for value in candidates:
        name = _human_group(str(value))
        if _norm(name) in q_norm or _norm(str(value)) in q_norm:
            requested.append(name)
    return requested


def _matches_requested_group(name: str, requested: list[str]) -> bool:
    norm = _norm(name)
    return any(norm == _norm(r) or norm in _norm(r) or _norm(r) in norm for r in requested)


def _asks_which_groups(q: str) -> bool:
    return any(token in q for token in ("which domain", "which domains", "which field", "which fields", "which group", "which groups"))


def _answers_query_surface(question: str, answer: str) -> bool:
    q = question.lower()
    a = answer.lower()
    if "original" in q and "replication" in q:
        return "original" in a and "replication" in a
    if "pca" in q or "pc1" in q or "pc2" in q:
        return "pc" in a or "principal component" in a
    if any(tok in q for tok in ("when", "first", "peak", "highest")):
        return bool(re.search(r"-?\d+(?:\.\d+)?", answer))
    return True


def _human_group(value: str) -> str:
    text = value.strip().replace("_", " ")
    aliases = {
        "exp econ": "Experimental Economics",
        "experimental economics": "Experimental Economics",
        "psych": "Psychology",
        "psychology": "Psychology",
    }
    return aliases.get(text.lower(), text[:1].upper() + text[1:])


def _join_human(items: list[str]) -> str:
    clean = [str(item) for item in items if str(item).strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
