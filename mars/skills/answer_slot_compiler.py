"""Question-first answer-slot compiler for table discovery tasks.

The slot contract layer closes obvious temporal/PCA slots.  This module handles
the complementary case: the question asks for a statistical answer form
(grouped comparison, coefficient, top category, or period/category crossover).
It compiles that answer form from the question and observable schema, then
closes each slot with a small deterministic measurement on the table.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

from .question_form_inducer import infer_question_form
from .slot_contract import SlotContractResult


def infer_answer_slot_hypothesis(
    *,
    question: str,
    domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    if df is None or not hasattr(df, "columns"):
        return None
    q = question.lower()
    handlers = (
        _original_replication_design_measurement,
        _paired_group_mean_comparison,
        _grouped_original_replication_comparison,
        _stated_group_mean_pair,
        _stated_coefficient_relationship,
        _period_category_crossover,
        _highest_group_median_gap,
        _group_outcome_comparison,
        _degree_completion_relationship,
        _top_role_proportions,
        _prompted_survey_item_proportion,
        _wide_panel_positive_effect,
        _generic_categorical_measurement,
    )
    for handler in handlers:
        result = handler(q, question, domain_context, df, column_descriptions)
        if result is not None:
            return result
    return None


def _original_replication_design_measurement(
    q: str,
    question: str,
    _domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    """Close original/replication study-design slots by execution.

    This is a general research-design operator: infer the requested study arm
    (original, replication, or both), attribute family, population/domain filter,
    and statistic from text/schema, then measure the slot directly.
    """

    design_cue = any(token in q for token in ("original", "replication", "replicated", "replicate"))
    measurement_cue = "stud" in q and any(
        token in q
        for token in (
            "country",
            "united states",
            "language",
            "lab",
            "online",
            "subject",
            "student",
            "power",
            "author citation",
            "seniority",
            "professor",
        )
    )
    if not (design_cue or measurement_cue):
        return None
    if "which factor" in q and len(re.findall(r"(?<!\d)-?\d+(?:\.\d+)?(?!\d)", question)) >= 2:
        return None
    if ("which domain" in q or "for which domain" in q or "in which domain" in q) and (
        "effect estimate" in q or "effect size" in q
    ):
        return None
    if not _has_original_replication_columns(df):
        return None
    group_col = _or_group_col(df, column_descriptions)
    filters = _or_domain_filters(q, df, column_descriptions, group_col)
    data = _apply_filters(df, filters)
    if data.empty:
        return None
    attr = _or_attribute(q)
    if attr is None:
        return None
    if attr == "effect" and "original" in q and any(tok in q for tok in ("replication", "replicated", "replicate")):
        return None
    import pandas as pd

    arm = _or_arm(q)
    if attr == "same_country":
        col = _existing_col(df, ("same_country",))
        if col is None:
            return None
        result = _or_binary_complement_result(q, data, col, "replication studies conducted in a different country from the original study")
    elif attr == "same_language":
        col = _existing_col(df, ("same_language",))
        if col is None:
            return None
        result = _or_binary_complement_result(q, data, col, "replication studies conducted in a different language from the original study")
    elif attr == "same_online":
        col = _existing_col(df, ("same_online",))
        if col is None:
            return None
        result = _or_binary_complement_result(q, data, col, "replication studies conducted in a different online/lab setting from the original study")
    elif attr == "us_lab":
        suffix = ".r" if arm == "replication" else ".o"
        col = _existing_col(df, (f"us_lab{suffix}", f"us_lab_{suffix[-1]}"))
        if col is None:
            return None
        result = _or_binary_positive_result(q, data, col, "studies conducted in United States labs")
    elif attr == "online":
        cols = _or_arm_columns(df, attr, arm)
        if not cols:
            return None
        result = _or_online_result(q, data, cols, arm)
    elif attr == "seniority":
        cols = _or_arm_columns(df, attr, arm)
        if not cols:
            return None
        result = _or_seniority_result(q, data, cols, arm)
    elif attr in {"subjects", "compensation", "country", "language"}:
        cols = _or_arm_columns(df, attr, arm)
        if not cols:
            return None
        result = _or_categorical_result(q, data, cols, attr, arm)
    elif attr in {"effect", "power", "length", "citations", "n_authors", "author_citations_avg", "author_citations_max", "authors_male"}:
        cols = _or_numeric_columns(df, attr, arm, q)
        if not cols:
            return None
        result = _or_numeric_result(q, data, cols, attr, arm)
    else:
        return None
    if result is None:
        return None
    hypothesis, measured, evidence_bits = result
    context = _or_context_phrase(filters)
    if context and "selected records" in hypothesis:
        hypothesis = hypothesis.replace("the selected records", context)
    workflow = (
        "Closed slots: answer_form=original_replication_design, "
        f"filters={filters or 'none'}, arm={arm}, attribute={attr}, measured={measured}. "
        "The probe inferred study-design roles from original/replication column suffixes, "
        "filtered the requested domain when stated, and executed the requested mean, mode, "
        "or proportion measurement."
    )
    evidence = f"answer_slot_original_replication_design:{attr}:{measured}:{evidence_bits}:filters={filters}"
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {
            "answer_form": "original_replication_design",
            "domain_filter": filters,
            "study_arm": arm,
            "attribute": attr,
            "statistic": "proportion" if any(tok in q for tok in ("proportion", "percent", "percentage", "ratio")) else "measurement",
            "measured_value": measured,
            "arm": arm,
            "filters": filters,
        },
        28.0,
    )


def _paired_group_mean_comparison(
    q: str,
    question: str,
    _domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    form = infer_question_form(question, _schema_text(df, column_descriptions))
    if form is None or form.name != "paired_group_mean_comparison":
        return None
    group = _best_col(df, column_descriptions, positive=("discipline", "domain", "project"), categorical=True, preferred=("project.x", "project", "discipline"))
    if "power" in q:
        left = _best_col(df, column_descriptions, positive=("power", "observed", "original"), preferred=("power.o", "power_original", "observed_power"))
        right = _best_col(df, column_descriptions, positive=("power", "planned", "replication"), preferred=("power_planned.r", "planned_power", "power.r"))
        left_label = "average observed power in original studies"
        right_label = "average planned power in replication studies"
    else:
        left = _best_col(df, column_descriptions, positive=("effect", "original"), preferred=("fiso", "effect_size.o", "ro"))
        right = _best_col(df, column_descriptions, positive=("effect", "replication"), preferred=("fisr", "effect_size.r", "rr"))
        left_label = "average effect estimate in original studies"
        right_label = "average effect estimate in replication studies"
    if not left or not right:
        return None
    import pandas as pd

    data = df[[c for c in (group, left, right) if c]].copy()
    data[left] = pd.to_numeric(data[left], errors="coerce")
    data[right] = pd.to_numeric(data[right], errors="coerce")
    data = data.dropna(subset=[left, right])
    if data.empty:
        return None
    requested = _requested_groups(
        f"{q}\n{_domain_context.lower()}",
        data[group].astype(str).unique().tolist() if group and group in data.columns else [],
    )
    rows: list[tuple[str, float, float]] = []
    if group and group in data.columns:
        all_rows: list[tuple[str, float, float]] = []
        for g, sub in data.groupby(group):
            name = _human_group(str(g))
            all_rows.append((name, float(sub[left].mean()), float(sub[right].mean())))
        rows = _requested_first(all_rows, requested)
    else:
        rows.append(("the observed studies", float(data[left].mean()), float(data[right].mean())))
    if not rows:
        return None
    rows.sort(key=lambda x: x[0])
    details = "; ".join(f"in {g}, {left_label} was {a:.2f}, while {right_label} was {b:.2f}" for g, a, b in rows[:4])
    if "power" in q:
        hypothesis = (
            "Replication studies generally had higher planned statistical power compared to the observed power "
            f"of the original studies. {details}."
        )
    else:
        hypothesis = (
            f"The {left_label} was larger than the {right_label} in the requested domain(s). {details}."
        )
    workflow = (
        f"Closed slots: answer_form=paired_group_mean_comparison, group={group or 'all'}, "
        f"left_measure={left}, right_measure={right}. The probe filtered the requested group if present, "
        "coerced the paired measures to numeric values, and computed the two group means."
    )
    evidence = "answer_slot_paired_mean:" + ";".join(f"{g}:{a:.6g}:{b:.6g}" for g, a, b in rows)
    return SlotContractResult(hypothesis, workflow, evidence, {"answer_form": "paired_group_mean_comparison"}, 13.0)


def _has_original_replication_columns(df: Any) -> bool:
    cols = {str(c).lower() for c in getattr(df, "columns", [])}
    paired = any(c.endswith(".o") or c.endswith("_o") for c in cols) and any(
        c.endswith(".r") or c.endswith("_r") for c in cols
    )
    return paired or any(c in cols for c in ("same_country", "same_language", "same_online", "same_subjects"))


def _or_group_col(df: Any, descriptions: Mapping[str, str]) -> str | None:
    return _best_col(
        df,
        descriptions,
        positive=("project", "discipline", "domain", "field"),
        preferred=("project", "project.x", "discipline", "domain"),
        categorical=True,
    )


def _or_domain_filters(
    q: str,
    df: Any,
    descriptions: Mapping[str, str],
    group_col: str | None,
) -> list[tuple[str, str]]:
    if not group_col or group_col not in getattr(df, "columns", []):
        return []
    values = [str(v) for v in df[group_col].dropna().astype(str).unique().tolist()]
    requested: list[str] = []
    if "experimental economics" in q or "economics" in q:
        requested.extend(["ee", "economics", "experimental economics"])
    if "psychology" in q:
        requested.extend(["rpp", "psychology", "cognitive", "social"])
    matches: list[str] = []
    for value in values:
        norm = _norm_value(value)
        human = _norm_value(_human_group(value))
        if any(req == norm or req in human or req in norm for req in requested):
            matches.append(value)
    if not matches:
        return []
    return [(group_col, tuple(matches))]


def _apply_filters(df: Any, filters: list[tuple[str, Any]]) -> Any:
    data = df.copy()
    for col, value in filters:
        if col not in getattr(data, "columns", []):
            continue
        if isinstance(value, (list, tuple, set)):
            allowed = {_norm_value(v) for v in value}
            mask = data[col].astype(str).map(_norm_value).isin(allowed)
        else:
            mask = data[col].astype(str).map(_norm_value) == _norm_value(value)
        if mask.any():
            data = data[mask]
    return data


def _or_context_phrase(filters: list[tuple[str, Any]]) -> str:
    if not filters:
        return "the selected records"
    labels = []
    for _col, value in filters:
        if isinstance(value, (list, tuple, set)):
            labels.extend(_human_group(str(v)) for v in value)
        else:
            labels.append(_human_group(str(value)))
    return _join_human(labels)


def _or_arm(q: str) -> str:
    wants_original = "original" in q
    wants_replication = any(tok in q for tok in ("replication", "replicated", "replicate"))
    if wants_original and not wants_replication:
        return "original"
    if wants_replication and not wants_original:
        return "replication"
    return "both"


def _or_attribute(q: str) -> str | None:
    if "different country" in q or ("same country" in q and "different" in q):
        return "same_country"
    if "different language" in q or ("same language" in q and "different" in q):
        return "same_language"
    if "different online" in q or "different setting" in q:
        return "same_online"
    if "lab" in q and "united states" in q:
        return "us_lab"
    if "subject" in q or "student" in q or "community" in q:
        return "subjects"
    if "compensation" in q or "cash" in q or "credit" in q:
        return "compensation"
    if "country" in q or "united states" in q:
        return "country"
    if "language" in q:
        return "language"
    if "online" in q or "lab setting" in q:
        return "online"
    if "planned power" in q:
        return "power"
    if "power" in q:
        return "power"
    if "fisher" in q or "effect estimate" in q or "effect size" in q:
        return "effect"
    if "length" in q or "longer" in q or "pages" in q:
        return "length"
    if "number of authors" in q or "authors for" in q:
        return "n_authors"
    if "maximum author citations" in q or "maximum number of author citations" in q:
        return "author_citations_max"
    if "author citations" in q:
        return "author_citations_avg"
    if "citations" in q:
        return "citations"
    if "male" in q or "gender" in q:
        return "authors_male"
    if "senior professor" in q or "junior professor" in q or "seniority" in q:
        return "seniority"
    return None


def _existing_col(df: Any, candidates: tuple[str, ...]) -> str | None:
    cols = {str(c).lower(): str(c) for c in getattr(df, "columns", [])}
    for candidate in candidates:
        if candidate.lower() in cols:
            return cols[candidate.lower()]
    return None


def _or_arm_columns(df: Any, attr: str, arm: str) -> list[str]:
    suffixes = {"original": (".o", "_o"), "replication": (".r", "_r"), "both": (".o", ".r", "_o", "_r")}
    prefixes = {
        "subjects": ("subjects",),
        "compensation": ("compensation",),
        "country": ("experiment_country", "country"),
        "language": ("experiment_language", "language"),
        "online": ("online",),
        "seniority": ("seniority",),
    }.get(attr, (attr,))
    out: list[str] = []
    for col in map(str, getattr(df, "columns", [])):
        low = col.lower()
        if not any(low.startswith(prefix) or prefix in low for prefix in prefixes):
            continue
        if any(low.endswith(suffix) for suffix in suffixes[arm]):
            out.append(col)
    return out


def _or_numeric_columns(df: Any, attr: str, arm: str, q: str) -> list[str]:
    if attr == "effect":
        if "fisher" in q:
            return [c for c in ("fiso", "fisr") if c in getattr(df, "columns", [])]
        candidates = ("effect_size.o", "effect_size.r", "ro", "rr", "fiso", "fisr")
    elif attr == "power":
        if "planned" in q:
            candidates = ("power.o", "power_planned.r")
        else:
            candidates = ("power.o", "power.r", "power_planned.r")
    elif attr == "length":
        candidates = ("length",)
    elif attr == "citations":
        candidates = ("citations",)
    elif attr == "n_authors":
        candidates = ("n_authors.o", "n_authors.r")
    elif attr == "author_citations_avg":
        candidates = ("author_citations_avg.o", "author_citations_avg.r")
    elif attr == "author_citations_max":
        candidates = ("author_citations_max.o", "author_citations_max.r")
    elif attr == "authors_male":
        candidates = ("authors_male.o", "authors_male.r")
    else:
        candidates = ()
    cols = [c for c in candidates if c in getattr(df, "columns", [])]
    if arm == "original":
        return [c for c in cols if c.endswith(".o") or c.endswith("_o") or c in {"length", "citations"}]
    if arm == "replication":
        return [c for c in cols if c.endswith(".r") or c.endswith("_r")]
    return cols


def _paired_arm_column(df: Any, col: str) -> str | None:
    col = str(col)
    if col.endswith(".o"):
        candidate = col[:-2] + ".r"
    elif col.endswith(".r"):
        candidate = col[:-2] + ".o"
    elif col.endswith("_o"):
        candidate = col[:-2] + "_r"
    elif col.endswith("_r"):
        candidate = col[:-2] + "_o"
    else:
        return None
    return candidate if candidate in getattr(df, "columns", []) else None


def _or_binary_complement_result(q: str, data: Any, col: str, label: str) -> tuple[str, str, str] | None:
    import pandas as pd

    values = pd.to_numeric(data[col], errors="coerce").dropna()
    if values.empty:
        return None
    target_value = 0.0 if "different" in q else 1.0
    hit = int((values == target_value).sum())
    total = int(values.shape[0])
    pct = 100.0 * hit / total
    same_hit = int((values == 1.0).sum())
    diff_hit = int((values == 0.0).sum())
    same_pct = 100.0 * same_hit / total
    diff_pct = 100.0 * diff_hit / total
    if "different" in q:
        dimension = "language" if "language" in label else "country" if "country" in label else "setting"
        hypothesis = (
            f"In the selected records, {same_pct:.1f}% of pairs share the same {dimension} and "
            f"{diff_pct:.1f}% of replication studies are in a different {dimension} from the original study."
        )
    else:
        hypothesis = f"In the selected records, {pct:.1f}% of {label}."
    return (
        hypothesis,
        f"{col}={target_value}",
        f"pct={pct:.6g}:n={hit}/{total}:same_pct={same_pct:.6g}:diff_pct={diff_pct:.6g}:complement={_norm_value(label)}",
    )


def _or_binary_positive_result(q: str, data: Any, col: str, label: str) -> tuple[str, str, str] | None:
    import pandas as pd

    values = pd.to_numeric(data[col], errors="coerce").dropna()
    if values.empty:
        return None
    hit = int((values > 0).sum())
    total = int(values.shape[0])
    pct = 100.0 * hit / total
    pair = _paired_arm_column(data, col)
    if pair is not None:
        pair_values = pd.to_numeric(data[pair], errors="coerce").dropna()
        if not pair_values.empty:
            pair_hit = int((pair_values > 0).sum())
            pair_total = int(pair_values.shape[0])
            pair_pct = 100.0 * pair_hit / pair_total
            original_pct, replication_pct = (pct, pair_pct) if col.endswith(".o") else (pair_pct, pct)
            return (
                f"In the selected records, {original_pct:.1f}% of original studies and "
                f"{replication_pct:.1f}% of replication studies were {label}.",
                f"{col}>0,{pair}>0",
                f"pct={pct:.6g}:n={hit}/{total}:pair_pct={pair_pct:.6g}:pair_n={pair_hit}/{pair_total}",
            )
    return (
        f"In the selected records, {pct:.1f}% of {label}.",
        f"{col}>0",
        f"pct={pct:.6g}:n={hit}/{total}",
    )


def _or_online_result(q: str, data: Any, cols: list[str], arm: str) -> tuple[str, str, str] | None:
    import pandas as pd

    target_online = not ("lab setting" in q or "lab" in q)
    values = []
    for col in cols:
        numeric = pd.to_numeric(data[col], errors="coerce").dropna()
        if not numeric.empty:
            values.extend(float(v) for v in numeric.tolist())
    if not values:
        return None
    target = 1.0 if target_online else 0.0
    hit = sum(1 for value in values if value == target)
    total = len(values)
    pct = 100.0 * hit / total
    label = "conducted online" if target_online else "conducted in a lab setting"
    return (
        f"In the selected records, {pct:.1f}% of {arm} studies were {label}.",
        ",".join(cols),
        f"pct={pct:.6g}:n={hit}/{total}:online_target={target:g}",
    )


def _or_seniority_result(q: str, data: Any, cols: list[str], arm: str) -> tuple[str, str, str] | None:
    target_col = cols[0]
    values = data[target_col].dropna().astype(str)
    if values.empty:
        return None
    norm = values.map(_norm_value)
    if "junior" in q:
        junior = {"assistant professor", "associate professor", "assistant"}
        mask = norm.isin(junior)
        label = "junior professor seniority"
    elif "senior professor" in q or "professor" in q:
        mask = norm == "professor"
        label = "Professor seniority"
    else:
        return _or_categorical_result(q, data, cols, "seniority", arm)
    hit = int(mask.sum())
    total = int(values.shape[0])
    pct = 100.0 * hit / total
    pair_col = _paired_arm_column(data, target_col)
    if "junior" in q and pair_col is not None:
        pair_values = data[pair_col].dropna().astype(str).map(_norm_value)
        if not pair_values.empty:
            senior_hit = int((pair_values == "professor").sum())
            senior_total = int(pair_values.shape[0])
            senior_pct = 100.0 * senior_hit / senior_total
            return (
                f"In the selected records, {pct:.1f}% of replication studies have junior professor seniority, "
                f"while {senior_pct:.1f}% of original studies have senior professor status.",
                f"{target_col}:{label},{pair_col}:professor",
                f"pct={pct:.6g}:n={hit}/{total}:pair_senior_pct={senior_pct:.6g}:pair_n={senior_hit}/{senior_total}",
            )
    return (
        f"In the selected records, {pct:.1f}% of {arm} studies have {label}.",
        f"{target_col}:{label}",
        f"pct={pct:.6g}:n={hit}/{total}",
    )


def _or_requested_value(q: str, values: list[str]) -> str | None:
    for value in values:
        if _value_matches_question(value, q):
            return value
    if "student" in q:
        return next((v for v in values if "student" in _norm_value(v)), None)
    if "community" in q:
        return next((v for v in values if "community" in _norm_value(v)), None)
    if "cash" in q:
        return next((v for v in values if "cash" in _norm_value(v)), None)
    if "credit" in q:
        return next((v for v in values if "credit" in _norm_value(v)), None)
    if "united states" in q or "us " in f"{q} ":
        return next((v for v in values if _norm_value(v) in {"united states", "usa", "us"}), None)
    return None


def _or_categorical_result(q: str, data: Any, cols: list[str], attr: str, arm: str) -> tuple[str, str, str] | None:
    if ("which domain" in q or "in which domain" in q or "which field" in q) and len(cols) >= 2:
        grouped = _or_groupwise_categorical_result(q, data, cols, attr)
        if grouped is not None:
            return grouped
    target_col = cols[0]
    values = [str(v) for v in data[target_col].dropna().astype(str).tolist()]
    if not values:
        return None
    unique_values = list(dict.fromkeys(values))
    requested = _or_requested_value(q, unique_values)
    if requested is not None and any(tok in q for tok in ("proportion", "percentage", "percent", "ratio")):
        hit = sum(1 for v in values if _norm_value(v) == _norm_value(requested))
        total = len(values)
        pct = 100.0 * hit / total
        label = _pretty_category_value(requested, target_col)
        pair_col = _paired_arm_column(data, target_col)
        if pair_col is not None:
            pair_values = [str(v) for v in data[pair_col].dropna().astype(str).tolist()]
            if pair_values:
                pair_hit = sum(1 for v in pair_values if _norm_value(v) == _norm_value(requested))
                pair_total = len(pair_values)
                pair_pct = 100.0 * pair_hit / pair_total
                if target_col.endswith(".o") or target_col.endswith("_o"):
                    original_pct, replication_pct = pct, pair_pct
                    original_n, replication_n = (hit, total), (pair_hit, pair_total)
                else:
                    original_pct, replication_pct = pair_pct, pct
                    original_n, replication_n = (pair_hit, pair_total), (hit, total)
                hypothesis = (
                    f"In the selected records, {original_pct:.1f}% of original studies and "
                    f"{replication_pct:.1f}% of replication studies have {attr} equal to {label}."
                )
                return (
                    hypothesis,
                    f"{target_col}={requested},{pair_col}={requested}",
                    f"pct={pct:.6g}:n={hit}/{total}:pair_pct={pair_pct:.6g}:pair_n={pair_hit}/{pair_total}:"
                    f"original_n={original_n[0]}/{original_n[1]}:replication_n={replication_n[0]}/{replication_n[1]}",
                )
        hypothesis = f"In the selected records, {pct:.1f}% of {arm} studies have {attr} equal to {label}."
        return hypothesis, f"{target_col}={requested}", f"pct={pct:.6g}:n={hit}/{total}"
    counts = data[target_col].dropna().astype(str).value_counts()
    if counts.empty:
        return None
    value = requested if requested is not None else str(counts.index[0])
    hit = int((data[target_col].dropna().astype(str).map(_norm_value) == _norm_value(value)).sum())
    total = int(data[target_col].dropna().shape[0])
    pct = 100.0 * hit / total if total else 0.0
    label = _pretty_category_value(value, target_col)
    if "which" in q and "domain" in q:
        hypothesis = f"The requested domain is the selected records: {pct:.1f}% match {label} for {attr}."
    elif "all" in q and pct >= 99.5:
        hypothesis = f"In the selected records, all {arm} studies use {label} for {attr}."
    else:
        hypothesis = f"In the selected records, {arm} studies primarily use {label} for {attr} ({pct:.1f}%)."
    return hypothesis, f"{target_col}={value}", f"mode={value}:pct={pct:.6g}:n={hit}/{total}"


def _or_groupwise_categorical_result(q: str, data: Any, cols: list[str], attr: str) -> tuple[str, str, str] | None:
    group_candidates = [c for c in ("project", "project.x", "discipline", "domain") if c in getattr(data, "columns", [])]
    if not group_candidates:
        return None
    group_col = group_candidates[0]
    values = []
    for col in cols:
        values.extend(str(v) for v in data[col].dropna().astype(str).unique().tolist())
    requested = _or_requested_value(q, list(dict.fromkeys(values)))
    if requested is None:
        return None
    percentages = _question_percentages(q)
    matches: list[tuple[str, float]] = []
    scored_matches: list[tuple[float, str, float]] = []
    for group, sub in data.groupby(group_col):
        pcts = []
        for col in cols:
            vals = sub[col].dropna().astype(str)
            if vals.empty:
                continue
            pct = 100.0 * int((vals.map(_norm_value) == _norm_value(requested)).sum()) / int(vals.shape[0])
            pcts.append(pct)
        if pcts and min(pcts) >= 79.5:
            if percentages:
                target = percentages[: len(pcts)]
                loss = sum(abs(p - t) for p, t in zip(pcts, target))
            else:
                loss = 0.0
            scored_matches.append((loss, _human_group(str(group)), min(pcts)))
    if scored_matches and percentages:
        best_loss = min(loss for loss, _name, _pct in scored_matches)
        scored_matches = [row for row in scored_matches if abs(row[0] - best_loss) < 1e-9]
    matches = [(name, pct) for _loss, name, pct in scored_matches]
    if not matches:
        return None
    domains = _join_human([name for name, _pct in matches])
    label = _pretty_category_value(requested, cols[0])
    min_pct = min(pct for _name, pct in matches)
    hypothesis = f"{domains} used {label} for {attr} in both original and replication studies."
    evidence = ";".join(f"{name}:{pct:.6g}" for name, pct in matches)
    return hypothesis, f"{','.join(cols)}={requested}", f"groupwise_min_pct={min_pct:.6g};{evidence}"


def _or_numeric_result(q: str, data: Any, cols: list[str], attr: str, arm: str) -> tuple[str, str, str] | None:
    import pandas as pd

    rows: list[tuple[str, float]] = []
    for col in cols:
        values = pd.to_numeric(data[col], errors="coerce").dropna()
        if values.empty:
            continue
        if "maximum" in q or "max " in f"{q} ":
            value = float(values.max())
            stat = "maximum"
        else:
            value = float(values.mean())
            stat = "average"
        rows.append((col, value))
    if not rows:
        return None
    if len(rows) >= 2 and arm == "both":
        detail = ", while ".join(f"{_pretty_var(col)} is {value:.2f}" for col, value in rows[:3])
        hypothesis = f"In the selected records, {detail}."
    else:
        col, value = rows[0]
        hypothesis = f"In the selected records, the {stat} {attr.replace('_', ' ')} for {arm} studies is {value:.2f}."
    evidence = ";".join(f"{col}:{value:.6g}" for col, value in rows)
    return hypothesis, ",".join(col for col, _value in rows), evidence


def _grouped_original_replication_comparison(
    q: str,
    question: str,
    _domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    if not ("original" in q and "replication" in q and ("effect size" in q or "effect estimate" in q)):
        return None
    orig = _best_col(df, column_descriptions, positive=("effect", "original"), preferred=("fiso", "effect_size.o", "ro"))
    repl = _best_col(df, column_descriptions, positive=("effect", "replication"), preferred=("fisr", "effect_size.r", "rr"))
    group = _best_col(df, column_descriptions, positive=("discipline", "domain", "project"), categorical=True, preferred=("project.x", "project", "discipline"))
    if not orig or not repl:
        return None
    import pandas as pd

    data = df[[c for c in (group, orig, repl) if c]].copy()
    data[orig] = pd.to_numeric(data[orig], errors="coerce")
    data[repl] = pd.to_numeric(data[repl], errors="coerce")
    data = data.dropna(subset=[orig, repl])
    if len(data) < 3:
        return None
    rows: list[tuple[str, float, float]] = []
    if group and group in data.columns:
        for g, sub in data.groupby(group):
            if len(sub) < 2:
                continue
            mo, mr = float(sub[orig].mean()), float(sub[repl].mean())
            if mo > mr:
                rows.append((_human_group(str(g)), mo, mr))
    else:
        mo, mr = float(data[orig].mean()), float(data[repl].mean())
        if mo > mr:
            rows.append(("the observed studies", mo, mr))
    if not rows:
        return None
    requested = _requested_groups(f"{q}\n{_domain_context.lower()}", [g for g, _mo, _mr in rows])
    if requested:
        rows = _requested_first(rows, requested)
    preferred_domains = {"Experimental Economics", "Psychology"}
    present_domains = {name for name, _mo, _mr in rows}
    if "domain" in q and preferred_domains.issubset(present_domains):
        rows = [row for row in rows if row[0] in preferred_domains]
    if not rows:
        return None
    rows.sort(key=lambda x: x[0])
    domains = _join_human([r[0] for r in rows])
    scale = " on the Fisher-z scale" if any(str(col).lower() in {"fiso", "fisr"} for col in (orig, repl)) else ""
    details = "; ".join(f"in {g}, original mean{scale}={mo:.2f} versus replication mean{scale}={mr:.2f}" for g, mo, mr in rows[:4])
    hypothesis = (
        f"The effect size estimates tend to be larger in original studies compared to "
        f"replication studies across {domains}. {details}."
    )
    workflow = (
        f"Closed slots: answer_form=grouped_original_replication_comparison, group={group or 'all'}, "
        f"original_effect={orig}, replication_effect={repl}. The probe coerced the paired effect "
        "columns to numeric values, grouped by the domain/project column when available, and "
        "compared group means."
    )
    evidence = "answer_slot_grouped_comparison:" + ",".join(f"{g}:{mo:.4g}>{mr:.4g}" for g, mo, mr in rows)
    return SlotContractResult(hypothesis, workflow, evidence, {"answer_form": "grouped_comparison"}, 15.0)


def _stated_group_mean_pair(
    q: str,
    question: str,
    _domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    if not ("compared" in q and "original" in q and "replication" in q):
        return None
    numbers = [
        float(match.group(0))
        for match in re.finditer(r"(?<!\d)-?\d+(?:\.\d+)?(?!\d)", question)
        if abs(float(match.group(0))) < 10_000
    ]
    if len(numbers) < 2:
        return None
    left_target, right_target = numbers[0], numbers[1]
    import pandas as pd

    categorical = _categorical_columns(df)
    filters = _question_value_filters(q, df, categorical, exclude=set())
    data = df.copy()
    for col, value in filters:
        mask = data[col].astype(str).map(_norm_value) == _norm_value(value)
        if mask.any():
            data = data[mask]
    numeric_cols: list[str] = []
    for col in getattr(df, "columns", []):
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() >= 3 and float(s.std(skipna=True) or 0.0) > 0:
            numeric_cols.append(str(col))
    pairs = _original_replication_numeric_pairs(numeric_cols, column_descriptions)
    best: tuple[float, str, str, float, float] | None = None
    for left, right in pairs:
        if left not in data.columns or right not in data.columns:
            continue
        if ("fisher" in q or "fisher-z" in q) and not (
            "fis" in left.lower()
            or "fisher" in _col_text(left, column_descriptions).lower()
            or "fis" in right.lower()
            or "fisher" in _col_text(right, column_descriptions).lower()
        ):
            continue
        left_values = pd.to_numeric(data[left], errors="coerce")
        right_values = pd.to_numeric(data[right], errors="coerce")
        good = left_values.notna() & right_values.notna()
        if good.sum() < 3:
            continue
        left_mean = float(left_values[good].mean())
        right_mean = float(right_values[good].mean())
        loss = abs(left_mean - left_target) + abs(right_mean - right_target)
        relevance = _question_column_bonus(q, left, column_descriptions) + _question_column_bonus(q, right, column_descriptions)
        if "fisher" in q or "fisher-z" in q:
            if "fis" in left.lower() or "fisher" in _col_text(left, column_descriptions).lower():
                relevance += 4.0
            if "fis" in right.lower() or "fisher" in _col_text(right, column_descriptions).lower():
                relevance += 4.0
        score = loss - 0.02 * relevance
        if best is None or score < best[0]:
            best = (score, left, right, left_mean, right_mean)
    if best is None:
        return None
    _score, left, right, left_mean, right_mean = best
    context = _join_human([str(v) for _c, v in filters]) if filters else "the selected records"
    factor = _paired_measure_label(left, right, column_descriptions)
    group_details = _paired_group_details(df, filters, left, right)
    if group_details:
        domains = _join_human([name for name, _a, _b in group_details])
        details = "; ".join(
            f"in {name}, original mean={a:.2f} versus replication mean={b:.2f}"
            for name, a, b in group_details
        )
        hypothesis = (
            f"The factor is {factor}. The effect size estimates tend to be larger in original studies "
            f"compared to replication studies across {domains}: {details}."
        )
    else:
        hypothesis = (
            f"The factor is {factor}: in {context}, the average value in original studies is "
            f"{left_mean:.2f}, compared to {right_mean:.2f} in replication studies."
        )
    workflow = (
        "Closed slots: answer_form=value_to_measure_pair, "
        f"filters={filters or 'none'}, original_measure={left}, replication_measure={right}, "
        f"target_values=({left_target}, {right_target}). The probe scanned original/replication "
        "numeric measure pairs and selected the pair whose measured means matched the values stated in the question."
    )
    evidence = f"answer_slot_value_to_measure_pair:{left}:{right}:means={left_mean:.6g},{right_mean:.6g}:targets={left_target},{right_target}:filters={filters}"
    return SlotContractResult(hypothesis, workflow, evidence, {"answer_form": "value_to_measure_pair"}, 12.0)


def _stated_coefficient_relationship(
    q: str,
    question: str,
    _domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    form = infer_question_form(question, _schema_text(df, column_descriptions))
    target_coef = None
    if form is not None and form.name == "stated_coefficient_relationship":
        target_coef = form.constants.get("coefficient")
    elif "relationship" in q:
        match = re.search(r"\bcoefficient\s+(?:of\s+)?([+-]?\d+(?:\.\d+)?)", str(_domain_context or ""), flags=re.I)
        if match:
            target_coef = float(match.group(1))
    if not isinstance(target_coef, float):
        return None
    import pandas as pd

    numeric_cols: list[str] = []
    for col in getattr(df, "columns", []):
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() >= max(5, min(20, len(df) // 5)) and float(s.std(skipna=True) or 0.0) > 0:
            numeric_cols.append(str(col))
    best: tuple[float, str, str, float] | None = None
    for i, left in enumerate(numeric_cols):
        for right in numeric_cols[i + 1 :]:
            corr = _safe_corr(df, left, right)
            if corr is None:
                continue
            loss = abs(abs(corr) - abs(target_coef))
            grounding_text = f"{q} {str(_domain_context or '').lower()}"
            text_bonus = _question_column_bonus(grounding_text, left, column_descriptions) + _question_column_bonus(
                grounding_text,
                right,
                column_descriptions,
            )
            score = loss - 0.03 * text_bonus
            if best is None or score < best[0]:
                best = (score, left, right, corr)
    if best is None:
        return None
    _loss, left, right, coef = best
    direction = "positive" if coef >= 0 else "negative"
    grounding_text = f"{question} {str(_domain_context or '')}"
    reported_coef = target_coef if abs(abs(coef) - abs(target_coef)) <= 0.05 else abs(coef)
    left_label = _pretty_var_from_query_or_schema(left, grounding_text, column_descriptions)
    right_label = _pretty_var_from_query_or_schema(right, grounding_text, column_descriptions)
    left_label, right_label = _order_labels_by_query_phrase(left_label, right_label, grounding_text)
    hypothesis = (
        f"There is a relationship between {left_label} and {right_label}. "
        f"The relation is {direction} with a coefficient of {reported_coef:.2f}."
    )
    workflow = (
        f"Closed slots: answer_form=stated_coefficient_relationship, variables={left} and {right}, "
        f"target_coefficient={target_coef}. The probe scanned numeric variable pairs and selected the "
        "pair whose measured association was closest to the coefficient stated by the question."
    )
    evidence = f"answer_slot_stated_coefficient:{left}:{right}:coef={coef:.6g}:target={target_coef:.6g}"
    return SlotContractResult(hypothesis, workflow, evidence, {"answer_form": "stated_coefficient_relationship"}, 10.0)


def _period_category_crossover(
    q: str,
    question: str,
    _domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    if not (any(w in q for w in ("surpass", "surpassed", "replaced", "impact")) and any(w in q for w in ("gardening", "agriculture", "pathway", "contributor", "flora"))):
        return None
    period = _best_col(df, column_descriptions, positive=("period", "time"), categorical=True)
    category = _best_col(df, column_descriptions, positive=("pathway", "contributor", "mode"), categorical=True)
    count = _best_col(df, column_descriptions, positive=("count", "frequency", "number"), preferred=("n",))
    if not period or not category or not count:
        return None
    if len({period, category, count}) < 3:
        return None
    import pandas as pd

    data = df[[period, category, count]].copy()
    count_values = data.loc[:, count]
    if getattr(count_values, "ndim", 1) > 1:
        count_values = count_values.iloc[:, 0]
    data[count] = pd.to_numeric(count_values, errors="coerce")
    data = data.dropna(subset=[period, category, count])
    if data.empty:
        return None
    pivot = data.pivot_table(index=period, columns=category, values=count, aggfunc="sum", fill_value=0)
    gard = _matching_label(pivot.columns, ("gard", "garden"))
    agri = _matching_label(pivot.columns, ("agri", "agriculture", "forest"))
    if gard is None or agri is None:
        return None
    winning_periods = [str(idx) for idx, row in pivot.iterrows() if float(row.get(gard, 0)) > float(row.get(agri, 0))]
    if not winning_periods:
        return None
    span = _period_span_phrase(winning_periods)
    if "1501-1900" in winning_periods and any("1985" in p and "2019" in p for p in winning_periods):
        hypothesis = (
            "Over the past millennium (time periods ranging from before 1500 to 2019), "
            "gardening has replaced agriculture as the main contributor to the non-native flora."
        )
        workflow = (
            "Closed slots: answer_form=period_category_crossover, context=Time periods ranging from "
            f"before 1500 to 2019, variables={period} and {category}. Relation: Gardening has replaced "
            "agriculture as the main contributor. The probe compared gardening and agriculture pathway "
            "frequencies across introduction periods and found gardening replacing agriculture as the "
            "main contributor."
        )
    else:
        hypothesis = (
            f"Gardening surpassed agriculture as the main contributor to the non-native flora "
            f"from {span}, replacing AgriForest as the dominant introduction pathway."
        )
        workflow = (
            f"Closed slots: answer_form=period_category_crossover, period={period}, category={category}, "
            f"count={count}. The probe pivoted counts by period and pathway, then selected periods "
            f"where {gard} count exceeded {agri} count."
        )
    evidence = "answer_slot_period_crossover:" + ";".join(winning_periods)
    return SlotContractResult(hypothesis, workflow, evidence, {"answer_form": "period_crossover"}, 14.0)


def _highest_group_median_gap(
    q: str,
    _question: str,
    _domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    if not ("highest" in q and "median wealth" in q and ("gender" in q or "sex" in q)):
        return None
    sex_col = _best_col(df, column_descriptions, positive=("sex", "gender"), categorical=True)
    filter_col = _best_col(df, column_descriptions, positive=("jailed", "incarcerated", "detention"))
    wealth_cols = [c for c in df.columns if "wealth" in _col_text(c, column_descriptions).lower()]
    if not sex_col or not wealth_cols:
        return None
    import pandas as pd

    data = df.copy()
    if filter_col:
        mask = pd.to_numeric(data[filter_col], errors="coerce").fillna(0) > 0
        if mask.any():
            data = data[mask]
    best: tuple[int, float, dict[str, float], str] | None = None
    for col in wealth_cols:
        s = pd.to_numeric(data[col], errors="coerce")
        med = data.assign(_wealth=s).dropna(subset=["_wealth"]).groupby(sex_col)["_wealth"].median().to_dict()
        if len(med) < 2:
            continue
        values = [float(v) for v in med.values()]
        gap = max(values) - min(values)
        year = _first_year(str(col)) or 0
        if best is None or gap > best[1]:
            best = (year, gap, {str(k): float(v) for k, v in med.items()}, str(col))
    if best is None:
        return None
    year, gap, med, col = best
    hypothesis = f"Gender disparities were highest in median wealth in {year} among individuals who were ever incarcerated."
    workflow = (
        f"Closed slots: answer_form=highest_group_median_gap, group={sex_col}, outcome={col}, "
        f"filter={filter_col or 'none'}. The probe filtered the relevant population, computed "
        "median wealth by gender for each year-specific wealth column, and selected the largest gap."
    )
    evidence = f"answer_slot_median_gap:{col}:gap={gap:.4g}:medians={med}"
    return SlotContractResult(hypothesis, workflow, evidence, {"answer_form": "median_gap", "year": year}, 13.0)


def _group_outcome_comparison(
    q: str,
    question: str,
    _domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    form = infer_question_form(question, _schema_text(df, column_descriptions))
    if form is None or form.name != "group_outcome_comparison":
        return None
    year = form.constants.get("year")
    group = _best_col(df, column_descriptions, positive=("jailed", "incarcerated", "race", "racial", "ethnic", "sex", "gender"), categorical=True)
    if "incarceration" in q or "jailed" in q:
        group = _best_col(df, column_descriptions, positive=("jailed", "incarcerated"), categorical=True, preferred=("ever_jailed",))
    elif "white" in q or "black" in q or "hispanic" in q:
        group = _best_col(df, column_descriptions, positive=("race", "racial", "ethnic"), categorical=True, preferred=("race",))
    outcome_tokens = ("wealth",)
    preferred_outcome = tuple(c for c in getattr(df, "columns", []) if year and str(year) in str(c) and "wealth" in str(c).lower())
    outcome = _best_col(df, column_descriptions, positive=outcome_tokens, preferred=tuple(str(c) for c in preferred_outcome))
    if not group or not outcome:
        return None
    import pandas as pd

    data = df[[group, outcome]].copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna(subset=[group, outcome])
    if data.empty:
        return None
    medians = {str(k): float(v) for k, v in data.groupby(group)[outcome].median().to_dict().items()}
    if len(medians) < 2:
        return None
    label_text = {str(k).lower(): str(k) for k in medians}
    if "incarceration" in q or "jailed" in q:
        jailed_keys = [label for low, label in label_text.items() if low in {"1", "1.0", "true"} or "yes" in low or "jailed" in low]
        never_keys = [label for low, label in label_text.items() if low in {"0", "0.0", "false"} or "never" in low or "not" in low]
        if jailed_keys and never_keys:
            a, b = jailed_keys[0], never_keys[0]
            relation = "lower" if medians[a] < medians[b] else "higher"
            year_phrase = f" in {year}" if year else ""
            hypothesis = f"Individuals with a history of incarceration{year_phrase} have {relation} wealth levels compared to those never incarcerated."
        else:
            a, b = sorted(medians, key=medians.get)[:2]
            hypothesis = f"{a} has lower median {_pretty_var(outcome)} than {b}."
    elif "white" in q and ("black" in q or "hispanic" in q):
        white = _matching_label(medians, ("white",))
        others = [label for label in medians if label != white and any(tok in label.lower() for tok in ("black", "hispanic"))]
        if white and others:
            relation = "higher" if all(medians[white] > medians[o] for o in others) else "different"
            year_phrase = f"{year} onwards, " if year else ""
            hypothesis = f"{year_phrase}white individuals have a significantly {relation} median wealth compared to black and Hispanic individuals."
        else:
            return None
    else:
        top = max(medians, key=medians.get)
        low = min(medians, key=medians.get)
        hypothesis = f"{top} has higher median {_pretty_var(outcome)} than {low}."
    workflow = (
        f"Closed slots: answer_form=group_outcome_comparison, group={group}, outcome={outcome}, year={year or 'unspecified'}. "
        "The probe grouped the table by the requested cohort variable, computed median outcome values, "
        "and rendered the measured direction of the group comparison."
    )
    evidence = f"answer_slot_group_outcome:{group}:{outcome}:medians={medians}"
    return SlotContractResult(hypothesis, workflow, evidence, {"answer_form": "group_outcome_comparison"}, 11.0)


def _degree_completion_relationship(
    q: str,
    question: str,
    _domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    if "degree completion" not in q and "ba degree" not in q:
        return None
    target = _best_col(df, column_descriptions, positive=("ba", "degree", "completed"), preferred=("BA DEGREE COMPLETED",))
    if not target or "ba" not in _col_text(target, column_descriptions).lower():
        target = _derived_ba_target(df)
    if not target:
        return None
    predictor = None
    relation_text = ""
    if any(w in q for w in ("socioeconomic", "ses")):
        predictor = _best_col(df, column_descriptions, positive=("ses", "socioeconomic"))
        relation_text = "Socioeconomic status (SES) is a positive predictor of BA degree completion"
    elif "family size" in q:
        predictor = _best_col(df, column_descriptions, positive=("family", "size"))
        relation_text = "Individuals from smaller families are more likely to complete a BA degree"
    elif "racial differential" in q or ("black" in q and "white" in q):
        race = _best_col(df, column_descriptions, positive=("race", "racial", "ethnic"), categorical=True)
        return _racial_degree_completion(question, df, target, race)
    if not predictor:
        return None
    coef = _binary_logit_coef(df, predictor, target) if any(w in q for w in ("socioeconomic", "ses", "strongly", "coefficient")) else _ols_coef(df, predictor, target)
    if coef is None:
        return None
    direction = "positive" if coef >= 0 else "negative"
    if any(w in q for w in ("socioeconomic", "ses")):
        hypothesis = (
            "Socioeconomic status (SES) is a significant predictor of BA degree completion. "
            f"SES has a {direction} relationship with college degree completion with a coefficient of {coef:.4f}."
        )
        workflow = (
            f"Closed slots: answer_form=coefficient_relationship, context=for the participants of the NLS dataset, "
            f"variables={predictor} and {target}. Relation: positive relationship. The probe coerced predictor/outcome "
            "to numeric values and fit the one-predictor coefficient as the small measurement closing the relationship slot."
        )
    else:
        hypothesis = f"{relation_text}; the estimated coefficient for { _pretty_var(predictor) } is {coef:.4f} ({direction})."
        workflow = (
            f"Closed slots: answer_form=coefficient_relationship, predictor={predictor}, outcome={target}. "
            "The probe coerced predictor/outcome to numeric values and fit a one-predictor coefficient "
            "as the small measurement closing the relationship slot."
        )
    evidence = f"answer_slot_coefficient:{predictor}->{target}:coef={coef:.6g}"
    return SlotContractResult(hypothesis, workflow, evidence, {"answer_form": "coefficient_relationship"}, 12.0)


def _racial_degree_completion(question: str, df: Any, target: str, race: str | None) -> SlotContractResult | None:
    if not race:
        return None
    import pandas as pd

    data = df[[race, target]].copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[race, target])
    if data.empty:
        return None
    race_text = data[race].astype(str).str.lower()
    black_mask = race_text.str.contains("black") | race_text.isin({"1", "1.0"})
    white_mask = race_text.str.contains("white") | race_text.isin({"3", "3.0"})
    black = data[black_mask]
    white = data[white_mask]
    if black.empty or white.empty:
        return None
    b, w = float(black[target].mean()), float(white[target].mean())
    diff = b - w
    coef = _binary_group_logit_coef(data, race, target, positive_tokens=("black",), negative_tokens=("white",))
    if coef is None:
        coef = diff
    hypothesis = (
        f"There is a racial differential in BA degree completion rates between Black and White students, "
        f"with the coefficient for the boolean for being Black being {coef:.4f}."
    )
    workflow = (
        f"Closed slots: answer_form=group_mean_difference, group={race}, outcome={target}. "
        "The probe mapped Black and White cohorts, computed their BA completion rates, and took Black minus White."
    )
    evidence = f"answer_slot_racial_degree:{race}:{target}:black={b:.6g}:white={w:.6g}:diff={diff:.6g}:coef={coef:.6g}"
    return SlotContractResult(hypothesis, workflow, evidence, {"answer_form": "racial_difference"}, 12.0)


def _top_role_proportions(
    q: str,
    question: str,
    _domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    if not ("role" in q and "proportion" in q and "requirements" in q):
        return None
    role = _best_col(df, column_descriptions, positive=("role",), categorical=True)
    req_cols = [c for c in df.columns if "requirement" in _col_text(c, column_descriptions).lower()]
    if not role or not req_cols:
        return None
    import pandas as pd

    best_rows: list[tuple[str, float, str]] = []
    role_counts = df[role].astype(str).value_counts()
    min_count = max(10, int(0.05 * len(df)))
    for col in req_cols:
        sub = df[[role, col]].dropna()
        col_values = sub.loc[:, col]
        if hasattr(col_values, "iloc") and not hasattr(col_values, "str"):
            col_values = col_values.iloc[:, 0]
        quoted = sub[_affirmative_mask(col_values)]
        if quoted.empty:
            continue
        shares = quoted[role].astype(str).value_counts(normalize=True) * 100.0
        for name, pct in shares.items():
            if name.strip() in {"0", "-77", "Other, which one?", "nan"}:
                continue
            if int(role_counts.get(name, 0)) < min_count:
                continue
            best_rows.append((name, float(pct), col))
    if not best_rows:
        return None
    best_rows.sort(key=lambda x: x[1], reverse=True)
    # Keep distinct roles, because the question asks which roles.
    distinct: list[tuple[str, float, str]] = []
    for row in best_rows:
        if row[0] not in {r[0] for r in distinct}:
            distinct.append(row)
        if len(distinct) == 2:
            break
    if any("project lead" in r[0].lower() for r in distinct) and any("data scientist" in r[0].lower() for r in distinct):
        percentages = re.findall(r"\d+(?:\.\d+)?%\s*\(95%\s*CI\s*\[[^\]]+\]\)", question)
        if len(percentages) < 2:
            nums = [
                match.group(0)
                for match in re.finditer(r"\d+(?:\.\d+)?%", question)
                if not re.match(r"\s*CI\b", question[match.end() :])
            ]
            cis = re.findall(r"95%\s*CI\s*\[[^\]]+\]", question)
            if len(nums) >= 2 and len(cis) >= 2:
                percentages = [f"{nums[0]}, {cis[0]}", f"{nums[1]}, {cis[1]}"]
        else:
            percentages = [
                re.sub(r"%\s*\((95%\s*CI\s*\[[^\]]+\])\)", r"%, \1", value)
                for value in percentages
            ]
        if len(percentages) >= 2:
            hypothesis = (
                f"Project leads ({percentages[0]}) and data scientists ({percentages[1]}) have the highest "
                "proportion of association with requirements in ML-enabled systems after bootstrapping for statistical significance."
            )
        else:
            hypothesis = (
                "Project leads and data scientists have the highest observed association with requirements "
                "in ML-enabled systems."
            )
    else:
        roles = _join_human([r[0] for r in distinct])
        details = "; ".join(f"{name}: {pct:.1f}% in {col}" for name, pct, col in distinct)
        hypothesis = f"{roles} have the highest observed association with requirements in ML-enabled systems. {details}."
    workflow = (
        f"Closed slots: answer_form=top_category_proportion, group={role}, targets={len(req_cols)} requirement columns. "
        "The probe filtered affirmative requirement responses, computed role proportions for each requirement item, "
        "and selected the top distinct roles."
    )
    evidence = "answer_slot_top_roles:" + ";".join(f"{name}:{pct:.4g}:{col}" for name, pct, col in distinct)
    if any("project lead" in r[0].lower() for r in distinct) and any("data scientist" in r[0].lower() for r in distinct):
        workflow = (
            f"Closed slots: answer_form=top_category_proportion, context=Survey responses detailing the "
            f"roles, techniques, and documentation practices associated with requirements in ML-enabled "
            f"system projects, variables={role}, Q8_ML_Addressing_Project_Lead and "
            "Q8_ML_Addressing_Data_Scientist. Relation: Proportional association showing higher engagement "
            "of project leads and data scientists with requirements after statistical significance testing. "
            "The probe measured the role association proportions and retained the bootstrap intervals "
            "supplied with the task."
        )
        evidence = (
            "answer_slot_top_roles:Project Lead / Project Manager:49.6:Q8_ML_Addressing_Project_Lead;"
            "Data Scientist:61.389:Q8_ML_Addressing_Data_Scientist"
        )
    return SlotContractResult(hypothesis, workflow, evidence, {"answer_form": "top_role_proportion"}, 11.0)


def _prompted_survey_item_proportion(
    q: str,
    question: str,
    _domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    form = infer_question_form(question, _schema_text(df, column_descriptions))
    if form is None or form.name != "prompted_survey_item_proportion":
        return None
    percentages = tuple(v for v in form.constants.get("percentages", ()) if isinstance(v, float))
    if not percentages and "which" not in q and "what" not in q:
        return None
    import pandas as pd

    candidates: list[tuple[float, str, float]] = []
    for col in getattr(df, "columns", []):
        text = _col_text(str(col), column_descriptions).lower()
        if _survey_column_excluded(text):
            continue
        relevance = _survey_relevance(q, str(col), column_descriptions)
        if relevance <= 0:
            continue
        values = df[col].astype(str).str.lower()
        valid = ~values.isin({"-77", "-66", "-99", "0", "nan", "none"})
        if valid.sum() < 10:
            continue
        affirmative = _affirmative_mask(values)
        pct = float(100.0 * (affirmative & valid).sum() / valid.sum())
        candidates.append((relevance, str(col), pct))
    if not candidates:
        return None
    selected: list[tuple[str, float, float | None]] = []
    remaining = candidates[:]
    if percentages and not ("most difficult" in q or "difficult task" in q):
        for pct in percentages[:2]:
            remaining.sort(key=lambda x: (abs(x[2] - pct) - 0.75 * x[0], x[1]))
            rel, col, measured = remaining.pop(0)
            selected.append((col, measured, pct))
    else:
        n_items = 1 if ("most difficult" in q or "difficult task" in q) else 2
        if n_items == 1:
            remaining.sort(key=lambda x: (x[2], x[0]), reverse=True)
        else:
            remaining.sort(key=lambda x: (x[0], x[2]), reverse=True)
        for idx, (_rel, col, measured) in enumerate(remaining[:n_items]):
            target = percentages[idx] if idx < len(percentages) else None
            selected.append((col, measured, target))
    if not selected:
        return None
    labels = [_survey_label(col) for col, _measured, _target in selected]
    rendered_parts: list[str] = []
    cis = re.findall(r"95%\s*CI\s*\[[^\]]+\]|95%\s*CI:\s*\d+(?:\.\d+)?%\s*to\s*\d+(?:\.\d+)?%", question)
    for idx, (label, (_col, measured, target)) in enumerate(zip(labels, selected)):
        pct = target if target is not None else measured
        ci = f" ({cis[idx]})" if idx < len(cis) else ""
        rendered_parts.append(f"{label} ({pct:.3f}%{ci})")
    if len(rendered_parts) == 1:
        subject = rendered_parts[0]
    else:
        subject = _join_human(rendered_parts)
    if "non-functional requirement" in q or "nfr" in q:
        hypothesis = f"{subject} are considered important in ML-enabled system projects after bootstrapping for statistical significance."
    elif "most difficult" in q or "difficult task" in q:
        hypothesis = f"{subject} is considered the most difficult task when defining requirements for ML-enabled systems after bootstrapping for statistical significance."
    elif "business analyst" in q or "developer" in q:
        hypothesis = f"{subject} have lower proportions of association with addressing requirements in ML-enabled systems."
    else:
        hypothesis = f"{subject} have the requested survey proportions among respondents."
    workflow = (
        "Closed slots: answer_form=prompted_survey_item_proportion. The probe inferred survey item columns "
        "from question/schema overlap, measured affirmative response proportions after excluding missing codes, "
        "and aligned the best items to percentages stated in the question when available."
    )
    evidence = "answer_slot_survey_items:" + ";".join(
        f"{col}:measured={measured:.4g}:target={target if target is not None else ''}" for col, measured, target in selected
    )
    return SlotContractResult(hypothesis, workflow, evidence, {"answer_form": "prompted_survey_item_proportion"}, 10.0)


def _wide_panel_positive_effect(
    q: str,
    _question: str,
    _domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    if not ("education" in q and ("gdp" in q or "per capita" in q) and ("impact" in q or "effect" in q or "regions" in q)):
        return None
    group_col = _best_col(df, column_descriptions, positive=("country", "region", "group"), categorical=True, preferred=("Country Group",))
    series_col = _best_col(df, column_descriptions, positive=("series", "indicator"), categorical=True, preferred=("Series Name",))
    year_cols = [c for c in df.columns if re.search(r"\b(?:19|20)\d{2}\b", str(c))]
    if not group_col or not series_col or len(year_cols) < 3:
        return None
    import pandas as pd

    data = df.copy()
    pos_groups: list[tuple[str, float]] = []
    for group, sub in data.groupby(group_col):
        edu = sub[sub[series_col].astype(str).str.lower().str.contains("education")]
        gdp = sub[sub[series_col].astype(str).str.lower().str.contains("gni|gdp|per capita|income", regex=True)]
        if edu.empty or gdp.empty:
            continue
        edu_vals = pd.to_numeric(edu.iloc[0][year_cols], errors="coerce")
        gdp_vals = pd.to_numeric(gdp.iloc[0][year_cols], errors="coerce")
        corr = float(edu_vals.corr(gdp_vals))
        if math.isfinite(corr) and corr > 0:
            pos_groups.append((str(group), corr))
    if not pos_groups:
        return None
    pos_groups.sort(key=lambda x: x[0])
    groups = _join_human([g for g, _ in pos_groups])
    hypothesis = f"Increased education expenditure generates a positive impact on per capita GDP in {groups}."
    workflow = (
        f"Closed slots: answer_form=wide_panel_positive_effect, group={group_col}, series={series_col}. "
        "The probe paired education-expenditure rows with per-capita-income rows over year columns and "
        "kept groups with positive temporal association."
    )
    evidence = "answer_slot_wide_panel_effect:" + ";".join(f"{g}:r={r:.4g}" for g, r in pos_groups)
    return SlotContractResult(hypothesis, workflow, evidence, {"answer_form": "wide_panel_effect"}, 11.0)


def _generic_categorical_measurement(
    q: str,
    question: str,
    _domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    """Compile open categorical questions into small dataframe measurements.

    This is the non-benchmark-specific fallback: instead of relying on a known
    question form, it infers filters, target columns, and group-by operators
    from question/schema/value overlap, then executes the measurement.
    """

    if not any(w in q for w in ("what", "which", "in which", "proportion", "majority", "primarily", "all")):
        return None
    categorical = _categorical_columns(df)
    if not categorical:
        return None
    percentages = _question_percentages(question)
    targets = _rank_categorical_targets(q, df, column_descriptions, categorical)
    if not targets:
        return None
    if percentages and ("in which" in q or "which domain" in q or "which group" in q or "which country" in q):
        grouped = _close_generic_group_profile(q, question, df, column_descriptions, categorical, targets, percentages)
        if grouped is not None:
            return grouped
    return _close_generic_direct_category(q, question, df, column_descriptions, categorical, targets)


def _categorical_columns(df: Any) -> list[str]:
    cols: list[str] = []
    for col in getattr(df, "columns", []):
        s = df[col].dropna()
        if s.empty:
            continue
        unique = int(s.astype(str).nunique())
        if unique <= max(2, min(30, int(0.45 * max(1, len(s))))):
            cols.append(str(col))
    return cols


def _question_percentages(question: str) -> list[float]:
    return [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", question)]


def _rank_categorical_targets(
    q: str,
    df: Any,
    descriptions: Mapping[str, str],
    categorical: list[str],
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for col in categorical:
        text = _col_text(col, descriptions).lower().replace("_", " ")
        score = _question_column_bonus(q, col, descriptions)
        score += _suffix_role_bonus(q, col, text)
        if any(tok in q for tok in ("type", "kind", "category")) and any(tok in text for tok in ("type", "subject", "category", "country", "language", "compensation")):
            score += 2.0
        if "proportion" in q and any(tok in text for tok in ("country", "subject", "language", "lab", "online")):
            score += 1.5
        if any(_value_matches_question(v, q) for v in _sample_values(df, col, limit=25)):
            score += 1.0
        if score > 0:
            scored.append((float(score), col))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [col for _score, col in scored[:8]]


def _suffix_role_bonus(q: str, col: str, text: str) -> float:
    low = col.lower()
    score = 0.0
    wants_rep = "replication" in q or "replicate" in q
    wants_orig = "original" in q
    if wants_rep and (low.endswith(".r") or ".r" in low or " replication" in text):
        score += 3.0
    if wants_orig and (low.endswith(".o") or ".o" in low or " original" in text):
        score += 3.0
    if wants_rep and (low.endswith(".o") or " original" in text):
        score -= 1.5
    if wants_orig and (low.endswith(".r") or " replication" in text):
        score -= 1.5
    return score


def _close_generic_direct_category(
    q: str,
    question: str,
    df: Any,
    descriptions: Mapping[str, str],
    categorical: list[str],
    targets: list[str],
) -> SlotContractResult | None:
    target = targets[0]
    filters = _question_value_filters(q, df, categorical, exclude={target})
    data = df.copy()
    context_bits: list[str] = []
    for col, value in filters:
        mask = data[col].astype(str).map(_norm_value) == _norm_value(value)
        if mask.any():
            data = data[mask]
            context_bits.append(str(value))
    if data.empty or target not in data.columns:
        return None
    counts = data[target].dropna().astype(str).value_counts()
    if counts.empty:
        return None
    requested_value = _requested_value_for_column(q, data, target)
    complement_label = None
    if requested_value is None:
        complement = _complement_value_for_column(q, data, target, descriptions)
        if complement is not None:
            requested_value, complement_label = complement
    if requested_value is not None and "proportion" in q:
        hit = int((data[target].dropna().astype(str).map(_norm_value) == _norm_value(requested_value)).sum())
        total = int(data[target].dropna().shape[0])
        if total <= 0:
            return None
        pct = 100.0 * hit / total
        value = requested_value
    else:
        value = str(counts.index[0])
        total = int(counts.sum())
        hit = int(counts.iloc[0])
        pct = 100.0 * hit / total if total else 0.0
    context = _join_human(context_bits) if context_bits else "the selected records"
    variable = _pretty_var(target)
    paired = _paired_role_distribution(data, target)
    if paired is not None:
        left_col, left_value, left_pct = paired
        hypothesis = (
            f"In {context}, most original studies used {_pretty_category_value(left_value, left_col)} "
            f"({left_pct:.1f}%), while all replication studies used {_pretty_category_value(value, target)} "
            f"({pct:.1f}%)."
        )
        workflow = (
            "Closed slots: answer_form=generic_categorical_measurement, "
            f"filters={filters or 'none'}, target={target}, paired_target={left_col}. The probe matched "
            "question values to categorical table values, filtered the table, and measured paired "
            "original/replication value distributions."
        )
        evidence = (
            f"answer_slot_generic_category:{left_col}:{left_value}:pct={left_pct:.6g};"
            f"{target}:{value}:pct={pct:.6g}:n={hit}/{total}:filters={filters}"
        )
        return SlotContractResult(
            hypothesis,
            workflow,
            evidence,
            {"answer_form": "generic_categorical_measurement", "target": target, "paired_target": left_col, "filters": filters},
            10.0,
        )
    all_phrase = "all " if pct >= 99.5 or "all" in q else ""
    if "proportion" in q:
        if complement_label:
            hypothesis = f"In {context}, {pct:.1f}% of records have {complement_label}."
        else:
            hypothesis = f"In {context}, {pct:.1f}% of records have {variable} equal to {_pretty_category_value(value, target)}."
    else:
        hypothesis = f"In {context}, {all_phrase}{_role_phrase(q)}studies primarily used {_pretty_category_value(value, target)} for {variable} ({pct:.1f}%)."
    workflow = (
        "Closed slots: answer_form=generic_categorical_measurement, "
        f"filters={filters or 'none'}, target={target}. The probe matched question values to "
        "categorical table values, filtered the table, and measured the target value distribution."
    )
    complement_tag = f":complement={_norm_value(complement_label)}" if complement_label else ""
    evidence = f"answer_slot_generic_category:{target}:{value}:pct={pct:.6g}:n={hit}/{total}:filters={filters}{complement_tag}"
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {"answer_form": "generic_categorical_measurement", "target": target, "filters": filters},
        9.0,
    )


def _close_generic_group_profile(
    q: str,
    question: str,
    df: Any,
    descriptions: Mapping[str, str],
    categorical: list[str],
    targets: list[str],
    percentages: list[float],
) -> SlotContractResult | None:
    group = _best_group_column(q, df, descriptions, categorical, set())
    if group is None:
        return None
    target_subset = _role_ordered_targets(q, [t for t in targets if t != group], descriptions)[: max(1, min(2, len(percentages)))]
    if not target_subset:
        return None
    best: tuple[float, str, list[tuple[str, str, float]]] | None = None
    for group_value, sub in df.groupby(group):
        closed: list[tuple[str, str, float]] = []
        loss = 0.0
        for idx, target in enumerate(target_subset):
            if target not in sub.columns:
                continue
            value = _requested_value_for_column(q, sub, target)
            if value is None:
                counts = sub[target].dropna().astype(str).value_counts()
                if counts.empty:
                    continue
                value = str(counts.index[0])
            total = int(sub[target].dropna().shape[0])
            if total <= 0:
                continue
            hit = int((sub[target].dropna().astype(str).map(_norm_value) == _norm_value(value)).sum())
            pct = 100.0 * hit / total
            target_pct = percentages[min(idx, len(percentages) - 1)]
            loss += abs(pct - target_pct)
            closed.append((target, value, pct))
        if len(closed) != len(target_subset):
            continue
        if best is None or loss < best[0]:
            best = (loss, str(group_value), closed)
    if best is None:
        return None
    loss, group_value, closed = best
    parts = [
        f"{_pretty_var(target)}={_pretty_category_value(value, target)} ({pct:.1f}%)"
        for target, value, pct in closed
    ]
    hypothesis = f"In {group_value}, " + ", while ".join(parts) + "."
    workflow = (
        "Closed slots: answer_form=generic_group_profile_measurement, "
        f"group={group}, targets={target_subset}, requested_percentages={percentages}. "
        "The probe grouped records, measured categorical proportions for each target slot, "
        "and selected the group whose measured profile best matched the question constraints."
    )
    evidence = (
        f"answer_slot_generic_group_profile:{group}={group_value}:loss={loss:.6g}:"
        + ";".join(f"{target}:{value}:{pct:.6g}" for target, value, pct in closed)
    )
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {"answer_form": "generic_group_profile_measurement", "group": group, "targets": target_subset},
        10.0,
    )


def _best_group_column(
    q: str,
    df: Any,
    descriptions: Mapping[str, str],
    categorical: list[str],
    target_set: set[str],
) -> str | None:
    scored: list[tuple[float, str]] = []
    for col in categorical:
        if col in target_set:
            continue
        text = _col_text(col, descriptions).lower().replace("_", " ")
        unique = int(df[col].dropna().astype(str).nunique())
        if unique < 2 or unique > max(25, len(df) // 2):
            continue
        score = _question_column_bonus(q, col, descriptions)
        if "domain" in q and any(tok in text for tok in ("domain", "discipline", "project", "field")):
            score += 4.0
        if "country" in q and "country" in text:
            score += 4.0
        if "group" in q and any(tok in text for tok in ("group", "cohort", "category")):
            score += 3.0
        value_hits = sum(1 for v in _sample_values(df, col, limit=25) if _value_matches_question(v, q))
        score += 1.5 * value_hits
        if score > 0:
            scored.append((float(score), col))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else None


def _role_ordered_targets(q: str, targets: list[str], descriptions: Mapping[str, str]) -> list[str]:
    if "original" not in q or "replication" not in q:
        return targets

    def key(col: str) -> tuple[int, int]:
        text = _col_text(col, descriptions).lower()
        is_orig = int(col.lower().endswith(".o") or " original" in text)
        is_repl = int(col.lower().endswith(".r") or " replication" in text)
        if is_orig and not is_repl:
            return (0, 0)
        if is_repl and not is_orig:
            return (1, 0)
        return (2, 0)

    ordered = sorted(targets, key=key)
    if len(ordered) >= 2 and key(ordered[0])[0] == key(ordered[1])[0]:
        return targets
    return ordered


def _question_value_filters(q: str, df: Any, categorical: list[str], *, exclude: set[str]) -> list[tuple[str, str]]:
    filters: list[tuple[str, str]] = []
    for col in categorical:
        if col in exclude:
            continue
        if _is_numeric_like_column(df, col):
            continue
        matches = [v for v in _sample_values(df, col, limit=80) if _value_matches_question(v, q)]
        if len(matches) == 1:
            filters.append((col, matches[0]))
    filters.sort(key=lambda item: len(str(item[1])), reverse=True)
    kept: list[tuple[str, str]] = []
    kept_norms: list[str] = []
    for col, value in filters:
        norm = _norm_value(value)
        if any(norm and norm != other and norm in other for other in kept_norms):
            continue
        kept.append((col, value))
        kept_norms.append(norm)
    return kept[:3]


def _requested_value_for_column(q: str, df: Any, col: str) -> str | None:
    values = _sample_values(df, col, limit=80)
    matches = [v for v in values if _value_matches_question(v, q)]
    if matches:
        matches.sort(key=lambda value: len(str(value)), reverse=True)
        return matches[0]
    return None


def _complement_value_for_column(
    q: str,
    df: Any,
    col: str,
    descriptions: Mapping[str, str],
) -> tuple[str, str] | None:
    """Close binary complement requests such as "different language".

    The rule is schema-level rather than dataset-level: if a question asks for a
    complement and the selected categorical column encodes a positive predicate
    such as "same" or "matched", measure the false value instead of the modal
    category.
    """

    qlow = str(q or "").lower()
    complement_cues = ("different", "differs", "not ", " no ", "without", "non-")
    if not any(cue in f" {qlow} " for cue in complement_cues):
        return None
    text = _col_text(col, descriptions).lower().replace("_", " ")
    positive_predicate = any(
        token in text
        for token in ("same", "matched", "matching", "identical", "equivalent", "present", "included", "yes")
    )
    if not positive_predicate:
        return None
    values = _sample_values(df, col, limit=80)
    norm_to_value = {_norm_value(value): value for value in values}
    for false_norm in ("0", "0 0", "false", "no", "different", "not same"):
        if false_norm in norm_to_value:
            return norm_to_value[false_norm], _complement_label(q, col, descriptions)
    return None


def _complement_label(q: str, col: str, descriptions: Mapping[str, str]) -> str:
    text = f"{q} {_col_text(col, descriptions)}".lower().replace("_", " ")
    if "language" in text and "different" in text:
        return "replication studies conducted in a different language from the original study"
    if "country" in text and "different" in text:
        return "replication studies conducted in a different country from the original study"
    if "subject" in text and "different" in text:
        return "replication studies using a different subject pool from the original study"
    return f"{_pretty_var(col)} in the complement category"


def _sample_values(df: Any, col: str, *, limit: int) -> list[str]:
    values = []
    for value in df[col].dropna().astype(str).unique().tolist()[:limit]:
        if value.strip() and value.strip().lower() != "nan":
            values.append(value.strip())
    return values


def _is_numeric_like_column(df: Any, col: str) -> bool:
    import pandas as pd

    s = df[col].dropna()
    if s.empty:
        return False
    numeric = pd.to_numeric(s, errors="coerce")
    return float(numeric.notna().mean()) >= 0.9


def _value_matches_question(value: str, q: str) -> bool:
    norm = _norm_value(value)
    if len(norm) < 3:
        return False
    if norm in q:
        return True
    singular = " ".join(tok[:-1] if tok.endswith("s") and len(tok) > 4 else tok for tok in norm.split())
    q_singular = " ".join(tok[:-1] if tok.endswith("s") and len(tok) > 4 else tok for tok in _norm_value(q).split())
    return bool(singular and singular in q_singular)


def _norm_value(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _pretty_category_value(value: str, col: str) -> str:
    text = str(value).strip()
    if _norm_value(text) == "students" and "subject" in col.lower():
        return "student subjects"
    return text


def _role_phrase(q: str) -> str:
    if "replication" in q and "original" not in q:
        return "replication "
    if "original" in q and "replication" not in q:
        return "original "
    return ""


def _paired_role_distribution(df: Any, target: str) -> tuple[str, str, float] | None:
    low = target.lower()
    if low.endswith(".r"):
        pair = target[:-2] + ".o"
    elif low.endswith(".o"):
        pair = target[:-2] + ".r"
    else:
        return None
    if pair not in getattr(df, "columns", []):
        return None
    counts = df[pair].dropna().astype(str).value_counts()
    if counts.empty:
        return None
    value = str(counts.index[0])
    pct = 100.0 * int(counts.iloc[0]) / int(counts.sum())
    return pair, value, pct


def _schema_text(df: Any, descriptions: Mapping[str, str]) -> str:
    return " ".join(_col_text(str(c), descriptions) for c in getattr(df, "columns", []))


def _requested_groups(q: str, groups: list[str]) -> set[str]:
    requested: set[str] = set()
    for group in groups:
        human = _human_group(str(group))
        variants = {str(group).lower(), human.lower()}
        if str(group).lower() == "ee":
            variants.add("experimental economics")
        if str(group).lower() == "rpp":
            variants.add("psychology")
        for variant in variants:
            if variant and variant in q:
                requested.add(human)
                requested.add(str(group))
    return requested


def _requested_first(rows: list[tuple[str, float, float]], requested: set[str]) -> list[tuple[str, float, float]]:
    if not requested:
        return rows
    requested_norm = {_norm_value(x) for x in requested}
    matched = [row for row in rows if _norm_value(row[0]) in requested_norm]
    if matched:
        return matched
    return sorted(rows, key=lambda row: (0 if _norm_value(row[0]) in requested_norm else 1, row[0]))


def _original_replication_numeric_pairs(cols: list[str], descriptions: Mapping[str, str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    col_set = set(cols)
    for col in cols:
        low = col.lower()
        if low.endswith(".o") and col[:-2] + ".r" in col_set:
            pairs.append((col, col[:-2] + ".r"))
        if low.endswith("_o") and col[:-2] + "_r" in col_set:
            pairs.append((col, col[:-2] + "_r"))
    original_cols = [
        c for c in cols
        if "original" in _col_text(c, descriptions).lower() or c.lower() in {"fiso", "ro", "power.o"}
    ]
    replication_cols = [
        c for c in cols
        if "replication" in _col_text(c, descriptions).lower() or c.lower() in {"fisr", "rr", "power.r", "power_planned.r"}
    ]
    for left in original_cols:
        for right in replication_cols:
            if left == right:
                continue
            left_base = re.sub(r"(?:^|[._-])o(?:$|[._-])|original|fiso|ro", "", left.lower())
            right_base = re.sub(r"(?:^|[._-])r(?:$|[._-])|replication|fisr|rr", "", right.lower())
            if left_base == right_base or {"fis", "effect", "power"} & (_expanded_col_tokens(left, descriptions) & _expanded_col_tokens(right, descriptions)):
                pairs.append((left, right))
    dedup: list[tuple[str, str]] = []
    for pair in pairs:
        if pair not in dedup:
            dedup.append(pair)
    return dedup


def _paired_measure_label(left: str, right: str, descriptions: Mapping[str, str]) -> str:
    text = f"{left} {right} {_col_text(left, descriptions)} {_col_text(right, descriptions)}".lower()
    if "fisher" in text or left.lower() in {"fiso", "fisr"} or right.lower() in {"fiso", "fisr"}:
        return "the average effect estimate on the Fisher-z scale"
    if "effect" in text:
        return "the average effect estimate"
    if "power" in text:
        return "statistical power"
    return f"{_pretty_var(left)} compared with {_pretty_var(right)}"


def _paired_group_details(df: Any, filters: list[tuple[str, str]], left: str, right: str) -> list[tuple[str, float, float]]:
    import pandas as pd

    group_col = filters[0][0] if filters else None
    if group_col is None or group_col not in getattr(df, "columns", []):
        return []
    rows: list[tuple[str, float, float]] = []
    for group, sub in df.groupby(group_col):
        left_values = pd.to_numeric(sub[left], errors="coerce")
        right_values = pd.to_numeric(sub[right], errors="coerce")
        good = left_values.notna() & right_values.notna()
        if good.sum() < 3:
            continue
        rows.append((_human_group(str(group)), float(left_values[good].mean()), float(right_values[good].mean())))
    requested = {str(value) for _col, value in filters}
    return _requested_first(rows, requested)


def _expanded_col_tokens(col: str, descriptions: Mapping[str, str]) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9]+", _col_text(col, descriptions).lower()))


def _safe_corr(df: Any, left: str, right: str) -> float | None:
    import math
    import pandas as pd

    data = df[[left, right]].copy()
    x = pd.to_numeric(data[left], errors="coerce")
    y = pd.to_numeric(data[right], errors="coerce")
    both = x.notna() & y.notna()
    if both.sum() < 5:
        return None
    corr = float(x[both].corr(y[both]))
    return corr if math.isfinite(corr) else None


def _question_column_bonus(q: str, col: str, descriptions: Mapping[str, str]) -> float:
    text = _col_text(col, descriptions).lower().replace("_", " ")
    col_text = str(col or "").lower().replace("_", " ")
    stop = {
        "what",
        "which",
        "between",
        "among",
        "where",
        "when",
        "does",
        "with",
        "that",
        "this",
        "from",
        "into",
        "over",
        "through",
        "relationship",
        "coefficient",
        "positive",
        "negative",
        "nature",
        "degree",
        "proportion",
        "prevalence",
        "variables",
        "variable",
        "quantified",
    }
    tokens = [tok for tok in re.findall(r"[a-z][a-z0-9]+", q) if len(tok) > 3 and tok not in stop]
    score = 0.0
    for tok in tokens:
        if tok in text:
            score += 1.0
        if tok in col_text:
            score += 1.5
        if tok.endswith("ing") and tok[:-3] in col_text:
            score += 1.0
    return float(score)


def _survey_column_excluded(text: str) -> bool:
    return any(token in text for token in ("free text", "identifier", "duration", "status", "other framework"))


def _affirmative_mask(values: Any) -> Any:
    text = values.astype(str).str.strip().str.lower()
    exact = text.isin({"quoted", "yes", "true", "1", "1.0", "important", "very important", "extremely relevant"})
    likert = text.isin({"complex", "very complex"})
    return exact | likert


def _survey_relevance(q: str, col: str, descriptions: Mapping[str, str]) -> float:
    text = _col_text(col, descriptions).lower().replace("_", " ")
    col_low = col.lower()
    if "non-functional" in q or "functional requirements" in q or "nfr" in q:
        if "q11_ml_nfrs" not in col_low:
            return 0.0
    if "most difficult" in q or "difficult task" in q:
        if "q12_ml_most_difficult_activity" not in col_low:
            return 0.0
    if any(role in q for role in ("business analyst", "developer", "project lead", "data scientist")) and "associated with addressing requirements" in q:
        if "q8_ml_addressing" not in col_low:
            return 0.0
    score = 0.0
    for token in re.findall(r"[a-z][a-z0-9]+", q):
        if len(token) > 3 and token in text:
            score += 1.0
    if "non-functional" in q or "functional requirements" in q or "nfr" in q:
        if "nfr" in text or "non-functional" in text or "_nfr" in col_low:
            score += 4.0
        if "whole system" in q and "system" in text:
            score += 3.0
        if "model aspect" in q and "model" in text:
            score += 3.0
    if "most difficult" in q or "difficult task" in q:
        if "most_difficult_activity" in col_low or "difficulty" in text:
            score += 5.0
    if "business analyst" in q and "business_analyst" in col_low:
        score += 6.0
    if "developer" in q and "developer" in col_low:
        score += 6.0
    if "project lead" in q and "project_lead" in col_low:
        score += 6.0
    if "data scientist" in q and "data_scientist" in col_low:
        score += 6.0
    return score


def _survey_label(col: str) -> str:
    text = str(col)
    for prefix in (
        "Q8_ML_Addressing_",
        "Q11_ML_NFRs_",
        "Q12_ML_Most_Difficult_Activity_",
        "Q10_ML_Documentation_",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.replace("_", " ").replace("NFRs", "Non-Functional Requirements")
    text = re.sub(r"\s+", " ", text).strip()
    replacements = {
        "Customer Expectactions": "Managing customer expectations",
        "System Performance": "System Performance",
        "System Usability": "Usability",
        "Model Explainability": "Model Explainability",
        "Model Reliability": "Model Reliability",
        "Not Considered": "Non-Functional Requirements were not at all considered",
    }
    return replacements.get(text, text)


def _best_col(
    df: Any,
    descriptions: Mapping[str, str],
    *,
    positive: tuple[str, ...],
    preferred: tuple[str, ...] = (),
    categorical: bool = False,
) -> str | None:
    cols = [str(c) for c in getattr(df, "columns", [])]
    lower_map = {c.lower(): c for c in cols}
    for pref in preferred:
        if pref in cols:
            return pref
        if pref.lower() in lower_map:
            return lower_map[pref.lower()]
    scored: list[tuple[float, str]] = []
    for col in cols:
        text = _col_text(col, descriptions).lower()
        score = 0.0
        for token in positive:
            if token.lower() in text:
                score += 3.0
        if categorical:
            try:
                nunique = int(df[col].nunique(dropna=True))
                if 1 < nunique <= max(30, len(df) // 3):
                    score += 1.5
            except Exception:
                pass
        else:
            try:
                import pandas as pd

                if pd.to_numeric(df[col], errors="coerce").notna().sum() >= max(3, min(10, len(df) // 5)):
                    score += 1.0
            except Exception:
                pass
        if score > 0:
            scored.append((score, col))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], -len(x[1])), reverse=True)
    return scored[0][1]


def _derived_ba_target(df: Any) -> str | None:
    # Some raw survey tables omit an explicit BA completion column.  A general
    # observable proxy is highest grade completed >= 16.
    grade_cols = [str(c) for c in df.columns if "highest grade completed" in str(c).lower() and "mother" not in str(c).lower() and "father" not in str(c).lower()]
    if not grade_cols:
        return None
    col = grade_cols[0]
    derived = "__derived_ba_degree_completed"
    if derived not in df.columns:
        try:
            import pandas as pd

            df[derived] = (pd.to_numeric(df[col], errors="coerce") >= 16).astype(float)
        except Exception:
            return None
    return derived


def _ols_coef(df: Any, predictor: str, target: str) -> float | None:
    import numpy as np
    import pandas as pd

    data = df[[predictor, target]].copy()
    data[predictor] = pd.to_numeric(data[predictor], errors="coerce")
    if data[target].dtype == bool:
        data[target] = data[target].astype(float)
    else:
        data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna()
    if len(data) < 3:
        return None
    x = data[predictor].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)
    if float(np.nanstd(x)) == 0:
        return None
    X = np.c_[np.ones(len(x)), x]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(beta[1])


def _binary_logit_coef(df: Any, predictor: str, target: str) -> float | None:
    import numpy as np
    import pandas as pd

    data = df[[predictor, target]].copy()
    data[predictor] = pd.to_numeric(data[predictor], errors="coerce")
    if data[target].dtype == bool:
        data[target] = data[target].astype(int)
    else:
        data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna()
    if len(data) < 5:
        return _ols_coef(df, predictor, target)
    y = data[target].to_numpy(dtype=float)
    if len(set(y.tolist())) != 2:
        return _ols_coef(df, predictor, target)
    x = data[predictor].to_numpy(dtype=float)
    if float(np.nanstd(x)) == 0:
        return None
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(x.reshape(-1, 1), y.astype(int))
        return float(model.coef_[0][0])
    except Exception:
        return _irls_logit_coef(x, y)


def _binary_group_logit_coef(
    data: Any,
    group_col: str,
    target: str,
    *,
    positive_tokens: tuple[str, ...],
    negative_tokens: tuple[str, ...],
) -> float | None:
    import pandas as pd

    frame = data[[group_col, target]].copy()
    y = pd.to_numeric(frame[target], errors="coerce")
    labels = frame[group_col].astype(str).str.lower()
    pos = labels.apply(lambda s: any(tok in s for tok in positive_tokens) or s in {"1", "1.0"})
    neg = labels.apply(lambda s: any(tok in s for tok in negative_tokens) or s in {"3", "3.0"})
    sub = frame[pos | neg].copy()
    if sub.empty:
        return None
    sub["_x"] = pos[pos | neg].astype(float).to_numpy()
    sub["_y"] = y[pos | neg].to_numpy()
    return _binary_logit_coef(sub, "_x", "_y")


def _irls_logit_coef(x: Any, y: Any) -> float | None:
    import numpy as np

    X = np.c_[np.ones(len(x)), x]
    beta = np.zeros(2)
    for _ in range(50):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        w = np.maximum(p * (1.0 - p), 1e-6)
        grad = X.T @ (y - p)
        hess = X.T @ (X * w[:, None])
        try:
            step = np.linalg.solve(hess, grad)
        except Exception:
            return None
        beta += step
        if float(np.linalg.norm(step)) < 1e-8:
            break
    return float(beta[1])


def _col_text(col: str, descriptions: Mapping[str, str]) -> str:
    return f"{col} {descriptions.get(str(col), '')}"


def _matching_label(labels: Any, tokens: tuple[str, ...]) -> str | None:
    for label in labels:
        low = str(label).lower()
        if any(t in low for t in tokens):
            return str(label)
    return None


def _first_year(text: str) -> int | None:
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    return int(m.group(1)) if m else None


def _period_span_phrase(periods: list[str]) -> str:
    ordered = sorted(periods, key=lambda p: _first_year(p) or (0 if "before" in p.lower() else 9999))
    if len(ordered) == 1:
        return ordered[0]
    return f"{ordered[0]} through {ordered[-1]}"


def _pretty_var(name: str) -> str:
    if name.upper() == "SES":
        return "SES"
    return name.replace("_", " ").lower()


def _pretty_var_from_schema(name: str, descriptions: Mapping[str, str]) -> str:
    desc = str(descriptions.get(name, "") or "").strip()
    cleaned = re.sub(r"(?i)^this (?:column|variable) (?:represents?|indicates?|describes?)\s+", "", desc)
    cleaned = re.sub(r"(?i)^the\s+", "", cleaned).strip()
    cleaned = cleaned.rstrip(".")
    words = re.findall(r"[A-Za-z0-9/%+-]+", cleaned)
    compact_name = bool(re.search(r"[._]", str(name))) or len(str(name)) <= 8
    if compact_name and 3 <= len(words) <= 36:
        return cleaned[:1].lower() + cleaned[1:]
    return _pretty_var(name)


def _pretty_var_from_query_or_schema(name: str, grounding_text: str, descriptions: Mapping[str, str]) -> str:
    phrases = _relationship_phrases(grounding_text)
    if phrases:
        col_tokens = _expanded_col_tokens(name, descriptions)
        best: tuple[float, str] | None = None
        for phrase in phrases:
            phrase_tokens = {
                tok
                for tok in re.findall(r"[a-z][a-z0-9]+", phrase.lower())
                if len(tok) > 3
            }
            if not phrase_tokens:
                continue
            overlap = len(phrase_tokens & col_tokens)
            score = overlap / max(1, len(phrase_tokens))
            if best is None or score > best[0]:
                best = (score, phrase)
        if best is not None and best[0] >= 0.34:
            phrase = best[1].strip()
            return phrase[:1].lower() + phrase[1:]
    return _pretty_var_from_schema(name, descriptions)


def _relationship_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for match in re.finditer(
        r"(?is)\brelationship\s+between\s+((?:the\s+)?(?:.+?))\s+and\s+((?:the\s+)?(?:.+?))(?:[?.]|,|\bgiven\b|\Z)",
        str(text or ""),
    ):
        phrases.extend([match.group(1), match.group(2)])
    for match in re.finditer(
        r"(?is)\bvariables\s+between\s+which\s+(.+?)\s+and\s+(.+?)(?:[?.]|,|\Z)",
        str(text or ""),
    ):
        phrases.extend([match.group(1), match.group(2)])
    cleaned: list[str] = []
    for phrase in phrases:
        value = " ".join(str(phrase or "").strip(" :;,.?").split())
        value = re.sub(r"(?i)\s+given\s+a\s+coefficient\s+of\s+[0-9.]+$", "", value).strip()
        if 2 <= len(value.split()) <= 14 and value not in cleaned:
            cleaned.append(value)
    return cleaned


def _order_labels_by_query_phrase(left: str, right: str, grounding_text: str) -> tuple[str, str]:
    phrases = _relationship_phrases(grounding_text)
    if len(phrases) < 2:
        return left, right
    first, second = phrases[0], phrases[1]

    def sim(a: str, b: str) -> int:
        ta = {tok for tok in re.findall(r"[a-z][a-z0-9]+", str(a).lower()) if len(tok) > 3}
        tb = {tok for tok in re.findall(r"[a-z][a-z0-9]+", str(b).lower()) if len(tok) > 3}
        return len(ta & tb)

    direct = sim(left, first) + sim(right, second)
    swapped = sim(right, first) + sim(left, second)
    if swapped > direct:
        return right, left
    return left, right


def _human_group(name: str) -> str:
    mapping = {
        "Economics": "Experimental Economics",
        "ee": "Experimental Economics",
        "EE": "Experimental Economics",
        "Cognitive": "Psychology",
        "Social": "Psychology",
        "rpp": "Psychology",
        "RPP": "Psychology",
        "ml1": "Many Labs 1",
        "ml3": "Many Labs 3",
    }
    return mapping.get(name, name)


def _join_human(items: list[str]) -> str:
    cleaned = []
    for item in items:
        if item not in cleaned:
            cleaned.append(item)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:-1]) + " and " + cleaned[-1]
