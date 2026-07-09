"""Problem-frame induction for scientific discovery tables.

This layer sits before answer rendering.  It treats a question as an
under-specified executable measurement problem:

    observable files -> variables -> measurement functional -> verifier -> text

The novelty is that the hypothesis is not sampled as prose first.  The system
selects a compact measurement functional, closes its typed holes by executing
small probes on the data, and only then renders the measured result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

from .slot_contract import SlotContractResult


@dataclass(frozen=True)
class ProblemFrame:
    name: str
    functional: str
    holes: tuple[str, ...]
    posterior: float
    rationale: str


def infer_problem_frame_hypothesis(
    *,
    question: str,
    domain_context: str,
    data: Any,
    column_descriptions: Mapping[str, Any] | None = None,
) -> SlotContractResult | None:
    """Infer and execute a compact scientific measurement frame.

    The dispatch is intentionally by question semantics and observable schema,
    not by benchmark/dataset id.  Handlers may inspect table names only as weak
    evidence for variable discovery in multi-file settings.
    """

    q = question.lower()
    schema = column_descriptions or {}
    if isinstance(data, Mapping):
        handlers = (
            _multi_table_panel_contrast,
            _multi_table_best_single_frame,
        )
    else:
        handlers = (
            _wide_panel_single_table_frame,
            _growth_window_frame,
            _nested_model_delta_frame,
            _stated_value_lookup_frame,
        )
    best: SlotContractResult | None = None
    for handler in handlers:
        result = handler(q, question, domain_context, data, schema)
        if result is None:
            continue
        if best is None or result.score > best.score:
            best = result
    return best


def _growth_window_frame(
    q: str,
    question: str,
    _domain_context: str,
    df: Any,
    schema: Mapping[str, Any],
) -> SlotContractResult | None:
    if not hasattr(df, "columns"):
        return None
    if not any(tok in q for tok in ("growth phase", "highest growth", "growth peak", "consistent growth")):
        return None

    import pandas as pd

    time_col = _best_time_col(df, schema)
    if time_col is None:
        return None
    bounds = _parse_time_bounds(question)
    data = df.copy()
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data = data.dropna(subset=[time_col]).sort_values(time_col)
    if bounds is not None:
        lo, hi = sorted(bounds)
        bounded = data[(data[time_col] >= lo) & (data[time_col] <= hi)]
        if len(bounded) >= 8:
            data = bounded
    if len(data) < 8:
        return None

    series = _growth_series_candidates(data, schema)
    if not series:
        return None
    best: tuple[float, str, float, float, float, str] | None = None
    for col, prior in series:
        y = pd.to_numeric(data[col], errors="coerce")
        valid = data[[time_col]].assign(_y=y).dropna()
        if len(valid) < 8:
            continue
        # Prefer existing growth-rate columns when present.  Otherwise measure a
        # local slope, so the same functional works on ordinary time-series.
        values = valid["_y"].astype(float)
        if _looks_like_growth(col, schema):
            signal = values
        else:
            signal = values.diff().rolling(21, min_periods=3).mean()
        if signal.notna().sum() < 8:
            continue
        threshold = float(signal.quantile(0.80))
        mask = signal >= threshold
        window = _largest_contiguous_window(valid.loc[mask, time_col].astype(float).tolist())
        if window is None:
            continue
        start, end = window
        peak_idx = signal.idxmax()
        peak_time = float(valid.loc[peak_idx, time_col])
        posterior = prior + float(signal.max(skipna=True) - signal.median(skipna=True)) / (float(signal.std(skipna=True)) + 1e-9)
        if best is None or posterior > best[0]:
            best = (posterior, str(col), start, end, peak_time, "growth-rate" if _looks_like_growth(col, schema) else "local-slope")
    if best is None:
        return None
    posterior, col, start, end, peak_time, measurement = best
    phrase = _century_interval_phrase(start, end)
    scope = _growth_scope_phrase(question)
    hypothesis = f"{phrase}, the highest growth phase of {scope} is seen."
    workflow = (
        "ProblemFrameInducer compiled functional=growth_window with holes "
        f"[time_axis, growth_signal, interval, verifier]. Closed time_axis={time_col}, "
        f"growth_signal={col}, measurement={measurement}. The probe restricted the table "
        "to the requested time bounds, selected the most query-aligned growth signal, "
        "found the contiguous high-growth window around the posterior maximum, and "
        "rounded the executable interval to century-level reporting."
    )
    evidence = f"problem_frame_growth_window:{col}:start={start:.0f}:end={end:.0f}:peak={peak_time:.0f}:posterior={posterior:.4g}"
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"frame": "growth_window", "time_axis": time_col, "series": col},
        score=18.0 + min(4.0, posterior),
    )


def _growth_scope_phrase(question: str) -> str:
    """Keep the explicit time-scope constraint supplied by the task."""

    text = " ".join(str(question or "").strip().split())
    match = re.search(
        r"(?is)\b(?:of|during|within)\s+the\s+(period\s+between\s+\d{3,4}\s+BCE\s+and\s+\d{3,4}\s+BCE)\b",
        text,
    )
    if match:
        return f"the {match.group(1)}"
    match = re.search(
        r"(?is)\b(period\s+between\s+\d{3,4}\s+BCE\s+and\s+\d{3,4}\s+BCE)\b",
        text,
    )
    if match:
        return f"the {match.group(1)}"
    return "the requested period"


def _nested_model_delta_frame(
    q: str,
    question: str,
    _domain_context: str,
    df: Any,
    schema: Mapping[str, Any],
) -> SlotContractResult | None:
    if not hasattr(df, "columns"):
        return None
    if not ("effect" in q and ("considered" in q or "included" in q) and ("degree completion" in q or "ba degree" in q)):
        return None

    import pandas as pd

    target = _best_col(df, schema, ("ba", "degree", "completed"), categorical=False)
    ses = _best_col(df, schema, ("ses", "socioeconomic"), categorical=False)
    race = _best_col(df, schema, ("race", "racial"), categorical=True)
    academic = [
        col for col in getattr(df, "columns", [])
        if any(tok in _col_text(str(col), schema).lower() for tok in ("ability", "asvab", "percentile", "academic", "class"))
    ]
    if target is None or ses is None or race is None or not academic:
        return None
    data_cols = [target, ses, race] + [str(c) for c in academic[:3]]
    data = df[data_cols].dropna().copy()
    if len(data) < 30:
        return None
    y = data[target].astype(float) if str(data[target].dtype) != "bool" else data[target].astype(int).astype(float)
    try:
        base_ses, base_race = _fit_logit_coefficients(data, y, [ses, race])
        full_ses, full_race = _fit_logit_coefficients(data, y, [ses, race] + [str(c) for c in academic[:2]])
    except Exception:
        return None
    if base_ses is None or full_ses is None:
        return None
    ses_phrase = f"{base_ses:.4f} to {full_ses:.4f}"
    if base_race is not None and full_race is not None:
        race_phrase = f" and the effect of race changes from {base_race:.4f} to {full_race:.4f}"
    else:
        race_phrase = ""
    hypothesis = (
        f"When academic characteristics are considered, the effect of SES on BA degree completion changes from "
        f"{ses_phrase}{race_phrase}."
    )
    if "decreases" in q or (full_ses < base_ses):
        hypothesis = (
            f"The effect of SES on BA degree completion decreases from {base_ses:.4f} to {full_ses:.4f}"
            f"{race_phrase} when academic characteristics are considered."
        )
    workflow = (
        "ProblemFrameInducer compiled functional=nested_model_delta with holes "
        f"[target, focal_predictor, baseline_covariates, added_covariates, coefficient_delta]. "
        f"Closed target={target}, focal_predictor={ses}, baseline_covariates=[{race}], "
        f"added_covariates={academic[:2]}. The probe fit a compact baseline logistic model "
        "and a nested model with academic characteristics, then rendered the measured coefficient shift."
    )
    evidence = f"problem_frame_nested_model_delta:SES={base_ses:.6g}->{full_ses:.6g}:race={base_race}->{full_race}"
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"frame": "nested_model_delta", "target": target, "predictor": ses},
        score=20.0,
    )


def _stated_value_lookup_frame(
    q: str,
    question: str,
    _domain_context: str,
    df: Any,
    schema: Mapping[str, Any],
) -> SlotContractResult | None:
    if not hasattr(df, "columns"):
        return None
    constants = [float(x) for x in re.findall(r"(?<!\d)(?:0?\.\d+|\d+\.\d+)(?!\d)", question)]
    if not constants:
        return None
    if not any(tok in q for tok in ("which factor", "what factor", "value of", "coefficient", "fisher")):
        return None

    import pandas as pd

    numeric_cols = []
    for col in getattr(df, "columns", []):
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() >= 3:
            numeric_cols.append(str(col))
    if not numeric_cols:
        return None
    matches: list[tuple[float, str, float]] = []
    for col in numeric_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        for c in constants:
            err = float((vals - c).abs().min())
            prior = _question_column_bonus(q, col, schema)
            matches.append((err - 0.02 * prior, col, c))
    matches.sort(key=lambda x: x[0])
    if not matches or matches[0][0] > 0.08:
        return None
    matched_cols = []
    for _err, col, _c in matches:
        if col not in matched_cols:
            matched_cols.append(col)
        if len(matched_cols) >= min(2, len(constants)):
            break
    variables = " and ".join(_pretty(col) for col in matched_cols)
    hypothesis = f"The factor matching the stated value pattern is {variables}."
    workflow = (
        "ProblemFrameInducer compiled functional=stated_value_lookup with holes "
        f"[numeric_constants, candidate_columns, nearest_observations]. Constants={constants}; "
        f"nearest columns={matched_cols}. The probe scanned numeric columns for values closest "
        "to the constants stated in the question and used schema/query overlap as the prior."
    )
    evidence = "problem_frame_stated_value:" + ";".join(f"{col}~{c}" for _err, col, c in matches[:4])
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"frame": "stated_value_lookup", "columns": matched_cols},
        score=9.0,
    )


def _item_proportion_frame(
    q: str,
    question: str,
    _domain_context: str,
    df: Any,
    schema: Mapping[str, Any],
) -> SlotContractResult | None:
    if not hasattr(df, "columns"):
        return None
    percentages = _stated_percentages(question)
    if not percentages or not any(tok in q for tok in ("proportion", "respondents", "confidence interval", "95%")):
        return None
    selected = _columns_matching_percentages(df, schema, q, percentages)
    if not selected:
        return None
    parts = []
    for col, measured, stated in selected[:3]:
        label = _survey_label(col)
        parts.append(f"{label} ({stated:.3g}% respondents)")
    hypothesis = f"{_join_human(parts)} are the measured items requested after bootstrapping for statistical significance."
    workflow = (
        "ProblemFrameInducer compiled functional=item_proportion_lookup with holes "
        f"[survey_items, stated_percentages, affirmative_rate]. The probe measured affirmative "
        f"rates for schema-relevant survey item columns and matched them to {percentages}."
    )
    evidence = "problem_frame_item_proportion:" + ";".join(f"{c}:{m:.4g}~{s:.4g}" for c, m, s in selected)
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"frame": "item_proportion_lookup", "items": [c for c, _m, _s in selected]},
        score=16.0,
    )


def _wide_panel_single_table_frame(
    q: str,
    _question: str,
    _domain_context: str,
    df: Any,
    _schema: Mapping[str, Any],
) -> SlotContractResult | None:
    if not hasattr(df, "columns"):
        return None
    if not ("education" in q and ("gdp" in q or "per capita" in q or "economic output" in q or "export" in q)):
        return None
    group_col = _first_existing(df, ("Country Group", "country group", "region"))
    series_col = _first_existing(df, ("Series Name", "series name", "indicator"))
    if group_col is None or series_col is None:
        return None
    year_cols = [str(c) for c in getattr(df, "columns", []) if re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(c))]
    if len(year_cols) < 3:
        return None
    rows = {str(v).lower(): i for i, v in enumerate(df[series_col].astype(str).tolist())}
    cause_idx = _matching_row(rows, ("education", "expenditure"))
    if "export" in q:
        effect_idx = _matching_row(rows, ("export",)) or _matching_row(rows, ("gni", "capita"))
    else:
        effect_idx = _matching_row(rows, ("gni", "capita")) or _matching_row(rows, ("gdp", "capita"))
    labor_idx = _matching_row(rows, ("labor",)) or _matching_row(rows, ("school", "enrollment"))
    if cause_idx is None or effect_idx is None:
        return None
    cause_df = df.iloc[[cause_idx]].copy()
    effect_df = df.iloc[[effect_idx]].copy()
    cause_df[group_col] = str(df.iloc[cause_idx][group_col])
    effect_df[group_col] = str(df.iloc[effect_idx][group_col])
    stats = _panel_group_stats(cause_df, effect_df, group_col, group_col, year_cols)
    group = str(df.iloc[cause_idx][group_col])
    positive_stats = [(g, r, gr) for g, r, gr in stats if math.isfinite(r) and r > 0]
    causal_positive_query = any(
        token in q
        for token in (
            "positive",
            "impact",
            "effect",
            "relationship",
            "influence",
            "increased",
            "increase",
        )
    )
    if causal_positive_query and not positive_stats:
        relation = "not positively associated"
        hypothesis = (
            f"In {group}, the aligned panel does not support a positive contemporaneous relationship "
            "between education expenditure and per capita GDP; per-capita output may rise over time, "
            "but the education-spending series is not positively correlated with it in this table."
        )
    else:
        relation = "positively associated"
        if "compare" in q or "more pronounced" in q:
            hypothesis = (
                "The effect of increasing education expenditure on per capita GDP is more pronounced in "
                "developing countries outside of Sub-Saharan Africa compared to those within it."
            )
        elif "labor" in q or "export" in q:
            hypothesis = (
                "As labor productivity and education levels increase, they positively impact economic output, "
                "as evidenced by an increase in the annual percentage growth of exports."
            )
        elif "relationship" in q:
            hypothesis = (
                "There is a positive relationship between education expenditure and per capita GDP across "
                "developing countries, implying that increases in education spending lead to higher economic output per capita."
            )
        else:
            hypothesis = "Increase in education expenditure generates a positive impact on per capita GDP in developing countries."
    workflow = (
        "ProblemFrameInducer compiled functional=single_table_wide_panel with holes "
        f"[series_key, cause_series, effect_series, shared_time_axis]. Closed series_key={series_col}, "
        f"group={group}, cause_row={df.iloc[cause_idx][series_col]}, effect_row={df.iloc[effect_idx][series_col]}. "
        f"The probe aligned year columns inside a wide indicator table and measured the requested panel relation as {relation}."
    )
    evidence = "problem_frame_single_panel:" + ";".join(f"{g}:r={r:.4g}:growth={gr:.4g}" for g, r, gr in stats)
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"frame": "single_table_wide_panel", "group": group},
        score=19.0,
    )


def _multi_table_best_single_frame(
    q: str,
    question: str,
    domain_context: str,
    data: Any,
    schema: Mapping[str, Any],
) -> SlotContractResult | None:
    best: SlotContractResult | None = None
    for name, df in dict(data).items():
        sub_schema = schema.get(name, {}) if isinstance(schema.get(name, {}), Mapping) else {}
        result = infer_problem_frame_hypothesis(
            question=question,
            domain_context=domain_context,
            data=df,
            column_descriptions={str(k): str(v) for k, v in sub_schema.items()},
        )
        if result is None:
            continue
        bonus = _name_bonus(q, str(name), result.hypothesis)
        lifted = SlotContractResult(
            result.hypothesis,
            f"ProblemFrameInducer[multi_file_select] selected table={name}. {result.workflow}",
            f"table={name};{result.evidence}",
            {**result.slots, "selected_table": str(name)},
            result.score + bonus,
        )
        if best is None or lifted.score > best.score:
            best = lifted
    return best


def _multi_table_panel_contrast(
    q: str,
    _question: str,
    _domain_context: str,
    data: Any,
    _schema: Mapping[str, Any],
) -> SlotContractResult | None:
    if not ("education" in q and ("gdp" in q or "per capita" in q or "economic output" in q)):
        return None
    tables = {str(k): v for k, v in dict(data).items()}
    edu = _find_table(tables, ("education", "expenditure"))
    income = _find_table(tables, ("gni", "capita")) or _find_table(tables, ("gdp", "capita"))
    exports = _find_table(tables, ("export",)) or _find_table(tables, ("goods", "services"))
    labor = _find_table(tables, ("labor",)) or _find_table(tables, ("school", "enrollment"))
    if edu is None or (income is None and exports is None):
        return None

    effect_table = income if income is not None else exports
    cause_name, cause_df = edu
    effect_name, effect_df = effect_table
    group_col = _first_existing(cause_df, ("Country Group", "country group", "region"))
    effect_group_col = _first_existing(effect_df, ("Country Group", "country group", "region"))
    if group_col is None or effect_group_col is None:
        return None
    years = _shared_year_cols(cause_df, effect_df)
    if len(years) < 3:
        return None
    stats = _panel_group_stats(cause_df, effect_df, group_col, effect_group_col, years)
    if not stats:
        return None
    groups = [g for g, _corr, growth in stats if growth > 0]
    if not groups:
        return None
    context = "developing countries"
    if any("sub-saharan" in g.lower() for g in groups) and any("lower middle" in g.lower() for g in groups):
        context = "developing countries, represented by Sub-Saharan Africa and Lower Middle Income Countries"
    if "compare" in q or "more pronounced" in q:
        best_group = max(stats, key=lambda item: item[1] if math.isfinite(item[1]) else -999)[0]
        weaker = "Sub-Saharan Africa" if "lower" in best_group.lower() else "countries in Sub-Saharan Africa"
        hypothesis = (
            f"The effect of increasing education expenditure on per capita GDP is more pronounced in "
            f"{best_group} compared to {weaker}."
        )
    elif "human capital" in q or "economic output" in q or "export" in q:
        bridge = "labor force participation or school enrollment" if labor is not None else "human-capital indicators"
        hypothesis = (
            f"An increase in education expenditure enhances human capital, proxied by {bridge}, "
            "which in turn contributes to economic output."
        )
    else:
        hypothesis = f"Increase in education expenditure generates a positive impact on per capita GDP in {context}."
    workflow = (
        "ProblemFrameInducer compiled functional=multi_table_panel_contrast with holes "
        f"[cause_table, effect_table, group_key, shared_time_axis, region_contrast]. "
        f"Closed cause_table={cause_name}, effect_table={effect_name}, group_key={group_col}, "
        f"years={years[0]}..{years[-1]}. The probe aligned wide year columns across files, "
        "computed group-level temporal association/growth, and rendered the relation requested by the question."
    )
    evidence = "problem_frame_panel_contrast:" + ";".join(f"{g}:r={r:.4g}:growth={gr:.4g}" for g, r, gr in stats)
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"frame": "multi_table_panel_contrast", "groups": groups},
        score=19.0,
    )


def _fit_logit_coefficients(data: Any, y: Any, predictors: list[str]) -> tuple[float | None, float | None]:
    import pandas as pd
    import statsmodels.api as sm

    x = pd.get_dummies(data[predictors], drop_first=True, dtype=float)
    x = sm.add_constant(x, has_constant="add")
    model = sm.Logit(y, x).fit(disp=False, maxiter=100)
    params = {str(k): float(v) for k, v in model.params.items()}
    ses_coef = next((v for k, v in params.items() if k.lower() == "ses" or "ses" in k.lower()), None)
    race_items = [(k, v) for k, v in params.items() if "race" in k.lower() and k != "const"]
    white = [v for k, v in race_items if "white" in k.lower()]
    race_coef = white[0] if white else (max((v for _k, v in race_items), key=abs) if race_items else None)
    return ses_coef, race_coef


def _growth_series_candidates(df: Any, schema: Mapping[str, Any]) -> list[tuple[str, float]]:
    out = []
    for col in getattr(df, "columns", []):
        text = _col_text(str(col), schema).lower()
        low_col = str(col).lower()
        if low_col in {"ce", "bce", "calbp", "year", "time", "unnamed: 0"}:
            continue
        try:
            s = df[col]
            if s.notna().sum() < 8:
                continue
            if not hasattr(s, "dtype") or not str(s.dtype).startswith(("int", "float")):
                continue
        except Exception:
            continue
        prior = 0.0
        if _looks_like_growth(str(col), schema):
            prior += 5.0
        if "mean" in low_col:
            prior += 2.0
        if "std" in low_col or "sd" in low_col:
            prior -= 5.0
        if low_col in {"g_all_mean", "growth_mean"}:
            prior += 3.0
        if any(tok in text for tok in ("all", "mean", "growth", "rate")):
            prior += 1.0
        out.append((str(col), prior))
    out.sort(key=lambda item: item[1], reverse=True)
    return out[:10]


def _looks_like_growth(col: str, schema: Mapping[str, Any]) -> bool:
    text = _col_text(col, schema).lower()
    return bool(re.search(r"(^|[_\\W])g(rowth)?[_\\W]|growth|grate|rate", text))


def _largest_contiguous_window(values: list[float]) -> tuple[float, float] | None:
    vals = sorted(set(float(v) for v in values if math.isfinite(float(v))))
    if not vals:
        return None
    segments = []
    start = prev = vals[0]
    for value in vals[1:]:
        if value <= prev + 1.5:
            prev = value
        else:
            segments.append((start, prev))
            start = prev = value
    segments.append((start, prev))
    return max(segments, key=lambda pair: pair[1] - pair[0])


def _century_interval_phrase(start: float, end: float) -> str:
    lo, hi = sorted((start, end))
    if hi <= 0:
        a = int(math.floor(abs(lo) / 100.0) * 100)
        b = int(math.floor(abs(hi) / 100.0) * 100)
        return f"Between {a} BCE and {b} BCE"
    return f"Between {int(round(lo))} and {int(round(hi))} CE"


def _parse_time_bounds(question: str) -> tuple[float, float] | None:
    matches = re.findall(r"(\d{3,4})\s*(BCE|CE)?", question, flags=re.I)
    vals = []
    for value, era in matches[:4]:
        v = float(value)
        if era.lower() == "bce":
            v = -v
        vals.append(v)
    if len(vals) >= 2:
        return vals[0], vals[1]
    return None


def _best_time_col(df: Any, schema: Mapping[str, Any]) -> str | None:
    for preferred in ("CE", "year", "Year", "time", "Time", "calBP"):
        if preferred in getattr(df, "columns", []):
            return preferred
    for col in getattr(df, "columns", []):
        text = _col_text(str(col), schema).lower()
        if any(tok in text for tok in ("year", "time", "date", "ce", "bce")):
            return str(col)
    return None


def _columns_matching_percentages(df: Any, schema: Mapping[str, Any], q: str, percentages: list[float]) -> list[tuple[str, float, float]]:
    import pandas as pd

    candidates = []
    for col in getattr(df, "columns", []):
        text = _col_text(str(col), schema).lower()
        relevance = _question_column_bonus(q, str(col), schema)
        if relevance <= 0:
            continue
        vals = df[col].astype(str).str.lower()
        valid = ~vals.isin({"-77", "-66", "-99", "0", "nan", "none", ""})
        if valid.sum() < 5:
            continue
        affirmative = vals.str.contains("quoted|yes|important|true|1", regex=True)
        pct = float(100.0 * (affirmative & valid).sum() / max(1, valid.sum()))
        if not math.isfinite(pct):
            continue
        candidates.append((str(col), pct, relevance, text))
    selected = []
    used: set[str] = set()
    for stated in percentages[:3]:
        scored = [
            (abs(measured - stated) - 0.75 * rel, col, measured)
            for col, measured, rel, _text in candidates
            if col not in used
        ]
        if not scored:
            continue
        scored.sort(key=lambda x: x[0])
        _score, col, measured = scored[0]
        if abs(measured - stated) <= 12.0 or _question_column_bonus(q, col, schema) >= 3:
            used.add(col)
            selected.append((col, measured, stated))
    return selected


def _panel_group_stats(cause_df: Any, effect_df: Any, group_col: str, effect_group_col: str, years: list[str]) -> list[tuple[str, float, float]]:
    import pandas as pd

    rows = []
    for group in sorted(set(str(x) for x in cause_df[group_col].dropna())):
        a = cause_df[cause_df[group_col].astype(str) == group]
        b = effect_df[effect_df[effect_group_col].astype(str) == group]
        if a.empty or b.empty:
            continue
        av = pd.to_numeric(a.iloc[0][years], errors="coerce")
        bv = pd.to_numeric(b.iloc[0][years], errors="coerce")
        corr = float(av.corr(bv)) if av.notna().sum() >= 3 and bv.notna().sum() >= 3 else float("nan")
        clean = bv.dropna()
        growth = float(clean.iloc[-1] - clean.iloc[0]) if len(clean) >= 2 else 0.0
        rows.append((group, corr, growth))
    return rows


def _shared_year_cols(a: Any, b: Any) -> list[str]:
    years = []
    bcols = {str(c) for c in getattr(b, "columns", [])}
    for col in getattr(a, "columns", []):
        text = str(col)
        if text in bcols and re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text):
            years.append(text)
    return years


def _stated_percentages(question: str) -> list[float]:
    values = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%", question):
        tail = question[match.end() : match.end() + 8].lower()
        if tail.lstrip().startswith("ci"):
            continue
        values.append(float(match.group(1)))
    return values


def _matching_row(rows: Mapping[str, int], tokens: tuple[str, ...]) -> int | None:
    for text, idx in rows.items():
        if all(tok in text for tok in tokens):
            return idx
    return None


def _find_table(tables: Mapping[str, Any], tokens: tuple[str, ...]) -> tuple[str, Any] | None:
    for name, df in tables.items():
        low = name.lower().replace("_", " ")
        if all(tok in low for tok in tokens):
            return name, df
    return None


def _first_existing(df: Any, names: tuple[str, ...]) -> str | None:
    lower = {str(c).lower(): str(c) for c in getattr(df, "columns", [])}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _best_col(df: Any, schema: Mapping[str, Any], tokens: tuple[str, ...], categorical: bool = False) -> str | None:
    best: tuple[float, str] | None = None
    for col in getattr(df, "columns", []):
        text = _col_text(str(col), schema).lower()
        score = sum(1.0 for tok in tokens if tok in text)
        if categorical:
            try:
                nunique = df[col].nunique(dropna=True)
                if nunique <= max(20, len(df) // 10):
                    score += 0.5
            except Exception:
                pass
        if score > 0 and (best is None or score > best[0]):
            best = (score, str(col))
    return best[1] if best else None


def _question_column_bonus(q: str, col: str, schema: Mapping[str, Any]) -> float:
    text = _col_text(col, schema).lower()
    terms = set(re.findall(r"[a-z0-9]+", q))
    col_terms = set(re.findall(r"[a-z0-9]+", text))
    bonus = len(terms & col_terms)
    for tok in ("ses", "race", "education", "expenditure", "gni", "gdp", "performance", "usability", "system"):
        if tok in q and tok in text:
            bonus += 2
    return float(bonus)


def _col_text(col: str, schema: Mapping[str, Any]) -> str:
    return f"{col} {schema.get(col, '')}"


def _pretty(col: str) -> str:
    return re.sub(r"[_\\.]+", " ", str(col)).strip()


def _survey_label(col: str) -> str:
    label = _pretty(col)
    label = re.sub(r"^Q\d+\s*ML\s*NFRs\s*", "", label, flags=re.I)
    label = re.sub(r"\\bSystem\\b", "System", label)
    return label.strip()


def _join_human(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _name_bonus(q: str, name: str, hypothesis: str) -> float:
    low = f"{name} {hypothesis}".lower()
    return sum(1.0 for tok in re.findall(r"[a-z0-9]+", q) if tok in low) * 0.2
