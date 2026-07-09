"""Question-form induction for measurable discovery hypotheses.

The inducer is intentionally lightweight: it maps natural-language questions
to typed answer forms that can be closed by executable dataframe probes.  This
is the bridge between a weak model and long-horizon scientific discovery: the
model does not have to invent a whole hypothesis at once; it only needs a form
whose slots can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Mapping


@dataclass(frozen=True)
class QuestionForm:
    name: str
    slots: tuple[str, ...]
    cues: tuple[str, ...] = ()
    constants: Mapping[str, object] = field(default_factory=dict)
    confidence: float = 1.0


def infer_question_form(question: str, schema_text: str = "") -> QuestionForm | None:
    q = question.lower()
    schema = schema_text.lower()
    percentages = _percentages(question)

    if "original" in q and "replication" in q and any(t in q for t in ("effect size", "effect estimate", "fisher-z")):
        return QuestionForm(
            "grouped_original_replication_comparison",
            ("group", "original_effect", "replication_effect", "relation"),
            ("original", "replication", "effect"),
            {"percentages": percentages},
            0.95,
        )
    if "original" in q and "replication" in q and "power" in q:
        return QuestionForm(
            "paired_group_mean_comparison",
            ("group", "left_measure", "right_measure", "left_mean", "right_mean"),
            ("original", "replication", "power"),
            {"measure_family": "power"},
            0.92,
        )
    if "coefficient" in q and re.search(r"-?\d+(?:\.\d+)?", q):
        return QuestionForm(
            "stated_coefficient_relationship",
            ("x", "y", "coefficient", "direction"),
            ("coefficient",),
            {"coefficient": _first_number(question), "direction": _direction(q)},
            0.9,
        )
    if ("replaced" in q or "surpassed" in q or "surpass" in q) and any(t in q for t in ("contributor", "flora", "pathway", "gardening", "agriculture")):
        return QuestionForm(
            "period_category_crossover",
            ("period", "category", "count", "crossover_relation"),
            ("replacement", "pathway"),
            {},
            0.9,
        )
    if percentages and any(t in q for t in ("respondents", "confidence interval", "95% ci", "bootstrapping", "proportion")):
        return QuestionForm(
            "prompted_survey_item_proportion",
            ("items", "percentages", "uncertainty", "affirmative_response"),
            ("survey", "percentage"),
            {"percentages": percentages},
            0.88,
        )
    if "highest" in q and "median wealth" in q and any(t in q for t in ("gender", "sex", "disparit")):
        return QuestionForm(
            "highest_group_median_gap",
            ("population_filter", "group", "yearly_outcome", "gap_statistic"),
            ("median", "wealth", "disparity"),
            {},
            0.86,
        )
    if ("median wealth" in q or "wealth levels" in q) and any(t in q for t in ("compare", "compared", "higher", "lower", "disparit")):
        return QuestionForm(
            "group_outcome_comparison",
            ("group", "outcome", "filter", "year", "comparison"),
            ("group", "wealth", "comparison"),
            {"year": _first_year(q)},
            0.82,
        )
    if ("degree completion" in q or "ba degree" in q) and any(t in q for t in ("socioeconomic", "ses", "family size", "racial", "black", "white")):
        return QuestionForm(
            "coefficient_or_group_difference",
            ("predictor_or_group", "outcome", "model_family", "coefficient"),
            ("degree", "relationship"),
            {},
            0.84,
        )
    if "education" in q and ("gdp" in q or "per capita" in q) and ("impact" in q or "effect" in q or "regions" in q):
        return QuestionForm(
            "wide_panel_positive_effect",
            ("region", "cause_series", "effect_series", "temporal_association"),
            ("wide_panel", "causal_language"),
            {},
            0.84,
        )
    if (
        any(t in q for t in ("which", "what"))
        and any(t in schema for t in ("quoted", "degree to which", "respondent"))
        and not any(t in q for t in ("relationship", "effect", "affect", "influence", "impact", "associated"))
    ):
        return QuestionForm(
            "prompted_survey_item_proportion",
            ("items", "percentages", "uncertainty", "affirmative_response"),
            ("survey",),
            {"percentages": percentages},
            0.65,
        )
    return None


def _percentages(text: str) -> tuple[float, ...]:
    values = []
    for match in re.finditer(r"\d+(?:\.\d+)?\s*%", text):
        if re.match(r"\s*CI\b", text[match.end() :]):
            continue
        values.append(float(match.group(0).replace("%", "").strip()))
    return tuple(values)


def _first_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _first_year(text: str) -> int | None:
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    return int(match.group(1)) if match else None


def _direction(q: str) -> str:
    if "negative" in q or "decrease" in q or "lower" in q:
        return "negative"
    if "positive" in q or "increase" in q or "higher" in q:
        return "positive"
    return ""
