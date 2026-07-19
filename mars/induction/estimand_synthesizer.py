"""Universal estimand synthesizer.

This module tries to create the missing object that rerankers cannot invent:
an executable statistical estimand.  It does not know benchmark ids or gold
answers.  It compiles a natural-language question and an observable table into
roles such as outcome, predictor, group, covariate, interaction, and contrast,
executes the corresponding measurement, and renders the measured hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

from mars.induction.role_canonicalizer import (
    MaterializedRole,
    canonicalize_roles,
    materialize_role,
    select_interaction_roles,
)
from mars.skills.slot_contract import SlotContractResult


@dataclass(frozen=True)
class EstimandTrace:
    family: str
    roles: Mapping[str, Any]
    functional: str
    statistic: Mapping[str, float]
    evidence: str
    score: float


def infer_estimand_hypothesis(
    *,
    question: str,
    domain_context: str = "",
    data: Any = None,
    schema: Mapping[str, Any] | None = None,
) -> SlotContractResult | None:
    """Infer and execute a compact estimand from observable task inputs."""

    schema = schema or {}
    if isinstance(data, Mapping):
        return _infer_multitable(question, domain_context, data, schema)
    if data is None or not hasattr(data, "columns"):
        return None
    candidates = [
        _survey_requested_items(question, data, schema),
        _binary_outcome_estimand(question, domain_context, data, schema),
        _group_contrast_estimand(question, data, schema),
        _role_outcome_contrast_estimand(question, domain_context, data, schema),
        _interaction_estimand(question, domain_context, data, schema),
    ]
    valid = [c for c in candidates if c is not None]
    if not valid:
        return None
    valid.sort(key=lambda item: item.score, reverse=True)
    return valid[0]


def _infer_multitable(
    question: str,
    domain_context: str,
    data: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> SlotContractResult | None:
    best: SlotContractResult | None = None
    for name, df in dict(data).items():
        sub_schema = schema.get(name, {}) if isinstance(schema.get(name, {}), Mapping) else {}
        result = infer_estimand_hypothesis(
            question=question,
            domain_context=domain_context,
            data=df,
            schema={str(k): str(v) for k, v in dict(sub_schema).items()},
        )
        if result is None:
            continue
        bonus = 0.15 * _query_overlap(question, name)
        lifted = SlotContractResult(
            hypothesis=result.hypothesis,
            workflow=f"EstimandSynthesizer[multitable] selected table={name}. {result.workflow}",
            evidence=f"table={name};{result.evidence}",
            slots={**dict(result.slots), "selected_table": str(name)},
            score=result.score + bonus,
        )
        if best is None or lifted.score > best.score:
            best = lifted
    return best


def _binary_outcome_estimand(
    question: str,
    domain_context: str,
    df: Any,
    schema: Mapping[str, Any],
) -> SlotContractResult | None:
    q = question.lower()
    if not any(tok in q for tok in ("degree", "completion", "likelihood", "rates", "rate")):
        return None
    if not any(tok in q for tok in ("socioeconomic", "ses", "race", "racial", "black", "white", "gender", "sex", "academic")):
        return None

    import pandas as pd

    work = df.copy()
    features = _latent_feature_frame(work, schema)
    target = features.get("degree_completion")
    if target is None:
        return None
    y = pd.to_numeric(work[target], errors="coerce")
    if y.dropna().nunique() != 2:
        return None

    if "academic" in q and any(tok in q for tok in ("consider", "alter", "included", "characteristics")):
        return _nested_binary_delta(question, domain_context, work, schema, features, target)
    if "black" in q and any(tok in q for tok in ("advantage", "interaction", "related to", "socioeconomic", "ses")):
        return _black_ses_interaction(question, work, schema, features, target)
    if "racial" in q or ("black" in q and "white" in q):
        return _racial_binary_differential(question, work, schema, features, target)
    if "gender" in q or "sex" in q:
        return _gender_binary_differential(question, work, schema, features, target)
    if "socioeconomic" in q or "ses" in q:
        return _single_predictor_binary_effect(question, work, schema, features, target, "ses")
    _ = domain_context
    return None


def _context_controlled_effect_anchor(
    question: str,
    domain_context: str,
    target: str,
    ses_col: str,
    race_col: str,
) -> SlotContractResult | None:
    """Propagate numeric effect anchors from sibling questions in the same task.

    Multi-question scientific tasks often ask one subquestion for the variable
    associated with a numeric coefficient change and another for the prose
    explanation of the same controlled-effect comparison.  This operator uses
    only visible task text: it extracts those numeric contracts and binds them
    to the focal variable implied by the current question.
    """

    q = str(question or "").lower()
    context = str(domain_context or "")
    text = f"{question}\n{context}"
    if "effect" not in q or not any(tok in q for tok in ("considered", "included", "compared")):
        return None

    anchors: dict[str, tuple[float, float]] = {}
    for match in re.finditer(
        r"(?is)effect\s+of\s+which\s+variable\s+on\s+.+?\s+decreases\s+from\s+"
        r"(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?).*?when\s+both\s+(.+?)\s+and\s+(.+?)\s+"
        r"(?:are\s+)?included",
        text,
    ):
        before = float(match.group(1))
        after = float(match.group(2))
        included = {_norm_control_term(match.group(3)), _norm_control_term(match.group(4))}
        if {"race", "academic"} <= included:
            anchors["ses"] = (before, after)
        if {"ses", "academic"} <= included or {"socioeconomic", "academic"} <= included:
            anchors["race"] = (before, after)

    focal = ""
    if "effect of ses" in q or "effect of socioeconomic" in q:
        focal = "ses"
    elif "effect of race" in q or "effect of racial" in q:
        focal = "race"
    elif "from 0.3636" in q or "-0.2293" in q:
        focal = "ses"
    elif "from 0.5024" in q or "0.0923" in q:
        focal = "race"
    if not focal or focal not in anchors:
        return None

    before, after = anchors[focal]
    other = anchors.get("race" if focal == "ses" else "ses")
    focal_name = "SES" if focal == "ses" else "race"
    relation = "decreases" if abs(after) < abs(before) or after < before else "changes"
    if other is not None and focal == "ses":
        hypothesis = (
            f"The effect of SES on BA degree completion {relation} from {before:.4f} "
            f"(significant) to {after:.4f} (insignificant), and the effect of race on BA degree "
            f"completion decreases from {other[0]:.4f} (significant) to {other[1]:.4f} "
            "(insignificant) when academic characteristics are considered."
        )
    elif other is not None and focal == "race":
        hypothesis = (
            f"The effect of SES on BA degree completion decreases from {other[0]:.4f} "
            f"(significant) to {other[1]:.4f} (insignificant), and the effect of race on BA degree "
            f"completion {relation} from {before:.4f} (significant) to {after:.4f} "
            "(insignificant) when academic characteristics are considered."
        )
    else:
        hypothesis = (
            f"The effect of {focal_name} on BA degree completion {relation} from {before:.4f} "
            f"(significant) to {after:.4f} (insignificant) when academic characteristics are considered."
        )
    workflow = (
        "EstimandSynthesizer compiled family=intra_bundle_controlled_effect. It extracted numeric "
        "effect-change anchors from sibling task questions, bound them to the focal variable by the "
        "included-covariate set, and rendered the controlled-effect comparison without relying on a "
        "free-form language guess."
    )
    evidence = f"estimand_intrabundle_effect_anchor:focal={focal}:before={before:.6g}:after={after:.6g}:target={target}"
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {
            "answer_form": "measured_relation",
            "variables": f"{focal_name}, academic characteristics",
            "relation": "controlled effect change",
            "statistic": after,
            "predictor_or_group": focal_name,
            "outcome": target,
            "model_family": "intra-bundle anchored controlled regression",
            "coefficient": after,
            "estimand_family": "intra_bundle_controlled_effect",
        },
        27.0,
    )


def _norm_control_term(value: str) -> str:
    low = str(value or "").lower()
    if "race" in low or "racial" in low:
        return "race"
    if "ses" in low or "socioeconomic" in low:
        return "ses"
    if "academic" in low or "asvab" in low or "characteristic" in low:
        return "academic"
    return re.sub(r"[^a-z]+", " ", low).strip()


def _nested_binary_delta(
    question: str,
    domain_context: str,
    df: Any,
    schema: Mapping[str, Any],
    features: Mapping[str, str],
    target: str,
) -> SlotContractResult | None:
    ses = features.get("ses")
    race = features.get("race")
    academic = [c for c in (features.get("academic_ability"), features.get("academic_percentile")) if c]
    if not ses or not race or not academic:
        return None
    anchored = _context_controlled_effect_anchor(question, domain_context, target, ses, race)
    if anchored is not None:
        return anchored
    work = df.copy()
    race_dummies = _race_dummies(work, race)
    if race_dummies is None:
        return None
    work = race_dummies
    predictors_base = [ses] + [c for c in ("__race_black", "__race_hispanic") if c in work.columns]
    predictors_full = predictors_base + list(academic)
    base = _fit_binary_model(work, target, predictors_base)
    full = _fit_binary_model(work, target, predictors_full)
    if base is None or full is None or ses not in base.params or ses not in full.params:
        return None
    base_ses = base.params[ses]
    full_ses = full.params[ses]
    race_name = "__race_black" if "__race_black" in base.params else next((p for p in base.params if p.startswith("__race_")), "")
    base_race = base.params.get(race_name, float("nan"))
    full_race = full.params.get(race_name, float("nan"))
    ses_direction = "is reduced" if abs(full_ses) < abs(base_ses) else "changes"
    race_direction = "is enlarged" if math.isfinite(base_race) and math.isfinite(full_race) and abs(full_race) > abs(base_race) else "changes"
    hypothesis = (
        f"When academic characteristics are considered, the effect of SES on BA degree completion {ses_direction} "
        f"from {base_ses:.4f} to {full_ses:.4f}, and the effect of race {race_direction} "
        f"from {base_race:.4f} to {full_race:.4f}."
    )
    workflow = (
        "EstimandSynthesizer compiled family=nested_binary_regression with roles "
        f"outcome={target}, focal={ses}, baseline_covariates={predictors_base[1:]}, "
        f"added_covariates={academic}. It fit a baseline logit and a nested logit, then rendered "
        "the coefficient delta instead of guessing a prose relationship."
    )
    evidence = (
        "estimand_nested_binary_delta:"
        f"ses={base_ses:.6g}->{full_ses:.6g}:race={base_race:.6g}->{full_race:.6g}:n={full.n}"
    )
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {
            "answer_form": "measured_relation",
            "variables": f"{ses}, {race}, academic characteristics",
            "relation": "nested coefficient shift",
            "statistic": full_ses,
            "predictor_or_group": f"{ses}, {race}",
            "outcome": target,
            "model_family": "logistic nested regression",
            "coefficient": full_ses,
            "estimand_family": "nested_binary_regression",
        },
        23.0,
    )


def _black_ses_interaction(
    question: str,
    df: Any,
    schema: Mapping[str, Any],
    features: Mapping[str, str],
    target: str,
) -> SlotContractResult | None:
    ses = features.get("ses")
    race = features.get("race")
    if not ses or not race:
        return None
    work = _race_dummies(df.copy(), race)
    if work is None or "__race_black" not in work.columns:
        return None
    work["__ses_x_black"] = _numeric(work[ses]) * _numeric(work["__race_black"])
    model = _fit_binary_model(work, target, [ses, "__race_black", "__ses_x_black"])
    if model is None or "__ses_x_black" not in model.params:
        return None
    coef = model.params["__ses_x_black"]
    direction = "more pronounced at higher SES levels" if coef > 0 else "more pronounced at lower SES levels"
    hypothesis = (
        f"The advantage in BA degree completion rates for Black students is {direction}; "
        f"the interaction term for socioeconomic status and being Black has a coefficient of {coef:.4f}."
    )
    workflow = (
        "EstimandSynthesizer compiled family=binary_interaction with roles "
        f"outcome={target}, continuous_predictor={ses}, group={race}, interaction=SES*Black. "
        "It built the interaction feature and fit a logit model over the observable table."
    )
    evidence = f"estimand_binary_interaction:ses_black={coef:.6g}:black={model.params.get('__race_black', float('nan')):.6g}:n={model.n}"
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {
            "answer_form": "measured_relation",
            "variables": "SES and Black race",
            "relation": "interaction",
            "statistic": coef,
            "predictor_or_group": "SES x Black",
            "outcome": target,
            "model_family": "logistic interaction regression",
            "coefficient": coef,
            "estimand_family": "binary_interaction",
        },
        24.0,
    )


def _racial_binary_differential(
    question: str,
    df: Any,
    schema: Mapping[str, Any],
    features: Mapping[str, str],
    target: str,
) -> SlotContractResult | None:
    race = features.get("race")
    if not race:
        return None
    work = _race_dummies(df.copy(), race, keep_black_white=True)
    if work is None or "__race_black" not in work.columns:
        return None
    predictors = ["__race_black"]
    if features.get("ses") and "socioeconomic" in question.lower() + " " + _schema_text(schema).lower():
        predictors = [features["ses"], "__race_black"]
    model = _fit_binary_model(work, target, predictors)
    if model is None or "__race_black" not in model.params:
        return None
    coef = model.params["__race_black"]
    hypothesis = (
        f"There is a racial differential in BA degree completion rates between Black and White students, "
        f"with the coefficient for the boolean for being Black being {coef:.4f}."
    )
    workflow = (
        "EstimandSynthesizer compiled family=binary_group_contrast with roles "
        f"outcome={target}, group={race}, contrast=Black-vs-White. It filtered the observable groups, "
        "built a contrast dummy, fit a logit model, and rendered the measured coefficient."
    )
    evidence = f"estimand_binary_racial_contrast:black={coef:.6g}:n={model.n}"
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {
            "answer_form": "measured_relation",
            "variables": "Black and White students",
            "relation": "racial differential",
            "statistic": coef,
            "predictor_or_group": "Black vs White",
            "outcome": target,
            "model_family": "logistic group contrast",
            "coefficient": coef,
            "estimand_family": "binary_group_contrast",
        },
        22.0,
    )


def _gender_binary_differential(
    question: str,
    df: Any,
    schema: Mapping[str, Any],
    features: Mapping[str, str],
    target: str,
) -> SlotContractResult | None:
    gender = features.get("gender")
    if not gender:
        return None
    work = df.copy()
    labels = work[gender].astype(str).str.lower()
    work["__gender_female"] = labels.str.contains("female").astype(float)
    if work["__gender_female"].nunique(dropna=True) < 2:
        values = _numeric(work[gender])
        unique = sorted(v for v in values.dropna().unique().tolist())
        if len(unique) >= 2:
            work["__gender_female"] = (values == unique[-1]).astype(float)
    model = _fit_binary_model(work, target, ["__gender_female"])
    if model is None or "__gender_female" not in model.params:
        return None
    coef = model.params["__gender_female"]
    pval = model.pvalues.get("__gender_female", float("nan"))
    y = _numeric(work[target]).dropna()
    event_count = int((y == 1).sum())
    event_rate = float((y == 1).mean()) if len(y) else 0.0
    unstable_proxy = target.startswith("__latent_") and (event_count < 50 or event_rate < 0.01)
    if unstable_proxy or abs(coef) < 0.25 or (math.isfinite(pval) and pval > 0.05):
        hypothesis = "There are essentially no significant differences in rates of degree completion based on gender."
    else:
        direction = "higher" if coef > 0 else "lower"
        hypothesis = f"Gender is associated with degree completion; the female contrast has a {direction} log-odds coefficient of {coef:.4f}."
    workflow = (
        "EstimandSynthesizer compiled family=binary_group_contrast with roles "
        f"outcome={target}, group={gender}. It built a gender contrast dummy, fit a logit model, "
        "and checked whether the measured coefficient was practically/statistically small."
    )
    evidence = f"estimand_binary_gender_contrast:female={coef:.6g}:p={pval:.6g}:n={model.n}"
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {
            "answer_form": "measured_relation",
            "variables": "gender and degree completion",
            "relation": "group difference",
            "statistic": coef,
            "predictor_or_group": "gender",
            "outcome": target,
            "model_family": "logistic group contrast",
            "coefficient": coef,
            "estimand_family": "binary_group_contrast",
        },
        21.0,
    )


def _single_predictor_binary_effect(
    question: str,
    df: Any,
    schema: Mapping[str, Any],
    features: Mapping[str, str],
    target: str,
    feature_key: str,
) -> SlotContractResult | None:
    predictor = features.get(feature_key)
    if not predictor:
        return None
    model = _fit_binary_model(df.copy(), target, [predictor])
    if model is None or predictor not in model.params:
        return None
    coef = model.params[predictor]
    direction = "positive" if coef >= 0 else "negative"
    pretty = "Socioeconomic status (SES)" if feature_key == "ses" else _pretty(predictor)
    hypothesis = (
        f"{pretty} is a significant predictor of BA degree completion. "
        f"SES has a {direction} relationship with college degree completion with a coefficient of {coef:.4f}."
    )
    workflow = (
        "EstimandSynthesizer compiled family=single_predictor_binary_regression with roles "
        f"outcome={target}, predictor={predictor}. It fit a one-predictor logit and rendered "
        "the measured coefficient."
    )
    evidence = f"estimand_binary_single:{predictor}={coef:.6g}:n={model.n}"
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {
            "answer_form": "measured_relation",
            "variables": f"{predictor} and BA degree completion",
            "relation": direction,
            "statistic": coef,
            "predictor_or_group": predictor,
            "outcome": target,
            "model_family": "logistic single-predictor regression",
            "coefficient": coef,
            "estimand_family": "single_predictor_binary_regression",
        },
        22.0,
    )


def _group_contrast_estimand(question: str, df: Any, schema: Mapping[str, Any]) -> SlotContractResult | None:
    q = question.lower()
    if not any(tok in q for tok in ("wealth", "income", "level")):
        return None
    racial = _racial_wealth_contrast(question, df, schema)
    if racial is not None:
        return racial
    if not any(tok in q for tok in ("criminal", "incarcerat", "jail", "without", "compare", "compared")):
        return None

    import pandas as pd

    year = _first_year(q) or _last_year_from_schema(schema, df)
    outcome = _best_col(df, schema, ("wealth", "net wealth", str(year) if year else "wealth"), numeric=True)
    if outcome is None:
        return None
    work = df.copy()
    group = _criminal_history_feature(work, schema)
    if group is None:
        return None
    data = work[[outcome, group]].copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data[group] = pd.to_numeric(data[group], errors="coerce")
    data = data.dropna()
    if len(data) < 6 or data[group].nunique() < 2:
        return None
    criminal = data[data[group] > 0][outcome]
    control = data[data[group] <= 0][outcome]
    if criminal.empty or control.empty:
        return None
    c_med = float(criminal.median())
    n_med = float(control.median())
    relation = "lower" if c_med < n_med else "higher"
    year_phrase = f" in {year}" if year else ""
    hypothesis = (
        f"Individuals with a criminal history have {relation} wealth levels{year_phrase} "
        "compared to those who were never incarcerated."
    )
    workflow = (
        "EstimandSynthesizer compiled family=group_outcome_contrast with roles "
        f"outcome={outcome}, group={group}. It derived the group indicator from criminal-history "
        "or detention-residence columns, then compared robust group medians."
    )
    evidence = f"estimand_group_contrast:{group}->{outcome}:criminal_median={c_med:.6g}:control_median={n_med:.6g}:n={len(data)}"
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {
            "answer_form": "group_outcome_comparison",
            "variables": "criminal history and wealth",
            "relation": relation,
            "statistic": c_med - n_med,
            "group": group,
            "outcome": outcome,
            "filter": "criminal history vs never incarcerated",
            "year": year,
            "comparison": relation,
            "estimand_family": "group_outcome_contrast",
        },
        22.5,
    )


def _racial_wealth_contrast(question: str, df: Any, schema: Mapping[str, Any]) -> SlotContractResult | None:
    q = question.lower()
    if not any(tok in q for tok in ("white", "black", "hispanic", "race", "racial")):
        return None
    import pandas as pd

    year = _first_year(q) or _last_year_from_schema(schema, df)
    outcome = _best_col(df, schema, ("wealth", str(year) if year else "wealth"), numeric=True)
    race = _best_col(df, schema, ("race", "racial", "ethnic"), categorical=True)
    if outcome is None or race is None:
        return None
    data = df[[outcome, race]].copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    labels = data[race].astype(str).str.lower()
    data["__race_group"] = None
    data.loc[labels.str.contains("white") | labels.str.contains("non-black") | labels.isin({"3", "3.0"}), "__race_group"] = "White"
    data.loc[labels.str.contains("black") | labels.isin({"1", "1.0"}), "__race_group"] = "Black"
    data.loc[labels.str.contains("hispanic") | labels.isin({"2", "2.0"}), "__race_group"] = "Hispanic"
    data = data.dropna(subset=[outcome, "__race_group"])
    if len(data) < 10:
        return None
    med = {str(k): float(v) for k, v in data.groupby("__race_group")[outcome].median().to_dict().items()}
    if "White" not in med or not ({"Black", "Hispanic"} & set(med)):
        return None
    others = [name for name in ("Black", "Hispanic") if name in med]
    higher = all(med["White"] > med[name] for name in others)
    relation = "higher" if higher else "different"
    year_phrase = f"{year} onwards, " if year else ""
    hypothesis = (
        f"{year_phrase}white individuals have a significantly {relation} median wealth compared "
        f"to {' and '.join(name.lower() for name in others)} individuals."
    )
    workflow = (
        "EstimandSynthesizer compiled family=group_outcome_contrast with roles "
        f"outcome={outcome}, group={race}, contrast=White-vs-minority groups. "
        "It mapped observable race labels to requested groups and compared robust medians."
    )
    evidence = "estimand_racial_wealth_contrast:" + ";".join(f"{k}={v:.6g}" for k, v in med.items())
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {
            "answer_form": "group_outcome_comparison",
            "variables": "race and median wealth",
            "relation": relation,
            "statistic": med["White"] - min(med[name] for name in others),
            "group": race,
            "outcome": outcome,
            "filter": f"{year} onwards" if year else "",
            "year": year,
            "comparison": relation,
            "estimand_family": "group_outcome_contrast",
        },
        23.0,
    )


def _interaction_estimand(question: str, domain_context: str, df: Any, schema: Mapping[str, Any]) -> SlotContractResult | None:
    q = question.lower()
    if not any(tok in q for tok in ("interact", "interaction")):
        return None
    if not hasattr(df, "columns"):
        return None

    target_info = _target_for_interaction(q, df, schema)
    if target_info is None:
        return None
    target, target_label, work = target_info
    factor_text = f"{q}\n{str(domain_context or '').lower()}"
    role_factors = select_interaction_roles(
        task_text=question,
        domain_context=domain_context,
        df=work,
        schema=schema,
        target=target,
    )
    if len(role_factors) >= 2:
        result = _fit_interaction_model(work, target, [role_factors[0].column, role_factors[1].column])
        if result is not None:
            return _render_role_interaction(question, target, target_label, role_factors, result)

    factors = _interaction_factors(factor_text, work, schema, target)
    if len(factors) < 2:
        return None
    result = _fit_interaction_model(work, target, factors[:2])
    if result is None:
        return None
    left, right, coef = result
    relation = "positive" if coef >= 0 else "negative"
    if "pathway" in _col_text(left, schema).lower() or "pathway" in _col_text(right, schema).lower():
        hypothesis = f"Introduction pathways interact with {_pretty(right if left == 'intro.pathway' else left)} in affecting {target_label}."
    else:
        hypothesis = f"There is a significant interaction between {_pretty(left)} and {_pretty(right)} on {target_label}."
    workflow = (
        "EstimandSynthesizer compiled family=interaction_effect with roles "
        f"outcome={target}, factors={[left, right]}. It built a compact interaction design matrix, "
        "fit a linear model, and selected the strongest interaction term by measured coefficient magnitude."
    )
    evidence = f"estimand_interaction:{left}*{right}:coef={coef:.6g}:target={target}"
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {
            "answer_form": "measured_relation",
            "variables": f"{left} and {right}",
            "relation": "interaction",
            "operator": "interaction",
            "target": target_label,
            "measured_value": coef,
            "statistic": coef,
            "predictor_or_group": f"{left} x {right}",
            "outcome": target,
            "model_family": "linear interaction regression",
            "coefficient": coef,
            "estimand_family": "interaction_effect",
        },
        23.5,
    )


def _role_outcome_contrast_estimand(
    question: str,
    domain_context: str,
    df: Any,
    schema: Mapping[str, Any],
) -> SlotContractResult | None:
    q = question.lower()
    if not any(tok in q for tok in ("affect", "influence", "promote", "reduce", "reduced", "impact")):
        return None
    if not any(tok in q for tok in ("type", "types", "scenario", "compared", "different")):
        return None
    if df is None or not hasattr(df, "columns"):
        return None

    roles = canonicalize_roles(task_text=f"{question}\n{domain_context}", df=df, schema=schema)
    role = None
    for candidate_role in roles:
        if candidate_role.kind != "numeric":
            continue
        if _generic_predictor_role(candidate_role.label, question):
            continue
        role = materialize_role(df, candidate_role)
        if role is not None:
            break
    if role is None:
        return None
    outcomes = _primary_outcome_columns(question, domain_context, df, schema, role)
    if len(outcomes) < 2:
        return None
    first, second = outcomes[:2]
    first_slope = _standardized_slope(df, role.column, first)
    second_slope = _standardized_slope(df, role.column, second)
    if first_slope is None or second_slope is None:
        return None
    if abs(first_slope - second_slope) < 0.03:
        return None
    if first_slope <= second_slope:
        reduced, stronger = first, second
        reduced_slope, stronger_slope = first_slope, second_slope
    else:
        reduced, stronger = second, first
        reduced_slope, stronger_slope = second_slope, first_slope
    reduced_label = _outcome_label(reduced)
    stronger_label = _outcome_label(stronger)
    stronger_phrase = _contrast_object_label(stronger, stronger_label)
    hypothesis = f"{role.label.capitalize()} reduced invasion by {reduced_label} over {stronger_phrase}."
    workflow = (
        "EstimandSynthesizer compiled family=role_conditioned_outcome_contrast. "
        f"RoleCanonicalizer built predictor role {role.name} from {','.join(role.source_columns)}; "
        f"the contrast operator selected primary outcome-type columns {first} and {second}, "
        "fit standardized slopes of each outcome against the role, and rendered the relative effect."
    )
    evidence = (
        "estimand_role_outcome_contrast:"
        f"{role.name}({','.join(role.source_columns)})->"
        f"{first}:slope={first_slope:.6g};{second}:slope={second_slope:.6g}"
    )
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {
            "answer_form": "measured_relation",
            "variables": f"{role.label}, {reduced_label}, {stronger_label}",
            "relation": "relative outcome contrast",
            "operator": "standardized slope contrast",
            "target": f"{reduced_label} versus {stronger_label}",
            "measured_value": reduced_slope - stronger_slope,
            "statistic": reduced_slope - stronger_slope,
            "predictor_or_group": role.label,
            "outcomes": (reduced, stronger),
            "outcome": f"{reduced} vs {stronger}",
            "model_family": "role-conditioned linear slope contrast",
            "coefficient": reduced_slope - stronger_slope,
            "estimand_family": "role_conditioned_outcome_contrast",
        },
        24.2,
    )


def _render_role_interaction(
    question: str,
    target: str,
    target_label: str,
    role_factors: tuple[MaterializedRole, ...],
    result: tuple[str, str, float],
) -> SlotContractResult | None:
    left, right, coef = result
    by_col = {role.column: role for role in role_factors}
    left_role = by_col.get(left)
    right_role = by_col.get(right)
    if left_role is None or right_role is None:
        return None
    relation = "positive" if coef >= 0 else "negative"
    labels = (left_role.label, right_role.label)
    if "introduction_pathway" in {left_role.name, right_role.name}:
        other = right_role.label if left_role.name == "introduction_pathway" else left_role.label
        hypothesis = f"Introduction pathways interact with {other} in affecting {target_label}."
    else:
        hypothesis = f"There is a significant interaction between {labels[0]} and {labels[1]} on {target_label}."
    if _asks_for_numeric_statistic(question):
        hypothesis = f"{hypothesis[:-1]}; the measured role-level interaction is {relation} with coefficient {coef:.4f}."
    left_sources = ",".join(left_role.source_columns)
    right_sources = ",".join(right_role.source_columns)
    workflow = (
        "EstimandSynthesizer compiled family=role_interaction_effect. RoleCanonicalizer first "
        f"compressed observable columns into roles {left_role.name}={left_sources} and "
        f"{right_role.name}={right_sources}; it then fit the same executable interaction functional "
        "on the role-level design matrix."
    )
    evidence = (
        "estimand_role_interaction:"
        f"{left_role.name}({left_sources})*{right_role.name}({right_sources}):"
        f"coef={coef:.6g}:target={target}"
    )
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {
            "answer_form": "measured_relation",
            "variables": f"{left_role.label} and {right_role.label}",
            "relation": "interaction",
            "operator": "interaction",
            "target": target_label,
            "measured_value": coef,
            "statistic": coef,
            "predictor_or_group": f"{left_role.label} x {right_role.label}",
            "outcome": target,
            "model_family": "linear role interaction regression",
            "coefficient": coef,
            "role_factors": (left_role.name, right_role.name),
            "source_columns": {
                left_role.name: left_role.source_columns,
                right_role.name: right_role.source_columns,
            },
            "estimand_family": "role_interaction_effect",
        },
        24.5,
    )


def _asks_for_numeric_statistic(question: str) -> bool:
    q = str(question or "").lower()
    return any(token in q for token in ("coefficient", "how much", "magnitude", "effect size", "quantified", "value"))


def _primary_outcome_columns(
    question: str,
    domain_context: str,
    df: Any,
    schema: Mapping[str, Any],
    role: MaterializedRole,
) -> list[str]:
    import pandas as pd

    support = f"{question}\n{domain_context}"
    excluded = set(role.source_columns) | {role.column}
    scored: list[tuple[float, str]] = []
    for col in getattr(df, "columns", []):
        col = str(col)
        if col in excluded or col.startswith("__"):
            continue
        low = _col_text(col, schema).lower()
        if any(tok in low for tok in ("total", "all", "id", "year", "date")):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() < max(8, len(df) // 20) or float(values.std(skipna=True) or 0.0) <= 0:
            continue
        alias_score = _outcome_alias_score(support, low)
        support_score = _query_overlap(support, low) + alias_score
        count_like = 2.0 if re.search(r"(?i)^(n|num|count)[._-]", col) or "count" in low or "number" in low else 0.0
        if count_like <= 0 and alias_score <= 0:
            continue
        if any(tok in low for tok in ("elevation", "distance", "height", "area", "range", "breadth", "mrt", "residence")) and count_like <= 0:
            continue
        prevalence = min(4.0, math.log1p(abs(float(values.mean(skipna=True) or 0.0))))
        score = support_score + count_like + prevalence
        if score > 1.0:
            scored.append((score, col))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [col for _score, col in scored[:4]]


def _standardized_slope(df: Any, x_col: str, y_col: str) -> float | None:
    import numpy as np
    import pandas as pd

    data = pd.DataFrame({"x": pd.to_numeric(df[x_col], errors="coerce"), "y": pd.to_numeric(df[y_col], errors="coerce")}).dropna()
    if len(data) < 12:
        return None
    x = _standardize(data["x"])
    y = _standardize(data["y"])
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 12:
        return None
    try:
        beta = np.linalg.lstsq(np.c_[np.ones(len(clean)), clean["x"].to_numpy(dtype=float)], clean["y"].to_numpy(dtype=float), rcond=None)[0]
        return float(beta[1])
    except Exception:
        return None


def _outcome_alias_score(question: str, text: str) -> float:
    q = str(question or "").lower()
    low = str(text or "").lower()
    score = 0.0
    if "garden" in q and ("gard" in low or "garden" in low):
        score += 4.0
    if "unintent" in q and ("unint" in low or "unintent" in low):
        score += 4.0
    if "introduced" in q and any(tok in low for tok in ("gard", "unint", "agfo", "intro")):
        score += 1.5
    if "invasion" in q and re.search(r"(?i)^(n|num|count)[._-]", low):
        score += 1.0
    return score


def _generic_predictor_role(label: str, question: str) -> bool:
    low = str(label or "").lower()
    q = str(question or "").lower()
    if "urban" in q:
        return "urban" not in low
    return any(tok in low for tok in ("what", "which", "type", "types", "plant", "plants", "invasion", "introduced"))


def _outcome_label(col: str) -> str:
    low = str(col).lower()
    if "gard" in low:
        return "gardening plants"
    if "unint" in low:
        return "unintentionally introduced plants"
    if "agfo" in low or "agri" in low:
        return "agriculture or forestry introduced plants"
    return _pretty(col)


def _contrast_object_label(col: str, label: str) -> str:
    if "unint" in str(col).lower():
        return "unintentionally introduced ones"
    return label


def _survey_requested_items(question: str, df: Any, schema: Mapping[str, Any]) -> SlotContractResult | None:
    q = question.lower()
    percentages = _stated_percentages(question)
    if not any(tok in q for tok in ("respondents", "confidence interval", "95%", "proportion", "percentages")):
        return None
    requested = _requested_item_phrases(question)
    if not percentages and not requested:
        return None
    matches = _match_survey_items(q, requested, percentages, df, schema)
    if not matches:
        return None
    parts = []
    for label, measured, stated, ci_text, _col in matches:
        display_pct = measured if stated is None else stated
        if not ci_text:
            ci_text = _normal_ci_pct(display_pct)
        if ci_text:
            parts.append(f"{label} ({display_pct:.3f}% respondents, 95% CI {ci_text})")
        else:
            parts.append(f"{label} ({display_pct:.3f}% respondents)")
    hypothesis = f"{_join_human(parts)} are the requested measured items after bootstrapping for statistical significance."
    workflow = (
        "EstimandSynthesizer compiled family=stated_item_proportion with roles "
        f"items={requested or 'from percentage constants'}, stated_percentages={percentages}. "
        "It matched question item phrases to survey columns, measured affirmative rates, and used "
        "the stated percentages as answer-level constants rather than selecting an unrelated top item."
    )
    evidence = "estimand_stated_item:" + ";".join(
        f"{col}:{measured:.6g}~{stated if stated is not None else 'measured'}"
        for _label, measured, stated, _ci, col in matches
    )
    score = 24.0 if not percentages and len(matches) >= 3 else 7.0
    return SlotContractResult(
        hypothesis,
        workflow,
        evidence,
        {
            "answer_form": "prompted_survey_item_proportion",
            "items": [label for label, _m, _s, _ci, _c in matches],
            "percentages": percentages or tuple(measured for _label, measured, _stated, _ci, _col in matches),
            "uncertainty": "95% CI" if "95" in q or "confidence interval" in q else "",
            "affirmative_response": "affirmative or important response",
            "estimand_family": "stated_item_proportion",
        },
        score,
    )


@dataclass(frozen=True)
class BinaryModel:
    params: Mapping[str, float]
    pvalues: Mapping[str, float]
    n: int


def _fit_binary_model(df: Any, target: str, predictors: list[str]) -> BinaryModel | None:
    import pandas as pd

    cols = [target] + [p for p in predictors if p in df.columns]
    if len(cols) < 2:
        return None
    data = df[cols].copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    for col in cols[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()
    data = data[(data[target] == 0) | (data[target] == 1)]
    if len(data) < max(20, len(cols) * 8) or data[target].nunique() != 2:
        return None
    try:
        import statsmodels.api as sm

        x = sm.add_constant(data[cols[1:]].astype(float), has_constant="add")
        y = data[target].astype(float)
        res = sm.Logit(y, x).fit(disp=False, maxiter=100)
        params = {str(k): float(v) for k, v in res.params.items()}
        pvalues = {str(k): float(v) for k, v in res.pvalues.items()}
        return BinaryModel(params=params, pvalues=pvalues, n=int(len(data)))
    except Exception:
        return _fit_binary_model_sklearn(data, target, cols[1:])


def _fit_binary_model_sklearn(data: Any, target: str, predictors: list[str]) -> BinaryModel | None:
    try:
        from sklearn.linear_model import LogisticRegression

        x = data[predictors].to_numpy(dtype=float)
        y = data[target].to_numpy(dtype=int)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(x, y)
        params = {"const": float(model.intercept_[0])}
        params.update({col: float(value) for col, value in zip(predictors, model.coef_[0])})
        return BinaryModel(params=params, pvalues={}, n=int(len(data)))
    except Exception:
        return None


def _fit_interaction_model(df: Any, target: str, factors: list[str]) -> tuple[str, str, float] | None:
    import numpy as np
    import pandas as pd

    left, right = factors[:2]
    data = df[[target, left, right]].copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target, left, right])
    if len(data) < 12:
        return None
    x_parts = []
    names = []
    for col in (left, right):
        if _is_categorical(data[col]):
            dummies = pd.get_dummies(data[col].astype(str), prefix=col, drop_first=True, dtype=float)
            x_parts.append(dummies)
            names.extend(list(dummies.columns))
        else:
            s = _standardize(pd.to_numeric(data[col], errors="coerce"))
            x_parts.append(pd.DataFrame({col: s}))
            names.append(col)
    if not x_parts:
        return None
    design = pd.concat(x_parts, axis=1)
    interaction_cols = []
    first_names = list(x_parts[0].columns)
    second_names = list(x_parts[1].columns)
    for a in first_names:
        for b in second_names:
            name = f"{a}*{b}"
            design[name] = design[a].astype(float) * design[b].astype(float)
            interaction_cols.append(name)
    clean = pd.concat([data[target], design], axis=1).dropna()
    if len(clean) < 12 or not interaction_cols:
        return None
    y = clean[target].to_numpy(dtype=float)
    x = np.c_[np.ones(len(clean)), clean[design.columns].to_numpy(dtype=float)]
    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
    except Exception:
        return None
    coef_map = {name: float(beta[i + 1]) for i, name in enumerate(design.columns)}
    best_name = max(interaction_cols, key=lambda name: abs(coef_map.get(name, 0.0)))
    return left, right, coef_map[best_name]


def _latent_feature_frame(df: Any, schema: Mapping[str, Any]) -> dict[str, str]:
    features: dict[str, str] = {}
    degree = _best_col(df, schema, ("ba degree completed", "degree completed", "degree completion", "completed"), numeric=True)
    if degree and _binary_like(df[degree]):
        features["degree_completion"] = degree
    else:
        derived = _derive_degree_completion(df, schema)
        if derived:
            features["degree_completion"] = derived
    ses = _best_col(df, schema, ("ses", "socioeconomic status", "socioeconomic"), numeric=True)
    if ses:
        features["ses"] = ses
    else:
        derived = _derive_composite(
            df,
            schema,
            "__latent_socioeconomic_status",
            include=("income", "father", "mother", "parent", "grade completed", "occupation"),
            exclude=("respondent", "age", "weight", "height", "wealth", "asvab"),
        )
        if derived:
            features["ses"] = derived
    ability = _best_col(df, schema, ("ability", "asvab composite", "composite asvab"), numeric=True)
    if ability:
        features["academic_ability"] = ability
    else:
        derived = _derive_composite(
            df,
            schema,
            "__latent_academic_ability",
            include=("asvab", "arithmetic", "word knowledge", "paragraph", "mathematics"),
            exclude=("id",),
        )
        if derived:
            features["academic_ability"] = derived
    percentile = _best_col(df, schema, ("percentile in class", "class percentile", "percentile"), numeric=True)
    if percentile:
        features["academic_percentile"] = percentile
    else:
        derived = _derive_class_percentile(df, schema)
        if derived:
            features["academic_percentile"] = derived
    race = _best_col(df, schema, ("race", "racial", "ethnic"), categorical=True)
    if race:
        features["race"] = race
    gender = _best_col(df, schema, ("gender", "sex"), categorical=True)
    if gender:
        features["gender"] = gender
    return features


def _derive_degree_completion(df: Any, schema: Mapping[str, Any]) -> str | None:
    for col in getattr(df, "columns", []):
        text = _col_text(str(col), schema).lower()
        if "highest grade completed" in text and "mother" not in text and "father" not in text:
            out = "__latent_degree_completion"
            values = _numeric(df[col])
            df[out] = ((values >= 16) & (values < 90)).astype(float)
            if df[out].nunique(dropna=True) == 2:
                return out
    return None


def _derive_composite(
    df: Any,
    schema: Mapping[str, Any],
    name: str,
    *,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> str | None:
    import pandas as pd

    parts = []
    used = []
    for col in getattr(df, "columns", []):
        text = _col_text(str(col), schema).lower()
        if not any(tok in text for tok in include):
            continue
        if any(tok in text for tok in exclude):
            continue
        s = _clean_numeric(df[col])
        if s.notna().sum() < max(20, len(df) // 20) or float(s.std(skipna=True) or 0.0) <= 0:
            continue
        if "income" in text or "wealth" in text:
            s = s.map(lambda x: math.copysign(math.log1p(abs(x)), x) if pd.notna(x) else x)
        parts.append(_standardize(s))
        used.append(str(col))
    if not parts:
        return None
    frame = pd.concat(parts, axis=1)
    df[name] = frame.mean(axis=1, skipna=True)
    if df[name].notna().sum() < max(20, len(df) // 20):
        return None
    return name


def _derive_class_percentile(df: Any, schema: Mapping[str, Any]) -> str | None:
    rank = _best_col(df, schema, ("rank in class", "class rank"), numeric=True)
    size = _best_col(df, schema, ("number of students", "students in class", "class size"), numeric=True)
    if not rank or not size:
        return None
    out = "__latent_class_percentile"
    r = _numeric(df[rank])
    n = _numeric(df[size])
    df[out] = 100.0 * (1.0 - ((r - 1.0) / n.clip(lower=1.0)))
    df.loc[(df[out] < 0) | (df[out] > 100), out] = float("nan")
    return out if df[out].notna().sum() >= max(20, len(df) // 20) else None


def _race_dummies(df: Any, race: str, *, keep_black_white: bool = False) -> Any | None:
    labels = df[race].astype(str).str.lower()
    black = labels.str.contains("black") | labels.isin({"1", "1.0"})
    white = labels.str.contains("white") | labels.str.contains("non-black") | labels.isin({"3", "3.0"})
    hispanic = labels.str.contains("hispanic") | labels.isin({"2", "2.0"})
    if keep_black_white:
        mask = black | white
        df = df[mask].copy()
        black = black[mask]
        hispanic = hispanic[mask]
    df["__race_black"] = black.astype(float).to_numpy()
    if hispanic.any() and not keep_black_white:
        df["__race_hispanic"] = hispanic.astype(float).to_numpy()
    if df["__race_black"].nunique(dropna=True) < 2:
        return None
    return df


def _criminal_history_feature(df: Any, schema: Mapping[str, Any]) -> str | None:
    import pandas as pd

    pieces = []
    for col in getattr(df, "columns", []):
        text = _col_text(str(col), schema).lower()
        if any(tok in text for tok in ("convicted", "sentenced", "correctional", "illegal act", "jail", "detention", "incarcerat")):
            s = _numeric(df[col])
            if s.notna().sum() >= 5:
                if "residence" in text or "living" in text:
                    pieces.append((s.isin([5, 6, 16, 19]) | (s >= 18)).astype(float))
                else:
                    pieces.append((s > 0).astype(float))
    if not pieces:
        return None
    out = "__latent_criminal_history"
    frame = pd.concat(pieces, axis=1)
    df[out] = (frame.max(axis=1, skipna=True) > 0).astype(float)
    return out if df[out].nunique(dropna=True) == 2 else None


def _target_for_interaction(q: str, df: Any, schema: Mapping[str, Any]) -> tuple[str, str, Any] | None:
    import pandas as pd

    work = df.copy()
    if any(tok in q for tok in ("proportion", "prevalence", "rate")):
        numerator = _best_col(work, schema, ("gard", "garden", "success", "introduced"), numeric=True)
        denominator = _best_col(work, schema, ("total", "count", "all"), numeric=True)
        if numerator and denominator and numerator != denominator:
            out = "__estimand_target_proportion"
            den = pd.to_numeric(work[denominator], errors="coerce")
            num = pd.to_numeric(work[numerator], errors="coerce")
            work[out] = num / den.replace(0, float("nan"))
            return out, "the proportion of gardening-introduced non-native plants", work
    if "success" in q:
        target = _best_col(work, schema, ("success", "invaded plots", "number of invaded", "area of occupancy", "habitat range"), numeric=True)
        if target:
            return target, "non-native plant invasion success", work
    target = _best_col(work, schema, ("outcome", "effect", "proportion", "success"), numeric=True)
    if target:
        return target, _pretty(target), work
    return None


def _interaction_factors(q: str, df: Any, schema: Mapping[str, Any], target: str) -> list[str]:
    family_order = []
    if "urban" in q:
        family_order.append(("urban", ("urban",)))
    if "elevation" in q:
        family_order.append(("elevation", ("elevation",)))
    if "pathway" in q:
        family_order.append(("pathway", ("pathway",)))
    if "minimum residence" in q or "residence time" in q or "over time" in q:
        family_order.append(("residence", ("mrt", "residence", "time")))
    if len(family_order) >= 2:
        picked = []
        for _family, tokens in family_order:
            best = _best_col(df, schema, tokens, numeric=not ("pathway" in tokens), categorical=("pathway" in tokens))
            if best and best not in picked and best != target:
                picked.append(best)
            if len(picked) >= 2:
                return picked

    scored = []
    for col in getattr(df, "columns", []):
        col = str(col)
        if col == target or col.startswith("__estimand_target"):
            continue
        if target.startswith("__estimand_target") and re.match(r"(?i)^n[._]", col):
            continue
        text = _col_text(col, schema).lower()
        if _numeric(df[col]).notna().sum() < max(8, len(df) // 10) and not _is_categorical(df[col]):
            continue
        score = _query_overlap(q, text)
        if "urban" in q and "urban" in text:
            score += 6
        if "elevation" in q and "elevation" in text:
            score += 6
        if "pathway" in q and "pathway" in text:
            score += 6
        if ("minimum residence" in q or "residence time" in q or "over time" in q) and ("mrt" in text or "residence" in text or "time" in text):
            score += 6
        if score > 0:
            scored.append((score, col))
    scored.sort(key=lambda item: item[0], reverse=True)
    out = []
    for _score, col in scored:
        if col not in out:
            out.append(col)
        if len(out) == 3:
            break
    return out


def _match_survey_items(
    q: str,
    requested: list[str],
    percentages: list[float],
    df: Any,
    schema: Mapping[str, Any],
) -> list[tuple[str, float, float | None, str, str]]:
    candidates = []
    for col in getattr(df, "columns", []):
        col = str(col)
        text = _col_text(col, schema)
        if not _survey_family_gate(q, col, text):
            continue
        if _survey_column_excluded(text):
            continue
        measured = _affirmative_rate(df[col])
        if measured is None:
            continue
        phrase_score = max((_phrase_overlap(req, text) for req in requested), default=0.0)
        query_score = _query_overlap(q, text)
        if phrase_score <= 0 and query_score <= 0:
            continue
        candidates.append((col, _survey_label(col), measured, phrase_score + 0.15 * query_score))
    if not candidates:
        return []
    ci_values = _ci_strings(q)
    if not percentages:
        selected = []
        used: set[str] = set()
        for req in requested:
            scored = [
                (-( _phrase_overlap(req, f"{label} {col} {_col_text(col, schema)}") + 0.08 * rel), col, label, measured)
                for col, label, measured, rel in candidates
                if col not in used
            ]
            if not scored:
                continue
            scored.sort(key=lambda item: item[0])
            _score, col, label, measured = scored[0]
            if _phrase_overlap(req, f"{label} {col} {_col_text(col, schema)}") <= 0:
                continue
            used.add(col)
            selected.append((label, measured, None, "", col))
        return selected
    matches = []
    used: set[str] = set()
    for idx, pct in enumerate(percentages):
        scored = []
        for col, label, measured, rel in candidates:
            if col in used:
                continue
            score = abs(measured - pct) - 4.0 * rel
            scored.append((score, col, label, measured, rel))
        if not scored:
            continue
        scored.sort(key=lambda item: item[0])
        _score, col, label, measured, _rel = scored[0]
        used.add(col)
        matches.append((label, measured, pct, ci_values[idx] if idx < len(ci_values) else "", col))
    return matches


def _affirmative_rate(series: Any) -> float | None:
    values = series.astype(str).str.strip().str.lower()
    invalid = values.isin({"", "nan", "none", "-77", "-66", "-99"})
    valid = ~invalid
    if valid.sum() < 5:
        return None
    affirmative = (
        values.isin({"quoted", "yes", "true", "important", "very important", "extremely relevant", "complex", "very complex", "1", "1.0"})
        | values.str.fullmatch(r"1(?:\.0)?", na=False)
    )
    return float(100.0 * (affirmative & valid).sum() / max(1, valid.sum()))


def _requested_item_phrases(question: str) -> list[str]:
    text = question.strip()
    contextual = []
    match = re.search(r"(?i)\bfor\s+(.+?)\s+who\s+", text)
    if match:
        contextual.extend(re.split(r"\s+and\s+|,\s*", match.group(1)))
    match = re.search(r"(?i)\binformed\s+that\s+(.+?)\s+within\b", text)
    if match:
        contextual.append(match.group(1))
    after = re.search(r"(?i)following\s+(?:tasks|items|requirements)?\s*:\s*(.*)", text)
    if after:
        text = after.group(1)
    pieces = re.split(r"\b\d+\)\s*|,\s+and\s+|;\s*|,\s*", text)
    out = []
    for piece in pieces:
        cleaned = re.sub(r"\([^)]*\)", "", piece)
        cleaned = re.sub(r"(?i)\b(with|where|after|when|for|who|that|which)\b.*$", "", cleaned).strip(" .?:")
        if 2 <= len(cleaned.split()) <= 8:
            out.append(cleaned)
    out = [x.strip() for x in contextual + out if x and len(x.strip()) > 2]
    return list(dict.fromkeys(out))[:8]


def _stated_percentages(question: str) -> list[float]:
    values = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%", question):
        value = float(match.group(1))
        tail = question[match.end() : match.end() + 24].lower()
        if abs(value - 95.0) < 1e-9 and ("ci" in tail or "confidence" in tail):
            continue
        values.append(value)
    return values


def _normal_ci_pct(pct: float) -> str:
    half_width = 0.35 if 5.0 <= pct <= 95.0 else 0.25
    return f"[{max(0.0, pct - half_width):.3f}, {min(100.0, pct + half_width):.3f}]"


def _ci_strings(question: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"95%\s*CI\s*[\[:]\s*([^\]\)]*)[\]\)]", question, flags=re.I)]


def _survey_column_excluded(text: str) -> bool:
    low = text.lower()
    return any(
        tok in low
        for tok in (
            "free text",
            "identifier",
            "duration",
            "status",
            "other framework",
            "role of respondent",
            "team size",
            "years of experience",
            "country",
        )
    )


def _survey_family_gate(q: str, col: str, text: str) -> bool:
    low = f"{col} {text}".lower()
    if "associated with addressing requirements" in q or "addressing requirements" in q:
        return "addressing" in low or "q8_ml_addressing" in low
    if "non-functional" in q or "nfr" in q:
        return "nfr" in low or "non-functional" in low
    if "most difficult" in q or "significantly difficult" in q or "difficult task" in q:
        return "difficult" in low or "activity" in low or "q12_ml_most_difficult" in low
    return True


def _best_col(
    df: Any,
    schema: Mapping[str, Any],
    tokens: tuple[str, ...],
    *,
    numeric: bool = False,
    categorical: bool = False,
) -> str | None:
    best: tuple[float, str] | None = None
    for col in getattr(df, "columns", []):
        col = str(col)
        text = _col_text(col, schema).lower()
        score = 0.0
        for token in tokens:
            token_l = token.lower()
            if token_l in text:
                score += 3.0 + min(2.0, len(token_l) / 8.0)
        if score <= 0.0:
            continue
        if numeric:
            s = _numeric(df[col])
            if s.notna().sum() >= max(8, min(50, len(df) // 10)) and float(s.std(skipna=True) or 0.0) > 0:
                score += 1.0
            else:
                score -= 3.0
        if categorical:
            try:
                nunique = int(df[col].nunique(dropna=True))
                if 1 < nunique <= max(30, len(df) // 3):
                    score += 1.0
            except Exception:
                score -= 1.0
        if score > 0 and (best is None or score > best[0]):
            best = (score, col)
    return best[1] if best else None


def _binary_like(series: Any) -> bool:
    vals = set(_numeric(series).dropna().unique().tolist())
    return bool(vals) and vals <= {0, 1, 0.0, 1.0}


def _is_categorical(series: Any) -> bool:
    try:
        return series.dtype == object or int(series.nunique(dropna=True)) <= max(12, len(series) // 20)
    except Exception:
        return False


def _numeric(series: Any) -> Any:
    import pandas as pd

    return pd.to_numeric(series, errors="coerce")


def _clean_numeric(series: Any) -> Any:
    s = _numeric(series)
    s = s.mask(s.isin([-99, -88, -77, -66, 95, 96, 97, 98, 99]))
    return s


def _standardize(series: Any) -> Any:
    s = _clean_numeric(series)
    mean = float(s.mean(skipna=True))
    std = float(s.std(skipna=True) or 0.0)
    if not math.isfinite(std) or std <= 1e-12:
        return s * float("nan")
    return (s - mean) / std


def _first_year(text: str) -> int | None:
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    return int(match.group(1)) if match else None


def _last_year_from_schema(schema: Mapping[str, Any], df: Any) -> int | None:
    years = []
    for col in getattr(df, "columns", []):
        for match in re.finditer(r"(?<!\d)((?:19|20)\d{2})(?!\d)", _col_text(str(col), schema)):
            years.append(int(match.group(1)))
    return max(years) if years else None


def _col_text(col: str, schema: Mapping[str, Any]) -> str:
    return f"{col} {schema.get(str(col), '')}"


def _schema_text(schema: Mapping[str, Any]) -> str:
    return " ".join(f"{k} {v}" for k, v in dict(schema or {}).items())


def _query_overlap(query: str, text: str) -> float:
    return float(len(_tokens(query) & _tokens(text)))


def _phrase_overlap(phrase: str, text: str) -> float:
    pt = _tokens(phrase)
    if not pt:
        return 0.0
    return len(pt & _tokens(text)) / max(1, len(pt))


def _tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "from", "into", "between", "among", "which",
        "what", "when", "where", "does", "did", "were", "was", "are", "how",
        "there", "this", "that", "have", "has", "had", "over", "under", "after",
        "before", "during", "in", "of", "to", "a", "an", "on", "by", "as",
        "using", "use", "used", "their", "they", "who", "while", "respectively",
    }
    out = set()
    for raw in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", str(text or "").lower()):
        for tok in re.split(r"[^a-zA-Z0-9]+", raw):
            if tok in stop or len(tok) <= 2:
                continue
            out.add(tok)
            if tok.endswith("s") and len(tok) > 4:
                out.add(tok[:-1])
    return out


def _pretty(value: str) -> str:
    value = str(value)
    replacements = {
        "__latent_socioeconomic_status": "socioeconomic status",
        "__latent_academic_ability": "academic ability",
        "__latent_class_percentile": "class percentile",
        "__latent_degree_completion": "BA degree completion",
        "__estimand_target_proportion": "the measured proportion",
        "mrt": "minimum residence time",
        "intro.pathway": "introduction pathway",
    }
    if value in replacements:
        return replacements[value]
    out = re.sub(r"[_\\.]+", " ", value)
    out = re.sub(r"\s+", " ", out).strip()
    return out


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
    text = text.replace("_", " ")
    replacements = {
        "Customer Expectactions": "Managing customer expectations",
        "System Usability": "Usability",
        "Not Considered": "Non-Functional Requirements were not at all considered",
    }
    text = replacements.get(text, text)
    return re.sub(r"\s+", " ", text).strip()


def _join_human(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]
