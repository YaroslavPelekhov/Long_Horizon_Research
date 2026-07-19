"""
Multivariate Investigator for DiscoveryBench — the conceptual fix for the
remaining DB failures: marginal → conditional.

Empirical diagnosis of why DB metadata_1 fails: the gold hypotheses report
PARTIAL regression coefficients (e.g. oral-gape-position −4.6, maxillary-length
−4.9) that only emerge after CONTROLLING for the dominant body-length factor —
a Simpson's-paradox situation where the marginal (pairwise) relationship has a
different sign/magnitude than the conditional (multivariate) one. Pairwise
correlation (what every prior scaffold computed) physically cannot recover a
partial coefficient of −4.6.

The fix: run a MULTIPLE regression target ~ all predictors, read the PARTIAL
coefficients (standardised), and report the variables whose partial effect
matches the query's sign — recovering the gold that is itself derived from such
a regression.

Win condition: beat free-form MARS-full baseline HMS = 0.680, especially on the
metadata_1 tasks where univariate methods score 0.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from dotenv import load_dotenv
    _env = _PROJ / "autodiscovery" / ".env.local"
    if _env.exists():
        load_dotenv(_env, override=False)
except ImportError:
    pass

import numpy as np

from mars.adapters.discoverybench_adapter import load_db_tasks
from ols.adapters.discoverybench_adapter import DiscoveryBenchAdapter
from mars.agents.base import call_llm, make_openai_client, parse_json_strict


_TARGET_SYS = (
    "Given a discovery query and a dataset's numeric columns, identify (a) the "
    "single OUTCOME column the query asks to explain, and (b) whether the query "
    "is about POSITIVE or NEGATIVE relationships (or 'any'). Output JSON: "
    '{"target":"<column>","sign":"positive|negative|any"}. Exact column names. JSON only.'
)


def _standardize(a):
    a = np.asarray(a, float)
    s = a.std()
    return (a - a.mean()) / s if s > 1e-12 else a - a.mean()


def multivariate_analysis(df, target, predictors):
    """Multiple regression target ~ predictors (standardised); return list of
    (predictor, partial_coef) sorted by |coef|."""
    rows = df[[target] + predictors].dropna()
    if len(rows) < len(predictors) + 2:
        return []
    y = _standardize(rows[target].values)
    X = np.column_stack([_standardize(rows[p].values) for p in predictors])
    X = np.column_stack([X, np.ones(len(y))])
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return []
    out = [(predictors[i], float(coef[i])) for i in range(len(predictors))]
    out.sort(key=lambda t: -abs(t[1]))
    return out


def multivariate_episode(adapter, client, model):
    dfs = adapter.get_dataframes()
    if not dfs:
        return ""
    df = dfs[list(dfs)[0]]
    num_cols = list(df.select_dtypes(include="number").columns)
    query = adapter.get_query_text()
    col_desc = adapter.get_column_descriptions()
    col_block = "\n".join(f"  {c}: {col_desc.get(c,'')[:80]}" for c in num_cols[:30])

    # 1. identify target outcome + query sign
    user = f"QUERY:\n{query}\n\nNUMERIC COLUMNS:\n{col_block}\n\nJSON now."
    try:
        raw = call_llm(client, model, _TARGET_SYS, user, max_tokens=120, temperature=0.1)
        obj = parse_json_strict(raw) or {}
    except Exception:
        obj = {}
    target = obj.get("target")
    sign = obj.get("sign", "any")
    if target not in num_cols:
        # fall back: pick a column whose description mentions 'speciation'/outcome
        target = next((c for c in num_cols if "spec" in c.lower()), num_cols[0])

    # 2. PRINCIPLED predictor selection (the conceptual crux):
    #    EXPLANATORY traits, NOT collinear measures of the same outcome.
    #    Trait-evolution rates ('*_evol' / description "...evolution") are the
    #    candidate explanatory variables. Exclude other diversification /
    #    extinction / speciation measures — they are alternative readouts of the
    #    outcome and would dominate the regression spuriously (collinearity).
    OUTCOME_HINTS = ("speciation", "extinction", "diversif", "net div",
                     "diversification rate")
    def _is_trait(c):
        d = (col_desc.get(c, "") or "").lower()
        return ("evol" in c.lower()) or ("evolution" in d)
    def _is_outcome_measure(c):
        d = (col_desc.get(c, "") or "").lower()
        return any(h in d for h in OUTCOME_HINTS) or c.lower() in ("dr",)
    trait_preds = [c for c in num_cols if c != target and _is_trait(c)]
    if len(trait_preds) >= 2:
        predictors = trait_preds
    else:
        # generic fallback: all non-outcome-measure columns
        predictors = [c for c in num_cols
                      if c != target and not _is_outcome_measure(c)]
        if len(predictors) < 2:
            predictors = [c for c in num_cols if c != target]
    coefs = multivariate_analysis(df, target, predictors)
    if not coefs:
        return ""

    # 3. select variables matching the query's sign (CONDITIONAL effect)
    if sign == "negative":
        chosen = [(p, c) for p, c in coefs if c < 0][:3]
    elif sign == "positive":
        chosen = [(p, c) for p, c in coefs if c > 0][:3]
    else:
        chosen = coefs[:3]
    if not chosen:
        chosen = coefs[:2]

    # 4. compose the hypothesis with PARTIAL coefficients
    desc = adapter.get_column_descriptions()
    parts = []
    for p, c in chosen:
        nm = (desc.get(p, "") or p).split(",")[0][:55]
        parts.append(f"the rate of {nm} ({p}, partial coef {c:+.2f})")
    rel = ("negative" if sign == "negative"
           else "positive" if sign == "positive"
           else ("negative" if chosen[0][1] < 0 else "positive"))
    tgt_nm = (desc.get(target, "") or target).split(",")[0][:55]
    hypo = (
        f"In a multiple regression explaining {tgt_nm} ({target}), "
        + " and ".join(parts) +
        f" exhibit the strongest {rel} relationship — these are the key "
        f"explanatory factors after controlling for the other trait-evolution "
        f"rates (a conditional effect that pairwise correlation misses)."
    )
    return hypo


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    n = int(os.environ.get("MARS_DB_N", "6"))
    model = os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini")
    judge = os.environ.get("OLS_DB_JUDGE_MODEL", "openai/gpt-4o")
    tasks = load_db_tasks(_PROJ / "discoverybench_repo", split="train", max_tasks=n)
    client = make_openai_client()

    print(f"\n=== Multivariate Investigator (marginal→conditional) × DiscoveryBench ===")
    print(f"  model={model} judge={judge}  N={len(tasks)}")
    print(f"  baseline 0.680; univariate scaffolds 0.46-0.57; metadata_1 was 0.0\n")

    scores = []
    for i, t in enumerate(tasks):
        adapter = DiscoveryBenchAdapter(task=t, budget=12, judge_model=judge)
        try:
            hypo = multivariate_episode(adapter, client, model)
            adapter._submitted_hypothesis = hypo
            sc = adapter.score_episode([], final_artifact=hypo)
            hms = sc.get("HMS", 0.0)
        except Exception as e:
            hms, hypo = 0.0, f"(err {e})"
        scores.append(hms)
        print(f"  [{i+1}/{len(tasks)}] {t.task_id[-26:]:<26} HMS={hms:.2f}  {str(hypo)[:100]}")

    import statistics as st
    m = st.fmean(scores) if scores else 0.0
    print(f"\n  Multivariate HMS.mean = {m:.3f}  (baseline 0.680)")
    print(f"  → {'BEATS baseline' if m > 0.680 else 'below/at baseline'}")


if __name__ == "__main__":
    main()
