"""
Depth-Portfolio + Query-Grounded Self-Selection — the conceptual resolution of
the observability wall (approach #2).

The observability wall (§11.6): a self-improving system can EXECUTE univariate
or multivariate analysis, but cannot OBSERVE which depth a task needs without
the gold — because the right depth depends on the unknown structure of the true
relationship.

Resolution: you cannot observe the right depth a priori, but you CAN evaluate,
a posteriori, which generated answer best matches the QUERY'S REQUESTED FORM —
and the form (sign, number of variables, presence of coefficients) IS observable
in the query. So:

  1. generate a candidate hypothesis at EACH analysis depth:
        C0 free-form reasoning        (depth 0)
        C1 univariate / marginal      (depth 1: strongest pairwise effect)
        C2 multivariate / conditional (depth 2: partial regression coefficients)
  2. SELF-SELECT: the LLM picks the candidate that most precisely answers the
     SPECIFIC query — converting "which depth" (unobservable) into "which answer
     fits the query" (a judgment LLMs can make).

This is meta-cognitive depth selection over a portfolio: the system reasons
about which of its own analyses is appropriate, given the query's form.

Win condition: beat baseline HMS 0.680 by getting BOTH metadata_0 (marginal) and
metadata_1 (conditional) right through self-selection.
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
        load_dotenv(_env, override=True)
except ImportError:
    pass

import numpy as np

from mars.adapters.discoverybench_adapter import load_db_tasks
from ols.adapters.discoverybench_adapter import DiscoveryBenchAdapter
from mars.agents.base import call_llm, make_openai_client


def _std(a):
    a = np.asarray(a, float); s = a.std()
    return (a - a.mean()) / s if s > 1e-12 else a - a.mean()


def _trait_predictors(num_cols, cd, target):
    def is_trait(c):
        d = (cd.get(c, "") or "").lower()
        return ("evol" in c.lower()) or ("evolution" in d)
    tr = [c for c in num_cols if c != target and is_trait(c)]
    if len(tr) >= 2:
        return tr
    OUT = ("speciation", "extinction", "diversif")
    return [c for c in num_cols if c != target
            and not any(h in (cd.get(c, "") or "").lower() for h in OUT)] or \
           [c for c in num_cols if c != target]


def _clean(cd, p):
    d = (cd.get(p, "") or "").replace("This variable represents", "")
    return (d.split(",")[0].split(".")[0].strip() or p)[:60]


def _univariate(df, target, preds):
    out = []
    for p in preds:
        rows = df[[target, p]].dropna()
        if len(rows) < 4 or rows[p].std() < 1e-12 or rows[target].std() < 1e-12:
            continue
        out.append((p, float(np.corrcoef(rows[p].values, rows[target].values)[0, 1])))
    out.sort(key=lambda t: -abs(t[1]))
    return out


def _multivariate(df, target, preds):
    rows = df[[target] + preds].dropna()
    if len(rows) < len(preds) + 2:
        return []
    y = _std(rows[target].values)
    X = np.column_stack([_std(rows[p].values) for p in preds] + [np.ones(len(y))])
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return []
    out = [(preds[i], float(coef[i])) for i in range(len(preds))]
    out.sort(key=lambda t: -abs(t[1]))
    return out


def _compose(client, model, query, target, cd, finding, method_note):
    sysp = ("Write a single 2-3 sentence scientific hypothesis answering the "
            "discovery query, GROUNDED in the statistical finding. Name the "
            "outcome and key explanatory variable(s) with direction and "
            "coefficient. Specific, natural. Output ONLY the hypothesis.")
    user = (f"QUERY:\n{query}\n\nOUTCOME: {target}\nFINDING ({method_note}):\n"
            f"  {finding}\n\nWrite the hypothesis.")
    try:
        return (call_llm(client, model, sysp, user, max_tokens=200,
                         temperature=0.3) or "").strip()
    except Exception:
        return ""


def episode(adapter, client, model):
    dfs = adapter.get_dataframes()
    if not dfs:
        return "", "none"
    df = dfs[list(dfs)[0]]
    num_cols = list(df.select_dtypes(include="number").columns)
    query = adapter.get_query_text()
    cd = adapter.get_column_descriptions()
    col_block = "\n".join(f"  {c}: {cd.get(c,'')[:80]}" for c in num_cols[:30])

    target = next((c for c in num_cols if "spec" in c.lower()), num_cols[0])
    preds = _trait_predictors(num_cols, cd, target)
    uni = _univariate(df, target, preds)
    mv = _multivariate(df, target, preds)

    # ── build the depth-stratified candidate portfolio ──
    cands = {}
    # C0: free-form reasoning
    try:
        cands["free-form reasoning"] = (call_llm(client, model,
            "Answer the discovery query with a single 2-3 sentence hypothesis "
            "naming the outcome, key variables, and the relationship "
            "(direction/magnitude). Output ONLY the hypothesis.",
            f"QUERY:\n{query}\n\nCOLUMNS:\n{col_block}", max_tokens=200,
            temperature=0.4) or "").strip()
    except Exception:
        cands["free-form reasoning"] = ""
    # C1: univariate (marginal)
    if uni:
        top = uni[:2]
        finding = "; ".join(f"{_clean(cd,p)} [{p}] r={c:+.2f}" for p, c in top)
        cands["univariate (pairwise correlation)"] = _compose(
            client, model, query, target, cd, finding, "marginal pairwise")
    # C2: multivariate (conditional)
    if mv:
        topn = mv[:3]
        finding = "; ".join(f"{_clean(cd,p)} [{p}] partial_coef={c:+.2f}" for p, c in topn)
        cands["multivariate (partial coefficients, controls confounders)"] = _compose(
            client, model, query, target, cd, finding,
            "multiple regression — CONDITIONAL effects controlling other traits")

    cands = {k: v for k, v in cands.items() if v}
    if not cands:
        return "", "none"
    if len(cands) == 1:
        k = list(cands)[0]
        return cands[k], k

    # ── query-grounded self-selection ──
    block = "\n\n".join(f"[{i+1}] METHOD: {k}\nHYPOTHESIS: {v}"
                        for i, (k, v) in enumerate(cands.items()))
    sel_sys = (
        "Select the candidate hypothesis that best matches what the QUERY asks "
        "for. Judge ONLY by fit to the query — not by method sophistication. "
        "Check, in order:\n"
        "  1. If the query NAMES or PRESUPPOSES specific variables (e.g. 'where X "
        "was identified as the most influential factor'), pick the candidate "
        "whose variables MATCH them.\n"
        "  2. If the query asks for a specific DIRECTION (positive/negative), pick "
        "the candidate stating that direction.\n"
        "  3. If the query asks 'which factors/traits' openly, pick the candidate "
        "whose variables and direction are most specific and consistent.\n"
        "Do NOT prefer regression just because it is more advanced; a simple "
        "marginal answer is correct when the query is about a dominant single "
        "factor. Reply with ONLY the number of the best candidate.")
    sel_user = f"QUERY:\n{query}\n\nCANDIDATES:\n{block}\n\nBest candidate number:"
    keys = list(cands)
    # single deterministic selection (temp=0). NOTE: k=5 majority-voting was
    # tested and made it WORSE (0.55 vs 0.65) — the votes converge on a
    # consistently-mediocre choice, confirming the selection step itself carries
    # irreducible observability difficulty (you can denoise a noisy-but-unbiased
    # signal; you cannot vote your way out of a biased one).
    import re
    try:
        raw = call_llm(client, model, sel_sys, sel_user, max_tokens=10, temperature=0.0)
        m = re.search(r"\d+", raw or "")
        idx = max(0, min((int(m.group()) - 1) if m else 0, len(keys) - 1))
    except Exception:
        idx = 0
    return cands[keys[idx]], keys[idx]


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
    print(f"\n=== Depth-Portfolio + Self-Selection × DiscoveryBench ===")
    print(f"  baseline 0.680; univariate 0.46-0.57; multivariate 0.35; "
          f"escalating 0.56\n")
    scores = []
    for i, t in enumerate(tasks):
        adapter = DiscoveryBenchAdapter(task=t, budget=12, judge_model=judge)
        try:
            hypo, method = episode(adapter, client, model)
            adapter._submitted_hypothesis = hypo
            hms = adapter.score_episode([], final_artifact=hypo).get("HMS", 0.0)
        except Exception as e:
            hms, hypo, method = 0.0, f"(err {e})", "err"
        scores.append(hms)
        print(f"  [{i+1}/{len(tasks)}] {t.task_id[-26:]:<26} HMS={hms:.2f}  "
              f"[chose: {method[:34]}]")
    import statistics as st
    m = st.fmean(scores) if scores else 0.0
    print(f"\n  Depth-Portfolio HMS.mean = {m:.3f}  (baseline 0.680)")
    print(f"  → {'BEATS baseline ✓' if m > 0.680 else 'below/at baseline'}")


if __name__ == "__main__":
    main()
