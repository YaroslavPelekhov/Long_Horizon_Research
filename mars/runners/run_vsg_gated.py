"""Margin-gated VSG — solve only where a FAITHFUL verifier exists, else ABSTAIN.

Combines: per-stage data verification (VSG) + verifier-validation (is the verifier
faithful or spurious?) + an abstention gate. For each task we compute the
faithfulness margin of the variable-selection verifier (V_grounded = relevance ×
fold-stability, vs adversarial low-relevance-high-corr negatives). If margin > 0
a faithful verifier exists -> solve and answer; else abstain (don't guess).

Tests honest universality: precision on ANSWERED tasks should be far above the
ungated baseline, because the system only claims where it can verify.

  python -m mars.runners.run_vsg_gated --run_id vsg_gated_v1
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_PROJ = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJ, _PROJ / "discoverybench_repo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=False)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.runners.run_db_official_eval import _run_official_eval, load_official_real_tasks
from mars.runners.run_verifier_validator import (
    _corr, _fold_stable_abs, _largest_csv, relevance_scores,
)

WEAK = "openai/gpt-4o-mini"
MARGIN_THRESH = 0.05
TASKS = [
    "archaeology:0:0", "archaeology:1:0",
    "introduction_pathways_non-native_plants:0:0", "introduction_pathways_non-native_plants:0:1",
    "meta_regression:0:0", "meta_regression:0:1",
    "nls_incarceration:0:0", "nls_incarceration:0:1",
    "nls_ses:0:0", "nls_ses:0:1",
    "requirements_engineering_for_ML_enabled_systems:0:0",
    "requirements_engineering_for_ML_enabled_systems:1:0",
    "worldbank_education_gdp:0:0", "worldbank_education_gdp:0:1",
]


def _ctx_discriminates(df, x, y):
    """Find a categorical subpopulation where corr(x,y) is stronger than complement."""
    best = ("the overall population", 0.0, None)
    for c in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 12]:
        for v in df[c].dropna().unique()[:8]:
            mask = df[c].astype(str) == str(v)
            sub, comp = df[mask], df[~mask]
            if len(sub) < 6 or len(comp) < 6:
                continue
            ci, co = abs(_corr(sub, x, y)), abs(_corr(comp, x, y))
            if ci > 0.2 and ci - co > best[1] and ci - co > 0.1:
                best = (f"the subpopulation where {c} = {v}", ci - co, 1 if _corr(sub, x, y) > 0 else -1)
    return best


def solve_gated(client, task):
    df = _largest_csv(task)
    if df is None:
        return {"answer": None, "reason": "no csv", "margin": None}
    q = task.task.query
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:10]
    pairs = list(itertools.combinations(num, 2))[:18]
    if len(pairs) < 3:
        return {"answer": None, "reason": "too few numeric pairs", "margin": None}
    col_desc = "\n".join(f"  {c}" for c in num)
    rel = relevance_scores(client, q, col_desc, pairs)
    rows = []
    for (x, y), r in zip(pairs, rel):
        stab = _fold_stable_abs(df, x, y)
        rows.append({"x": x, "y": y, "rel": r, "stab": stab,
                     "V_naive": stab, "V_grounded": r * stab})
    target = max(rows, key=lambda r: r["V_grounded"])
    negs = [r for r in rows if r["rel"] <= 0.34 and r["stab"] >= 0.3] or \
           sorted([r for r in rows if r["rel"] <= 0.5], key=lambda r: -r["stab"])[:3]
    margin = target["V_grounded"] - max((r["V_grounded"] for r in negs), default=0.0)

    if margin <= MARGIN_THRESH:
        return {"answer": None, "reason": f"abstain: verifier margin={margin:.3f} (no faithful verifier)",
                "margin": round(margin, 3), "target": (target["x"], target["y"])}

    # faithful verifier exists -> compose answer from verified stages
    x, y = target["x"], target["y"]
    sign = 1 if _corr(df, x, y) > 0 else -1
    ctx_desc, _, ctx_sign = _ctx_discriminates(df, x, y)
    if ctx_sign:
        sign = ctx_sign
    rel_word = "positively associated with" if sign > 0 else "negatively associated with"
    hypo = call_llm(client, model=WEAK,
        system="Write one precise scientific hypothesis from ONLY the verified facts. No new claims.",
        user=(f"Question: {q}\nVerified facts:\n- context: {ctx_desc}\n- variables: {x}, {y}\n"
              f"- relation: {x} is {rel_word} {y}\nWrite a single-sentence hypothesis."),
        max_tokens=120, temperature=0.2).strip()
    return {"answer": hypo, "margin": round(margin, 3), "vars": (x, y),
            "context": ctx_desc, "relation": rel_word}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="vsg_gated_v1")
    ap.add_argument("--judge", default="openai/gpt-4-turbo")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "vsg_gated" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    tasks = {t.task_key: t for t in load_official_real_tasks(
        _PROJ / "discoverybench_repo", max_tasks=None, datasets=None, task_keys=set(args.tasks))}
    trace_f = io.StringIO()
    results, answered_hms = {}, []
    n_answer = n_abstain = 0
    print("=== Margin-gated VSG: solve only where verifier is faithful, else abstain ===\n")
    for key in args.tasks:
        t = tasks.get(key)
        if not t:
            continue
        r = solve_gated(client, t)
        if r["answer"] is None:
            n_abstain += 1
            results[key] = {"status": "ABSTAIN", **r}
            print(f"  {key:48} ABSTAIN  ({r['reason']})")
        else:
            ev = _run_official_eval(t, pred_hypo=r["answer"], pred_workflow="",
                                    judge_model=args.judge, trace_f=trace_f)
            hms = 100.0 * float(ev.get("final_score", 0.0))
            answered_hms.append(hms)
            n_answer += 1
            results[key] = {"status": "ANSWER", "HMS": round(hms, 1), **r}
            print(f"  {key:48} ANSWER HMS={hms:5.1f} margin={r['margin']} | {r['answer'][:55]}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_answer + n_abstain
    prec = sum(answered_hms) / len(answered_hms) if answered_hms else 0.0
    overall = sum(answered_hms) / n if n else 0.0          # abstain counts as 0 if forced to answer
    print(f"\n=== RESULT ===")
    print(f"  coverage: answered {n_answer}/{n}  abstained {n_abstain}/{n}")
    print(f"  PRECISION on answered  : mean HMS = {prec:.2f}")
    print(f"  if forced (abstain=0)  : mean HMS = {overall:.2f}")
    print(f"  ungated baseline (all answered, no gate): mini~14, 4o~15")
    print(f"  → honest universality: only claim where verifiable; precision >> baseline = robust.")
    summary = {"run_id": args.run_id, "margin_thresh": MARGIN_THRESH,
               "answered": n_answer, "abstained": n_abstain,
               "precision_answered": round(prec, 2), "forced_mean": round(overall, 2),
               "results": results}
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
