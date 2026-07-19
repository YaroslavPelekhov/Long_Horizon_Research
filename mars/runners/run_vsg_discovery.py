"""VSG — Verified Staged Generation on real DiscoveryBench.

The missing piece: make EVERY generation stage verifiable by a small program on
the data, so a weak model stays robust. A scientific claim = (context, variables,
relation). We generate it in stages; the weak model only PROPOSES candidates per
stage; the DATA verifies each stage (correlation stability across folds,
in-slice vs out-slice discrimination). Unverified candidates are dropped.

Verification moves from the final answer (HMS, no oracle) to the STAGES (the CSV
verifies them). Directly attacks the dominant WRONG-CONTEXT failure: the context
stage SEARCHES for the subpopulation where the relation is strongest, verified.

  python -m mars.runners.run_vsg_discovery --run_id vsg_v1 --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import io
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

DEFAULT_TASKS = [
    "meta_regression:0:0", "meta_regression:0:1",
    "nls_incarceration:0:0", "nls_incarceration:0:1",
    "nls_ses:0:0", "nls_ses:0:1",
    "requirements_engineering_for_ML_enabled_systems:0:0",
    "requirements_engineering_for_ML_enabled_systems:1:0",
    "worldbank_education_gdp:0:0", "worldbank_education_gdp:0:1",
]


# ---------------------------------------------------------------------------
# Small executable verifiers (the "micro-programs") — pure pandas, no LLM
# ---------------------------------------------------------------------------

def _folds(df, k=3, seed=0):
    idx = np.array(df.index)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    return [df.loc[idx[i::k]] for i in range(k)]


def _corr(df, x, y):
    try:
        s = df[[x, y]].dropna()
        if len(s) < 4:
            return None
        c = s[x].astype(float).corr(s[y].astype(float))
        return None if c != c else float(c)
    except Exception:
        return None


def verify_pair(df, x, y, seed=0):
    """Micro-program: |corr(x,y)| stable across folds. Returns (mean_abs, sign, ok)."""
    cs = [_corr(f, x, y) for f in _folds(df, 3, seed)]
    cs = [c for c in cs if c is not None]
    if len(cs) < 2:
        return 0.0, 0, False
    mean = float(np.mean(cs))
    stable = all((c > 0) == (mean > 0) for c in cs)        # sign-consistent across folds
    return abs(mean), (1 if mean > 0 else -1), (abs(mean) > 0.1 and stable)


def verify_context(df, x, y, mask):
    """Micro-program: relation STRONGER inside the slice than its complement,
    and present. Returns (corr_in, corr_out, discriminates)."""
    sub, comp = df[mask], df[~mask]
    if len(sub) < 6 or len(sub) == len(df):
        return None, None, False
    ci, co = _corr(sub, x, y), _corr(comp, x, y)
    if ci is None:
        return None, None, False
    co = co if co is not None else 0.0
    discr = abs(ci) > 0.15 and abs(ci) > abs(co) + 0.1     # stronger inside than outside
    return ci, co, discr


# ---------------------------------------------------------------------------
# Stage candidate proposal (weak model PROPOSES; data verifies)
# ---------------------------------------------------------------------------

def _ask_json(client, model, prompt, max_tokens=500):
    raw = call_llm(client, model=model, system="Return only valid JSON.",
                   user=prompt, max_tokens=max_tokens, temperature=0.3)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:])
        t = t.split("```")[0]
    try:
        return json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return {}


def propose_pairs(client, model, question, col_desc, numeric_cols):
    obj = _ask_json(client, model,
        f"Question: {question}\nNumeric columns:\n{col_desc}\n\n"
        f"Propose up to 6 candidate (x,y) numeric column pairs whose relationship "
        f"could answer the question. Use EXACT column names.\n"
        f'Return JSON: {{"pairs": [["x","y"], ...]}}')
    out = []
    for p in obj.get("pairs", []):
        if isinstance(p, list) and len(p) == 2 and p[0] in numeric_cols and p[1] in numeric_cols and p[0] != p[1]:
            out.append((p[0], p[1]))
    return out[:6]


def propose_filters(client, model, question, df, cat_cols):
    vals = {c: [str(v) for v in df[c].dropna().unique()[:8]] for c in cat_cols}
    obj = _ask_json(client, model,
        f"Question: {question}\nCategorical columns and their values:\n{json.dumps(vals)[:1500]}\n\n"
        f"Which subpopulation(s) might the answer be specific to? Propose up to 6 "
        f"candidate filters as (column, value) pairs. Use EXACT names/values.\n"
        f'Return JSON: {{"filters": [["col","value"], ...]}}')
    out = []
    for f in obj.get("filters", []):
        if isinstance(f, list) and len(f) == 2 and f[0] in cat_cols:
            out.append((f[0], f[1]))
    return out[:6]


# ---------------------------------------------------------------------------
# VSG solve: staged, each stage data-verified
# ---------------------------------------------------------------------------

def _is_identifier(df, c):
    """Generic data hygiene: drop id/year/index-like columns (not real variables)."""
    name = str(c).lower()
    if any(k in name for k in ("year", "yr", " id", "id ", "code", "index", "_id", "serial")):
        return True
    s = df[c].dropna()
    if len(s) and s.nunique() / len(s) > 0.9:          # near-unique → identifier
        return True
    try:                                                # integer-valued in a year range
        v = s.astype(float)
        if ((v >= 1850) & (v <= 2100)).mean() > 0.8 and (v == v.round()).mean() > 0.95:
            return True
    except Exception:
        pass
    return False


def vsg_solve(client, model, df, question, col_desc):
    trace = []
    numeric_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c]) and not _is_identifier(df, c)]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
                and df[c].nunique() <= 20]
    col_desc = "\n".join(f"  {c} ({df[c].dtype})" for c in numeric_cols[:30])

    # Stage 1: variables — model proposes in RELEVANCE order; data only REJECTS
    # unstable ones; we keep the first relevant pair that verifies (relevance + robustness).
    pairs = propose_pairs(client, model, question, col_desc, numeric_cols)
    chosen = None
    for (x, y) in pairs:                                # preserve model's relevance order
        mag, sign, ok = verify_pair(df, x, y)
        if ok:
            chosen = (mag, sign, x, y)
            break
    if chosen is None:
        return None, ["no relevant variable pair passed fold-stability verification"]
    mag, sign, x, y = chosen
    trace.append(f"variables VERIFIED: ({x},{y}) |corr|={mag:.2f} sign={'+' if sign>0 else '-'}")

    # Stage 2: context (proposed by model, verified by in-slice vs out-slice discrimination)
    filters = propose_filters(client, model, question, df, cat_cols)
    ctx_desc, ctx_sign = "the overall population", sign
    best_discr = 0.0
    for (col, val) in filters:
        mask = df[col].astype(str) == str(val)
        ci, co, discr = verify_context(df, x, y, mask)
        if discr and abs(ci) - abs(co or 0) > best_discr:
            best_discr = abs(ci) - abs(co or 0)
            ctx_desc = f"the subpopulation where {col} = {val}"
            ctx_sign = 1 if ci > 0 else -1
            trace.append(f"context VERIFIED: {col}={val} corr_in={ci:.2f} corr_out={(co or 0):.2f} (discriminates)")
    if best_discr == 0.0:
        trace.append("context VERIFIED: overall population (no subpopulation discriminated)")

    # Stage 3: relation (verified sign)
    rel = "positively associated with" if ctx_sign > 0 else "negatively associated with"
    trace.append(f"relation VERIFIED: {rel}")

    # Compose hypothesis from VERIFIED facts (model verbalizes, adds no new claims)
    hypo = call_llm(client, model=model,
        system="Write one precise scientific hypothesis using ONLY the given verified facts. No new claims.",
        user=(f"Question: {question}\nVerified facts:\n"
              f"- context: {ctx_desc}\n- variables: {x} and {y}\n- relation: {x} is {rel} {y}\n\n"
              f"Write a single-sentence hypothesis answering the question."),
        max_tokens=120, temperature=0.2).strip()
    return hypo, trace


def _largest_csv(task):
    best, best_n = None, -1
    for name, path in task.task.csv_paths.items():
        try:
            df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
            if len(df) > best_n:
                best, best_n = df, len(df)
        except Exception:
            continue
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="vsg_v1")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--judge", default="openai/gpt-4-turbo")
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = _PROJ / "lmw" / "vsg_discovery" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; --overwrite: {out_path}")

    client = make_openai_client()
    tasks = load_official_real_tasks(_PROJ / "discoverybench_repo", max_tasks=None,
                                     datasets=None, task_keys=set(args.tasks))
    task_by = {t.task_key: t for t in tasks}
    trace_f = io.StringIO()
    results, scores = {}, []
    print(f"=== VSG (verified staged generation) on real DiscoveryBench — solver={args.model} ===\n")
    for key in args.tasks:
        t = task_by.get(key)
        if t is None:
            continue
        df = _largest_csv(t)
        if df is None:
            continue
        col_desc = "\n".join(f"  {c} ({df[c].dtype})" for c in df.columns[:30])
        try:
            hypo, stages = vsg_solve(client, args.model, df, t.task.query, col_desc)
        except Exception as e:
            hypo, stages = None, [f"error: {type(e).__name__}: {e}"]
        if not hypo:
            results[key] = {"HMS": 0.0, "stages": stages, "hypo": ""}
            scores.append(0.0)
            print(f"  {key:48} HMS=  0.0  [{stages[-1][:50] if stages else 'no hypo'}]")
            continue
        ev = _run_official_eval(t, pred_hypo=hypo, pred_workflow="", judge_model=args.judge, trace_f=trace_f)
        hms = 100.0 * float(ev.get("final_score", 0.0))
        scores.append(hms)
        results[key] = {"HMS": round(hms, 1), "hypo": hypo, "stages": stages,
                        "recall_context": ev.get("recall_context")}
        print(f"  {key:48} HMS={hms:5.1f}  | {hypo[:70]}")
        for s in stages:
            print(f"        · {s}")
        out_path.write_text(json.dumps({"run_id": args.run_id, "model": args.model,
                                        "results": results}, indent=2))

    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"\n=== VSG RESULT (solver={args.model}) ===")
    print(f"  mean HMS = {mean:.2f}  (N={len(scores)})")
    print(f"  baseline (same tasks, no per-stage verification): mini~14, 4o~15")
    summary = {"run_id": args.run_id, "model": args.model, "judge": args.judge,
               "mean_HMS": round(mean, 2), "n": len(scores), "results": results}
    out_path.write_text(json.dumps(summary, indent=2))
    (out_dir / "judge_trace.log").write_text(trace_f.getvalue())
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
