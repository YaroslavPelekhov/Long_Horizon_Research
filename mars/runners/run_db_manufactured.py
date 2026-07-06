"""DiscoveryBench with a MANUFACTURED code-invariant: predictive consistency.

Principle from the arithmetic case, transferred: the model only STRUCTURES the
hypothesis (which variables / population / relation — transcription, reliable);
CODE imposes the manufactured invariant — a correct relationship must PREDICT
HELD-OUT rows (sign + strength stable on data the fit never saw); a spurious one
won't. Claims passing the predictive invariant are emitted; else abstain.

Honest prior: DB's bottleneck is matching the gold's SEMANTIC framing (verifier-
poor), not computation — so this likely confirms the ~14-15 ceiling. Run to see.

  python -m mars.runners.run_db_manufactured --run_id dbm_v1 --model openai/gpt-4o-mini
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
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=True)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.runners.run_db_official_eval import _run_official_eval, load_official_real_tasks

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


def _largest(task):
    best, n = None, -1
    for _, p in task.task.csv_paths.items():
        try:
            df = pd.read_csv(p, encoding="utf-8", on_bad_lines="skip")
            if len(df) > n:
                best, n = df, len(df)
        except Exception:
            pass
    return best


def _corr(d, x, y):
    s = d[[x, y]].dropna()
    if len(s) < 5:
        return None
    try:
        c = s[x].astype(float).corr(s[y].astype(float))
        return None if c != c else float(c)
    except Exception:
        return None


def predictive_invariant(df, x, y, filt):
    """Manufactured invariant: fit sign/strength on TRAIN, require it to hold on HELD-OUT."""
    d = df
    if filt and filt[0] in df.columns:
        d = df[df[filt[0]].astype(str) == str(filt[1])]
    d = d[[x, y]].dropna()
    if len(d) < 12:
        return 0.0, None
    k = int(len(d) * 0.6)
    tr, ho = d.iloc[:k], d.iloc[k:]
    ctr, cho = _corr(tr, x, y), _corr(ho, x, y)
    if ctr is None or cho is None:
        return 0.0, None
    sign_match = (ctr > 0) == (cho > 0)
    score = abs(cho) if (sign_match and abs(cho) > 0.15) else 0.0   # predicts held-out
    return score, (1 if cho > 0 else -1)


def propose_structures(client, model, q, df):
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:10]
    cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 12]
    catvals = {c: [str(v) for v in df[c].dropna().unique()[:6]] for c in cat[:5]}
    obj = call_llm(client, model=model, system="Return only JSON.",
        user=(f"Question: {q}\nNumeric columns: {num}\nCategorical filters: {json.dumps(catvals)[:1000]}\n\n"
              f"Propose up to 6 candidate hypotheses as STRUCTURE (exact names):\n"
              f'JSON: {{"cands":[{{"x":"..","y":"..","filter":["col","val"] or null}}]}}'),
        max_tokens=500, temperature=0.4)
    t = obj.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        cands = json.loads(t).get("cands", [])
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: cands = json.loads(t[i:j+1]).get("cands", [])
        except Exception: cands = []
    out = []
    for c in cands:
        if isinstance(c, dict) and c.get("x") in num and c.get("y") in num and c["x"] != c["y"]:
            out.append((c["x"], c["y"], c.get("filter")))
    return out[:6]


def solve(client, model, task):
    df = _largest(task)
    if df is None:
        return None, "no csv"
    cands = propose_structures(client, model, task.task.query, df)
    best = None
    for (x, y, filt) in cands:
        score, sign = predictive_invariant(df, x, y, filt)
        if score > 0 and (best is None or score > best[0]):
            best = (score, x, y, filt, sign)
    if best is None:
        return None, "abstain: no candidate predicts held-out (manufactured invariant fails)"
    _, x, y, filt, sign = best
    rel = "positively associated with" if sign > 0 else "negatively associated with"
    ctx = f"the subpopulation where {filt[0]} = {filt[1]}" if filt else "the overall population"
    hypo = call_llm(client, model=model,
        system="One precise hypothesis from ONLY the verified facts; question's framing; no new claims.",
        user=(f"Question: {task.task.query}\nVerified: context={ctx}; {x} is {rel} {y} "
              f"(holds on held-out data).\nOne sentence."),
        max_tokens=120, temperature=0.2).strip()
    return hypo, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="dbm_v1")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--judge", default="openai/gpt-4-turbo")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "db_manufactured" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    tasks = {t.task_key: t for t in load_official_real_tasks(
        _PROJ / "discoverybench_repo", max_tasks=None, datasets=None, task_keys=set(args.tasks))}
    trace = io.StringIO()
    results, answered = {}, []
    n_ans = n_abs = 0
    print(f"=== DB manufactured-invariant (predictive held-out) — {args.model} ===\n")
    for key in args.tasks:
        t = tasks.get(key)
        if not t:
            continue
        try:
            hypo, why = solve(client, args.model, t)
        except Exception as e:
            hypo, why = None, f"err {type(e).__name__}"
        if hypo is None:
            n_abs += 1; results[key] = {"status": "ABSTAIN", "why": why}
            print(f"  {key:46} ABSTAIN ({why})")
        else:
            ev = _run_official_eval(t, pred_hypo=hypo, pred_workflow="", judge_model=args.judge, trace_f=trace)
            hms = 100.0 * float(ev.get("final_score", 0.0))
            answered.append(hms); n_ans += 1
            results[key] = {"status": "ANSWER", "HMS": round(hms, 1), "hypo": hypo}
            print(f"  {key:46} HMS={hms:5.1f} | {hypo[:55]}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_ans + n_abs
    prec = sum(answered) / len(answered) if answered else 0.0
    overall = sum(answered) / n if n else 0.0
    print(f"\n=== RESULT ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  precision on answered = {prec:.2f} HMS   |  forced(abstain=0) = {overall:.2f}")
    print(f"  prior DB ceiling (data-first, paper judge): ~14-15")
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "precision_answered": round(prec, 2), "forced_mean": round(overall, 2),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
