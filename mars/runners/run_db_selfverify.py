"""General self-verifying portfolio on DiscoveryBench (test polygon, zero bench code).

Model authors diverse strategies; for EACH candidate it authors a claim-specific
verifier + negatives; the shell selects the candidate whose own verifier is
FAITHFUL (margin>0) and passed, else abstains. The only bench-touching code builds
a generic ctx (df, columns, question) and judges with official HMS.

  python -m mars.runners.run_db_selfverify --run_id sv_v1 --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from pathlib import Path

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

from mars.agents.base import make_openai_client
from mars.induction.self_portfolio import SelfPortfolioEngine
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="sv_v1")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--judge", default="openai/gpt-4-turbo")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "db_selfverify" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    engine = SelfPortfolioEngine(model=args.model, client=client, k_strategies=4)
    tasks = {t.task_key: t for t in load_official_real_tasks(
        _PROJ / "discoverybench_repo", max_tasks=None, datasets=None, task_keys=set(args.tasks))}
    trace = io.StringIO()
    results, answered = {}, []
    n_ans = n_abs = 0
    print(f"=== Self-verifying portfolio (per-candidate verifier + margin) — {args.model} ===\n")
    for key in args.tasks:
        t = tasks.get(key)
        if not t:
            continue
        df = _largest(t)
        if df is None:
            n_abs += 1; results[key] = {"status": "ABSTAIN", "note": "no csv"}
            print(f"  {key:46} ABSTAIN (no csv)"); continue
        ctx = {"df": df, "columns": list(df.columns), "question": t.task.query}
        ctx_desc = ("ctx['df']: pandas DataFrame; ctx['columns']: column names; ctx['question']: "
                    "the discovery question. Columns: " + ", ".join(map(str, df.columns[:25])))
        try:
            ans, tr = engine.run(t.task.query, ctx, ctx_desc)
        except Exception as e:
            n_abs += 1; results[key] = {"status": "ERROR", "note": str(e)[:100]}
            print(f"  {key:46} ERROR {str(e)[:45]}"); continue
        if ans is None:
            n_abs += 1
            results[key] = {"status": "ABSTAIN", "note": tr.note}
            print(f"  {key:46} ABSTAIN ({tr.note})")
        else:
            ev = _run_official_eval(t, pred_hypo=ans, pred_workflow="", judge_model=args.judge, trace_f=trace)
            hms = 100.0 * float(ev.get("final_score", 0.0))
            answered.append(hms); n_ans += 1
            results[key] = {"status": "ANSWER", "HMS": round(hms, 1), "margin": tr.margin,
                            "selected": tr.selected, "answer": ans[:120]}
            print(f"  {key:46} HMS={hms:5.1f} margin={tr.margin} | {ans[:48]}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_ans + n_abs
    prec = sum(answered) / len(answered) if answered else 0.0
    forced = sum(answered) / n if n else 0.0
    print(f"\n=== RESULT (N={n}) ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  PRECISION on answered = {prec:.2f} HMS")
    print(f"  forced (abstain=0)    = {forced:.2f} HMS")
    print(f"  oracle-max portfolio ceiling = 32.89 | single method ~15 | SOTA 24.5")
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "precision_answered": round(prec, 2), "forced_mean": round(forced, 2),
                                    "results": results}, indent=2, default=str))
    (out_dir / "judge_trace.log").write_text(trace.getvalue())
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
