"""Run the task-agnostic Self-Portfolio shell on real DiscoveryBench.

The ONLY benchmark-touching code here builds a generic ctx (df, columns,
question) and judges with the official HMS. The strategies, verifier, and
negatives are ALL authored by the model at runtime — nothing hand-coded per task.

  python -m mars.runners.run_self_portfolio --run_id sp_v1 --model openai/gpt-4o-mini
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
    "archaeology:0:0", "meta_regression:0:1", "nls_ses:0:0",
    "requirements_engineering_for_ML_enabled_systems:0:0",
    "nls_incarceration:0:0", "worldbank_education_gdp:0:0",
]


def _largest_csv(task):
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
    ap.add_argument("--run_id", default="sp_v1")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--judge", default="openai/gpt-4-turbo")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "self_portfolio" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    engine = SelfPortfolioEngine(model=args.model, client=client)
    tasks = {t.task_key: t for t in load_official_real_tasks(
        _PROJ / "discoverybench_repo", max_tasks=None, datasets=None, task_keys=set(args.tasks))}
    trace_f = io.StringIO()
    results, answered = {}, []
    n_ans = n_abs = 0
    print(f"=== Self-Portfolio (model authors everything) — solver={args.model} ===\n")
    for key in args.tasks:
        t = tasks.get(key)
        if not t:
            continue
        df = _largest_csv(t)
        if df is None:
            results[key] = {"status": "ABSTAIN", "note": "no csv"}; n_abs += 1
            print(f"  {key:48} ABSTAIN (no csv)"); continue
        ctx = {"df": df, "columns": list(df.columns), "question": t.task.query}
        ctx_desc = ("ctx['df']: a pandas DataFrame; ctx['columns']: list of column names; "
                    "ctx['question']: the discovery question (str). Columns: "
                    + ", ".join(map(str, df.columns[:25])))
        try:
            ans, tr = engine.run(t.task.query, ctx, ctx_desc)
        except Exception as e:
            ans, tr = None, None
            results[key] = {"status": "ERROR", "note": str(e)[:120]}; n_abs += 1
            print(f"  {key:48} ERROR {str(e)[:50]}"); continue
        if ans is None:
            n_abs += 1
            results[key] = {"status": "ABSTAIN", "note": tr.note, "margin": tr.margin,
                            "strategies": tr.strategies}
            print(f"  {key:48} ABSTAIN ({tr.note})")
            out_path.write_text(json.dumps(results, indent=2, default=str)); continue
        ev = _run_official_eval(t, pred_hypo=ans, pred_workflow="", judge_model=args.judge, trace_f=trace_f)
        hms = 100.0 * float(ev.get("final_score", 0.0))
        answered.append(hms); n_ans += 1
        results[key] = {"status": "ANSWER", "HMS": round(hms, 1), "margin": tr.margin,
                        "selected": tr.selected, "answer": ans, "n_strategies": len(tr.strategies)}
        print(f"  {key:48} ANSWER HMS={hms:5.1f} margin={tr.margin} via '{tr.selected}' | {ans[:50]}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_ans + n_abs
    prec = sum(answered) / len(answered) if answered else 0.0
    print(f"\n=== RESULT ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  PRECISION on answered = {prec:.2f} HMS")
    print(f"  baselines: hand data-first ~15, oracle-portfolio ceiling ~23")
    print(f"  (NB: strategies/verifier/negatives ALL model-authored — zero benchmark code)")
    out_path.write_text(json.dumps({"run_id": args.run_id, "model": args.model,
                                    "answered": n_ans, "abstained": n_abs,
                                    "precision_answered": round(prec, 2), "results": results},
                                   indent=2, default=str))
    (out_dir / "judge_trace.log").write_text(trace_f.getvalue())
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
