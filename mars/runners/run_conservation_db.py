"""DiscoveryBench as a THIN ADAPTER over the SAME universal conservation-discovery engine
used for NewtonBench. Proves one method, two benchmarks, zero benchmark logic in the core.

Difference from the NB adapter is ONLY the adapter wiring:
  - candidates carry a `quantity` (the signed scalar the prose claim asserts), not a `fidelity`
    (DB answers are hypotheses, not laws fit to data);
  - transforms are bootstrap/subsample of the dataframe;
  - the scorer is the official HMS judge (paper judge gpt-4-turbo).
The engine's selection/abstention (conservation of the quantity, no tuned constants) is identical.

  python -m mars.runners.run_conservation_db --run_id cdb_v1 --model openai/gpt-4o-mini
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
from mars.induction import conservation_discovery as cd
from mars.induction.self_portfolio import _compile
from mars.runners.run_db_invariants import author_candidates, TASKS, _largest
from mars.runners.run_db_official_eval import _run_official_eval, load_official_real_tasks


def build_adapter(client, model, task, df, judge, trace, k):
    ctx0 = {"df": df, "columns": list(df.columns)}
    ctx_desc = ("ctx['df']: pandas DataFrame; ctx['columns']: column names. "
                "Columns: " + ", ".join(map(str, df.columns[:25])))

    def generate_candidates(ctx):
        out = []
        for ans, mcode in author_candidates(client, model, task.task.query, ctx_desc, k):
            fn = _compile(mcode, "measure", {})
            if fn is None:
                continue
            out.append(cd.Candidate(answer=ans, quantity=fn, meta={}))
        return out

    adapter = cd.DiscoveryAdapter(
        generate_candidates=generate_candidates,
        transforms=cd.bootstrap_transforms(df_key="df", n=20),
        score=lambda ans: 100.0 * float(_run_official_eval(
            task, pred_hypo=ans, pred_workflow="", judge_model=judge, trace_f=trace).get("final_score", 0.0)),
        noise_floor=lambda c: 0.0,     # claims path uses conservation of the quantity, not fidelity
    )
    return adapter, ctx0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="cdb_v1")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--judge", default="openai/gpt-4-turbo")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "conservation_db" / args.run_id
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
    print(f"=== DiscoveryBench via universal conservation engine — {args.model} ===\n")
    for key in args.tasks:
        t = tasks.get(key)
        if not t:
            continue
        df = _largest(t)
        if df is None:
            n_abs += 1; results[key] = {"status": "ABSTAIN", "note": "no csv"}
            print(f"  {key:46} ABSTAIN (no csv)"); continue
        adapter, ctx = build_adapter(client, args.model, t, df, args.judge, trace, args.k)
        sel = cd.select(adapter, ctx)
        if sel.abstained:
            n_abs += 1; results[key] = {"status": "ABSTAIN", "reason": sel.reason}
            print(f"  {key:46} ABSTAIN ({sel.reason})"); continue
        hms = adapter.score(sel.answer)
        answered.append(hms); n_ans += 1
        results[key] = {"status": "ANSWER", "HMS": round(hms, 1), "answer": sel.answer[:110]}
        print(f"  {key:46} HMS={hms:5.1f} | {sel.answer[:44]}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_ans + n_abs
    prec = sum(answered) / len(answered) if answered else 0.0
    forced = sum(answered) / n if n else 0.0
    print(f"\n=== RESULT (N={n}) ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  PRECISION on answered = {prec:.2f} HMS  |  forced (abstain=0) = {forced:.2f} HMS")
    print(f"  same engine as NB; baseline first-candidate=4.25 forced; SOTA 24.5")
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "precision_answered": round(prec, 2), "forced_mean": round(forced, 2),
                                    "results": results}, indent=2, default=str))
    (out_dir / "judge_trace.log").write_text(trace.getvalue())
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
