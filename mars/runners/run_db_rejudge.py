"""Re-judge existing DiscoveryBench predictions with the PAPER-compatible judge.

Isolates the judge effect: same generated hypotheses, scored by gpt-4-1106-preview
(the paper's HMS judge), repeated N times per task to quantify judge
non-determinism. Use after run_universal_discovery_real_eval.

  python -m mars.runners.run_db_rejudge --src db_real_dpsr_diverse \
      --judge openai/gpt-4-1106-preview --repeats 2
"""
from __future__ import annotations

import argparse
import io
import json
import statistics as st
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=False)
except ImportError:
    pass

from mars.runners.run_db_official_eval import _run_official_eval, load_official_real_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="db_real_dpsr_diverse")
    ap.add_argument("--judge", default="openai/gpt-4-1106-preview")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    src_dir = _PROJ / "lmw" / "universal_discovery_real" / args.src
    preds = [json.loads(l) for l in open(src_dir / "predictions.jsonl")]
    pred_by_key = {p["task_key"]: p for p in preds}

    tasks = load_official_real_tasks(_PROJ / "discoverybench_repo", max_tasks=None,
                                     datasets=None, task_keys=set(pred_by_key))
    task_by_key = {t.task_key: t for t in tasks}

    trace = io.StringIO()
    per_task = {}
    print(f"=== Re-judge {args.src} with PAPER judge {args.judge} (repeats={args.repeats}) ===\n")
    for key, p in pred_by_key.items():
        t = task_by_key.get(key)
        if t is None:
            continue
        scores = []
        for _ in range(args.repeats):
            try:
                ev = _run_official_eval(t, pred_hypo=p.get("pred_hypo", ""),
                                        pred_workflow=p.get("pred_workflow", ""),
                                        judge_model=args.judge, trace_f=trace)
                scores.append(100.0 * float(ev.get("final_score", 0.0)))
            except Exception as e:
                scores.append(0.0)
                trace.write(f"\n[err {key}] {type(e).__name__}: {e}\n")
        mean_s = sum(scores) / len(scores)
        per_task[key] = {"scores": scores, "mean": round(mean_s, 1)}
        spread = f"(runs: {scores})" if len(set(scores)) > 1 else ""
        print(f"  {key:55} HMS={mean_s:5.1f}  {spread}", flush=True)

    means = [v["mean"] for v in per_task.values()]
    overall = sum(means) / len(means) if means else 0.0
    # judge non-determinism: how many tasks gave different scores across repeats
    flipped = sum(1 for v in per_task.values() if len(set(v["scores"])) > 1)
    print(f"\n=== RESULT (paper judge, same predictions) ===")
    print(f"  mean HMS = {overall:.2f}  (N={len(means)})")
    print(f"  tasks with judge-score flips across repeats: {flipped}/{len(means)}")
    print(f"  published (gpt-4o solver): React/CodeGen ~15.4, Reflexion+Oracle 24.5")
    print(f"  NOTE: our solver = gpt-4o-mini (weaker than their gpt-4o baselines)")

    out = src_dir / f"rejudge_{args.judge.split('/')[-1]}.json"
    out.write_text(json.dumps({"judge": args.judge, "repeats": args.repeats,
                               "mean_HMS": round(overall, 2), "flipped": flipped,
                               "per_task": per_task}, indent=2))
    (src_dir / "rejudge_trace.log").write_text(trace.getvalue())
    print(f"\nsummary → {out}")


if __name__ == "__main__":
    main()
