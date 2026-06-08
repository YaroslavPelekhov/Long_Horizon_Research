"""Hybrid DB sweep: QD-CPI + meta-routing on the stratified-20 task set.

For each of the 20 tasks from the official stratified-20 evaluation:
  1. Classify question type (temporal_occurrence / statistical_relation / etc.)
  2. Run QD-CPI (question decomposition chain + code execution)
  3. Score with the same HMS judge as the existing db_cpi.py baseline
  4. Compare: QD-CPI (zero hand-written operators) vs db_cpi.py (hand-written)

Key comparison baseline: stratified20_db_cpi_gpt4omini_gpt4o_v1
  HMS_mean = 25.16, solved 7/20 tasks

Run
---
python -m mars.runners.run_db_hybrid_sweep \\
    --run_id hybrid_sweep_v1 \\
    --judge_model openai/gpt-4o --synth_model openai/gpt-4o --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
_DB_REPO = _PROJ / "discoverybench_repo"
for _p in (_PROJ, _DB_REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv
    _env = _PROJ / "autodiscovery" / ".env.local"
    if _env.exists():
        load_dotenv(_env, override=True)
except ImportError:
    pass

import pandas as pd  # noqa: E402

from mars.agents.base import call_llm, make_openai_client  # noqa: E402
from mars.induction.db_grammar_synthesizer import (  # noqa: E402
    extract_column_descriptions,
    extract_schema_description,
)
from mars.induction.qd_cpi import classify_interface, solve as qd_solve  # noqa: E402
from mars.runners.run_db_official_eval import (  # noqa: E402
    load_official_real_tasks,
    _run_official_eval,
)


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

_BASELINE_EVAL = _PROJ / "lmw" / "db_cpi" / "stratified20_db_cpi_gpt4omini_gpt4o_v1" / "official_eval.jsonl"


def load_stratified_official_tasks() -> list[Any]:
    """Load the 20 official task objects matching the stratified-20 baseline."""
    with open(_BASELINE_EVAL) as f:
        evals = [json.loads(l) for l in f if l.strip()]
    baseline_by_key = {e["task_key"]: e["HMS_100"] for e in evals}
    all_keys = set(baseline_by_key.keys())
    all_datasets = {e["dataset"] for e in evals}

    tasks = load_official_real_tasks(
        _DB_REPO,
        max_tasks=None,
        datasets=all_datasets,
        task_keys=all_keys,
    )
    # Attach baseline score
    for t in tasks:
        t.baseline_hms = baseline_by_key.get(t.task_key, 0.0)
    return tasks


def load_df_for_task(task: Any) -> tuple[pd.DataFrame | None, dict[str, str]]:
    """Load the richest dataset for a task, with column descriptions."""
    best_df = None
    best_col_descs: dict[str, str] = {}

    for ds in task.task.datasets:
        csv_path = task.metadata_path.parent / ds.get("name", "")
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].str.replace(",", ".", regex=False).astype(float)
                except Exception:
                    pass
        col_descs: dict[str, str] = {}
        for col_info in ds.get("columns", {}).get("raw", []):
            name = col_info.get("name", "")
            desc = col_info.get("description", "")
            if name and desc:
                col_descs[name] = desc
        score = len(col_descs) * 2 + min(df.shape[0], 1000)
        if best_df is None or score > (len(best_col_descs) * 2 + min(best_df.shape[0], 1000)):
            best_df = df
            best_col_descs = col_descs
    return best_df, best_col_descs


def _official_hms(task: Any, hypothesis: str, judge_model: str) -> float:
    """Score using the official DiscoveryBench evaluator (same as baseline)."""
    import io as _io
    trace = _io.StringIO()
    try:
        result = _run_official_eval(
            task,
            pred_hypo=hypothesis,
            pred_workflow="",
            judge_model=judge_model,
            trace_f=trace,
        )
        return 100.0 * float(result.get("final_score", 0.0))
    except Exception as e:
        return 0.0


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------

def run_one_task(
    task: Any,
    synth_model: str,
    judge_model: str,
) -> dict[str, Any]:
    question = task.task.query if isinstance(task.task.query, str) else str(task.task.query)
    meta = json.load(open(task.metadata_path))
    domain_knowledge = meta.get("domain_knowledge", "")[:600]
    interface_type = classify_interface(question + " " + domain_knowledge[:200])

    df, col_descs = load_df_for_task(task)

    if df is None:
        return {
            "task_key": task.task_key,
            "interface_type": interface_type,
            "qd_hms": 0.0,
            "qd_answer": "no dataset",
            "baseline_hms": task.baseline_hms,
            "hybrid_hms": task.baseline_hms,
            "error": "no dataset loaded",
        }

    t0 = time.time()
    chain = qd_solve(
        question,
        domain_knowledge,
        df,
        column_descriptions=col_descs,
        model=synth_model,
    )
    qd_time = time.time() - t0

    # Score with OFFICIAL DiscoveryBench evaluator (same as baseline)
    qd_hms = _official_hms(task, chain.final_answer, judge_model)
    baseline_hms = task.baseline_hms
    hybrid_hms = max(qd_hms, baseline_hms)

    return {
        "task_key": task.task_key,
        "dataset": task.task_key.split(":")[0],
        "question": question[:80],
        "interface_type": interface_type,
        "n_steps": len(chain.steps),
        "n_verified": sum(1 for s in chain.steps if s.verified),
        "qd_answer": chain.final_answer[:250],
        "qd_hms": qd_hms,
        "baseline_hms": baseline_hms,
        "hybrid_hms": hybrid_hms,
        "qd_time_s": qd_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=os.environ.get("MARS_DB_HYBRID_RUN_ID", "hybrid_sweep_smoke"))
    parser.add_argument("--synth_model", default=os.environ.get("MARS_SYNTH_MODEL", "openai/gpt-4o"))
    parser.add_argument("--judge_model", default=os.environ.get("MARS_JUDGE_MODEL", "openai/gpt-4o"))
    parser.add_argument("--max_tasks", type=int, default=20)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = _PROJ / "lmw" / "db_hybrid" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; pass --overwrite: {out_path}")

    tasks = load_stratified_official_tasks()
    if args.datasets:
        ds_filter = {x.strip() for x in args.datasets.split(",") if x.strip()}
        tasks = [t for t in tasks if t.task_key.split(":")[0] in ds_filter]
    tasks = tasks[:args.max_tasks]

    print(f"Running hybrid sweep: {len(tasks)} tasks (model={args.synth_model})")
    print(f"Baseline: {_BASELINE_EVAL.parent.name}  HMS_mean=25.16")
    print(f"Judge: official DiscoveryBench evaluator (same as baseline)")
    print()

    results = []
    for i, task in enumerate(tasks):
        question = task.task.query if isinstance(task.task.query, str) else str(task.task.query)
        itype = classify_interface(question)
        print(f"[{i+1}/{len(tasks)}] {task.task_key}  itype={itype}")
        try:
            row = run_one_task(task, args.synth_model, args.judge_model)
        except Exception as e:
            row = {"task_key": task.task_key, "qd_hms": 0.0,
                   "baseline_hms": task.baseline_hms, "hybrid_hms": task.baseline_hms, "error": str(e)}
        results.append(row)
        print(f"  qd={row.get('qd_hms',0):.1f}  baseline={row.get('baseline_hms',0):.1f}"
              f"  hybrid={row.get('hybrid_hms',0):.1f}  answer={row.get('qd_answer','')[:60]}")

    n = len(results)
    mean_qd = sum(r.get("qd_hms", 0) for r in results) / n if n else 0
    mean_baseline = sum(r.get("baseline_hms", 0) for r in results) / n if n else 0
    mean_hybrid = sum(r.get("hybrid_hms", 0) for r in results) / n if n else 0

    summary = {
        "run_id": args.run_id,
        "score_type": "hybrid-qd-cpi-vs-db-cpi",
        "synth_model": args.synth_model,
        "judge_model": args.judge_model,
        "n_tasks": n,
        "mean_qd_hms": mean_qd,
        "mean_baseline_hms": mean_baseline,
        "mean_hybrid_hms": mean_hybrid,
        "qd_better_count": sum(1 for r in results if r.get("qd_hms",0) > r.get("baseline_hms",0)),
        "baseline_better_count": sum(1 for r in results if r.get("baseline_hms",0) > r.get("qd_hms",0)),
        "tied_count": sum(1 for r in results if r.get("qd_hms",0) == r.get("baseline_hms",0)),
        "results": results,
    }

    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 55)
    print("HYBRID SWEEP RESULTS")
    print("=" * 55)
    print(f"{'Condition':<22} {'Mean HMS':>10}")
    print("-" * 35)
    print(f"{'QD-CPI (zero ops)':<22} {mean_qd:>9.1f}")
    print(f"{'db_cpi.py (hand ops)':<22} {mean_baseline:>9.1f}")
    print(f"{'Hybrid (max both)':<22} {mean_hybrid:>9.1f}")
    print("-" * 35)
    print(f"QD better: {summary['qd_better_count']}/{n}  "
          f"baseline better: {summary['baseline_better_count']}/{n}  "
          f"tied: {summary['tied_count']}/{n}")
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
