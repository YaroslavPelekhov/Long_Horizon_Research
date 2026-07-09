"""Run DiscoveryBench CPI predictions through the official HMS evaluator."""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from dotenv import load_dotenv

    env_path = _PROJ / "autodiscovery" / ".env.local"
    if env_path.exists():
        load_dotenv(env_path, override=True)
except ImportError:
    pass

from mars.induction.db_cpi import build_evidence_bundle, synthesize_hypothesis  # noqa: E402
from mars.runners.run_db_official_eval import (  # noqa: E402
    _run_official_eval,
    load_official_real_tasks,
)


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _write_jsonl(f, row: dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    f.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=os.environ.get("MARS_DB_CPI_RUN_ID", "db_cpi_official"))
    parser.add_argument("--max_tasks", type=int, default=5)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--task_keys", nargs="*", default=None)
    parser.add_argument("--generator_model", default=os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--judge_model", default=os.environ.get("MARS_DB_OFFICIAL_JUDGE_MODEL", "openai/gpt-4o"))
    parser.add_argument("--skip_official_eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo = _PROJ / "discoverybench_repo"
    datasets = set(_parse_csv(args.datasets)) or None
    task_keys = {x.strip() for x in args.task_keys or [] if x.strip()} or None
    tasks = load_official_real_tasks(
        repo,
        max_tasks=args.max_tasks if args.max_tasks > 0 else None,
        datasets=datasets,
        task_keys=task_keys,
        start_index=args.start_index,
    )
    if not tasks:
        raise SystemExit("no DiscoveryBench tasks selected")

    out_dir = _PROJ / "lmw" / "db_cpi" / args.run_id
    pred_path = out_dir / "predictions.jsonl"
    eval_path = out_dir / "official_eval.jsonl"
    trace_path = out_dir / "eval_trace.log"
    summary_path = out_dir / "summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in [pred_path, eval_path, trace_path, summary_path]:
        if p.exists() and not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {p}")
        if p.exists() and args.overwrite:
            p.unlink()

    print("\n=== DiscoveryBench CPI official HMS ===")
    print(f"run_id={args.run_id} N={len(tasks)} gen={args.generator_model} judge={args.judge_model}")
    print(f"out_dir={out_dir}\n")

    pred_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    t0 = time.time()
    with pred_path.open("w", encoding="utf-8") as pred_f, eval_path.open("w", encoding="utf-8") as eval_f, trace_path.open("w", encoding="utf-8") as trace_f:
        for i, task in enumerate(tasks, 1):
            metadata = json.loads(task.metadata_path.read_text(encoding="utf-8"))
            evidence = build_evidence_bundle(
                metadata=metadata,
                csv_paths=task.task.csv_paths,
                query=task.task.query,
            )
            pred_hypo, pred_workflow, raw = synthesize_hypothesis(
                query=task.task.query,
                domain=task.task.domain,
                query_type=task.task.query_type,
                metadata=metadata,
                evidence=evidence,
                model=args.generator_model,
            )
            pred_row = {
                "task_key": task.task_key,
                "task_id": task.task.task_id,
                "dataset": task.dataset_name,
                "metadata_id": task.metadata_id,
                "query_id": task.query_id,
                "domain": task.task.domain,
                "query_type": task.task.query_type,
                "query": task.task.query,
                "metadata_path": str(task.metadata_path),
                "generator_model": args.generator_model,
                "pred_hypo": pred_hypo,
                "pred_workflow": pred_workflow,
                "evidence": [
                    {"analyzer": e.analyzer, "dataset": e.dataset, "text": e.text, "score": e.score}
                    for e in evidence
                ],
                "raw_synthesis": raw,
            }
            _write_jsonl(pred_f, pred_row)
            pred_rows.append(pred_row)

            eval_row: dict[str, Any] = {
                "task_key": task.task_key,
                "dataset": task.dataset_name,
                "metadata_id": task.metadata_id,
                "query_id": task.query_id,
                "judge_model": args.judge_model,
                "official_eval_skipped": bool(args.skip_official_eval),
            }
            if not args.skip_official_eval:
                try:
                    eval_result = _run_official_eval(
                        task,
                        pred_hypo=pred_hypo,
                        pred_workflow=pred_workflow,
                        judge_model=args.judge_model,
                        trace_f=trace_f,
                    )
                    eval_row["eval_result"] = eval_result
                    eval_row["final_score"] = float(eval_result.get("final_score", 0.0))
                    eval_row["HMS_100"] = 100.0 * eval_row["final_score"]
                except Exception as e:
                    eval_row["error"] = f"{type(e).__name__}: {e}"
                    eval_row["final_score"] = 0.0
                    eval_row["HMS_100"] = 0.0
            _write_jsonl(eval_f, eval_row)
            eval_rows.append(eval_row)
            score = "skip" if args.skip_official_eval else f"{float(eval_row.get('HMS_100', 0.0)):.1f}"
            print(f"[{i}/{len(tasks)}] {task.task_key} HMS={score} evidence={len(evidence)}", flush=True)

    scores = [
        float(r.get("final_score", 0.0))
        for r in eval_rows
        if not r.get("error") and not r.get("official_eval_skipped")
    ]
    summary = {
        "run_id": args.run_id,
        "score_type": "official-HMS-compatible" if not args.skip_official_eval else "prediction-export",
        "generator_model": args.generator_model,
        "judge_model": args.judge_model if not args.skip_official_eval else None,
        "n_tasks_selected": len(tasks),
        "n_predictions": len(pred_rows),
        "n_official_eval_success": len(scores),
        "official_eval_skipped": bool(args.skip_official_eval),
        "HMS_mean_0_1": st.fmean(scores) if scores else 0.0,
        "HMS_mean_100": 100.0 * st.fmean(scores) if scores else 0.0,
        "HMS_sd_100": 100.0 * st.stdev(scores) if len(scores) > 1 else 0.0,
        "wall_time_s": time.time() - t0,
        "paths": {
            "predictions": str(pred_path),
            "official_eval": str(eval_path),
            "eval_trace": str(trace_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n=== Summary ===")
    print(f"HMS_mean_100={summary['HMS_mean_100']:.2f} N={summary['n_official_eval_success']}/{len(tasks)}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
