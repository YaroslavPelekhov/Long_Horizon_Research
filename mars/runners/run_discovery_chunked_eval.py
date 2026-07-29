"""Run an immutable DiscoveryBench evaluation in parallel task chunks.

This runner is only valid when language promotion is disabled.  Each worker
receives a disjoint subset of tasks and the parent merges the native evaluator
rows into the same artifact format as the serial runner.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path
from typing import Any

from mars.runners.run_db_official_eval import (
    load_official_real_tasks,
    load_official_train_tasks,
)


_PROJ = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _run_chunk(command: list[str], env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=_PROJ, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout[-8000:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--data_split", choices=("train", "test"), default="test")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--judge_model", default="openai/gpt-4o")
    parser.add_argument("--n_proposals", type=int, default=4)
    parser.add_argument("--max_rounds", type=int, default=1)
    parser.add_argument("--task_timeout_s", type=int, default=600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete child chunks without recomputing completed evaluations.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if os.environ.get("MARS_PROMOTE_SELF_MODULES", "1").lower() not in {"0", "false", "no", "off"}:
        raise SystemExit("parallel evaluation requires MARS_PROMOTE_SELF_MODULES=0")
    if os.environ.get("MARS_PROMOTE_SELF_LAYERS", "1").lower() not in {"0", "false", "no", "off"}:
        raise SystemExit("parallel evaluation requires MARS_PROMOTE_SELF_LAYERS=0")
    repo = _PROJ / "discoverybench_repo"
    loader = load_official_train_tasks if args.data_split == "train" else load_official_real_tasks
    tasks = loader(repo, max_tasks=None, datasets=None, task_keys=None)
    if not tasks:
        raise SystemExit("no tasks")
    workers = min(max(1, args.workers), len(tasks))
    chunks = [tasks[index::workers] for index in range(workers)]
    out_dir = _PROJ / "lmw" / "universal_discovery_real" / args.run_id
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.resume:
            raise SystemExit("refusing to overwrite a non-empty merged run; pass --resume")
        if (out_dir / "summary.json").exists():
            raise SystemExit("merged summary already exists; choose a new run_id")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    commands: list[list[str]] = []
    for index, chunk in enumerate(chunks):
        child_id = f"{args.run_id}__chunk{index:02d}"
        command = [
            sys.executable,
            "-m",
            "mars.runners.run_universal_discovery_real_eval",
            "--run_id", child_id,
            "--data_split", args.data_split,
            "--max_tasks", "0",
            "--model", args.model,
            "--judge_model", args.judge_model,
            "--n_proposals", str(args.n_proposals),
            "--max_rounds", str(args.max_rounds),
            "--task_timeout_s", str(args.task_timeout_s),
            "--fail_on_eval_error",
            "--overwrite",
            "--task_keys", *[task.task_key for task in chunk],
        ]
        if args.resume:
            command.append("--resume")
        commands.append(command)

    print(f"parallel DiscoveryBench: split={args.data_split} N={len(tasks)} workers={workers}", flush=True)
    outputs: list[tuple[int, str]] = []
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(_run_chunk, command, base_env) for command in commands]
        for index, future in enumerate(pending):
            code, output = future.result()
            outputs.append((code, output))
            print(f"chunk {index + 1}/{workers} exit={code}", flush=True)
    failed = [(index, output) for index, (code, output) in enumerate(outputs) if code != 0]
    if failed:
        for index, output in failed:
            print(f"--- failed chunk {index} ---\n{output}", file=sys.stderr)
        raise SystemExit("one or more chunks failed")

    predictions: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    language_states: list[dict[str, Any]] = []
    for index in range(workers):
        child_dir = _PROJ / "lmw" / "universal_discovery_real" / f"{args.run_id}__chunk{index:02d}"
        summary = json.loads((child_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("language_mutated"):
            raise SystemExit(f"frozen language mutated in chunk {index}")
        language_states.append(summary.get("language_state_after", {}))
        predictions.extend(_read_jsonl(child_dir / "predictions.jsonl"))
        evaluations.extend(_read_jsonl(child_dir / "official_eval.jsonl"))
    # A child can have written a prediction before an evaluator transport
    # failure.  On resume the fresh prediction is the final row for that key.
    predictions_by_key = {
        str(row.get("task_key", "")): row
        for row in predictions
        if row.get("task_key")
    }
    predictions = [predictions_by_key[key] for key in sorted(predictions_by_key)]
    evaluations.sort(key=lambda row: str(row.get("task_key", "")))
    task_keys = [str(row.get("task_key", "")) for row in evaluations]
    if len(evaluations) != len(tasks) or len(set(task_keys)) != len(tasks):
        raise SystemExit("merged evaluation has missing or duplicate tasks")
    if len(predictions) != len(tasks):
        raise SystemExit("merged predictions have missing task rows")
    pred_path = out_dir / "predictions.jsonl"
    eval_path = out_dir / "official_eval.jsonl"
    pred_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    eval_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in evaluations), encoding="utf-8")
    scores = [float(row.get("HMS_100", 0.0) or 0.0) for row in evaluations]
    consistency = [float(row.get("HMS_consistency_100", row.get("HMS_100", 0.0)) or 0.0) for row in evaluations]
    summary = {
        "run_id": args.run_id,
        "score_type": "universal-cpi-real-discovery",
        "data_split": args.data_split,
        "model": args.model,
        "judge_model": args.judge_model,
        "n_tasks": len(evaluations),
        "n_tasks_requested": len(tasks),
        "n_proposals": args.n_proposals,
        "max_rounds": args.max_rounds,
        "HMS_mean_100": st.fmean(scores),
        "HMS_sd_100": st.stdev(scores) if len(scores) > 1 else 0.0,
        "HMS_mean_consistency_100": st.fmean(consistency),
        "HMS_sd_consistency_100": st.stdev(consistency) if len(consistency) > 1 else 0.0,
        "scores": scores,
        "consistency_scores": consistency,
        "task_timeout_s": args.task_timeout_s,
        "parallel_workers": workers,
        "resumed": bool(args.resume),
        "language_mutated": False,
        "language_state_after": language_states[0] if language_states else {},
        "predictions_path": str(pred_path),
        "official_eval_path": str(eval_path),
        "child_run_ids": [f"{args.run_id}__chunk{index:02d}" for index in range(workers)],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"HMS.mean={summary['HMS_mean_100']:.2f} N={len(evaluations)}", flush=True)


if __name__ == "__main__":
    main()
