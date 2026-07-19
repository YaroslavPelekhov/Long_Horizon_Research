"""Retry failed official HMS evaluations for external DiscoveryBench baselines."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics as st
import sys
import time
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from dotenv import load_dotenv

    env_path = _PROJ / "autodiscovery" / ".env.local"
    if env_path.exists():
        load_dotenv(env_path, override=False)
except ImportError:
    pass

from mars.runners.run_db_official_eval import _run_official_eval, load_official_real_tasks  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_summary(summary_path: Path, summary: dict, rows: list[dict], eval_path: Path) -> None:
    scores = [float(row.get("HMS_100", 0.0)) for row in rows]
    summary.update(
        {
            "mean_HMS": float(st.mean(scores)) if scores else 0.0,
            "errors": sum(1 for row in rows if "error" in row),
            "official_eval_path": str(eval_path),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--max_tasks", type=int, default=30)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--judge_model", default="openai/gpt-4o")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = _PROJ / run_dir
    eval_path = run_dir / "official_eval.jsonl"
    pred_path = run_dir / "predictions.jsonl"
    summary_path = run_dir / "summary.json"
    trace_path = run_dir / "eval_trace_retry.log"

    rows = _read_jsonl(eval_path)
    preds = {row["task_key"]: row for row in _read_jsonl(pred_path)}
    tasks = {
        task.task_key: task
        for task in load_official_real_tasks(
            _PROJ / "discoverybench_repo",
            max_tasks=args.max_tasks if args.max_tasks > 0 else None,
            datasets=None,
            task_keys=None,
            start_index=args.start_index,
        )
    }

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    initial_fixed = int(summary.get("retry_fixed_evaluator_errors", 0))
    fixed = 0
    with trace_path.open("a", encoding="utf-8") as trace_f:
        for row in rows:
            if "error" not in row:
                continue
            key = row["task_key"]
            pred = preds[key]
            task = tasks[key]
            last_error = str(row["error"])
            for attempt in range(args.retries):
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        ev = _run_official_eval(
                            task,
                            pred_hypo=pred.get("pred_hypo", ""),
                            pred_workflow=pred.get("pred_workflow", ""),
                            judge_model=args.judge_model,
                            trace_f=trace_f,
                        )
                    row.pop("error", None)
                    row.update(ev)
                    row["HMS_100"] = 100.0 * float(ev.get("final_score", 0.0))
                    fixed += 1
                    summary["retry_fixed_evaluator_errors"] = initial_fixed + fixed
                    _write_jsonl(eval_path, rows)
                    _write_summary(summary_path, summary, rows, eval_path)
                    print(f"fixed {key} HMS={row['HMS_100']:.2f}")
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    time.sleep(2.0 * (attempt + 1))
            else:
                row["error"] = last_error
                _write_jsonl(eval_path, rows)
                _write_summary(summary_path, summary, rows, eval_path)
                print(f"still_error {key}: {last_error}")

    summary["retry_fixed_evaluator_errors"] = initial_fixed + fixed
    _write_jsonl(eval_path, rows)
    _write_summary(summary_path, summary, rows, eval_path)
    print(f"summary mean_HMS={summary['mean_HMS']:.2f} errors={summary['errors']} fixed={fixed}")


if __name__ == "__main__":
    main()
