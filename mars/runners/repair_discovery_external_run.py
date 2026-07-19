"""Repair external DiscoveryBench baseline runs without changing good rows.

The script builds a complete 239-task artifact from one or more prior run
directories. Existing valid predictions/evaluations are preserved. Rows whose
prediction is an API error, rows with evaluator errors, and missing rows are
regenerated/evaluated and checkpointed after every repaired task.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJ, _PROJ / "discoverybench_repo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv

    env_path = _PROJ / "autodiscovery" / ".env.local"
    if env_path.exists():
        load_dotenv(env_path, override=False)
except ImportError:
    pass

from mars.agents.base import make_openai_client  # noqa: E402
from mars.runners.run_db_official_eval import _run_official_eval, load_official_real_tasks  # noqa: E402
from mars.runners.run_discovery_external_baselines import _call_baseline, _task_context  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_bad_prediction(row: dict[str, Any] | None) -> bool:
    if not row:
        return True
    blob = " ".join(str(row.get(k, "")) for k in ("pred_hypo", "pred_workflow", "raw_response"))
    needles = (
        "OpenAI-compatible HTTP 402",
        "Insufficient credits",
        "can only afford",
    )
    return any(x in blob for x in needles)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _checkpoint(
    *,
    out_dir: Path,
    tasks: list[Any],
    pred_by_key: dict[str, dict[str, Any]],
    eval_by_key: dict[str, dict[str, Any]],
    summary_extra: dict[str, Any],
) -> None:
    pred_rows = [pred_by_key[t.task_key] for t in tasks if t.task_key in pred_by_key]
    eval_rows = [eval_by_key[t.task_key] for t in tasks if t.task_key in eval_by_key]
    _write_jsonl(out_dir / "predictions.jsonl", pred_rows)
    _write_jsonl(out_dir / "official_eval.jsonl", eval_rows)
    scores = [float(row.get("HMS_100", 0.0)) for row in eval_rows]
    summary = {
        **summary_extra,
        "n_predictions": len(pred_rows),
        "n": len(eval_rows),
        "coverage_ok": len(eval_rows) == len(tasks) and len(pred_rows) == len(tasks),
        "mean_HMS": float(st.mean(scores)) if scores else 0.0,
        "errors": sum(1 for row in eval_rows if "error" in row),
        "bad_predictions_remaining": sum(_is_bad_prediction(pred_by_key.get(t.task_key)) for t in tasks),
        "prediction_path": str(out_dir / "predictions.jsonl"),
        "official_eval_path": str(out_dir / "official_eval.jsonl"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_run_dirs", nargs="+", required=True)
    ap.add_argument("--out_run_id", required=True)
    ap.add_argument("--method", choices=["direct", "stats", "react", "selfdebug", "codeact"], required=True)
    ap.add_argument("--model", default=os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini"))
    ap.add_argument("--judge_model", default=os.environ.get("MARS_DB_OFFICIAL_JUDGE_MODEL", "openai/gpt-4o"))
    ap.add_argument("--max_repairs", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tasks = load_official_real_tasks(
        _PROJ / "discoverybench_repo",
        max_tasks=None,
        datasets=None,
        task_keys=None,
        start_index=0,
    )
    task_by_key = {task.task_key: task for task in tasks}
    out_dir = _PROJ / "lmw" / "discovery_external" / args.out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in ("predictions.jsonl", "official_eval.jsonl", "summary.json", "repair_trace.log"):
            path = out_dir / name
            if path.exists():
                path.unlink()

    pred_by_key: dict[str, dict[str, Any]] = {}
    eval_by_key: dict[str, dict[str, Any]] = {}
    for raw_dir in args.source_run_dirs:
        src = Path(raw_dir)
        if not src.is_absolute():
            src = _PROJ / src
        for row in _read_jsonl(src / "predictions.jsonl"):
            if row.get("task_key") in task_by_key:
                pred_by_key[row["task_key"]] = row
        for row in _read_jsonl(src / "official_eval.jsonl"):
            if row.get("task_key") in task_by_key:
                eval_by_key[row["task_key"]] = row

    client = make_openai_client()
    repaired = 0
    summary_extra = {
        "run_id": args.out_run_id,
        "method": args.method,
        "model": args.model,
        "judge_model": args.judge_model,
        "source_run_dirs": args.source_run_dirs,
        "created_unix": time.time(),
    }
    _checkpoint(out_dir=out_dir, tasks=tasks, pred_by_key=pred_by_key, eval_by_key=eval_by_key, summary_extra=summary_extra)

    with (out_dir / "repair_trace.log").open("a", encoding="utf-8") as trace_f:
        for task in tasks:
            key = task.task_key
            pred = pred_by_key.get(key)
            ev = eval_by_key.get(key)
            needs_prediction = _is_bad_prediction(pred)
            needs_eval = needs_prediction or ev is None or "error" in ev
            if not needs_eval:
                continue
            if args.max_repairs > 0 and repaired >= args.max_repairs:
                break

            if needs_prediction:
                context = _task_context(task, method=args.method)
                hypo, workflow, raw = _call_baseline(client, args.model, context, method=args.method)
                pred = {
                    "task_key": task.task_key,
                    "dataset": task.dataset_name,
                    "metadata_id": task.metadata_id,
                    "query_id": task.query_id,
                    "method": args.method,
                    "pred_hypo": hypo,
                    "pred_workflow": workflow,
                    "raw_response": raw[:3000],
                    "repair_generated": True,
                }
                pred_by_key[key] = pred

            row = {
                "task_key": task.task_key,
                "dataset": task.dataset_name,
                "metadata_id": task.metadata_id,
                "query_id": task.query_id,
                "method": args.method,
                "judge_model": args.judge_model,
            }
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    scored = _run_official_eval(
                        task,
                        pred_hypo=str(pred.get("pred_hypo", "")),
                        pred_workflow=str(pred.get("pred_workflow", "")),
                        judge_model=args.judge_model,
                        trace_f=trace_f,
                    )
                row.update(scored)
                row["HMS_100"] = 100.0 * float(scored.get("final_score", 0.0))
                print(f"repaired {key} HMS={row['HMS_100']:.2f}")
            except Exception as exc:
                row.update({"error": f"{type(exc).__name__}: {exc}", "HMS_100": 0.0})
                print(f"still_error {key}: {row['error']}")
            eval_by_key[key] = row
            repaired += 1
            summary_extra["repaired_rows"] = repaired
            _checkpoint(out_dir=out_dir, tasks=tasks, pred_by_key=pred_by_key, eval_by_key=eval_by_key, summary_extra=summary_extra)

    _checkpoint(out_dir=out_dir, tasks=tasks, pred_by_key=pred_by_key, eval_by_key=eval_by_key, summary_extra=summary_extra)
    final = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    print(
        "summary "
        f"pred={final['n_predictions']} eval={final['n']} "
        f"HMS={final['mean_HMS']:.2f} errors={final['errors']} "
        f"bad_pred={final['bad_predictions_remaining']}"
    )


if __name__ == "__main__":
    main()
