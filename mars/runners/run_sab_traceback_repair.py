"""Build the next SAB program generation from execution traces.

This runner is SAB-shaped only at the I/O boundary: it reads the official
ScienceAgentBench run logs to map instance ids back to generated program files.
The repair operators themselves are benchmark-agnostic and live in
``mars.skills.code_repair_compiler``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from mars.runners.run_sab_official_eval import _static_program_checks  # noqa: E402
from mars.skills.code_repair_compiler import compile_code_repairs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="sab_traceback_repair")
    parser.add_argument("--pred_program_path", required=True)
    parser.add_argument(
        "--run_log_dir",
        nargs="+",
        required=True,
        help="One or more official run log directories whose traces form the outer-loop evidence.",
    )
    parser.add_argument("--benchmark_path", default=str(_PROJ / "scienceagentbench_repo" / "benchmark"))
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_dir = Path(args.pred_program_path).resolve()
    log_dirs = [Path(p).resolve() for p in args.run_log_dir]
    benchmark_path = Path(args.benchmark_path).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else _PROJ / "lmw" / "sab_official" / args.run_id
    pred_out = out_dir / "pred_programs"
    summary_path = out_dir / "summary.json"

    if summary_path.exists() and not args.overwrite:
        raise SystemExit(f"summary exists; pass --overwrite: {summary_path}")
    if pred_out.exists() and args.overwrite:
        shutil.rmtree(pred_out)
    pred_out.mkdir(parents=True, exist_ok=True)

    failures = _load_failures(log_dirs)
    pred_files = sorted(src_dir.glob("pred_*.py"))
    before = _static_program_checks(pred_files=pred_files, benchmark_path=benchmark_path)
    rows: list[dict[str, Any]] = []
    t0 = time.time()

    for pred in pred_files:
        source = pred.read_text(encoding="utf-8", errors="ignore")
        failure_text = failures.get(pred.name, "")
        repaired = compile_code_repairs(
            source,
            benchmark_path=benchmark_path,
            output_path="",
            failure_text=failure_text,
        )
        (pred_out / pred.name).write_text(repaired.source, encoding="utf-8")
        rows.append(
            {
                "program": pred.name,
                "had_failure_trace": bool(failure_text),
                "changed": repaired.source != source,
                "repairs": list(repaired.repairs),
                "warnings": list(repaired.warnings),
                "failure_head": failure_text[:500],
                "chars_before": len(source),
                "chars_after": len(repaired.source),
            }
        )

    after_files = sorted(pred_out.glob("pred_*.py"))
    after = _static_program_checks(pred_files=after_files, benchmark_path=benchmark_path)
    summary = {
        "run_id": args.run_id,
        "score_type": "outer-loop-traceback-repair",
        "source_pred_program_path": str(src_dir),
        "run_log_dirs": [str(p) for p in log_dirs],
        "pred_program_path": str(pred_out),
        "benchmark_path": str(benchmark_path),
        "n_programs": len(pred_files),
        "n_failure_traces": sum(1 for row in rows if row["had_failure_trace"]),
        "n_changed": sum(1 for row in rows if row["changed"]),
        "repair_counts": _counts(rows, "repairs"),
        "warning_counts": _counts(rows, "warnings"),
        "static_before": before,
        "static_after": after,
        "rows": rows,
        "wall_time_s": time.time() - t0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary={summary_path}")
    print(f"pred_program_path={pred_out}")
    print(f"failure_traces={summary['n_failure_traces']}/{len(pred_files)}")
    print(f"changed={summary['n_changed']}/{len(pred_files)}")
    print(f"repair_counts={summary['repair_counts']}")


def _load_failures(log_dirs: list[Path]) -> dict[str, str]:
    failures: dict[str, str] = {}
    for log_dir in log_dirs:
        for input_path in sorted(log_dir.glob("*/input/input.json")):
            try:
                spec = json.loads(input_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            gold_name = Path(str(spec.get("gold_program_name", ""))).name
            if not gold_name:
                continue
            pred_name = "pred_" + gold_name
            instance_dir = input_path.parent.parent
            failure_text = _failure_text_for(instance_dir)
            if failure_text:
                failures[pred_name] = "\n".join(
                    part for part in (failures.get(pred_name, ""), failure_text) if part
                )
    return failures


def _failure_text_for(instance_dir: Path) -> str:
    result_path = instance_dir / "output" / "result.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            result = None
        if isinstance(result, list) and len(result) >= 4:
            return str(result[3])
        if isinstance(result, dict):
            return str(result.get("log_info", ""))
    log_path = instance_dir / "run_instance.log"
    if log_path.exists():
        return "\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-160:])
    return ""


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for item in row.get(key, []):
            counts[str(item)] = counts.get(str(item), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


if __name__ == "__main__":
    main()
