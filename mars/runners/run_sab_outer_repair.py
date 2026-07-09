"""Apply the universal outer-loop repair compiler to SAB prediction files."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from mars.runners.run_sab_official_eval import _static_program_checks  # noqa: E402
from mars.skills.code_repair_compiler import compile_code_repairs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="sab_outer_repair")
    parser.add_argument("--pred_program_path", required=True)
    parser.add_argument("--benchmark_path", default=str(_PROJ / "scienceagentbench_repo" / "benchmark"))
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_dir = Path(args.pred_program_path)
    benchmark_path = Path(args.benchmark_path)
    out_dir = Path(args.out_dir) if args.out_dir else _PROJ / "lmw" / "sab_official" / args.run_id
    pred_out = out_dir / "pred_programs"
    summary_path = out_dir / "summary.json"

    if summary_path.exists() and not args.overwrite:
        raise SystemExit(f"summary exists; pass --overwrite: {summary_path}")
    if pred_out.exists() and args.overwrite:
        shutil.rmtree(pred_out)
    pred_out.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(src_dir.glob("pred_*.py"))
    before = _static_program_checks(pred_files=pred_files, benchmark_path=benchmark_path)
    rows: list[dict] = []
    t0 = time.time()
    for pred in pred_files:
        source = pred.read_text(encoding="utf-8", errors="ignore")
        repaired = compile_code_repairs(
            source,
            benchmark_path=benchmark_path,
            output_path="",
        )
        out_path = pred_out / pred.name
        out_path.write_text(repaired.source, encoding="utf-8")
        rows.append(
            {
                "program": pred.name,
                "changed": repaired.source != source,
                "repairs": list(repaired.repairs),
                "warnings": list(repaired.warnings),
                "chars_before": len(source),
                "chars_after": len(repaired.source),
            }
        )

    after_files = sorted(pred_out.glob("pred_*.py"))
    after = _static_program_checks(pred_files=after_files, benchmark_path=benchmark_path)
    summary = {
        "run_id": args.run_id,
        "score_type": "outer-loop-static-repair",
        "source_pred_program_path": str(src_dir),
        "pred_program_path": str(pred_out),
        "benchmark_path": str(benchmark_path),
        "n_programs": len(pred_files),
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
    print(f"changed={summary['n_changed']}/{len(pred_files)}")
    print(f"static_before={before['n_static_warnings']} static_after={after['n_static_warnings']}")


def _counts(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for item in row.get(key, []):
            counts[str(item)] = counts.get(str(item), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


if __name__ == "__main__":
    main()
