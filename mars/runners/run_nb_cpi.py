"""Run NewtonBench CPI trajectory-to-law induction.

This runner is intentionally deterministic. It tests the SB-CPI idea on
NewtonBench by building a measurement analyzer for non-vanilla systems, fitting
a compact law program, and evaluating it with NewtonBench's own evaluator.
"""

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

from mars.induction.nb_cpi import run_nb_cpi_task  # noqa: E402


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=os.environ.get("MARS_NB_CPI_RUN_ID", "nb_cpi_smoke"))
    parser.add_argument("--modules", default="m0_gravity")
    parser.add_argument("--difficulties", default="easy,medium,hard")
    parser.add_argument("--law_versions", default="v0,v1,v2")
    parser.add_argument("--systems", default="vanilla_equation,simple_system,complex_system")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--judge_model", default=os.environ.get("MARS_NB_JUDGE_MODEL", "gpt41"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = _PROJ / "lmw" / "nb_cpi" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run.jsonl"
    summary_path = out_dir / "summary.json"
    if run_log.exists() and not args.overwrite:
        raise SystemExit(f"run log exists; pass --overwrite: {run_log}")
    if args.overwrite and run_log.exists():
        run_log.unlink()

    modules = _parse_csv(args.modules)
    difficulties = _parse_csv(args.difficulties)
    law_versions = _parse_csv(args.law_versions)
    systems = _parse_csv(args.systems)

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    with run_log.open("w", encoding="utf-8") as f:
        for module in modules:
            for difficulty in difficulties:
                for law_version in law_versions:
                    for system in systems:
                        row = run_nb_cpi_task(
                            module_name=module,
                            difficulty=difficulty,
                            law_version=law_version,
                            system=system,
                            judge_model=args.judge_model,
                            noise_level=args.noise,
                        )
                        rows.append(row)
                        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                        f.flush()
                        print(
                            f"{module}/{difficulty}/{law_version}/{system} "
                            f"SA={row['SA']:.1f} rmsle={row['rmsle']} "
                            f"kind={(row.get('best') or {}).get('kind')}",
                            flush=True,
                        )

    scores = [float(r.get("SA", 0.0)) for r in rows]
    nas = [float(r.get("numerical_accuracy", 0.0)) for r in rows]
    by_system: dict[str, dict[str, Any]] = {}
    for system in systems:
        subset = [r for r in rows if r["system"] == system]
        by_system[system] = {
            "n": len(subset),
            "SA_mean": st.fmean([float(r.get("SA", 0.0)) for r in subset]) if subset else 0.0,
            "numerical_accuracy_mean": st.fmean([float(r.get("numerical_accuracy", 0.0)) for r in subset]) if subset else 0.0,
            "symbolic_match": sum(1 for r in subset if r.get("symbolic_equivalent")),
        }

    summary = {
        "run_id": args.run_id,
        "score_type": "official-evaluator-compatible",
        "modules": modules,
        "difficulties": difficulties,
        "law_versions": law_versions,
        "systems": systems,
        "noise": args.noise,
        "judge_model": args.judge_model,
        "n": len(rows),
        "SA_mean": st.fmean(scores) if scores else 0.0,
        "numerical_accuracy_mean": st.fmean(nas) if nas else 0.0,
        "symbolic_match": sum(1 for r in rows if r.get("symbolic_equivalent")),
        "by_system": by_system,
        "wall_time_s": time.time() - t0,
        "paths": {"run_log": str(run_log), "summary": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n=== NewtonBench CPI summary ===")
    print(f"n={summary['n']} SA_mean={summary['SA_mean']:.3f} num_acc={summary['numerical_accuracy_mean']:.3f}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
