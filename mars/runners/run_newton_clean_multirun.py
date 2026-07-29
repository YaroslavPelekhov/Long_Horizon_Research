"""Prepare or execute matched full-grid NewtonBench repetitions.

Every repetition evaluates all 324 configurations with one fixed judge and no
selective rejudging.  Preparation is the default; use ``--execute`` to run.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_PROJ = Path(__file__).resolve().parents[2]
_MODULES = (
    "m0_gravity,m1_coulomb_force,m2_magnetic_force,m3_fourier_law,"
    "m4_snell_law,m5_radioactive_decay,m6_underdamped_harmonic,"
    "m7_malus_law,m8_sound_speed,m9_hooke_law,m10_be_distribution,"
    "m11_heat_transfer"
)


def _command(run_id: str, model: str, judge_model: str, seed: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mars.runners.run_nb_activeprobe",
        "--run_id",
        run_id,
        "--model",
        model,
        "--judge_model",
        judge_model,
        "--seed",
        str(seed),
        "--modules",
        _MODULES,
        "--difficulties",
        "easy,medium,hard",
        "--law_versions",
        "v0,v1,v2",
        "--systems",
        "vanilla_equation,simple_system,complex_system",
        "--overwrite",
    ]


def _summary_path(run_id: str) -> Path:
    return _PROJ / "lmw" / "nb_activeprobe" / run_id / "summary.json"


def _read_summary(path: Path, *, expected_judge: str, expected_seed: int) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if int(value.get("n", -1)) != 324:
        raise ValueError(f"{path}: expected n=324, found {value.get('n')}")
    if str(value.get("judge_model")) != expected_judge:
        raise ValueError(f"{path}: judge mismatch")
    if int(value.get("seed", -1)) != expected_seed:
        raise ValueError(f"{path}: seed mismatch")
    if "audited_SA_all" in value or "rejudge_file" in value:
        raise ValueError(f"{path}: selective rejudge artifact is not admissible")
    return value


def _aggregate(run_id: str, model: str, judge_model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    sa_all = [float(row["SA_all"]) for row in rows]
    sa_answered = [float(row["SA_answered"]) for row in rows]
    return {
        "protocol_version": "newton-full324-multirun-v1",
        "run_id": run_id,
        "model": model,
        "judge_model": judge_model,
        "n_repetitions": len(rows),
        "n_per_repetition": 324,
        "n_total_judgments": 324 * len(rows),
        "selective_rejudge": False,
        "SA_all_mean": st.fmean(sa_all),
        "SA_all_sd": st.stdev(sa_all) if len(sa_all) > 1 else 0.0,
        "SA_answered_mean": st.fmean(sa_answered),
        "SA_answered_sd": st.stdev(sa_answered) if len(sa_answered) > 1 else 0.0,
        "runs": [
            {
                "seed": row["seed"],
                "SA_all": row["SA_all"],
                "SA_answered": row["SA_answered"],
                "answered": row["answered"],
                "abstained": row["abstained"],
            }
            for row in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=f"aaai27_newton_clean_{time.strftime('%Y%m%d')}")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--judge_model", default="gpt41")
    parser.add_argument("--seeds", default="2027,2028,2029,2030")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise SystemExit("no seeds")
    out_dir = _PROJ / "lmw" / "newton_clean_multirun" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = []
    for seed in seeds:
        child_id = f"{args.run_id}_seed{seed}"
        plan.append(
            {
                "seed": seed,
                "run_id": child_id,
                "command": _command(child_id, args.model, args.judge_model, seed),
                "summary": str(_summary_path(child_id)),
            }
        )
    (out_dir / "protocol.json").write_text(
        json.dumps(
            {
                "protocol_version": "newton-full324-multirun-v1",
                "model": args.model,
                "judge_model": args.judge_model,
                "seeds": seeds,
                "grid": {
                    "modules": 12,
                    "difficulties": ["easy", "medium", "hard"],
                    "law_versions": ["v0", "v1", "v2"],
                    "systems": ["vanilla_equation", "simple_system", "complex_system"],
                    "n": 324,
                },
                "selective_rejudge": False,
                "runs": plan,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"protocol: {out_dir / 'protocol.json'}")
    if not args.execute:
        print(f"prepared {len(plan)} x 324 runs; pass --execute to start")
        return

    summaries: list[dict[str, Any]] = []
    for item in plan:
        path = Path(item["summary"])
        if not (args.resume and path.exists()):
            print(f"\n=== Newton seed {item['seed']} ===", flush=True)
            proc = subprocess.run(item["command"], cwd=_PROJ)
            if proc.returncode != 0:
                raise SystemExit(f"Newton child failed for seed {item['seed']}")
        summaries.append(
            _read_summary(
                path,
                expected_judge=args.judge_model,
                expected_seed=int(item["seed"]),
            )
        )

    aggregate = _aggregate(args.run_id, args.model, args.judge_model, summaries)
    (out_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    with (out_dir / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("seed", "SA_all", "SA_answered", "answered", "abstained"),
        )
        writer.writeheader()
        writer.writerows(aggregate["runs"])
    print(
        f"SA_all={aggregate['SA_all_mean']:.4f} +/- {aggregate['SA_all_sd']:.4f} "
        f"over {len(summaries)} complete runs"
    )


if __name__ == "__main__":
    main()
