"""Build and optionally execute a theory-guided 2x2 benchmark matrix.

The matrix is:

  4o-mini raw       4o raw
  4o-mini + MARS    4o + MARS

`raw` means the closest local baseline runner without BenchmarkTheory guidance
and with minimal MARS scaffolding where the local runner requires a Coordinator.
`+MARS` means the same core model with MARS-full and the self-induced
BenchmarkTheory injected into the Generator prompt/runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))


MODELS = {
    "4o-mini": "openai/gpt-4o-mini",
    "4o": "openai/gpt-4o",
}


@dataclass(frozen=True)
class Theory2x2Cell:
    benchmark: str
    model_label: str
    core_model: str
    system_label: str
    uses_mars: bool
    uses_benchmark_theory: bool
    runner_kind: str
    command: tuple[str, ...]
    env: dict[str, str]
    score_target: str
    status: str
    notes: str


def _base_env(benchmark: str, theory_path: Path, *, use_theory: bool) -> dict[str, str]:
    env = {
        "PYTHONPATH": str(_PROJ),
    }
    if use_theory:
        env.update(
            {
                "MARS_USE_BENCHMARK_THEORY": "1",
                "MARS_BENCHMARK_THEORY_PATH": str(theory_path),
                "MARS_BENCHMARK_THEORY_NAME": benchmark,
            }
        )
    return env


def _model_env(model: str) -> dict[str, str]:
    return {
        "MARS_GENERATOR_MODEL": model,
        "MARS_REFLECTOR_MODEL": model,
    }


def _cell(
    *,
    benchmark: str,
    model_label: str,
    system_label: str,
    uses_mars: bool,
    runner_kind: str,
    command: list[str],
    env: dict[str, str],
    score_target: str,
    status: str = "planned",
    notes: str = "",
) -> Theory2x2Cell:
    core_model = MODELS[model_label]
    return Theory2x2Cell(
        benchmark=benchmark,
        model_label=model_label,
        core_model=core_model,
        system_label=system_label,
        uses_mars=uses_mars,
        uses_benchmark_theory=uses_mars,
        runner_kind=runner_kind,
        command=tuple(command),
        env={**_model_env(core_model), **env},
        score_target=score_target,
        status=status,
        notes=notes,
    )


def build_plan(theory_path: Path, *, smoke: bool = True) -> list[Theory2x2Cell]:
    py = sys.executable
    cells: list[Theory2x2Cell] = []

    for model_label in MODELS:
        for uses_mars, system_label, ablation in [
            (False, f"{model_label} raw", "MARS-all-off"),
            (True, f"{model_label}+MARS", "MARS-full"),
        ]:
            env = _base_env("DiscoveryBench", theory_path, use_theory=uses_mars)
            env.update(
                {
                    "MARS_DB_N": "1" if smoke else "25",
                    "MARS_DB_BUDGET": "8" if smoke else "12",
                    "MARS_ABLATIONS": ablation,
                    "MARS_RUN_LABEL": f"theory2x2_{model_label.replace('-', '')}_{'mars' if uses_mars else 'raw'}",
                    "OLS_DB_JUDGE_MODEL": "openai/gpt-4o",
                }
            )
            cells.append(
                _cell(
                    benchmark="DiscoveryBench",
                    model_label=model_label,
                    system_label=system_label,
                    uses_mars=uses_mars,
                    runner_kind="mars_db_train_proxy",
                    command=[py, "-m", "mars.runners.run_db"],
                    env=env,
                    score_target="HMS",
                    notes="Local train/proxy runner; official DB-CPI full run remains separate.",
                )
            )

            env = _base_env("NewtonBench", theory_path, use_theory=uses_mars)
            env.update(
                {
                    "MARS_NB_MODULES": "m0_gravity" if smoke else "m0_gravity,m1_coulomb_force,m3_fourier_law,m4_snell_law,m5_radioactive_decay,m7_malus_law,m8_sound_speed,m9_hooke_law,m11_heat_transfer",
                    "MARS_NB_DIFFICULTIES": "easy" if smoke else "easy,medium,hard",
                    "MARS_NB_LAW_VERSIONS": "v0" if smoke else "v0,v1,v2",
                    "MARS_NB_SYSTEMS": "vanilla_equation" if smoke else "vanilla_equation,simple_system,complex_system",
                    "MARS_NB_BUDGET": "6" if smoke else "10",
                    "MARS_NB_EPISODE_TIMEOUT_S": "180" if smoke else "240",
                    "MARS_NB_RESUME": "1",
                    "MARS_ABLATIONS": ablation,
                    "MARS_NB_JUDGE_MODEL": "gpt41",
                    "MARS_RUN_LABEL": f"theory2x2_{model_label.replace('-', '')}_{'mars' if uses_mars else 'raw'}",
                }
            )
            cells.append(
                _cell(
                    benchmark="NewtonBench",
                    model_label=model_label,
                    system_label=system_label,
                    uses_mars=uses_mars,
                    runner_kind="mars_nb_official_compatible",
                    command=[py, "-m", "mars.runners.run_nb"],
                    env=env,
                    score_target="SA/RMSLE",
                    notes="Coordinator-based NewtonBench runner; CPI full-grid runner should be used for final law-induction claim.",
                )
            )

            env = _base_env("UltraHorizon", theory_path, use_theory=uses_mars)
            run_id = f"theory2x2_uh_seq_{model_label.replace('-', '')}_{'mars' if uses_mars else 'raw'}"
            cells.append(
                _cell(
                    benchmark="UltraHorizon",
                    model_label=model_label,
                    system_label=system_label,
                    uses_mars=uses_mars,
                    runner_kind="uh_official_env",
                    command=[
                        py,
                        "-m",
                        "mars.runners.run_uh_official",
                        "--env",
                        "seq",
                        "--run_id",
                        run_id,
                        "--steps",
                        "5" if smoke else "7",
                        "--action_budget",
                        "8" if smoke else "12",
                        "--seeds",
                        "42" if smoke else "17,42,101",
                        "--difficulty",
                        "hard",
                        "--ablation",
                        ablation,
                        "--generator_model",
                        MODELS[model_label],
                        "--reflector_model",
                        MODELS[model_label],
                        "--judge_model",
                        "openai/gpt-4o",
                        "--overwrite",
                    ],
                    env=env,
                    score_target="official-env-compatible score",
                    notes="Sequence only in smoke plan; full UltraHorizon needs grid+seq+bio.",
                )
            )

            env = _base_env("ScienceAgentBench", theory_path, use_theory=uses_mars)
            run_id = f"theory2x2_sab_verified4_{model_label.replace('-', '')}_{'mars' if uses_mars else 'raw'}"
            cells.append(
                _cell(
                    benchmark="ScienceAgentBench",
                    model_label=model_label,
                    system_label=system_label,
                    uses_mars=uses_mars,
                    runner_kind="sab_official_export_pre_eval",
                    command=[
                        py,
                        "-m",
                        "mars.runners.run_sab_official_export",
                        "--run_id",
                        run_id,
                        "--instance_ids",
                        "1",
                        "2",
                        "3",
                        "4",
                        "--ablation",
                        ablation,
                        "--generator_model",
                        MODELS[model_label],
                        "--reflector_model",
                        MODELS[model_label],
                        "--budget",
                        "4",
                        "--overwrite",
                    ],
                    env=env,
                    score_target="official export; SR/VER requires Docker eval",
                    notes="Export is runnable; official SR/VER remains blocked if Docker daemon is unavailable.",
                )
            )

    return cells


def _markdown(cells: list[Theory2x2Cell]) -> str:
    lines = [
        "# Theory-Guided 2x2 Benchmark Run Plan",
        "",
        "| Benchmark | Cell | Core model | Theory? | Runner | Score target | Status | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        theory = "yes" if cell.uses_benchmark_theory else "no"
        lines.append(
            f"| {cell.benchmark} | {cell.system_label} | {cell.core_model} | {theory} | "
            f"`{cell.runner_kind}` | {cell.score_target} | {cell.status} | {cell.notes} |"
        )
    lines += [
        "",
        "## Execution Command Template",
        "",
        "Each cell stores its exact command and environment in `plan.json`.",
        "For `+MARS` cells, the environment includes:",
        "",
        "```text",
        "MARS_USE_BENCHMARK_THEORY=1",
        "MARS_BENCHMARK_THEORY_PATH=<compiled theories.json>",
        "MARS_BENCHMARK_THEORY_NAME=<benchmark>",
        "```",
    ]
    return "\n".join(lines)


def _run_cell(cell: Theory2x2Cell, *, cwd: Path, timeout_s: int | None) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(cell.env)
    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            list(cell.command),
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
        returncode = proc.returncode
        output = proc.stdout
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        output = out + f"\n[TIMEOUT] cell exceeded timeout_s={timeout_s}\n"
    return {
        "cell": asdict(cell),
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_time_s": time.time() - t0,
        "output_tail": output[-6000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="theory_2x2_plan_20260618")
    parser.add_argument("--theory_path", default=str(_PROJ / "lmw" / "benchmark_theory" / "four_benchmark_theory_20260618" / "theories.json"))
    parser.add_argument("--full", action="store_true", help="Plan larger runs instead of smoke defaults.")
    parser.add_argument("--execute", action="store_true", help="Execute selected cells after writing the plan.")
    parser.add_argument("--benchmarks", default="", help="Comma-separated benchmark filter for execution.")
    parser.add_argument("--systems", default="", help="Comma-separated system label filter for execution.")
    parser.add_argument("--timeout_s", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    theory_path = Path(args.theory_path)
    out_dir = _PROJ / "lmw" / "theory_2x2" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "plan.json"
    md_path = out_dir / "plan.md"
    results_path = out_dir / "results.jsonl"
    if plan_path.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {out_dir}")

    cells = build_plan(theory_path, smoke=not args.full)
    plan = {
        "run_id": args.run_id,
        "theory_path": str(theory_path),
        "mode": "full" if args.full else "smoke",
        "n_cells": len(cells),
        "cells": [asdict(c) for c in cells],
    }
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_markdown(cells), encoding="utf-8")

    print("=== theory 2x2 plan ===")
    print(f"cells={len(cells)} mode={plan['mode']}")
    print(f"plan={plan_path}")
    print(f"md={md_path}")

    if not args.execute:
        return

    bench_filter = {x.strip() for x in args.benchmarks.split(",") if x.strip()}
    system_filter = {x.strip() for x in args.systems.split(",") if x.strip()}
    selected = [
        c
        for c in cells
        if (not bench_filter or c.benchmark in bench_filter)
        and (not system_filter or c.system_label in system_filter)
    ]
    timeout = args.timeout_s if args.timeout_s > 0 else None
    with results_path.open("w", encoding="utf-8") as f:
        for i, cell in enumerate(selected, 1):
            print(f"[{i}/{len(selected)}] {cell.benchmark} {cell.system_label} ...", flush=True)
            row = _run_cell(cell, cwd=_PROJ, timeout_s=timeout)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            f.flush()
            print(f"  returncode={row['returncode']} wall={row['wall_time_s']:.1f}s", flush=True)
    print(f"results={results_path}")


if __name__ == "__main__":
    main()
