"""Aggregate pre-registered residual-language transfer runs.

This utility performs no model calls.  It checks that each paired run uses the
same source and held-out split, then reports seed-level and task-level effects
for the residual-selected operator and a type-compatible counterfactual.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


_PROJECT = Path(__file__).resolve().parents[2]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def _load(run_id: str) -> dict[str, Any]:
    path = _PROJECT / "lmw" / "language_transfer" / run_id / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_artifact"] = str(path)
    return payload


def _validate_pair(selected: dict[str, Any], control: dict[str, Any]) -> None:
    for field in ("seed", "source_modules", "heldout_modules", "difficulty", "law_version", "system"):
        if selected.get(field) != control.get(field):
            raise ValueError(f"paired runs disagree on {field}: {selected.get(field)!r} != {control.get(field)!r}")
    if control.get("counterfactual_family") != "positive_additive_lattice":
        raise ValueError("control arm is not the declared additive counterfactual")
    if selected.get("counterfactual_family") is not None:
        raise ValueError("selected arm must not override the induced operator")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected_runs",
        default=(
            "aaai27_newton_transfer_geometry_pilot_seed13,"
            "aaai27_newton_transfer_geometry_seed31,"
            "aaai27_newton_transfer_geometry_seed57"
        ),
    )
    parser.add_argument(
        "--control_runs",
        default=(
            "aaai27_newton_transfer_additive_control_seed13,"
            "aaai27_newton_transfer_additive_control_seed31,"
            "aaai27_newton_transfer_additive_control_seed57"
        ),
    )
    parser.add_argument("--run_id", default="aaai27_newton_language_growth_control_v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selected_ids = [item.strip() for item in args.selected_runs.split(",") if item.strip()]
    control_ids = [item.strip() for item in args.control_runs.split(",") if item.strip()]
    if not selected_ids or len(selected_ids) != len(control_ids):
        raise SystemExit("selected and control run lists must have equal nonzero length")
    selected = [_load(run_id) for run_id in selected_ids]
    controls = [_load(run_id) for run_id in control_ids]
    for selected_run, control_run in zip(selected, controls):
        _validate_pair(selected_run, control_run)

    selected_gain = [float(run["mean_relative_gain"]) for run in selected]
    control_gain = [float(run["mean_relative_gain"]) for run in controls]
    task_rows = []
    for selected_run, control_run in zip(selected, controls):
        for selected_row, control_row in zip(selected_run["paired"], control_run["paired"]):
            if selected_row["module"] != control_row["module"]:
                raise ValueError("paired task ordering differs between selected and control arms")
            task_rows.append({
                "seed": selected_run["seed"],
                "module": selected_row["module"],
                "l0_loss": float(selected_row["l0_loss"]),
                "selected_loss": float(selected_row["l1_loss"]),
                "control_loss": float(control_row["l1_loss"]),
                "selected_gain": float(selected_row["relative_gain"]),
                "control_gain": float(control_row["relative_gain"]),
            })
    selected_task_gain = [row["selected_gain"] for row in task_rows]
    control_task_gain = [row["control_gain"] for row in task_rows]
    differences = [left - right for left, right in zip(selected_gain, control_gain)]
    selected_audits = [run.get("language_audit", {}) for run in selected]
    source_audits = [run.get("source_language_audit", {}) for run in controls]

    summary = {
        "protocol": "typed-residual-language-growth-control-v1",
        "selected_family": "positive_monomial_lattice",
        "counterfactual_family": "positive_additive_lattice",
        "n_seeds": len(selected),
        "n_heldout_tasks_per_seed": len(selected[0]["paired"]),
        "n_paired_task_evaluations": len(task_rows),
        "source_modules": selected[0]["source_modules"],
        "heldout_modules": selected[0]["heldout_modules"],
        "selected_gain_mean": _mean(selected_gain),
        "selected_gain_seed_sd": _sample_sd(selected_gain),
        "control_gain_mean": _mean(control_gain),
        "control_gain_seed_sd": _sample_sd(control_gain),
        "selection_advantage_mean": _mean(differences),
        "selection_advantage_seed_sd": _sample_sd(differences),
        "selected_task_gain_mean": _mean(selected_task_gain),
        "control_task_gain_mean": _mean(control_task_gain),
        "selected_candidate_wins": sum(int(run["candidate_wins"]) for run in selected),
        "control_candidate_wins": sum(int(run["candidate_wins"]) for run in controls),
        "selected_operator_identifier_free": all(
            bool(audit.get("source_task_identifier_free")) for audit in selected_audits
        ),
        "source_operator_identifier_free": all(
            bool(audit.get("source_task_identifier_free")) for audit in source_audits
        ),
        "selected_runs": [run["_artifact"] for run in selected],
        "control_runs": [run["_artifact"] for run in controls],
        "task_rows": task_rows,
    }
    out_dir = _PROJECT / "lmw" / "language_transfer" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists: {out_path}")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"task_rows", "selected_runs", "control_runs"}}, indent=2))


if __name__ == "__main__":
    main()
