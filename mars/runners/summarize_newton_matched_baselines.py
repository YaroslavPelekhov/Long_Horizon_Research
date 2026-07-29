"""Audit and summarize full NewtonBench matched-backbone baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MODULES = (
    "m0_gravity",
    "m1_coulomb_force",
    "m2_magnetic_force",
    "m3_fourier_law",
    "m4_snell_law",
    "m5_radioactive_decay",
    "m6_underdamped_harmonic",
    "m7_malus_law",
    "m8_sound_speed",
    "m9_hooke_law",
    "m10_be_distribution",
    "m11_heat_transfer",
)
DIFFICULTIES = ("easy", "medium", "hard")
SYSTEMS = ("vanilla_equation", "simple_system", "complex_system")
VERSIONS = ("v0", "v1", "v2")
BACKENDS = ("vanilla_agent", "code_assisted_agent")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _version(path: Path) -> int:
    match = re.search(r"_v(\d+)$", path.parent.parent.name)
    return int(match.group(1)) if match else 0


def _config_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["module_name"]),
        str(row["equation_difficulty"]),
        str(row["model_system"]),
        str(row["law_version"]),
    )


def _expected() -> set[tuple[str, str, str, str]]:
    return {
        (module, difficulty, system, version)
        for module in MODULES
        for difficulty in DIFFICULTIES
        for system in SYSTEMS
        for version in VERSIONS
    }


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return (100.0 * (center - radius), 100.0 * (center + radius))


def _usage(paths: Iterable[Path]) -> dict[str, Any]:
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
    )
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            model = str(row.get("provider_model") or row.get("model_name"))
            target = totals[model]
            target["calls"] += 1
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                target[field] += int(row.get(field) or 0)
            target["cost_usd"] += float(row.get("cost") or 0.0)
    by_model = {
        model: {
            **values,
            "cost_usd": round(values["cost_usd"], 6),
        }
        for model, values in sorted(totals.items())
    }
    return {
        "by_model": by_model,
        "calls": int(sum(row["calls"] for row in totals.values())),
        "prompt_tokens": int(
            sum(row["prompt_tokens"] for row in totals.values())
        ),
        "completion_tokens": int(
            sum(row["completion_tokens"] for row in totals.values())
        ),
        "total_tokens": int(
            sum(row["total_tokens"] for row in totals.values())
        ),
        "cost_usd": round(
            sum(row["cost_usd"] for row in totals.values()), 6
        ),
    }


def _aggregate(
    rows: list[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    output = []
    for name, subset in sorted(grouped.items()):
        correct = sum(int(row["exact_accuracy"]) for row in subset)
        low, high = _wilson(correct, len(subset))
        output.append(
            {
                "dimension": field,
                "value": name,
                "n": len(subset),
                "correct": correct,
                "SA_all_100": 100.0 * correct / len(subset),
                "wilson95_low": low,
                "wilson95_high": high,
            }
        )
    return output


def _summarize_backend(root: Path, backend: str) -> dict[str, Any]:
    candidates: dict[
        tuple[str, str, str, str], list[tuple[int, Path, dict[str, Any]]]
    ] = defaultdict(list)
    for path in root.glob(
        f"gpt4omini/*/{backend}/*/*/*/trials/trial0.json"
    ):
        row = _read_json(path)
        if not row:
            continue
        if row.get("evaluation", {}).get("exact_accuracy") not in (0, 1):
            continue
        if not str(row.get("submitted_law") or "").strip():
            continue
        candidates[_config_key(row)].append((_version(path), path, row))

    selected: dict[
        tuple[str, str, str, str], tuple[int, Path, dict[str, Any]]
    ] = {
        key: min(values, key=lambda item: (item[0], item[1].stat().st_mtime))
        for key, values in candidates.items()
    }
    expected = _expected()
    missing = sorted(expected - set(selected))
    unexpected = sorted(set(selected) - expected)
    duplicate_valid = {
        "|".join(key): [str(item[1]) for item in values]
        for key, values in candidates.items()
        if len(values) > 1
    }

    task_rows = []
    for key in sorted(selected):
        version, path, raw = selected[key]
        evaluation = raw["evaluation"]
        submitted = bool(str(raw.get("submitted_law") or "").strip())
        task_rows.append(
            {
                "backend": backend,
                "module_name": key[0],
                "equation_difficulty": key[1],
                "model_system": key[2],
                "law_version": key[3],
                "experiment_version": version,
                "answered": int(submitted),
                "exact_accuracy": int(evaluation["exact_accuracy"]),
                "symbolic_equivalent": int(
                    bool(evaluation.get("symbolic_equivalent"))
                ),
                "rounds": int(raw.get("rounds") or 0),
                "num_experiments": int(raw.get("num_experiments") or 0),
                "agent_model": str(raw.get("model_name")),
                "judge_model": str(raw.get("LLM judge")),
                "terminal_status": str(raw.get("status") or ""),
                "trial_path": str(path),
            }
        )

    answered = [row for row in task_rows if row["answered"]]
    correct = sum(row["exact_accuracy"] for row in task_rows)
    answered_correct = sum(row["exact_accuracy"] for row in answered)
    low, high = _wilson(correct, len(task_rows))
    selected_usage_paths = [
        path.parent / "api_usage.jsonl" for _, path, _ in selected.values()
    ]
    clean_usage = _usage(selected_usage_paths)
    incurred_usage = _usage(
        root.glob(f"gpt4omini/*/{backend}/*/*/*/trials/api_usage.jsonl")
    )
    slices = []
    for field in ("module_name", "equation_difficulty", "model_system"):
        slices.extend(_aggregate(task_rows, field))

    complete = not missing and not unexpected and len(task_rows) == 324
    return {
        "backend": backend,
        "complete": complete,
        "n_expected": 324,
        "n_valid_unique": len(task_rows),
        "n_missing": len(missing),
        "missing": [list(key) for key in missing],
        "unexpected": [list(key) for key in unexpected],
        "n_duplicate_valid": len(duplicate_valid),
        "duplicate_valid": duplicate_valid,
        "n_answered": len(answered),
        "coverage_100": (
            100.0 * len(answered) / len(task_rows) if task_rows else None
        ),
        "SA_all_100": (
            100.0 * correct / len(task_rows) if task_rows else None
        ),
        "SA_all_wilson95": [low, high],
        "SA_answered_100": (
            100.0 * answered_correct / len(answered) if answered else None
        ),
        "mean_rounds": (
            sum(row["rounds"] for row in task_rows) / len(task_rows)
            if task_rows
            else None
        ),
        "mean_experiments": (
            sum(row["num_experiments"] for row in task_rows)
            / len(task_rows)
            if task_rows
            else None
        ),
        "judge_models": sorted(
            {row["judge_model"] for row in task_rows}
        ),
        "terminal_status_counts": {
            status: sum(
                row["terminal_status"] == status for row in task_rows
            )
            for status in sorted(
                {row["terminal_status"] for row in task_rows}
            )
        },
        "selected_run_usage": clean_usage,
        "incurred_usage_including_failed_attempts": incurred_usage,
        "slices": slices,
        "tasks": task_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    summaries = {
        backend: _summarize_backend(root, backend)
        for backend in BACKENDS
    }
    output = {
        "protocol": "newtonbench-full-matched-backbone-v1",
        "duplicate_resolution": (
            "earliest valid version; failed files are never eligible"
        ),
        "root": str(root),
        "model": "openai/gpt-4o-mini",
        "judge": "openai/gpt-4.1",
        "n_configurations_per_backend": 324,
        "backends": {
            backend: {
                key: value
                for key, value in summary.items()
                if key not in {"tasks", "slices"}
            }
            for backend, summary in summaries.items()
        },
    }
    (root / "audited_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    for backend, summary in summaries.items():
        _write_csv(root / f"{backend}_tasks.csv", summary["tasks"])
        _write_csv(root / f"{backend}_slices.csv", summary["slices"])

    headline = [
        {
            "backend": backend,
            "complete": summary["complete"],
            "n_valid_unique": summary["n_valid_unique"],
            "n_missing": summary["n_missing"],
            "coverage_100": summary["coverage_100"],
            "SA_all_100": summary["SA_all_100"],
            "SA_answered_100": summary["SA_answered_100"],
            "wilson95_low": summary["SA_all_wilson95"][0],
            "wilson95_high": summary["SA_all_wilson95"][1],
            "selected_cost_usd": summary["selected_run_usage"]["cost_usd"],
            "selected_calls": summary["selected_run_usage"]["calls"],
        }
        for backend, summary in summaries.items()
    ]
    _write_csv(root / "audited_headline.csv", headline)
    print(json.dumps(headline, indent=2))
    if not all(summary["complete"] for summary in summaries.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
