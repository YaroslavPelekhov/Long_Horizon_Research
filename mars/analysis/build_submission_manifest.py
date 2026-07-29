"""Build a strict, source-linked manifest from completed benchmark runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return value


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path) -> str:
    """Keep manifests portable when the workspace moves to another machine."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _discovery(path: Path) -> dict[str, Any]:
    value = _read(path)
    if int(value.get("n_tasks", -1)) != 239:
        raise ValueError(f"DiscoveryBench must contain 239 tasks: {path}")
    if value.get("data_split", "test") != "test":
        raise ValueError(f"DiscoveryBench headline artifact is not test split: {path}")
    if value.get("HMS_mean_100") is None:
        raise ValueError(f"DiscoveryBench official evaluation is missing: {path}")
    return {
        "benchmark": "DiscoveryBench",
        "metric": "HMS / consistency-HMS",
        "primary": float(value["HMS_mean_100"]),
        "secondary": float(value["HMS_mean_consistency_100"]),
        "n": 239,
        "model": value.get("model"),
        "judge_model": value.get("judge_model"),
        "protocol": "official real/test evaluator",
        "source": _portable(path),
        "sha256": _hash(path),
        "score_source": _portable(path),
        "score_sha256": _hash(path),
    }


def _newton(path: Path) -> dict[str, Any]:
    value = _read(path)
    if int(value.get("n", -1)) != 324:
        raise ValueError(f"NewtonBench must contain 324 configurations: {path}")
    answered = int(value.get("answered", -1))
    abstained = int(value.get("abstained", -1))
    if answered < 0 or abstained < 0 or answered + abstained != 324:
        raise ValueError(f"NewtonBench answer accounting must sum to 324: {path}")
    if "SA_all" not in value or "SA_answered" not in value:
        raise ValueError(f"NewtonBench symbolic-agreement scores are missing: {path}")
    rows_path = Path(str(value.get("rows_csv", "")))
    if not rows_path.is_file():
        raise ValueError(f"NewtonBench per-row evaluator log is missing: {path}")
    with rows_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 324:
        raise ValueError(f"NewtonBench evaluator log must contain 324 rows: {rows_path}")
    exact_all = 100.0 * sum(float(row["SA"]) for row in rows) / len(rows)
    answered_rows = [row for row in rows if row.get("status") == "ANSWER"]
    if len(answered_rows) != answered:
        raise ValueError(f"NewtonBench evaluator-log answer count disagrees with summary: {path}")
    exact_answered = 100.0 * sum(float(row["SA"]) for row in answered_rows) / len(answered_rows)
    return {
        "benchmark": "NewtonBench",
        "metric": "SA-all / SA-answered",
        "primary": exact_all,
        "secondary": exact_answered,
        "n": 324,
        "model": value.get("model"),
        "judge_model": value.get("judge_model"),
        "protocol": "complete frozen-kernel full grid; promotion gates disabled",
        "source": _portable(path),
        "sha256": _hash(path),
        "score_source": _portable(rows_path),
        "score_sha256": _hash(rows_path),
    }


def _ultrahorizon(path: Path) -> dict[str, Any]:
    value = _read(path)
    n = int(value.get("n", value.get("n_tasks", -1)) or -1)
    if n != 96:
        raise ValueError(f"UltraHorizon must contain 96 official episodes: {path}")
    score = value.get("mean_score")
    if score is None:
        raise ValueError(f"UltraHorizon mean score is missing: {path}")
    if value.get("paper_style_judge") is not True or value.get("exact_paper_judge") is not True:
        raise ValueError(f"UltraHorizon headline artifact must use the paper-style judge: {path}")
    if int(value.get("steps", -1)) != 50:
        raise ValueError(f"UltraHorizon headline artifact must use the fixed 50-step horizon: {path}")
    if value.get("n_measurement_bootstrap_applied") != 0:
        raise ValueError(f"UltraHorizon headline artifact must not use measurement bootstraps: {path}")
    if set(value.get("use_env_hints_values", {})) != {"False"}:
        raise ValueError(f"UltraHorizon headline artifact must disable environment hints: {path}")
    if set(value.get("enable_fallback_commit_values", {})) != {"False"}:
        raise ValueError(f"UltraHorizon headline artifact must disable fallback commits: {path}")
    return {
        "benchmark": "UltraHorizon",
        "metric": "paper-style score",
        "primary": float(score),
        "secondary": None,
        "n": 96,
        "model": value.get("generator_model", value.get("model")),
        "judge_model": value.get("judge_model"),
        "protocol": "official 3 environments x 32 seeds",
        "source": _portable(path),
        "sha256": _hash(path),
        "score_source": _portable(path),
        "score_sha256": _hash(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path)
    parser.add_argument("--newton", type=Path)
    parser.add_argument("--ultrahorizon", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    if args.discovery:
        rows.append(_discovery(args.discovery.resolve()))
    if args.newton:
        rows.append(_newton(args.newton.resolve()))
    if args.ultrahorizon:
        rows.append(_ultrahorizon(args.ultrahorizon.resolve()))
    if not rows:
        raise SystemExit("provide at least one benchmark summary")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_version": "submission-results-v1",
        "rows": rows,
        "all_sources_hashed": True,
        "manual_score_entry": False,
    }
    (args.output_dir / "results_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "results_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output_dir / 'results_manifest.json'}")


if __name__ == "__main__":
    main()
