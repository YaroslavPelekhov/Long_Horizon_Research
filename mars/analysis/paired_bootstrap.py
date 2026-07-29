"""Paired bootstrap confidence intervals for benchmark JSONL evaluations.

The analysis is intentionally read-only: it aligns task keys present in both
files and resamples those pairs. It never invokes a judge or rewrites a run.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path
from typing import Any


def _load_scores(path: Path, score_key: str) -> dict[str, float]:
    rows: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("task_key", ""))
        if not key:
            continue
        try:
            rows[key] = float(row.get(score_key, 0.0) or 0.0)
        except (TypeError, ValueError):
            rows[key] = 0.0
    return rows


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


def paired_bootstrap(
    baseline: dict[str, float],
    treatment: dict[str, float],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    keys = sorted(set(baseline) & set(treatment))
    if not keys:
        raise ValueError("the two files have no shared task_key values")
    deltas = [treatment[key] - baseline[key] for key in keys]
    rng = random.Random(seed)
    draws: list[float] = []
    n = len(deltas)
    for _ in range(samples):
        draws.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    mean_delta = st.fmean(deltas)
    return {
        "n_pairs": n,
        "baseline_mean": st.fmean(baseline[key] for key in keys),
        "treatment_mean": st.fmean(treatment[key] for key in keys),
        "mean_delta": mean_delta,
        "ci95": [_quantile(draws, 0.025), _quantile(draws, 0.975)],
        "improved_tasks": sum(delta > 0 for delta in deltas),
        "regressed_tasks": sum(delta < 0 for delta in deltas),
        "tied_tasks": sum(delta == 0 for delta in deltas),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--treatment", required=True, type=Path)
    parser.add_argument("--score_key", default="HMS_100")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = paired_bootstrap(
        _load_scores(args.baseline, args.score_key),
        _load_scores(args.treatment, args.score_key),
        seed=args.seed,
        samples=args.samples,
    )
    result.update(
        {
            "baseline": str(args.baseline),
            "treatment": str(args.treatment),
            "score_key": args.score_key,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
