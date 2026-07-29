"""Deterministic nonparametric confidence intervals for paper headline runs."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / (
    "lmw/universal_discovery_real/"
    "aaai27_language_growth_protocol_v1_fixed_l0_test239/"
    "official_eval.jsonl"
)
DB_CONTROLS = {
    "direct": ROOT
    / (
        "lmw/discovery_external/"
        "aaai27_discovery_external_direct_full239_20260717/"
        "official_eval.jsonl"
    ),
    "react": ROOT
    / (
        "lmw/discovery_external/"
        "aaai27_discovery_external_react_full239_repaired_20260718/"
        "official_eval.jsonl"
    ),
    "codeact": ROOT
    / (
        "lmw/discovery_external/"
        "aaai27_discovery_external_codeact_full239_v2_combined_20260718/"
        "official_eval.jsonl"
    ),
}
NB = ROOT / (
    "lmw/nb_activeprobe/"
    "aaai27_newton_full324_no_promotion_gates_20260717/"
    "summary.json"
)
UH = ROOT / (
    "lmw/uh_official/"
    "aaai27_uh_full96_marsfull_strict_agent_paperjudge_20260717/"
    "run.jsonl"
)
OUTPUT = ROOT / "lmw/paper_safe_results/bootstrap_headline_ci_20260729.json"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_mean(
    values: list[float],
    rng: random.Random,
    *,
    draws: int,
) -> dict[str, float | int]:
    n = len(values)
    estimates = [
        mean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(draws)
    ]
    return {
        "n": n,
        "mean": mean(values),
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
        "bootstrap_draws": draws,
    }


def paired_bootstrap_gain(
    treatment: dict[str, float],
    control: dict[str, float],
    rng: random.Random,
    *,
    draws: int,
) -> dict[str, float | int]:
    keys = sorted(set(treatment) & set(control))
    if len(keys) != len(treatment) or len(keys) != len(control):
        raise ValueError(
            "Paired DiscoveryBench comparison requires identical task keys: "
            f"treatment={len(treatment)}, control={len(control)}, shared={len(keys)}"
        )
    deltas = [treatment[key] - control[key] for key in keys]
    result = bootstrap_mean(deltas, rng, draws=draws)
    result["treatment_mean"] = mean(treatment[key] for key in keys)
    result["control_mean"] = mean(control[key] for key in keys)
    return result


def discovery_hms(row: dict) -> float:
    if "HMS_raw_100" in row:
        return float(row["HMS_raw_100"])
    return float(row["HMS_100"])


def stratified_bootstrap_mean(
    groups: dict[str, list[float]],
    rng: random.Random,
    *,
    draws: int,
) -> dict[str, float | int]:
    names = sorted(groups)
    estimates = []
    for _ in range(draws):
        group_means = []
        for name in names:
            values = groups[name]
            group_means.append(
                mean(values[rng.randrange(len(values))] for _ in values)
            )
        estimates.append(mean(group_means))
    return {
        "n": sum(len(groups[name]) for name in names),
        "strata": len(names),
        "mean": mean(mean(groups[name]) for name in names),
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
        "bootstrap_draws": draws,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, default=DB)
    parser.add_argument("--newton", type=Path, default=NB)
    parser.add_argument("--ultrahorizon", type=Path, default=UH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    draws = args.draws
    rng = random.Random(args.seed)

    db_rows = read_jsonl(args.discovery)
    nb_summary = json.loads(args.newton.read_text(encoding="utf-8"))
    nb_rows = list(nb_summary["rows"])
    uh_rows = read_jsonl(args.ultrahorizon)

    db_raw = [discovery_hms(row) for row in db_rows]
    db_consistency = [float(row["HMS_consistency_100"]) for row in db_rows]
    nb_all = [float(row.get("SA", 0.0)) for row in nb_rows]
    nb_answered = [
        float(row.get("SA", 0.0))
        for row in nb_rows
        if row.get("status") == "ANSWER"
    ]
    uh_groups: dict[str, list[float]] = defaultdict(list)
    for row in uh_rows:
        uh_groups[str(row["env"])].append(float(row["final_score"]))

    db_by_key = {str(row["task_key"]): discovery_hms(row) for row in db_rows}
    paired_controls = {}
    for index, (name, path) in enumerate(DB_CONTROLS.items(), start=1):
        rows = read_jsonl(path)
        control_by_key = {
            str(row["task_key"]): discovery_hms(row)
            for row in rows
        }
        paired_controls[name] = paired_bootstrap_gain(
            db_by_key,
            control_by_key,
            random.Random(args.seed + index),
            draws=draws,
        )

    result = {
        "protocol": {
            "method": "nonparametric percentile bootstrap over evaluation units",
            "seed": args.seed,
            "draws": draws,
            "ultrahorizon": "stratified by environment, 32 seeds per stratum",
            "sources": {
                "discovery": str(args.discovery),
                "newton": str(args.newton),
                "ultrahorizon": str(args.ultrahorizon),
            },
        },
        "discoverybench_hms": bootstrap_mean(db_raw, rng, draws=draws),
        "discoverybench_consistency_hms": bootstrap_mean(
            db_consistency, rng, draws=draws
        ),
        "discoverybench_paired_controls": paired_controls,
        "newtonbench_sa_all": bootstrap_mean(nb_all, rng, draws=draws),
        "newtonbench_sa_answered": bootstrap_mean(
            nb_answered, rng, draws=draws
        ),
        "newtonbench_coverage": {
            "answered": len(nb_answered),
            "total": len(nb_all),
            "rate": len(nb_answered) / len(nb_all),
        },
        "ultrahorizon_strict_score": stratified_bootstrap_mean(
            dict(uh_groups), rng, draws=draws
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
