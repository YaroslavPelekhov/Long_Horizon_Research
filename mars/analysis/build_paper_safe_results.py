"""Build a paper-safe benchmark table for RG-HLI.

The goal of this script is deliberately conservative.  It separates:

* main universal results: one shared MARS spine, no late benchmark-engineered
  ceiling runs;
* enhanced-but-still-reported results: shared induction/repair layers that were
  evaluated on full official-compatible splits;
* diagnostic ceiling runs: useful for debugging, excluded from paper claims.

The script reads existing run artifacts and writes both JSON and Markdown so the
paper/README cannot accidentally mix the categories.
"""

from __future__ import annotations

import json
import statistics as st
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "lmw" / "paper_safe_results"
OUT_JSON = OUT_DIR / "summary.json"
OUT_MD = ROOT / "PAPER_SAFE_RESULTS.md"


@dataclass(frozen=True)
class Artifact:
    benchmark: str
    tier: str
    metric: str
    path: Path
    note: str


ARTIFACTS = [
    Artifact(
        benchmark="DiscoveryBench",
        tier="headline",
        metric="HMS / consistency-HMS",
        path=ROOT
        / "lmw/universal_discovery_real/aaai27_language_growth_protocol_v1_fixed_l0_test239/summary.json",
        note=(
            "Complete 239-task native-HMS evaluation with four per-task "
            "proposals and a frozen hypothesis language."
        ),
    ),
    Artifact(
        benchmark="NewtonBench",
        tier="headline",
        metric="SA-all / SA-answered",
        path=ROOT
        / "lmw/nb_activeprobe/aaai27_newton_full324_no_promotion_gates_20260717/summary.json",
        note="Complete 324-configuration frozen-kernel evaluation; 240 submitted laws and 84 abstentions.",
    ),
    Artifact(
        benchmark="UltraHorizon",
        tier="headline strict",
        metric="paper-style score",
        path=ROOT
        / "lmw/uh_official/aaai27_uh_full96_marsfull_strict_agent_paperjudge_20260717/summary.json",
        note="Complete 96-episode hard run: 50 steps, no hints, fallback commits, or measurement bootstraps.",
    ),
    Artifact(
        benchmark="UltraHorizon",
        tier="diagnostic_ceiling_excluded",
        metric="paper-style judge score",
        path=ROOT
        / "lmw/uh_official/uh_full_official_nohint_grid_32seeds_priorfix_20260714/summary.json",
        note="Part of a later ceiling/debug run. Kept out of the main table.",
    ),
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _score_for_artifact(artifact: Artifact, data: dict[str, Any]) -> dict[str, Any]:
    if "missing" in data:
        return {"primary": None, "secondary": None, "n": None, "status": "missing"}

    if artifact.benchmark == "DiscoveryBench":
        return {
            "primary": data.get("HMS_mean_100"),
            "secondary": data.get("HMS_mean_consistency_100"),
            "n": data.get("n_tasks"),
            "status": "ok",
        }

    if artifact.benchmark == "NewtonBench":
        rows_path = Path(str(data.get("rows_csv", "")))
        if not rows_path.is_file():
            raise FileNotFoundError(f"NewtonBench rows are required for exact SA: {rows_path}")
        with rows_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 324:
            raise ValueError(f"NewtonBench rows must contain 324 configurations: {rows_path}")
        answered_rows = [row for row in rows if row.get("status") == "ANSWER"]
        return {
            "primary": 100.0 * sum(float(row["SA"]) for row in rows) / len(rows),
            "secondary": 100.0 * sum(float(row["SA"]) for row in answered_rows) / len(answered_rows),
            "n": data.get("n"),
            "status": "ok",
            "answered": data.get("answered"),
            "abstained": data.get("abstained"),
        }

    if artifact.benchmark == "UltraHorizon":
        return {
            "primary": data.get("mean_score"),
            "secondary": None,
            "n": data.get("n"),
            "status": "ok",
            "committed": data.get("committed", data.get("n_committed")),
            "exact_paper_judge": data.get("exact_paper_judge"),
        }

    if artifact.benchmark == "UltraHorizon Bio":
        return {
            "primary": data.get("mean_score"),
            "secondary": None,
            "n": data.get("n"),
            "status": "ok",
        }

    return {"primary": None, "secondary": None, "n": None, "status": "unknown"}


def _uh_ceiling_from_parts() -> dict[str, Any]:
    """Summarize the late UH ceiling run without promoting it to main results."""

    paths = [
        ROOT / "lmw/uh_official/uh_full_official_nohint_grid_32seeds_priorfix_20260714/run.jsonl",
        ROOT / "lmw/uh_official/uh_full_official_nohint_seq_32seeds_renderfix_20260714/run.jsonl",
        ROOT / "lmw/uh_official/uh_full_official_nohint_bio_32seeds_latfit3_20260714/run.jsonl",
        ROOT / "lmw/uh_official/uh_full_official_nohint_bio_latfit3_s04_11_20260714/run.jsonl",
        ROOT / "lmw/uh_official/uh_full_official_nohint_bio_latfit3_s12_19_20260714/run.jsonl",
        ROOT / "lmw/uh_official/uh_full_official_nohint_bio_latfit3_s20_27_20260714/run.jsonl",
        ROOT / "lmw/uh_official/uh_full_official_nohint_bio_latfit3_s28_31_20260714/run.jsonl",
    ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        env = row.get("env")
        seed = row.get("seed")
        if env is None or seed is None:
            continue
        by_key.setdefault((str(env), int(seed)), row)

    scores = [float(row.get("final_score") or 0.0) for row in by_key.values()]
    return {
        "tier": "diagnostic_ceiling_excluded",
        "benchmark": "UltraHorizon",
        "n": len(scores),
        "mean_score": round(st.fmean(scores), 3) if scores else None,
        "score_distribution": {
            str(score): scores.count(score) for score in sorted(set(scores))
        },
        "excluded_reason": (
            "Late run uses UH-specific measurement bootstraps/rendering fixes. "
            "It is useful as an engineering ceiling, not as the paper's universal-method result."
        ),
    }


def build() -> dict[str, Any]:
    rows = []
    for artifact in ARTIFACTS:
        data = _read_json(artifact.path)
        row = {
            "benchmark": artifact.benchmark,
            "tier": artifact.tier,
            "metric": artifact.metric,
            "path": str(artifact.path.relative_to(ROOT)),
            "note": artifact.note,
            **_score_for_artifact(artifact, data),
        }
        rows.append(row)

    result = {
        "method_scope": {
            "paper_main": (
                "Use the three main rows as the current defensible paper table: "
                "one full run each for DiscoveryBench, NewtonBench, and UltraHorizon."
            ),
            "excluded": (
                "diagnostic_ceiling_excluded rows must not be used as main SOTA claims."
            ),
        },
        "rows": [row for row in rows if row["tier"] != "diagnostic_ceiling_excluded"],
        "excluded_rows": [row for row in rows if row["tier"] == "diagnostic_ceiling_excluded"],
        "diagnostic_ceiling": _uh_ceiling_from_parts(),
    }
    return result


def write_markdown(result: dict[str, Any]) -> None:
    lines = [
        "# Paper-Safe RG-HLI Results",
        "",
        "This file separates defensible universal-method results from late diagnostic ceiling runs.",
        "",
        "## Main Table",
        "",
        "| Benchmark | Tier | Metric | Primary | Secondary | N | Notes |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in result["rows"]:
        primary = "" if row.get("primary") is None else f"{float(row['primary']):.3f}"
        secondary = "" if row.get("secondary") is None else f"{float(row['secondary']):.3f}"
        n = "" if row.get("n") is None else str(row["n"])
        lines.append(
            f"| {row['benchmark']} | {row['tier']} | {row['metric']} | "
            f"{primary} | {secondary} | {n} | {row['note']} |"
        )

    ceiling = result["diagnostic_ceiling"]
    lines += [
        "",
        "## Excluded Diagnostic Ceiling",
        "",
        (
            f"UltraHorizon late ceiling aggregate: N={ceiling['n']}, "
            f"mean={ceiling['mean_score']}, distribution={ceiling['score_distribution']}."
        ),
        "",
        f"Excluded reason: {ceiling['excluded_reason']}",
        "",
        "## Paper Wording",
        "",
        (
            "For the main paper, describe RG-HLI as a residual-guided hypothesis-language "
            "induction system and report the three headline rows above. Do not present the "
            "diagnostic ceiling as SOTA."
        ),
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = build()
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result)
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
