"""Build a paper-safe benchmark table for the universal MARS method.

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
        tier="main_universal",
        metric="HMS / HMS-consistency",
        path=ROOT
        / "lmw/universal_discovery_real/research_cycle_discovery_full239_scopegate_intrabundle_20260712/summary.json",
        note="Full 239-task real DiscoveryBench run with the shared slot/contract compiler; no task-specific gold labels in generation.",
    ),
    Artifact(
        benchmark="NewtonBench",
        tier="main_universal",
        metric="SA-all / SA-answered",
        path=ROOT
        / "lmw/nb_activeprobe/nb_full324_asymptotic_lift_v5_20260713/summary.json",
        note="Full 324-task NewtonBench run with the shared universal equation-induction stack.",
    ),
    Artifact(
        benchmark="UltraHorizon",
        tier="main_universal_clean",
        metric="paper-style judge score",
        path=ROOT
        / "lmw/uh_official/uh_clean_universal_full96_20260715/summary.json",
        note="Full 96-task hard UltraHorizon run with paper-style judge, no env hints, and measurement bootstraps disabled.",
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
        return {
            "primary": data.get("SA_all"),
            "secondary": data.get("SA_answered"),
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
        "# Paper-Safe MARS Results",
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
            "For the main paper, describe MARS as a universal hypothesis-induction "
            "system and report the three full-run rows above. Do not present the "
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
