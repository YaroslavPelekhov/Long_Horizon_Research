"""Collect fresh cross-benchmark MARS ablation results for the paper.

The script is intentionally read-only over experiment outputs: it does not
rescore predictions or fill missing rows. Missing runs are reported explicitly so
the paper table cannot silently mix old and new artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "lmw" / "submission_ablations_20260715"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _discovery_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    module_path = (
        ROOT
        / "lmw"
        / "universal_discovery_real"
        / "ablation_discovery_modules30_20260715"
        / "module_ablation.json"
    )
    module = _read_json(module_path)
    if module:
        for result in module.get("results", []):
            rows.append(
                {
                    "benchmark": "DiscoveryBench",
                    "condition": result.get("tag"),
                    "n": len(module.get("task_keys", [])),
                    "metric": "HMS",
                    "score": round(float(result.get("mean_hms", 0.0)), 2),
                    "artifact": _artifact(Path(result.get("summary_path", ""))),
                }
            )
    full_path = (
        ROOT
        / "lmw"
        / "universal_discovery_real"
        / "ablation_discovery_full30_mars_full_20260715"
        / "summary.json"
    )
    full = _read_json(full_path)
    if full:
        rows.append(
            {
                "benchmark": "DiscoveryBench",
                "condition": "MARS-full",
                "n": int(full.get("n_tasks", 0) or 0),
                "metric": "HMS / Cons-HMS",
                "score": round(float(full.get("HMS_mean_100", 0.0)), 2),
                "secondary": round(float(full.get("HMS_mean_consistency_100", 0.0)), 2),
                "artifact": _artifact(full_path),
            }
        )
    return rows


def _newton_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        (
            "MARS-full",
            ROOT / "lmw" / "nb_activeprobe" / "nb_full324_asymptotic_lift_v5_20260713" / "summary.json",
        ),
        (
            "no residual-born operator charts",
            ROOT
            / "lmw"
            / "nb_activeprobe"
            / "ablation_newton_full324_no_operator_charts_20260715"
            / "summary.json",
        ),
    ]
    for condition, path in specs:
        data = _read_json(path)
        if not data:
            continue
        n = int(data.get("n", 0) or 0)
        if n <= 0:
            continue
        rows.append(
            {
                "benchmark": "NewtonBench",
                "condition": condition,
                "n": n,
                "metric": "SA-all / SA-answered",
                "score": round(100.0 * float(data.get("SA_all", 0.0)), 1),
                "secondary": round(100.0 * float(data.get("SA_answered", 0.0)), 1),
                "artifact": _artifact(path),
            }
        )
    return rows


def _uh_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs: list[tuple[str, list[Path]]] = [
        (
            "MARS-full (grid+seq)",
            [ROOT / "lmw" / "uh_official" / "ablation_uh_seed0_3_full_20260715" / "run.jsonl"],
        ),
        (
            "MARS-no-ref (grid+seq)",
            [
                ROOT / "lmw" / "uh_official" / "ablation_uh_seed0_3_no_ref_grid_20260715" / "run.jsonl",
                ROOT / "lmw" / "uh_official" / "ablation_uh_seed0_3_no_ref_seq_20260715" / "run.jsonl",
            ],
        ),
        (
            "MARS-all-off (grid+seq)",
            [
                ROOT / "lmw" / "uh_official" / "ablation_uh_seed0_3_all_off_grid_20260715" / "run.jsonl",
                ROOT / "lmw" / "uh_official" / "ablation_uh_seed0_3_all_off_seq_20260715" / "run.jsonl",
            ],
        ),
    ]
    for condition, paths in specs:
        scores: list[float] = []
        artifacts: list[str] = []
        for path in paths:
            if not path.exists():
                continue
            artifacts.append(_artifact(path))
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("env") not in {"grid", "seq"}:
                    continue
                scores.append(float(row.get("final_score", 0.0) or 0.0))
        if not scores:
            continue
        rows.append(
            {
                "benchmark": "UltraHorizon",
                "condition": condition,
                "n": len(scores),
                "metric": "paper-style score",
                "score": round(sum(scores) / len(scores), 2),
                "artifact": "; ".join(artifacts),
            }
        )
    return rows


def _json_n(path: Path) -> int:
    data = _read_json(path)
    if not data:
        return 0
    return int(data.get("n", 0) or 0)


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Submission Ablation Results",
        "",
        "Fresh cross-benchmark ablation artifacts for the MARS paper.",
        "",
        "| Benchmark | Condition | N | Metric | Score | Secondary | Artifact |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for row in rows:
        secondary = row.get("secondary")
        secondary_s = "" if secondary is None else str(secondary)
        lines.append(
            f"| {row['benchmark']} | {row['condition']} | {row['n']} | {row['metric']} | "
            f"{row['score']} | {secondary_s} | `{row['artifact']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _discovery_rows() + _newton_rows() + _uh_rows()
    newton_no_charts = (
        ROOT
        / "lmw"
        / "nb_activeprobe"
        / "ablation_newton_full324_no_operator_charts_20260715"
        / "summary.json"
    )
    uh_full = ROOT / "lmw" / "uh_official" / "ablation_uh_seed0_3_full_20260715" / "run.jsonl"
    uh_no_ref_grid = ROOT / "lmw" / "uh_official" / "ablation_uh_seed0_3_no_ref_grid_20260715" / "run.jsonl"
    uh_no_ref_seq = ROOT / "lmw" / "uh_official" / "ablation_uh_seed0_3_no_ref_seq_20260715" / "run.jsonl"
    uh_all_off_grid = ROOT / "lmw" / "uh_official" / "ablation_uh_seed0_3_all_off_grid_20260715" / "run.jsonl"
    uh_all_off_seq = ROOT / "lmw" / "uh_official" / "ablation_uh_seed0_3_all_off_seq_20260715" / "run.jsonl"
    missing = {
        "newton_no_charts_full324": _json_n(newton_no_charts) != 324,
        "uh_full_gridseq_seed0_3": sum(1 for _ in uh_full.read_text(encoding="utf-8").splitlines()) < 8
        if uh_full.exists()
        else True,
        "uh_no_ref_gridseq_seed0_3": not (uh_no_ref_grid.exists() and uh_no_ref_seq.exists()),
        "uh_all_off_gridseq_seed0_3": not (uh_all_off_grid.exists() and uh_all_off_seq.exists()),
    }
    payload = {"rows": rows, "missing": missing}
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "summary.md").write_text(_markdown(rows), encoding="utf-8")
    print(_markdown(rows))
    if any(missing.values()):
        print("Missing:", {k: v for k, v in missing.items() if v})


if __name__ == "__main__":
    main()
