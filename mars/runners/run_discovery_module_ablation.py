"""Run DiscoveryBench module-stack ablations.

This is a presentation-friendly protocol: the same official tasks are evaluated
under increasingly capable universal modules, so each row gives an observable
increment from a module rather than a hand-written story.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent


PRESETS: list[tuple[str, str, str]] = [
    (
        "probes_llm",
        "llm_synthesis",
        "Executable table probes + LLM verbalization; no typed hypothesis contracts.",
    ),
    (
        "workbench",
        "workbench,llm_synthesis",
        "Adds metric-oriented candidate repair/ranking over probe evidence.",
    ),
    (
        "slot_contract",
        "slot_contract,temporal,workbench,llm_synthesis",
        "Adds typed temporal/PCA/comparison slots closed by deterministic measurements.",
    ),
    (
        "answer_slot_full",
        "answer_slot,slot_contract,temporal,workbench,llm_synthesis",
        "Adds question-first answer-form slots such as grouped comparisons, coefficients, gaps, and category crossovers.",
    ),
]


DEFAULT_TASK_KEYS = [
    "introduction_pathways_non-native_plants:0:0",
    "meta_regression:0:0",
    "meta_regression_raw:0:0",
    "nls_incarceration:0:0",
    "nls_raw:2:0",
    "nls_ses:0:0",
    "nls_ses:1:0",
    "requirements_engineering_for_ML_enabled_systems:0:0",
    "worldbank_education_gdp:0:0",
    "worldbank_education_gdp_indicators:0:0",
]


def _run_condition(args: argparse.Namespace, tag: str, modules: str, task_keys: list[str]) -> dict[str, Any]:
    run_id = f"{args.run_id}_{tag}"
    cmd = [
        sys.executable,
        "-m",
        "mars.runners.run_universal_discovery_real_eval",
        "--run_id",
        run_id,
        "--max_tasks",
        "0",
        "--model",
        args.model,
        "--judge_model",
        args.judge_model,
        "--n_proposals",
        str(args.n_proposals),
        "--max_rounds",
        str(args.max_rounds),
        "--discovery_modules",
        modules,
        "--overwrite",
        "--task_keys",
        *task_keys,
    ]
    print(f"\n=== {tag} ===")
    print("modules:", modules)
    subprocess.run(cmd, cwd=_PROJ, check=True)
    summary_path = _PROJ / "lmw" / "universal_discovery_real" / run_id / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    eval_path = Path(summary["official_eval_path"])
    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "tag": tag,
        "modules": modules,
        "description": dict((name, desc) for name, _mods, desc in PRESETS).get(tag, ""),
        "summary_path": str(summary_path),
        "official_eval_path": str(eval_path),
        "mean_hms": float(summary["HMS_mean_100"]),
        "scores": {row["task_key"]: float(row.get("HMS_100") or 0.0) for row in rows},
    }


def _markdown(results: list[dict[str, Any]], task_keys: list[str]) -> str:
    previous = None
    lines = [
        "# DiscoveryBench Module Ablation",
        "",
        "Same official tasks, same model/judge, cumulative module stack.",
        "",
        "| Stack | Modules | Mean HMS | Delta | What it adds |",
        "|---|---|---:|---:|---|",
    ]
    for row in results:
        delta = 0.0 if previous is None else row["mean_hms"] - previous
        previous = row["mean_hms"]
        lines.append(
            f"| `{row['tag']}` | `{row['modules']}` | {row['mean_hms']:.2f} | {delta:+.2f} | {row['description']} |"
        )
    lines.extend(["", "## Per Task", ""])
    header = "| Task | " + " | ".join(f"`{r['tag']}`" for r in results) + " |"
    lines.append(header)
    lines.append("|---|" + "|".join("---:" for _ in results) + "|")
    for key in task_keys:
        vals = " | ".join(f"{r['scores'].get(key, 0.0):.1f}" for r in results)
        lines.append(f"| `{key}` | {vals} |")
    lines.extend(["", "## Artifacts", ""])
    for row in results:
        lines.append(f"- `{row['tag']}`: `{row['summary_path']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="discovery_module_ablation")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--judge_model", default="openai/gpt-4o")
    parser.add_argument("--n_proposals", type=int, default=0)
    parser.add_argument("--max_rounds", type=int, default=0)
    parser.add_argument("--task_keys", nargs="*", default=None)
    args = parser.parse_args()

    task_keys = args.task_keys or DEFAULT_TASK_KEYS
    results = [_run_condition(args, tag, modules, task_keys) for tag, modules, _desc in PRESETS]
    means = [r["mean_hms"] for r in results]
    out_dir = _PROJ / "lmw" / "universal_discovery_real" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "module_ablation.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "model": args.model,
                "judge_model": args.judge_model,
                "task_keys": task_keys,
                "mean_hms_by_stack": {r["tag"]: r["mean_hms"] for r in results},
                "total_delta": means[-1] - means[0] if means else 0.0,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    md = _markdown(results, task_keys)
    (out_dir / "module_ablation.md").write_text(md, encoding="utf-8")
    print("\n=== MODULE ABLATION SUMMARY ===")
    print(md)


if __name__ == "__main__":
    main()
