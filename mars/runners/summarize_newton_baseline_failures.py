"""Audit the form of submitted NewtonBench baseline laws.

The matched-control headline reports strict symbolic accuracy.  This audit
separates parser failures from executable but scientifically incorrect laws so
that a low score cannot be mistaken for an empty-output harness.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


BACKENDS = ("vanilla_agent", "code_assisted_agent")
REPRESENTATIVE_MODULES = (
    "m0_gravity",
    "m5_radioactive_decay",
    "m9_hooke_law",
)


def classify_law(source: str, exact: bool) -> str:
    if exact:
        return "exact"
    source = source.strip()
    if not source:
        return "empty"
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "invalid_syntax"

    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        return "non_function"

    function = functions[0]
    arguments = {argument.arg for argument in function.args.args}
    used_names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Return) and node.value is not None:
            used_names.update(
                child.id
                for child in ast.walk(node.value)
                if isinstance(child, ast.Name)
            )
    if not arguments.intersection(used_names):
        return "constant_only"
    return "valid_but_incorrect"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_trial(row: dict[str, str]) -> dict[str, Any]:
    with Path(row["trial_path"]).open(encoding="utf-8") as handle:
        return json.load(handle)


def audit(evidence_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    representatives: list[dict[str, Any]] = []

    for backend in BACKENDS:
        rows = _load_rows(evidence_root / f"{backend}_tasks.csv")
        categories: Counter[str] = Counter()
        chosen: set[str] = set()

        for row in rows:
            trial = _load_trial(row)
            evaluation = dict(trial.get("evaluation") or {})
            source = str(trial.get("submitted_law") or "")
            category = classify_law(
                source, bool(evaluation.get("exact_accuracy", 0))
            )
            categories[category] += 1

            module = str(trial.get("module_name") or "")
            if module in REPRESENTATIVE_MODULES and module not in chosen:
                representatives.append(
                    {
                        "backend": backend,
                        "module": module,
                        "difficulty": trial.get("equation_difficulty"),
                        "system": trial.get("model_system"),
                        "law_version": trial.get("law_version"),
                        "category": category,
                        "terminal_status": trial.get("status"),
                        "rounds": trial.get("rounds"),
                        "num_experiments": trial.get("num_experiments"),
                        "submitted_law": source.replace("\n", "\\n"),
                        "rmsle": evaluation.get("rmsle"),
                        "exact_accuracy": evaluation.get("exact_accuracy"),
                    }
                )
                chosen.add(module)

        for category in (
            "empty",
            "invalid_syntax",
            "non_function",
            "constant_only",
            "valid_but_incorrect",
            "exact",
        ):
            summary.append(
                {
                    "backend": backend,
                    "category": category,
                    "count": categories[category],
                    "total": len(rows),
                }
            )

    return summary, representatives


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    summary, representatives = audit(args.evidence_root)
    _write_csv(args.output_root / "baseline_failure_audit.csv", summary)
    _write_csv(
        args.output_root / "baseline_representative_outputs.csv",
        representatives,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
