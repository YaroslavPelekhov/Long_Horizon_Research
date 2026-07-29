"""Read-only operator-use audit for DiscoveryBench prediction traces.

The audit joins a completed prediction trace with its official evaluator JSONL
by task key.  It reports which executable operator supplied the selected
artifact, how often it was reused, and the observed HMS of those tasks.  No
model, judge, or benchmark input is invoked or modified.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _selected_operator(row: dict[str, Any]) -> str:
    for item in row.get("selection_scores", []):
        source = str(item.get("source", ""))
        if source.startswith("universal_kernel:"):
            return source.split(":", 1)[1]
    return "other"


def audit(predictions: list[dict[str, Any]], official: list[dict[str, Any]]) -> dict[str, Any]:
    hms_by_key = {
        str(row.get("task_key")): float(row.get("HMS_100", 0.0) or 0.0)
        for row in official
        if row.get("task_key")
    }
    grouped: dict[str, list[float]] = defaultdict(list)
    domains: dict[str, set[str]] = defaultdict(set)
    for row in predictions:
        key = str(row.get("task_key", ""))
        if key not in hms_by_key:
            continue
        operator = _selected_operator(row)
        grouped[operator].append(hms_by_key[key])
        domains[operator].add(str(row.get("domain", "unknown")))
    items = []
    for operator, scores in grouped.items():
        items.append(
            {
                "operator": operator,
                "tasks_selected": len(scores),
                "mean_HMS_100": st.fmean(scores),
                "nonzero_HMS_tasks": sum(score > 0 for score in scores),
                "domains": sorted(domains[operator]),
                "n_domains": len(domains[operator]),
            }
        )
    items.sort(key=lambda item: (-item["tasks_selected"], item["operator"]))
    return {
        "n_joined_tasks": sum(len(scores) for scores in grouped.values()),
        "operators": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(_rows(args.predictions), _rows(args.official))
    result.update({"predictions": str(args.predictions), "official": str(args.official)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
