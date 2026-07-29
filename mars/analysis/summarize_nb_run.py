"""Summarize NewtonBench active-probe runs into paper-ready diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _bucket(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    out = {}
    for name, group in sorted(groups.items()):
        n = len(group)
        answered = sum(r.get("status") == "ANSWER" for r in group)
        exact = sum(float(r.get("SA", 0.0) or 0.0) for r in group)
        num002 = sum(
            r.get("status") == "ANSWER" and float(r.get("held_out_rmsle", 9.9) or 9.9) <= 0.02
            for r in group
        )
        num010 = sum(
            r.get("status") == "ANSWER" and float(r.get("held_out_rmsle", 9.9) or 9.9) <= 0.10
            for r in group
        )
        out[name] = {
            "n": n,
            "answered": answered,
            "SA_all": round(exact / n, 4) if n else 0.0,
            "numeric_pass_002_all": round(num002 / n, 4) if n else 0.0,
            "numeric_pass_010_all": round(num010 / n, 4) if n else 0.0,
        }
    return out


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text())
    rows = list(data.get("rows", []))
    n = len(rows)
    answered = sum(r.get("status") == "ANSWER" for r in rows)
    exact = sum(float(r.get("SA", 0.0) or 0.0) for r in rows)
    num002 = [
        r for r in rows
        if r.get("status") == "ANSWER" and float(r.get("held_out_rmsle", 9.9) or 9.9) <= 0.02
    ]
    num010 = [
        r for r in rows
        if r.get("status") == "ANSWER" and float(r.get("held_out_rmsle", 9.9) or 9.9) <= 0.10
    ]
    numeric_only = [
        r for r in num010
        if float(r.get("SA", 0.0) or 0.0) <= 0.0
    ]
    methods = Counter(str(r.get("method", "")) for r in rows)
    statuses = Counter(str(r.get("status", "")) for r in rows)
    return {
        "source": str(path),
        "n": n,
        "answered": answered,
        "abstained": n - answered,
        "SA_all": round(exact / n, 4) if n else 0.0,
        "SA_answered": round(exact / answered, 4) if answered else 0.0,
        "numeric_pass_002_all": round(len(num002) / n, 4) if n else 0.0,
        "numeric_pass_010_all": round(len(num010) / n, 4) if n else 0.0,
        "numeric_only_010": len(numeric_only),
        "status_counts": dict(statuses.most_common()),
        "method_counts": dict(methods.most_common()),
        "by_module": _bucket(rows, "module"),
        "by_system": _bucket(rows, "system"),
        "by_difficulty": _bucket(rows, "difficulty"),
        "numeric_only_examples": [
            {
                "module": r.get("module"),
                "difficulty": r.get("difficulty"),
                "law_version": r.get("law_version"),
                "system": r.get("system"),
                "held_out_rmsle": r.get("held_out_rmsle"),
                "method": r.get("method"),
                "law": r.get("law", ""),
            }
            for r in numeric_only[:20]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    summary = summarize(args.summary_json)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
