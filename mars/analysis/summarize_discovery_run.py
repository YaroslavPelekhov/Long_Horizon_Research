"""Summarize DiscoveryBench real-eval runs into domain and failure diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _norm(text: object) -> str:
    s = str(text or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _score(row: dict, key: str) -> float:
    value = row.get(key)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bucket(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    out: dict[str, dict] = {}
    for name, group in sorted(groups.items()):
        raw = [_score(r, "HMS_raw_100") for r in group]
        consistency = [_score(r, "HMS_consistency_100") for r in group]
        zeros = [r for r in group if _score(r, "HMS_raw_100") <= 0.0]
        exact_text_zero = [
            r for r in zeros
            if _norm(r.get("HypoA")) == _norm(r.get("HypoB")) and _norm(r.get("HypoA"))
        ]
        out[name] = {
            "n": len(group),
            "HMS_raw_100": round(mean(raw), 4) if raw else 0.0,
            "HMS_consistency_100": round(mean(consistency), 4) if consistency else 0.0,
            "raw_zero_count": len(zeros),
            "exact_text_zero_count": len(exact_text_zero),
        }
    return out


def summarize(run_dir: Path) -> dict:
    eval_rows = _load_jsonl(run_dir / "official_eval.jsonl")
    pred_rows = _load_jsonl(run_dir / "predictions.jsonl")
    pred_by_key = {str(r.get("task_key")): r for r in pred_rows}
    for row in eval_rows:
        pred = pred_by_key.get(str(row.get("task_key")), {})
        row.setdefault("query_type", pred.get("query_type"))
        row.setdefault("domain", pred.get("domain"))
        row.setdefault("metadata_path", pred.get("metadata_path"))
    raw = [_score(r, "HMS_raw_100") for r in eval_rows]
    consistency = [_score(r, "HMS_consistency_100") for r in eval_rows]
    zeros = [r for r in eval_rows if _score(r, "HMS_raw_100") <= 0.0]
    exact_text_zero = [
        r for r in zeros
        if _norm(r.get("HypoA")) == _norm(r.get("HypoB")) and _norm(r.get("HypoA"))
    ]
    return {
        "run_dir": str(run_dir),
        "n_eval": len(eval_rows),
        "n_predictions": len(pred_rows),
        "HMS_raw_100": round(mean(raw), 4) if raw else 0.0,
        "HMS_consistency_100": round(mean(consistency), 4) if consistency else 0.0,
        "raw_zero_count": len(zeros),
        "exact_text_zero_count": len(exact_text_zero),
        "by_dataset": _bucket(eval_rows, "dataset"),
        "by_domain": _bucket(eval_rows, "domain"),
        "by_query_type": _bucket(eval_rows, "query_type"),
        "zero_examples": [
            {
                "task_key": r.get("task_key"),
                "dataset": r.get("dataset"),
                "query_type": r.get("query_type"),
                "query": r.get("query"),
                "HMS_consistency_100": r.get("HMS_consistency_100"),
                "gold": r.get("HypoA"),
                "pred": r.get("HypoB"),
            }
            for r in zeros[:25]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    summary = summarize(args.run_dir)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
