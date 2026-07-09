"""Analyze DiscoveryBench failures through metric-aligned slots.

Usage:
  python3 -m mars.runners.analyze_db_metric_slots \
    --run_dir lmw/db_cpi/full239_db_cpi_gpt4omini_gpt4o_20260617

The script reads official_eval.jsonl and writes slot_failure_summary.json into
the same run directory.  It does not call any model.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping

from mars.skills.metric_compiler import (
    CompiledHypothesis,
    MetricContract,
    residuals_from_score,
    score_contract,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_sub_hypo(container: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    subs = ((container.get(key) or {}).get("sub_hypo") or [])
    return subs[0] if subs else None


def _compiled_pair(eval_result: Mapping[str, Any]) -> tuple[CompiledHypothesis, CompiledHypothesis] | None:
    gold = _first_sub_hypo(eval_result, "gold_sub_hypo")
    gen = _first_sub_hypo(eval_result, "gen_sub_hypo")
    if not gold or not gen:
        return None
    return (
        CompiledHypothesis.from_discoverybench_sub_hypo(gold),
        CompiledHypothesis.from_discoverybench_sub_hypo(gen),
    )


def _official_pair_scores(eval_result: Mapping[str, Any]) -> dict[str, float]:
    """Extract the official slot scores when DB matched a generated subhypothesis."""

    matched = eval_result.get("matched_gold_gen_subh_evals") or {}
    if not matched:
        return {}
    context_scores: list[float] = []
    variable_scores: list[float] = []
    relation_scores: list[float] = []
    accuracy_scores: list[float] = []
    for item in matched.values():
        context_scores.append(_num(((item.get("context") or {}).get("score")), 0.0))
        variable_scores.append(_num((((item.get("var") or {}).get("score") or {}).get("f1")), 0.0))
        relation_scores.append(_num(((item.get("rel") or {}).get("score")), 0.0))
        accuracy_scores.append(_num(item.get("accuracy_score"), 0.0))
    return {
        "official_context": sum(context_scores) / len(context_scores),
        "official_variables": sum(variable_scores) / len(variable_scores),
        "official_relation": sum(relation_scores) / len(relation_scores),
        "official_accuracy": sum(accuracy_scores) / len(accuracy_scores),
    }


def _slot_failure_names(eval_result: Mapping[str, Any], lexical_failed: tuple[str, ...]) -> tuple[str, ...]:
    """Prefer official slot diagnostics, fall back to lexical compiler diagnostics."""

    official = _official_pair_scores(eval_result)
    if not official:
        if _num(eval_result.get("recall_context"), 0.0) <= 0:
            return ("context", "variables", "relation")
        return lexical_failed
    failed: list[str] = []
    if official["official_context"] < 0.72:
        failed.append("context")
    if official["official_variables"] < 0.72:
        failed.append("variables")
    if official["official_relation"] < 0.72:
        failed.append("relation")
    return tuple(failed)


def analyze_run(run_dir: Path) -> dict[str, Any]:
    rows = _load_jsonl(run_dir / "official_eval.jsonl")
    predictions = {
        str(row.get("task_key")): row
        for row in _load_jsonl(run_dir / "predictions.jsonl")
        if row.get("task_key") is not None
    }
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "n_rows": len(rows),
        "n_scored": 0,
        "n_errors_or_skipped": 0,
        "hms_mean_100": 0.0,
        "slot_failures": Counter(),
        "by_dataset": defaultdict(lambda: {"n": 0, "hms_sum": 0.0, "failures": Counter()}),
        "by_query_type": defaultdict(lambda: {"n": 0, "hms_sum": 0.0, "failures": Counter()}),
        "residual_skill_schemas": Counter(),
        "examples": [],
    }

    hms_values: list[float] = []
    for row in rows:
        eval_result = row.get("eval_result") or {}
        pred_row = predictions.get(str(row.get("task_key")), {})
        dataset = str(row.get("dataset", "unknown"))
        query_type = str(row.get("query_type") or pred_row.get("query_type") or "unknown")
        skipped = row.get("official_eval_skipped") or not eval_result
        if skipped:
            summary["n_errors_or_skipped"] += 1
            continue

        hms = _num(row.get("HMS_100"), 0.0)
        hms_values.append(hms)
        summary["n_scored"] += 1
        summary["by_dataset"][dataset]["n"] += 1
        summary["by_dataset"][dataset]["hms_sum"] += hms
        summary["by_query_type"][query_type]["n"] += 1
        summary["by_query_type"][query_type]["hms_sum"] += hms

        pair = _compiled_pair(eval_result)
        lexical_failed: tuple[str, ...] = ()
        lexical_residuals = ()
        if pair is not None:
            gold, gen = pair
            contract = MetricContract(name=str(row.get("task_key", "db_task")), expected=gold)
            contract_score = score_contract(contract, gen)
            lexical_failed = contract_score.failed_slots
            lexical_residuals = residuals_from_score(contract_score)

        failed_slots = _slot_failure_names(eval_result, lexical_failed)
        for slot in failed_slots:
            summary["slot_failures"][slot] += 1
            summary["by_dataset"][dataset]["failures"][slot] += 1
            summary["by_query_type"][query_type]["failures"][slot] += 1

        for residual in lexical_residuals:
            if residual.slot in failed_slots:
                summary["residual_skill_schemas"][" ".join(residual.suggested_skill_schema)] += 1

        if failed_slots and len(summary["examples"]) < 12:
            summary["examples"].append(
                {
                    "task_key": row.get("task_key"),
                    "dataset": dataset,
                    "query_type": query_type,
                    "hms_100": hms,
                    "failed_slots": failed_slots,
                    "query": eval_result.get("query"),
                    "gold": eval_result.get("HypoA"),
                    "generated": eval_result.get("HypoB"),
                }
            )

    summary["hms_mean_100"] = sum(hms_values) / len(hms_values) if hms_values else 0.0

    def finalize_group(group: Mapping[str, Any]) -> dict[str, Any]:
        n = int(group["n"])
        return {
            "n": n,
            "hms_mean_100": group["hms_sum"] / n if n else 0.0,
            "failures": dict(group["failures"]),
        }

    summary["slot_failures"] = dict(summary["slot_failures"])
    summary["by_dataset"] = {key: finalize_group(value) for key, value in summary["by_dataset"].items()}
    summary["by_query_type"] = {key: finalize_group(value) for key, value in summary["by_query_type"].items()}
    summary["residual_skill_schemas"] = dict(summary["residual_skill_schemas"].most_common(30))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary = analyze_run(run_dir)
    out_path = Path(args.output) if args.output else run_dir / "slot_failure_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
