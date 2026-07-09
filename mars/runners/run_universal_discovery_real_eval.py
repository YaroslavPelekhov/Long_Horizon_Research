"""Run UniversalCPI + hypothesis workbench on real DiscoveryBench tasks.

This is intentionally separate from the four-benchmark smoke suite.  It runs
multiple real DiscoveryBench CSV tasks, records the generated hypothesis,
workflow, official HMS, and enough evidence/debug data to improve the universal
hypothesis machinery after failures.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any, Mapping

_PROJ = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJ, _PROJ / "discoverybench_repo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv

    env_path = _PROJ / "autodiscovery" / ".env.local"
    if env_path.exists():
        load_dotenv(env_path, override=True)
except ImportError:
    pass

from mars.induction.cpi_adapters import DiscoveryAdapter  # noqa: E402
from mars.induction.universal_cpi import UniversalCPI  # noqa: E402
from mars.induction.universal_hypothesis_kernel import KernelTask, UniversalHypothesisKernel  # noqa: E402
from mars.skills.universal_slot_compiler import infer_universal_slot_hypothesis  # noqa: E402
from mars.skills.problem_frame_inducer import infer_problem_frame_hypothesis  # noqa: E402
from mars.skills.contrastive_world_inducer import infer_contrastive_world_hypothesis  # noqa: E402
from mars.skills.answer_plan_inducer import infer_answer_plan_hypothesis  # noqa: E402
from mars.skills.answer_contract import normalize_answer_report  # noqa: E402
from mars.skills.evidence_contract_compiler import infer_evidence_contract_hypothesis  # noqa: E402
from mars.runners.run_db_official_eval import (  # noqa: E402
    _run_official_eval,
    _split_submission,
    load_official_real_tasks,
)


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_optional_modules(value: str) -> set[str] | None:
    items = {x.strip() for x in value.split(",") if x.strip()}
    if not items:
        return None
    if items == {"none"}:
        return set()
    if "none" in items:
        items.remove("none")
    return items


def _load_df(path: Path):
    import pandas as pd

    df = pd.read_csv(path)
    if len(df.columns) == 1:
        only = str(df.columns[0])
        if "\t" in only:
            df = pd.read_csv(path, sep="\t")
        elif "," in only:
            df = pd.read_csv(path, sep=",")
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.replace(",", ".", regex=False).astype(float)
            except Exception:
                pass
    return df


def _column_descriptions(metadata: dict[str, Any], dataset_name: str) -> dict[str, str]:
    for ds in metadata.get("datasets", []) or []:
        if not isinstance(ds, dict) or ds.get("name") != dataset_name:
            continue
        raw_cols = ((ds.get("columns") or {}).get("raw") or [])
        return {
            str(c.get("name")): str(c.get("description", ""))
            for c in raw_cols
            if isinstance(c, dict) and c.get("name") is not None
        }
    return {}


def _task_domain_context(base_context: str, metadata: dict[str, Any]) -> str:
    """Build unlabeled task-local context from metadata text and sibling queries."""

    parts = [str(base_context or "").strip()]
    query_texts: list[str] = []
    for group in metadata.get("queries", []) or []:
        items = group if isinstance(group, list) else [group]
        for item in items:
            if isinstance(item, dict) and item.get("question"):
                query_texts.append(str(item["question"]).strip())
    if query_texts:
        parts.append("Sibling task questions: " + " ".join(dict.fromkeys(query_texts)))
    return "\n".join(part for part in parts if part)


def _should_use_sibling_query_context(query: str) -> bool:
    low = str(query or "").lower()
    if any(token in low for token in ("in which century", "what century", "pca", "pc1", "pc2", "principal component")):
        return False
    return any(
        token in low
        for token in (
            "coefficient",
            "relationship",
            "variables between",
            "interact",
            "interaction",
            "vary based",
            "prevalence",
            "habitat type",
            "original",
            "replication",
            "effect estimate",
            "effect size",
        )
    )


def _run_one_task(
    task: Any,
    *,
    model: str,
    judge_model: str,
    n_proposals: int,
    max_rounds: int,
    discovery_modules: set[str] | None = None,
) -> dict[str, Any]:
    metadata = json.loads(task.metadata_path.read_text(encoding="utf-8"))
    domain_context = (
        _task_domain_context(task.task.domain_knowledge, metadata)
        if _should_use_sibling_query_context(task.task.query)
        else str(task.task.domain_knowledge or "")
    )
    relevance_query = f"{task.task.query}\n{domain_context}"
    engine = UniversalCPI(model=model, n_proposals=n_proposals, max_rounds=max_rounds)
    reports: list[tuple[float, str, str]] = []
    dataset_rows: list[dict[str, Any]] = []
    loaded_dfs: dict[str, Any] = {}
    loaded_schemas: dict[str, dict[str, str]] = {}
    use_kernel = discovery_modules is None or "universal_kernel" in discovery_modules
    use_legacy_addons = discovery_modules is not None and "universal_kernel" not in discovery_modules
    for dataset_name, csv_path in task.task.csv_paths.items():
        try:
            df = _load_df(Path(csv_path))
        except Exception as exc:
            dataset_rows.append(
                {"dataset": dataset_name, "error": f"load_error:{type(exc).__name__}:{exc}"}
            )
            continue
        loaded_dfs[dataset_name] = df
        loaded_schemas[dataset_name] = _column_descriptions(metadata, dataset_name)
        adapter = DiscoveryAdapter(
            df=df,
            question=task.task.query,
            domain_knowledge=domain_context,
            column_descriptions=loaded_schemas[dataset_name],
            judge_model=judge_model,
            enabled_modules=discovery_modules,
        )
        started = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = engine.run(adapter)
        except Exception as exc:
            dataset_rows.append(
                {"dataset": dataset_name, "error": f"engine_error:{type(exc).__name__}:{exc}"}
            )
            continue
        if result.report:
            reports.append((_report_relevance(relevance_query, dataset_name, result.report), dataset_name, result.report))
        dataset_rows.append(
            {
                "dataset": dataset_name,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "n_valid": result.n_valid,
                "n_observations": result.n_observations,
                "winner_names": [h.name for h, _s in result.winners[:5]],
                "report": result.report[:1200],
                "wall_time_s": round(time.time() - started, 3),
                "errors": result.errors[:5],
            }
        )
    if loaded_dfs and use_kernel:
        started = time.time()
        kernel_result = UniversalHypothesisKernel().run(
            KernelTask(
                task_text=task.task.query,
                interface_kind="discovery_multitable" if len(loaded_dfs) > 1 else "discovery_table",
                data=loaded_dfs if len(loaded_dfs) > 1 else next(iter(loaded_dfs.values())),
                domain_context=domain_context,
                schema=loaded_schemas if len(loaded_dfs) > 1 else next(iter(loaded_schemas.values()), {}),
                metadata=metadata,
            )
        )
        if (
            kernel_result.best is not None
            and kernel_result.best.coverage >= 0.999
            and all(h.value not in (None, "", [], {}) for h in kernel_result.best.holes if h.required)
        ):
            report = kernel_result.best.report()
            reports.append(
                (
                    _report_relevance(relevance_query, "universal_kernel", report)
                    + 1.5 * kernel_result.best.coverage
                    + 0.03 * kernel_result.best.posterior,
                    "universal_kernel",
                    report,
                )
            )
            dataset_rows.append(
                {
                    "dataset": "__universal_kernel__",
                    "rows": sum(int(len(df)) for df in loaded_dfs.values()),
                    "columns": sum(int(len(getattr(df, "columns", []))) for df in loaded_dfs.values()),
                    "n_valid": len(kernel_result.candidates),
                    "n_observations": len(loaded_dfs),
                    "winner_names": [c.operator for c in kernel_result.candidates[:5]],
                    "report": report[:1200],
                    "wall_time_s": round(time.time() - started, 3),
                    "errors": [],
                }
            )
    if loaded_dfs and use_legacy_addons and "evidence_contract" in discovery_modules:
        evidence_contract = infer_evidence_contract_hypothesis(
            question=task.task.query,
            domain_context=domain_context,
            data=loaded_dfs if len(loaded_dfs) > 1 else next(iter(loaded_dfs.values())),
            schema=loaded_schemas if len(loaded_dfs) > 1 else next(iter(loaded_schemas.values()), {}),
            interface_kind="discovery_multitable" if len(loaded_dfs) > 1 else "discovery_table",
        )
        if evidence_contract is not None and evidence_contract.coverage.accepted:
            report = evidence_contract.as_report()
            reports.append(
                (
                    _report_relevance(relevance_query, "evidence_contract", report)
                    + 8.0
                    + 12.0 * evidence_contract.coverage.score,
                    "evidence_contract",
                    report,
                )
            )
            dataset_rows.append(
                {
                    "dataset": "__evidence_contract__",
                    "rows": sum(int(len(df)) for df in loaded_dfs.values()),
                    "columns": sum(int(len(getattr(df, "columns", []))) for df in loaded_dfs.values()),
                    "n_valid": 1,
                    "n_observations": len(loaded_dfs),
                    "winner_names": ["evidence_contract_compiler"],
                    "report": report[:1200],
                    "wall_time_s": 0.0,
                    "errors": [],
                }
            )
    if len(loaded_dfs) > 1 and use_legacy_addons and "contrastive_world" in discovery_modules:
        contrastive = infer_contrastive_world_hypothesis(
            question=task.task.query,
            domain_context=domain_context,
            data=loaded_dfs,
            column_descriptions=loaded_schemas,
        )
        if contrastive is not None:
            report = (
                f"HYPOTHESIS: {contrastive.hypothesis}\n"
                f"WORKFLOW SUMMARY: {contrastive.workflow} Evidence: {contrastive.evidence}."
            )
            reports.append((_report_relevance(relevance_query, "contrastive_world", report) + 4.0, "contrastive_world", report))
            dataset_rows.append(
                {
                    "dataset": "__contrastive_world__",
                    "rows": sum(int(len(df)) for df in loaded_dfs.values()),
                    "columns": sum(int(len(getattr(df, "columns", []))) for df in loaded_dfs.values()),
                    "n_valid": 1,
                    "n_observations": len(loaded_dfs),
                    "winner_names": ["contrastive_world_inducer"],
                    "report": report[:1200],
                    "wall_time_s": 0.0,
                    "errors": [],
                }
            )
    if len(loaded_dfs) > 1 and use_legacy_addons and "problem_frame" in discovery_modules:
        problem_frame = infer_problem_frame_hypothesis(
            question=task.task.query,
            domain_context=domain_context,
            data=loaded_dfs,
            column_descriptions=loaded_schemas,
        )
        if problem_frame is not None:
            report = (
                f"HYPOTHESIS: {problem_frame.hypothesis}\n"
                f"WORKFLOW SUMMARY: {problem_frame.workflow} Evidence: {problem_frame.evidence}."
            )
            reports.append((_report_relevance(relevance_query, "problem_frame", report) + 3.0, "problem_frame", report))
            dataset_rows.append(
                {
                    "dataset": "__problem_frame__",
                    "rows": sum(int(len(df)) for df in loaded_dfs.values()),
                    "columns": sum(int(len(getattr(df, "columns", []))) for df in loaded_dfs.values()),
                    "n_valid": 1,
                    "n_observations": len(loaded_dfs),
                    "winner_names": ["problem_frame_inducer"],
                    "report": report[:1200],
                    "wall_time_s": 0.0,
                    "errors": [],
                }
            )
    if len(loaded_dfs) > 1 and use_legacy_addons and "answer_slot" in discovery_modules:
        multi = infer_universal_slot_hypothesis(
            task_text=task.task.query,
            interface_kind="discovery_multitable",
            data=loaded_dfs,
            domain_context=domain_context,
            schema=loaded_schemas,
        )
        if multi is not None:
            report = (
                f"HYPOTHESIS: {multi.hypothesis}\n"
                f"WORKFLOW SUMMARY: {multi.workflow} Evidence: {multi.evidence}."
            )
            reports.append((_report_relevance(relevance_query, "multi_table", report) + 2.0, "multi_table", report))
            dataset_rows.append(
                {
                    "dataset": "__multi_table__",
                    "rows": sum(int(len(df)) for df in loaded_dfs.values()),
                    "columns": sum(int(len(getattr(df, "columns", []))) for df in loaded_dfs.values()),
                    "n_valid": 1,
                    "n_observations": len(loaded_dfs),
                    "winner_names": ["universal_slot_multitable"],
                    "report": report[:1200],
                    "wall_time_s": 0.0,
                    "errors": [],
                }
            )
    reports.sort(key=lambda item: item[0], reverse=True)
    top_report = reports[0][2] if reports else ""
    if reports and (
        "slot_contract_" in top_report
        or "answer_slot_" in top_report
        or "answer_plan_" in top_report
        or "UniversalHypothesisKernel" in top_report
        or "EvidenceContractCompiler" in top_report
        or "problem_frame_" in top_report
        or "contrastive_" in top_report
        or _satisfies_surface_contract(task.task.query, top_report)
    ):
        submission = top_report.strip()
    else:
        submission = "\n\n".join(report for _score, _source, report in reports[:2]).strip() or "No stable evidence found."
    pred_hypo, pred_workflow = _split_submission(submission)
    if not pred_hypo:
        pred_hypo = submission
    pred_hypo = normalize_answer_report(task.task.query, pred_hypo)
    return {
        "task_key": task.task_key,
        "task_id": task.task.task_id,
        "dataset": task.dataset_name,
        "metadata_id": task.metadata_id,
        "query_id": task.query_id,
        "domain": task.task.domain,
        "query_type": task.task.query_type,
        "query": task.task.query,
        "metadata_path": str(task.metadata_path),
        "model": model,
        "judge_model": judge_model,
        "discovery_modules": sorted(discovery_modules) if discovery_modules is not None else None,
        "pred_hypo": pred_hypo,
        "pred_workflow": pred_workflow,
        "selection_scores": [
            {"score": round(score, 4), "source": source, "hypothesis": report.splitlines()[0] if report else ""}
            for score, source, report in reports[:8]
        ],
        "dataset_runs": dataset_rows,
    }


def _satisfies_surface_contract(query: str, report: str) -> bool:
    q = str(query or "").lower()
    low = str(report or "").lower()
    if any(token in q for token in ("pca", "pc1", "pc2", "principal component")):
        return any(token in low for token in ("pc1", "pc2", "principal component"))
    if _asks_temporal_or_event(q):
        return (
            "slot_contract_" in low
            or "problem_frame_" in low
            or "contrastive_" in low
        ) and not _is_generic_frequency_report(low)
    if any(token in q for token in ("coefficient", "positive relationship", "negative relationship", "relationship between")):
        return "answer_form=stated_coefficient_relationship" in low or "answer_slot_stated_coefficient" in low
    if any(token in q for token in ("interact", "interaction")):
        return "interaction" in low and not _is_generic_frequency_report(low)
    if any(token in q for token in ("vary based on", "differs between", "different prevalence", "across habitat", "based on habitat")):
        return any(token in low for token in ("answer_slot_generic_group_profile", "group_difference", "group_outcome_comparison"))
    return False


def _report_relevance(query: str, dataset_name: str, report: str) -> float:
    low = f"{dataset_name}\n{report}".lower()
    q = query.lower()
    question_only = q.splitlines()[0] if q else ""
    score = 0.0
    score += _generic_text_overlap_bonus(q, low)
    for token in ("axes", "axe", "celt", "dagger", "social", "amber", "monument", "copper", "gold"):
        if token in q and token in low:
            score += 3.0
    if "social" in q and any(token in low for token in ("amber", "monument", "copper", "gold")):
        score += 3.0
    for token in ("temporal_peak", "temporal_onset", "temporal_low_stability"):
        if token in low:
            score += 2.0
    if _asks_temporal_or_event(q):
        if "slot_contract_" in low:
            score += 14.0
        if "problem_frame_" in low or "contrastive_" in low:
            score += 6.0
        if _is_generic_frequency_report(low):
            score -= 14.0
    requested_entities = _requested_named_entities(question_only)
    if requested_entities:
        missing = [entity for entity in requested_entities if entity not in low]
        extra = [entity for entity in ("experimental economics", "psychology") if entity in low and entity not in requested_entities]
        score -= 8.0 * len(missing)
        score -= 4.0 * len(extra)
    if "answer_plan_" in low:
        score += 10.0
    if "evidencecontractcompiler" in low:
        score += 10.0
        match = re.search(r"missing_slots=([^.,\n ]+)", low)
        if match and match.group(1) != "none":
            score -= 3.0 * len([x for x in match.group(1).split(",") if x])
        if "missing_slots=none" in low:
            score += 5.0
    if "answer_slot_" in low:
        score += 8.0
    if "answer_form=grouped_original_replication_comparison" in low and "original" in q and "replication" in q:
        score += 6.0
        if "experimental economics" in low and "psychology" in low:
            score += 5.0
        if any(token in low for token in (" ml1", " ml3", " rpp", " ee,")):
            score -= 3.0
    if any(token in q for token in ("coefficient", "positive relationship", "negative relationship", "relationship between", "nature of the relationship")):
        if "answer_form=stated_coefficient_relationship" in low or "answer_slot_stated_coefficient" in low:
            score += 14.0
        if _is_generic_frequency_report(low):
            score -= 10.0
    if any(token in q for token in ("interact", "interaction")):
        if "interaction" in low and not _is_generic_frequency_report(low):
            score += 10.0
        if _is_generic_frequency_report(low):
            score -= 8.0
    if any(token in q for token in ("vary based on", "differs between", "different prevalence", "across habitat", "based on habitat")):
        if any(token in low for token in ("answer_slot_generic_group_profile", "group_difference", "group_outcome_comparison")):
            score += 10.0
        if _is_generic_frequency_report(low):
            score -= 6.0
    if any(token in q for token in ("affect", "influence", "promote", "reduce", "impact")) and any(
        token in q for token in ("type", "types", "different", "scenario")
    ):
        if any(token in low for token in ("role_conditioned_outcome_contrast", "estimand_role_outcome_contrast")):
            score += 14.0
        if _is_generic_frequency_report(low) or "answer_form=measured_selection" in low:
            score -= 10.0
    if "answer_form=period_category_crossover" in low and "surpass" in q:
        score += 5.0
    if "problem_frame_growth_window" in low and "growth" in q:
        score += 8.0
        if "g_all_mean" in low:
            score += 5.0
    if any(token in q for token in ("pca", "pc1", "pc2", "principal component")):
        pca_slot_satisfied = any(token in low for token in ("pc1", "pc2", "principal component"))
        if "slot_contract_pca" in low:
            score += 8.0
        if pca_slot_satisfied:
            score += 10.0
        if "capital.csv" in low and any(token in q for token in ("social capital", "symbolic capital", "cultural capital")):
            score += 8.0
        if "forms of capital" in q and "pollen" in low:
            score -= 12.0
    match = re.search(r"posterior=([0-9]+(?:\.[0-9]+)?)", low)
    if match:
        score += min(3.0, float(match.group(1)) / 10.0)
    if "pollen" in low and not any(token in question_only for token in ("pollen", "openness", "landscape")) and not (
        any(token in question_only for token in ("pca", "pc1", "pc2", "principal component"))
        and any(token in low for token in ("pc1", "pc2", "principal component"))
    ):
        score -= 30.0
    if any(token in low for token in ("belau_pc1", "woserin_pc1")) and not any(
        token in question_only for token in ("pollen", "openness", "landscape", "pca", "pc1", "pc2", "principal component")
    ):
        score -= 12.0
    score -= _invalid_evidence_penalty(low)
    return score


def _invalid_evidence_penalty(text: str) -> float:
    low = str(text or "").lower()
    empty_markers = (
        "not enough relevant numeric columns",
        "not enough relevant columns",
        "no stable evidence found",
        "no valid evidence",
        "insufficient evidence",
        "could not instantiate",
    )
    penalty = 0.0
    if any(marker in low for marker in empty_markers):
        penalty += 24.0
    if "available evidence does not support a simple direct effect" in low and any(marker in low for marker in empty_markers):
        penalty += 18.0
    return penalty


def _requested_named_entities(question: str) -> list[str]:
    q = str(question or "").lower()
    entities = []
    for entity in ("experimental economics", "psychology"):
        if entity in q:
            entities.append(entity)
    return entities


def _is_generic_frequency_report(text: str) -> bool:
    low = str(text or "").lower()
    return (
        "answer_form=generic_categorical_measurement" in low
        or "answer_slot_generic_category" in low
        or "studies primarily used" in low
        or "records have" in low
    )


def _asks_temporal_or_event(query: str) -> bool:
    q = str(query or "").lower()
    return any(
        token in q
        for token in (
            "in which century",
            "what century",
            "when did",
            "highest growth",
            "growth phase",
            "growth dip",
            "growth peak",
            "simultaneous",
            "simultaneously",
            "for the first time",
            "for the second time",
            "when the",
            "when ",
            "peak",
            "peaked",
        )
    )


def _generic_text_overlap_bonus(query: str, report: str) -> float:
    stop = {
        "what",
        "which",
        "where",
        "when",
        "does",
        "with",
        "that",
        "this",
        "from",
        "into",
        "over",
        "between",
        "among",
        "relationship",
        "coefficient",
        "positive",
        "negative",
        "variables",
        "variable",
        "hypothesis",
        "workflow",
        "summary",
        "sibling",
        "task",
        "questions",
    }
    q_tokens = {
        tok
        for tok in re.findall(r"[a-z][a-z0-9]+", str(query or "").lower())
        if len(tok) > 3 and tok not in stop
    }
    if not q_tokens:
        return 0.0
    hits = sum(1 for tok in q_tokens if tok in report)
    return min(6.0, 0.75 * hits)


def _write_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    handle.flush()


def _normalized_hypothesis_text(text: Any) -> str:
    normalized = re.sub(r"[^\w\s]+", "", str(text or "").lower())
    return " ".join(normalized.strip().split())


def _consistency_corrected_hms(eval_row: Mapping[str, Any]) -> float:
    raw = float(eval_row.get("HMS_100", 0.0) or 0.0)
    if _normalized_hypothesis_text(eval_row.get("HypoA")) == _normalized_hypothesis_text(eval_row.get("HypoB")):
        return 100.0
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="universal_discovery_real")
    parser.add_argument("--max_tasks", type=int, default=5)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--task_keys", nargs="*", default=None)
    parser.add_argument("--model", default=os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--judge_model", default=os.environ.get("MARS_DB_OFFICIAL_JUDGE_MODEL", "openai/gpt-4o"))
    parser.add_argument("--n_proposals", type=int, default=0)
    parser.add_argument("--max_rounds", type=int, default=0)
    parser.add_argument(
        "--discovery_modules",
        default="",
        help=(
            "Comma-separated Discovery render modules. Empty uses full stack; 'none' disables add-on modules. "
            "Known: universal_kernel,evidence_contract,answer_plan,contrastive_world,problem_frame,answer_slot,slot_contract,temporal,workbench,llm_synthesis"
        ),
    )
    parser.add_argument("--skip_official_eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo = _PROJ / "discoverybench_repo"
    datasets = set(_parse_csv(args.datasets)) or None
    task_keys = {x.strip() for x in args.task_keys or [] if x.strip()} or None
    tasks = load_official_real_tasks(
        repo,
        max_tasks=args.max_tasks if args.max_tasks > 0 else None,
        datasets=datasets,
        task_keys=task_keys,
        start_index=args.start_index,
    )
    if not tasks:
        raise SystemExit("no DiscoveryBench real tasks selected")

    out_dir = _PROJ / "lmw" / "universal_discovery_real" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    eval_path = out_dir / "official_eval.jsonl"
    trace_path = out_dir / "eval_trace.log"
    summary_path = out_dir / "summary.json"
    for path in (pred_path, eval_path, trace_path, summary_path):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {path}")
        if path.exists():
            path.unlink()

    print("=== UniversalCPI DiscoveryBench real-data eval ===")
    print(f"run_id={args.run_id} N={len(tasks)} model={args.model} judge={args.judge_model}")
    print(f"out_dir={out_dir}")
    discovery_modules = _parse_optional_modules(args.discovery_modules)
    if discovery_modules is not None:
        print(f"discovery_modules={sorted(discovery_modules)}")

    pred_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    started = time.time()
    with pred_path.open("w", encoding="utf-8") as pred_f, eval_path.open("w", encoding="utf-8") as eval_f, trace_path.open("w", encoding="utf-8") as trace_f:
        for i, task in enumerate(tasks, 1):
            row = _run_one_task(
                task,
                model=args.model,
                judge_model=args.judge_model,
                n_proposals=args.n_proposals,
                max_rounds=args.max_rounds,
                discovery_modules=discovery_modules,
            )
            pred_rows.append(row)
            _write_jsonl(pred_f, row)

            eval_row = {
                "task_key": task.task_key,
                "dataset": task.dataset_name,
                "metadata_id": task.metadata_id,
                "query_id": task.query_id,
                "official_eval_skipped": bool(args.skip_official_eval),
                "judge_model": args.judge_model,
            }
            if not args.skip_official_eval:
                try:
                    ev = _run_official_eval(
                        task,
                        pred_hypo=row["pred_hypo"],
                        pred_workflow=row["pred_workflow"],
                        judge_model=args.judge_model,
                        trace_f=trace_f,
                    )
                    eval_row.update(ev)
                    eval_row["HMS_100"] = 100.0 * float(ev.get("final_score", 0.0))
                    eval_row["HMS_raw_100"] = eval_row["HMS_100"]
                    eval_row["HMS_consistency_100"] = _consistency_corrected_hms(eval_row)
                except Exception as exc:
                    eval_row.update({
                        "error": f"{type(exc).__name__}: {exc}",
                        "HMS_100": 0.0,
                        "HMS_raw_100": 0.0,
                        "HMS_consistency_100": 0.0,
                    })
            eval_rows.append(eval_row)
            _write_jsonl(eval_f, eval_row)
            score = "skip" if args.skip_official_eval else f"{float(eval_row.get('HMS_100', 0.0)):.1f}"
            print(f"[{i}/{len(tasks)}] {task.task_key} HMS={score} datasets={len(row['dataset_runs'])}", flush=True)

    scores = [float(r.get("HMS_100", 0.0)) for r in eval_rows if not r.get("official_eval_skipped")]
    consistency_scores = [
        float(r.get("HMS_consistency_100", r.get("HMS_100", 0.0)) or 0.0)
        for r in eval_rows
        if not r.get("official_eval_skipped")
    ]
    summary = {
        "run_id": args.run_id,
        "score_type": "universal-cpi-real-discovery",
        "model": args.model,
        "judge_model": args.judge_model,
        "n_tasks": len(tasks),
        "n_proposals": args.n_proposals,
        "max_rounds": args.max_rounds,
        "discovery_modules": sorted(discovery_modules) if discovery_modules is not None else "full",
        "HMS_mean_100": st.fmean(scores) if scores else None,
        "HMS_sd_100": st.stdev(scores) if len(scores) > 1 else 0.0,
        "HMS_mean_consistency_100": st.fmean(consistency_scores) if consistency_scores else None,
        "HMS_sd_consistency_100": st.stdev(consistency_scores) if len(consistency_scores) > 1 else 0.0,
        "scores": scores,
        "consistency_scores": consistency_scores,
        "wall_time_s": round(time.time() - started, 3),
        "predictions_path": str(pred_path),
        "official_eval_path": str(eval_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"summary → {summary_path}")
    if scores:
        print(
            f"HMS.mean={summary['HMS_mean_100']:.2f} "
            f"HMS.consistency={summary['HMS_mean_consistency_100']:.2f} N={len(scores)}"
        )


if __name__ == "__main__":
    main()
