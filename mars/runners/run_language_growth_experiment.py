"""Prepare and execute the clean DiscoveryBench language-growth experiment.

The protocol grows a fresh operator library on the released development split,
freezes it, and evaluates the frozen language on the complete official test
split.  All conditions keep the per-task search budget and rendering stack
fixed; only cross-task language loading and promotion differ.

Preparation is the default.  API-backed runs start only with ``--execute``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mars.runners.run_db_official_eval import (
    OfficialDBTask,
    load_official_real_tasks,
    load_official_train_tasks,
)
from mars.skills.self_layer_registry import validate_self_layer_source
from mars.skills.self_module_registry import validate_self_module_source


_PROJ = Path(__file__).resolve().parents[2]
_RUNNER = "mars.runners.run_universal_discovery_real_eval"
_PARALLEL_TEST_RUNNER = "mars.runners.run_discovery_chunked_eval"
_CONDITIONS = ("fixed_l0", "promote_all", "gated_growth")


@dataclass(frozen=True)
class ProtocolTask:
    split: str
    task_key: str
    group_key: str
    dataset: str
    metadata_path: str


@dataclass(frozen=True)
class CommandSpec:
    name: str
    condition: str
    phase: str
    command: list[str]
    env: dict[str, str]
    expected_summary: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_key(task: OfficialDBTask) -> str:
    return f"{task.dataset_name}:metadata_{task.metadata_id}"


def _stable_group_split(
    tasks: list[OfficialDBTask],
    *,
    seed: int,
    induction_fraction: float,
) -> tuple[list[OfficialDBTask], list[OfficialDBTask]]:
    groups: dict[str, list[OfficialDBTask]] = {}
    for task in tasks:
        groups.setdefault(_group_key(task), []).append(task)
    ranked = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
    )
    n_induction = max(1, min(len(ranked) - 1, round(len(ranked) * induction_fraction)))
    induction_groups = set(ranked[:n_induction])
    induction = [task for task in tasks if _group_key(task) in induction_groups]
    promotion = [task for task in tasks if _group_key(task) not in induction_groups]
    return induction, promotion


def _task_rows(tasks: list[OfficialDBTask], split: str) -> list[ProtocolTask]:
    return [
        ProtocolTask(
            split=split,
            task_key=task.task_key,
            group_key=_group_key(task),
            dataset=task.dataset_name,
            metadata_path=str(task.metadata_path),
        )
        for task in tasks
    ]


def _language_env(root: Path, condition: str, phase: str) -> dict[str, str]:
    env = {
        "MARS_SELF_MODULE_ROOT": str(root / "modules"),
        "MARS_SELF_LAYER_ROOT": str(root / "layers"),
        "MARS_RESIDUAL_LEDGER_ROOT": str(root / "residual_classes"),
        "MARS_SELF_LAYER_NAMESPACE": "universal",
        "MARS_SELF_WRITE_MODULES": "1",
        "MARS_SELF_WRITE_LAYERS": "1",
        "MARS_PROPOSE_SELF_LAYERS": "1",
        "MARS_RESIDUAL_OPERATOR_LAYERS": "1",
        "MARS_OPENAI_TIMEOUT_S": "120",
        "MARS_LOAD_CANDIDATE_MODULES": "0",
        "MARS_LOAD_CANDIDATE_LAYERS": "0",
    }
    if condition == "fixed_l0":
        env.update(
            {
                "MARS_SELF_WRITE_MODULES": "0",
                "MARS_SELF_WRITE_LAYERS": "0",
                "MARS_LOAD_TRUSTED_MODULES": "0",
                "MARS_LOAD_TRUSTED_LAYERS": "0",
                "MARS_PROMOTE_SELF_MODULES": "0",
                "MARS_PROMOTE_SELF_LAYERS": "0",
            }
        )
    elif phase == "induce":
        env.update(
            {
                "MARS_LOAD_TRUSTED_MODULES": "1",
                "MARS_LOAD_TRUSTED_LAYERS": "1",
                "MARS_PROMOTE_SELF_MODULES": "1",
                "MARS_PROMOTE_SELF_LAYERS": "1",
            }
        )
        if condition == "gated_growth":
            # Source tasks may place a safe executable proposal in quarantine,
            # but cannot grant it language membership.
            env.update(
                {
                    "MARS_QUARANTINE_PROMOTION": "1",
                    "MARS_RESIDUAL_CLASS_INDUCTION": "1",
                    # Isolate residual-class induction from the legacy path
                    # that lifts a source-task winner into a library record.
                    "MARS_RESIDUAL_CLASS_ONLY_PROMOTION": "1",
                    "MARS_RESIDUAL_CLASS_MIN_SUPPORT": "2",
                    "MARS_SELF_MODULE_MIN_PROBE_KEYS": "2",
                    "MARS_SELF_LAYER_MIN_PROBE_KEYS": "2",
                    "MARS_SELF_MODULE_MIN_KEYS": "2",
                    "MARS_SELF_LAYER_MIN_KEYS": "2",
                    "MARS_SELF_MODULE_TRUSTED_MIN_EXACT": "0.20",
                    "MARS_SELF_MODULE_TRUSTED_MAX_LOSS": "0.80",
                    "MARS_SELF_MODULE_MIN_PROBE_RELATIVE_GAIN": "0.01",
                    "MARS_SELF_LAYER_TRUSTED_MIN_GAIN": "0.20",
                    "MARS_SELF_LAYER_TRUSTED_MAX_LOSS": "0.80",
                    "MARS_SELF_LAYER_MIN_PROBE_RELATIVE_GAIN": "0.01",
                }
            )
    elif phase == "promotion_probe":
        # Candidate artifacts are evaluated on held-out development groups.
        # No new candidate is proposed here: this pass only records transfer
        # evidence and may admit candidates that satisfy the cross-task gate.
        env.update(
            {
                "MARS_SELF_WRITE_MODULES": "1",
                "MARS_SELF_WRITE_LAYERS": "1",
                "MARS_LOAD_TRUSTED_MODULES": "1",
                "MARS_LOAD_TRUSTED_LAYERS": "1",
                "MARS_LOAD_CANDIDATE_MODULES": "1",
                "MARS_LOAD_CANDIDATE_LAYERS": "1",
                "MARS_PROPOSE_SELF_LAYERS": "0",
                "MARS_PROMOTE_SELF_MODULES": "1",
                "MARS_PROMOTE_SELF_LAYERS": "1",
                "MARS_PROMOTION_PROBE": "1",
                "MARS_RESIDUAL_CLASS_ONLY_PROMOTION": "1",
                "MARS_CANDIDATE_MODULE_BUDGET": "6",
                "MARS_CANDIDATE_LAYER_BUDGET": "4",
                "MARS_SELF_MODULE_MIN_PROBE_KEYS": "2",
                "MARS_SELF_LAYER_MIN_PROBE_KEYS": "2",
                "MARS_SELF_MODULE_MIN_KEYS": "2",
                "MARS_SELF_LAYER_MIN_KEYS": "2",
                "MARS_SELF_MODULE_TRUSTED_MIN_EXACT": "0.20",
                "MARS_SELF_MODULE_TRUSTED_MAX_LOSS": "0.80",
                "MARS_SELF_MODULE_MIN_PROBE_RELATIVE_GAIN": "0.01",
                "MARS_SELF_LAYER_TRUSTED_MIN_GAIN": "0.20",
                "MARS_SELF_LAYER_TRUSTED_MAX_LOSS": "0.80",
                "MARS_SELF_LAYER_MIN_PROBE_RELATIVE_GAIN": "0.01",
                "MARS_PROMOTION_EVIDENCE_CANDIDATES": "8",
            }
        )
    else:
        # Frozen development/test evaluation: reuse is allowed, mutation is not.
        env.update(
            {
                "MARS_SELF_WRITE_MODULES": "0",
                "MARS_SELF_WRITE_LAYERS": "0",
                "MARS_LOAD_TRUSTED_MODULES": "1",
                "MARS_LOAD_TRUSTED_LAYERS": "1",
                "MARS_PROMOTE_SELF_MODULES": "0",
                "MARS_PROMOTE_SELF_LAYERS": "0",
            }
        )
    if condition == "promote_all":
        env.update(
            {
                "MARS_SELF_MODULE_MIN_EXACT": "0",
                "MARS_SELF_MODULE_MAX_LOSS": "1",
                "MARS_SELF_MODULE_MIN_KEYS": "1",
                "MARS_SELF_MODULE_TRUSTED_MIN_EXACT": "0",
                "MARS_SELF_MODULE_TRUSTED_MAX_LOSS": "1",
                "MARS_SELF_LAYER_MIN_GAIN": "0",
                "MARS_SELF_LAYER_MAX_LOSS": "1",
                "MARS_SELF_LAYER_MIN_KEYS": "1",
                "MARS_SELF_LAYER_TRUSTED_MIN_GAIN": "0",
                "MARS_SELF_LAYER_TRUSTED_MAX_LOSS": "1",
            }
        )
    return env


def _child_command(
    *,
    run_id: str,
    split: str,
    task_keys: list[str],
    model: str,
    judge_model: str,
    n_proposals: int,
    max_rounds: int,
    skip_official_eval: bool,
    test_workers: int,
) -> list[str]:
    # Frozen full-test tasks are conditionally independent: the language is
    # read-only, so chunking only changes wall time, not the protocol.
    if split == "test" and not task_keys and not skip_official_eval:
        return [
            sys.executable,
            "-m",
            _PARALLEL_TEST_RUNNER,
            "--run_id",
            run_id,
            "--data_split",
            "test",
            "--model",
            model,
            "--judge_model",
            judge_model,
            "--n_proposals",
            str(n_proposals),
            "--max_rounds",
            str(max_rounds),
            "--task_timeout_s",
            "600",
            "--workers",
            str(test_workers),
            "--overwrite",
        ]
    command = [
        sys.executable,
        "-m",
        _RUNNER,
        "--run_id",
        run_id,
        "--data_split",
        split,
        "--max_tasks",
        "0",
        "--model",
        model,
        "--judge_model",
        judge_model,
        "--n_proposals",
        str(n_proposals),
        "--max_rounds",
        str(max_rounds),
        "--discovery_modules",
        "full",
        "--task_timeout_s",
        "600",
        "--fail_on_eval_error",
        "--overwrite",
    ]
    if task_keys:
        command.extend(["--task_keys", *task_keys])
    if skip_official_eval:
        command.append("--skip_official_eval")
    return command


def _summary_path(run_id: str) -> Path:
    return _PROJ / "lmw" / "universal_discovery_real" / run_id / "summary.json"


def _command_specs(
    *,
    run_id: str,
    conditions: list[str],
    induction_waves: list[list[str]],
    promotion_keys: list[str],
    model: str,
    judge_model: str,
    n_proposals: int,
    max_rounds: int,
    checkpoint_every: int,
    test_workers: int,
    experiment_dir: Path,
) -> list[CommandSpec]:
    specs: list[CommandSpec] = []
    for condition in conditions:
        root = experiment_dir / "language" / condition
        if condition != "fixed_l0":
            for wave_index, wave_keys in enumerate(induction_waves, 1):
                child_id = f"{run_id}_{condition}_induce_w{wave_index:02d}"
                specs.append(
                    CommandSpec(
                        name=child_id,
                        condition=condition,
                        phase="induce",
                        command=_child_command(
                            run_id=child_id,
                            split="train",
                            task_keys=wave_keys,
                            model=model,
                            judge_model=judge_model,
                            n_proposals=n_proposals,
                            max_rounds=max_rounds,
                            skip_official_eval=True,
                            test_workers=test_workers,
                        ),
                        env=_language_env(root, condition, "induce"),
                        expected_summary=str(_summary_path(child_id)),
                    )
                )
                if (
                    checkpoint_every > 0
                    and wave_index % checkpoint_every == 0
                    and wave_index < len(induction_waves)
                ):
                    checkpoint_id = (
                        f"{run_id}_{condition}_dev_after_w{wave_index:02d}"
                    )
                    specs.append(
                        CommandSpec(
                            name=checkpoint_id,
                            condition=condition,
                            phase="dev_checkpoint",
                            command=_child_command(
                                run_id=checkpoint_id,
                                split="train",
                                task_keys=promotion_keys,
                                model=model,
                                judge_model=judge_model,
                            n_proposals=n_proposals,
                            max_rounds=max_rounds,
                            # Promotion is decided from executable task-local
                            # validation, never from a benchmark judge or gold
                            # answer.  The official judge is reserved for the
                            # final frozen test evaluation.
                            skip_official_eval=condition == "gated_growth",
                                test_workers=test_workers,
                            ),
                            env=_language_env(
                                root,
                                condition,
                                "promotion_probe" if condition == "gated_growth" else "frozen_eval",
                            ),
                            expected_summary=str(_summary_path(checkpoint_id)),
                        )
                    )
        dev_id = f"{run_id}_{condition}_dev"
        specs.append(
            CommandSpec(
                name=dev_id,
                condition=condition,
                phase="dev_eval",
                command=_child_command(
                    run_id=dev_id,
                    split="train",
                    task_keys=promotion_keys,
                    model=model,
                    judge_model=judge_model,
                    n_proposals=n_proposals,
                    max_rounds=max_rounds,
                    skip_official_eval=condition == "gated_growth",
                    test_workers=test_workers,
                ),
                env=_language_env(
                    root,
                    condition,
                    "promotion_probe" if condition == "gated_growth" else "frozen_eval",
                ),
                expected_summary=str(_summary_path(dev_id)),
            )
        )
        test_id = f"{run_id}_{condition}_test239"
        specs.append(
            CommandSpec(
                name=test_id,
                condition=condition,
                phase="test_eval",
                command=_child_command(
                    run_id=test_id,
                    split="test",
                    task_keys=[],
                    model=model,
                    judge_model=judge_model,
                    n_proposals=n_proposals,
                    max_rounds=max_rounds,
                    skip_official_eval=False,
                    test_workers=test_workers,
                ),
                env=_language_env(root, condition, "frozen_eval"),
                expected_summary=str(_summary_path(test_id)),
            )
        )
    return specs


def _source_complexity(path: Path) -> tuple[int, int]:
    try:
        source = path.read_text(encoding="utf-8")
        return len(source), sum(1 for _ in ast.walk(ast.parse(source)))
    except Exception:
        return 0, 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _promotion_ledger(language_root: Path, condition: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind, base, validator in (
        ("module", language_root / "modules", validate_self_module_source),
        ("layer", language_root / "layers", validate_self_layer_source),
    ):
        trusted_hashes: set[str] = set()
        for trusted_path in base.glob("*/trusted_manifest.jsonl"):
            trusted_hashes.update(
                str(row.get("source_hash", "")) for row in _read_jsonl(trusted_path)
            )
        for manifest in base.glob("*/manifest.jsonl"):
            for record in _read_jsonl(manifest):
                path = Path(str(record.get("path", "")))
                try:
                    code = path.read_text(encoding="utf-8")
                except Exception:
                    code = ""
                safe, safety_reason = validator(code)
                n_chars, n_ast_nodes = _source_complexity(path)
                score = dict(record.get("score") or {})
                rows.append(
                    {
                        "condition": condition,
                        "kind": kind,
                        "namespace": record.get("namespace"),
                        "name": record.get("name"),
                        "source_hash": record.get("source_hash"),
                        "validation_key": score.get("validation_key"),
                        "loss_mean": score.get("loss_mean"),
                        "exact_rate": score.get("exact_rate"),
                        "gain": score.get("gain"),
                        "complexity": score.get("complexity"),
                        "source_chars": n_chars,
                        "ast_nodes": n_ast_nodes,
                        "static_safe": safe,
                        "safety_reason": safety_reason,
                        "decision": (
                            "trusted"
                            if str(record.get("source_hash", "")) in trusted_hashes
                            else "candidate_only"
                        ),
                        "path": str(path),
                    }
                )
    return rows


def _write_ledger(experiment_dir: Path, conditions: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        rows.extend(
            _promotion_ledger(experiment_dir / "language" / condition, condition)
        )
    jsonl_path = experiment_dir / "promotion_ledger.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_path = experiment_dir / "promotion_ledger.csv"
    fields = list(rows[0]) if rows else ["condition", "kind", "decision"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _language_inventory(language_root: Path) -> dict[str, int]:
    inventory = {
        "module_candidates": 0,
        "module_trusted": 0,
        "layer_candidates": 0,
        "layer_trusted": 0,
    }
    for kind, base in (("module", language_root / "modules"), ("layer", language_root / "layers")):
        candidate_hashes: set[str] = set()
        trusted_hashes: set[str] = set()
        for path in base.glob("*/manifest.jsonl"):
            candidate_hashes.update(
                str(row.get("source_hash", "")) for row in _read_jsonl(path)
            )
        for path in base.glob("*/trusted_manifest.jsonl"):
            trusted_hashes.update(
                str(row.get("source_hash", "")) for row in _read_jsonl(path)
            )
        candidate_hashes.discard("")
        trusted_hashes.discard("")
        inventory[f"{kind}_candidates"] = len(candidate_hashes)
        inventory[f"{kind}_trusted"] = len(trusted_hashes)
    return inventory


def _write_reuse_ledger(experiment_dir: Path, specs: list[CommandSpec]) -> None:
    uses: dict[tuple[str, str, str, str], set[str]] = {}
    for spec in specs:
        if spec.phase != "test_eval":
            continue
        summary_path = Path(spec.expected_summary)
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        prediction_path = Path(str(summary.get("predictions_path", "")))
        for task_row in _read_jsonl(prediction_path):
            task_key = str(task_row.get("task_key", ""))
            for dataset_run in task_row.get("dataset_runs", []) or []:
                for winner in dataset_run.get("winner_details", []) or []:
                    tags = ",".join(str(tag) for tag in winner.get("tags", []) or [])
                    key = (
                        spec.condition,
                        str(winner.get("name", "")),
                        str(winner.get("source_hash", "")),
                        tags,
                    )
                    uses.setdefault(key, set()).add(task_key)
    rows = [
        {
            "condition": condition,
            "operator": name,
            "source_hash": source_hash,
            "tags": tags,
            "n_test_tasks_reused": len(task_keys),
            "test_task_keys": sorted(task_keys),
        }
        for (condition, name, source_hash, tags), task_keys in uses.items()
    ]
    rows.sort(key=lambda row: (-row["n_test_tasks_reused"], row["condition"], row["operator"]))
    (experiment_dir / "operator_reuse.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (experiment_dir / "operator_reuse.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ("condition", "operator", "source_hash", "tags", "n_test_tasks_reused")
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_spec(spec: CommandSpec, *, resume: bool) -> dict[str, Any]:
    summary_path = Path(spec.expected_summary)
    if resume and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if spec.phase == "test_eval" and summary.get("language_mutated"):
            return {
                "name": spec.name,
                "condition": spec.condition,
                "phase": spec.phase,
                "status": "failed",
                "reason": "resumed frozen evaluation mutated its language store",
                "summary": str(summary_path),
            }
        return {
            "name": spec.name,
            "condition": spec.condition,
            "phase": spec.phase,
            "status": "reused",
            "summary": str(summary_path),
        }
    env = os.environ.copy()
    env.update(spec.env)
    started = time.time()
    proc = subprocess.run(spec.command, cwd=_PROJ, env=env)
    result = {
        "name": spec.name,
        "condition": spec.condition,
        "phase": spec.phase,
        "status": "complete" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "wall_time_s": round(time.time() - started, 3),
        "summary": str(summary_path),
    }
    if proc.returncode == 0 and spec.phase == "test_eval":
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("language_mutated"):
            result["status"] = "failed"
            result["reason"] = "frozen language store mutated during evaluation"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=f"aaai27_language_growth_{time.strftime('%Y%m%d')}")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--judge_model", default="openai/gpt-4o")
    parser.add_argument("--conditions", default=",".join(_CONDITIONS))
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--induction_fraction", type=float, default=0.65)
    parser.add_argument("--n_proposals", type=int, default=4)
    parser.add_argument("--max_rounds", type=int, default=1)
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=3,
        help="Evaluate the held-out development groups every N induction waves; 0 disables checkpoints.",
    )
    parser.add_argument(
        "--test_workers",
        type=int,
        default=4,
        help="Parallel workers for immutable full-test evaluation only.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Run only induction and held-out promotion probes; do not touch the official test split.",
    )
    parser.add_argument("--overwrite_protocol", action="store_true")
    args = parser.parse_args()

    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    unknown = sorted(set(conditions) - set(_CONDITIONS))
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}")
    if not 0.0 < args.induction_fraction < 1.0:
        raise SystemExit("--induction_fraction must be between 0 and 1")
    if args.test_workers < 1:
        raise SystemExit("--test_workers must be positive")

    repo = _PROJ / "discoverybench_repo"
    train_tasks = load_official_train_tasks(
        repo, max_tasks=None, datasets=None, task_keys=None
    )
    test_tasks = load_official_real_tasks(
        repo, max_tasks=None, datasets=None, task_keys=None
    )
    induction, promotion = _stable_group_split(
        train_tasks,
        seed=args.seed,
        induction_fraction=args.induction_fraction,
    )
    if set(task.task_key for task in induction) & set(task.task_key for task in promotion):
        raise SystemExit("induction/promotion task overlap")
    if len(test_tasks) != 239:
        raise SystemExit(f"expected 239 official test tasks, found {len(test_tasks)}")

    experiment_dir = _PROJ / "lmw" / "aaai27_language_growth" / args.run_id
    protocol_path = experiment_dir / "protocol.json"
    if protocol_path.exists() and not (args.resume or args.overwrite_protocol):
        raise SystemExit(f"protocol exists; pass --resume or --overwrite_protocol: {protocol_path}")
    if args.overwrite_protocol and experiment_dir.exists() and not args.execute:
        shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    specs = _command_specs(
        run_id=args.run_id,
        conditions=conditions,
        induction_waves=[
            [task.task_key for task in induction if _group_key(task) == group]
            for group in sorted({_group_key(task) for task in induction})
        ],
        promotion_keys=[task.task_key for task in promotion],
        model=args.model,
        judge_model=args.judge_model,
        n_proposals=args.n_proposals,
        max_rounds=args.max_rounds,
        checkpoint_every=args.checkpoint_every,
        test_workers=args.test_workers,
        experiment_dir=experiment_dir,
    )
    if args.skip_test:
        specs = [spec for spec in specs if spec.phase != "test_eval"]
    protocol = {
        "protocol_version": "aaai27-language-growth-v1",
        "run_id": args.run_id,
        "created_at": time.time(),
        "model": args.model,
        "judge_model": args.judge_model,
        "seed": args.seed,
        "induction_fraction": args.induction_fraction,
        "n_proposals": args.n_proposals,
        "max_rounds": args.max_rounds,
        "checkpoint_every": args.checkpoint_every,
        "test_workers": args.test_workers,
        "test_deferred": args.skip_test,
        "conditions": conditions,
        "causal_variable": "cross-task hypothesis-language promotion and reuse",
        "fixed_factors": [
            "official task split",
            "per-task search budget",
            "evidence contracts and validators",
            "Discovery renderer stack",
            "generator and judge models",
        ],
        "leakage_guards": [
            "language roots start experiment-local",
            "only released train tasks can mutate the language",
            "promotion development groups do not overlap induction groups",
            "test evaluation loads trusted artifacts but disables all writes",
            "all 239 test tasks are evaluated exactly once per declared condition",
        ],
        "counts": {
            "train_total": len(train_tasks),
            "induction": len(induction),
            "promotion_dev": len(promotion),
            "official_test": len(test_tasks),
        },
        "induction_tasks": [asdict(row) for row in _task_rows(induction, "train_induction")],
        "promotion_tasks": [asdict(row) for row in _task_rows(promotion, "train_promotion")],
        "test_tasks": [asdict(row) for row in _task_rows(test_tasks, "test")],
        "source_hashes": {
            "universal_cpi": _sha256(_PROJ / "mars" / "induction" / "universal_cpi.py"),
            "discovery_runner": _sha256(_PROJ / "mars" / "runners" / "run_universal_discovery_real_eval.py"),
            "experiment_runner": _sha256(Path(__file__)),
        },
        "commands": [asdict(spec) for spec in specs],
    }
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    command_path = experiment_dir / "commands.sh"
    command_path.write_text(
        "\n\n".join(
            " ".join(f"{key}={shlex.quote(value)}" for key, value in spec.env.items())
            + " \\\n+  "
            + " ".join(shlex.quote(part) for part in spec.command)
            for spec in specs
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"protocol: {protocol_path}")
    print(
        f"train={len(train_tasks)} induction={len(induction)} "
        f"promotion_dev={len(promotion)} test={len(test_tasks)}"
    )
    print(f"planned child runs: {len(specs)}")
    if not args.execute:
        print("prepared only; pass --execute to start API-backed runs")
        return

    run_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    for spec in specs:
        print(f"\n=== {spec.condition}/{spec.phase}: {spec.name} ===", flush=True)
        result = _run_spec(spec, resume=args.resume)
        run_rows.append(result)
        (experiment_dir / "run_status.json").write_text(
            json.dumps(run_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if result.get("status") == "failed":
            raise SystemExit(f"child run failed: {spec.name}")
        if spec.phase == "induce":
            _write_ledger(experiment_dir, conditions)
            trajectory_rows.append(
                {
                    "condition": spec.condition,
                    "induction_step": sum(
                        1
                        for row in run_rows
                        if row.get("condition") == spec.condition
                        and row.get("phase") == "induce"
                    ),
                    **_language_inventory(
                        experiment_dir / "language" / spec.condition
                    ),
                }
            )
            (experiment_dir / "language_trajectory.json").write_text(
                json.dumps(trajectory_rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if spec.phase in {"dev_checkpoint", "dev_eval"}:
            summary = json.loads(Path(spec.expected_summary).read_text(encoding="utf-8"))
            performance_rows.append(
                {
                    "condition": spec.condition,
                    "phase": spec.phase,
                    "induction_step": sum(
                        1
                        for row in run_rows
                        if row.get("condition") == spec.condition
                        and row.get("phase") == "induce"
                    ),
                    "HMS_mean_100": summary.get("HMS_mean_100"),
                    "HMS_mean_consistency_100": summary.get(
                        "HMS_mean_consistency_100"
                    ),
                    **_language_inventory(
                        experiment_dir / "language" / spec.condition
                    ),
                    "summary": spec.expected_summary,
                }
            )
            (experiment_dir / "performance_trajectory.json").write_text(
                json.dumps(performance_rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    _write_ledger(experiment_dir, conditions)
    _write_reuse_ledger(experiment_dir, specs)
    print(f"completed: {experiment_dir}")


if __name__ == "__main__":
    main()
