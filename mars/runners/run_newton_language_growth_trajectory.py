"""Audited multi-step hypothesis-language growth on NewtonBench interfaces.

The experiment uses only observable input-output rows during induction.  It
partitions modules by typed residual geometry, induces one operator from two
source modules, promotes it on disjoint modules/seeds, freezes the resulting
language, and measures transfer on a third disjoint partition.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from mars.induction.universal_cpi import UniversalCPI
from mars.runners.run_newton_language_transfer import _adapter, _run_task
from mars.skills.self_layer_registry import SelfLayerRecord, SelfLayerRegistry


_PROJECT = Path(__file__).resolve().parents[2]
_MODULES = (
    "m0_gravity",
    "m1_coulomb_force",
    "m2_magnetic_force",
    "m3_fourier_law",
    "m4_snell_law",
    "m5_radioactive_decay",
    "m6_underdamped_harmonic",
    "m7_malus_law",
    "m8_sound_speed",
    "m9_hooke_law",
    "m10_be_distribution",
    "m11_heat_transfer",
)


def _module_index(name: str) -> int:
    match = re.match(r"m(\d+)_", name)
    return int(match.group(1)) if match else 10_000


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _common_env(language_root: Path) -> dict[str, str]:
    return {
        "MARS_SELF_MODULE_ROOT": str(language_root / "modules"),
        "MARS_SELF_LAYER_ROOT": str(language_root / "layers"),
        "MARS_RESIDUAL_LEDGER_ROOT": str(language_root / "residual_classes"),
        "MARS_SELF_LAYER_NAMESPACE": "universal",
        "MARS_SELF_WRITE_MODULES": "1",
        "MARS_SELF_WRITE_LAYERS": "1",
        "MARS_LOAD_TRUSTED_MODULES": "0",
        "MARS_LOAD_TRUSTED_LAYERS": "1",
        "MARS_LOAD_CANDIDATE_MODULES": "0",
        "MARS_LOAD_CANDIDATE_LAYERS": "0",
        "MARS_PROMOTE_SELF_MODULES": "0",
        "MARS_PROMOTE_SELF_LAYERS": "0",
        "MARS_PROPOSE_SELF_LAYERS": "0",
        "MARS_RESIDUAL_OPERATOR_LAYERS": "0",
        "MARS_TYPED_PRIORS": "0",
        "MARS_RESIDUAL_KERNEL": "0",
        "MARS_DPSR": "0",
        "MARS_QUARANTINE_PROMOTION": "1",
        "MARS_RESIDUAL_CLASS_INDUCTION": "1",
        "MARS_RESIDUAL_CLASS_ONLY_PROMOTION": "1",
        "MARS_RESIDUAL_CLASS_MIN_SUPPORT": "2",
    }


def _profile_modules(
    *,
    difficulty: str,
    law_version: str,
    system: str,
    seed: int,
) -> tuple[dict[str, str], dict[str, str]]:
    geometries: dict[str, str] = {}
    skipped: dict[str, str] = {}
    engine = UniversalCPI(
        model="openai/gpt-4o-mini",
        n_proposals=1,
        max_rounds=0,
        temperature=0.0,
    )
    for module in _MODULES:
        try:
            adapter = _adapter(
                module,
                difficulty=difficulty,
                law_version=law_version,
                system=system,
                seed=seed,
            )
            observations = adapter.collect_observations()
            fingerprint = engine._residual_fingerprint(
                adapter, observations, None
            )
            geometries[module] = str(fingerprint["residual_geometry"])
        except Exception as exc:
            skipped[module] = f"{type(exc).__name__}: {exc}"
    return geometries, skipped


def _partitions(geometries: dict[str, str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for module, geometry in geometries.items():
        grouped.setdefault(geometry, []).append(module)
    for modules in grouped.values():
        modules.sort(key=_module_index)

    plans = []
    for geometry in ("positive_log_structured", "positive_nonlog_structured"):
        modules = grouped.get(geometry, [])
        if len(modules) < 4:
            continue
        promotion_count = 2 if len(modules) >= 5 else 1
        plans.append(
            {
                "geometry": geometry,
                "source": modules[:2],
                "promotion": modules[2 : 2 + promotion_count],
                "transfer": modules[2 + promotion_count :],
            }
        )
    return plans


def _candidate_records(registry: SelfLayerRegistry) -> dict[str, SelfLayerRecord]:
    records: dict[str, SelfLayerRecord] = {}
    for record in registry.load("universal"):
        if record.layer_type != "residual_class_operator":
            continue
        records.setdefault(record.source_hash, record)
    return records


def _evaluate(
    modules: list[str],
    *,
    model: str,
    difficulty: str,
    law_version: str,
    system: str,
    seed: int,
    load_candidates: bool,
) -> list[dict[str, Any]]:
    os.environ["MARS_LOAD_CANDIDATE_LAYERS"] = "1" if load_candidates else "0"
    rows = []
    for module in modules:
        row = _run_task(
            module,
            model=model,
            difficulty=difficulty,
            law_version=law_version,
            system=system,
            seed=seed,
            max_rounds=0,
        )
        rows.append(row)
    return rows


def _mean_loss(rows: list[dict[str, Any]]) -> float:
    return (
        sum(float(row["loss"]) for row in rows) / len(rows)
        if rows
        else 1.0
    )


def _source_complexity(path: Path) -> dict[str, float]:
    source = path.read_text(encoding="utf-8")
    nodes = sum(1 for _ in ast.walk(ast.parse(source)))
    return {
        "source_chars": len(source),
        "ast_nodes": nodes,
        "K": math.log1p(nodes),
    }


def _record_probe_evidence(
    registry: SelfLayerRegistry,
    candidate: SelfLayerRecord,
    baseline: list[dict[str, Any]],
    augmented: list[dict[str, Any]],
    *,
    geometry: str,
    seed: int,
) -> None:
    source = Path(candidate.path).read_text(encoding="utf-8")
    for left, right in zip(baseline, augmented):
        registry.promote(
            namespace="universal",
            name=candidate.name,
            layer_type=candidate.layer_type,
            code=source,
            contract=candidate.contract,
            description=candidate.description,
            score={
                "gain": max(0.0, 1.0 - float(right["loss"])),
                "loss_mean": float(right["loss"]),
                "validation_key": (
                    f"promotion:{geometry}:{right['module']}:seed{seed}"
                ),
                "relative_gain": (
                    float(left["loss"]) - float(right["loss"])
                ),
                "evidence_phase": "promotion_probe",
            },
            min_gain=0.0,
            max_loss_mean=1.0,
        )


def _trusted_hashes(registry: SelfLayerRegistry) -> set[str]:
    return {
        record.source_hash
        for record in registry.load_trusted("universal")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id",
        default=f"newton_language_growth_{time.strftime('%Y%m%d')}",
    )
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--law_version", default="v0")
    parser.add_argument("--system", default="vanilla_equation")
    parser.add_argument("--source_seed", type=int, default=13)
    parser.add_argument("--promotion_seed", type=int, default=31)
    parser.add_argument("--transfer_seed", type=int, default=57)
    parser.add_argument("--redundancy_seed", type=int, default=73)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--min_net_gain", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = _PROJECT / "lmw" / "language_growth_newton" / args.run_id
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        raise SystemExit(f"exists: {summary_path}")

    language_root = out_dir / "language"
    os.environ.update(_common_env(language_root))
    registry = SelfLayerRegistry(language_root / "layers")

    geometries, skipped = _profile_modules(
        difficulty=args.difficulty,
        law_version=args.law_version,
        system=args.system,
        seed=args.source_seed,
    )
    plans = _partitions(geometries)
    protocol = {
        "protocol": "residual-guided-language-trajectory-v1",
        "run_id": args.run_id,
        "model": args.model,
        "difficulty": args.difficulty,
        "law_version": args.law_version,
        "system": args.system,
        "source_seed": args.source_seed,
        "promotion_seed": args.promotion_seed,
        "transfer_seed": args.transfer_seed,
        "redundancy_seed": args.redundancy_seed,
        "beta": args.beta,
        "min_net_gain": args.min_net_gain,
        "module_geometry": geometries,
        "skipped_modules": skipped,
        "waves": plans,
        "selection_rule": (
            "accept iff mean held-out loss reduction - beta*log(1+AST nodes) "
            "> min_net_gain"
        ),
        "leakage_guards": [
            "partitions are generated from residual geometry and module index",
            "source, promotion, and transfer module sets are disjoint per wave",
            "source, promotion, and transfer use distinct random seeds",
            "operator code receives no module identifier or gold law",
            "transfer tasks cannot mutate the language",
        ],
    }
    (out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    started = time.time()
    trajectory = [
        {
            "language": "L0",
            "trusted_operators": 0,
            "trusted_hashes": [],
        }
    ]
    decisions = []

    for wave_index, plan in enumerate(plans, start=1):
        before_candidates = set(_candidate_records(registry))
        os.environ.update(
            {
                "MARS_LOAD_CANDIDATE_LAYERS": "0",
                "MARS_PROMOTE_SELF_LAYERS": "1",
            }
        )
        source_rows = []
        for module in plan["source"]:
            row = _run_task(
                module,
                model=args.model,
                difficulty=args.difficulty,
                law_version=args.law_version,
                system=args.system,
                seed=args.source_seed,
                max_rounds=1,
            )
            source_rows.append(row)
            print(
                f"wave {wave_index} source {module}: loss={row['loss']:.4f}",
                flush=True,
            )

        after_records = _candidate_records(registry)
        new_hashes = sorted(set(after_records) - before_candidates)
        if not new_hashes:
            decisions.append(
                {
                    "wave": wave_index,
                    "geometry": plan["geometry"],
                    "decision": "rejected",
                    "reason": "no executable residual-class operator was induced",
                    "source": source_rows,
                }
            )
            continue
        candidate = after_records[new_hashes[0]]

        os.environ["MARS_PROMOTE_SELF_LAYERS"] = "0"
        promotion_l0 = _evaluate(
            plan["promotion"],
            model=args.model,
            difficulty=args.difficulty,
            law_version=args.law_version,
            system=args.system,
            seed=args.promotion_seed,
            load_candidates=False,
        )
        promotion_plus = _evaluate(
            plan["promotion"],
            model=args.model,
            difficulty=args.difficulty,
            law_version=args.law_version,
            system=args.system,
            seed=args.promotion_seed,
            load_candidates=True,
        )
        complexity = _source_complexity(Path(candidate.path))
        promotion_gain = _mean_loss(promotion_l0) - _mean_loss(promotion_plus)
        net_gain = promotion_gain - args.beta * complexity["K"]
        accepted = net_gain > args.min_net_gain

        transfer_before = _evaluate(
            plan["transfer"],
            model=args.model,
            difficulty=args.difficulty,
            law_version=args.law_version,
            system=args.system,
            seed=args.transfer_seed,
            load_candidates=False,
        )
        _record_probe_evidence(
            registry,
            candidate,
            promotion_l0,
            promotion_plus,
            geometry=plan["geometry"],
            seed=args.promotion_seed,
        )
        if accepted:
            registry.refresh_trusted_manifest(
                "universal",
                min_validation_keys=2,
                min_mean_gain=0.0,
                max_mean_loss=1.0,
                min_probe_validation_keys=len(plan["promotion"]),
                min_mean_probe_relative_gain=args.min_net_gain,
            )
        transfer_after = _evaluate(
            plan["transfer"],
            model=args.model,
            difficulty=args.difficulty,
            law_version=args.law_version,
            system=args.system,
            seed=args.transfer_seed,
            load_candidates=False,
        )
        transfer_gain = _mean_loss(transfer_before) - _mean_loss(transfer_after)
        decision = {
            "wave": wave_index,
            "from_language": f"L{wave_index - 1}",
            "to_language": f"L{wave_index}" if accepted else f"L{wave_index - 1}",
            "geometry": plan["geometry"],
            "operator": candidate.name,
            "source_hash": candidate.source_hash,
            "source": source_rows,
            "promotion_modules": plan["promotion"],
            "promotion_baseline": promotion_l0,
            "promotion_augmented": promotion_plus,
            "promotion_gain": promotion_gain,
            "complexity": complexity,
            "complexity_penalty": args.beta * complexity["K"],
            "net_gain": net_gain,
            "decision": "accepted" if accepted else "rejected",
            "transfer_modules": plan["transfer"],
            "transfer_before": transfer_before,
            "transfer_after": transfer_after,
            "transfer_gain": transfer_gain,
        }
        decisions.append(decision)
        print(
            f"wave {wave_index} {decision['decision']}: "
            f"promotion_gain={promotion_gain:.4f} "
            f"net={net_gain:.4f} transfer_gain={transfer_gain:.4f}",
            flush=True,
        )
        trajectory.append(
            {
                "language": f"L{wave_index}",
                "trusted_operators": len(_trusted_hashes(registry)),
                "trusted_hashes": sorted(_trusted_hashes(registry)),
                "promotion_gain": promotion_gain,
                "transfer_gain": transfer_gain,
            }
        )

    accepted_decisions = [
        row for row in decisions if row.get("decision") == "accepted"
    ]
    if accepted_decisions:
        first = accepted_decisions[0]
        trusted_by_hash = {
            record.source_hash: record
            for record in registry.load_trusted("universal")
        }
        repeated = trusted_by_hash.get(str(first["source_hash"]))
        if repeated is not None:
            challenge_modules = list(first["promotion_modules"])
            baseline = _evaluate(
                challenge_modules,
                model=args.model,
                difficulty=args.difficulty,
                law_version=args.law_version,
                system=args.system,
                seed=args.redundancy_seed,
                load_candidates=False,
            )
            augmented = _evaluate(
                challenge_modules,
                model=args.model,
                difficulty=args.difficulty,
                law_version=args.law_version,
                system=args.system,
                seed=args.redundancy_seed,
                load_candidates=True,
            )
            complexity = _source_complexity(Path(repeated.path))
            promotion_gain = _mean_loss(baseline) - _mean_loss(augmented)
            net_gain = promotion_gain - args.beta * complexity["K"]
            decisions.append(
                {
                    "wave": len(plans) + 1,
                    "from_language": f"L{len(accepted_decisions)}",
                    "to_language": f"L{len(accepted_decisions)}",
                    "geometry": first["geometry"],
                    "operator": repeated.name,
                    "source_hash": repeated.source_hash,
                    "proposal_origin": (
                        "repeat occurrence of an already covered residual class"
                    ),
                    "deduplicated_by_source_hash": True,
                    "promotion_modules": challenge_modules,
                    "promotion_baseline": baseline,
                    "promotion_augmented": augmented,
                    "promotion_gain": promotion_gain,
                    "complexity": complexity,
                    "complexity_penalty": args.beta * complexity["K"],
                    "net_gain": net_gain,
                    "decision": "rejected",
                    "reason": (
                        "semantic duplicate has no marginal held-out value"
                    ),
                    "transfer_modules": [],
                    "transfer_before": [],
                    "transfer_after": [],
                    "transfer_gain": 0.0,
                }
            )
            trajectory.append(
                {
                    "language": f"L{len(accepted_decisions)}",
                    "trusted_operators": len(_trusted_hashes(registry)),
                    "trusted_hashes": sorted(_trusted_hashes(registry)),
                    "promotion_gain": promotion_gain,
                    "transfer_gain": 0.0,
                    "decision": "rejected_redundant_operator",
                }
            )
            print(
                f"wave {len(plans) + 1} rejected: "
                f"redundant operator net={net_gain:.4f}",
                flush=True,
            )

    trusted = registry.load_trusted("universal")
    operator_sources = []
    for record in trusted:
        source = Path(record.path).read_text(encoding="utf-8")
        operator_sources.append(
            {
                "name": record.name,
                "source_hash": record.source_hash,
                "sha256": hashlib.sha256(
                    source.encode("utf-8")
                ).hexdigest(),
                "source_task_identifier_hits": [
                    module
                    for module in geometries
                    if module.lower() in source.lower()
                ],
            }
        )
    summary = {
        **protocol,
        "decisions": decisions,
        "trajectory": trajectory,
        "accepted": sum(row.get("decision") == "accepted" for row in decisions),
        "rejected": sum(row.get("decision") == "rejected" for row in decisions),
        "trusted_operator_sources": operator_sources,
        "wall_time_s": round(time.time() - started, 3),
        "official_score_used": False,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (out_dir / "trajectory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "language",
                "trusted_operators",
                "decision",
                "promotion_gain",
                "transfer_gain",
            ],
        )
        writer.writeheader()
        for step, row in enumerate(trajectory):
            writer.writerow(
                {
                    "step": step,
                    "language": row["language"],
                    "trusted_operators": row["trusted_operators"],
                    "decision": row.get("decision", "initial_or_accepted"),
                    "promotion_gain": row.get("promotion_gain", ""),
                    "transfer_gain": row.get("transfer_gain", ""),
                }
            )
    with (out_dir / "operator_decisions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "wave",
                "from_language",
                "to_language",
                "geometry",
                "operator",
                "promotion_gain",
                "complexity_penalty",
                "net_gain",
                "decision",
                "transfer_gain",
                "reason",
            ],
        )
        writer.writeheader()
        for row in decisions:
            writer.writerow(
                {
                    "wave": row.get("wave"),
                    "from_language": row.get("from_language"),
                    "to_language": row.get("to_language"),
                    "geometry": row.get("geometry"),
                    "operator": row.get("operator"),
                    "promotion_gain": row.get("promotion_gain"),
                    "complexity_penalty": row.get("complexity_penalty"),
                    "net_gain": row.get("net_gain"),
                    "decision": row.get("decision"),
                    "transfer_gain": row.get("transfer_gain"),
                    "reason": row.get("reason", ""),
                }
            )
    print(f"summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
