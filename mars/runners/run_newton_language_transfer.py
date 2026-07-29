"""Strict development test for residual-guided language transfer on NewtonBench.

The script deliberately evaluates only observable held-out loss.  It is a
mechanism experiment, not an official-score run: source modules may create a
quarantined typed operator; the operator is then frozen and compared with the
same fixed language on disjoint modules.  The official NewtonBench evaluator is
reserved for a later, selected confirmation run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


_PROJECT = Path(__file__).resolve().parents[2]
_NEWTON = _PROJECT / "newtonbench_repo"
for _path in (_PROJECT, _NEWTON):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mars.induction.cpi_adapters import NewtonAdapter
from mars.induction.universal_cpi import UniversalCPI
from mars.runners.run_nb_official_dpsr import _params, collect_official
from mars.skills.self_layer_registry import SelfLayerRegistry
from mars.skills.typed_operator_plan import compile_typed_operator_plan


def _adapter(module_name: str, *, difficulty: str, law_version: str, system: str, seed: int) -> NewtonAdapter:
    module = importlib.import_module(f"modules.{module_name}")
    params = _params(str(module.FUNCTION_SIGNATURE).strip())
    rows = collect_official(
        module,
        params,
        difficulty=difficulty,
        law_version=law_version,
        system=system,
        n=24,
        seed=seed,
    )
    if len(rows) < 6:
        raise RuntimeError(f"{module_name}: too few observable rows")
    return NewtonAdapter(rows, var_names=params, target_name="y")


def _run_task(
    module_name: str,
    *,
    model: str,
    difficulty: str,
    law_version: str,
    system: str,
    seed: int,
    max_rounds: int,
) -> dict[str, Any]:
    result = UniversalCPI(
        model=model,
        n_proposals=2,
        max_rounds=max_rounds,
        holdout_frac=0.4,
        temperature=0.0,
    ).run(_adapter(
        module_name,
        difficulty=difficulty,
        law_version=law_version,
        system=system,
        seed=seed,
    ))
    if not result.winners:
        return {"module": module_name, "loss": 1.0, "winner": None, "tags": [], "n_valid": 0}
    winner, score = result.winners[0]
    return {
        "module": module_name,
        "loss": float(score.loss_mean),
        "winner": winner.name,
        "tags": list(winner.tags),
        "n_valid": result.n_valid,
        "candidate_won": "self_layer_program" in winner.tags,
        "residual_notes": [note for note in result.errors if "residual class" in note],
    }


def _mean(rows: list[dict[str, Any]]) -> float:
    return sum(float(row["loss"]) for row in rows) / len(rows) if rows else 1.0


def _provenance_audit(language_root: Path, source_modules: list[str]) -> dict[str, Any]:
    """Record that a persisted operator has no source-task identifier in code."""

    layer_dir = language_root / "layers" / "universal"
    files = sorted(path for path in layer_dir.glob("*.py") if path.name != "__init__.py")
    sources = [path.read_text(encoding="utf-8") for path in files]
    lowered = "\n".join(sources).lower()
    identifier_hits = [
        module for module in source_modules
        if module.lower() in lowered
    ]
    return {
        "operator_files": [path.name for path in files],
        "operator_source_hashes": [
            hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            for text in sources
        ],
        "source_task_identifier_hits": identifier_hits,
        "source_task_identifier_free": not identifier_hits,
    }


def _install_counterfactual_layer(language_root: Path, family: str) -> Path:
    """Persist an admissible, source-free operator for a paired control arm.

    The control is intentionally constructed from the same typed compiler as
    the induced layer.  It receives no task identifier, observation value,
    answer, or source-task text.  Only its operator family differs.
    """

    item = compile_typed_operator_plan(
        {"family": family, "description": f"typed counterfactual: {family}"},
        input_type="dict",
        target_type="float",
        signature_hint="def law(inputs: dict) -> float",
    )
    if item is None:
        raise RuntimeError(f"counterfactual family is not admissible: {family}")
    control_root = language_root / "counterfactual"
    layer_root = control_root / "layers"
    registry = SelfLayerRegistry(layer_root)
    record = registry.promote(
        namespace="universal",
        name=f"counterfactual_{family}",
        layer_type="typed_counterfactual_operator",
        code=str(item["code"]),
        contract=dict(item["contract"]),
        description=str(item["description"]),
        score={
            "gain": 0.0,
            "loss_mean": 1.0,
            "evidence_phase": "counterfactual_control",
        },
        min_gain=0.0,
        max_loss_mean=1.0,
    )
    if record is None:
        raise RuntimeError(f"could not persist counterfactual family: {family}")
    return layer_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=f"newton_language_transfer_{time.strftime('%Y%m%d')}")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--source_modules", default="m0_gravity,m1_coulomb_force")
    parser.add_argument("--heldout_modules", default="m2_magnetic_force,m3_fourier_law,m5_radioactive_decay,m9_hooke_law")
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--law_version", default="v0")
    parser.add_argument("--system", default="vanilla_equation")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--counterfactual_family",
        default="",
        help=(
            "Optional admissible typed family for a paired negative-control arm. "
            "The source phase still induces a layer; only the frozen held-out "
            "candidate is replaced by this source-free counterfactual."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = [item.strip() for item in args.source_modules.split(",") if item.strip()]
    heldout = [item.strip() for item in args.heldout_modules.split(",") if item.strip()]
    if not source or not heldout or set(source) & set(heldout):
        raise SystemExit("source and held-out module lists must be non-empty and disjoint")

    out_dir = _PROJECT / "lmw" / "language_transfer" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists: {out_path}")

    language_root = out_dir / "language"
    os.environ.update({
        "MARS_SELF_MODULE_ROOT": str(language_root / "modules"),
        "MARS_SELF_LAYER_ROOT": str(language_root / "layers"),
        "MARS_RESIDUAL_LEDGER_ROOT": str(language_root / "residual_classes"),
        "MARS_SELF_LAYER_NAMESPACE": "universal",
        "MARS_SELF_WRITE_MODULES": "1",
        "MARS_SELF_WRITE_LAYERS": "1",
        "MARS_LOAD_TRUSTED_MODULES": "0",
        "MARS_LOAD_TRUSTED_LAYERS": "0",
        "MARS_PROMOTE_SELF_MODULES": "1",
        "MARS_PROMOTE_SELF_LAYERS": "1",
        "MARS_QUARANTINE_PROMOTION": "1",
        "MARS_RESIDUAL_CLASS_INDUCTION": "1",
        "MARS_RESIDUAL_CLASS_ONLY_PROMOTION": "1",
        "MARS_RESIDUAL_CLASS_MIN_SUPPORT": "2",
        "MARS_PROPOSE_SELF_LAYERS": "0",
        "MARS_RESIDUAL_OPERATOR_LAYERS": "0",
        # L0 is intentionally sparse.  The induced layer is the only
        # language extension between the paired held-out conditions.
        "MARS_TYPED_PRIORS": "0",
        "MARS_RESIDUAL_KERNEL": "0",
        "MARS_DPSR": "0",
    })

    started = time.time()
    source_rows = []
    for module in source:
        row = _run_task(module, model=args.model, difficulty=args.difficulty,
                        law_version=args.law_version, system=args.system, seed=args.seed,
                        max_rounds=1)
        source_rows.append(row)
        print(f"source {module}: loss={row['loss']:.4f}", flush=True)

    source_language_audit = _provenance_audit(language_root, source)
    counterfactual_family = str(args.counterfactual_family or "").strip()
    if counterfactual_family:
        control_root = _install_counterfactual_layer(language_root, counterfactual_family)
        os.environ["MARS_SELF_LAYER_ROOT"] = str(control_root)

    # Both held-out arms are a deterministic executable comparison.  No new
    # free-form candidates, promotions, or language mutations are permitted.
    os.environ.update({
        "MARS_LOAD_CANDIDATE_LAYERS": "0",
        "MARS_PROMOTE_SELF_MODULES": "0",
        "MARS_PROMOTE_SELF_LAYERS": "0",
    })
    l0_rows = []
    for module in heldout:
        row = _run_task(module, model=args.model, difficulty=args.difficulty,
                        law_version=args.law_version, system=args.system, seed=args.seed,
                        max_rounds=0)
        l0_rows.append(row)
        print(f"L0 {module}: loss={row['loss']:.4f}", flush=True)

    os.environ.update({
        "MARS_LOAD_CANDIDATE_LAYERS": "1",
        "MARS_CANDIDATE_LAYER_BUDGET": "1",
    })
    l1_rows = []
    for module in heldout:
        row = _run_task(module, model=args.model, difficulty=args.difficulty,
                        law_version=args.law_version, system=args.system, seed=args.seed,
                        max_rounds=0)
        l1_rows.append(row)
        print(f"L1 {module}: loss={row['loss']:.4f} candidate={row['candidate_won']}", flush=True)

    paired = [
        {
            "module": left["module"],
            "l0_loss": left["loss"],
            "l1_loss": right["loss"],
            "relative_gain": float(left["loss"]) - float(right["loss"]),
            "l1_candidate_won": right["candidate_won"],
        }
        for left, right in zip(l0_rows, l1_rows)
    ]
    summary = {
        "protocol": "typed-residual-language-transfer-v1",
        "run_id": args.run_id,
        "model": args.model,
        "difficulty": args.difficulty,
        "law_version": args.law_version,
        "system": args.system,
        "seed": args.seed,
        "counterfactual_family": counterfactual_family or None,
        "source_modules": source,
        "heldout_modules": heldout,
        "source": source_rows,
        "l0": l0_rows,
        "l1": l1_rows,
        "paired": paired,
        "mean_l0_loss": _mean(l0_rows),
        "mean_l1_loss": _mean(l1_rows),
        "mean_relative_gain": sum(row["relative_gain"] for row in paired) / len(paired),
        "candidate_wins": sum(int(row["l1_candidate_won"]) for row in paired),
        "source_language_audit": source_language_audit,
        "language_audit": _provenance_audit(
            (language_root / "counterfactual") if counterfactual_family else language_root,
            source,
        ),
        "official_score_used": False,
        "wall_time_s": round(time.time() - started, 2),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "mean_l0_loss": summary["mean_l0_loss"],
        "mean_l1_loss": summary["mean_l1_loss"],
        "mean_relative_gain": summary["mean_relative_gain"],
        "candidate_wins": summary["candidate_wins"],
    }), flush=True)


if __name__ == "__main__":
    main()
