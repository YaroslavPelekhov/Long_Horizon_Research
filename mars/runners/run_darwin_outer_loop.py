"""Evolve MARS's regulatory hypothesis-generator genome from CPI traces.

Example:
  python -m mars.runners.run_darwin_outer_loop \\
      --input_run_id regulatory_suite_20260702 \\
      --output_dir lmw/darwin_outer/regulatory_g1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from mars.darwin import (  # noqa: E402
    GeneratorTrace,
    RegulatoryGenome,
    default_regulatory_genome,
    evolve_regulatory_genome,
)


def _load_genome(path: str) -> RegulatoryGenome:
    if path:
        return RegulatoryGenome.load_json(path)
    return default_regulatory_genome()


def _trace_rows(cpi_log: dict[str, Any], genome: RegulatoryGenome | None = None) -> list[GeneratorTrace]:
    rows: list[GeneratorTrace] = []
    gene_by_name = {gene.name: gene for gene in genome.genes} if genome is not None else {}
    for i, result in enumerate(cpi_log.get("results", []) or []):
        if not isinstance(result, dict):
            continue
        benchmark = str(result.get("benchmark", "unknown"))
        supermetrics = result.get("supermetrics") or {}
        try:
            fitness = float(supermetrics.get("universal_score", 0.0) or 0.0)
        except Exception:
            fitness = 0.0
        for proposal in result.get("proposals_raw", []) or []:
            if not isinstance(proposal, dict) or proposal.get("kind") != "darwin_trace":
                continue
            for trace in proposal.get("trace", []) or []:
                if not isinstance(trace, dict):
                    continue
                dev = trace.get("regulatory_development") or {}
                active_rows = dev.get("active_genes") or []
                active = tuple(
                    str(row.get("gene"))
                    for row in active_rows
                    if isinstance(row, dict) and row.get("gene")
                )
                emitted = tuple(str(x) for x in dev.get("emitted_families", []) or [])
                prompt_scaffolds = tuple(
                    str(x) for x in dev.get("emitted_prompt_scaffolds", []) or []
                )
                code_probes = tuple(str(x) for x in dev.get("emitted_code_probes", []) or [])
                contract_validators = tuple(
                    str(x) for x in dev.get("emitted_contract_validators", []) or []
                )
                if active and not prompt_scaffolds and not code_probes and not contract_validators:
                    prompt_scaffolds, code_probes, contract_validators = _ip_expression_from_active(
                        active,
                        gene_by_name,
                    )
                if not active and not emitted:
                    continue
                rows.append(
                    GeneratorTrace(
                        task_id=f"{benchmark}:{i}",
                        active_genes=active,
                        emitted_families=emitted,
                        fitness=fitness,
                        emitted_prompt_scaffolds=prompt_scaffolds,
                        emitted_code_probes=code_probes,
                        emitted_contract_validators=contract_validators,
                        metadata={
                            "benchmark": benchmark,
                            "best": (result.get("winners") or [{}])[0].get("name")
                            if result.get("winners")
                            else "",
                            "n_valid": result.get("n_valid"),
                            "supermetrics": supermetrics,
                        },
                    )
                )
    return rows


def _ip_expression_from_active(
    active: tuple[str, ...],
    gene_by_name: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    prompt_scaffolds: list[str] = []
    code_probes: list[str] = []
    contract_validators: list[str] = []
    for name in active:
        gene = gene_by_name.get(str(name))
        if gene is None:
            continue
        prompt_scaffolds.extend(gene.emit_prompt_scaffolds)
        code_probes.extend(gene.emit_code_probes)
        contract_validators.extend(gene.emit_contract_validators)
    return (
        tuple(dict.fromkeys(prompt_scaffolds)),
        tuple(dict.fromkeys(code_probes)),
        tuple(dict.fromkeys(contract_validators)),
    )


def _external_deltas(baseline_summary: str, current_summary: str) -> dict[str, float]:
    if not baseline_summary or not current_summary:
        return {}
    base_path = Path(baseline_summary)
    curr_path = Path(current_summary)
    if not base_path.exists() or not curr_path.exists():
        return {}
    base = json.loads(base_path.read_text(encoding="utf-8"))
    curr = json.loads(curr_path.read_text(encoding="utf-8"))
    base_rows = {
        str(row.get("benchmark")): float(row.get("normalized", 0.0) or 0.0)
        for row in base.get("results", []) or []
        if isinstance(row, dict)
    }
    deltas: dict[str, float] = {}
    for row in curr.get("results", []) or []:
        if not isinstance(row, dict):
            continue
        bench = str(row.get("benchmark"))
        if bench not in base_rows:
            continue
        deltas[bench] = float(row.get("normalized", 0.0) or 0.0) - base_rows[bench]
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_run_id", default="")
    parser.add_argument("--input_cpi_log", default="")
    parser.add_argument("--input_genome", default="")
    parser.add_argument("--output_dir", default="lmw/darwin_outer/latest")
    parser.add_argument("--success_threshold", type=float, default=0.65)
    parser.add_argument("--failure_threshold", type=float, default=0.35)
    parser.add_argument("--relative_margin", type=float, default=0.08)
    parser.add_argument("--baseline_summary", default="")
    parser.add_argument("--current_summary", default="")
    args = parser.parse_args()

    if args.input_cpi_log:
        cpi_path = Path(args.input_cpi_log)
    elif args.input_run_id:
        cpi_path = _PROJ / "lmw" / "universal_cpi" / args.input_run_id / "cpi_results.json"
    else:
        raise SystemExit("pass --input_run_id or --input_cpi_log")
    if not cpi_path.exists():
        raise SystemExit(f"missing CPI log: {cpi_path}")

    genome = _load_genome(args.input_genome)
    cpi_log = json.loads(cpi_path.read_text(encoding="utf-8"))
    traces = _trace_rows(cpi_log, genome)
    deltas = _external_deltas(args.baseline_summary, args.current_summary)
    evolved = evolve_regulatory_genome(
        genome,
        traces,
        success_threshold=args.success_threshold,
        failure_threshold=args.failure_threshold,
        relative_margin=args.relative_margin,
        external_deltas=deltas,
    )

    out_dir = (_PROJ / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    genome_path = out_dir / "regulatory_genome.json"
    trace_path = out_dir / "outer_loop_trace.json"
    evolved.genome.save_json(genome_path)
    trace_path.write_text(
        json.dumps(
            {
                "input_cpi_log": str(cpi_path),
                "input_genome": args.input_genome or "default",
                "external_deltas": deltas,
                "n_generator_traces": len(traces),
                "traces": [
                    {
                        "task_id": t.task_id,
                        "active_genes": list(t.active_genes),
                        "emitted_families": list(t.emitted_families),
                        "emitted_prompt_scaffolds": list(t.emitted_prompt_scaffolds),
                        "emitted_code_probes": list(t.emitted_code_probes),
                        "emitted_contract_validators": list(t.emitted_contract_validators),
                        "fitness": t.fitness,
                        "metadata": dict(t.metadata or {}),
                    }
                    for t in traces
                ],
                "outer_loop": evolved.trace,
                "promoted": list(evolved.promoted),
                "demoted": list(evolved.demoted),
                "macro_genes": list(evolved.macro_genes),
                "genome_path": str(genome_path),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"loaded traces: {len(traces)}")
    print(f"promoted: {', '.join(evolved.promoted) or '-'}")
    print(f"demoted: {', '.join(evolved.demoted) or '-'}")
    print(f"macro_genes: {', '.join(evolved.macro_genes) or '-'}")
    print(f"genome → {genome_path}")
    print(f"trace → {trace_path}")


if __name__ == "__main__":
    main()
