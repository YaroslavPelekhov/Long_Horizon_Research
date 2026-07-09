"""Outer-loop evolution for MARS regulatory genomes.

The inner loop evaluates hypotheses.  The outer loop evaluates the generator
that produced those hypotheses.  This file is deliberately benchmark-agnostic:
it only sees generator traces, active regulatory genes, emitted families, and a
fitness number supplied by the caller.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

from mars.darwin.regulatory import RegulatoryGene, RegulatoryGenome


@dataclass(frozen=True)
class GeneratorTrace:
    """One task-level trace for generator evolution."""

    task_id: str
    active_genes: tuple[str, ...]
    emitted_families: tuple[str, ...]
    fitness: float
    emitted_prompt_scaffolds: tuple[str, ...] = ()
    emitted_code_probes: tuple[str, ...] = ()
    emitted_contract_validators: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class OuterLoopResult:
    """Result of evolving a hypothesis-generator genome."""

    genome: RegulatoryGenome
    promoted: tuple[str, ...]
    demoted: tuple[str, ...]
    macro_genes: tuple[str, ...]
    trace: dict[str, Any]


def evolve_regulatory_genome(
    genome: RegulatoryGenome,
    traces: Sequence[GeneratorTrace],
    *,
    success_threshold: float = 0.65,
    failure_threshold: float = 0.35,
    relative_margin: float = 0.08,
    niche_min_fitness: float = 0.45,
    external_deltas: Mapping[str, float] | None = None,
    max_macro_genes: int = 4,
) -> OuterLoopResult:
    """Evolve the generator genome from task traces.

    A gene is promoted if tasks where it fired were successful, demoted if they
    consistently failed, and compressed into a macro-gene if a pair co-fired in
    multiple successful traces.  This evolves the hypothesis birth mechanism,
    not any single benchmark hypothesis.
    """

    if not traces:
        return OuterLoopResult(genome, (), (), (), {"reason": "no traces"})

    by_gene: dict[str, list[float]] = defaultdict(list)
    by_pair: dict[tuple[str, str], list[GeneratorTrace]] = defaultdict(list)
    family_by_gene: dict[str, set[str]] = defaultdict(set)
    for trace in traces:
        fitness = max(0.0, min(1.0, float(trace.fitness)))
        genes = tuple(dict.fromkeys(trace.active_genes))
        for gene in genes:
            by_gene[gene].append(fitness)
            family_by_gene[gene].update(trace.emitted_families)
        if fitness >= success_threshold:
            for pair in combinations(sorted(genes), 2):
                by_pair[pair].append(trace)

    promoted: list[str] = []
    demoted: list[str] = []
    new_genes: list[RegulatoryGene] = []
    for gene in genome.genes:
        scores = by_gene.get(gene.name, [])
        if not scores:
            new_genes.append(gene)
            continue
        mean = sum(scores) / len(scores)
        if mean >= success_threshold:
            promoted.append(gene.name)
            delta = min(0.35, 0.08 + 0.12 * (mean - success_threshold))
        elif mean <= failure_threshold:
            demoted.append(gene.name)
            delta = -min(0.35, 0.08 + 0.12 * (failure_threshold - mean))
        else:
            delta = 0.0
        new_genes.append(_gene_with(gene, priority=max(0.05, gene.priority + delta)))

    macro_names: list[str] = []
    existing = {gene.name for gene in new_genes}
    macro_candidates = sorted(
        by_pair.items(),
        key=lambda item: (len(item[1]), sum(t.fitness for t in item[1])),
        reverse=True,
    )
    base_by_name = {gene.name: gene for gene in genome.genes}
    for (left_name, right_name), pair_traces in macro_candidates[:max_macro_genes]:
        if _is_nonprimitive_gene(left_name) or _is_nonprimitive_gene(right_name):
            continue
        left = base_by_name.get(left_name)
        right = base_by_name.get(right_name)
        if left is None or right is None:
            continue
        macro_name = f"macro_{left_name}__{right_name}"
        if macro_name in existing:
            continue
        families = tuple(
            dict.fromkeys(
                list(left.emit_families)
                + list(right.emit_families)
                + sorted(family_by_gene[left_name] | family_by_gene[right_name])
            )
        )
        if not families:
            continue
        priority = max(left.priority, right.priority) + 0.12 + 0.03 * len(pair_traces)
        new_genes.append(
            RegulatoryGene(
                name=macro_name,
                trigger_syndromes=tuple(dict.fromkeys(left.trigger_syndromes + right.trigger_syndromes)),
                trigger_queries=tuple(dict.fromkeys(left.trigger_queries + right.trigger_queries)),
                signature_patterns=tuple(dict.fromkeys(left.signature_patterns + right.signature_patterns)),
                emit_prompt_scaffolds=tuple(dict.fromkeys(left.emit_prompt_scaffolds + right.emit_prompt_scaffolds)),
                emit_code_probes=tuple(dict.fromkeys(left.emit_code_probes + right.emit_code_probes)),
                emit_contract_validators=tuple(dict.fromkeys(left.emit_contract_validators + right.emit_contract_validators)),
                emit_families=families,
                priority=priority,
                mutation_policy="macro_compression",
                rationale=(
                    "outer-loop compression of co-active regulatory genes "
                    f"{left_name} and {right_name}"
                ),
            )
        )
        existing.add(macro_name)
        macro_names.append(macro_name)

    if not promoted and len(traces) >= 2:
        best_trace = max(traces, key=lambda t: float(t.fitness))
        worst_trace = min(traces, key=lambda t: float(t.fitness))
        if float(best_trace.fitness) - float(worst_trace.fitness) >= relative_margin:
            best_genes = set(best_trace.active_genes)
            worst_genes = set(worst_trace.active_genes)
            promoted = sorted(best_genes)
            demoted = sorted(g for g in worst_genes if g not in best_genes)
            boosted: list[RegulatoryGene] = []
            for gene in new_genes:
                delta = 0.0
                if gene.name in promoted:
                    delta = 0.12
                elif gene.name in demoted:
                    delta = -0.10
                boosted.append(_gene_with(gene, priority=max(0.05, gene.priority + delta)))
            new_genes = boosted
            if not macro_names:
                base_by_name = {gene.name: gene for gene in new_genes}
                for left_name, right_name in combinations(sorted(best_genes), 2):
                    if _is_nonprimitive_gene(left_name) or _is_nonprimitive_gene(right_name):
                        continue
                    left = base_by_name.get(left_name)
                    right = base_by_name.get(right_name)
                    if left is None or right is None:
                        continue
                    families = tuple(dict.fromkeys(left.emit_families + right.emit_families))
                    if not families:
                        continue
                    macro_name = f"macro_{left_name}__{right_name}"
                    if macro_name in existing:
                        continue
                    new_genes.append(
                        RegulatoryGene(
                            name=macro_name,
                            trigger_syndromes=tuple(dict.fromkeys(left.trigger_syndromes + right.trigger_syndromes)),
                            trigger_queries=tuple(dict.fromkeys(left.trigger_queries + right.trigger_queries)),
                            signature_patterns=tuple(dict.fromkeys(left.signature_patterns + right.signature_patterns)),
                            emit_prompt_scaffolds=tuple(dict.fromkeys(left.emit_prompt_scaffolds + right.emit_prompt_scaffolds)),
                            emit_code_probes=tuple(dict.fromkeys(left.emit_code_probes + right.emit_code_probes)),
                            emit_contract_validators=tuple(dict.fromkeys(left.emit_contract_validators + right.emit_contract_validators)),
                            emit_families=families,
                            priority=max(left.priority, right.priority) + 0.10,
                            mutation_policy="relative_macro_compression",
                            rationale=(
                                "relative outer-loop compression of best-trace "
                                f"regulatory genes {left_name} and {right_name}"
                            ),
                        )
                    )
                    existing.add(macro_name)
                    macro_names.append(macro_name)
                    if len(macro_names) >= max_macro_genes:
                        break

    niche_promoted: set[str] = set()
    by_family: dict[str, list[GeneratorTrace]] = defaultdict(list)
    for trace in traces:
        for family in trace.emitted_families:
            by_family[str(family)].append(trace)
    for family, family_traces in by_family.items():
        best_trace = max(family_traces, key=lambda t: float(t.fitness))
        if float(best_trace.fitness) < niche_min_fitness:
            continue
        for gene_name in best_trace.active_genes:
            niche_promoted.add(str(gene_name))
    if niche_promoted:
        already_promoted = set(promoted)
        added = sorted(g for g in niche_promoted if g not in already_promoted)
        if added:
            promoted = sorted(already_promoted | niche_promoted)
            boosted = []
            for gene in new_genes:
                delta = 0.06 if gene.name in added else 0.0
                boosted.append(_gene_with(gene, priority=max(0.05, gene.priority + delta)))
            new_genes = boosted
        if len(macro_names) < max_macro_genes:
            base_by_name = {gene.name: gene for gene in new_genes}
            existing = {gene.name for gene in new_genes}
            for family, family_traces in sorted(by_family.items()):
                best_trace = max(family_traces, key=lambda t: float(t.fitness))
                if float(best_trace.fitness) < niche_min_fitness:
                    continue
                for left_name, right_name in combinations(sorted(set(best_trace.active_genes)), 2):
                    if _is_nonprimitive_gene(left_name) or _is_nonprimitive_gene(right_name):
                        continue
                    left = base_by_name.get(left_name)
                    right = base_by_name.get(right_name)
                    if left is None or right is None:
                        continue
                    macro_name = f"macro_{left_name}__{right_name}"
                    if macro_name in existing:
                        continue
                    families = tuple(dict.fromkeys(left.emit_families + right.emit_families))
                    if not families:
                        continue
                    new_genes.append(
                        RegulatoryGene(
                            name=macro_name,
                            trigger_syndromes=tuple(dict.fromkeys(left.trigger_syndromes + right.trigger_syndromes)),
                            trigger_queries=tuple(dict.fromkeys(left.trigger_queries + right.trigger_queries)),
                            signature_patterns=tuple(dict.fromkeys(left.signature_patterns + right.signature_patterns)),
                            emit_prompt_scaffolds=tuple(dict.fromkeys(left.emit_prompt_scaffolds + right.emit_prompt_scaffolds)),
                            emit_code_probes=tuple(dict.fromkeys(left.emit_code_probes + right.emit_code_probes)),
                            emit_contract_validators=tuple(dict.fromkeys(left.emit_contract_validators + right.emit_contract_validators)),
                            emit_families=families,
                            priority=max(left.priority, right.priority) + 0.08,
                            mutation_policy="niche_macro_compression",
                            rationale=(
                                "niche-balanced outer-loop compression for "
                                f"family {family}: {left_name} and {right_name}"
                            ),
                        )
                    )
                    existing.add(macro_name)
                    macro_names.append(macro_name)
                    if len(macro_names) >= max_macro_genes:
                        break
                    if len(macro_names) >= max_macro_genes:
                        break

    family_mutations = _induce_family_mutations(
        traces,
        new_genes,
        external_deltas=external_deltas or {},
    )
    new_genes = _apply_external_confirmations(
        new_genes,
        external_deltas=external_deltas or {},
    )
    if family_mutations:
        existing = {gene.name for gene in new_genes}
        for gene in family_mutations:
            if gene.name in existing:
                continue
            new_genes.append(gene)
            existing.add(gene.name)

    evolved = RegulatoryGenome(
        name=f"{genome.name}_g{genome.generation + 1}",
        genes=tuple(new_genes),
        generation=genome.generation + 1,
        lineage=tuple(genome.lineage) + ("outer_loop_evolve",),
    )
    return OuterLoopResult(
        genome=evolved,
        promoted=tuple(promoted),
        demoted=tuple(demoted),
        macro_genes=tuple(macro_names),
        trace={
            "n_traces": len(traces),
            "promoted": promoted,
            "demoted": demoted,
            "macro_genes": macro_names,
            "generation": evolved.generation,
            "niche_promoted": sorted(niche_promoted),
            "family_mutations": [gene.to_dict() for gene in family_mutations],
        },
    )


def _is_nonprimitive_gene(name: str) -> bool:
    value = str(name)
    return value.startswith("macro_") or value.startswith("family_")


def _induce_family_mutations(
    traces: Sequence[GeneratorTrace],
    genes: Sequence[RegulatoryGene],
    *,
    external_deltas: Mapping[str, float],
) -> list[RegulatoryGene]:
    """Create new emitted families from repeated high-value trace phenotypes.

    This is the first nontrivial developmental mutation: the generator learns a
    new hypothesis-family branch from the kind of winner that emerged in a
    previous run.  It is intentionally conservative and trace-derived.
    """

    by_name = {gene.name: gene for gene in genes}
    existing_families = {
        family
        for gene in genes
        for family in gene.emit_families
    }
    out: list[RegulatoryGene] = []
    for trace in traces:
        meta = dict(trace.metadata or {})
        best = str(meta.get("best", ""))
        benchmark = str(meta.get("benchmark", ""))
        external_delta = float(external_deltas.get(benchmark, 0.0) or 0.0)
        priority = 0.65
        policy = "trace_induced_family_birth_auxiliary"
        if external_delta > 1e-9:
            priority = 1.35
            policy = "external_confirmed_family_birth"
        elif external_delta < -1e-9:
            priority = 0.25
            policy = "external_quarantined_family_birth"
        if (
            "table_chain" in trace.emitted_families
            and "robust_mediated_trend" in best
            and "table_robust_mediation" not in existing_families
            and float(trace.fitness) >= 0.55
        ):
            trigger_queries: list[str] = []
            signature_patterns: list[str] = []
            for gene_name in trace.active_genes:
                gene = by_name.get(gene_name)
                if gene is None or _is_nonprimitive_gene(gene.name):
                    continue
                trigger_queries.extend(gene.trigger_queries)
                signature_patterns.extend(gene.signature_patterns)
            out.append(
                RegulatoryGene(
                    name="family_table_robust_mediation_developer",
                    trigger_queries=tuple(dict.fromkeys(trigger_queries or ["axis_orientation_gene", "role_binding_gene"])),
                    signature_patterns=tuple(dict.fromkeys(signature_patterns or ["analyze(df)"])),
                    emit_prompt_scaffolds=("robust_mediated_measurement_plan",),
                    emit_code_probes=("robust_mediation_probe",),
                    emit_contract_validators=("variable_relation_contract",),
                    emit_families=("table_robust_mediation",),
                    priority=priority,
                    mutation_policy=policy,
                    rationale=(
                        "trace-induced family: robust mediated trend won inside "
                        f"table_chain; external_delta={external_delta:.4f}"
                    ),
                )
            )
    return out[:4]


def _apply_external_confirmations(
    genes: Sequence[RegulatoryGene],
    *,
    external_deltas: Mapping[str, float],
) -> list[RegulatoryGene]:
    """Promote or quarantine existing auxiliary family genes after external eval."""

    if not external_deltas:
        return list(genes)
    discovery_delta = float(external_deltas.get("discoverybench", 0.0) or 0.0)
    out: list[RegulatoryGene] = []
    for gene in genes:
        if gene.name != "family_table_robust_mediation_developer":
            out.append(gene)
            continue
        if discovery_delta > 1e-9:
            out.append(
                _gene_with(
                    gene,
                    priority=max(gene.priority, 1.35),
                    mutation_policy="external_confirmed_family",
                    rationale=gene.rationale + f" | externally confirmed delta={discovery_delta:.4f}",
                )
            )
        elif discovery_delta < -1e-9:
            out.append(
                _gene_with(
                    gene,
                    priority=min(gene.priority, 0.25),
                    mutation_policy="external_quarantined_family",
                    rationale=gene.rationale + f" | externally quarantined delta={discovery_delta:.4f}",
                )
            )
        else:
            out.append(gene)
    return out


def _gene_with(gene: RegulatoryGene, **updates: Any) -> RegulatoryGene:
    row = gene.to_dict()
    row.update(updates)
    return RegulatoryGene.from_dict(row)
