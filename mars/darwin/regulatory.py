"""Regulatory genomes for hypothesis-generator evolution.

A TheoryGenome represents one hypothesis family.  A RegulatoryGenome represents
the developmental mechanism that decides which hypothesis families should be
born from an observed interface and failure pressure.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from mars.darwin.failure_algebra import FailureSyndrome, MissingGeneQuery
from mars.darwin.probes import ProbeGenome


@dataclass(frozen=True)
class RegulatoryGene:
    """A triggerable developmental rule for cognitive phenotype birth.

    The original Darwin layer emitted only hypothesis families.  The IP-Genome
    extension makes a gene express the whole inference prior around a frozen API
    model: prompt scaffolds, executable probes, contract validators, and finally
    hypothesis families.
    """

    name: str
    trigger_syndromes: tuple[str, ...] = ()
    trigger_queries: tuple[str, ...] = ()
    signature_patterns: tuple[str, ...] = ()
    emit_prompt_scaffolds: tuple[str, ...] = ()
    emit_code_probes: tuple[str, ...] = ()
    emit_contract_validators: tuple[str, ...] = ()
    emit_families: tuple[str, ...] = ()
    suppress_families: tuple[str, ...] = ()
    priority: float = 1.0
    mutation_policy: str = "priority_shift"
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trigger_syndromes": list(self.trigger_syndromes),
            "trigger_queries": list(self.trigger_queries),
            "signature_patterns": list(self.signature_patterns),
            "emit_prompt_scaffolds": list(self.emit_prompt_scaffolds),
            "emit_code_probes": list(self.emit_code_probes),
            "emit_contract_validators": list(self.emit_contract_validators),
            "emit_families": list(self.emit_families),
            "suppress_families": list(self.suppress_families),
            "priority": self.priority,
            "mutation_policy": self.mutation_policy,
            "rationale": self.rationale,
        }

    @staticmethod
    def from_dict(row: Mapping[str, Any]) -> "RegulatoryGene":
        name = str(row.get("name", "regulatory_gene"))
        prompt_scaffolds = tuple(str(x) for x in row.get("emit_prompt_scaffolds", ()) or ())
        code_probes = tuple(str(x) for x in row.get("emit_code_probes", ()) or ())
        contract_validators = tuple(str(x) for x in row.get("emit_contract_validators", ()) or ())
        if not prompt_scaffolds and not code_probes and not contract_validators:
            prompt_scaffolds, code_probes, contract_validators = _ip_expression_defaults(name)
        return RegulatoryGene(
            name=name,
            trigger_syndromes=tuple(str(x) for x in row.get("trigger_syndromes", ()) or ()),
            trigger_queries=tuple(str(x) for x in row.get("trigger_queries", ()) or ()),
            signature_patterns=tuple(str(x) for x in row.get("signature_patterns", ()) or ()),
            emit_prompt_scaffolds=prompt_scaffolds,
            emit_code_probes=code_probes,
            emit_contract_validators=contract_validators,
            emit_families=tuple(str(x) for x in row.get("emit_families", ()) or ()),
            suppress_families=tuple(str(x) for x in row.get("suppress_families", ()) or ()),
            priority=float(row.get("priority", 1.0)),
            mutation_policy=str(row.get("mutation_policy", "priority_shift")),
            rationale=str(row.get("rationale", "")),
        )


@dataclass(frozen=True)
class RegulatoryGenome:
    """A compact genome for the hypothesis generator itself."""

    name: str
    genes: tuple[RegulatoryGene, ...]
    generation: int = 0
    lineage: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "generation": self.generation,
            "lineage": list(self.lineage),
            "genes": [gene.to_dict() for gene in self.genes],
        }

    @staticmethod
    def from_dict(row: Mapping[str, Any]) -> "RegulatoryGenome":
        return RegulatoryGenome(
            name=str(row.get("name", "regulatory_genome")),
            genes=tuple(
                RegulatoryGene.from_dict(gene)
                for gene in row.get("genes", ()) or ()
                if isinstance(gene, dict)
            ),
            generation=int(row.get("generation", 0) or 0),
            lineage=tuple(str(x) for x in row.get("lineage", ()) or ()),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load_json(path: str | Path) -> "RegulatoryGenome":
        row = json.loads(Path(path).read_text(encoding="utf-8"))
        genome = RegulatoryGenome.from_dict(row)
        if not genome.genes:
            raise ValueError(f"regulatory genome has no genes: {path}")
        return genome

    def without_genes(self, names: Sequence[str]) -> "RegulatoryGenome":
        remove = {str(name) for name in names}
        return RegulatoryGenome(
            name=f"{self.name}_ablated",
            genes=tuple(gene for gene in self.genes if gene.name not in remove),
            generation=self.generation,
            lineage=tuple(self.lineage) + tuple(f"ablate:{name}" for name in sorted(remove)),
        )

    def with_gene_priority(self, name: str, priority: float) -> "RegulatoryGenome":
        target = str(name)
        genes = []
        for gene in self.genes:
            if gene.name != target:
                genes.append(gene)
                continue
            genes.append(
                RegulatoryGene(
                    name=gene.name,
                    trigger_syndromes=gene.trigger_syndromes,
                    trigger_queries=gene.trigger_queries,
                    signature_patterns=gene.signature_patterns,
                    emit_prompt_scaffolds=gene.emit_prompt_scaffolds,
                    emit_code_probes=gene.emit_code_probes,
                    emit_contract_validators=gene.emit_contract_validators,
                    emit_families=gene.emit_families,
                    suppress_families=gene.suppress_families,
                    priority=float(priority),
                    mutation_policy=gene.mutation_policy,
                    rationale=gene.rationale,
                )
            )
        return RegulatoryGenome(
            name=f"{self.name}_priority_{target}",
            genes=tuple(genes),
            generation=self.generation,
            lineage=tuple(self.lineage) + (f"priority:{target}:{priority}",),
        )


@dataclass(frozen=True)
class DevelopmentResult:
    """Result of developing a regulatory genome on a task interface."""

    active_genes: tuple[RegulatoryGene, ...]
    emitted_prompt_scaffolds: tuple[str, ...]
    emitted_code_probes: tuple[str, ...]
    emitted_contract_validators: tuple[str, ...]
    emitted_families: tuple[str, ...]
    suppressed_families: tuple[str, ...]
    trace: dict[str, Any]


def default_regulatory_genome() -> RegulatoryGenome:
    """Return the initial universal generator genome.

    These genes are not benchmark answers.  They are interface/failure-triggered
    developmental rules that decide which hypothesis family should be explored.
    """

    return RegulatoryGenome(
        name="mars_regulatory_seed",
        genes=(
            RegulatoryGene(
                name="axis_orientation_developer",
                trigger_queries=("axis_orientation_gene",),
                signature_patterns=("analyze(df)",),
                emit_prompt_scaffolds=("axis_role_decomposition",),
                emit_code_probes=("axis_orientation_scan", "row_column_role_probe"),
                emit_contract_validators=("table_answer_contract",),
                emit_families=("table_chain",),
                priority=1.4,
                rationale="ambiguous axes should first develop object/axis hypotheses",
            ),
            RegulatoryGene(
                name="semantic_role_developer",
                trigger_queries=("role_binding_gene",),
                signature_patterns=("analyze(df)",),
                emit_prompt_scaffolds=("semantic_slot_binding",),
                emit_code_probes=("schema_token_binding_probe",),
                emit_contract_validators=("variable_relation_contract",),
                emit_families=("table_chain",),
                priority=1.2,
                rationale="semantic pressure should develop role-binding hypotheses",
            ),
            RegulatoryGene(
                name="scale_law_developer",
                trigger_queries=("scale_relation_gene",),
                signature_patterns=("law(inputs",),
                emit_prompt_scaffolds=("typed_numeric_law_decomposition",),
                emit_code_probes=("loglog_exponent_probe", "residual_shape_probe"),
                emit_contract_validators=("numeric_law_signature_contract",),
                emit_families=("numeric_power_law",),
                priority=1.3,
                rationale="numeric scalar interfaces should develop scale-law hypotheses",
            ),
            RegulatoryGene(
                name="transition_developer",
                trigger_queries=("transition_rule_gene",),
                signature_patterns=("rule(current",),
                emit_prompt_scaffolds=("transition_state_decomposition",),
                emit_code_probes=("string_delta_probe", "positionwise_edit_probe"),
                emit_contract_validators=("transition_io_contract",),
                emit_families=("string_transition",),
                priority=1.25,
                rationale="string transition interfaces should develop transition rules",
            ),
            RegulatoryGene(
                name="state_role_developer",
                trigger_queries=("state_role_gene",),
                signature_patterns=("predict(parent1",),
                emit_prompt_scaffolds=("latent_role_decomposition",),
                emit_code_probes=("state_role_aggregation_probe",),
                emit_contract_validators=("structured_state_contract",),
                emit_families=("dict_state_predictor",),
                priority=1.1,
                rationale="structured state interfaces should develop role-level predictors",
            ),
            RegulatoryGene(
                name="table_fallback_developer",
                signature_patterns=("analyze(df)",),
                emit_prompt_scaffolds=("generic_table_measurement_plan",),
                emit_code_probes=("basic_table_profile",),
                emit_contract_validators=("generic_answer_contract",),
                emit_families=("table_chain",),
                priority=0.45,
                rationale="fallback table theory family when no stronger syndrome fires",
            ),
            RegulatoryGene(
                name="numeric_fallback_developer",
                signature_patterns=("law(inputs",),
                emit_prompt_scaffolds=("generic_numeric_measurement_plan",),
                emit_code_probes=("basic_numeric_profile",),
                emit_contract_validators=("numeric_output_contract",),
                emit_families=("numeric_power_law",),
                priority=0.45,
                rationale="fallback numeric theory family when no stronger syndrome fires",
            ),
        ),
        lineage=("seed",),
    )


def develop_regulatory_genome(
    genome: RegulatoryGenome,
    *,
    signature_hint: str,
    syndromes: Sequence[FailureSyndrome],
    queries: Sequence[MissingGeneQuery],
    probes: Sequence[ProbeGenome],
) -> DevelopmentResult:
    """Develop a generator genome into hypothesis-family emissions."""

    active: list[tuple[float, RegulatoryGene, dict[str, Any]]] = []
    syndrome_names = {s.kind for s in syndromes}
    syndrome_severity = {s.kind: float(s.severity) for s in syndromes}
    query_names = {q.gene_type for q in queries}
    for gene in genome.genes:
        matched_syndromes = tuple(s for s in gene.trigger_syndromes if s in syndrome_names)
        matched_queries = tuple(q for q in gene.trigger_queries if q in query_names)
        matched_signature = tuple(
            p for p in gene.signature_patterns if _signature_matches(signature_hint, p)
        )
        if not matched_syndromes and not matched_queries and not matched_signature:
            continue
        severity_bonus = sum(syndrome_severity.get(s, 0.0) for s in matched_syndromes)
        query_bonus = 0.25 * len(matched_queries)
        signature_bonus = 0.1 * len(matched_signature)
        score = float(gene.priority) + severity_bonus + query_bonus + signature_bonus
        active.append(
            (
                score,
                gene,
                {
                    "matched_syndromes": matched_syndromes,
                    "matched_queries": matched_queries,
                    "matched_signature": matched_signature,
                    "activation_score": score,
                },
            )
        )
    active.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    prompt_scaffolds: list[str] = []
    code_probes: list[str] = []
    contract_validators: list[str] = []
    emitted: list[str] = []
    suppressed: list[str] = []
    activation_rows = []
    for score, gene, row in active:
        activation_rows.append({"gene": gene.name, **row})
        for scaffold in gene.emit_prompt_scaffolds:
            if scaffold not in prompt_scaffolds:
                prompt_scaffolds.append(scaffold)
        for probe in gene.emit_code_probes:
            if probe not in code_probes:
                code_probes.append(probe)
        for validator in gene.emit_contract_validators:
            if validator not in contract_validators:
                contract_validators.append(validator)
        for family in gene.suppress_families:
            if family not in suppressed:
                suppressed.append(family)
        for family in gene.emit_families:
            if family not in emitted:
                emitted.append(family)
    emitted = [family for family in emitted if family not in set(suppressed)]
    return DevelopmentResult(
        active_genes=tuple(g for _, g, _ in active),
        emitted_prompt_scaffolds=tuple(prompt_scaffolds),
        emitted_code_probes=tuple(code_probes),
        emitted_contract_validators=tuple(contract_validators),
        emitted_families=tuple(emitted),
        suppressed_families=tuple(suppressed),
        trace={
            "regulatory_genome": genome.name,
            "generation": genome.generation,
            "lineage": list(genome.lineage),
            "active_genes": activation_rows,
            "emitted_prompt_scaffolds": prompt_scaffolds,
            "emitted_code_probes": code_probes,
            "emitted_contract_validators": contract_validators,
            "emitted_families": emitted,
            "suppressed_families": suppressed,
            "n_syndromes": len(syndromes),
            "n_missing_gene_queries": len(queries),
            "n_probes": len(probes),
        },
    )


def _signature_matches(signature_hint: str, pattern: str) -> bool:
    return bool(re.search(re.escape(pattern), str(signature_hint)))


def _ip_expression_defaults(name: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Backfill IP-Genome expression fields for old saved regulatory genomes."""

    base: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
        "axis_orientation_developer": (
            ("axis_role_decomposition",),
            ("axis_orientation_scan", "row_column_role_probe"),
            ("table_answer_contract",),
        ),
        "semantic_role_developer": (
            ("semantic_slot_binding",),
            ("schema_token_binding_probe",),
            ("variable_relation_contract",),
        ),
        "scale_law_developer": (
            ("typed_numeric_law_decomposition",),
            ("loglog_exponent_probe", "residual_shape_probe"),
            ("numeric_law_signature_contract",),
        ),
        "transition_developer": (
            ("transition_state_decomposition",),
            ("string_delta_probe", "positionwise_edit_probe"),
            ("transition_io_contract",),
        ),
        "state_role_developer": (
            ("latent_role_decomposition",),
            ("state_role_aggregation_probe",),
            ("structured_state_contract",),
        ),
        "table_fallback_developer": (
            ("generic_table_measurement_plan",),
            ("basic_table_profile",),
            ("generic_answer_contract",),
        ),
        "numeric_fallback_developer": (
            ("generic_numeric_measurement_plan",),
            ("basic_numeric_profile",),
            ("numeric_output_contract",),
        ),
        "family_table_robust_mediation_developer": (
            ("robust_mediated_measurement_plan",),
            ("robust_mediation_probe",),
            ("variable_relation_contract",),
        ),
    }
    if name in base:
        return base[name]
    if name.startswith("macro_") and "__" in name:
        prompt_scaffolds: list[str] = []
        code_probes: list[str] = []
        contract_validators: list[str] = []
        for part in name.removeprefix("macro_").split("__"):
            prompts, probes, validators = _ip_expression_defaults(part)
            prompt_scaffolds.extend(prompts)
            code_probes.extend(probes)
            contract_validators.extend(validators)
        return (
            tuple(dict.fromkeys(prompt_scaffolds)),
            tuple(dict.fromkeys(code_probes)),
            tuple(dict.fromkeys(contract_validators)),
        )
    return (), (), ()
