import pandas as pd
import math
from unittest.mock import patch

from mars.darwin import (
    DarwinSynthesizer,
    GeneratorTrace,
    default_regulatory_genome,
    develop_regulatory_genome,
    evolve_regulatory_genome,
    infer_failure_syndromes,
    missing_gene_queries,
)
from mars.darwin.archive import TheoryArchive
from mars.darwin.probes import build_probe_genomes
from mars.induction.cpi_adapters import DiscoveryAdapter, NewtonAdapter
from mars.induction.universal_cpi import Observation, UniversalCPI


def test_failure_algebra_turns_wide_table_into_missing_axis_gene():
    df = pd.DataFrame(
        {
            "Series Name": [
                "education expenditure",
                "labor force participation",
                "GNI per capita",
            ],
            "1990 [YR1990]": [2.0, 50.0, 100.0],
            "1991 [YR1991]": [3.0, 55.0, 130.0],
            "1992 [YR1992]": [4.0, 60.0, 160.0],
            "1993 [YR1993]": [5.0, 65.0, 190.0],
            "1994 [YR1994]": [6.0, 70.0, 220.0],
        }
    )
    adapter = DiscoveryAdapter(
        df,
        "How does education expenditure influence human capital and economic output?",
        "Human capital can be proxied by labor force; output by GNI per capita.",
        {"Series Name": "indicator name"},
        enabled_modules=set(),
    )
    syndromes = infer_failure_syndromes(
        signature_hint=adapter.signature_hint(),
        observations=adapter.collect_observations(),
        question=adapter.question,
        column_descriptions=adapter.column_descriptions,
        domain_context=adapter.domain_knowledge,
    )
    queries = missing_gene_queries(syndromes)
    assert any(s.kind == "wide_table_axis_ambiguity" for s in syndromes)
    assert any(q.gene_type == "axis_orientation_gene" for q in queries)


def test_darwin_numeric_law_gene_solves_power_law_without_llm():
    rows = []
    for x, z in [(1.0, 2.0), (2.0, 2.0), (3.0, 3.0), (4.0, 2.0), (5.0, 5.0)]:
        rows.append({"x": x, "z": z, "y": 3.0 * x * x / z})
    adapter = NewtonAdapter(rows, ["x", "z"], "y")
    observations = adapter.collect_observations()

    result = DarwinSynthesizer(max_programs=8).synthesize(
        signature_hint=adapter.signature_hint(),
        observations=observations,
        interface_name=adapter.name,
    )
    assert any(g.kind == "numeric_power_law" for g in result.genomes)
    assert any(
        q["gene_type"] == "scale_relation_gene"
        for row in result.trace
        for q in row.get("missing_gene_queries", [])
    )

    engine = UniversalCPI(model="unused", max_rounds=0, n_proposals=0)
    with patch.dict("os.environ", {"MARS_DARWIN": "1"}, clear=False):
        programs, errors, proposals = engine._darwin_programs(adapter, observations)
    assert not errors
    assert any(p.name == "darwin_numeric_power_law_scale_relation" for p in programs)
    ranked = engine._rank(programs, observations, adapter)
    assert ranked[0][1].loss_mean < 1e-6
    assert any(p.get("kind") == "darwin_trace" for p in proposals)


def test_darwin_numeric_gene_emits_bounded_trig_relation_without_llm():
    rows = []
    for n1, n2, angle1 in [
        (1.0, 1.5, 20.0),
        (1.2, 1.7, 35.0),
        (1.1, 1.6, 45.0),
        (1.3, 1.8, 55.0),
        (1.0, 1.4, 65.0),
    ]:
        angle2 = math.degrees(math.asin((n1 / n2) * math.sin(math.radians(angle1))))
        rows.append({"n1": n1, "n2": n2, "angle1": angle1, "angle2": angle2})
    adapter = NewtonAdapter(rows, ["n1", "n2", "angle1"], "angle2")
    observations = adapter.collect_observations()

    result = DarwinSynthesizer(max_programs=16).synthesize(
        signature_hint=adapter.signature_hint(),
        observations=observations,
        interface_name=adapter.name,
    )
    assert any(g.kind == "numeric_trig_relation" for g in result.genomes)

    engine = UniversalCPI(model="unused", max_rounds=0, n_proposals=0)
    with patch.dict("os.environ", {"MARS_DARWIN": "1"}, clear=False):
        programs, errors, _proposals = engine._darwin_programs(adapter, observations)
    assert not errors
    ranked = engine._rank(programs, observations, adapter)
    assert ranked[0][0].name.startswith("darwin_numeric_bounded_trig_relation")
    assert ranked[0][1].loss_mean < 1e-6


def test_darwin_string_transition_gene_is_interface_generic():
    observations = [
        Observation(inputs="AB", target="ABAB", context={}),
        Observation(inputs="C", target="CC", context={}),
        Observation(inputs="XYZ", target="XYZXYZ", context={}),
    ]
    result = DarwinSynthesizer(max_programs=8).synthesize(
        signature_hint="def rule(current: str, context: dict) -> str",
        observations=observations,
        interface_name="synthetic_transition",
    )
    names = [src["name"] for src in result.program_sources]
    assert "darwin_string_transition_double_current" in names


def test_theory_archive_keeps_elite_per_niche():
    observations = [
        Observation(inputs="A", target="AA", context={}),
        Observation(inputs="BC", target="BCBC", context={}),
    ]
    result = DarwinSynthesizer(max_programs=8).synthesize(
        signature_hint="def rule(current: str, context: dict) -> str",
        observations=observations,
    )
    archive = TheoryArchive()
    for genome in result.genomes:
        archive.add(genome, fitness=1.0 / max(1.0, genome.complexity))
    assert archive.elites()
    assert archive.to_trace()[0]["genome"].startswith("string_transition")


def test_regulatory_genome_develops_hypothesis_families_from_missing_genes():
    observations = [
        Observation(inputs={"x": 1.0}, target=2.0, context={}),
        Observation(inputs={"x": 2.0}, target=4.0, context={}),
        Observation(inputs={"x": 3.0}, target=6.0, context={}),
    ]
    signature = "def law(inputs: dict) -> float"
    syndromes = infer_failure_syndromes(signature_hint=signature, observations=observations)
    queries = missing_gene_queries(syndromes)
    probes = build_probe_genomes(signature_hint=signature, observations=observations)
    development = develop_regulatory_genome(
        default_regulatory_genome(),
        signature_hint=signature,
        syndromes=syndromes,
        queries=queries,
        probes=probes,
    )
    assert "numeric_power_law" in development.emitted_families
    assert "typed_numeric_law_decomposition" in development.emitted_prompt_scaffolds
    assert "loglog_exponent_probe" in development.emitted_code_probes
    assert "numeric_law_signature_contract" in development.emitted_contract_validators
    assert any(g.name == "scale_law_developer" for g in development.active_genes)
    assert development.trace["n_missing_gene_queries"] >= 1
    assert "emitted_prompt_scaffolds" in development.trace
    assert "emitted_code_probes" in development.trace
    assert "emitted_contract_validators" in development.trace


def test_ip_genome_context_guides_llm_prompt_and_result_trace():
    rows = [
        {"x": 1.0, "y": 2.0},
        {"x": 2.0, "y": 4.0},
        {"x": 3.0, "y": 6.0},
        {"x": 4.0, "y": 8.0},
    ]
    adapter = NewtonAdapter(rows, ["x"], "y")
    observations = adapter.collect_observations()
    engine = UniversalCPI(model="unused", max_rounds=0, n_proposals=0)

    prompt = engine._build_prompt(adapter, observations)
    assert "INFERENCE-PRIOR GENOME EXPRESSION" in prompt
    assert "typed_numeric_law_decomposition" in prompt
    assert "loglog_exponent_probe" in prompt
    assert "numeric_law_signature_contract" in prompt

    result = engine.run(adapter)
    ip_rows = [p for p in result.proposals_raw if p.get("kind") == "ip_genome_context"]
    assert ip_rows
    assert "numeric_power_law" in ip_rows[0]["hypothesis_families"]


def test_outer_loop_evolves_generator_not_hypothesis():
    genome = default_regulatory_genome()
    traces = [
        GeneratorTrace(
            task_id="t1",
            active_genes=("axis_orientation_developer", "semantic_role_developer"),
            emitted_families=("table_chain",),
            fitness=0.9,
        ),
        GeneratorTrace(
            task_id="t2",
            active_genes=("axis_orientation_developer", "semantic_role_developer"),
            emitted_families=("table_chain",),
            fitness=0.85,
        ),
        GeneratorTrace(
            task_id="t3",
            active_genes=("numeric_fallback_developer",),
            emitted_families=("numeric_power_law",),
            fitness=0.2,
        ),
    ]
    evolved = evolve_regulatory_genome(genome, traces)
    assert evolved.genome.generation == genome.generation + 1
    assert "axis_orientation_developer" in evolved.promoted
    assert "numeric_fallback_developer" in evolved.demoted
    assert any(name.startswith("macro_axis_orientation") for name in evolved.macro_genes)
    assert any(g.mutation_policy == "macro_compression" for g in evolved.genome.genes)
    macro = next(g for g in evolved.genome.genes if g.mutation_policy == "macro_compression")
    assert "axis_role_decomposition" in macro.emit_prompt_scaffolds
    assert "schema_token_binding_probe" in macro.emit_code_probes
    assert "variable_relation_contract" in macro.emit_contract_validators


def test_regulatory_genome_json_roundtrip(tmp_path):
    genome = default_regulatory_genome()
    path = tmp_path / "regulatory_genome.json"
    genome.save_json(path)
    loaded = type(genome).load_json(path)
    assert loaded.name == genome.name
    assert len(loaded.genes) == len(genome.genes)
    assert loaded.genes[0].name == genome.genes[0].name
    assert loaded.genes[0].emit_prompt_scaffolds == genome.genes[0].emit_prompt_scaffolds
    assert loaded.genes[0].emit_code_probes == genome.genes[0].emit_code_probes
    assert loaded.genes[0].emit_contract_validators == genome.genes[0].emit_contract_validators


def test_regulatory_genome_ablation_and_priority_edit():
    genome = default_regulatory_genome()
    ablated = genome.without_genes(["transition_developer"])
    assert all(gene.name != "transition_developer" for gene in ablated.genes)
    assert len(ablated.genes) == len(genome.genes) - 1

    edited = genome.with_gene_priority("scale_law_developer", 2.5)
    scale_gene = next(gene for gene in edited.genes if gene.name == "scale_law_developer")
    assert scale_gene.priority == 2.5


def test_high_table_regulatory_pressure_emits_robust_table_phenotype():
    df = pd.DataFrame(
        {
            "Series Name": [
                "Adjusted savings: education expenditure",
                "Labor force participation rate",
                "GNI per capita",
            ],
            "1990 [YR1990]": [2.0, 50.0, 100.0],
            "1991 [YR1991]": [3.0, 55.0, 130.0],
            "1992 [YR1992]": [4.0, 60.0, 160.0],
            "1993 [YR1993]": [5.0, 65.0, 190.0],
            "1994 [YR1994]": [6.0, 70.0, 220.0],
            "1995 [YR1995]": [7.0, 75.0, 250.0],
        }
    )
    adapter = DiscoveryAdapter(
        df,
        "How does increased education expenditure influence human capital and economic output?",
        "Human capital can be proxied by labor force; economic output can be proxied by GNI per capita.",
        {"Series Name": "indicator or measured variable"},
        enabled_modules=set(),
    )
    traces = [
        GeneratorTrace(
            task_id="table_success",
            active_genes=("axis_orientation_developer", "semantic_role_developer"),
            emitted_families=("table_chain",),
            fitness=0.9,
        ),
        GeneratorTrace(
            task_id="state_failure",
            active_genes=("state_role_developer",),
            emitted_families=("dict_state_predictor",),
            fitness=0.3,
        ),
    ]
    evolved = evolve_regulatory_genome(default_regulatory_genome(), traces).genome
    result = DarwinSynthesizer(max_programs=16, regulatory_genome=evolved).synthesize(
        signature_hint=adapter.signature_hint(),
        observations=adapter.collect_observations(),
        interface_name=adapter.name,
        question=adapter.question,
        column_descriptions=adapter.column_descriptions,
        domain_context=adapter.domain_knowledge,
    )
    assert any(src["name"] == "darwin_rows_as_variables_robust_mediated_trend" for src in result.program_sources)


def test_outer_loop_births_new_emit_family_from_successful_phenotype():
    genome = default_regulatory_genome()
    traces = [
        GeneratorTrace(
            task_id="discovery_success",
            active_genes=(
                "axis_orientation_developer",
                "semantic_role_developer",
                "table_fallback_developer",
            ),
            emitted_families=("table_chain",),
            fitness=0.8,
            metadata={"best": "darwin_rows_as_variables_robust_mediated_trend"},
        )
    ]
    evolved = evolve_regulatory_genome(genome, traces)
    names = [gene.name for gene in evolved.genome.genes]
    assert "family_table_robust_mediation_developer" in names
    family_gene = next(g for g in evolved.genome.genes if g.name == "family_table_robust_mediation_developer")
    assert family_gene.emit_families == ("table_robust_mediation",)
    assert family_gene.priority < 1.0
    assert family_gene.mutation_policy == "trace_induced_family_birth_auxiliary"


def test_external_delta_confirms_family_birth():
    traces = [
        GeneratorTrace(
            task_id="discovery_success",
            active_genes=("axis_orientation_developer", "semantic_role_developer"),
            emitted_families=("table_chain",),
            fitness=0.8,
            metadata={
                "benchmark": "discoverybench",
                "best": "darwin_rows_as_variables_robust_mediated_trend",
            },
        )
    ]
    evolved = evolve_regulatory_genome(
        default_regulatory_genome(),
        traces,
        external_deltas={"discoverybench": 0.05},
    )
    family_gene = next(g for g in evolved.genome.genes if g.name == "family_table_robust_mediation_developer")
    assert family_gene.priority >= 1.35
    assert family_gene.mutation_policy == "external_confirmed_family_birth"


def test_external_delta_quarantines_family_birth():
    traces = [
        GeneratorTrace(
            task_id="discovery_regression",
            active_genes=("axis_orientation_developer", "semantic_role_developer"),
            emitted_families=("table_chain",),
            fitness=0.8,
            metadata={
                "benchmark": "discoverybench",
                "best": "darwin_rows_as_variables_robust_mediated_trend",
            },
        )
    ]
    evolved = evolve_regulatory_genome(
        default_regulatory_genome(),
        traces,
        external_deltas={"discoverybench": -0.05},
    )
    family_gene = next(g for g in evolved.genome.genes if g.name == "family_table_robust_mediation_developer")
    assert family_gene.priority <= 0.25
    assert family_gene.mutation_policy == "external_quarantined_family_birth"
