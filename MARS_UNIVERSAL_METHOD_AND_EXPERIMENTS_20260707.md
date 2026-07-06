# MARS Universal Method and Experiments

Status: working research snapshot, 2026-07-07.

This document describes the current MARS system, its novelty claim, the latest
experiments, and the remaining gaps before a stronger paper/demo claim.

## 1. Motivation

Small language models fail on long scientific tasks for two related reasons:

1. They cannot reliably invent the right high-level hypothesis in one shot.
2. They cannot hold a long chain of reasoning well enough to choose among many
   weakly specified hypotheses.

MARS attacks this by moving the hard part out of free-form prose and into
measured artifacts. A hypothesis is treated as a partially specified program:
it has typed holes, evidence requirements, executable measurements, and a
posterior score. The language model is used as a proposer and compiler, while
data execution and metamorphic checks supply the pressure that a weak model
lacks internally.

## 2. Method Summary

### 2.1 UniversalHypothesisKernel

`UniversalHypothesisKernel` is the top-level search object. It receives a
`KernelTask`:

- `task_text`
- `interface_kind`
- optional data/table/files
- schema and metadata
- domain context

It compiles an evidence contract, calls a set of artifact-producing operators,
rescoring each candidate with metamorphic checks, then sorts candidates by a
shared posterior.

The kernel currently includes operators for evidence contracts, slot contracts,
answer plans, problem frames, contrastive worlds, universal slots, metamorphic
estimands, and executable estimands.

Key file:

- `mars/induction/universal_hypothesis_kernel.py`

### 2.2 Evidence Contracts

The Evidence Contract Compiler turns a task into a typed checklist:

- expected answer form
- required slots
- measurable evidence
- relation/object/scope constraints
- rejection reasons for incomplete or mismatched evidence

This prevents the system from accepting a fluent answer that measures a nearby
but different question.

Key file:

- `mars/skills/evidence_contract_compiler.py`

### 2.3 Metamorphic Estimand Kernel

MEK is a label-free verifier. It checks whether a candidate answer remains
aligned with the question under task-preserving transformations:

- required typed holes must be closed;
- answer form must match the contract;
- temporal and categorical scopes must not drift;
- query role tokens must survive rendering;
- period/category questions must not be answered by row-level numeric proxies;
- underspecified coefficient questions must anchor variables to context.

The output is a residual signature such as `answer_form_drift`,
`period_axis_not_categorical`, or `semantic_role_anchor_missing`. These
signatures are intended to drive later operator mutation.

Key file:

- `mars/induction/metamorphic_estimand_kernel.py`

### 2.4 EstimandSynthesizer

The EstimandSynthesizer creates missing statistical hypotheses when reranking
cannot help. It compiles a question and observable table into an executable
estimand:

- population/filter
- outcome
- predictor or group
- covariates
- interaction or contrast
- model family
- statistic or coefficient
- rendering contract

It then runs a small measurement, such as a binary regression, nested
coefficient delta, group contrast, interaction regression, role-conditioned
outcome contrast, or multi-item survey proportion.

Key file:

- `mars/induction/estimand_synthesizer.py`

### 2.5 Role Canonicalization

The role layer maps raw columns into query-supported semantic roles before an
estimand is fit. This is important because real datasets often expose multiple
proxy columns for the same latent object.

Example:

- `urban.1956.50m`
- `urban.1993.50m`
- `urban.2009.50m`

can be materialized as one `urban_land_use` role rather than treated as three
unrelated variables.

The current implementation has two parts:

1. A small seed ontology of common scientific/social roles.
2. A dynamic role inducer that clusters schema/query evidence without benchmark
   IDs or gold labels.

Key file:

- `mars/induction/role_canonicalizer.py`

## 3. Novelty Claim

The main novelty is not "more agents" or "better prompts." It is a
self-building causal/program induction layer with executable hypothesis
materialization:

1. Hypotheses are structured artifacts, not only text.
2. Natural-language questions are compiled into typed contracts and typed holes.
3. Holes are closed by small measurements over the available environment.
4. Candidate answers are rejected by metamorphic invariants before judging.
5. The same posterior ranks statistical estimands, symbolic probes, and
   program-induction rules.

The important mechanism is decompositional generation. Instead of sampling the
whole hypothesis as prose, the system searches over smaller typed objects whose
values are grounded by execution. This helps a weak model because the model no
longer has to carry the entire derivation in its hidden chain of thought.

## 4. Benchmarks

### DiscoveryBench

Task type: scientific discovery from real tables, schema descriptions, and
paper-derived questions.

Evaluation: HMS on a 0 to 100 scale, with an additional consistency variant
used in our run logs.

Current run:

- `lmw/universal_discovery_real/universal_role_layers_hard_buckets_20260707/summary.json`
- model core: `openai/gpt-4o-mini`
- judge model: `openai/gpt-4o`
- tasks: 30
- raw HMS mean: 40.50
- consistency HMS mean: 60.50

Layer progression:

| System | Raw HMS | Consistency HMS |
|---|---:|---:|
| MEK only | 16.67 | 30.00 |
| EstimandSynthesizer gated | 40.50 | 50.50 |
| Universal role layers | 40.50 | 60.50 |

Dataset-level result for the latest role-layer run:

| Dataset | Tasks | Raw HMS |
|---|---:|---:|
| `introduction_pathways_non-native_plants` | 16 | 37.50 |
| `nls_raw` | 7 | 59.29 |
| `requirements_engineering_for_ML_enabled_systems` | 7 | 28.57 |
| total | 30 | 40.50 |

Interpretation: the system is much stronger than the earlier prompt/contract
versions, especially on NLS-style statistical questions, but raw DiscoveryBench
HMS is not yet a SOTA claim.

### NewtonBench

Task type: symbolic/numeric scientific law induction from generated data.

Evaluation in our runner: symbolic accuracy over all tasks (`SA_all`) and over
answered tasks only (`SA_answered`), with abstentions counted against `SA_all`.

Comparable hard slice:

- `lmw/nb_activeprobe/role_layers_newton_hard_comparable_20260707/summary.json`
- model core: `openai/gpt-4o-mini`
- modules: gravity, Snell, Malus, heat transfer
- law versions: v0 and v1
- tasks: 8
- answered: 5
- abstained: 3
- `SA_answered = 1.000`
- `SA_all = 0.625`

Full hard slice:

- `lmw/nb_activeprobe/role_layers_newton_hard_20260707/summary.json`
- model core: `openai/gpt-4o-mini`
- modules: 12
- law versions: v0 and v1
- tasks: 24
- answered: 12
- abstained: 12
- `SA_answered = 0.833`
- `SA_all = 0.417`

Interpretation: when the executable probe finds the right coordinate chart, the
answers are often exact. The remaining weakness is coverage: the system
abstains or fails to compile on several harder families.

### UltraHorizon

Task type: long-horizon sequence/program induction under sparse feedback.

Current sequence run:

- `lmw/uh_seq_cpi/role_layers_gate_seq_hard_s42_commit_20260707/summary.json`
- model core: `openai/gpt-4o-mini`
- difficulty: hard
- seed: 42
- steps: 7
- committed: true
- exact program fit: 5/5
- final score: 80.0

Interpretation: the sequential CPI layer works well on this checked sequence
slice, where hypothesis programs can be compared directly against observations.

### ScienceAgentBench

Task type: realistic scientific workflows with tool execution and multi-step
research operations.

Current status: not yet a competitive full result in this branch. The
universal kernel design is compatible with this benchmark, but the remaining
missing component is a robust multi-file/tool-execution assembler that can
create, run, inspect, and repair project-level workflows without benchmark
specific patches.

## 5. Latest Test Status

Full local test run:

```text
176 passed, 6 warnings
```

Relevant tests:

- `tests/test_estimand_synthesizer.py`
- `tests/test_universal_hypothesis_kernel.py`
- `tests/test_evidence_contract_compiler.py`

## 6. Reproduction Commands

Tests:

```bash
pytest -q tests
```

DiscoveryBench:

```bash
python -m mars.runners.run_universal_discovery_real_eval \
  --run_id universal_role_layers_hard_buckets_20260707 \
  --max_tasks 30 \
  --datasets nls_raw,worldbank_education_gdp,introduction_pathways_non-native_plants,requirements_engineering_for_ML_enabled_systems \
  --model openai/gpt-4o-mini \
  --judge_model openai/gpt-4o \
  --n_proposals 0 \
  --max_rounds 0 \
  --overwrite
```

NewtonBench comparable hard slice:

```bash
python -m mars.runners.run_nb_activeprobe \
  --run_id role_layers_newton_hard_comparable_20260707 \
  --model openai/gpt-4o-mini \
  --difficulty hard \
  --law_versions v0,v1 \
  --modules m0_gravity,m4_snell_law,m7_malus_law,m11_heat_transfer \
  --overwrite
```

NewtonBench full hard slice:

```bash
python -m mars.runners.run_nb_activeprobe \
  --run_id role_layers_newton_hard_20260707 \
  --model openai/gpt-4o-mini \
  --difficulty hard \
  --law_versions v0,v1 \
  --overwrite
```

UltraHorizon sequence:

```bash
python -m mars.runners.run_uh_seq_cpi \
  --run_id role_layers_gate_seq_hard_s42_commit_20260707 \
  --difficulty hard \
  --seed 42 \
  --steps 7 \
  --judge_model openai/gpt-4o \
  --commit \
  --overwrite
```

## 7. Failure Analysis

### DiscoveryBench

The system still loses raw HMS when:

- the answer requires exact paper-specific phrasing rather than a measured
  relation;
- metadata has to be assembled across multiple files;
- survey item labels need precise normalization;
- the correct scientific object is implicit in the paper rather than explicit
  in the table/schema;
- several plausible measured estimands are statistically true but only one is
  the benchmark answer.

The largest improvement from role layers was consistency: the system now more
often measures the right object even when rendering or exact benchmark phrasing
is imperfect.

### NewtonBench

The main issue is coverage, not precision. Answered tasks are usually correct,
but the system still lacks enough universal coordinate families and compilation
repair to answer every hard law.

### UltraHorizon

The checked sequence slice is strong. More seeds and more environment variants
are needed before claiming benchmark-wide dominance.

### ScienceAgentBench

The missing layer is project-level execution synthesis: a universal assembler
that can infer required files, run tools, read failures, and repair artifacts
without SAB-specific code.

## 8. What Should Be Improved Next

The next universal addition should be an outer-loop residual compiler:

1. Collect residual signatures from MEK, Newton probe failures, and workflow
   execution failures.
2. Cluster them into reusable missing operator families.
3. Propose a minimal operator patch.
4. Test the patch across at least DiscoveryBench, NewtonBench, and
   UltraHorizon before accepting it.
5. Reject patches that only improve one benchmark while regressing the others.

This would connect the current system to a more serious self-improvement loop:
not a growing bag of benchmark tricks, but a compression pressure over repeated
failure signatures.

## 9. Current Evidence Artifacts

Committed summary artifacts are intentionally small:

- `lmw/universal_discovery_real/universal_role_layers_hard_buckets_20260707/summary.json`
- `lmw/nb_activeprobe/role_layers_newton_hard_comparable_20260707/summary.json`
- `lmw/nb_activeprobe/role_layers_newton_hard_20260707/summary.json`
- `lmw/uh_seq_cpi/role_layers_gate_seq_hard_s42_commit_20260707/summary.json`

Raw benchmark repositories, private data, and large prediction logs should not
be committed to the public repository.
