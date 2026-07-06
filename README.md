# MARS: Universal Self-Induced Hypothesis System

MARS is a research prototype for making a smaller model behave more like a
scientific problem solver: instead of asking the model to guess a final answer
in prose, the system builds executable hypothesis artifacts, tests them against
the available evidence, and ranks them with a shared posterior.

The current version focuses on a universal layer that can run across scientific
discovery, symbolic law induction, and long-horizon program induction
benchmarks without writing a separate benchmark-specific solver for each task.

## Core Idea

The central object is a self-building causal/program induction layer:

1. Compile the natural-language task and available data into typed evidence
   contracts.
2. Build candidate artifacts with explicit holes: variables, roles, outcomes,
   operators, transforms, estimands, and rendering contracts.
3. Close those holes by executable probes over the data rather than by prose
   guessing.
4. Validate the resulting answer with metamorphic and contract checks.
5. Reuse only compressed abstractions that improve evidence fit under a common
   scoring rule.

This keeps the method different from a pure ReAct or prompt-agent loop. The
model does not just "think longer"; it writes or selects small measurable
objects whose outputs can be executed, falsified, and compared.

## Current Universal Stack

The main implementation lives in:

- `mars/induction/universal_hypothesis_kernel.py` - one posterior over
  artifact-producing operators.
- `mars/skills/evidence_contract_compiler.py` - typed evidence contracts from a
  task interface.
- `mars/induction/metamorphic_estimand_kernel.py` - label-free validation of
  answer form, scope, and role preservation.
- `mars/induction/estimand_synthesizer.py` - executable statistical estimands
  for tabular discovery tasks.
- `mars/induction/role_canonicalizer.py` - query/schema role materialization
  before fitting estimands.
- `mars/runners/run_universal_discovery_real_eval.py` - DiscoveryBench real-data
  evaluation runner.
- `mars/runners/run_nb_activeprobe.py` - NewtonBench active-probe runner.
- `mars/runners/run_uh_seq_cpi.py` - UltraHorizon sequence induction runner.

See `MARS_UNIVERSAL_METHOD_AND_EXPERIMENTS_20260707.md` for the full method
description, benchmark status, commands, and failure analysis.

## Latest Checked Results

| Benchmark slice | Model core | Method | Main result |
|---|---|---|---|
| DiscoveryBench hard buckets, 30 tasks | GPT-4o mini | UHK + MEK + EstimandSynthesizer + role layers | HMS 40.50, consistency HMS 60.50 |
| NewtonBench hard comparable slice, 8 tasks | GPT-4o mini | Active probes + universal coordinate charts | SA all 0.625, SA answered 1.000 |
| NewtonBench hard full slice, 24 tasks | GPT-4o mini | Active probes + universal coordinate charts | SA all 0.417, SA answered 0.833 |
| UltraHorizon sequence hard, seed 42 | GPT-4o mini | Sequential CPI with committed rule induction | exact program fit 5/5, score 80.0 |
| Unit tests | local | current code | 176 passed |

ScienceAgentBench remains an open gap for this branch: the current universal
method needs a stronger multi-file/tool-execution assembler before claiming a
competitive full result there.

## Reproducing the Main Checks

Run tests:

```bash
pytest -q tests
```

DiscoveryBench hard bucket run:

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

UltraHorizon sequence induction:

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

## Repository Hygiene

Raw benchmark repositories, private data directories, virtual environments, and
API keys are intentionally ignored. Use environment variables for credentials
such as OpenRouter/OpenAI keys; do not commit `.env` files.
