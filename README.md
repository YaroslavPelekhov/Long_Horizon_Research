# MARS: Typed Executable Hypothesis Induction

MARS is a research prototype for making a smaller language model behave more
like a scientific problem solver. Instead of asking the model to guess a final
answer in prose, the system builds executable hypothesis artifacts, tests them
against available evidence, and ranks them with a shared posterior-style score.

The current version focuses on one universal hypothesis-state layer that can
run across data-driven discovery, symbolic law induction, and long-horizon rule
induction without writing a separate solver for each benchmark.

The paper draft is available as [`draft.pdf`](draft.pdf). The LaTeX source is
[`MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.tex`](MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.tex).

## Core Idea

The central object is a typed executable hypothesis-induction layer:

1. Compile the natural-language task and available data into typed evidence
   contracts.
2. Build candidate artifacts with explicit holes: variables, roles, outcomes,
   operators, transforms, estimands, and rendering contracts.
3. Close those holes by executable probes over the data rather than by prose
   guessing.
4. Validate the resulting answer with metamorphic and contract checks.
5. Promote recurring residual patterns into reusable operators only when they
   improve held-out checks after a complexity penalty.

This keeps the method different from a pure ReAct or prompt-agent loop. The
model does not just "think longer"; it proposes small measurable objects whose
outputs can be executed, falsified, and compared.

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

Supporting modules include `mars/skills/` for task contracts, metric compilers,
slot compilers, and residual kernels; `mars/darwin/` for residual-conditioned
outer-loop experiments; and `tests/` for focused unit coverage.

## Latest Checked Results

| Benchmark | Model core | Method | Main result |
|---|---|---|---|
| DiscoveryBench | GPT-4o mini | UHK + MEK + estimands + role layers | HMS 40.50, consistency HMS 60.50 |
| NewtonBench | GPT-4o mini | active probes + universal coordinate charts | SA all 0.417, SA answered 0.833 |
| UltraHorizon | GPT-4o mini | sequential CPI with committed rule induction | exact recovery 100.0%, score 80.0 |
| Unit tests | local | current code | run with `pytest -q tests` |

ScienceAgentBench remains the main open workflow-execution target: the current
universal method needs a stronger multi-file/tool-execution assembler before it
can be treated as a primary result.

## Reproducing the Main Checks

Run tests:

```bash
OPENAI_API_KEY=dummy pytest -q tests
```

The dummy value is enough for offline unit tests that instantiate the OpenAI
client but do not make network calls. Real benchmark runs should use
`OPENAI_API_KEY` or `OPENROUTER_API_KEY`.

DiscoveryBench runner:

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

NewtonBench runner:

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
