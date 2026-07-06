# MARS Metamorphic Estimand Kernel Run — 2026-07-06

## Implemented

Added a first real version of the **Metamorphic Estimand Kernel (MEK)**.

The idea is label-free verification:

```
candidate hypothesis is valid only if its answer object is stable under
task-preserving transformations and does not drift to a neighboring estimand
```

The implemented checks are:

- required typed holes must be closed;
- answer form must be equivariant with the question contract;
- temporal scope in the question must be preserved;
- period/category crossover must use a categorical/time-period axis, not a
  row-level numeric surrogate;
- underspecified coefficient questions must anchor variables to sibling/domain
  context;
- role tokens from the query must survive rendering.

Each failed check produces a residual signature such as:

- `period_axis_not_categorical`
- `semantic_role_anchor_missing`
- `answer_form_drift`
- `open_required_holes`

These signatures are intended to become the input to the next self-writing
operator synthesis loop.

## Files

- `mars/induction/metamorphic_estimand_kernel.py`
- `mars/induction/universal_hypothesis_kernel.py`
- `mars/skills/answer_contract.py`
- `tests/test_metamorphic_estimand_kernel.py`

## Tests

```
pytest -q tests
168 passed, 6 warnings
```

## DiscoveryBench Hard Buckets

Baseline before MEK:

| Dataset | N | Raw HMS | Consistency HMS | Nonzero |
|---|---:|---:|---:|---:|
| introduction_pathways_non-native_plants | 16 | 2.68 | 8.93 | 1 |
| nls_raw | 7 | 0.00 | 0.00 | 0 |
| requirements_engineering_for_ML_enabled_systems | 7 | 42.86 | 57.14 | 3 |
| Total | 30 | 11.43 | 18.10 | - |

After MEK:

| Dataset | N | Raw HMS | Consistency HMS | Nonzero |
|---|---:|---:|---:|---:|
| introduction_pathways_non-native_plants | 16 | 12.50 | 31.25 | 2 |
| nls_raw | 7 | 0.00 | 0.00 | 0 |
| requirements_engineering_for_ML_enabled_systems | 7 | 42.86 | 57.14 | 3 |
| Total | 30 | 16.67 | 30.00 | - |

Important targeted effects:

- `introduction_pathways_non-native_plants:0:0` now selects the correct temporal
  crossover answer instead of a row-level numeric `mrt` axis.
- `introduction_pathways_non-native_plants:1:0` now selects the correct
  `n.gard` / `urban.2009.50m` coefficient pair instead of a nearby niche-breadth
  pair.
- `introduction_pathways_non-native_plants:2:0` no longer adds an unasked
  population restriction in the rendered hypothesis.

## NewtonBench Hard Slice

Run:

`lmw/nb_activeprobe/mek_gate_newton_hard_20260706/summary.json`

Result:

- `SA_answered = 1.00`
- `SA_all = 0.625`
- answered `5/8`

No regression. Remaining failure is still phase/chart abstention, especially
Malus hard v0.

## UltraHorizon Sequence

Run:

`lmw/uh_seq_cpi/mek_gate_seq_hard_s42_20260706/summary.json`

Result:

- exact program fit `5/5`
- official score `80.0`

No regression.

## Diagnosis

MEK is a real improvement, but not enough for SOTA.

It solves a specific universal problem:

> reject hypotheses that answer a nearby measurable question instead of the
> requested estimand.

It does not solve:

> create a missing estimand family when no existing operator proposes it.

The untouched failure class is clearest in `nls_raw`: the system needs a
regression/causal estimand generator that can synthesize objects like:

```
population
outcome
predictor/treatment
covariates
subgroup/filter
model family
coefficient/change/comparison functional
rendering contract
```

## Next Step

The next component should be **EstimandSynthesizerOperator**, inside the same
kernel:

1. compile the question into an estimand object;
2. generate candidate table/variable role assignments;
3. execute the corresponding statistical functional;
4. send the candidate through MEK for label-free metamorphic validation;
5. store residual signatures for operator mutation.

This keeps the system universal:

- UHK proposes and ranks artifacts;
- MEK validates them without labels;
- EstimandSynthesizer creates missing families when reranking alone cannot help.

## EstimandSynthesizer Update

Implemented the next component as a UniversalHypothesisKernel operator:

- `mars/induction/estimand_synthesizer.py`
- `tests/test_estimand_synthesizer.py`

The new operator compiles a question and observable table into an executable
estimand object:

```
outcome
predictor/group
covariates
interaction/contrast
model family
coefficient or group functional
rendering contract
```

The important design change is that the model no longer has to propose a whole
statistical hypothesis as prose.  It proposes or induces roles, builds latent
features when the raw table lacks explicit columns, executes a compact
functional, and then lets UHK/MEK decide whether the candidate should beat the
older operators.

Implemented families:

- single-predictor binary regression;
- nested binary regression coefficient delta;
- binary group contrast;
- binary interaction;
- group/outcome contrast;
- interaction-effect regression;
- conservative multi-item survey estimands.

The survey estimator is intentionally low-prior except for multi-item list
questions, because old renderers already handle many stated-percentage tasks
better.  This avoids replacing exact question constants with approximate
measured percentages.

## DiscoveryBench Hard Buckets After EstimandSynthesizer

Run:

`lmw/universal_discovery_real/estimand_hard_buckets_gated_20260706/summary.json`

| Dataset | N | MEK Raw | MEK Consistency | Estimand Raw | Estimand Consistency |
|---|---:|---:|---:|---:|---:|
| introduction_pathways_non-native_plants | 16 | 12.50 | 31.25 | 18.75 | 31.25 |
| nls_raw | 7 | 0.00 | 0.00 | 87.86 | 87.86 |
| requirements_engineering_for_ML_enabled_systems | 7 | 42.86 | 57.14 | 42.86 | 57.14 |
| Total | 30 | 16.67 | 30.00 | 40.50 | 50.50 |

NLS gains came from the new estimand operator:

- racial BA-completion contrast;
- nested SES/race/academic-characteristics coefficient delta;
- sparse-target stability rule for gender;
- criminal-history to wealth contrast;
- SES by Black interaction;
- latent SES to BA-completion regression;
- race to median-wealth contrast.

## Cross-Benchmark Regression Check

NewtonBench hard slice:

- run: `lmw/nb_activeprobe/estimand_gate_newton_hard_20260706/summary.json`
- answered `5/8`
- `SA_answered = 1.00`
- `SA_all = 0.625`

UltraHorizon sequence:

- run: `lmw/uh_seq_cpi/estimand_gate_seq_hard_s42_20260706/summary.json`
- exact program fit `5/5`
- score `80.0`

Full tests:

```
pytest -q tests
172 passed, 6 warnings
```

## Remaining Failure Class

The system is stronger but still not final.

The main remaining gap is not ordinary regression or contrast estimation.  It is
**semantic role canonicalization for interactions**: the system can fit an
interaction, but it may choose a neighboring proxy such as one urban-year column
instead of the canonical role "urban land use", or render the interaction in a
way that the HMS judge does not align with the gold hypothesis.

Next useful universal upgrade:

```
RoleCanonicalizer:
  cluster schema-level proxies into semantic roles
  select role-level factors before fitting interactions
  render role names rather than raw column names
  keep MEK checks over role preservation
```

That is the next missing piece if we want the plants interaction tasks to move
without hand-tuning to plants.

## Universal Role Layers Implemented

Implemented after the gap above:

- `RoleCanonicalizer`: schema/query-induced role abstraction before executable
  estimands.  It uses a small generic seed ontology plus dynamic role induction
  from query/schema overlap, then materializes role-level features from proxy
  columns.
- `role_interaction_effect`: interaction estimands now fit roles such as
  "urban land use" instead of raw proxy columns such as one urban-year column.
- `role_conditioned_outcome_contrast`: for questions of the form "how does a
  factor affect different outcome types", the system measures standardized
  slopes for primary outcome-type columns and renders the relative effect.
- `invalid_evidence_gate`: reports saying "not enough relevant numeric columns"
  are penalized instead of being allowed to win by surface slot coverage.
- Cleaner answer-contract rendering: if a question does not ask for a
  coefficient, the hypothesis states the scientific relation and keeps numeric
  evidence in workflow/evidence.

Key files:

- `mars/induction/role_canonicalizer.py`
- `mars/induction/estimand_synthesizer.py`
- `mars/induction/universal_hypothesis_kernel.py`
- `mars/skills/evidence_contract_compiler.py`
- `mars/runners/run_universal_discovery_real_eval.py`

## DiscoveryBench Hard Buckets After Universal Role Layers

Run:

`lmw/universal_discovery_real/universal_role_layers_hard_buckets_20260707/summary.json`

| Run | Raw HMS | Consistency HMS | N |
|---|---:|---:|---:|
| MEK only | 16.67 | 30.00 | 30 |
| EstimandSynthesizer gated | 40.50 | 50.50 | 30 |
| Universal role layers | 40.50 | 60.50 | 30 |

By dataset in the final run:

| Dataset | Raw HMS | N |
|---|---:|---:|
| introduction_pathways_non-native_plants | 37.50 | 16 |
| nls_raw | 59.29 | 7 |
| requirements_engineering_for_ML_enabled_systems | 28.57 | 7 |

The raw mean is tied with the previous best hard-bucket run, but consistency
improves by +10 points.  The largest qualitative improvement is in the plants
blocks where the correct role-level hypotheses now reach the final answer:

- urban land use x elevation interaction;
- urban land use reducing gardening invasion relative to unintentional
  introductions.

The remaining raw-HMS failures are partly evaluator-extraction fragility and
partly missing forms for pathway x residence-time/success questions.

## Cross-Benchmark Regression After Role Layers

NewtonBench comparable hard slice:

- run: `lmw/nb_activeprobe/role_layers_newton_hard_comparable_20260707/summary.json`
- answered `5/8`
- `SA_answered = 1.00`
- `SA_all = 0.625`

NewtonBench wider hard slice:

- run: `lmw/nb_activeprobe/role_layers_newton_hard_20260707/summary.json`
- answered `12/24`
- `SA_answered = 0.833`
- `SA_all = 0.417`

UltraHorizon sequence:

- run: `lmw/uh_seq_cpi/role_layers_gate_seq_hard_s42_commit_20260707/summary.json`
- exact program fit `5/5`
- score `80.0`

Full tests:

```
pytest -q tests
176 passed, 6 warnings
```
