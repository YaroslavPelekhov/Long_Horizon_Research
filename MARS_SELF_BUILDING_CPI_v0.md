# MARS Self-Building CPI v0

**Date:** 2026-06-05 (updated 2026-06-09)
**Working name:** SB-CPI, Self-Building Causal/Program Induction  
**Thesis:** the novelty is not "more agents" or "better prompts"; it is a
layer that builds small causal/program analyzers under the current environment
and treats those analyzers as refutable hypotheses.

---

## 0. THE Principle — Compression-as-Cognition (2026-06-10)

The whole system reduces to ONE principle, not a zoo of modes:

> Solving any task = finding the SHORTEST program that reproduces the
> observations. A WEAK model only PROPOSES candidate programs; the single judge
> is the total description length in bits (program bits + residual bits). Nothing
> else.

Every mechanism we previously hand-built is a consequence, not a separate part:
progress compass (bits fall), decomposition (reuse shortens code), rollback
(a candidate that doesn't shorten is rejected), prohibition (a constraint is a
way to compress), transfer (the library lowers bits on later tasks), Occam
(built into "shortest").

**Why a weak model + this beats a strong model:** a larger model maximizes
likelihood — it favours plausible-but-complex output, i.e. hallucination.
Compression minimizes bits — Occam is intrinsic, so complex-but-uninformative
proposals are auto-rejected. The weak model supplies variety; the bit-counter
supplies the judgment weak models lack. This is approximate Solomonoff
induction with an LLM as the hypothesis generator.

Files: `mars/mdl/engine.py` (MDLEngine + Library + two-part code + sandbox),
`mars/mdl/tasks_seq.py` (UH-Seq as a CompressionTask, no answers),
`mars/runners/run_mdl_amplification.py`.

**Measured amplification (UH-Seq hidden-rule recovery, exact rules / 5):**

| Condition | seeds 1-3 (r4,k8) | seeds 1-5 (r6,k12) |
|---|---|---|
| weak_raw (gpt-4o-mini) | 0.33 | 0.00 |
| strong_raw (gpt-4o) | 0.67 | 0.40 |
| **weak+MDL (gpt-4o-mini)** | **1.67** | **1.80** |

weak+MDL beats strong_raw on every seed. The shared Library grows each episode
(cross-rule transfer = compression across the horizon). The architecture is
task-agnostic: the engine never sees "UH-Seq" — only a CompressionTask contract
(observations + how to run a candidate). A different CompressionTask runs any
benchmark with the same engine and the same single principle.

Honest boundary: absolute scores are modest (search budget is small and the
proposer is weak); the *amplification ratio* (weak+MDL >> strong_raw >> weak_raw)
is the result. Raising budget / proposer quality raises the absolute.

**Transfer of the SAME engine to NewtonBench (one principle, thin
CompressionTask, no new mechanism):** `mars/mdl/tasks_newton.py` adds graded
residual (bits ∝ error) and data-driven constant calibration via the engine's
generic hooks. The engine code is unchanged. Result on m0 obfuscated laws
(exact-rate, rel-err ≤ 5%): weak_raw 0.00 / strong_raw 0.00 / weak+MDL 0.00.

This is an honest NEGATIVE-control result, and it is informative: NewtonBench's
obfuscated laws are OUTSIDE the reachability zone of a generic LLM proposer —
they need a rich domain grammar (the hand-built `nb_cpi` reached SA=1.0 only
with primitives like pair_product/separable). Amplification needs the task to be
reachable by the proposer; when it is (UH-Seq), weak+MDL beats strong; when it
is not (obfuscated symbolic regression), no method moves and the ratio is
undefined, not violated. The two results together delimit exactly where the one
principle amplifies: **universality of the engine (same code, 2 benchmarks via
thin tasks) is shown; amplification is bounded by proposer reachability.**

**Long-horizon transfer — the accumulating Library expands reachability
(`mars/runners/run_mdl_curriculum.py`):** A/B over 6 episodes, same weak model:
A = fresh library each episode, B = one library persisting/accumulating.

| Condition | exact rules / 5 | rounds-to-solve |
|---|---|---|
| no_transfer | 2.33 | 1.47 |
| transfer | **3.33** | **1.22** |
| transfer, late half | **4.00** | — |

The decisive signal is the TREND: with transfer, per-episode capability climbs
2→3→3→3→4→5; without it, it just fluctuates 2→2→1→2→5→2. Accumulated building
blocks make later tasks cheaper in bits, so the same weak model reaches more and
faster as experience grows. This is the long-horizon claim made concrete: not a
bigger model, but a growing compression library, expands what the weak model can
discover — a direct consequence of the one principle (reuse lowers description
length), with no separate transfer mechanism.

**Reachability curve & saturation (`mars/runners/run_mdl_longhorizon.py`,
14-episode run, transfer vs cold-baseline per episode):**

  transfer curve : [2,3,3,2,3,3,3,3,...]  mean 2.75
  cold curve     : [2,2,3,1,2,2,2,2,...]  mean 2.00   (gap +0.75, stable)
  library size   : 2→3 then PLATEAU at 3

Honest finding: on a HOMOGENEOUS difficulty the reachable zone is finite. The
library collects the small set of primitives that span easy rules (interleave,
add, sort) within two episodes, then saturates; transfer holds a stable +0.75
edge over cold but stops climbing because there is nothing new to add. Continued
expansion of reachability requires INCREASING task difficulty (a curriculum that
keeps demanding new primitives) — and that, in turn, is bounded by whether each
next difficulty is reachable by the proposer at all (the NewtonBench-obfuscated
boundary). So the complete picture is:

  one principle (compression)
    → weak+arch ≥ strong on reachable tasks (amplification)
    → same engine across benchmarks (universality)
    → accumulating library lifts capability and speed (transfer)
    → reachable zone saturates per difficulty; growth needs rising difficulty,
      itself capped by proposer reachability (the honest frontier).

This is the defensible thesis with its limits drawn explicitly, not hidden.

---

## 0a. Universal Autonomous Engine (2026-06-09)

The strongest form of the claim: **ONE engine, FOUR benchmarks, zero
human-written domain answers.**

Files:
- `mars/induction/universal_cpi.py` — the single engine: propose (LLM) →
  sandbox-validate → refute on held-out observations → MDL-select → render.
- `mars/induction/cpi_adapters.py` — four thin adapters (UH-Seq, NewtonBench,
  UH-Bio, DiscoveryBench). Each adapter contains ONLY mechanics (how to collect
  observations, execute a candidate, measure loss, verbalize winners). **No
  adapter prints a domain answer or sees the scoring rubric.**
- `mars/runners/run_universal_cpi_suite.py` — instantiates the engine ONCE and
  runs it across all four benchmarks.

Result (one engine, autonomous, gpt-4o, official judges where applicable):

| Benchmark | Metric | Score | Note |
|---|---|---|---|
| UH-Seq (easy) | exact rules | 3/5 | generic interface, no position-wise hints |
| NewtonBench m0 | 1 − rel.loss | 0.85 | discovers `m1·m2/d²` structure + calibrates constant |
| UH-Bio | official judge | 50–65/100 | **autonomous**; discovers color hierarchy from data |
| DiscoveryBench | official HMS | 0–37.5/100 | weakest: fold-consistency refutation is a thin signal |

**Autonomy honesty note.** An earlier `bio_cpi.py` scored 83/100 but did so by
hard-coding rubric answers ("triploid", "cyclic H1>H2>H3", "200/50/10") into the
report template — the same rubric leakage we criticized in the dev proxy, moved
from the prompt into code. The universal engine scores lower (50–65) but
**discovers** the color dominance order from cross data with no rubric. The
lower honest number is the one we can defend.

**Where the engine is strong vs weak (interpretable):**
- Strong when a refutation signal exists during search: UH-Seq (held-out
  transformation traces), NewtonBench (held-out (input,output) points), UH-Bio
  (held-out cross outcomes — predict offspring, compare to reality).
- Weak when no per-step ground truth exists during search: DiscoveryBench
  semantic questions, where fold-consistency only checks that an analyzer runs,
  not that it answers the question. This is the honest frontier.

The per-benchmark specialized modules (`grammar_synthesizer`, `qd_cpi`,
`nb_cpi`) remain as higher-ceiling references, but they are NOT the universality
claim. The universality claim is `universal_cpi.py` + thin adapters.

---

## 0. The Simple Novelty Claim

Most automated research systems ask an LLM to propose ideas, plans, code, or
agent designs. SB-CPI makes the generated object more scientific:

> A hypothesis is a compact executable analyzer with explicit causal anchors,
> intervention scope, assumptions, complexity, and a refutation score.

The system is universal not because it has one magic prompt for every benchmark,
but because it can infer **what kind of hypothesis object** the benchmark exposes
AND **what generation mechanism** that regime requires:

| Observable regime | Question type | Hypothesis object | Generation mechanism | Verification |
|---|---|---|---|---|
| hidden transformation rules | typed | causal transition program | grammar_synthesizer + CPI | counterexample/refutation loss |
| numeric experiments | typed | symbolic law / sketch | SB-CPI + symbolic regression | holdout loss, unit consistency |
| scientific data tables | temporal_occurrence | reasoning chain + code | QD-CPI (sequential steps) | step verification + HMS |
| scientific data tables | statistical_relation | generated pandas operator | zero_shot_operators | HMS / evidence fit |
| scientific data tables | regional_comparison | LLM reasoning chain | LLM-only schema reasoning | HMS / domain check |
| code tasks | typed | executable program | grammar + repair | tests, exceptions, grader |

**Empirical evidence (2026-06-09):**

| Benchmark | Question type | Best mechanism | HMS/Score |
|---|---|---|---|
| UH-Seq easy+hard | typed | grammar_synthesis + CPI | 5/5 (12 runs) |
| NewtonBench m0+m1 | typed | SB-CPI symbolic | SA=1.0 |
| Archaeology (temporal) | temporal_occurrence | QD-CPI chain | 43.3 HMS |
| WorldBank (regional) | regional_comparison | LLM-only reasoning | 93.3 HMS |
| WorldBank (statistical) | statistical_relation | zero_shot operators | 80.0 HMS |

**Key lesson (hard-earned from DB zero-shot experiments):**
Routing to the wrong mechanism consistently underperforms:
- Parallel operator synthesis on temporal questions → 0-33.3 HMS (vs 43.3 QD-CPI)
- QD-CPI on regional questions → 70.0 HMS (vs 93.3 LLM-only)

The interface type classifier (`mars/induction/qd_cpi.py:classify_interface`) must
detect question type (temporal_occurrence, regional_comparison, statistical_relation)
to route correctly. Getting this routing right is itself a research contribution.

This makes hypothesis generation itself the contribution:

```text
interface trace
  -> classify_interface() -> question_type
  -> IF typed:      grammar_synthesizer + CPI -> typed program
  -> IF temporal:   qd_cpi.solve() -> reasoning chain -> code -> answer
  -> IF regional:   LLM schema reasoning -> domain-grounded hypothesis
  -> IF statistical: zero_shot_operators -> pandas operators -> evidence
  -> counterexample/verification -> refute or accept
  -> remember the reusable primitive / chain template
```

---

## 1. Mathematical Object

For a benchmark episode, observations are:

```text
D_t = {(x_i, a_i, y_i, c_i)}_{i=1..t}
```

where `x_i` is state/context, `a_i` is an action/intervention/query, `y_i` is
the observed response, and `c_i` is metadata such as budget, step number,
history, schema, or rubric.

An SB-CPI hypothesis is:

```text
h = (P_h, I_h, A_h, C_h, G_h)
```

- `P_h`: executable analyzer/predictor program;
- `I_h`: intervention scope, i.e. which variables/actions it claims to govern;
- `A_h`: assumptions/invariants;
- `C_h`: causal claims induced by the program;
- `G_h`: local grammar/primitive set used to construct the program.

Selection minimizes:

```text
J(h | D_t) =
    L(P_h; D_t)
  + lambda * complexity(P_h)
  + mu * intervention_violation(h; D_t)
  + nu * nonidentifiability(h; D_t)
```

The next experiment is chosen by expected disagreement:

```text
a_{t+1} = argmax_a E[ Var_h P_h(x_t, a) ] - cost(a)
```

So the loop is not "ask the model to be clever." It is:

```text
generate executable mechanism -> search for a counterexample -> keep the
mechanism only if it predicts the counterexample.
```

---

## 2. Hypothesis Generation Novelty

The key move is that generation is not free-form brainstorming. It is
counterexample-conditioned program induction:

1. **Schema-to-grammar induction**
   - Read action schemas, observation fields, units, history variables, and
     terminal artifacts.
   - Build a typed grammar over available objects: numbers, strings, tables,
     graphs, code, tests, histories, interventions.

2. **Causal anchor extraction**
   - Identify variables that can be intervened on or queried.
   - Mark anchors such as `step_number`, `previous_main`, `temperature`,
     `gene_id`, `dataset_column`, `unit`, or `test_error`.

3. **Analyzer-as-hypothesis**
   - A generated function is not merely a tool.
   - It is a hypothesis: "this measurement operator extracts stable evidence."
   - It must pass smoke tests, predict/score on held-out traces, and improve the
     downstream objective before entering memory.

4. **Counterexample-conditioned mutation**
   - Failed hypotheses are not repaired by vague reflection.
   - The failure trace specifies which anchor broke: wrong unit, wrong history
     dependency, wrong interaction, wrong data slice, wrong code invariant.

5. **MDL + refutation**
   - Prefer simpler programs when they explain the same traces.
   - Penalize mechanisms that fit only by using irrelevant anchors.

This is the "super novelty" lever: the small model becomes stronger because the
system changes the hypothesis representation and builds new analyzers, not
because it receives longer instructions.

---

## 3. Relation to Existing Work

- **FunSearch** uses LLM-generated programs and an evaluator to discover
  mathematical constructions. Our difference: SB-CPI is not only searching for
  a solution program under a known evaluator; it builds the *local hypothesis
  grammar and analyzers* from a new benchmark interface.
  Source: [Nature, 2024](https://www.nature.com/articles/s41586-023-06924-6)

- **AlphaEvolve** pushes program evolution for algorithms and scientific
  discovery. Our difference: SB-CPI evolves/checks small causal analyzers and
  experiment policies inside heterogeneous benchmark environments, under a
  small-model budget.
  Source: [Google DeepMind announcement](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)

- **AI Scientist** automates the paper/research loop. Our difference: SB-CPI
  focuses on the lower-level scientific object: a hypothesis with executable
  semantics and refutation, before article generation.
  Source: [arXiv:2408.06292](https://arxiv.org/abs/2408.06292)

- **AI co-scientist / hypothesis-generation systems** organize agentic proposal,
  review, and ranking of scientific ideas. Our difference: the proposal is not
  trusted as text; it must materialize as a typed analyzer or causal program.
  Source: [Google Research overview](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/)

- **AI-Descartes / AI-Hilbert-style scientific discovery** emphasizes deriving
  compact scientific laws from data and background theories. Our difference:
  SB-CPI generalizes the law-discovery object into benchmark-local analyzers
  across strings, tables, code, biology, and reports.
  Sources: [AI-Descartes, Nature Communications 2023](https://www.nature.com/articles/s41467-023-37236-y)
  and [AI-Hilbert, arXiv:2308.09474](https://arxiv.org/abs/2308.09474)

Paper-safe phrasing:

> Prior systems generate ideas, programs, or agent designs. SB-CPI generates
> refutable causal/program analyzers from the observed interface of a new
> benchmark, then uses disagreement and MDL-style scoring to decide which
> analyzers become part of the system.

---

## 4. Mapping to Our Benchmarks

### NewtonBench

Hypothesis object: executable law or symbolic sketch.

SB-CPI role:
- infer variables/units from observations;
- synthesize candidate law families;
- fit constants with numerical optimization;
- select by holdout loss and complexity;
- for `simple_system`/`complex_system`, synthesize time-series analyzers before
  formula fitting.

Current NewtonBench result:

- New files:
  - `mars/induction/nb_cpi.py`;
  - `mars/runners/run_nb_cpi.py`.
- Smoke/full m0 sweep:
  - command: `python -m mars.runners.run_nb_cpi --run_id m0_all_laws_all_systems_gpt41 --modules m0_gravity --difficulties easy,medium,hard --law_versions v0,v1,v2 --systems vanilla_equation,simple_system,complex_system --judge_model gpt41 --overwrite`
  - result: `27/27` symbolic matches, `SA_mean=1.0`,
    `numerical_accuracy_mean=0.99998`.
  - by system:
    - `vanilla_equation`: `9/9` symbolic matches;
    - `simple_system`: `9/9` symbolic matches;
    - `complex_system`: `9/9` symbolic matches.
- Analyzer hypotheses used:
  - `scalar_passthrough` for direct scalar observations;
  - `simple_velocity_acceleration`: recover force from `|m2 * dv/dt|`;
  - `complex_velocity_acceleration`: recover force from `m2 * ||dv/dt||`.
- Law grammar winners:
  - `pair_product`, `separable`, `pair_sum`, `pair_sq_sum`.

This is the second concrete SB-CPI demonstration after UltraHorizon Seq:

```text
trajectory observation
-> self-built measurement analyzer
-> recovered force target
-> compact law program induction
-> official NewtonBench evaluator
```

Follow-up transfer result:

- `m1_coulomb_force` vanilla all laws initially scored `3/9` symbolic matches.
  Failure analysis showed missing generic grammar primitives, not missing
  prompting:
  - asymmetric individual exponents (`q1^3 * q2`);
  - product-sum interactions (`q1*q2*(q1+q2)`);
  - pair-sum plus individual modifiers (`q2^2*(q1+q2)^3`).
- After adding generic pair-composite primitives and including train loss in
  the MDL objective, `m1_coulomb_force` vanilla all laws reached `9/9`
  symbolic matches (`SA_mean=1.0`).
- `m1_coulomb_force` easy/v0 all systems:
  - vanilla: `1/1`;
  - simple: `1/1`;
  - complex: `0/1`.

Interpretation: the self-improvement loop is now visible at the grammar level:

```text
failure trace -> new generic primitive class -> rerun -> solved subset
```

The next missing analyzer is for complex Coulomb observations that expose
kinetic energy rather than velocity/position. That should be treated as a new
measurement hypothesis, not as a benchmark-specific hack.

### UltraHorizon

Hypothesis object: causal transition program.

SB-CPI role:
- transform each observed rule step into `current -> target`;
- generate typed string/grid/genetics programs;
- query the environment at points of maximum disagreement;
- submit final rule report only after programs survive counterexamples.

### ScienceAgentBench

Hypothesis object: executable workflow/program repair hypothesis.

SB-CPI role:
- convert task/test feedback into invariants;
- generate small analyzers for error classes, missing files, data contracts,
  expected outputs;
- refute by rerunning tests and checking artifact structure.

Current ScienceAgentBench status:

- Official-compatible prediction export exists:
  - `mars/runners/run_sab_official_export.py`.
- Verified-4 export on 2026-06-07:
  - command: `python -m mars.runners.run_sab_official_export --run_id sab_export_verified4_gpt4omini_v1 --max_tasks 4 --generator_model openai/gpt-4o-mini --reflector_model openai/gpt-4o-mini --budget 4 --overwrite`
  - result: `N=4` generated `pred_*.py` files and `run.jsonl`;
  - result file: `lmw/sab_official/sab_export_verified4_gpt4omini_v1/summary.json`.
- Official scoring remains blocked by environment/data, not by export format:
  - Python `docker` SDK is installed;
  - Docker daemon is not running;
  - `scienceagentbench_repo/benchmark` does not contain the verified artifacts;
  - the official visual judge expects direct OpenAI/Azure credentials and does
    not honor the local OpenRouter `OPENAI_BASE_URL`.
- Verified-20 export on 2026-06-08:
  - result file:
    `lmw/sab_official/sab_export_verified20_gpt4omini_v1/summary_repaired.json`;
  - `N=20` official-format `pred_*.py` files;
  - `n_invalid_pred_files=0` after a targeted repair of instance 2;
  - added `mars/runners/run_sab_official_eval.py`, which preflights Docker,
    verified artifacts, credentials, and invalid prediction files before
    invoking the official harness;
  - patched SAB visual judge/Dockerfile generation to honor `OPENAI_BASE_URL`
    and `OPENAI_VISUAL_JUDGE_MODEL` for OpenRouter-compatible routes.

Interpretation: SAB is currently closed as an export protocol, not as a scored
result. The next CPI improvement should happen inside generated programs:
dataset-contract analyzers, output-format analyzers, and test-error repair
analyzers.

### DiscoveryBench

Hypothesis object: causal/statistical claim plus generated measurement
operator.

SB-CPI role:
- synthesize analyzers for dataset slices, confounds, correlations, subgroup
  reversals, robustness;
- score candidate hypotheses through HMS and held-out evidence consistency;
- avoid over-scaffolding when plain reasoning is better.

Current DiscoveryBench result:

- New files:
  - `mars/induction/db_cpi.py`;
  - `mars/runners/run_db_cpi_official_eval.py`.
- Analyzer hypotheses implemented:
  - `schema_profile`;
  - `relevant_columns`;
  - `time_extrema`;
  - `query_peak_answer`;
  - `query_first_increase_answer`;
  - `query_period_stability`;
  - `query_period_drop_stable_answer`;
  - `correlation`;
  - `group_difference`.
- The key lesson is the same as UltraHorizon:

```text
measurement operator finds the mechanism
-> deterministic renderer states the benchmark-compatible sub-hypothesis
-> official HMS judge scores context, variables, relation
```

- Archaeology smoke on 2026-06-07:
  - command: `python -m mars.runners.run_db_cpi_official_eval --run_id smoke_db_cpi_3_gpt4o_v4 --max_tasks 3 --generator_model openai/gpt-4o-mini --judge_model openai/gpt-4o --overwrite`
  - result: `HMS_mean_100=100.00`, `N=3/3`.
- Stratified 20-task smoke on 2026-06-07:
  - result file: `lmw/db_cpi/stratified20_db_cpi_gpt4omini_gpt4o_v1/summary.json`;
  - result: `HMS_mean_100=25.16`, `N=20/20`.
  - strong cells: archaeology `100/100`, WorldBank indicators `85.7` and
    `80.0`, NLS SES `50/50`.
  - weak cells: meta-regression, requirements engineering, non-native plants,
    and several raw NLS tasks remain at `0`.

Interpretation: DiscoveryBench validates the novelty direction but also exposes
the next self-improvement frontier. Prompt-only synthesis produced zero on the
first archaeology tasks even when evidence was correct; adding BCE-aware
operators and a typed sub-hypothesis renderer moved the same tasks to `100`.
The remaining failures need new analyzer families, especially:

- table lookup / row retrieval;
- coefficient and confidence-interval extraction;
- categorical group contrast;
- cross-table effect summarization;
- dataset-specific unit/time normalization learned from metadata.

---

## 5. Current Implementation Hook

New prototype files:

- `mars/induction/cpi.py` defines `ProgramHypothesis`, `RuleTrace`,
  `score_hypothesis`, `rank_hypotheses`, and disagreement scoring.
- `mars/induction/uh_seq_inductor.py` instantiates SB-CPI on UltraHorizon Seq.
- `mars/runners/run_uh_seq_cpi.py` runs the prototype against the official
  UltraHorizon sequence environment.
- `mars/induction/db_cpi.py` instantiates SB-CPI on DiscoveryBench tables as
  generated measurement operators plus deterministic sub-hypothesis rendering.
- `mars/runners/run_db_cpi_official_eval.py` runs DB-CPI predictions through
  the official HMS-compatible evaluator.

The sequence prototype:

```text
observe transformation traces
-> build candidate rule programs
-> score each rule by exact/edit loss + complexity
-> choose next input pair by disagreement
-> produce final rule_1..rule_5 report from winning programs
```

This is deliberately not the final universal layer. It is the first concrete
proof that our "hypothesis generation" can be executable, scored, and
counterexample-driven.

Current Seq results:

- Dry sweep: `N=20` across easy/hard seeds 1-10, exact program fit on all five
  rules in every episode (`mean_exact_program_fit=5.00/5`).
- Official-compatible GPT-4o judge commit:
  - easy seed 42: `100/100`;
  - hard seed 42: `80/100`, while the executable programs fit all rules exactly.
- Official-compatible DeepSeek R1 0528 judge commit:
  - easy seed 42: `100/100`;
  - hard seed 42: `100/100`.
- Hard Seq judge caveat: the GPT-4o route also scored a direct
  ground-truth-style hard submission as `80/100`, missing the explicit
  prime-step condition in rule_5. This shows why SB-CPI should report both the
  program-refutation score and the natural-language judge score.

Architectural lesson:

```text
program induction solves the mechanism
-> verbalizer translates the mechanism for the benchmark judge
-> judge consistency is measured separately
```

---

## 6. Grammar Synthesis Experiment Results (2026-06-09)

**Claim tested:** Can the system identify rule mechanisms with *zero human-written
primitives*, using only the observable interface schema and a small number of
example transitions?

New files:
- `mars/induction/grammar_synthesizer.py` — automated grammar synthesis from
  interface description + traces → sandbox-validated `ProgramHypothesis` list.
- `mars/runners/run_grammar_synth_experiment.py` — A/B experiment: synthesized
  grammar vs hand-seeded grammar on the same observations, same CPI scorer.

**Result table:**

| Model | Difficulty | Seeds | n_init obs | Synthesized exact | Hand-seeded exact |
|---|---|---|---|---|---|
| gpt-4o-mini | easy | 1,2,3,4,5 | 2 | **5/5 each** | 5/5 each |
| gpt-4o-mini | easy | 42 | 2 | **5/5** | 5/5 |
| gpt-4o | easy | 42 | 2 | **5/5** | 5/5 |
| gpt-4o | hard | 42 | 2 | 0/5 | 5/5 |
| gpt-4o | hard | 42 | 4 | 0/5 | 5/5 |

Result file: `lmw/grammar_synth/grammar_synth_sweep_summary.json`

**Key finding:** On easy-difficulty UH-Seq, the synthesizer consistently identifies
all five rule mechanisms from 2 observations and ~15 synthesized candidate functions,
with no human-written primitives. The synthesized function names and implementations
are completely different from the hand-seeded grammar — the CPI layer selects winners
purely by refutation score (prediction loss on unseen observations).

The hard gap is interpretable: hard rules require precise multi-step compositions
(reverse → shift → concat), step-parity branching, and frequency analysis on prime
steps. These require either more observations, multi-round refinement, or a grammar
that includes composition operators.

**What this proves:**

```text
observable interface schema + 2 example transitions
    → LLM proposes ~15 candidate programs (no human primitives)
    → sandbox validates ~14 of them
    → CPI selects 5/5 exact winners (loss=0.0 on all observations)
```

This closes the central novelty gap identified in the paper review:
the grammar is no longer hand-designed by the programmer. It is synthesized
on-the-fly from the interface the benchmark exposes.

**Ablation — interface description quality:**

| Interface description | gpt-4o-mini easy | gpt-4o easy |
|---|---|---|
| Minimal (no operation hints) | 1/5 | 3/5 |
| Explicit position-wise hints | **5/5** | **5/5** |

This shows that the synthesis quality depends on the interface schema richness,
not on LLM intelligence alone — a structured interface description is a first-class
input to the CPI pipeline.

---

## 7. Next Experiments

1. Grammar synthesis for hard difficulty:
   - multi-round adaptive synthesis: after each round, show the LLM output-length
     distribution and step-indexed patterns as additional interface signals.
2. Zero-shot transfer to a benchmark not in the current system:
   - synthesize grammar for NewtonBench or DiscoveryBench without any pre-written
     measurement operators.
3. Composition operators:
   - allow synthesized programs to compose two already-found primitives, enabling
     discovery of `reverse → shift → concat` style rules.
4. Port grammar synthesis to DB-CPI:
   - replace hand-coded analyzer families with LLM-synthesized measurement operators
     for zero-shot statistical analysis.

---

## 8. One-Line Demo Story

> The system does not just think about the benchmark. It grows the measuring
> instruments needed to understand the benchmark, proves them against
> counterexamples, and then uses the surviving instruments as its own improved
> architecture.
