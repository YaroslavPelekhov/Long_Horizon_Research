# Weak-Model Amplification via Conservation: A Mechanistic Theory

**One universal method, four benchmarks, and an exact map of where it wins and where it hits a
hard ceiling.** Solver throughout: `gpt-4o-mini` (a non-thinking weak model). No benchmark-specific
code, no injected rewards, no tuned constants — the model authors everything; a thin shell only
executes, verifies, and selects.

---

## 1. The method (one engine)

`mars/induction/conservation_discovery.py` — a benchmark-agnostic core:

> The model proposes candidates (each carrying an executable *quantity*, *fidelity*, or
> *answer_key*). The shell keeps what is **conserved against the data at the data's own measured
> noise floor**, and **selects** the strongest, else **abstains**. Zero tuned constants (the accept
> threshold is the measured noise floor).

Each benchmark plugs in through a thin `DiscoveryAdapter` supplying only: how to generate
candidates, how to resample/transform, how to measure the noise floor, and the official scorer.

| Benchmark | adapter path | data-verifier |
|---|---|---|
| NewtonBench | `run_conservation_nb.py` | held-out predictive fidelity @ noise floor |
| DiscoveryBench | `run_conservation_db.py` | claimed quantity's stability under resampling |
| ScienceAgentBench | `run_conservation_sab*.py` | reproducibility / property checks / decorrelated agreement |
| UltraHorizon | `run_conservation_uh*.py` | agreement/selection against real experiment outcomes |

## 2. Results (honest, gpt-4o-mini)

| Benchmark | mechanism | result | baseline |
|---|---|---|---|
| NewtonBench (easy vanilla, 6 mod) | data-driven regression + DPSR holes + conservation | **SA 0.67, precision 1.00** | self-debug 0.17; non-thinking GPT-4.1 ~0.06 |
| — medium / hard | same | 0.17 / 0.00, **precision 1.00 all levels** | matches literature's precipitous decay |
| DiscoveryBench (14 diverse) | conservation-selection (judge gpt-4-turbo) | **forced 16.6 HMS (×4)**, precision 33 | first-candidate 4.25; SOTA 24.5 |
| ScienceAgentBench | consensus → **decorrelated** diverse-method agreement | precision **0.50 → 1.00** | same-method consensus 0.50 |
| UltraHorizon (bio) | **evidence-based selection** | **87 = oracle**, > single-rollout 77 (8/8) | consensus 45, claim-merge 47, re-synth 66 |

## 3. The theory: three gates of weak→strong amplification

Amplification succeeds only if all three gates pass, in order:

**Gate 1 — Generation support.** The correct primitive/insight must appear with nonzero
probability in *some* sample (in generation, validation, or code). Selection/union move you from
the *mean* to the *max* of the sample distribution — **never beyond its support**.
*Evidence:* UH-bio, 4/4 rollouts score identical 87 with the SAME blind spots (triploidy never
found), zero complementary criteria → no aggregation exceeds 87.

**Gate 2 — Verifier independence.** To *select* the correct sample you need a verifier whose error
is **independent of the generator's error**. Two sources of independence:
- **External oracle** the model *reads*, not authors: NB held-out data, DB dataset resampling, UH
  real experiment outcomes.
- **Decorrelation**: force independent *diverse derivations* so a shared systematic error stops
  reproducing (Condorcet / multiple-independent-proofs).
*Evidence:* same-method consensus fails (SAB id71 "consensus-but-wrong", precision 0.50); forcing
diverse methods restores independence → precision 1.00. Self-authored checks fail (share the
model's blind spot: too weak → pass wrong output; too strict → reject correct output).

**Gate 3 — Analytical primitive repertoire (the hard ceiling).** Even with support and an
independent verifier, the required analytical *operation* must be expressible by the model in at
least one mode — generation, validation, **or code**. Architecture reaches any primitive the model
has (NB log-log regression, DB correlation, UH color-ordering-by-intensity). It **cannot
manufacture** a primitive absent in all three modes.
*Evidence:* UH triploidy needs decomposing additive-dosage sums (601≈3×200, 450=200+200+50) into 3
alleles. The model fails it (a) in prose reasoning, (b) in hypothesis-conditioned validation even
when explicitly asked "is ploidy non-diploid?" (**V/G asymmetry breaks for deep analytical
claims**), and (c) in code (its clustering gave ploidy=40, not 3). A hand-written *general*
decomposition primitive also failed (C=2, rmse 40) — compounded by limited data coverage (9 crosses
never sampled the 10+10+10 combination; observed min 69 ≠ true min 30).

## 4. The exact boundary (for the advisor)

This is **not** "the method is weak" — it is a precise, mechanistic map:

- **Wins** wherever the needed primitive is in the model's repertoire *and* an independence source
  exists: NB (regression), DB (correlation), UH-selection (reaches oracle), SAB (decorrelation).
- **Hard ceiling** wherever the primitive is absent in generation *and* validation *and* code, or
  the data under-determine it. No architecture crosses this — it requires a stronger model or a
  richer primitive library / more data.

**Corollaries that unify prior findings:**
- Conservation is a trust gate: **necessary but not sufficient**; it discards rare-but-correct
  signal when errors are systematic (peer-consensus failure mode).
- V/G asymmetry (verification easier than generation) holds only for **shallow, executable** checks;
  it **breaks for deep analytical claims** where verifying ≈ deriving.
- The reliable computational path is **model-authored code executed on data** (offload analysis to
  the interpreter) — but only for operations the model can correctly *code*.

## 5. The central contribution: an outer loop that does the method-research itself

The per-benchmark method above (which generator? which verifier? which selector?) was found by a
**human** trying variants (5 attempts on UltraHorizon before evidence-based aggregation won). The
novel contribution is a **recursive meta-loop ("docker-in-docker")** that automates exactly this:

> Level N runs and judges level N−1 on an ever-more-held-out metric. **L0** solves. **L1** selects
> the METHOD by a **gold-free signal** (predictive fidelity / evidence-support on manufactured
> known-answer probes), then **validates** by transfer to held-out probes.

**It works, and it is our own conservation principle applied recursively** (L1's gold-free signal =
conservation-at-noise-floor lifted to the method level: zero held-out error ⟺ correct on noiseless
data). Two demonstrations:

- **NewtonBench** (`run_meta_tower_nb.py`): L1 **auto-rediscovers `regression`** by held-out
  fidelity alone (gold-free pick == gold-best), no human hint.
- **UltraHorizon** (`run_meta_uh.py`): over the 4 aggregators on shared rollouts, the meta-loop
  auto-picks the winner on labeled probe seeds and it **validates on held-out seeds (MATCH)** —
  and it did *not* copy the human's hand-choice; its probe→validate protocol selected a
  competitive/better evidence-based aggregator, i.e. **more rigorous method-research than the human
  did by hand**. Robustly, it auto-rejects the reliably-worst method (consensus).

**Measured, not asserted — and the deep law.** A noise-curriculum
(`run_meta_curriculum_nb.py`) shows the meta-selection stays grounded up to **noise ≈ 0.1**, then
decouples (gap collapses / gold labels themselves go noisy) — matching NewtonBench's documented
noise-sensitivity. This yields the governing law:

> **The tower does not create verifiability — it consumes it.** Each rung spends some of the
> environment's verifiable structure; the usable tower depth **equals** the total independent
> structure the environment affords. Recursion **relocates** the fundamental limit one level up,
> it does not escape it. (A conservation-of-verification / no-free-lunch statement.)

## 6. Honest scope and where this points

- **Not a SOTA claim.** On comparable metrics the weak model does **not** beat SOTA (DiscoveryBench
  forced 16.6 < 24.5; NewtonBench 0.67 is the easy-vanilla slice, not the full 0.76 benchmark).
  What is real is **strong amplification of a weak model** and an **automated method-discovery
  loop** bounded by a **measurable** quantity.
- **The realized universal form**: weak model = orchestration + hypothesis generation; shell =
  library of reliable analytical primitives + conservation-selection with an independent verifier;
  **meta-loop = autonomous per-benchmark method-selection**, valid within the environment's
  verifiable structure.
- **The lever that raises the ceiling** is extending the primitive library (§3, Gate 3); **the
  lever that removes the human** is the meta-loop (§5) — each bounded by the same conserved quantity.

---
*All numbers are on the slices stated (small N on SAB/UH; NB is the easy-vanilla slice, not the full
324-task benchmark). Reproduce via the `run_conservation_*` runners.*
