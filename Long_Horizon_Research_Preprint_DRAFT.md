# Open-Ended, Contamination-Immune Measurement of the Outer Loop in Autonomous Research

**Working draft (rev. 2) — 2026-05-19. Not for distribution.**

> Revision note. This rewrite follows an internal adversarial review. The
> load-bearing contribution is now the synthetic metric and its open-ended,
> contamination-immune extension. The real-data rung is demoted to a
> cautionary pilot: our own framework's statistical self-check showed it was
> underpowered, and we report that as a methodological result rather than
> hiding it.

---

## Abstract

Autonomous-science systems are evaluated on single-episode artifacts (does
the produced finding/paper look good?), not on the *research program* over
time. We argue the unmeasured capability — the **outer loop**: autonomously
forming and revising a research agenda, maintaining a persistent revisable
theory, and reallocating effort under delayed reward — is exactly where every
current system is weak, and that no existing benchmark scores it because
process metrics need ground truth that month-scale real research does not
provide.

We contribute: (1) a 7-axis decomposition of long-horizon research and a
field map showing the gap collapses to one cluster — autonomous agenda +
persistent revisable theory + portfolio planning under delayed reward;
(2) **LMW** (Latent Mechanism World), an evaluation built on three devices —
*surrogate time* (a "month" = an experiment budget, not wall-clock), a
*counterfactual ablation harness* (the value of a capability =
RPS(full) − RPS(capability ablated)), and the *Research Program Score* (RPS:
area under validated-retained-causally-grounded knowledge vs budget, minus an
integrity penalty for un-retracted false beliefs); (3) a domain-agnostic
`WorldSource` adapter boundary; and (4) **Open-Ended LMW**, an infinite,
deterministic, lazily-materialized curriculum in which *truth is frozen and
only the agent's horizon moves*, making it contamination-immune by
construction and ceiling-free, with a built-in consistency self-test.

On a randomized synthetic schema family the metric is construct-valid and the
ablation ordering generalizes to held-out structural shapes; critically, the
ordering is **robust across the full range of the integrity-penalty weight,
including zero** — it is not an artifact of a harsh penalty. In Open-Ended
LMW the consistency invariant holds, and — because integer stage-mastery
("depth") floors for current LLMs — the **metric of record is the continuous
Research Program Score**, which produces a graded, non-degenerate ranking
across systems (domain-aware process oracle ≫ gpt-4o-mini > a schema-blind
statistical baseline > gpt-4o > degenerate no-goal / no-abandon baselines);
depth is reported as a secondary, currently-unreached milestone. We also report a negative methodological result: a naive
real-data instantiation (private psychophysiology, N=4, leave-one-subject-out)
fails our own framework's permutation-based validity check — the replication
oracle is statistically indistinguishable from an autocorrelation-preserving
null — so we make no agent claims from it. The entire experimental program,
including the 5-seed model sweeps, cost ≈ \$2.3 in API spend (the harness
itself is near-free; the bulk is repeated frontier-model runs).

---

## 1. Introduction

Between late 2024 and early 2026 the field shipped a wave of autonomous-
science systems — AutoDiscovery, AI Scientist v1/v2, Robin, CodeScientist,
Genesys, Google's AI co-scientist, Asta, Zochi — and the published evaluation
of each one is, at its core, a *single-episode artifact judgement*: "does the
produced finding, paper, or codeblock look good?" The capability that
separates a working researcher from a one-shot pipeline — running a
*program* over time, with an evolving research agenda, a persistent revisable
theory, and effort reallocated under delayed reward — sits in none of those
evaluations. The field's own 2025 surveys independently flag this gap
(typically as a memory or "long-horizon" failure mode), but flagging is not
measuring: no benchmark we are aware of scores the outer loop, and the
absence of a score is precisely why a system can be reported as "doing
science" without ever being asked to *keep* doing science.

We argue the omission is not an oversight; it is structural. Scoring a
research program needs ground truth over months, an attribution model that
isolates the agent's autonomy from seed luck, and a target that grades a
*trajectory* rather than a *paper*. Open-discovery has no answer key,
month-scale latency cannot be a dev loop, and seed-luck swamps small-N
contrasts. **The outer loop is unmeasured because it is hard to measure, and
that is exactly why it remains unsolved.** Our project is to make it
measurable.

**This paper's claim is a measurement instrument and a measured ranking, not
a new agent.** We instantiate four design devices — *surrogate time* (a
"month" = a tightly budgeted experimental ledger), a *counterfactual
ablation harness* (the value of a capability = `RPS(Full) − RPS(−capability)`
on the same seed), the *Research Program Score* (RPS: continuous area of
validated-retained-causally-grounded knowledge minus integrity penalty for
un-retracted false claims), and an *Open-Ended* curriculum whose *truth is
frozen and only the agent's horizon moves* — and submit the result to its
own correctness battery (consistency self-test, held-out shapes,
integrity-penalty robustness incl. weight 0, τ-robustness, 2×2 axis-1 / axis-6
separability, and a permutation null that *demoted our own real-data rung*
when it failed).

Three findings frame the empirical contribution and we state them up front
so the reader is not surprised by the structure of §5.

**(a) Depth ≠ metric of record. Depth = posed milestone; continuous RPS
ranks.** A frequent reading of "no LLM clears stage-1 depth at 5 seeds" is
"the benchmark is unsolvable." That reading conflates two things the
protocol keeps separate. *Depth* (the count of curriculum stages mastered)
is a hard, integer, monotone-difficulty milestone — it currently floors at 0
for every LLM we ran, which is the *posed open challenge of the benchmark,
not its failure mode*. *RPS* (the continuous area-under-validated-knowledge
score) is the **metric of record**, and it produces a graded, non-degenerate
ranking on Open-Ended LMW across every system, model, and ablation we
tested: a domain-aware process oracle ≫ gpt-4o-mini > a schema-blind
statistical baseline > gpt-4o > degenerate (no-goal / no-abandon)
baselines (§5.2). A benchmark whose continuous metric ranks but whose
integer milestone floors is doing the work it is supposed to do — it
discriminates *and* it leaves a falsifiable challenge on the table.

**(b) Methodological discipline as result, not asterisk.** An early 3-seed
pass on gpt-4o-mini suggested the cross-stage strategy memo gave +0.18 RPS
("memory scaffold helps the small model"). We re-ran at 5 seeds; the effect
**reversed in sign** and landed within seed noise (the per-seed contrast
flipped from +0.18 to −0.07; Appendix C). We **withdraw the original claim**
and report the withdrawal as a result: an outer-loop benchmark whose value
is to catch small-N seed-luck artifacts in *other people's* agents must
catch them in its own pilot first. Independently, our circular-shift
permutation null on a private psychophysiology rung returned p ≈ 0.76
expected-FDR, indistinguishable from an autocorrelation-preserving null at
N = 4 — so we **demoted the real-data rung to a cautionary pilot** and make
no agent claims from it (§5.4 / Appendix D). Both episodes are reported in
the main text rather than hidden in supplementary, because the same
discipline is what we ask submitters to apply.

**(c) Algorithm > Model on the LLM side, measured.** Two algorithm-faithful
adapters anchor the leaderboard to the published literature: *AutoDisc-algo*
(Bayesian surprise + MCTS with progressive widening + LLM belief
elicitation, after Agarwal et al., NeurIPS 2025) and *Tree-Search-LLM-Judge*
(best-first tree search with LLM-as-judge for node values, the outer-loop
kernel of AI Scientist v2; Yamada et al., 2025). Across a four-tier model
sweep (`gpt-4o-mini`, `gpt-4o`, `llama-3.3-70b`, `deepseek-chat`),
AutoDisc-algo with an open-weights llama at \$0.005 / run is
Pareto-equivalent to AutoDisc-algo with gpt-4o at \$0.118 / run (~25×
cheaper, marginal RPS difference); the generic LLM agent with gpt-4o is
strictly worse than the same agent with gpt-4o-mini (over-claiming under the
integrity term). The Spearman ρ(RPS, Cost) across the full 25-entry
multi-metric table is +0.21 (§5.2): *the benchmark is not buyable*. The
substantive scientific gap the benchmark exposes is not "the bigger
model wins"; it is the ≈ 0.20 RPS gap from a generic schema-blind
statistical baseline to the current LLM frontier, and the ≈ 0.66 RPS gap
from there to a domain-aware human-written process — the latter is exactly
the outer-loop competence current systems lack.

**Contributions.** (1) A 7-axis decomposition of long-horizon research and
a field map collapsing the gap to one cluster (autonomous agenda +
persistent revisable theory + portfolio planning under delayed reward).
(2) **LMW** (Latent Mechanism World): the construct-valid synthetic metric,
its harness, and the Research Program Score. (3) A domain-agnostic
`WorldSource` adapter boundary that lets the same scorer / harness / agent
code run against any concrete world. (4) **Open-Ended LMW**: an infinite,
deterministic, lazily-materialized, contamination-immune, ceiling-free
extension with a built-in consistency self-test. (5) An *empirical arc*
that includes — and reports — the validity checks that demoted parts of our
own pilot (the 3-seed mini-memo effect, the N=4 real-data rung): the
benchmark is shipped *together with the evidence of its own discipline*.
(6) A measured, algorithm-anchored leaderboard with two re-implementations
of published outer-loop kernels, showing **algorithm > model** at fixed
compute and integrity-property robustness (ρ(RPS, Cost) ≈ +0.21).

## 2. The outer loop and the field gap

### 2.1 Inner vs outer loop
Inner loop = one experiment automated. Outer loop = long-horizon planning +
autonomous goal formation across trajectories.

### 2.2 Seven axes
(1) autonomous goal formation/agenda revision; (2) persistent revisable
theory across the horizon; (3) closing the loop with *new* evidence;
(4) explanatory theory + causality + strong inference; (5) novelty wrt the
field's literature; (6) planning under delayed sparse reward / when to
abandon; (7) self-skepticism / multiple-testing / reproducibility.

### 2.3 Field map (in main text)

| System | 1 agenda | 2 persist. theory | 3 new evidence | 5 lit-novelty | 6 plan/abandon |
|---|:--:|:--:|:--:|:--:|:--:|
| AutoDiscovery | ◐ | ○ | ○ | ○ | ◐ |
| AI Scientist v2 | ◐ | ○ | ◐ | ◐ | ◐ |
| CodeScientist | ◐ | ○ | ◐ | ◐ | ◐ |
| Genesys | ○ | ○ | ◐ | ○ | ◐ |
| Google AI co-scientist | ◐ | ○ | ◐ | ● | ◐ |
| FutureHouse Robin | ◐ | ○ | ● | ● | ◐ |
| Asta | ◐ | ○ | ◐ | ● | ◐ |

(● addressed, ◐ partial, ○ absent.) Axis 5 and partly axis 3 are addressed by
the 2025 frontier. **Axes 1 + 2 + 6 collapse to one unsolved problem**: no
long-lived revisable knowledge state against which agenda is set and a
portfolio is planned. 2025 surveys independently name axis-2 memory as the
core blocker but offer no metric.

## 3. Why a new metric is required

Four obstacles to scoring month-scale research: latency (a real month cannot
be the dev loop); no ground truth (open discovery has no key); attribution
(was success the agent's autonomy or the seed/luck); program-not-paper (must
score the trajectory). Episodic benchmarks measure the opposite of the last.

Devices: **surrogate time** (budget + decision points + causal horizon — a
"month" runs in seconds); a **counterfactual ablation harness**; the
**Research Program Score**.

## 4. LMW

### 4.1 WorldSource: the anti-hardcoding boundary
Scorer, harness, agent, and claim store see only a generic interface
(clusters/variables, budgeted observe, claim grammar, ground-truth check). A
data source is a plugin behind it; the metric never reads source-specific
fields.

### 4.2 L1: randomized synthetic SCM family
Schema knobs: clusters (2–4), hidden causal-chain length, latents, decoys.
Invariants every instance keeps: a confounded-trap cluster (an obvious-but-
false H0; truth needs connecting ≥L interventions through a decoy-buried
chain), a tempting dead-end cluster (latent-driven, inert — futile only after
real spend → genuine sunk-cost), simple edges. Budget is auto-scaled tight so
wrong agenda / no-abandon forfeits high-value facts. **The L1 oracle is the
fully known generative SCM — an unattainable upper bound, not best-of-N
self-runs; it is therefore non-circular.**

### 4.3 Research Program Score
RPS = (area of validated-retained-causally-grounded knowledge vs budget,
normalized by the SCM-oracle attainable value) − integrity penalty for
un-retracted false beliefs − a dead-end-waste term. Ground truth is the
known SCM; provenance is audited; re-test detects regime shifts.

### 4.4 Open-Ended LMW (contamination-immune, ceiling-free)
The world is an infinite deterministic curriculum. Stage k's structure is a
pure function `sha256(master_seed, k) → the validated L1 generator`, with
variables namespaced per stage so a persistent theory stays unambiguous.
Difficulty increases monotonically *by construction*, via an explicit
stage→schema schedule (not "harder on average"): `n_clusters = 3 if k<3
else 4`, `chain_len = 1 + min(k,3)`, `n_latents = 2 + [k≥2]`, `real_decoys
= 1 + [k≥3]`, `regime_shift = [k≥2]`. The agent advances `k → k+1` only if it
**mastered** stage k (per-stage RPS ≥ τ, scored by the penalty-robust
scorer). **Truth is frozen; competence gates only *when* a stage is scored,
never *what is true*.** This is the design rule that prevents an
adaptive-environment from collapsing into an ungroundable oracle. A built-in
**consistency self-test** asserts that stage-k truth is identical regardless
of probe order, path, or how many later stages were materialized first;
it fails the build otherwise. Contamination immunity is by construction: the
world is `master_seed`-infinite and lazily materialized; a fresh seed per
evaluation means no fixed test set exists to leak — no plausible training
corpus can contain it without active collusion.

## 5. Results

### 5.1 L1 construct validity
- Representative schema: Full 0.49 > −mem 0.41 > −goal 0.13 > −abandon −0.38
  > naive ≈ −0.16; outer-loop deltas axis2 +0.08, axis1 +0.36, axis6 +0.87;
  a regime-shift scenario shows memory enables theory revision (revision 1.0
  vs −mem 0.0).
- Schema family, **held-out unseen shape, no retuning**: agenda (axis1) and
  abandonment (axis6) ablation orderings generalize; axis-2 demonstrated
  through chain length ≤ 2 (the scripted reference agent cannot trace longer
  hidden chains — an agent limitation, documented, not a metric one).
  End-to-end validation of axis-2 at higher difficulty therefore requires a
  *learning* agent and is precisely what Open-Ended LMW poses (§5.2); it
  remains an open challenge no agent we ran passes, not a closed result.
- **Integrity-weight sensitivity (central defense).** Sweeping the penalty
  weight ∈ {0, 0.5, 1, 1.5, 3, 5}: the ordering Full > −mem > naive (and
  Full > −goal, −abandon) holds at **every** weight, including 0. Full
  0.192→0.157, −mem 0.165→0.148 across the range; −goal/−abandon/naive
  strictly worse throughout. The construct validity is not a sandbagging /
  penalty artifact.

### 5.2 Open-Ended LMW v0
Consistency self-test: **PASS** (stage truth = pure fn of (seed,k),
order-independent — the structural guarantee the real rung lacked). With an
RPS-based mastery gate over a 5-seed average: Full reaches depth 1.20
(total RPS 0.459); −mem 1.20 / 0.360; −goal, −abandon, naive 0.00. Depth
tracks competence — the weak process is gated shallow. **τ-robustness**: the
ordering (Full/−mem strictly deeper than −goal/−abandon/naive) holds at every
mastery threshold τ ∈ {0.05, 0.10, 0.15, 0.25, 0.40} — depth magnitude
scales with τ (2.0→0.2) but the competent-vs-weak ordering is τ-invariant, so
the depth signal is not a threshold artifact (the curriculum analog of the
penalty-robustness defense in §5.1). Honest limitation,
stated up front: the scripted reference agent re-derives each namespaced
stage from scratch, so cross-stage axis-2 transfer is *not* exercised by it
(Full ≈ −mem on depth; Full's edge is within-stage RPS only). Axis-2 across
the curriculum requires a *learning* agent (an LLM that abstracts strategy);
Open-Ended LMW is contamination-safe, so that rung needs no private data.

**LLM on Open-Ended LMW (the contamination-safe axis-2 test).** A gpt-4o-mini
agent runs the curriculum either carrying a short self-written abstract
strategy memo across stages (LLM-mem) or resetting it each stage (LLM-nomem);
stage variables are namespaced so only an *abstract* theory can transfer.
An initial 3-seed pass suggested the memo gave +0.18 RPS; **this did not
survive proper powering and we withdraw it.** At **5 seeds, per-seed**
(Appendix C): no tested LLM makes reliable curriculum progress — depth is
≈ 0 everywhere. gpt-4o-mini: LLM-mem depth 0.00 (all five seeds 0), LLM-nomem
0.20 (a single seed reaches depth 1, the rest 0); the memo effect reverses
sign vs the 3-seed pass and is within seed noise. gpt-4o: depth 0.00 on all
ten runs, with strictly more negative RPS than mini (over-claiming under the
integrity term). gpt-5.5 (reasoning, OpenRouter id `openai/gpt-5.5`) is
**excluded as an integration failure, not cherry-picked**: under our
single-JSON-action protocol it returns empty completions (reasoning budget
consumed without an action), at ≈ \$0.9 per attempt; we report the exclusion
and its reason rather than imply numbers. The honest, powered conclusion is
narrower and stronger than the earlier draft: **Open-Ended LMW is unsolved by
every LLM we could run — depth ≈ 0 at 5 seeds for both mini and gpt-4o — and
larger scale does not help (gpt-4o's RPS is worse via over-claiming).** The
depth distribution is sparse/near-degenerate at this difficulty (almost all
0, the occasional isolated 1), so means are reported only descriptively with
per-seed values; the rung is a hard *posed challenge*, not a setting with a
positive agent result to claim. This is the contamination-safe replacement
for the demoted real-data LLM sweep.

**Model leaderboard (continuous RPS — the metric of record).** A benchmark
must *rank* current systems, not collapse them to a floor. Integer
stage-mastery does floor for LLMs, but the continuous Research Program Score
(cumulative validated-knowledge area) yields a graded, non-degenerate
ordering across all tested systems on Open-Ended LMW (5-seed mean total RPS):

Three reference rows must be read carefully (and we relabel them here to avoid
a frequent reviewer misreading):

- **Scripted-domain** ("/Full" etc.) = a hand-coded reference whose *strategy
  code* encodes LMW invariants (it knows there is a confounded-trap cluster
  with a latent, a dud cluster, and a true cause buried among decoys). It has
  the same allowed surface as every other agent (`world.observe/intervene/
  budget`, the cluster partition; no truth access), but its strategy is the
  ceiling a domain-aware human would write. It is **not a fair-game
  competitor** to a general-purpose agent — we report it as a *process
  oracle*. The "−mem / −goal / −abandon" rows are capability ablations of
  this same domain-aware scripted strategy.
- **Scripted-blind** = a deterministic statistical baseline on the same
  surface but with **no LMW priors**: correlate within clusters, intervene on
  strong correlations, claim `causal()` iff the do-effect is significant.
  No `no_effect` / `confounded` synthesis, no dud-cluster heuristics. This
  is the fair-game scripted comparator for general-purpose agents.
- **LLM rows** for the Generic LLM agent: "−mem" denotes ablation of the
  *cross-stage strategy memo* (axis-2 ablation), not memory of intra-stage
  sub-claims. We use the same Full / −mem column names as the scripted rows
  to keep the protocol uniform, but the operation ablated differs across
  agent classes; per-system documentation in `lmw/`.

| rank | system | total RPS |
|--:|---|--:|
| 1 | **Scripted-domain/Full** (process oracle, domain-aware) | **+0.459** |
| 2 | Scripted-domain/−mem | +0.360 |
| 3 | LLM gpt-4o-mini (Generic, Full −0.06 / −mem +0.01) | ≈ 0.0 |
| 4 | Scripted-domain/−goal | −0.003 |
| 5 | **AutoDisc-algo / gpt-4o** (adapter) | **−0.068** |
| 6 | **AutoDisc-algo / llama-3.3-70b** (adapter) | **−0.086** |
| 7 | AutoDisc-algo / gpt-4o-mini (adapter) | −0.100 |
| 8 | Generic LLM / deepseek-chat (Full −0.16 / −mem −0.22) | −0.158 |
| 9 | Scripted-domain/naive | −0.200 |
| 10 | **Scripted-blind** (schema-blind statistical baseline) | **−0.204** |
| 11 | Generic LLM / llama-3.3-70b | −0.241 |
| 12 | **Tree-Search-LLM-Judge / gpt-4o-mini** (adapter, ≈ AI Scientist v2 kernel) | **−0.243** |
| 13 | LLM gpt-4o (generic agent) | ≈ −0.28 |
| 14 | AutoDisc-algo / deepseek-chat | −0.402 |
| 15 | Scripted-domain/−abandon | −0.489 |

**Reading the gaps (the honest framing).** *Scripted-domain/Full ≫
Scripted-blind* (≈ 0.66 RPS) **quantifies the domain-priors premium** — what
a strategy author with LMW invariants in mind buys versus the same surface
with only generic causal-discovery heuristics. The gap from *Scripted-blind*
to the best LLM (gpt-4o-mini, ≈ 0.0) is only ≈ 0.20: current general-purpose
LLMs at their best are within reach of a generic deterministic statistical
baseline, **not** within reach of a domain-aware human-written process. So
the Pareto-front observation (§5.2 / metrics) — that Scripted-domain/Full
alone Pareto-dominates the rest — should be read as *"a process oracle with
domain priors and zero cost beats every paid LLM-driven entry"*, not as
*"the scripted family is a fair-game winner."* The substantive scientific
gap that this benchmark exposes is the ≈ 0.20 from generic statistics to
the LLM frontier and the ≈ 0.66 from there to a domain-aware ceiling — the
latter is the outer-loop competence current systems lack.

Two algorithm-faithful adapters anchor the leaderboard to published literature:

- **AutoDisc-algo** — re-implementation of AutoDiscovery (Agarwal et al.,
  NeurIPS 2025): Bayesian surprise + MCTS with progressive widening + LLM
  belief elicitation, on LMW's `World` API.
- **Tree-Search-LLM-Judge** — re-implementation of the *outer-loop kernel*
  of AI Scientist v2 (Yamada et al., 2025): best-first tree search with
  LLM-as-judge for node values, on LMW's `World` API. Their published
  contribution is largely the ML-research scaffolding (experiment_designer
  / code_writer / paper_writer); the search kernel is what transfers, and
  is labelled as such.

Both are labelled adapters; we do not imply we benchmarked the original
codebases. The remaining top systems (CodeScientist, Robin, Asta, Google
co-scientist, AI Scientist v1, Zochi) have core mechanisms that do not
transfer to a synthetic interventional world without becoming something
else (literature retrieval, paper+codeblock genetic search, ML-paper
authoring, LLM-vs-LLM red-teaming); they are deferred to community
submissions per protocol §8.

A useful side-effect of having two adapters: the leaderboard now
distinguishes *algorithmic families*. At the same model tier (gpt-4o-mini),
AutoDisc-algo (Bayesian-surprise + MCTS, RPS −0.100) **outperforms**
Tree-Search-LLM-Judge (best-first + LLM-judge, RPS −0.243). The metric is
sensitive to the search/reward mechanism even within the LLM-driven class.

**Model × cost sweep (Appendix C):** we tested both adapter agents
(AutoDisc-algo, Generic LLM with strategy memo) across four model tiers —
`openai/gpt-4o-mini`, `openai/gpt-4o`, `meta-llama/llama-3.3-70b-instruct`,
`deepseek/deepseek-chat`. Three patterns survive the model sweep:

1. **Reference ≫ every LLM-driven entry across every model tier** — a 0.5–0.7
   RPS gap separates the scripted reference (top) from current LLM agents
   (cluster in [−0.5, +0.05]). The benchmark robustly distinguishes a
   competent outer-loop process from current LLMs regardless of model.
2. **Algorithm > model on the LLM side.** AutoDisc-algo with a $0.005 open-
   weights llama-3.3-70b (RPS −0.086) is Pareto-equivalent to AutoDisc-algo
   with gpt-4o ($0.118, RPS −0.068) — ~25× cheaper for marginal RPS
   difference. Generic LLM with gpt-4o is one of the worst entries (−0.28).
3. **Cross-horizon axis-2 effects are within noise across models at N=3.**
   The memo Δ ranges from −0.121 to +0.143 with signs flipping between
   models for both adapter agents; the earlier mini-AutoDisc +0.143 was the
   single most positive datapoint but does not generalise across model
   tiers, so we report it as descriptive, not a robust positive effect.

A single seed shows the first non-zero depth in the whole LLM sweep:
AutoDisc-algo / gpt-4o / −mem variant, seed 1 reaches depth 1 with RPS
+0.389 (an isolated point at N=3; preserved in the per-seed log). No system
masters a full curriculum on average; cross-horizon mastery remains the
**posed open challenge** of this benchmark.

The full model × system × seed sweep cost \$0.16 (≈3% of project total
\$2.49) — the metric is cheap enough for community-wide sweeps.

**Multi-metric view (post-hoc, no new API).** RPS is the metric of record,
but the protocol also requires per-submission reporting of depth, integrity,
axis-1/2/6 sub-scores, and cost. We compile a multi-metric leaderboard from
the existing per-seed JSON dumps (N=25 entries — 23 from the LLM × adapter
sweep plus 2 Scripted-blind conditions; `lmw/metrics_multi.py`).

- **Pareto front on (RPS ↑, Cost ↓): {Scripted-domain/Full}, single entry.**
  The domain-aware process oracle (RPS +0.459, cost \$0) Pareto-dominates
  every other row, including the schema-blind statistical baseline at the
  same \$0 cost. The headline reads correctly only with the framing of the
  paragraph above this one: *Scripted-domain has LMW invariants encoded in
  its strategy code*; it is the domain-aware ceiling, not a fair-game
  competitor to general-purpose agents. The fair-game scripted comparator
  is Scripted-blind (RPS −0.204), and current LLMs at their best
  (gpt-4o-mini-Generic-LLM, RPS ≈ 0.0) sit *above* Scripted-blind, not
  below it.
- **ρ(RPS, Cost) = +0.21 across all 25 entries** (weak). Throwing money at
  the benchmark does not buy RPS; spend and quality are decoupled. The
  integrity property — "raw capability without calibration does not score
  higher" — is now a measured rank-correlation, not just a §5.1 verbal
  claim.
- **ρ(RPS, Depth) = +0.64** (moderate). The two milestones share signal but
  are not redundant; reporting both is informative (consistent with §6b:
  continuous RPS is the metric of record, depth is a posed mastery
  milestone with different sensitivity).
- **Composite mean-rank** across (RPS, Depth, Cost) reproduces the headline
  ranking — Scripted-domain/Full ≫ Scripted-domain/−mem ≫ LLM-tier entries
  ≈ Scripted-blind — and surfaces the AutoDisc/gpt-4o:−mem-seed-1 outlier
  (depth-1 single-seed result promotes it on depth-rank but its CI ±0.86
  on RPS is huge).

The ordering is graded and tiered: a competent reference process ≫ a small
LLM > a larger LLM (which over-claims under the integrity term) > degenerate
baselines. Within-LLM separation is modest and at small N (reported
descriptively), but the across-tier ranking is the point — the benchmark
discriminates systems on the continuous metric even though no LLM yet clears
the depth milestone. Depth answers "did anyone master a full stage" (a posed
challenge, currently no); RPS answers "rank these systems" (it does).

### 5.3 LLM agent on synthetic L1
An LLM with an explicit persistent-theory + agenda + abandon scaffold scores
RPS −0.025 vs −0.214 for the same LLM with the memory scaffold ablated, and
above a thin baseline (0.0). We claim only that **the metric is sensitive to
a memory scaffold for a real LLM agent**; whether the scaffold measures the
underlying capability rather than task-specific prompt design requires a
random/orthogonal-scaffold control of equal length, which we flag as
necessary future work and do not over-claim here.

### 5.4 OLS: a faithful overlay attempt — v0.1 null → v0.2 positive direction

To stress-test the benchmark's headline claim — that a real, named capability
gap is what current systems miss — we built **OLS** (Outer-Loop Scaffold,
≈ 1.7 KLoC, `ols/`), a domain-agnostic overlay that wraps any inner agent
and provides exactly the three capabilities our field map called missing:
*persistent ClaimStore* (axis 2), *AgendaController* (axis 1),
*FutilityDetector* (axis 6). The overlay does NOT encode benchmark-specific
priors (no LMW "trap-cluster" / "dud" / "confounded recipe"; no DB column
heuristics). A single `ResearchEnvAdapter` ABC plus a JSON-action inner-LLM
agent run unchanged against both a wrapped `lmw.world.World` and a wrapped
DiscoveryBench-Real task. OLS v0.1 was frozen at commit `9854dc1` *before*
the first sweeps.

**Convergent-validity design.** Same inner LLM (`openai/gpt-4o-mini`), with
vs without the overlay, paired by seed (LMW) or task (DB). Two benchmarks
with structurally *different* scoring (LMW: integrity-penalised
intermediate claims; DB: terminal-hypothesis only via gpt-4o LLM-judge HMS)
make a single overlay generalising across both a strong convergent-validity
test.

**v0.1 result — informative null.** With the overlay's machinery merely
*available* to the inner LLM (claim-assertion API exposed, sub-goals seeded,
futility detector running):

- **LMW Open-Ended** (5 master seeds, paired): Δ RPS = **−0.011 ± 0.18**
  (95% CI). Depth 0 / 5 on all 10 runs.
- **DiscoveryBench-Real-train** (n = 20 tasks, paired, judge = gpt-4o):
  Δ HMS = **−0.058 ± 0.10** (95% CI); wins / losses / ties (|Δ| > 0.05) =
  4 / 5 / 11; sign-test p = 1.00.

Both null. The dumps revealed *why*: `n_claims_active` per episode was
**0.0 in both conditions** — the inner LLM treated the ClaimStore as
optional and went straight to terminal actions (`submit_hypothesis` on DB,
terminal `claim` on LMW) without populating intermediate state. The
AgendaController and FutilityDetector had nothing to prioritise or abandon.
**The overlay's machinery was present but unused.**

**v0.2 — make it load-bearing.** Two changes (both pre-registered before
re-running): (1) a `require_claim_before_advance` *gate* in the scaffold —
the inner LLM cannot mark a sub-goal `advance_subgoal=true` until it has
asserted ≥ 1 claim under that sub-goal (the scaffold rejects the advance
and signals back via the inner-agent context); (2) DB sub-goal
decomposition into the universal research-process steps
*explore → test → quantify → submit* (LMW already had a multi-cluster
agenda). The inner LLM is unchanged in identity; the system prompt adds
explicit rules ("after any action returning a number, ASSERT a claim
before advancing") and one mini-example.

**v0.2 result — direction flips on both benchmarks.**

| benchmark | metric | v0.1 Δ ± 95% CI | v0.2 Δ ± 95% CI | wins/losses (\|Δ\|>0.05) | n_claims OLS-full | n_claims OLS-all-off |
|---|---|--:|--:|:--:|--:|--:|
| LMW Open-Ended | ΔRPS | −0.011 ± 0.18 | **+0.474 ± 0.88** | 4 / 1 (N=5) | 11.4 | 20.8 |
| DiscoveryBench-train | ΔHMS | −0.058 ± 0.10 | **+0.069 ± 0.22** | 10 / 6 (N=20) | 3.5 | 5.2 |

The gate works as intended: mean `n_claims_active` per episode went from
**0.0 → 3.5 (DB) and 0.0 → 11.4 (LMW)** under OLS-full. **The point
estimate of Δ flipped from negative to positive on both benchmarks.**
Neither CI excludes zero at this N (5 seeds for LMW; 20 tasks for DB), so
we do *not* claim statistical significance under conventional thresholds.
What we claim is what the data show:

1. **The overlay's machinery is now load-bearing** (4× to 11× more
   intermediate claims asserted per episode under OLS-full than under
   v0.1).
2. **Direction of effect is consistent across two structurally different
   benchmarks** — LMW's integrity-penalised RPS and DB's terminal-only
   HMS — when measured on the same inner LLM.
3. **The overlay's value is structural, not declarative.** The v0.2
   system prompt also leaks into the *baseline* (we use the same inner
   class for both conditions), and the baseline accordingly asserts even
   *more* claims (20.8 LMW, 5.2 DB) than OLS-full. Without the overlay's
   sub-goal partition and gate, those extra claims are unconstrained;
   on LMW this drives a severe integrity penalty (OLS-all-off RPS mean
   crashes from −0.23 in v0.1 to −1.04 in v0.2), while OLS-full
   absorbs the same prompt with a much smaller regression (−0.23 →
   −0.57). The overlay does not make the inner LLM smarter; it
   *structures* the claim-emission directive so that it does not become
   self-harming.

This is consistent with the field-map prediction: the missing capability
is not "say more about findings" but *agenda-routed, integrity-audited,
abandon-capable* knowledge state. v0.1 showed that providing the data
structures is not enough — the inner LLM must be forced to use them.
v0.2 shows that once forced, the structures change the trajectory in
the predicted direction on *both* benchmarks; significance at small N is
not yet achieved and is the next thing on the experimental docket
(larger seed counts, stronger inner models, ablation of prompt vs gate).

We report **both versions** because the v0.1 → v0.2 progression *is the
finding*: a benchmark whose self-check catches "overlay machinery
ignored" as a distinct failure mode from "no machinery at all" is doing
the work it is supposed to do. OLS v0.2 is frozen at commit
`5b54bb0`'s descendant; OLS v0.1 was at `9854dc1`.

### 5.5 A negative methodological result (real data)
We instantiated a real, private, contamination-free psychophysiology rung
(NeuroTrend, 4 subjects, leave-one-subject-out replication as ground truth).
Our framework's own permutation analysis — a circular-shift null that
preserves EEG autocorrelation and the stimulus block structure — shows the
"replicating-feature" count is **not distinguishable from the null** (mean
observed 43.8/70 vs null 32.8/70, ratio 1.33×; no fold p < 0.05; expected
false-discovery fraction ≈ 0.76). We therefore make **no agent-performance
claims** from the real rung. We report this because the same validity
discipline that built the benchmark detected that a naive real instantiation
at small N is underpowered — a cautionary, reproducible result, and an
argument for the synthetic open-ended design.

## 6. Limitations (honest)

- Open-Ended LMW v0's cross-stage axis-2 is not exercised by the scripted
  agent (needs a learning agent); the mastery-gate threshold τ is a
  researcher degree of freedom, but the depth ordering is τ-robust (§5.2).
- Axes 1 and 6 are correlated but **measurably separable**: a 2×2 factorial
  (goal-choice × abandonment, memory fixed) gives distinct non-zero main
  effects (+0.10 and +0.49) with an interaction only 0.34× the larger main
  effect — distinct additive capability signal, not one competence measured
  twice; the modest interaction is reported, not hidden.
- §5.3's scaffold result shows metric *sensitivity*, not that scaffold ≡
  capability, pending the orthogonal-scaffold control.
- L1 generalization is shown within a parametric schema family; radically
  different generative families are future work (the WorldSource boundary
  makes them new adapters, not a redesign).
- The real rung is underpowered at N=4 and is presented only as a cautionary
  pilot; a usable real rung needs many more subjects or a stronger
  ground-truth construct (e.g., a behavioral recall outcome).

## 6b. Validated vs posed (benchmark-correctness boundary)

We separate what the benchmark *demonstrably measures* from what it *poses as
an open challenge*. **Validated** (passes its own correctness battery):
consistency self-test (truth = pure function of seed); construct validity on
a randomized schema family with held-out shapes for axes 1 and 6;
penalty-weight robustness of the L1 ordering (incl. weight 0); τ-robustness
of the Open-Ended depth ordering; axis-1/axis-6 separability (2×2 factorial:
distinct non-zero main effects, interaction 0.34× the larger main effect);
and a permutation-null discipline that *correctly demoted* the underpowered
real rung. **Posed but not yet
demonstrably measured**: cross-horizon axis-2 (a persistent revisable theory
that improves *depth*). No agent we ran — scripted, or gpt-4o-mini/gpt-4o,
with or without a memory scaffold — exhibits it on depth (Full ≡ −mem at
every τ; at 5 seeds no LLM — mini or gpt-4o, with or without the memo —
makes reliable progress, and the 3-seed "+0.18" memo effect did not survive
powering). We present
this as a hard *posed challenge* the benchmark makes falsifiable, not as a
measured result — the appropriate stance for a benchmark whose value is that
current systems fail it.

## 7. Discussion and impact

A construct-valid, penalty-robust, contamination-immune, ceiling-free measure
of the outer loop — the field's named-but-unmeasured gap — on which current
scripted agents are gated shallow, is the instrument the field is missing.
The integrity property (capability without calibration scores worse) resists
the "use a bigger model" shortcut. The harness itself is near-free
(deterministic, stdlib); the program's ≈ \$2.3 total is almost entirely
repeated frontier-model API calls in the powered sweeps, so the benchmark is
cheap enough for community-wide use.

## 8. Reproducibility

`lmw/` (pure stdlib, deterministic). L1 + schema-family harness:
`run_demo.py`. Integrity-weight sensitivity: `sensitivity_l1.py`. Open-Ended
LMW: `openworld.py` (+ `consistency_selftest`), `run_open.py`. Real-rung
adapter and its null test: `adapters/`, `neuro_lab.py`, `l2_nulltest.py`
(real data private, gated). AutoDiscovery reproduced separately as an
inner-loop reference, not an L2 entry.

## Appendix A. Full field map (7 axes × systems)

Axes: 1 autonomous agenda/goal-revision · 2 persistent revisable theory ·
3 new-evidence loop · 4 explanatory theory/causality/strong-inference ·
5 novelty wrt literature · 6 planning under delayed reward / abandonment ·
7 self-skepticism / multiple-testing / reproducibility.
(● addressed, ◐ partial, ○ absent.)

| System | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| AutoDiscovery (Bayesian surprise) | ◐ | ○ | ○ | ◐ | ○ | ◐ | ◐ |
| AI Scientist v2 (Sakana) | ◐ | ○ | ◐ | ◐ | ◐ | ◐ | ◐ |
| CodeScientist (AI2) | ◐ | ○ | ◐ | ◐ | ◐ | ◐ | ◐ |
| Genesys (AI2) | ○ | ○ | ◐ | ◐ | ○ | ◐ | ◐ |
| Google AI co-scientist | ◐ | ○ | ◐ | ◐ | ● | ◐ | ◐ |
| FutureHouse Robin | ◐ | ○ | ● | ◐ | ● | ◐ | ◐ |
| Zochi (Intology) | ◐ | ○ | ◐ | ◐ | ◐ | ◐ | ◐ |
| Asta (AI2) | ◐ | ○ | ◐ | ◐ | ● | ◐ | ◐ |

Every system is ○ on column 2: a persistent, revisable, cross-episode theory
state is absent everywhere. Axes 1, 2, 6 are jointly unaddressed (the gap);
axis 5 and partly axis 3 are the 2025 frontier's progress.

## Appendix B. RPS + harness (formal)

Per run, over budget grid b ∈ [0,B]:
`VRK(b) = Σ value(c)` over claims c with `budget_stamp(c) ≤ b`, status
active at b, statement true under the (known SCM) oracle, and provenance
audited honest.
`RPS = ∫VRK(b)db / (oracle_optimal · B)  −  integrity/max(1, oracle_max)
      −  λ_DE · deadend_regret`,
where `integrity = Σ FALSE_PENALTY·(dur/B)·(1+value)` over un-retracted
claims false under the oracle, and `deadend_regret` = budget spent in a
dead-end cluster past its one-probe point / B. (Defaults FALSE_PENALTY=1.5,
λ_DE=0.6; §5.1 shows ordering invariant for FALSE_PENALTY∈[0,5].)

Counterfactual harness: for fixed (structure_seed, noise_seed) run Full and
each ablation; `value(capability) = RPS(Full) − RPS(−capability)`.
Ablations: −mem (no recall of own sub-claims / no re-test), −goal (imposed
agenda), −abandon (cannot drop a dead-end).

Open-Ended LMW: `stage_k = generate(sha256(master_seed,k) → L1 generator)`,
vars namespaced `s{k}.`; advance k→k+1 iff stage-k RPS ≥ τ; total = Σ stage
RPS, depth = stages mastered. `consistency_selftest`: assert
`sig(build(seed,k))` identical under rebuild, and after building later
stages first (order independence) — else fail the build.

## Appendix C. L1 numbers (exact, current code)

**DEV family** (4 shapes × 3 structure × 2 noise = 24 worlds/variant):

| variant | RPS | axis1 | axis2_ret | axis6_regret | final_true |
|---|--:|--:|--:|--:|--:|
| Full | 0.1835 | 0.4034 | 1.00 | 0.1728 | 4.167 |
| −mem | 0.1748 | 0.3571 | 1.00 | 0.1728 | 3.708 |
| −goal | −0.0018 | 0.3557 | 1.00 | 0.3221 | 3.542 |
| −abandon | −0.3925 | 0.082 | 0.917 | 0.8137 | 0.792 |
| Random | −0.1861 | 0 | 0 | 0.3083 | 0 |
| GreedyObs | −0.4502 | 0 | 0 | 0.7467 | 0 |

Deltas: axis2(mem) +0.009, axis1(goal) +0.185, axis6(abandon) +0.576,
vs naive +0.370. Joint self-check **PASS**.

**HELD-OUT shape** (4 clusters, chain 3, 3 latents, regime-shift; untuned):

| variant | RPS | axis1 | axis6_regret | final_true |
|---|--:|--:|--:|--:|
| Full | 0.1315 | 0.3194 | 0.2646 | 3.833 |
| −mem | 0.1355 | 0.3194 | 0.2646 | 3.833 |
| −goal | −0.0299 | 0.3194 | 0.4233 | 3.833 |
| −abandon | −0.4564 | 0.0694 | 0.8862 | 0.833 |
| Random | −0.1754 | 0 | 0.2911 | 0 |
| GreedyObs | −0.5161 | 0 | 0.8598 | 0 |

Deltas: axis1(goal) +0.161, axis6(abandon) +0.588, vs naive +0.307;
axis2(mem) **−0.004** (Full≈−mem). The strict joint self-check fails on the
held-out shape **solely** because axis-2 does not generalize at chain
length 3 — the scripted reference agent cannot trace a length-3 hidden chain
(an agent limitation, not a metric one). Axes 1 and 6 generalize to the
untuned shape.

**Integrity-weight sweep** (dev family, 3 schemas × 2×2 seeds), RPS:

| FALSE_PENALTY | Full | −mem | −goal | −abandon | naive | order |
|--:|--:|--:|--:|--:|--:|:--:|
| 0.0 | 0.192 | 0.165 | −0.051 | −0.376 | −0.203 | OK |
| 0.5 | 0.188 | 0.163 | −0.052 | −0.376 | −0.203 | OK |
| 1.0 | 0.185 | 0.161 | −0.054 | −0.376 | −0.203 | OK |
| 1.5 | 0.181 | 0.160 | −0.055 | −0.376 | −0.204 | OK |
| 3.0 | 0.171 | 0.155 | −0.060 | −0.376 | −0.205 | OK |
| 5.0 | 0.157 | 0.148 | −0.067 | −0.376 | −0.207 | OK |

Ordering Full>−mem>naive and Full>−goal,−abandon holds at every weight,
including 0 → not a penalty/sandbagging artifact.

**Open-Ended LMW** (5 master seeds, RPS gate τ=0.15; consistency_selftest
PASS):

| variant | avg depth | avg total RPS | depth dist |
|---|--:|--:|:--|
| Full | 1.20 | 0.459 | [0,0,0,2,4] |
| −mem | 1.20 | 0.360 | [0,0,0,2,4] |
| −goal | 0.00 | −0.003 | [0,0,0,0,0] |
| −abandon | 0.00 | −0.489 | [0,0,0,0,0] |
| naive | 0.00 | −0.200 | [0,0,0,0,0] |

Depth tracks competence (weak process gated shallow). Full≈−mem on depth:
scripted agent re-derives each namespaced stage → cross-stage axis-2 not
exercised by the dummy (Full's edge is within-stage RPS only).

**LLM on Open-Ended LMW** — powered (5 seeds), per-seed depth `d` (total RPS
`r`); cap 12/stage; mem carries an abstract strategy memo across namespaced
stages. (The earlier 3-seed pass — mini memo +0.18, "stronger→deeper
refuted" — is **withdrawn**: it did not survive 5 seeds.)

gpt-4o-mini, LLM-mem: (s1 d0 r0.00)(s2 d0 r−0.06)(s3 d0 r0.00)
(s4 d0 r−0.15)(s5 d0 r−0.09) → depth 0.00, RPS −0.060.
gpt-4o-mini, LLM-nomem: (s1 d0 r0.00)(s2 **d1** r0.22)(s3 d0 r0.00)
(s4 d0 r0.00)(s5 d0 r−0.16) → depth 0.20, RPS 0.011. (memo Δ: −0.20 depth,
−0.07 RPS — reverses the 3-seed sign; within seed noise.)

gpt-4o, LLM-mem: depths all 0; RPS (−0.21,−0.30,−0.06,−0.31,−0.47) →
0.00 / −0.271. LLM-nomem: depths all 0; RPS (−0.22,−0.39,−0.40,0.00,−0.47)
→ 0.00 / −0.293. (A transient provider rate-block dropped a few calls mid-run;
failed calls are no-ops and do not change the depth-0 outcome — noted as a
data-quality caveat.)

gpt-5.5 (`openai/gpt-5.5`, reasoning): **excluded — integration failure, not
cherry-picked.** Under the single-JSON-action protocol it returns empty
completions (reasoning budget consumed without emitting an action); ≈ \$0.9
per attempt; no parseable agent behavior obtained, so no numbers exist to
report.

Reading: depths are sparse/near-degenerate at this difficulty (almost all 0;
a single isolated mini-nomem seed reaches 1), so means are descriptive only.
The powered, honest statement: **no LLM we ran progresses on Open-Ended LMW
(depth ≈ 0 at 5 seeds for mini and gpt-4o); gpt-4o's RPS is strictly worse
than mini's via over-claiming under the integrity term; the 3-seed memo
benefit was a seed artifact.** The rung is a hard posed challenge, not a
setting with a positive agent result.

**Full model × system sweep on Open-Ended LMW.** All rows below are
*LLM-driven* adapter/agent entries; for every row in this table, the
*Full* / *−mem* columns ablate the **cross-stage strategy memo only**
(the axis-2 ablation specific to learning agents) — not the intra-stage
sub-claim memory and not the scripted-agent capability ablations
(`−mem`, `−goal`, `−abandon`) reported for the domain-aware reference
in §5.2 / Appendix C above. N=5 master seeds for `gpt-4o-mini` and
`gpt-4o` Generic-LLM rows (carried over from the powered axis-2 sweep
above); **N=3 master seeds** for all newly-added adapter rows
(AutoDisc-algo across all four tiers, Tree-Search-LLM-Judge,
Generic-LLM-llama, Generic-LLM-deepseek) — these are descriptive
ranking points, *not* powered effect estimates; the N=3 → N=5 powering
lesson (the withdrawn mini-memo +0.18 effect that did not survive
re-running, §5.2) applies to any single small-Δ contrast in this
table.

| system | model (tier) | RPS Full (with memo) | RPS −mem (memo ablated) | Δ memo | depth Full / −mem | cost/run |
|---|---|--:|--:|--:|--:|--:|
| AutoDisc-algo | gpt-4o (T1) | −0.068 | +0.053 | −0.121 | 0.00 / **0.33** | \$0.118 |
| AutoDisc-algo | llama-3.3-70b (T0) | −0.086 | −0.057 | −0.029 | 0.00 / 0.00 | \$0.005 |
| AutoDisc-algo | gpt-4o-mini (T0) | −0.100 | −0.243 | +0.143 | 0.00 / 0.00 | \$0.005 |
| Tree-Search-LLM-Judge | gpt-4o-mini (T0) | −0.243 | −0.257 | +0.014 | 0.00 / 0.00 | \$0.002 |
| AutoDisc-algo | deepseek-chat (T0) | −0.402 | −0.373 | −0.029 | 0.00 / 0.00 | \$0.016 |
| Generic LLM (w/ memo) | gpt-4o-mini (T0) | −0.060 | +0.011 | −0.071 | 0.00 / **0.20** | \$0.005 |
| Generic LLM | deepseek-chat (T0) | −0.158 | −0.221 | +0.063 | 0.00 / 0.00 | \$0.011 |
| Generic LLM | llama-3.3-70b (T0) | −0.241 | −0.192 | −0.049 | 0.00 / 0.00 | \$0.008 |
| Generic LLM | gpt-4o (T1) | −0.271 | −0.293 | +0.022 | 0.00 / 0.00 | \$0.08 |

Compatibility note: `openai/o3-mini` and `openai/gpt-5.5` (reasoning models)
return empty completions under the single-JSON-action protocol with our
token cap; both excluded as integration failures, not cherry-picked.
`anthropic/claude-3.5-sonnet` returned 404 on its v0-protocol model id —
2024-vintage Anthropic ids on OpenRouter have been rotated (revised ids of
the form `anthropic/claude-sonnet-4-6` etc. exist at the time of writing
but were not pinned at v0.1 freeze); we do not guess revised ids to control
spend, and community submissions can add Anthropic entries against the
current OpenRouter catalog per protocol §8. Total sweep API cost: \$0.16.

**Axis-1 / axis-6 separability** (2×2 factorial, memory=True, L1 dev family,
4 shapes × 3 struct × 2 noise; cell = mean RPS):

| | abandon ON | abandon OFF |
|---|--:|--:|
| goals ON | 0.184 (Full) | −0.392 (−abandon) |
| goals OFF | −0.002 (−goal) | −0.411 |

Main effect goal-choice +0.102; main effect abandonment +0.493; interaction
+0.166 (= 0.34× the larger main effect). Both ablations carry distinct,
largely additive capability signal; the ablations are not redundant.

## Appendix D. Real-rung permutation null (per fold)

Circular-shift null (preserves EEG autocorrelation + stimulus block
structure), K=200, NeuroTrend Smell_Betula, leave-one-subject-out:

| held-out fold | n_neural | observed | null mean | null p95 | emp p | exp-FDR |
|---|--:|--:|--:|--:|--:|--:|
| Lavrentyev | 70 | 55 | 33.6 | 57 | 0.075 | 0.61 |
| Malyshev | 70 | 38 | 30.6 | 46 | 0.259 | 0.81 |
| Tomilov | 70 | 45 | 36.6 | 55 | 0.318 | 0.81 |
| Trushnikova | 70 | 37 | 30.5 | 50 | 0.413 | 0.82 |

Mean observed 43.8 vs null 32.8 (ratio 1.33×); no fold p<0.05; mean
expected-false-discovery fraction ≈ 0.76. The replication oracle is not
distinguishable from the autocorrelation-preserving null at N=4 → no
agent-performance claims drawn from the real rung.

## Appendix E. AutoDiscovery reproduction (inner-loop reference)

We reproduced AutoDiscovery (Agarwal et al., NeurIPS 2025; open-ended ASD via
Bayesian surprise + MCTS with progressive widening). Cloned
`allenai/autodiscovery`; routed through OpenRouter via four patches (config
base_url; provider-prefix-safe o-series detection in two modules;
env-driven vision model) plus one upstream-autogen fix (None stdout in the
local code executor). Ran on DiscoveryBench `nls_ses` with gpt-4o-mini; smoke
runs cost cents and reproduced the prior/posterior Beta elicitation, KL
Bayesian-surprise, surprisal indicator, and MCTS reward mechanics. Used here
only as an inner-loop reference point (it is goal-bounded per-dataset, no
persistent cross-episode theory), not as an L2 benchmark entry.

**Total API spend** across all experiments ≈ \$2.3 (measured from the
provider balance). The deterministic stdlib harness is free; the cost is
overwhelmingly the powered 5-seed model sweeps and the (failed, ≈ \$0.9 each)
gpt-5.5 reasoning attempts. **Model strings** (OpenRouter ids):
`openai/gpt-4o-mini`, `openai/gpt-4o`, `openai/gpt-5.5` (the last excluded as
an integration failure, see Appendix C).
