# Open-Ended LMW v0 — spec

## The one design rule (prevents repeating the L2 failure)

Separate **what is true** from **what is revealed**.

- The full world `W(master_seed)` is an *infinite, frozen, deterministic*
  object. Every fact is a pure function of `(master_seed, address)`. It NEVER
  depends on the agent's history. → ground truth is well-defined; the
  consistency invariant holds by construction.
- "Competence-tied growth" = the agent's **horizon** expands to the next
  pre-determined stage when it has mastered the current one. The newly
  revealed stage was *always* true; the agent's competence gates only *when it
  becomes scored/reachable*, never *what it is*.

Truth frozen, horizon moves. (If truth ever depended on the agent → L2 redux.)

## Structure

- **Curriculum**: a deterministic, monotonically harder sequence of stages
  `k = 0,1,2,…`. `schema_k` is a fixed difficulty schedule (longer hidden
  chain, more decoys/latents, regime shift switches on). `structure_k =
  make_structure(seed=H(master_seed,k), schema_k)`, variable names namespaced
  `s{k}.X…` so stages never collide and a persistent theory stays unambiguous.
- **Persistent theory across stages**: one `ClaimStore` carried through the
  whole curriculum. This is what makes axis-2 (persistent revisable theory)
  have *long-horizon* teeth: an agent that cannot carry validated theory
  forward re-derives each stage and fails the gate sooner.
- **Mastery gate**: advance `k → k+1` iff the agent's validated knowledge of
  stage `k` (per-stage `axis1_agenda_ratio` = honest replicated value / full-
  SCM oracle) ≥ τ within its per-stage budget. Computed by the existing,
  penalty-robust scorer. Deterministic given the run.
- **Score**: area of validated-retained knowledge summed over stages reached,
  + depth reached. Unbounded, competence-tied, no ceiling for strong agents.

## Why this answers the open problems

- **Contamination**: world is `master_seed`-infinite and lazily materialized;
  issue a fresh seed per evaluation → no fixed test set exists to leak.
  Stronger than rolling real data; needs no private data.
- **No ceiling**: a better outer-loop agent advances through more stages →
  strictly higher score; metric stays discriminative for any future model.
- **Ground truth**: per-stage = full known SCM (upper bound, not self-runs →
  not circular).

## Validity guardrails (built in from day 0 — our L2 insurance)

1. **Consistency invariant**: stage-`k` truth is identical regardless of probe
   order, traversal path, or how many later stages were materialized first.
2. **Self-consistency autotest** (CI-style, fails the build if violated):
   materialize random `(master_seed,k)` via independent calls / different
   orders / different frontier sizes → assert identical structure & oracle.
   This is the structural check L2 lacked.
3. **Construct-validity protocol carries over** and must pass on the new
   generator before any claim: counterfactual ablations (Full > −mem/−goal/
   −abandon > naive), held-out curriculum schedule (train on one difficulty
   progression, test on an unseen one), integrity-weight sensitivity, and
   **gate-τ sensitivity** (ordering robust across τ — the new researcher DoF).

## v0 scope (minimal, reuses validated code)

`openworld.py`: OpenEndedLMW (stage schedule, namespaced lazy structures,
per-stage World+Oracle, persistent ClaimStore, mastery gate, run loop) +
`consistency_selftest()`. `run_open.py`: counterfactual harness over the
curriculum (Full vs −mem/−goal/−abandon vs naive), reports stages-reached and
total score. No API, deterministic. Pass autotest + show Full reaches deeper
than −mem (axis-2 now long-horizon) before any claim.
