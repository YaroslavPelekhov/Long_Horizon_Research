# AutoDiscovery → LMW Adapter (v0)

## What this is, honestly

A faithful re-implementation of the **AutoDiscovery algorithm** (Agarwal et al.,
NeurIPS 2025: open-ended discovery via *Bayesian surprise* + *MCTS with
progressive widening* + *LLM belief elicitation*) on top of LMW's `World` API.

This is **not** a wrap of the original AutoDiscovery codebase. We considered
three integration paths and chose this one with eyes open:

| Path | What it measures | v0 feasibility | Verdict |
|---|---|---|---|
| (a) Wrap their codebase, feed LMW samples as a CSV (passive only) | Their system on data **without** `do()` — by construction it falls into the trap | Cheap | **Misleading** — sets the published system up to fail |
| (b) Wrap their codebase, inject `do()` helpers into their code-execution sandbox | Their *implementation* on our interventional API | Weeks of integration, high cost | Defer |
| (c) **Re-implement the published algorithm** against `World.observe/intervene`, label as such | The *algorithm* on the LMW action space | Bounded engineering, cents-of-API | **Chosen for v0** |

The leaderboard entry is therefore labelled
`AutoDisc-algo (adapter, Agarwal et al. 2025)` — never just `AutoDiscovery`.
This avoids implying we benchmarked their codebase.

## Contract (matches `BENCHMARK_PROTOCOL_v0.md` §3)

- **Reads only the allowed surface.** `world._scm.structure.subdomains`
  (opaque cluster ids → variable names — allowed), `world.observe`,
  `world.intervene`, `world.budget_left/total`. Forbidden: `Oracle.*`,
  `Structure.roles/edges/spurious_pair/true_cause/fake_const`, raw `Edge.coeff`.
- **Emits claims** in the strict grammar
  `causal(A,B,+) | causal(A,B,-) | no_effect(A,B) | confounded(A,B,Z) |
  mean(V)~const` with **honest provenance** — every claim carries the `eid`s
  of the world experiments that produced its supporting evidence.

## Algorithm summary (what gets implemented)

Per the paper:

1. **Hypothesis generator** (LLM): given the cluster/variable surface and the
   path of prior hypotheses, propose ONE new hypothesis in the strict grammar.
2. **Experiment** (our world): parse the hypothesis statement and execute the
   corresponding `intervene/observe` call(s); record the resulting effect /
   correlation / mean. The eids are stored as provenance.
3. **Belief elicitation** (LLM): sample n boolean answers (true/false) to "is
   the hypothesis true?" both **before** the experiment (prior) and **after**
   (posterior, conditioning on the experimental evidence). Count → Beta(α,β).
4. **Bayesian surprise**: KL divergence between posterior Beta and prior Beta.
   A claim is committed to the `ClaimStore` iff the surprise crosses
   `kl_thresh` (analogue of `surprisal_width`).
5. **MCTS with progressive widening**: nodes = hypotheses, parent expansion
   limited to `k · N^α` children; selection by UCB1 (avg surprisal + explore
   term); reward backpropagated. Loop until budget exhausted or call cap.

The Beta-Bernoulli + KL math follows the paper (§3.1); we use
`scipy.special.betaln/digamma` when available, with a math fallback.

## v0 hyperparameters (cost-disciplined)

- `model = openai/gpt-4o-mini` (env-configurable via `AUTODISC_MODEL`).
- `max_iterations = 12`, `n_belief_samples = 3`, `k = 1.0`, `α = 0.5`,
  `surprisal_kl = 0.1`, `ucb_c = 1.0`, hard LLM-call cap = 40.

## Expected cost (declared up front)

Per `(seed × condition)` run: roughly `12 propose calls + 12 × 2 belief
elicitations × 3 samples ≈ 80–100` LLM calls at gpt-4o-mini ≈ **\$0.05–0.15
per seed**. Leaderboard v0 entry on Open-Ended LMW with 3 seeds → expected
**\$0.15–0.45** total. Reported alongside the result per protocol §4.

## Leaderboard placement

- **Category B** (LLM-driven). **Compute tier T1** (~\$0.10–\$1 per evaluation).
- Two declared variants: `AutoDisc-algo (with memo)` and
  `AutoDisc-algo (no memo)` — same mandatory `−mem` ablation as any other
  submission (the memo carries an abstract strategy across stages on
  Open-Ended LMW, just as in our other LLM agents).
- Label string in the leaderboard MUST include "(adapter, not original
  codebase)" so a reader cannot mistake this for a measurement of the
  published implementation.

## Convergent-validity role

This is the v0 anchor we committed to in `BENCHMARK_PROTOCOL_v0.md` §9.4 — a
single real published algorithm on the leaderboard alongside the scripted
ladder, against which Ordering A (literature) can be rank-correlated. The
strong form (multiple published systems) accumulates via community
submissions; this is the seed.

## Honest limitations

- Algorithm faithfulness ≠ implementation faithfulness. Specific scaffold
  choices in the original codebase (their experiment_programmer prompts,
  their re-test/dedup logic) are not replicated 1:1.
- LLM hypothesis generation is intrinsically stochastic; per-seed numbers
  will be noisy at small N. v0 reports per-seed (per protocol §4).
- The published algorithm is goal-driven within a single dataset; running it
  per-stage in Open-Ended LMW is itself an extension we declare.

## Files

- `lmw/autodisc_agent.py` — the adapter agent class.
- `lmw/run_autodisc.py` — runner script: Open-Ended LMW × seeds × {mem,nomem}
  → JSON dump + leaderboard line, mirroring `run_open_llm.py`.

## Next step (gated on user confirmation of API cost)

Implementation skeleton lands first (this turn); a live run on Open-Ended
LMW (3 seeds × 2 conditions, ≈ \$0.30–0.90 expected) is the next step.
