# LMW Benchmark — Submission Protocol & Convergent-Validity Plan (v0)

This document specifies (i) how a third party submits an agent to the
benchmark, (ii) how we score and audit the submission, and (iii) the
procedure by which we evaluate convergent validity against existing
literature without claiming to run every SOTA system ourselves. It is a
*living* protocol; the version + git hash pinned in §13 is the frozen
snapshot a submission is evaluated against.

---

## 1. Purpose and scope

The benchmark measures the **outer loop** of autonomous research (autonomous
agenda + persistent revisable theory + portfolio planning under delayed
reward; see paper §2). Submissions are agents that produce claims into a
typed store while interacting with a domain-agnostic `WorldSource`. The
primary scored rung is **Open-Ended LMW** (contamination-immune by
construction); L1 schema-family is an additional sanity rung; the real-data
rung is a cautionary pilot (paper §5.4) and is **not** scored for
submissions in v0.

Out of scope for v0: real-data-rung scoring; multi-domain (physics/bio)
adapters (`WorldSource` makes them future additions, not redesigns).

---

## 2. The frozen benchmark instance

A submission is scored against a *frozen configuration* (so submissions
across labs are comparable):

- **Code anchor.** `lmw/` at the protocol's pinned git hash (§13). The
  consistency self-test (`openworld.consistency_selftest`) must pass against
  the pinned code; submissions assert this.
- **Difficulty schedule** (`openworld._schema_k`, monotone in `k`,
  pinned exactly as in code).
- **Mastery gate τ.** Pinned at `0.15` for the leaderboard. τ-robustness
  across `[0.05, 0.40]` is independently validated (paper §5.2) and
  submissions need not re-establish it.
- **Per-stage budget.** `Oracle.reference_budget() × TIGHTNESS` with
  `TIGHTNESS = 1.05` (`harness._budget_for`).
- **Max stages.** `openworld.MAX_STAGES = 4` in v0.
- **Seeds.**
  - *Public dev seeds* (visible, for tuning): `1..20`.
  - *Hidden test seeds* (released only at evaluation time, or run by an
    eval-server): `100..120`. **Submissions MUST NOT tune against test
    seeds.** Reported numbers are computed on test seeds.
- **Metric of record.** Mean **continuous total RPS** over the hidden test
  seeds, as defined in `scorer.score` and `openworld.run_curriculum`.
  Secondary numbers: depth, axis-1/2/6 metrics, integrity penalty.

---

## 3. Agent contract (the API a submission may use)

A submission is a Python object exposing `.run(world, store)` where:

- `world: WorldSource` (see `lmw/adapters/base.py`). The agent may call only
  the public surface: `clusters()`, `observe(cluster)`,
  `intervene(cause, target, n)`, `budget_left`, `budget_total`.
- `store: ClaimStore` (`lmw/claims.py`). The agent populates it with claims
  in the **strict claim grammar**:
  `causal(A,B,+) | causal(A,B,-) | no_effect(A,B) | confounded(A,B,Z) | mean(V)~const`.
  Every claim must carry honest provenance — a list of `eid`s from
  experiments the agent actually ran in this world.

**Forbidden inputs / actions** (cause for disqualification):

- Reading any ground-truth surface: `WorldSource.replicates`,
  `WorldSource.value`, `WorldSource.oracle_value`, `Oracle.*`,
  `Structure.spurious_pair`, `Structure.true_cause`, `Structure.roles`,
  `Structure.edges`, `Structure.fake_const`, raw `Edge.coeff`. The agent
  sees variable names and cluster ids; *roles* and *truth* are scorer-only.
- Probing future stages before mastering the current one (e.g., calling
  `build_stage(seed, k+1)` from inside the agent).
- Persisting cross-evaluation state, retrieving the held-out seeds, or
  fingerprinting the seed/structure.

**Determinism.** The agent must be deterministic given (test seed, declared
agent seed, declared model version). Stochastic agents declare their RNG
seed; LLM agents declare exact model id (e.g. `openai/gpt-4o-2024-11-20`) and
sampling parameters.

---

## 4. Submission package (mandatory artifacts)

A submission PR/upload must include:

1. **Agent code** + minimal runner that takes `(world, store)`.
2. **All numbers**: per-test-seed `(seed, depth, total_RPS, axis1_ratio,
   axis6_deadend_regret, integrity_penalty, final_true_value)` plus their
   means.
3. **ClaimStore JSON** for each run (statements, provenance eids,
   budget-stamps, status — for audit).
4. **Counterfactual ablations.** Required: the submission's own `−mem`
   variant (axis-2 ablated, however the submitter implements it) and, where
   meaningful, `−goal` and `−abandon`. The leaderboard reports
   `value(capability) = RPS(Full) − RPS(−capability)` for each.
5. **Audit output.** `scorer.score` results + a tool-emitted log proving:
   (i) consistency self-test passed against the pinned code; (ii)
   provenance honesty audit passed for every claim; (iii) no forbidden-API
   call (a sandboxed runner verifying allowed-surface attribute access is
   provided).
6. **Compute & cost receipt.** Total API spend, tokens, wall-clock; model
   strings; hardware tier.
7. **Category declaration** (§7).

---

## 5. Mandatory ablations (defense against scaffold-engineering)

A bare RPS number is gameable by scaffold/prompt-engineering specific to the
benchmark surface. Requiring the submitter to publish its `−mem` (and where
meaningful `−goal`, `−abandon`) means a high RPS coupled with small
ablation deltas is *itself* a red flag: the agent is not exercising the
capabilities the benchmark targets, only their scaffold. The leaderboard
displays the ablation deltas alongside the headline RPS for this reason.

---

## 6. Audit & integrity

- **Provenance audit.** Every claim's cited `eid`s must exist in
  `world.log` and be of the kind that could support the statement (e.g., a
  `causal/no_effect/confounded` claim must cite at least one intervention on
  the named cause). Dishonest provenance → claim discarded; if pervasive →
  submission rejected.
- **Integrity penalty** (`scorer.FALSE_PENALTY`, default 1.5) penalises
  un-retracted false beliefs. Reported as a separate column; not negotiable.
  Penalty-weight robustness validated in paper §5.1 (RPS ordering invariant
  for `FALSE_PENALTY ∈ [0, 5]`).
- **Consistency self-test.** Must pass against the pinned code, asserting
  stage truth is a pure function of `(master_seed, k)` and order-independent
  in materialisation.
- **No seed shopping.** Submissions report all test-seed runs; "best of N"
  is forbidden.
- **No truth leakage.** A sandbox runner restricts the agent's attribute
  access to the allowed surface in §3.

---

## 7. Submission categories

The leaderboard buckets entries so a $0 scripted agent and a $50 frontier
LLM are not compared without context:

- **A. Reference / scripted** — hand-coded; no learned components.
- **B. LLM-driven** — declared model + version + sampling.
- **C. Hybrid / learning** — any agent that updates parameters or memory
  beyond a single run.

Plus a **compute-tier tag**: `T0` (< \$0.10 per evaluation),
`T1` (\$0.10–\$1), `T2` (\$1–\$10), `T3` (> \$10). RPS is reported per
category-tier; a global "Pareto frontier" view ranks across tiers.

---

## 8. Submission process

- **Public dev evaluation** (open): submitter runs locally on dev seeds
  `1..20`; numbers self-reported in the PR for peer review.
- **Hidden test evaluation** (gated): an eval-server runs the submitted
  agent on hidden seeds `100..120`; the server returns the audited numbers
  and writes them to the public leaderboard. The hidden seeds are rotated
  on a schedule (annually) so the benchmark does not become a fixed test
  set submitters can over-fit to over time — analogous to LiveBench's
  rolling renewal.
- **Reproducibility window.** Submissions must remain reproducible from
  their declared code/version for 12 months; if the model version is
  retired by the provider, submitters annotate accordingly.

---

## 9. Convergent-validity plan

Submissions accrue into a leaderboard ordering. Convergent validity asks
whether that ordering agrees with independent, literature-derived orderings
of capability.

### 9.1 Two orderings

- **Ordering A — literature-derived (independent of our metric).**
  We commit *up front* to multiple operationalisations of A (no single
  cherry-picked one):
  - **A1: Field-map axis coverage.** Number of the seven outer-loop axes a
    system addresses (●=1, ◐=0.5, ○=0; see paper §2.3 / Appendix A).
  - **A2: Independent peer-review signal.** Whether the system has a
    paper at a major venue (NeurIPS / ICLR / ICML / ACL); a coarse 0/1 (or
    main-conference vs workshop coded).
  - **A3: Validated discoveries.** Public count of independently-validated
    discoveries the system produced (where reported), normalised by
    declared budget.
  - **A4: Independent benchmark wins** (AstaBench, DiscoveryBench, etc.)
    where the system has a comparable score.

- **Ordering B — our metric.**
  Mean continuous total RPS on the hidden test seeds, taken from the
  leaderboard.

### 9.2 Reporting

For each Ai we report **Spearman ρ** between Ai-rank and B-rank, with a
bootstrap 95% CI (re-sample systems) and an exact permutation p-value (the
distribution is small; permutation is honest at small N). We **pre-register**
that "convergent validity holds for axis Ai" iff ρ > 0 with 95% CI not
crossing 0 (i.e. not merely directionally positive). We report all Ai,
including ones that fail — no cherry-picking.

The headline number for the paper is the **median ρ across A1–A4** and the
fraction of Ai's that pass; the verdict is one of {strong, partial,
no} convergent validity, defined ex ante in §9.3.

### 9.3 Pre-registered verdicts

- **Strong**: ≥ 3 of 4 Ai pass (ρ > 0, 95% CI excludes 0).
- **Partial**: exactly 2 of 4 pass.
- **No / weak**: ≤ 1 passes. We report it as such, not bury it.

### 9.4 v0 anchor (what is feasible now, without months of integration)

At protocol v0 we publish initial Ordering-B entries built from:

- Scripted ladder (`Full`, `−mem`, `−goal`, `−abandon`, `naive`,
  `GreedyObs`, `RandomAgent`) — their a-priori capability ordering is
  unambiguous from design and yields a high internal-consistency anchor.
- **AutoDiscovery** (Agarwal et al., NeurIPS 2025; already cloned, patched,
  runnable in our repo) wrapped as a `WorldSource` agent — one real
  published system on the leaderboard. We commit to this as v0's reality
  check; further real systems are community submissions over time.

This is honest: v0 convergent validity is anchored on a small set; the
strong form depends on the community submitting real SOTA. The paper
states this explicitly rather than over-claiming a literature correlation
established by us alone.

### 9.5 Divergent validity (no free lunch)

We also report at least one *negative* expectation: a system / setting
where our metric should NOT track an external ordering (e.g., a
single-paper benchmark whose task structure is orthogonal to the outer
loop). If our metric happens to correlate there too, that itself is a
warning we must explain — convergent without divergent is not validity.

---

## 10. Pre-registration discipline for our own future SOTA agent

To prevent the "designed both the benchmark and the winner" critique:

- The protocol version + git hash in §13 is the **frozen snapshot** any
  future work by the benchmark authors is evaluated against.
- An author-built agent is developed and tuned **only on the public dev
  seeds** (`1..20`). Final reported numbers are run on the hidden test
  seeds and never optimised against them; the run is performed by the
  eval-server, not the authors.
- An author-built agent is identified as such in the leaderboard
  (`category: A/B/C, authors=yes`) so reviewers can discount.
- Any change to scorer / generator / gate after a frozen snapshot bumps the
  protocol version; prior leaderboard numbers stay attached to the version
  they were measured under (no silent re-scoring).

---

## 11. Anti-gaming summary

- Hidden, rotated test seeds + no "best of N" + no seed-shopping.
- Mandatory ablations — RPS without small `−mem`/`−goal`/`−abandon` deltas
  is suspect (scaffold-engineering).
- Provenance audit + integrity penalty (penalty-robust ordering, §5.1).
- Consistency self-test (truth = pure fn of seed) — submissions cannot
  exploit construction-order side-effects, because there are none by
  construction.
- Compute-tier disclosure — bigger model alone is not capability (see paper
  §5.2: `gpt-4o` strictly worse than `gpt-4o-mini` on continuous RPS via
  over-claiming).
- Sandbox runner restricts attribute access to the allowed agent surface.

---

## 12. Known limitations / open issues (honestly)

- v0 convergent validity is anchored on a scripted ladder + AutoDiscovery.
  Strong-form correlation against the literature depends on community
  submissions accumulating real systems over time. We do not pretend
  otherwise.
- The four Ai operationalisations are themselves not gold-standard; we
  expect honest debate about each. We report all four, not a chosen one.
- The compute tier is coarse; a system spending \$8 vs \$2 may sit in the
  same tier despite a 4× capability advantage *budget*. Future versions
  may refine tiers.
- Hidden-seed rotation cadence (annual) is a policy choice and may need
  shortening if leaderboard saturation occurs faster.
- The benchmark currently has no learning-agent reference that exercises
  cross-stage axis-2 on depth; that is the **posed open challenge** (paper
  §6b), not a closed result.

---

## 13. Versioning & changelog

- **v0.1 (this document).** Protocol drafted from paper rev.2.
- **Frozen code snapshot** (pinned): `lmw/` at commit
  `daf85268c467e0e41680ed2b7f27e95046bfaa02` (git tag `v0.1`). Submissions
  against protocol v0.1 are scored on this commit; any change to scorer /
  generator / gate τ / budget formula bumps the protocol version and is
  recorded in this changelog with a new hash.
- **First leaderboard cut** (planned): scripted ladder + AutoDiscovery
  wrapped to `WorldSource`, against dev seeds `1..20` and hidden seeds
  `100..120`.

Changes to scorer, generator, gate τ, or budget formula bump the protocol
version (and may invalidate prior numbers; the changelog will say so).
