# RG-HLI A* Evidence Matrix

This file records the current defensible evidence package for the RG-HLI paper.
It is intentionally stricter than the experimental log: only full-run,
paper-safe rows are treated as main claims.

## Central Claim

RG-HLI is a universal residual-guided hypothesis-language induction system. The
same hypothesis state is used across scientific discovery, symbolic law
induction, and long-horizon rule discovery. Benchmark-specific adapters expose
observations and native scoring only; the core loop remains:

1. evidence contract compilation,
2. typed hypothesis artifact construction,
3. executable slot closure,
4. metamorphic and contract validation,
5. typed residualization,
6. held-out operator promotion with a complexity cost.

The paper claim is not that every benchmark is solved. The claim is that
small-model reasoning improves when hypothesis construction is externalized as
typed executable induction and failures update the hypothesis language rather
than remaining free-form feedback.

## Canonical Main Rows

| Benchmark | N | Metric | Primary | Secondary | Canonical artifact | Status |
|---|---:|---|---:|---:|---|---|
| DiscoveryBench | 239 | HMS / Cons-HMS | 29.9439 | 34.9648 | `lmw/universal_discovery_real/research_cycle_discovery_full239_scopegate_intrabundle_20260712/summary.json` | main |
| NewtonBench | 324 | audited SA-all / audited SA-answered | 0.4969 | 0.6708 | `lmw/nb_activeprobe/nb_full324_compression_tournament_v9_20260713/audited_summary.json` | main |
| UltraHorizon | 96 | paper-style score | 75.3646 |  | `lmw/uh_official/uh_clean_universal_full96_20260715/summary.json` | main |

## Per-Benchmark Artifact Audit

| Benchmark | Required evidence | Present artifacts | Current assessment |
|---|---|---|---|
| DiscoveryBench | full predictions, official evaluation, summary, judge model, task count | `predictions.jsonl`, `official_eval.jsonl`, `diagnostics.json`, `eval_trace.log`, `summary.json` | Ready as a full 239-task paper-safe row. Needs exact launch command copied into a reproducibility appendix. |
| NewtonBench | full 324-row result grid, answered/abstained counts, SA-all, SA-answered, audit trace | `rows.csv`, `summary.json`, `audited_summary.json`, `rejudge_regressions_gpt41_openrouter_x3.json` | Ready as the canonical universal audited row. The raw v9 run was affected by judge/API instability; the audit reruns only affected regressions with the official evaluator path. |
| UltraHorizon | full 96-run hard protocol, per-env scores, paper-style judge, no env hints, no measurement bootstrap | `run.jsonl`, `summary.json` | Ready as clean universal full96. The 100.0 diagnostic ceiling must remain excluded from all SOTA claims. |

## Excluded Rows

| Benchmark | Artifact | Score | Exclusion reason |
|---|---|---:|---|
| UltraHorizon | `lmw/uh_official/uh_full_official_nohint_grid_32seeds_priorfix_20260714/summary.json` and related ceiling aggregate | 100.0 | Uses late UH-specific measurement bootstraps/rendering fixes. Useful as an engineering ceiling, not as a universal-method paper claim. |

## Result Interpretation

DiscoveryBench is the strongest support for the residual-guided hypothesis
language claim: full-run HMS is above the listed published references, and the
component ablation shows a large gap between strict-none and full RG-HLI.

NewtonBench exceeds the o4-mini reference after consistency audit but not frontier
agents. It is the clearest evidence that universal executable scaffolding helps
only when the system can generate or induce the right representation.

UltraHorizon is strong under the clean paper-style full96 protocol, but the
environment breakdown matters: Grid remains weaker than Seq and Bio. The paper
should report the breakdown to avoid a misleading single-score story.

## Current A* Gaps

| Gap | Why it matters | Action |
|---|---|---|
| Exact commands are not centralized for every canonical run. | Reproducibility reviewers need a clear command path from code to result. | Add a reproducibility appendix with commands or runner configs for the three canonical artifacts. |
| Ablations are uneven across benchmarks. | A* reviewers will ask whether the claimed mechanism, not incidental engineering, explains gains. | Keep DiscoveryBench as the primary component ablation and add smaller cross-benchmark ablations only where they are protocol-clean. |
| NewtonBench raw single-run judge instability lowers the unaudited v9 score. | Reviewers need to understand why the canonical number differs from the raw single pass. | Report the audited full-324 result with raw and rejudge artifacts, and keep the older no-chart run as a robustness reference rather than the main Newton row. |
| UltraHorizon has a tempting excluded 100.0 ceiling. | Using it would weaken credibility. | Keep the 75.36 clean universal row as the main result and mention the ceiling only in internal engineering notes, not in the paper. |
| Method name transition is incomplete in legacy logs. | Mixed naming makes the project look unstable. | Use RG-HLI in paper, README, and paper-safe files; leave old run IDs unchanged when they are filesystem identifiers. |

## Paper-Safe Wording

Use: "RG-HLI obtains 29.94 HMS on DiscoveryBench, 49.7% audited SA-all on NewtonBench,
and 75.36 on UltraHorizon under the clean paper-style protocol."

Do not use: "solves UltraHorizon", "beats SOTA on all benchmarks", "fully
self-improving", or any claim based on the excluded 100.0 diagnostic ceiling.
