# Headline Claim Audit

This internal manifest maps every result used in the paper to its retained
artifact. It distinguishes benchmark performance from the separate
language-growth mechanism audit. The document is not a benchmark result and is
not used to select operators or rewrite reported scores.

## Full benchmark claims

| Claim in manuscript | Retained artifact | Proposal core | Native evaluator | Evaluation units | Verified point estimate | 95% bootstrap interval |
|---|---|---|---|---:|---:|---|
| DiscoveryBench HMS | `lmw/universal_discovery_real/aaai27_language_growth_protocol_v1_fixed_l0_test239/{summary.json,official_eval.jsonl}` | `openai/gpt-4o-mini` | `openai/gpt-4o` HMS judge | 239 tasks | 28.6989 | [23.2375, 34.2759] |
| DiscoveryBench consistency-HMS | same retained run | `openai/gpt-4o-mini` | consistency-HMS | 239 tasks | 34.5567 | [28.8424, 40.3341] |
| NewtonBench SA-all | `lmw/nb_activeprobe/aaai27_newton_full324_no_promotion_gates_20260717/{summary.json,rows.csv}` | `openai/gpt-4o-mini` | repository `gpt41` symbolic-law judge | 324 tasks | 49.3827% | [44.1358, 54.9383] |
| NewtonBench SA-answered | same retained run | `openai/gpt-4o-mini` | repository `gpt41` symbolic-law judge | 240 submitted laws | 66.6667% | [60.8333, 72.5000] |
| UltraHorizon strict score | `lmw/uh_official/aaai27_uh_full96_marsfull_strict_agent_paperjudge_20260717/{summary.json,run.jsonl}` | `openai/gpt-4o-mini` | reproduced paper-style `DeepSeek-R1-0528-BF16` judge | 96 episodes | 51.0417 | [44.7917, 56.8750] |

Intervals are taken from
`lmw/paper_safe_results/bootstrap_headline_ci_20260729.json`: 10,000
nonparametric percentile draws with seed 2027. DiscoveryBench and NewtonBench
resample completed task rows. UltraHorizon stratifies the resampling by its
three environments, with 32 seeds in each stratum.

## Protocol facts that must remain true in the manuscript

- The DiscoveryBench result is a 239-task run with four per-task proposals and
  one repair round. Cross-task self-module and self-layer promotion are disabled,
  the language-state hash remains unchanged, and the retained evaluator is the
  native HMS judge.
- The NewtonBench result covers all 324 configurations. It contains 240
  submitted laws and 84 abstentions, hence 74.07% coverage. SA-all gives
  abstentions zero credit; SA-answered is conditional on a submitted law.
- The canonical NewtonBench headline run has promotion gates disabled. It is
  evidence for the frozen typed executable kernel, not by itself evidence that
  online language growth caused its SA-all score.
- The UltraHorizon result is exactly the 96-episode hard, fixed-50-step strict
  configuration. Hints, fallback commits, and measurement bootstraps are
  disabled for all 96 rows. Environment means are Grid 59.375, Seq 93.750, and
  Bio 0.000; their equal-weight mean is 51.0417.
- The matched UltraHorizon ablation is
  `lmw/uh_official/uh_full96_alloff_strict_paperjudge_20260716/summary.json`.
  It uses the same GPT-4o-mini core, 96 environment episodes, hard fixed
  50-step horizon, action budget 220, disabled hints/fallbacks, and the same
  paper-style judge. Its 0.00 score is therefore a valid all-components-off
  control, but it is not an external ReAct or CodeAct baseline.
- A retained 100/100 Seq trace is an illustrative episode only. It must never
  be represented as an UltraHorizon average or as evidence about the Bio
  environment.

## Mechanism claim: frozen language growth

The language-growth claim is supported separately by the frozen NewtonBench
audit stored in `paper_assets/evidence/newton_language_growth/` and
`lmw/newton_external/aaai27_newton_language_growth_trajectory_final_v2_20260724/`.
The relevant evidence is not an official benchmark score:

1. Gravity and Coulomb development interfaces yield compatible residuals.
2. A type-compatible positive-monomial / log-coordinate operator is selected
   using held-out utility after its complexity cost.
3. The selected operator is frozen before disjoint transfer evaluation.
4. Across the retained interface-split audit, the selected operator reduces
   executable loss by `0.171 +/- 0.012` relative to the additive control,
   without additional proposal-model calls.

Therefore the paper must phrase the evidence in two layers:

- **Full runs:** one typed executable kernel is evaluated across three native
  benchmark interfaces.
- **Mechanism audit:** residual-conditioned operator selection changes the
  frozen language and improves held-out executable loss under controlled test
  time compute.

The manuscript should not imply that the full NewtonBench SA-all number was
obtained through online promotion on that same reported evaluation set.
