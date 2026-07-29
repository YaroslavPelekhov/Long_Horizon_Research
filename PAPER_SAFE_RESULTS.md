# Paper-Safe RG-HLI Results

This file separates defensible universal-method results from late diagnostic ceiling runs.

## Main Table

| Benchmark | Tier | Metric | Primary | Secondary | N | Notes |
|---|---:|---|---:|---:|---:|---|
| DiscoveryBench | headline | HMS / consistency-HMS | 28.699 | 34.557 | 239 | Complete 239-task native-HMS evaluation with four per-task proposals and a frozen hypothesis language. |
| NewtonBench | headline | SA-all / SA-answered | 49.383 | 66.667 | 324 | Complete 324-configuration frozen-kernel evaluation; 240 submitted laws and 84 abstentions. |
| UltraHorizon | headline strict | paper-style score | 51.042 |  | 96 | Complete 96-episode hard run: 50 steps, no hints, fallback commits, or measurement bootstraps. |

## Excluded Diagnostic Ceiling

UltraHorizon late ceiling aggregate: N=96, mean=100.0, distribution={'100.0': 96}.

Excluded reason: Late run uses UH-specific measurement bootstraps/rendering fixes. It is useful as an engineering ceiling, not as the paper's universal-method result.

## Complete proposal-core diagnostics

These rows support provider robustness and are not substituted for the
headline configurations.

| Benchmark | Proposal cores | Complete units | Result range |
|---|---:|---:|---|
| DiscoveryBench | 3 | 239 each | 28.58--29.14 HMS |
| NewtonBench diagnostic kernel | 5 | 324 each | 94--95 correct laws |
| UltraHorizon controller-enabled | 3 | 96 each | 24.64--75.36 overall |

The machine-validated table and source hashes are stored in
`paper_assets/evidence/model_substitution_audit.json`.

## Paper Wording

For the main paper, describe RG-HLI as a residual-guided hypothesis-language induction system and report the three headline rows above. Do not present the diagnostic ceiling as SOTA.
