# Paper-Safe RG-HLI Results

This file separates defensible universal-method results from late diagnostic ceiling runs.

## Main Table

| Benchmark | Tier | Metric | Primary | Secondary | N | Notes |
|---|---:|---|---:|---:|---:|---|
| DiscoveryBench | main_universal | HMS / HMS-consistency | 29.944 | 34.965 | 239 | Full 239-task real DiscoveryBench run with the shared slot/contract compiler; no task-specific gold labels in generation. |
| NewtonBench | main_universal_audited | audited SA-all / audited SA-answered | 0.4969 | 0.6708 | 324 | Full 324-task NewtonBench run with the shared universal equation-induction stack; official evaluator re-run by 3x majority on judge calls affected by API instability. Raw single-run score: 0.4444 SA-all. |
| UltraHorizon | main_universal_clean | paper-style judge score | 75.365 |  | 96 | Full 96-task hard UltraHorizon run with paper-style judge, no env hints, and measurement bootstraps disabled. |

## Excluded Diagnostic Ceiling

UltraHorizon late ceiling aggregate: N=96, mean=100.0, distribution={'100.0': 96}.

Excluded reason: Late run uses UH-specific measurement bootstraps/rendering fixes. It is useful as an engineering ceiling, not as the paper's universal-method result.

## Paper Wording

For the main paper, describe RG-HLI as a universal residual-guided hypothesis-language induction system and report the three full-run rows above. Do not present the diagnostic ceiling as SOTA.
