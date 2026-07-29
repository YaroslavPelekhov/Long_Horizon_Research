# Retained RG-HLI Results

These are the complete benchmark rows used by the submission.

| Benchmark | Evaluation units | Native metric | Result |
|---|---:|---|---:|
| DiscoveryBench | 239 tasks | HMS / Cons-HMS | 28.70 / 34.56 |
| NewtonBench | 324 tasks | SA-all / SA-answered | 49.38% / 66.67% |
| UltraHorizon | 96 episodes | strict paper-style score | 51.04 |

The headline language is frozen before evaluation. DiscoveryBench uses four
proposals and one search round. NewtonBench submits 240 laws and abstains on 84
tasks, giving 74.1% coverage. UltraHorizon uses 32 hard seeds in each of three
environments, a 50-step horizon, disabled environment hints, and no fallback
commit.

Every row is linked to its complete evaluator artifact and SHA-256 digest in
`paper_assets/evidence/evidence_index.json`. Development-time language growth
is evaluated separately through disjoint induction, promotion, and transfer
interfaces; benchmark test scores never update the language.
