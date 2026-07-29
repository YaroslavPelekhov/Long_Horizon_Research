# RG-HLI Reproducibility Notes

This file records the commands and artifacts for the current paper-safe RG-HLI
results. The JSON/CSV/JSONL artifacts listed here are the canonical evidence
for the paper tables.

## Environment

Set credentials through environment variables, never through committed files:

```bash
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

The paper runs use `openai/gpt-4o-mini` as the small-model core. DiscoveryBench
uses `openai/gpt-4o` as the HMS judge. UltraHorizon uses the paper-style judge
path exposed by `mars.runners.run_uh_official`.

The official benchmark repositories are not redistributed. Clone or unpack
them directly under the project root using the following directory names:

```text
discoverybench_repo/
newtonbench_repo/
ultrahorizon_repo/
```

The released runners resolve these locations relative to the project root.

## DiscoveryBench

Canonical result:

- HMS: `28.698944012751546`
- Cons-HMS: `34.5566845985256`
- N: `239`
- Summary: `lmw/universal_discovery_real/aaai27_language_growth_protocol_v1_fixed_l0_test239/summary.json`
- Predictions: `lmw/universal_discovery_real/aaai27_language_growth_protocol_v1_fixed_l0_test239/predictions.jsonl`
- Official evaluation: `lmw/universal_discovery_real/aaai27_language_growth_protocol_v1_fixed_l0_test239/official_eval.jsonl`

Reproduction command:

```bash
MARS_PROMOTE_SELF_MODULES=0 MARS_PROMOTE_SELF_LAYERS=0 \
python3 -m mars.runners.run_discovery_chunked_eval \
  --run_id aaai27_language_growth_protocol_v1_fixed_l0_test239 \
  --data_split test \
  --model openai/gpt-4o-mini \
  --judge_model openai/gpt-4o \
  --n_proposals 4 \
  --max_rounds 1 \
  --workers 8 \
  --overwrite
```

## NewtonBench

Canonical result:

- SA-all: `0.49382716049382713`
- SA-answered: `0.6666666666666666`
- N: `324`
- Answered: `240`
- Abstained/no-answer: `84`
- Summary: `lmw/nb_activeprobe/aaai27_newton_full324_no_promotion_gates_20260717/summary.json`
- Rows: `lmw/nb_activeprobe/aaai27_newton_full324_no_promotion_gates_20260717/rows.csv`

Reproduction command:

```bash
python3 -m mars.runners.run_nb_activeprobe \
  --run_id aaai27_newton_full324_no_promotion_gates_20260717 \
  --model openai/gpt-4o-mini \
  --modules m0_gravity,m1_coulomb_force,m2_magnetic_force,m3_fourier_law,m4_snell_law,m5_radioactive_decay,m6_underdamped_harmonic,m7_malus_law,m8_sound_speed,m9_hooke_law,m10_be_distribution,m11_heat_transfer \
  --difficulties easy,medium,hard \
  --law_versions v0,v1,v2 \
  --systems vanilla_equation,simple_system,complex_system \
  --disable_promotion_gates \
  --overwrite
```

This is the retained single-pass evaluator result. The flag disables optional
inference-time residual-risk holds; it is distinct from the development-time
held-out operator-promotion protocol, which is frozen during headline
evaluation.

## UltraHorizon

Canonical result:

- Mean score: `51.041666666666664`
- N: `96`
- Summary: `lmw/uh_official/aaai27_uh_full96_marsfull_strict_agent_paperjudge_20260717/summary.json`
- Combined run log: `lmw/uh_official/aaai27_uh_full96_marsfull_strict_agent_paperjudge_20260717/run.jsonl`

Reproduction command:

```bash
python3 -m mars.runners.run_uh_official \
  --run_id aaai27_uh_full96_marsfull_strict_agent_paperjudge_20260717 \
  --env all \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --difficulty hard \
  --steps 50 \
  --generator_model openai/gpt-4o-mini \
  --reflector_model openai/gpt-4o-mini \
  --paper_style_judge \
  --require_paper_judge \
  --disable_env_hints \
  --disable_fallback_commit \
  --overwrite
```

The command above is the strict, frozen-language headline configuration.

## Headline manifest

Before packaging a submission, rebuild the source-hashed manifest that checks
the retained benchmark cardinalities and strict UltraHorizon settings:

```bash
python3 -m mars.analysis.build_submission_manifest \
  --discovery lmw/universal_discovery_real/aaai27_language_growth_protocol_v1_fixed_l0_test239/summary.json \
  --newton lmw/nb_activeprobe/aaai27_newton_full324_no_promotion_gates_20260717/summary.json \
  --ultrahorizon lmw/uh_official/aaai27_uh_full96_marsfull_strict_agent_paperjudge_20260717/summary.json \
  --output_dir paper_assets/evidence/headline_manifest
```

The manifest recomputes NewtonBench SA from the retained 324-row evaluator log
rather than the rounded fields in its run summary. It does not execute model
calls or modify benchmark results.

## Proposal-core portability audit

The portability audit changes only the proposal endpoint within a frozen
benchmark configuration and keeps the task set, language, executable
interfaces, parser, and native evaluator fixed. The audit is intentionally
separate from the three headline configurations.

Validate and export the 11 complete rows:

```bash
python -m mars.analysis.build_model_substitution_audit
python -m mars.analysis.build_paper_evidence_index
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_model_substitution_audit.py \
  tests/test_paper_evidence_index.py -q
```

The resulting `paper_assets/evidence/model_substitution_audit.json` stores the
protocol label, task cardinality, metric, source path, and SHA-256 digest for
every row. The evidence index additionally includes the complete
DiscoveryBench native-evaluator JSONL records for both new endpoints.
