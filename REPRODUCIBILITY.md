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

## DiscoveryBench

Canonical result:

- HMS: `29.943879922959418`
- Cons-HMS: `34.96480042505147`
- N: `239`
- Summary: `lmw/universal_discovery_real/research_cycle_discovery_full239_scopegate_intrabundle_20260712/summary.json`
- Predictions: `lmw/universal_discovery_real/research_cycle_discovery_full239_scopegate_intrabundle_20260712/predictions.jsonl`
- Official evaluation: `lmw/universal_discovery_real/research_cycle_discovery_full239_scopegate_intrabundle_20260712/official_eval.jsonl`

Reproduction command:

```bash
python3 -m mars.runners.run_universal_discovery_real_eval \
  --run_id research_cycle_discovery_full239_scopegate_intrabundle_20260712 \
  --model openai/gpt-4o-mini \
  --judge_model openai/gpt-4o \
  --n_proposals 0 \
  --max_rounds 0 \
  --discovery_modules full \
  --overwrite
```

## NewtonBench

Canonical result:

- Audited SA-all: `0.49691358024691357`
- Audited SA-answered: `0.6708333333333333`
- Raw single-run SA-all: `0.4444444444444444`
- N: `324`
- Answered: `240`
- Abstained/no-answer: `84`
- Summary: `lmw/nb_activeprobe/nb_full324_compression_tournament_v9_20260713/summary.json`
- Audit summary: `lmw/nb_activeprobe/nb_full324_compression_tournament_v9_20260713/audited_summary.json`
- Rejudge file: `lmw/nb_activeprobe/nb_full324_compression_tournament_v9_20260713/rejudge_regressions_gpt41_openrouter_x3.json`
- Rows: `lmw/nb_activeprobe/nb_full324_compression_tournament_v9_20260713/rows.csv`

Reproduction command:

```bash
python3 -m mars.runners.run_nb_activeprobe \
  --run_id nb_full324_compression_tournament_v9_20260713 \
  --model openai/gpt-4o-mini \
  --modules m0_gravity,m1_coulomb_force,m2_magnetic_force,m3_fourier_law,m4_snell_law,m5_radioactive_decay,m6_underdamped_harmonic,m7_malus_law,m8_sound_speed,m9_hooke_law,m10_be_distribution,m11_heat_transfer \
  --difficulties easy,medium,hard \
  --law_versions v0,v1,v2 \
  --systems vanilla_equation,simple_system,complex_system \
  --overwrite
```

The audit uses the official NewtonBench `evaluate_law` symbolic prompt/model
path and reruns only the 19 v9-v5 regressions affected by judge/API
instability, with 3x majority voting. The audited count is `161 / 324`.

## UltraHorizon

Canonical result:

- Mean score: `75.36458333333333`
- N: `96`
- Grid: `54.375`
- Seq: `81.25`
- Bio: `90.46875`
- Summary: `lmw/uh_official/uh_clean_universal_full96_20260715/summary.json`
- Combined run log: `lmw/uh_official/uh_clean_universal_full96_20260715/run.jsonl`
- Source env logs:
  - `lmw/uh_official/uh_clean_universal_grid_full32_20260715/run.jsonl`
  - `lmw/uh_official/uh_clean_universal_seq_full32_20260715/run.jsonl`
  - `lmw/uh_official/uh_clean_universal_bio_full32_20260715/run.jsonl`

Reproduction command:

```bash
python3 -m mars.runners.run_uh_official \
  --run_id uh_clean_universal_full96_20260715 \
  --env all \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --difficulty hard \
  --steps 50 \
  --action_budget 220 \
  --generator_model openai/gpt-4o-mini \
  --reflector_model openai/gpt-4o-mini \
  --paper_style_judge \
  --require_paper_judge \
  --disable_env_hints \
  --overwrite
```

The excluded 100.0 UltraHorizon diagnostic ceiling is not a paper result. It is
kept only as an engineering note in `PAPER_SAFE_RESULTS.md`.
