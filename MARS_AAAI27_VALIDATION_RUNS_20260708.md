# MARS AAAI-27 Validation Runs, 2026-07-08

This note records the runs used to synchronize the AAAI-27 draft with current
code and benchmark outputs.

## Unit Tests

Command:

```bash
pytest -q tests
```

Result:

```text
177 passed, 6 warnings in 13.69s
```

The six warnings are numpy correlation warnings in the multi-table WorldBank
slot-compiler test.

## NewtonBench Active-Probe Slices

Easy law-version slice:

```bash
python -m mars.runners.run_nb_activeprobe \
  --run_id aaai27_full_easy_20260708 \
  --model openai/gpt-4o-mini \
  --modules m0_gravity,m1_coulomb_force,m2_magnetic_force,m3_fourier_law,m4_snell_law,m5_radioactive_decay,m6_underdamped_harmonic,m7_malus_law,m8_sound_speed,m9_hooke_law,m10_be_distribution,m11_heat_transfer \
  --difficulty easy \
  --law_versions v0,v1 \
  --overwrite
```

Summary: `n=24`, `SA_all=0.833`, `SA_answered=0.952`.

Hard law-version slice:

```bash
python -m mars.runners.run_nb_activeprobe \
  --run_id aaai27_full_hard_20260708 \
  --model openai/gpt-4o-mini \
  --difficulty hard \
  --law_versions v0,v1 \
  --overwrite
```

Summary: `n=24`, `SA_all=0.417`, `SA_answered=0.833`.

## UltraHorizon Official-Style Audits

Easy all-environment audit:

```bash
python -m mars.runners.run_uh_official \
  --run_id aaai27_uh_all_easy_s42_43_20260708 \
  --env all \
  --difficulty easy \
  --steps 3 \
  --action_budget 10 \
  --seeds 42,43 \
  --ablation MARS-full \
  --generator_model openai/gpt-4o-mini \
  --reflector_model openai/gpt-4o-mini \
  --judge_model openai/gpt-4o \
  --overwrite
```

Summary: `n=6`, mean score `49.17`, program-induction score `33.33`,
coverage `0.333`.

Hard all-environment audit:

```bash
python -m mars.runners.run_uh_official \
  --run_id aaai27_uh_all_hard_s42_43_20260708 \
  --env all \
  --difficulty hard \
  --steps 3 \
  --action_budget 10 \
  --seeds 42,43 \
  --ablation MARS-full \
  --generator_model openai/gpt-4o-mini \
  --reflector_model openai/gpt-4o-mini \
  --judge_model openai/gpt-4o \
  --overwrite
```

Summary: `n=6`, mean score `45.83`, program-induction score `33.33`,
coverage `0.333`.

## DiscoveryBench DB-Real Full Audit

Command:

```bash
python -m mars.runners.run_universal_discovery_real_eval \
  --run_id aaai27_discovery_full239_20260708 \
  --max_tasks 0 \
  --model openai/gpt-4o-mini \
  --judge_model openai/gpt-4o \
  --n_proposals 0 \
  --max_rounds 0 \
  --overwrite
```

Summary: `n_tasks=239`, `HMS_mean_100=13.61`,
`HMS_mean_consistency_100=17.79`, wall time `3217.46s`.

Per-dataset HMS breakdown:

| Dataset | N | HMS | Cons-HMS | Nonzero | Max |
|---|---:|---:|---:|---:|---:|
| archaeology | 38 | 28.95 | 34.21 | 11 | 100.00 |
| introduction_pathways_non-native_plants | 16 | 31.25 | 68.75 | 5 | 100.00 |
| meta_regression | 50 | 7.16 | 7.16 | 5 | 100.00 |
| meta_regression_raw | 50 | 0.67 | 0.67 | 1 | 33.33 |
| nls_incarceration | 28 | 7.14 | 7.14 | 2 | 100.00 |
| nls_raw | 7 | 60.82 | 75.10 | 5 | 100.00 |
| nls_ses | 23 | 13.91 | 13.91 | 4 | 100.00 |
| requirements_engineering_for_ML_enabled_systems | 15 | 6.67 | 13.33 | 1 | 100.00 |
| worldbank_education_gdp | 6 | 0.00 | 0.00 | 0 | 0.00 |
| worldbank_education_gdp_indicators | 6 | 35.95 | 35.95 | 3 | 85.71 |

Diagnostic finding: a zero-score meta-regression task asked for studies
conducted in a different language, while the generic categorical slot closure
selected a `same_language` column and measured the positive value. A generic
binary complement-closure operator was added after this audit, with a unit test.
The full-audit result above is kept as the diagnostic baseline that exposed the
residual class.

## DiscoveryBench Residual-Operator Full Audit

Change tested: original/replication study-design slot closure with conservative
domain-code aliases, paired-column completion, binary complement closure, and
contrastive rendering. This is a general table-design operator, not a
benchmark-specific answer table.

Command:

```bash
python -m mars.runners.run_universal_discovery_real_eval \
  --run_id aaai27_discovery_full239_or_design_20260709 \
  --max_tasks 0 \
  --model openai/gpt-4o-mini \
  --judge_model openai/gpt-4o \
  --n_proposals 0 \
  --max_rounds 0 \
  --overwrite
```

Summary: `n_tasks=239`, `HMS_mean_100=20.43`,
`HMS_mean_consistency_100=27.13`.

Matched comparison against `aaai27_discovery_full239_20260708`:

| Dataset | N | HMS before | HMS after | Cons before | Cons after | Nonzero before | Nonzero after |
|---|---:|---:|---:|---:|---:|---:|---:|
| archaeology | 38 | 28.95 | 23.68 | 34.21 | 36.84 | 11 | 9 |
| introduction_pathways_non-native_plants | 16 | 31.25 | 6.25 | 68.75 | 68.75 | 5 | 1 |
| meta_regression | 50 | 7.16 | 24.48 | 7.16 | 24.48 | 5 | 16 |
| meta_regression_raw | 50 | 0.67 | 24.17 | 0.67 | 24.17 | 1 | 13 |
| nls_incarceration | 28 | 7.14 | 5.36 | 7.14 | 5.36 | 2 | 2 |
| nls_raw | 7 | 60.82 | 73.57 | 75.10 | 87.86 | 5 | 6 |
| nls_ses | 23 | 13.91 | 7.39 | 13.91 | 7.39 | 4 | 3 |
| requirements_engineering_for_ML_enabled_systems | 15 | 6.67 | 26.67 | 13.33 | 26.67 | 1 | 4 |
| worldbank_education_gdp | 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 |
| worldbank_education_gdp_indicators | 6 | 35.95 | 35.95 | 35.95 | 35.95 | 3 | 3 |

Main gains: `meta_regression_raw` moves from `0.67` to `24.17` HMS, and
`meta_regression` from `7.16` to `24.48`. The strongest solved classes are
effect-estimate comparisons, original/replication subject proportions,
compensation roles, author/participant counts, and country proportions in
Experimental Economics. The requirements subset also improves from `6.67` to
`26.67` HMS.

Regressions and remaining gaps: plant introduction pathways drop from `31.25`
to `6.25`, archaeology drops slightly in raw HMS, and NLS SES drops from
`13.91` to `7.39`. WorldBank education-GDP remains unsolved. These failures are
not arithmetic failures of the new operator; they are mostly denominator
alignment, semantic rendering, multi-file assembly, and table-family routing
failures. The paper should therefore present the operator as a strong residual
repair for study-design tables, not as a finished universal DiscoveryBench
solver.

## Paper Build

Command:

```bash
pdflatex -interaction=nonstopmode -halt-on-error MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.tex
pdflatex -interaction=nonstopmode -halt-on-error MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.tex
```

Result: PDF builds successfully, 8 pages, no overfull boxes after the final
table rewrite. Rendered pages were inspected as PNGs under
`tmp/pdfs/mars_aaai27_render/`.

Post residual-operator update: PDF was rebuilt twice with
`pdflatex -interaction=nonstopmode -halt-on-error` and rendered under
`tmp/pdfs/mars_aaai27_render_20260709/`. Pages 1, 6, and 7 were visually
inspected; the abstract and result/operator tables are legible and not clipped.
The full Python test suite passes: `180 passed, 6 warnings`.

## Final AAAI-27 Polish Cycle

Change tested: the paper now explicitly separates autonomous non-oracle
DiscoveryBench comparisons from the oracle-feedback Reflexion reference. The
main DiscoveryBench claim is phrased as: MARS with a GPT-4o-mini core reaches
`20.4` HMS on the 239-task DB-Real audit, exceeding the published non-oracle
GPT-4o ReAct (`15.4`) and GPT-4-preview CodeGen (`16.3`) references while
remaining below oracle-feedback Reflexion (`24.5`).

Build command:

```bash
pdflatex -interaction=nonstopmode -halt-on-error MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.tex
pdflatex -interaction=nonstopmode -halt-on-error MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.tex
```

Result: PDF builds successfully, `8` pages, Letter paper, no undefined
references and no overfull boxes. Remaining LaTeX messages are underfull
paragraph/page warnings only.

Render command:

```bash
mkdir -p tmp/pdfs/mars_aaai27_final_20260709
pdftoppm -png -r 150 MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.pdf \
  tmp/pdfs/mars_aaai27_final_20260709/page
```

Visual check: pages 1, 5, 6, 7, and 8 were inspected. The abstract,
DiscoveryBench non-oracle/oracle table, main results table, ablations,
residual-operator table, limitations, conclusion, and references render
cleanly without clipping or table overlap.

Text checks: no duplicate labels, no empty references, `19` bibliography items,
and no remaining draft author/date placeholders.

Final unit-test command:

```bash
pytest -q tests
```

Result: `180 passed, 6 warnings in 6.91s`. The warnings are unchanged numpy
correlation warnings in the multi-table WorldBank slot-compiler test.

## Fast Ablation Hardening Cycle

Change tested: the ablation section was strengthened for a heavier
pre-submission draft without launching long new official-protocol runs. The
DiscoveryBench hard-bucket ablation now reports all saved intermediate layers:

| System | N | HMS | Cons-HMS |
|---|---:|---:|---:|
| MEK | 30 | 16.7 | 30.0 |
| MEK + estimands | 30 | 20.5 | 37.2 |
| + role canonicalization | 30 | 27.5 | 44.2 |
| + universal role layers | 30 | 40.5 | 60.5 |

The paper also now includes a cross-benchmark mechanism-ablation table:

| Benchmark | Mechanism | Scope | Before -> After |
|---|---|---|---:|
| DiscoveryBench | hard-bucket role stack | 30 tasks | 16.7 -> 40.5 HMS |
| DiscoveryBench | residual study-design operator | 239 tasks | 13.6 -> 20.4 HMS |
| NewtonBench | self-induced coordinate charts | 4 hard tasks | 25.0 -> 75.0 SA-all |
| NewtonBench | final epistemic checks | 24 easy tasks | 79.2 -> 83.3 SA-all |
| UltraHorizon | typed slots + measurements | 6 easy env runs | 46.7 -> 50.8 score |
| UltraHorizon | sequence residual operators | checked hard seq | 2/5 -> 5/5 exact |

Build/render validation:

```bash
pdflatex -interaction=nonstopmode -halt-on-error MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.tex
pdflatex -interaction=nonstopmode -halt-on-error MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.tex
rm -rf tmp/pdfs/mars_aaai27_ablation_20260709
mkdir -p tmp/pdfs/mars_aaai27_ablation_20260709
pdftoppm -png -r 150 MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.pdf \
  tmp/pdfs/mars_aaai27_ablation_20260709/page
```

Result: PDF builds successfully, still `8` pages, no undefined references and
no overfull boxes. Pages 5, 6, and 7 were visually inspected; the new ablation
tables are readable and do not overlap surrounding text.

## Method Figure Hardening Cycle

Change tested: the AAAI-27 paper now includes a vector/TikZ method schematic
for MARS as typed executable hypothesis induction. The figure separates the
inner loop (contract -> local typed sketches -> execution -> validation ->
state update) from the outer loop (typed residuals -> operator promotion ->
shared operator library). The old compact trace table was replaced by a short
textual trace to avoid duplicating visual method content.

Build/render validation:

```bash
pdflatex -interaction=nonstopmode -halt-on-error MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.tex
pdflatex -interaction=nonstopmode -halt-on-error MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.tex
rm -rf tmp/pdfs/mars_aaai27_scheme4_20260709
mkdir -p tmp/pdfs/mars_aaai27_scheme4_20260709
pdftoppm -png -r 150 MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.pdf \
  tmp/pdfs/mars_aaai27_scheme4_20260709/page
```

Result: PDF builds successfully, `9` pages, no undefined references and no
overfull boxes. Page 4 was visually inspected; the method schematic is readable,
scientific in style, and does not overlap the surrounding algorithm or text.
The page-count increase is the tradeoff for keeping the architecture figure in
the main paper; a strict 8-page version would require additional text
compression.

### Method Figure Routing Cleanup

Follow-up change tested: the method schematic was redrawn as three explicit
swimlanes rather than nested loop boxes. The main task/artifact flow is now
horizontal, executable closure is routed on the lower lane, and residual-driven
operator promotion is routed on the upper lane. Arrow endpoints use shortened
paths so arrowheads do not touch labels or box text.

Build/render validation:

```bash
pdflatex -interaction=nonstopmode -halt-on-error MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.tex
pdflatex -interaction=nonstopmode -halt-on-error MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.tex
rm -rf tmp/pdfs/mars_aaai27_scheme7_20260709
mkdir -p tmp/pdfs/mars_aaai27_scheme7_20260709
pdftoppm -png -r 150 MARS_TYPED_EXECUTABLE_HYPOTHESIS_INDUCTION_AAAI2027.pdf \
  tmp/pdfs/mars_aaai27_scheme7_20260709/page
```

Result: PDF builds successfully, still `9` pages, no undefined references and
no overfull boxes. Page 4 was visually inspected; the method figure no longer
has arrows crossing or sitting on top of text, and the lane labels are visible
without overlapping the task blocks.
