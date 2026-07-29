# RG-HLI: Residual-Guided Hypothesis Language Induction

This repository contains the current research prototype and paper artifacts for
Residual-Guided Hypothesis Language Induction (RG-HLI), a universal
hypothesis-induction system for scientific reasoning with small language
models.

RG-HLI does not treat a scientific answer as a single free-form completion. It
turns the task into typed executable hypothesis artifacts: evidence contracts,
slots, measurements, validators, residuals, and promoted operators. The language
model proposes local typed objects, while execution closes uncertain slots and
stores failures as structured residuals.

## Core Method

The same kernel is used across tabular discovery, symbolic law induction, and
long-horizon rule discovery:

1. Compile the task interface into a typed evidence contract.
2. Propose a hypothesis artifact with explicit holes.
3. Close holes with executable probes over the available data or environment.
4. Validate the artifact with contract and metamorphic checks.
5. Convert failures into typed residuals.
6. Promote only operators that improve held-out evidence fit relative to their
   complexity cost.

The main novelty is the residual-guided update of the hypothesis language:
failures are not kept as natural-language feedback; they become typed objects
that reveal missing expressivity in the current language and drive operator
induction.

## Main Results

| Benchmark | N | Metric | RG-HLI result | Main artifact |
|---|---:|---|---:|---|
| DiscoveryBench | 239 | HMS / Cons-HMS | 28.70 / 34.56 | `lmw/universal_discovery_real/aaai27_language_growth_protocol_v1_fixed_l0_test239/summary.json` |
| NewtonBench | 324 | SA-all / SA-answered | 49.38% / 66.67% | `lmw/nb_activeprobe/aaai27_newton_full324_no_promotion_gates_20260717/summary.json` |
| UltraHorizon | 96 | strict paper-style score | 51.04 | `lmw/uh_official/aaai27_uh_full96_marsfull_strict_agent_paperjudge_20260717/summary.json` |

These are the complete single-pass artifacts used by the submission. The
headline language is frozen before evaluation; development-time promotion is
tested separately by the frozen-transfer and multi-step mechanism audits.

## Submission Experiments

Prepare the causal language-growth protocol without making API calls:

```bash
python3 -m mars.runners.run_language_growth_experiment \
  --run_id aaai27_language_growth_protocol_v1 --overwrite_protocol
```

The generated protocol fixes GPT-4o-mini, the official evaluator, the per-task
budget, and the full Discovery stack. It compares fixed `L0`, ungated
promotion, and gated language growth. Operator induction uses only the released
train split; the library is frozen before the complete 239-task test run.

Prepare four clean full-grid NewtonBench repetitions:

```bash
python3 -m mars.runners.run_newton_clean_multirun \
  --run_id aaai27_newton_clean_protocol_v1
```

Both commands prepare manifests by default. Add `--execute` only when the model
and judge credentials are configured. Newton aggregation rejects partial grids,
judge mismatches, and selective-rejudge artifacts.

## Important Files

- `residual_guided_hli_submission.tex` - current AAAI-style paper source.
- `residual_guided_hli_supplement.tex` - supplementary material.
- `residual_guided_hli_refs.bib` - bibliography for the paper.
- `residual_guided_hli_submission.pdf` - compiled submission draft.
- `RESULTS.md` - retained complete benchmark rows.
- `REPRODUCIBILITY.md` - commands and artifact paths for the three canonical
  full runs.
- `ANONYMOUS_CODE_RELEASE.md` - reviewer map for the anonymous code archive.
- `paper_assets/figures/` - generated figures used by the draft.
- `paper_assets/evidence/HEADLINE_CLAIM_AUDIT.md` - internal mapping from every
  headline claim to its retained full-run artifact, evaluator, interval, and
  controlled language-growth evidence.
- `paper_assets/evidence/evidence_index.csv` - source-hashed registry for every
  quantitative artifact cited in the paper and supplement.
- `paper_assets/evidence/model_substitution_audit.json` - protocol validation
  for the complete multi-backbone experiments.
- `paper_assets/make_paper_figures.py` - figure-generation script.
- `mars/` - implementation modules and benchmark runners.

## Reproducing Checks

Run unit tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

Rebuild paper figures:

```bash
python3 paper_assets/make_paper_figures.py
```

Compile the paper:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error residual_guided_hli_submission.tex
```

Build clean, compile-verified Overleaf, checklist, and anonymous-code archives:

```bash
python -m mars.analysis.build_submission_packages \
  --output_root submission_release
```

## Repository Hygiene

Raw benchmark repositories, private data directories, virtual environments, and
API keys are intentionally ignored. Use environment variables for credentials
and do not commit `.env` files.
