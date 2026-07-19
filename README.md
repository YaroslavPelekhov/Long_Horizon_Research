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

## Main Paper-Safe Results

| Benchmark | N | Metric | RG-HLI result | Main artifact |
|---|---:|---|---:|---|
| DiscoveryBench | 239 | HMS / Cons-HMS | 29.94 / 34.96 | `lmw/universal_discovery_real/research_cycle_discovery_full239_scopegate_intrabundle_20260712/summary.json` |
| NewtonBench | 324 | audited SA-all / audited SA-answered | 49.7% / 67.1% | `lmw/nb_activeprobe/nb_full324_compression_tournament_v9_20260713/audited_summary.json` |
| UltraHorizon | 96 | paper-style score | 75.36 | `lmw/uh_official/uh_clean_universal_full96_20260715/summary.json` |

These are the defensible full-run rows used for the current paper draft. Late
diagnostic ceiling runs are kept for engineering analysis but are not used as
main claims.

## Important Files

- `residual_guided_hli_aaai2027.tex` - current AAAI-style paper draft.
- `residual_guided_hli_refs.bib` - bibliography for the paper.
- `residual_guided_hli_aaai2027.pdf` - compiled draft.
- `PAPER_SAFE_RESULTS.md` - canonical result rows and exclusions.
- `REPRODUCIBILITY.md` - commands and artifact paths for the three canonical
  full runs.
- `paper_assets/figures/` - generated figures used by the draft.
- `paper_assets/make_paper_figures.py` - figure-generation script.
- `mars/` - implementation modules and benchmark runners.
- `lmw/paper_safe_results/summary.json` - machine-readable paper-safe result
  registry.

## Reproducing Checks

Run unit tests:

```bash
pytest -q tests
```

Rebuild paper figures:

```bash
python3 paper_assets/make_paper_figures.py
```

Compile the paper:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error residual_guided_hli_aaai2027.tex
```

## Repository Hygiene

Raw benchmark repositories, private data directories, virtual environments, and
API keys are intentionally ignored. Use environment variables for credentials
and do not commit `.env` files.
