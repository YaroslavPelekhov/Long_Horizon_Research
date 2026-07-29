# RG-HLI Anonymous Code Release

This archive contains the implementation, evaluation runners, tests, paper
sources, and compact evidence index for:

> Residual-Guided Hypothesis Language Induction for Scientific Reasoning with
> Small Language Models

## Environment

The retained local environment uses Python 3.12.7. Install the pinned analysis
and API dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-paper.txt
```

Set model credentials only through environment variables:

```bash
export OPENROUTER_API_KEY=...
```

No credential, private benchmark dataset, author identity, or unindexed model
cache is included in the archive. The release does include the 39 retained
table-facing evaluator artifacts named by
`paper_assets/evidence/evidence_index.json`; every included artifact is
content-hashed and can be checked independently.

For double-blind release, the packager removes only machine-local storage
prefixes from path-valued fields (for example, `run_log` and `trial_path`) and
then rebuilds the model audit and SHA-256 evidence index inside the archive.
Task keys, prompts, outputs, evaluator decisions, metrics, and protocol fields
are not changed.

## Contents

- `mars/`: RG-HLI kernel, benchmark interfaces, runners, and analysis tools.
- `tests/`: protocol, resume, evidence-index, and implementation tests.
- `paper_assets/evidence/`: compact summaries, complete retained evaluator
  records, and a SHA-256 index linking each quantitative claim to its source.
- `paper_assets/figures/`: the exact figures compiled into the paper and
  supplement.
- `residual_guided_hli_submission.tex`: seven-page main paper plus references.
- `residual_guided_hli_supplement.tex`: experimental details and audits.
- `REPRODUCIBILITY.md`: canonical commands and complete-run artifact paths.
- `PAPER_SAFE_RESULTS.md`: retained headline rows and excluded diagnostics.

The benchmark repositories are not redistributed. Obtain DiscoveryBench,
NewtonBench, and UltraHorizon from their cited official releases and place
them at the paths documented in `REPRODUCIBILITY.md`.

## Verification

Rebuild and validate the quantitative evidence registry:

```bash
python -m mars.analysis.build_paper_evidence_index
python -m mars.analysis.build_model_substitution_audit
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_paper_evidence_index.py \
  tests/test_model_substitution_audit.py \
  tests/test_main_experiment_protocols.py -q
```

Compile the main paper and supplement:

```bash
LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error \
  residual_guided_hli_submission.tex
LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error \
  residual_guided_hli_supplement.tex
LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error \
  reproducibility_checklist_main.tex
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_submission_artifacts.py \
  tests/test_submission_metadata.py -q
```

See `REPRODUCIBILITY.md` for the exact commands corresponding to each
headline benchmark row.
