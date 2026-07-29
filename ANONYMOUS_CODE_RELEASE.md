# RG-HLI Anonymous Code Release

This archive contains the implementation and evaluator artifacts for:

> Residual-Guided Hypothesis Language Induction for Scientific Reasoning with
> Small Language Models

The release is organized around the paper's central mechanism rather than the
chronology of the research project.

## Reviewer Map

| Question | Primary location |
|---|---|
| What is the shared hypothesis state? | `mars/induction/universal_hypothesis_kernel.py` |
| How are failures represented? | `mars/skills/residual_kernel.py` and `mars/skills/residual_class_ledger.py` |
| How are candidate operators typed and checked? | `mars/skills/typed_operator_plan.py` |
| How is language growth audited? | `mars/runners/run_newton_language_growth_trajectory.py` |
| How are the three headline runs executed? | `mars/runners/run_discovery_chunked_eval.py`, `run_nb_activeprobe.py`, and `run_uh_official.py` |
| Where are reported rows tied to source artifacts? | `paper_assets/evidence/evidence_index.json` |

## Directory Layout

```text
mars/
  adapters/      benchmark observation and evaluator interfaces
  agents/        proposal and reflection calls
  analysis/      evidence indexing, intervals, and integrity checks
  induction/     shared hypothesis-language and operator-induction kernels
  mdl/           executable hypothesis scoring
  runners/       retained experiment entry points
  skills/        typed contracts, slots, residuals, and operator plans
ols/             compatibility interfaces used by the benchmark adapters
tests/           mechanism, protocol, and evidence-integrity tests
paper_assets/
  evidence/      machine-readable claim-to-artifact registry
lmw/             retained per-task evaluator outputs referenced by the registry
```

The separate paper, supplement, and checklist archives contain the LaTeX
submission files. They are intentionally not duplicated here.

## Environment

The retained environment uses Python 3.12. Install the released dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set model credentials through environment variables only:

```bash
export OPENROUTER_API_KEY=...
```

No credential, author identity, private benchmark dataset, or model cache is
included. Obtain the official DiscoveryBench, NewtonBench, and UltraHorizon
repositories from their cited releases and place them at:

```text
discoverybench_repo/
newtonbench_repo/
ultrahorizon_repo/
```

## Fast Verification

The following command checks the typed-state mechanism, language-growth
protocol, complete-run cardinalities, and source hashes without API calls:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Rebuild the quantitative evidence registry:

```bash
python -m mars.analysis.build_paper_evidence_index
python -m mars.analysis.build_submission_manifest \
  --discovery lmw/universal_discovery_real/aaai27_language_growth_protocol_v1_fixed_l0_test239/summary.json \
  --newton lmw/nb_activeprobe/aaai27_newton_full324_no_promotion_gates_20260717/summary.json \
  --ultrahorizon lmw/uh_official/aaai27_uh_full96_marsfull_strict_agent_paperjudge_20260717/summary.json \
  --output_dir paper_assets/evidence/headline_manifest
```

See `REPRODUCIBILITY.md` for the exact benchmark commands and `RESULTS.md` for
the retained paper rows.
