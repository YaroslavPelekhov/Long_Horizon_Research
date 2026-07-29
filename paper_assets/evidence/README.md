# Audited Paper Evidence

This directory contains compact, table-level evidence for the experiments
added during the final paper audit. Raw model transcripts remain under `lmw/`.

## NewtonBench matched controls

`newton_matched_controls/` contains complete 324-task summaries for the
repository's interactive vanilla agent and CodeAct agent with GPT-4o-mini,
plus per-task results, slice summaries, and an exact-law evaluator sanity
check. The audit resolves duplicate attempts by the earliest valid result and
never selects by score.

Recreate the summaries with:

```bash
python mars/runners/summarize_newton_matched_baselines.py \
  --root lmw/newton_external/aaai27_newton_gpt4omini_matched_full324_20260724
```

`baseline_failure_audit.csv` separates empty, invalid, constant-only,
structured-incorrect, and exact submissions. Representative retained laws are
in `baseline_representative_outputs.csv`. Recreate both with:

```bash
python -m mars.runners.summarize_newton_baseline_failures \
  --evidence-root paper_assets/evidence/newton_matched_controls \
  --output-root paper_assets/evidence/newton_matched_controls
```

## NewtonBench language growth

`newton_language_growth/` records two accepted held-out language promotions
and one rejected redundant proposal. Source, promotion, and transfer modules
are disjoint within each update, use different random seeds, and do not expose
official benchmark scores to induction.

Recreate the trajectory with:

```bash
python -m mars.runners.run_newton_language_growth_trajectory \
  --run_id aaai27_newton_language_growth_trajectory_final_v2_20260724
```

## Headline manifest

`headline_manifest/` is a source-hashed, machine-readable registry of the
three headline benchmark rows. It verifies benchmark cardinality, the retained
NewtonBench per-row symbolic-agreement log, and the strict UltraHorizon
configuration before writing the manifest. Recreate it with:

```bash
python -m mars.analysis.build_submission_manifest \
  --discovery lmw/universal_discovery_real/aaai27_language_growth_protocol_v1_fixed_l0_test239/summary.json \
  --newton lmw/nb_activeprobe/aaai27_newton_full324_no_promotion_gates_20260717/summary.json \
  --ultrahorizon lmw/uh_official/aaai27_uh_full96_marsfull_strict_agent_paperjudge_20260717/summary.json \
  --output_dir paper_assets/evidence/headline_manifest
```

## Complete paper evidence index

`evidence_index.csv` and `evidence_index.json` map every quantitative artifact
cited by the paper and supplement to its retained source, record count, byte
size, and SHA-256 digest. This includes headline runs, matched baselines,
language-growth audits, uncertainty estimates, and proposal-core
substitutions on DiscoveryBench, NewtonBench, and UltraHorizon. The companion
`model_substitution_audit.csv` and `.json` files validate 11 complete rows
against their protocol fields and source hashes. Recreate both registries with:

```bash
python -m mars.analysis.build_model_substitution_audit
python -m mars.analysis.build_paper_evidence_index
```
