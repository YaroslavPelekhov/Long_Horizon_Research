"""Build a source-hashed index for every quantitative artifact cited in the paper."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "paper_assets" / "evidence"

EVIDENCE = (
    (
        "headline.discovery.summary",
        "headline",
        "DiscoveryBench",
        "lmw/universal_discovery_real/"
        "aaai27_language_growth_protocol_v1_fixed_l0_test239/summary.json",
        "Complete 239-task headline summary",
    ),
    (
        "headline.discovery.rows",
        "headline",
        "DiscoveryBench",
        "lmw/universal_discovery_real/"
        "aaai27_language_growth_protocol_v1_fixed_l0_test239/official_eval.jsonl",
        "Native HMS decisions for all 239 tasks",
    ),
    (
        "headline.newton.summary",
        "headline",
        "NewtonBench",
        "lmw/nb_activeprobe/"
        "aaai27_newton_full324_no_promotion_gates_20260717/summary.json",
        "Complete 324-task headline summary",
    ),
    (
        "headline.newton.rows",
        "headline",
        "NewtonBench",
        "lmw/nb_activeprobe/"
        "aaai27_newton_full324_no_promotion_gates_20260717/rows.csv",
        "Symbolic-agreement decisions for all 324 tasks",
    ),
    (
        "headline.ultrahorizon.summary",
        "headline",
        "UltraHorizon",
        "lmw/uh_official/"
        "aaai27_uh_full96_marsfull_strict_agent_paperjudge_20260717/summary.json",
        "Strict 96-episode headline summary",
    ),
    (
        "headline.ultrahorizon.rows",
        "headline",
        "UltraHorizon",
        "lmw/uh_official/"
        "aaai27_uh_full96_marsfull_strict_agent_paperjudge_20260717/run.jsonl",
        "Paper-style judge records for all 96 episodes",
    ),
    (
        "baseline.discovery.direct",
        "matched baseline",
        "DiscoveryBench",
        "lmw/aaai27_mechanism_audit_20260721/direct_vs_rghli_bootstrap.json",
        "Paired Direct comparison",
    ),
    (
        "baseline.discovery.react",
        "matched baseline",
        "DiscoveryBench",
        "lmw/aaai27_mechanism_audit_20260721/react_vs_rghli_bootstrap.json",
        "Paired ReAct comparison",
    ),
    (
        "baseline.discovery.codeact",
        "matched baseline",
        "DiscoveryBench",
        "lmw/aaai27_mechanism_audit_20260721/codeact_vs_rghli_bootstrap.json",
        "Paired CodeAct comparison",
    ),
    (
        "baseline.newton.summary",
        "matched baseline",
        "NewtonBench",
        "paper_assets/evidence/newton_matched_controls/audited_summary.json",
        "Complete matched-control summary",
    ),
    (
        "baseline.newton.aggregate",
        "matched baseline",
        "NewtonBench",
        "paper_assets/evidence/newton_matched_controls/audited_headline.csv",
        "Audited matched-control aggregate rows",
    ),
    (
        "baseline.newton.vanilla_rows",
        "matched baseline",
        "NewtonBench",
        "paper_assets/evidence/newton_matched_controls/vanilla_agent_tasks.csv",
        "Complete 324-task Vanilla/ReAct-like control records",
    ),
    (
        "baseline.newton.codeact_rows",
        "matched baseline",
        "NewtonBench",
        "paper_assets/evidence/newton_matched_controls/code_assisted_agent_tasks.csv",
        "Complete 324-task CodeAct control records",
    ),
    (
        "mechanism.operator_usage",
        "mechanism",
        "DiscoveryBench",
        "lmw/aaai27_mechanism_audit_20260721/discovery_full239_operator_usage.json",
        "Operator reuse counts over the complete run",
    ),
    (
        "mechanism.language_growth.summary",
        "mechanism",
        "NewtonBench",
        "paper_assets/evidence/newton_language_growth/summary.json",
        "Frozen-transfer and multi-step trajectory summary",
    ),
    (
        "mechanism.language_growth.trajectory",
        "mechanism",
        "NewtonBench",
        "paper_assets/evidence/newton_language_growth/trajectory.csv",
        "Accepted and rejected language updates",
    ),
    (
        "mechanism.language_growth.decisions",
        "mechanism",
        "NewtonBench",
        "paper_assets/evidence/newton_language_growth/operator_decisions.csv",
        "Promotion decision audit",
    ),
    (
        "uncertainty.headline",
        "uncertainty",
        "All",
        "lmw/paper_safe_results/bootstrap_headline_ci_20260729.json",
        "Ten-thousand-draw headline bootstrap intervals",
    ),
    (
        "backbone.gpt4o_mini",
        "proposal-core substitution",
        "NewtonBench",
        "lmw/nb_activeprobe/"
        "aaai27_newton_clean_20260724_seed2031_seed2031/summary.json",
        "Complete 324-task diagnostic",
    ),
    (
        "backbone.discovery.gemini_flash_lite.summary",
        "proposal-core substitution",
        "DiscoveryBench",
        "lmw/universal_discovery_real/"
        "aaai27_modelcore_gemini_flash_lite_discovery_full239_matched_20260729/"
        "summary.json",
        "Complete 239-task matched substitution",
    ),
    (
        "backbone.discovery.gemini_flash_lite.rows",
        "proposal-core substitution",
        "DiscoveryBench",
        "lmw/universal_discovery_real/"
        "aaai27_modelcore_gemini_flash_lite_discovery_full239_matched_20260729/"
        "official_eval.jsonl",
        "Native HMS decisions for all 239 tasks",
    ),
    (
        "backbone.discovery.qwen3_30b_a3b.summary",
        "proposal-core substitution",
        "DiscoveryBench",
        "lmw/universal_discovery_real/"
        "aaai27_modelcore_qwen3_30b_discovery_full239_matched_20260729/"
        "summary.json",
        "Complete 239-task matched substitution",
    ),
    (
        "backbone.discovery.qwen3_30b_a3b.rows",
        "proposal-core substitution",
        "DiscoveryBench",
        "lmw/universal_discovery_real/"
        "aaai27_modelcore_qwen3_30b_discovery_full239_matched_20260729/"
        "official_eval.jsonl",
        "Native HMS decisions for all 239 tasks",
    ),
    (
        "backbone.gemini_flash_lite",
        "proposal-core substitution",
        "NewtonBench",
        "lmw/nb_activeprobe/"
        "aaai27_modelcore_gemini_flash_lite_newton_full324_20260721/summary.json",
        "Complete 324-task diagnostic",
    ),
    (
        "backbone.qwen3_30b_a3b",
        "proposal-core substitution",
        "NewtonBench",
        "lmw/nb_activeprobe/"
        "aaai27_modelcore_qwen3_30b_newton_full324_20260721/summary.json",
        "Complete 324-task diagnostic",
    ),
    (
        "backbone.gpt4o",
        "proposal-core substitution",
        "NewtonBench",
        "lmw/nb_activeprobe/"
        "aaai27_modelcore_gpt4o_newton_full324_rghli_20260721/summary.json",
        "Complete 324-task diagnostic",
    ),
    (
        "backbone.deepseek_v4_flash",
        "proposal-core substitution",
        "NewtonBench",
        "lmw/nb_activeprobe/"
        "aaai27_modelcore_deepseek_v4_flash_newton_full324_rghli_20260721/"
        "summary.json",
        "Complete 324-task diagnostic",
    ),
    (
        "backbone.ultrahorizon.gpt4o_mini",
        "proposal-core substitution",
        "UltraHorizon",
        "lmw/uh_official/uh_clean_universal_full96_20260715/summary.json",
        "Complete 96-episode controller-enabled substitution",
    ),
    (
        "backbone.ultrahorizon.gpt4o_mini.grid",
        "proposal-core substitution",
        "UltraHorizon",
        "lmw/uh_official/uh_clean_universal_grid_full32_20260715/summary.json",
        "Complete 32-seed Grid substitution source",
    ),
    (
        "backbone.ultrahorizon.gpt4o_mini.seq",
        "proposal-core substitution",
        "UltraHorizon",
        "lmw/uh_official/uh_clean_universal_seq_full32_20260715/summary.json",
        "Complete 32-seed Seq substitution source",
    ),
    (
        "backbone.ultrahorizon.gpt4o_mini.bio",
        "proposal-core substitution",
        "UltraHorizon",
        "lmw/uh_official/uh_clean_universal_bio_full32_20260715/summary.json",
        "Complete 32-seed Bio substitution source",
    ),
    (
        "backbone.ultrahorizon.gemini.grid",
        "proposal-core substitution",
        "UltraHorizon",
        "lmw/uh_official/"
        "aaai27_modelcore_gemini_uh_grid_full32_20260721/summary.json",
        "Complete 32-seed Grid substitution",
    ),
    (
        "backbone.ultrahorizon.gemini.seq",
        "proposal-core substitution",
        "UltraHorizon",
        "lmw/uh_official/"
        "aaai27_modelcore_gemini_uh_seq_full32_20260721/summary.json",
        "Complete 32-seed Seq substitution",
    ),
    (
        "backbone.ultrahorizon.gemini.bio",
        "proposal-core substitution",
        "UltraHorizon",
        "lmw/uh_official/"
        "aaai27_modelcore_gemini_uh_bio_full32_20260721/summary.json",
        "Complete 32-seed Bio substitution",
    ),
    (
        "backbone.ultrahorizon.qwen.grid",
        "proposal-core substitution",
        "UltraHorizon",
        "lmw/uh_official/"
        "aaai27_modelcore_qwen_uh_grid_full32_20260721/summary.json",
        "Complete 32-seed Grid substitution",
    ),
    (
        "backbone.ultrahorizon.qwen.seq",
        "proposal-core substitution",
        "UltraHorizon",
        "lmw/uh_official/"
        "aaai27_modelcore_qwen_uh_seq_full32_20260721/summary.json",
        "Complete 32-seed Seq substitution",
    ),
    (
        "backbone.ultrahorizon.qwen.bio",
        "proposal-core substitution",
        "UltraHorizon",
        "lmw/uh_official/"
        "aaai27_modelcore_qwen_uh_bio_full32_20260721/summary.json",
        "Complete 32-seed Bio substitution",
    ),
    (
        "audit.model_substitution.csv",
        "audit",
        "All",
        "paper_assets/evidence/model_substitution_audit.csv",
        "Protocol-validated model-substitution table",
    ),
    (
        "audit.model_substitution.json",
        "audit",
        "All",
        "paper_assets/evidence/model_substitution_audit.json",
        "Source hashes and validated model-substitution rows",
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_count(path: Path) -> int | None:
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if path.suffix == ".json":
        value: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            for key in ("n", "n_tasks"):
                if isinstance(value.get(key), int):
                    return int(value[key])
            if isinstance(value.get("results"), dict):
                return len(value["results"])
            if isinstance(value.get("rows"), list):
                return len(value["rows"])
    return None


def main() -> None:
    rows: list[dict[str, Any]] = []
    for claim_id, category, benchmark, relative_path, description in EVIDENCE:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing evidence for {claim_id}: {path}")
        rows.append(
            {
                "claim_id": claim_id,
                "category": category,
                "benchmark": benchmark,
                "artifact": relative_path,
                "records": _record_count(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "description": description,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "evidence_index.csv"
    json_path = OUTPUT_DIR / "evidence_index.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "index_version": "aaai27-paper-evidence-v1",
                "all_sources_hashed": True,
                "entries": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"indexed {len(rows)} artifacts")


if __name__ == "__main__":
    main()
