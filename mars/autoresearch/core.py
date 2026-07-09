"""Workspace-level autonomous research cycle.

The lower SB-CPI modules build analyzers inside a benchmark episode. This layer
builds the research agenda around those modules: what has been tried, what
failed, which missing analyzer family is most valuable, and how to verify the
next patch.

No benchmark is hard-routed here. Benchmarks appear only as observed artifacts
in the workspace: repo names, run summaries, evaluator logs, and protocol docs.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DOC_GLOBS = [
    "MARS_*.md",
    "BENCHMARK*.md",
    "AUTODISCOVERY*.md",
]


@dataclass
class RunSummary:
    path: str
    benchmark: str
    score_type: str
    metrics: dict[str, Any]


@dataclass
class ResearchCandidate:
    name: str
    priority: float
    hypothesis: str
    evidence: list[str]
    next_experiment: str
    verification_command: str
    expected_result: str
    failure_interpretation: str


@dataclass
class ResearchState:
    created_at: str
    workspace: str
    summaries: list[RunSummary] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    benchmark_findings: dict[str, Any] = field(default_factory=dict)
    candidates: list[ResearchCandidate] = field(default_factory=list)
    selected: ResearchCandidate | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "workspace": self.workspace,
            "summaries": [s.__dict__ for s in self.summaries],
            "environment": self.environment,
            "benchmark_findings": self.benchmark_findings,
            "candidates": [c.__dict__ for c in self.candidates],
            "selected": self.selected.__dict__ if self.selected else None,
        }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_quiet(args: list[str], cwd: Path, timeout: float = 6.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip()
    except Exception as exc:
        return 999, f"{type(exc).__name__}: {exc}"


def _infer_benchmark(path: Path, data: dict[str, Any]) -> str:
    text = " ".join([str(path), str(data.get("run_id", "")), str(data.get("score_type", ""))]).lower()
    if "nb_cpi" in text or "newton" in text:
        return "NewtonBench"
    if "uh_seq" in text or "uh_official" in text or "ultrahorizon" in text:
        return "UltraHorizon"
    if "sab" in text or "scienceagentbench" in text:
        return "ScienceAgentBench"
    if "db_cpi" in text or "db_official" in text or "discovery" in text:
        return "DiscoveryBench"
    return "unknown"


def _extract_metrics(data: dict[str, Any]) -> dict[str, Any]:
    metric_keys = [
        "HMS_mean_100",
        "HMS_mean_0_1",
        "SA_mean",
        "symbolic_match_rate",
        "symbolic_match",
        "numerical_accuracy_mean",
        "mean_exact_program_fit",
        "program_fit_rules",
        "final_score",
        "n_tasks",
        "n_tasks_selected",
        "n_predictions",
        "n_official_eval_success",
        "official_eval_skipped",
        "score_type",
    ]
    return {k: data[k] for k in metric_keys if k in data}


def _best_numeric(values: list[Any]) -> float | None:
    nums: list[float] = []
    for v in values:
        try:
            nums.append(float(v))
        except Exception:
            continue
    return max(nums) if nums else None


class ResearchCycle:
    """Scan the workspace and compile the next self-improvement agenda."""

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace).resolve()

    def collect_summaries(self) -> list[RunSummary]:
        out: list[RunSummary] = []
        for path in sorted((self.workspace / "lmw").glob("**/summary.json")):
            data = _read_json(path)
            if not data:
                continue
            bench = _infer_benchmark(path.relative_to(self.workspace), data)
            out.append(
                RunSummary(
                    path=str(path),
                    benchmark=bench,
                    score_type=str(data.get("score_type", "")),
                    metrics=_extract_metrics(data),
                )
            )
        return out

    def collect_doc_signals(self) -> dict[str, Any]:
        docs: list[dict[str, Any]] = []
        for glob in DOC_GLOBS:
            for path in sorted(self.workspace.glob(glob)):
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                lower = text.lower()
                docs.append(
                    {
                        "path": str(path),
                        "chars": len(text),
                        "mentions": {
                            "self_building": lower.count("self-building") + lower.count("self building"),
                            "cpi": lower.count("cpi"),
                            "blocker": lower.count("blocker"),
                            "sota": lower.count("sota"),
                            "official": lower.count("official"),
                        },
                    }
                )
        return {"docs": docs, "n_docs": len(docs)}

    def collect_environment(self) -> dict[str, Any]:
        env: dict[str, Any] = {}
        env["python"] = shutil.which("python") or ""
        env["docker_cli"] = shutil.which("docker") or ""
        docker_code, docker_out = _run_quiet(["docker", "info"], self.workspace, timeout=8)
        env["docker_daemon_ok"] = docker_code == 0
        env["docker_info_error"] = "" if docker_code == 0 else docker_out[:500]

        df_code, df_out = _run_quiet(["df", "-h", "."], self.workspace, timeout=5)
        env["disk"] = df_out if df_code == 0 else ""

        sab_benchmark = self.workspace / "scienceagentbench_repo" / "benchmark"
        env["sab_benchmark_has_verified_artifacts"] = all(
            (sab_benchmark / name).exists()
            for name in ["datasets", "eval_programs", "gold_programs", "scoring_rubrics"]
        )

        openai_key = os.environ.get("OPENAI_API_KEY", "")
        openai_base = os.environ.get("OPENAI_BASE_URL", "")
        env["openai_key_present"] = bool(openai_key)
        env["openai_key_route"] = "openrouter-like" if openai_key.startswith("sk" + "-or-") else "direct-or-unknown"
        env["openai_base_url_present"] = bool(openai_base)
        return env

    def infer_findings(self, summaries: list[RunSummary]) -> dict[str, Any]:
        by_bench: dict[str, list[RunSummary]] = {}
        for s in summaries:
            by_bench.setdefault(s.benchmark, []).append(s)

        findings: dict[str, Any] = {}
        for bench, rows in by_bench.items():
            hms = _best_numeric([r.metrics.get("HMS_mean_100") for r in rows])
            sa = _best_numeric([r.metrics.get("SA_mean") for r in rows])
            fit = _best_numeric([r.metrics.get("mean_exact_program_fit") for r in rows])
            final = _best_numeric([r.metrics.get("final_score") for r in rows])
            findings[bench] = {
                "n_runs": len(rows),
                "best_HMS_mean_100": hms,
                "best_SA_mean": sa,
                "best_mean_exact_program_fit": fit,
                "best_final_score": final,
                "latest_runs": [r.path for r in rows[-5:]],
            }

        db_eval = self.workspace / "lmw" / "db_cpi" / "stratified20_db_cpi_gpt4omini_gpt4o_v1" / "official_eval.jsonl"
        if db_eval.exists():
            rows = []
            for line in db_eval.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
            zeros = [r.get("task_key") for r in rows if float(r.get("HMS_100", 0.0)) == 0.0]
            positives = {r.get("task_key"): r.get("HMS_100") for r in rows if float(r.get("HMS_100", 0.0)) > 0.0}
            scores = [float(r.get("HMS_100", 0.0)) for r in rows]
            findings.setdefault("DiscoveryBench", {})
            findings["DiscoveryBench"]["stratified20_HMS_mean_100"] = sum(scores) / len(scores) if scores else None
            findings["DiscoveryBench"]["stratified20_task_keys"] = [str(r.get("task_key")) for r in rows if r.get("task_key")]
            findings["DiscoveryBench"]["stratified20_zero_tasks"] = zeros
            findings["DiscoveryBench"]["stratified20_positive_tasks"] = positives

        return findings

    def propose_candidates(self, findings: dict[str, Any], env: dict[str, Any]) -> list[ResearchCandidate]:
        candidates: list[ResearchCandidate] = []

        db = findings.get("DiscoveryBench", {})
        db_hms = float(db.get("stratified20_HMS_mean_100") or db.get("best_HMS_mean_100") or 0.0)
        db_zeros = db.get("stratified20_zero_tasks") or []
        db_keys = " ".join(db.get("stratified20_task_keys") or [])
        db_key_args = db_keys or "<task keys selected by prior autonomous scan>"
        if db_hms < 33.7 or db_zeros:
            candidates.append(
                ResearchCandidate(
                    name="discoverybench_operator_synthesis",
                    priority=0.96,
                    hypothesis=(
                        "DiscoveryBench failures are dominated by missing generated measurement "
                        "operators, not by final-answer prompting. Adding table lookup, coefficient "
                        "extraction, CI extraction, and categorical group-contrast analyzers should "
                        "raise stratified HMS."
                    ),
                    evidence=[
                        f"Observed DB mixed-slice HMS mean: {db_hms:.2f}",
                        f"Zero-score tasks in stratified20: {len(db_zeros)}",
                        "Archaeology moved from 0 to 100 after adding typed operators and renderer.",
                    ],
                    next_experiment=(
                        "Extend DB-CPI with generic table/row retrieval, coefficient/CI parser, "
                        "and group-contrast renderer; rerun the same stratified20 split."
                    ),
                    verification_command=(
                        "python -m mars.runners.run_db_cpi_official_eval "
                        "--run_id stratified20_db_cpi_autoresearch_v2 --max_tasks 0 "
                        f"--task_keys {db_key_args} --generator_model openai/gpt-4o-mini "
                        "--judge_model openai/gpt-4o --overwrite"
                    ),
                    expected_result="HMS_mean_100 should exceed 25.16 and reduce zero-score tasks.",
                    failure_interpretation=(
                        "If HMS does not improve, the missing piece is likely not evidence extraction "
                        "but benchmark-compatible sub-hypothesis decomposition or exact judge routing."
                    ),
                )
            )

        nb = findings.get("NewtonBench", {})
        nb_sa = float(nb.get("best_SA_mean") or 0.0)
        candidates.append(
            ResearchCandidate(
                name="newtonbench_complex_energy_analyzer",
                priority=0.88 if nb_sa >= 1.0 else 0.74,
                hypothesis=(
                    "NewtonBench complex Coulomb failures require a new observation-to-force "
                    "measurement hypothesis for kinetic-energy traces, not a benchmark-specific law."
                ),
                evidence=[
                    f"Best observed NB subset SA: {nb_sa:.2f}",
                    "m1 Coulomb vanilla solved after generic grammar expansion.",
                    "m1 easy/v0 all-systems still fails on complex system.",
                ],
                next_experiment=(
                    "Synthesize and test an energy-difference analyzer that recovers force or "
                    "potential-like targets from complex-system observations."
                ),
                verification_command=(
                    "python -m mars.runners.run_nb_cpi --run_id m1_energy_analyzer_v1 "
                    "--modules m1_coulomb_force --difficulties easy --law_versions v0 "
                    "--systems vanilla_equation,simple_system,complex_system --judge_model gpt41 --overwrite"
                ),
                expected_result="m1 easy/v0 all-systems should move from 2/3 to 3/3 symbolic match.",
                failure_interpretation=(
                    "If complex remains 0/1, inspect the raw observation schema and generate a "
                    "different measurement operator rather than changing the symbolic grammar."
                ),
            )
        )

        sab_blockers = []
        if not env.get("docker_daemon_ok"):
            sab_blockers.append("Docker daemon is not running")
        if not env.get("sab_benchmark_has_verified_artifacts"):
            sab_blockers.append("verified benchmark artifacts are not unpacked")
        if env.get("openai_key_route") == "openrouter-like":
            sab_blockers.append("official visual judge needs direct OpenAI/Azure credentials")
        candidates.append(
            ResearchCandidate(
                name="scienceagentbench_official_unblock_then_contract_analyzers",
                priority=0.82 if sab_blockers else 0.9,
                hypothesis=(
                    "SAB progress has two layers: first unblock official SR/VER/CBS; then add "
                    "dataset-contract, output-format, and error-repair analyzers inside generated programs."
                ),
                evidence=sab_blockers or ["Official export exists and scoring can now be attempted."],
                next_experiment=(
                    "Start Docker, unpack benchmark_verified.zip, provide direct OpenAI/Azure for visual "
                    "judge, run one official instance; then mine failures for code-contract analyzers."
                ),
                verification_command=(
                    "cd scienceagentbench_repo && python -m evaluation.harness.run_evaluation "
                    "--benchmark_path benchmark --pred_program_path ../lmw/sab_official/"
                    "sab_export_verified4_gpt4omini_v1/pred_programs --log_fname ../lmw/sab_official/"
                    "sab_export_verified4_gpt4omini_v1/official_eval_attempt.jsonl --run_id "
                    "sab_export_verified4_gpt4omini_v1 --split verified --instance_ids 1 --max_workers 1"
                ),
                expected_result="Official eval should produce SR/VER/CBS row for instance 1.",
                failure_interpretation="Any crash before a metric row is infrastructure; any failed row is analyzer feedback.",
            )
        )

        uh = findings.get("UltraHorizon", {})
        uh_fit = float(uh.get("best_mean_exact_program_fit") or 0.0)
        candidates.append(
            ResearchCandidate(
                name="ultrahorizon_generalize_seq_to_grid_bio",
                priority=0.76,
                hypothesis=(
                    "UltraHorizon Seq is solved by program induction; Grid and Bio need the same "
                    "self-built transition-program idea over spatial and inheritance traces."
                ),
                evidence=[
                    f"Best observed UH Seq program fit: {uh_fit:.2f}",
                    "DeepSeek R1 0528 judge gave 100/100 on easy and hard Seq seed 42.",
                    "No comparable CPI layer exists yet for Grid/Bio.",
                ],
                next_experiment=(
                    "Build generic transition trace extractor for grid state deltas and genetics "
                    "trait crosses; verify with dry program-fit before judge commit."
                ),
                verification_command=(
                    "python -m mars.runners.run_uh_official --run_id uh_grid_probe_autoresearch_v1 "
                    "--env grid --steps 5 --free --action_budget 12 --seeds 42 "
                    "--judge_model deepseek/deepseek-r1-0528 --overwrite"
                ),
                expected_result="Probe logs should expose enough transition traces for a first grid CPI grammar.",
                failure_interpretation="If traces are under-informative, improve query policy by disagreement rather than renderer.",
            )
        )

        candidates.sort(key=lambda c: c.priority, reverse=True)
        return candidates

    def run(self) -> ResearchState:
        summaries = self.collect_summaries()
        env = self.collect_environment()
        env["doc_signals"] = self.collect_doc_signals()
        findings = self.infer_findings(summaries)
        candidates = self.propose_candidates(findings, env)
        selected = candidates[0] if candidates else None
        return ResearchState(
            created_at=datetime.now(timezone.utc).isoformat(),
            workspace=str(self.workspace),
            summaries=summaries,
            environment=env,
            benchmark_findings=findings,
            candidates=candidates,
            selected=selected,
        )


def render_markdown(state: ResearchState) -> str:
    selected = state.selected
    lines = [
        "# MARS Autonomous Research Cycle",
        "",
        f"Created: `{state.created_at}`",
        f"Workspace: `{state.workspace}`",
        "",
        "## Selected Next Research Hypothesis",
        "",
    ]
    if selected is None:
        lines.append("No candidate selected.")
        return "\n".join(lines) + "\n"

    lines += [
        f"**{selected.name}**",
        "",
        f"Priority: `{selected.priority:.2f}`",
        "",
        selected.hypothesis,
        "",
        "### Evidence",
        "",
    ]
    lines += [f"- {x}" for x in selected.evidence]
    lines += [
        "",
        "### Next Experiment",
        "",
        selected.next_experiment,
        "",
        "```bash",
        selected.verification_command,
        "```",
        "",
        f"Expected result: {selected.expected_result}",
        "",
        f"If it fails: {selected.failure_interpretation}",
        "",
        "## Ranked Agenda",
        "",
    ]
    for i, c in enumerate(state.candidates, 1):
        lines += [
            f"{i}. **{c.name}** (`priority={c.priority:.2f}`)",
            f"   - {c.hypothesis}",
            f"   - Verify: `{c.verification_command}`",
        ]
    lines += [
        "",
        "## Benchmark Findings",
        "",
        "```json",
        json.dumps(state.benchmark_findings, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(state.environment, indent=2, ensure_ascii=False),
        "```",
    ]
    return "\n".join(lines) + "\n"
