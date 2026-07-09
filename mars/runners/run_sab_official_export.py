"""
Generate MARS predictions for official ScienceAgentBench evaluation.

This runner intentionally does NOT compute the proxy LLM-judge score used by
`run_sab.py`. It only produces:

  1. predicted Python files named `pred_<gold_program_name>` in a directory
     accepted by the official SAB docker harness;
  2. a JSONL run log with one line per official verified task, including a
     `cost` field required by `calculate_metrics.py`.

Official evaluation is a separate step:

  cd scienceagentbench_repo
  python -m evaluation.harness.run_evaluation \
    --benchmark_path benchmark \
    --pred_program_path ../lmw/sab_official/<run_id>/pred_programs \
    --log_fname ../lmw/sab_official/<run_id>/eval.jsonl \
    --run_id <run_id> \
    --split verified \
    --instance_ids <id...>

For a publishable SAB result, run three independent attempts and aggregate with
`scienceagentbench_repo/calculate_metrics.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st_stats
import sys
import time
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from dotenv import load_dotenv

    _env_path = _PROJ / "autodiscovery" / ".env.local"
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass

from mars.adapters.scienceagentbench_adapter import (  # noqa: E402
    SABTask,
    ScienceAgentBenchAdapter,
    load_sab_tasks,
)
from mars.agents.generator import Generator  # noqa: E402
from mars.agents.memory_selector import MemorySelector  # noqa: E402
from mars.agents.reflector import Reflector  # noqa: E402
from mars.coordinator import Coordinator  # noqa: E402
from mars.skills.code_repair_compiler import compile_code_repairs  # noqa: E402


class OfficialExportSABAdapter(ScienceAgentBenchAdapter):
    """SAB adapter variant that skips proxy judging during export."""

    def _judge(self, code: str) -> tuple[float, str]:
        return 0.0, "official export: proxy judge skipped"

    @property
    def submitted_program(self) -> str:
        return self._submitted_program or ""

    @property
    def submissions(self) -> list[dict]:
        return list(self._submissions)


ABLATIONS = {
    "MARS-full": dict(
        use_reflector=True,
        use_memory_selector=True,
        use_futility_detector=True,
    ),
    "MARS-no-ref": dict(
        use_reflector=False,
        use_memory_selector=True,
        use_futility_detector=True,
    ),
    "MARS-no-mem": dict(
        use_reflector=True,
        use_memory_selector=False,
        use_futility_detector=True,
    ),
    "MARS-no-fut": dict(
        use_reflector=True,
        use_memory_selector=True,
        use_futility_detector=False,
    ),
    "MARS-all-off": dict(
        use_reflector=False,
        use_memory_selector=False,
        use_futility_detector=False,
    ),
}


def _strip_code_fences(code: str) -> str:
    code = (code or "").strip()
    if not code:
        return "ERROR\n"
    match = re.search(r"```(?:python)?\s*(.*?)```", code, re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1).strip()
    return code.rstrip() + "\n"


def _best_submitted_code(adapter: OfficialExportSABAdapter) -> str:
    """Choose the last non-empty program submission.

    Coordinator histories can contain a good program followed by an empty
    submit action.  The official export should preserve the best executable
    artifact rather than blindly taking the final action.
    """

    candidates: list[str] = []
    for submission in adapter.submissions:
        code = ""
        if isinstance(submission, dict):
            args = submission.get("args", {})
            if isinstance(args, dict):
                code = str(args.get("code", "") or "")
        stripped = _strip_code_fences(code)
        if stripped.strip() and stripped.strip() != "ERROR":
            candidates.append(stripped)
    if candidates:
        return candidates[-1]
    return _strip_code_fences(adapter.submitted_program)


def _safe_cost_placeholder(rep) -> float:
    """Return a deterministic cost placeholder until token accounting exists.

    Official SAB primary metrics (SR/VER/CBS) do not depend on this value, but
    `calculate_metrics.py` expects it. We keep a transparent placeholder rather
    than fabricating token prices.
    """
    return 0.0


def _load_tasks(
    *,
    max_tasks: int | None,
    domains: list[str] | None,
    instance_ids: list[str] | None,
) -> list[SABTask]:
    tasks = load_sab_tasks(max_tasks=None, domains=domains)
    if instance_ids:
        wanted = {str(x) for x in instance_ids}
        tasks = [t for t in tasks if str(t.instance_id) in wanted]
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return tasks


def _run_one(
    task: SABTask,
    *,
    ablation_name: str,
    gen_model: str,
    ref_model: str,
    budget: float,
    max_turns_buffer: int,
) -> tuple[dict, str]:
    adapter = OfficialExportSABAdapter(
        task=task,
        budget=budget,
        judge_model="official-export-no-proxy",
    )
    gen = Generator(model=gen_model, max_tokens=2600)
    ref = Reflector(model=ref_model, max_tokens=350)
    sel = MemorySelector(k=4)
    coord = Coordinator(
        adapter=adapter,
        generator=gen,
        reflector=ref,
        memory_selector=sel,
        max_turns_per_subgoal=int(budget) + 1,
        max_total_turns=int(budget) + max_turns_buffer,
        verbose=False,
        **ABLATIONS[ablation_name],
    )

    t0 = time.time()
    rep = coord.run_episode()
    elapsed = time.time() - t0
    code = _best_submitted_code(adapter)
    if code.strip() == "ERROR":
        code = "ERROR\n"
    else:
        repaired = compile_code_repairs(
            code,
            benchmark_path=_PROJ / "scienceagentbench_repo" / "benchmark",
            output_path=task.output_fname,
        )
        code = repaired.source

    log = {
        "instance_id": str(task.instance_id),
        "domain": task.domain,
        "gold_program_name": task.gold_program_name,
        "eval_script_name": task.eval_script_name,
        "output_fname": task.output_fname,
        "model_name_or_path": gen_model,
        "agent": ablation_name,
        "cost": _safe_cost_placeholder(rep),
        "cost_note": "0.0 placeholder; token accounting not yet implemented",
        "n_generator_calls": rep.n_generator_calls,
        "n_reflector_calls": rep.n_reflector_calls,
        "n_turns": rep.n_turns,
        "n_actions": rep.n_actions,
        "n_submissions": len(adapter.submissions),
        "wall_time_s": elapsed,
        "history": rep.history,
        "program_chars": len(code),
    }
    if code.strip() != "ERROR":
        log["compiled_repairs"] = list(repaired.repairs)
        log["compiled_warnings"] = list(repaired.warnings)
    return log, code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=os.environ.get("MARS_SAB_RUN_ID", "mars_sab_official_run1"))
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--pred_program_path", default="")
    parser.add_argument("--run_log", default="")
    parser.add_argument("--max_tasks", type=int, default=int(os.environ.get("MARS_SAB_N", "0")))
    parser.add_argument("--instance_ids", nargs="*", default=None)
    parser.add_argument("--domains", default=os.environ.get("MARS_SAB_DOMAINS", ""))
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default=os.environ.get("MARS_ABLATION", "MARS-full"))
    parser.add_argument("--generator_model", default=os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--reflector_model", default=os.environ.get("MARS_REFLECTOR_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--budget", type=float, default=float(os.environ.get("MARS_SAB_BUDGET", "4")))
    parser.add_argument("--max_turns_buffer", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    max_tasks = args.max_tasks if args.max_tasks > 0 else None
    domains = [d.strip() for d in args.domains.split(",") if d.strip()] or None

    base_out = Path(args.out_dir) if args.out_dir else _PROJ / "lmw" / "sab_official" / args.run_id
    pred_dir = Path(args.pred_program_path) if args.pred_program_path else base_out / "pred_programs"
    run_log = Path(args.run_log) if args.run_log else base_out / "run.jsonl"
    summary_path = base_out / "summary.json"

    base_out.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    run_log.parent.mkdir(parents=True, exist_ok=True)

    if run_log.exists() and not args.overwrite:
        raise SystemExit(f"run log exists; pass --overwrite: {run_log}")
    if args.overwrite and run_log.exists():
        run_log.unlink()

    tasks = _load_tasks(
        max_tasks=max_tasks,
        domains=domains,
        instance_ids=args.instance_ids,
    )
    if not tasks:
        raise SystemExit("no SAB tasks selected")

    print("\n=== MARS -> ScienceAgentBench official prediction export ===")
    print(f"  run_id={args.run_id}")
    print(f"  pred_program_path={pred_dir}")
    print(f"  run_log={run_log}")
    print(f"  gen={args.generator_model}  ref={args.reflector_model}")
    print(f"  ablation={args.ablation}  budget={args.budget}")
    print(f"  N={len(tasks)} tasks\n")

    rows: list[dict] = []
    t_start = time.time()
    with run_log.open("w", encoding="utf-8") as log_f:
        for i, task in enumerate(tasks, 1):
            out_file = pred_dir / f"pred_{task.gold_program_name}"
            try:
                row, code = _run_one(
                    task,
                    ablation_name=args.ablation,
                    gen_model=args.generator_model,
                    ref_model=args.reflector_model,
                    budget=args.budget,
                    max_turns_buffer=args.max_turns_buffer,
                )
                out_file.write_text(code, encoding="utf-8")
                status = "OK" if code.strip() != "ERROR" else "ERROR"
            except Exception as exc:
                row = {
                    "instance_id": str(task.instance_id),
                    "domain": task.domain,
                    "gold_program_name": task.gold_program_name,
                    "model_name_or_path": args.generator_model,
                    "agent": args.ablation,
                    "cost": 0.0,
                    "error": str(exc)[:500],
                    "history": [],
                }
                out_file.write_text("ERROR\n", encoding="utf-8")
                status = "EXC"

            rows.append(row)
            log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_f.flush()
            print(
                f"  [{i}/{len(tasks)}] id={task.instance_id:<4} "
                f"{task.domain[:28]:<28} {status:<5} "
                f"chars={row.get('program_chars', 0):<5} "
                f"g={row.get('n_generator_calls', 0):<2} "
                f"r={row.get('n_reflector_calls', 0):<2} "
                f"({row.get('wall_time_s', 0):.0f}s)",
                flush=True,
            )

    program_lengths = [r.get("program_chars", 0) for r in rows]
    summary = {
        "run_id": args.run_id,
        "score_type": "official-prediction-export",
        "generator_model": args.generator_model,
        "reflector_model": args.reflector_model,
        "ablation": args.ablation,
        "budget": args.budget,
        "n_tasks": len(tasks),
        "pred_program_path": str(pred_dir),
        "run_log": str(run_log),
        "mean_program_chars": st_stats.fmean(program_lengths) if program_lengths else 0.0,
        "wall_time_s": time.time() - t_start,
        "official_eval_next_step": (
            "Run scienceagentbench_repo/evaluation/harness/run_evaluation.py "
            "with --split verified and this pred_program_path."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[written predictions] {pred_dir}")
    print(f"[written run log]     {run_log}")
    print(f"[written summary]     {summary_path}")


if __name__ == "__main__":
    main()
