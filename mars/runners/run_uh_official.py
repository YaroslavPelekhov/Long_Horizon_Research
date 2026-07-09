"""
Run MARS against the official UltraHorizon environments.

This is a bridge runner, not the older Bio-only dev proxy. It instantiates the
official `ultrahorizon_repo/envs/*/env.py` classes, exposes their `[agent tool]`
methods as MARS actions, and uses each environment's own `commit_final_result`
judge path.

Score labels:
  * `official-env-compatible`: same official env object and scoring prompt,
    but the judge model is whatever `--judge_model` routes to.
  * exact paper scoring still requires the paper judge
    `DeepSeek-R1-0528-BF16` via a working provider config.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import statistics as st_stats
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
_UH_REPO = _PROJ / "ultrahorizon_repo"
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))
if str(_UH_REPO) not in sys.path:
    sys.path.insert(0, str(_UH_REPO))

try:
    from dotenv import load_dotenv

    _env_path = _PROJ / "autodiscovery" / ".env.local"
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass

from mars.agents.generator import Generator  # noqa: E402
from mars.agents.memory_selector import MemorySelector  # noqa: E402
from mars.agents.reflector import Reflector  # noqa: E402
from mars.coordinator import Coordinator  # noqa: E402
from mars.skills.program_induction_ledger import induce_trace_rule_report  # noqa: E402
from ols.adapters.base import BudgetExhausted, EnvHandle, ResearchEnvAdapter  # noqa: E402
from ols.core.abandon import FutilityDetector  # noqa: E402
from ols.core.types import ActionSpec, Claim, ExperimentResult  # noqa: E402


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _openai_compatible_judge_config(judge_model: str) -> dict[str, str]:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if key.startswith("sk-or-") and not base_url:
        base_url = "https://openrouter.ai/api/v1"
    return {"model": judge_model, "base_url": base_url, "api_key": key}


def _json_shrink(obj: Any, max_chars: int = 2600) -> Any:
    """Return a prompt-sized representation while preserving JSON structure."""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    if len(s) <= max_chars:
        return obj
    return {"truncated": True, "preview": s[:max_chars]}


def _extract_final_score(final_result: dict[str, Any]) -> float:
    jr = (final_result or {}).get("judge_result", {})
    if isinstance(jr, dict):
        try:
            return float(jr.get("final_score", 0.0))
        except Exception:
            return 0.0
    return 0.0


def _program_induction_score(row: dict[str, Any]) -> float | None:
    diagnostics = row.get("program_induction", {})
    slots = diagnostics.get("slots", []) if isinstance(diagnostics, dict) else []
    if not isinstance(slots, list) or not slots:
        return None
    strong = 0
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        try:
            loss = float(slot.get("loss_mean", 1.0))
            exact = float(slot.get("exact_rate", 0.0))
        except Exception:
            continue
        if exact >= 0.8 or loss <= 0.05:
            strong += 1
    return 100.0 * strong / max(1, len(slots))


def _induction_strong_enough_to_replace(diagnostics: dict[str, Any]) -> bool:
    """Decide whether executable induction should replace the agent report.

    Low-confidence programs are still valuable evidence, but replacing a
    richer textual hypothesis with weak code is a bad final-artifact policy.
    This gate is benchmark-agnostic: it only reads validation residuals.
    """

    slots = diagnostics.get("slots", []) if isinstance(diagnostics, dict) else []
    if not isinstance(slots, list) or not slots:
        return False
    useful = 0
    losses: list[float] = []
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        try:
            loss = float(slot.get("loss_mean", 1.0))
            exact = float(slot.get("exact_rate", 0.0))
        except Exception:
            continue
        losses.append(loss)
        if exact >= 0.5 or loss <= 0.12:
            useful += 1
    if not losses:
        return False
    return useful >= max(2, len(losses) // 2) and (sum(losses) / len(losses)) <= 0.18


@dataclass
class UHTask:
    env_name: str
    seed: int
    steps: int
    free: bool
    difficulty: str
    judge_model: str
    action_budget: int

    def label(self) -> str:
        mode = "free" if self.free else "fixed"
        return f"uh_{self.env_name}_{self.difficulty}_s{self.seed}_steps{self.steps}_{mode}"


class UltraHorizonOfficialAdapter(ResearchEnvAdapter):
    """ResearchEnvAdapter over the official UltraHorizon env objects."""

    def __init__(self, task: UHTask):
        from envs.common import Difficulty  # noqa: PLC0415
        from envs.grid_env.env import MysteryGridEnvironment  # noqa: PLC0415
        from envs.seq_env.env import SequenceExploreEnvironment  # noqa: PLC0415
        from envs.bio_env.env import GeneticsLabEnvironment  # noqa: PLC0415

        env_map = {
            "grid": MysteryGridEnvironment,
            "seq": SequenceExploreEnvironment,
            "bio": GeneticsLabEnvironment,
        }
        if task.env_name not in env_map:
            raise ValueError(f"unknown UltraHorizon env: {task.env_name}")
        self.task = task
        self._budget_total = float(task.action_budget)
        self._budget_spent = 0.0
        self._eid = 0
        self._submitted = ""
        self._induction_diagnostics: dict[str, Any] = {}
        self._last_commit_result: dict[str, Any] = {}
        self._history_raw: list[dict[str, Any]] = []

        env_cls = env_map[task.env_name]
        judge_cfg = _openai_compatible_judge_config(task.judge_model)
        env_cls.load_judge_config = lambda _self: dict(judge_cfg)
        diff = getattr(Difficulty, task.difficulty.upper())

        # The official env constructors print the full prompt/config. Suppress
        # that so runner logs stay readable.
        with contextlib.redirect_stdout(io.StringIO()):
            if task.env_name == "bio":
                self._env = env_cls(
                    seed=task.seed,
                    required_steps=task.steps,
                    difficulty=diff,
                    free=task.free,
                )
            else:
                # grid/seq constructors do not expose seed; seed their modules
                # through Python's random via constructor internals.
                import random
                import numpy as np

                random.seed(task.seed)
                np.random.seed(task.seed)
                self._env = env_cls(
                    difficulty=diff,
                    required_steps=task.steps,
                    free=task.free,
                )

    def handle(self) -> EnvHandle:
        return EnvHandle(
            description=(
                self._env.env_prompt
                + "\n\nMARS EXECUTION NOTE: use the listed tools to run "
                "experiments and maintain hypotheses in claims. "
                "`commit_final_result` is TERMINAL: call it exactly once when "
                "you have a complete answer, or when the remaining action "
                "budget is low."
            ),
            subdomains=[
                (
                    "explore",
                    "Run informative experiments/actions to identify hidden mechanisms. "
                    "Assert compact claims after observations.",
                ),
                (
                    "synthesize",
                    "Consolidate the claims into the final answer and call "
                    "commit_final_result.",
                ),
            ],
            actions=self._actions(),
            budget_total=self._budget_total,
        )

    def _actions(self) -> list[ActionSpec]:
        if self.task.env_name == "grid":
            return [
                ActionSpec(
                    "move",
                    {"direction": "str: up/down/left/right"},
                    1.0,
                    "Move in the grid; official environment tool.",
                ),
                ActionSpec(
                    "get_current_state",
                    {},
                    1.0,
                    "Get current game state and nearby tiles; official environment tool.",
                ),
                ActionSpec(
                    "get_full_map",
                    {},
                    1.0,
                    "Get complete map state with coordinates; official environment tool.",
                ),
                ActionSpec(
                    "reset",
                    {},
                    1.0,
                    "Reset for a new game; official environment tool.",
                ),
                ActionSpec(
                    "commit_final_result",
                    {"content": "str: exact mapping A-E to effect rules"},
                    0.0,
                    "TERMINAL official judge call. Submit final A-E effect mapping.",
                ),
            ]
        if self.task.env_name == "seq":
            return [
                ActionSpec(
                    "input_sequences",
                    {
                        "main_sequence": "str: exactly 5 chars from A-E",
                        "vice_sequence": "str: exactly 5 chars from A-E",
                    },
                    1.0,
                    "Input two sequences and observe the official transformation chain.",
                ),
                ActionSpec(
                    "commit_final_result",
                    {"content": "str: exact rule_1..rule_5 mechanisms"},
                    0.0,
                    "TERMINAL official judge call. Submit final transformation rules.",
                ),
            ]
        return [
            ActionSpec(
                "conduct_cross",
                {
                    "parent1_id": "int",
                    "parent2_id": "int",
                    "num_offspring": "int 1-100",
                },
                1.0,
                "Conduct breeding experiment; official environment tool.",
            ),
            ActionSpec(
                "query_organisms",
                {"start_id": "int", "end_id": "int optional"},
                1.0,
                "Query organisms by ID range; official environment tool.",
            ),
            ActionSpec(
                "remove_organisms",
                {"organism_ids": "list[int]"},
                1.0,
                "Remove organisms to manage capacity; official environment tool.",
            ),
            ActionSpec(
                "get_lab_status",
                {},
                1.0,
                "Get lab status; official environment tool.",
            ),
            ActionSpec(
                "commit_final_result",
                {"content": "str: full formal inheritance-rule report"},
                0.0,
                "TERMINAL official judge call. Submit final genetics report.",
            ),
        ]

    def budget_left(self) -> float:
        if getattr(self._env, "committed", False):
            return 0.0
        return max(0.0, self._budget_total - self._budget_spent)

    def submit_action_names(self) -> set[str]:
        return {"commit_final_result"}

    def execute(self, action: str, args: dict) -> ExperimentResult:
        if self.budget_left() <= 0:
            raise BudgetExhausted("UltraHorizon official action budget exhausted")
        self._eid += 1
        args = dict(args or {})

        if action == "commit_final_result":
            self._submitted = str(args.get("content", "") or "")
            if os.environ.get("MARS_USE_PROGRAM_INDUCTION_LEDGER", "1") not in (
                "0",
                "false",
                "False",
                "no",
            ):
                induced = induce_trace_rule_report(self._history_raw)
                if induced and induced.artifact:
                    self._induction_diagnostics = {
                        "mode": induced.mode,
                        "matched_rules": list(induced.matched_rules),
                        "n_traces": induced.n_traces,
                        **(induced.diagnostics or {}),
                    }
                    if _induction_strong_enough_to_replace(induced.diagnostics or {}):
                        self._submitted = induced.artifact
                        args["content"] = induced.artifact
                    elif self._submitted.strip():
                        self._submitted = (
                            self._submitted.rstrip()
                            + "\n\nExecutable induction evidence "
                            "(validation-gated, not used as sole answer):\n"
                            + induced.artifact
                        )
                        args["content"] = self._submitted
                    else:
                        self._submitted = induced.artifact
                        args["content"] = induced.artifact
        method = getattr(self._env, action, None)
        if method is None:
            result = {"success": False, "message": f"unknown action: {action}"}
        else:
            try:
                result = _run_async(method(**args))
            except Exception as e:
                result = {"success": False, "message": f"{type(e).__name__}: {e}"}

        cost = 0.0 if action == "commit_final_result" else 1.0
        self._budget_spent += cost
        if action == "commit_final_result":
            self._last_commit_result = result if isinstance(result, dict) else {}

        self._history_raw.append(
            {"eid": self._eid, "action": action, "args": args, "result": result}
        )
        return ExperimentResult(
            eid=self._eid,
            action=action,
            args=args,
            cost=cost,
            raw=result,
            summary=_json_shrink(result),
        )

    def verify_claim(self, claim: Claim):
        return None

    def score_episode(self, claim_store_active: list[Claim], final_artifact=None) -> dict:
        final_result = getattr(self._env, "final_result", {}) or {}
        final_score = _extract_final_score(final_result)
        return {
            "primary": final_score,
            "final_score": final_score,
            "submitted": self._submitted,
            "committed": bool(getattr(self._env, "committed", False)),
            "judge_result": final_result.get("judge_result", {}),
            "final_result": final_result,
            "env_tool_history": self._history_raw,
            "program_induction": self._induction_diagnostics,
        }


ABLATIONS = {
    "MARS-full": dict(
        use_reflector=True,
        use_memory_selector=True,
        use_futility_detector=True,
    ),
    "MARS-all-off": dict(
        use_reflector=False,
        use_memory_selector=False,
        use_futility_detector=False,
    ),
    "MARS-no-ref": dict(
        use_reflector=False,
        use_memory_selector=True,
        use_futility_detector=True,
    ),
}


def run_one(task: UHTask, *, ablation_name: str, gen_model: str, ref_model: str) -> dict:
    adapter = UltraHorizonOfficialAdapter(task)
    gen = Generator(model=gen_model, max_tokens=1800)
    ref = Reflector(model=ref_model, max_tokens=350)
    sel = MemorySelector(k=6)
    coord = Coordinator(
        adapter=adapter,
        generator=gen,
        reflector=ref,
        memory_selector=sel,
        futility=FutilityDetector(theta_futile=2.0, min_absolute_spend=1e9),
        max_turns_per_subgoal=max(2, task.action_budget // 2),
        max_total_turns=task.action_budget,
        verbose=False,
        **ABLATIONS[ablation_name],
    )
    t0 = time.time()
    rep = coord.run_episode()
    score_dict = dict(rep.score_dict or {})
    fallback_committed = False
    if not score_dict.get("committed", False):
        try:
            adapter.execute(
                "commit_final_result",
                {
                    "content": (
                        "No explicit final answer was committed before budget "
                        "exhaustion. Synthesize the best validation-gated "
                        "hypothesis from the executed experiments."
                    )
                },
            )
            score_dict = adapter.score_episode([])
            fallback_committed = True
        except Exception:
            score_dict = dict(rep.score_dict or {})
    elapsed = time.time() - t0
    return {
        "task_label": task.label(),
        "env": task.env_name,
        "seed": task.seed,
        "steps": task.steps,
        "free": task.free,
        "difficulty": task.difficulty,
        "judge_model": task.judge_model,
        "agent": ablation_name,
        "generator_model": gen_model,
        "reflector_model": ref_model,
        "final_score": score_dict.get("final_score", 0.0),
        "committed": score_dict.get("committed", False),
        "fallback_committed": fallback_committed,
        "submitted_preview": score_dict.get("submitted", "")[:500],
        "program_induction": score_dict.get("program_induction", {}),
        "judge_result": score_dict.get("judge_result", {}),
        "n_turns": rep.n_turns,
        "n_actions": rep.n_actions,
        "n_generator_calls": rep.n_generator_calls,
        "n_reflector_calls": rep.n_reflector_calls,
        "wall_time_s": elapsed,
        "history": rep.history,
        "env_tool_history": score_dict.get("env_tool_history", []),
    }


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["grid", "seq", "bio", "all"], default="seq")
    parser.add_argument("--run_id", default=os.environ.get("MARS_UH_RUN_ID", "mars_uh_official_smoke"))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("MARS_UH_STEPS", "5")))
    parser.add_argument("--action_budget", type=int, default=int(os.environ.get("MARS_UH_ACTION_BUDGET", "12")))
    parser.add_argument("--seeds", default=os.environ.get("MARS_UH_SEEDS", "42"))
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=os.environ.get("MARS_UH_DIFFICULTY", "hard"))
    parser.add_argument("--free", action="store_true", default=os.environ.get("MARS_UH_FREE", "0") in ("1", "true", "True", "yes"))
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default=os.environ.get("MARS_ABLATION", "MARS-full"))
    parser.add_argument("--generator_model", default=os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--reflector_model", default=os.environ.get("MARS_REFLECTOR_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--judge_model", default=os.environ.get("MARS_UH_OFFICIAL_JUDGE_MODEL", "openai/gpt-4o"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    envs = ["grid", "seq", "bio"] if args.env == "all" else [args.env]
    seeds = [int(x) for x in _parse_csv(args.seeds)]
    tasks = [
        UHTask(
            env_name=e,
            seed=s,
            steps=args.steps,
            free=args.free,
            difficulty=args.difficulty,
            judge_model=args.judge_model,
            action_budget=args.action_budget,
        )
        for e in envs
        for s in seeds
    ]

    out_dir = _PROJ / "lmw" / "uh_official" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run.jsonl"
    summary_path = out_dir / "summary.json"
    if run_log.exists() and not args.overwrite:
        raise SystemExit(f"run log exists; pass --overwrite: {run_log}")
    if args.overwrite and run_log.exists():
        run_log.unlink()

    print("\n=== MARS -> UltraHorizon official-env bridge ===")
    print(f"  run_id={args.run_id}")
    print(f"  envs={envs} seeds={seeds} steps={args.steps} free={args.free}")
    print(f"  action_budget={args.action_budget} difficulty={args.difficulty}")
    print(f"  gen={args.generator_model} ref={args.reflector_model} judge={args.judge_model}")
    print(f"  N={len(tasks)}\n")

    rows = []
    t_start = time.time()
    with run_log.open("w", encoding="utf-8") as f:
        for i, task in enumerate(tasks, 1):
            print(f"  [{i}/{len(tasks)}] {task.label()} ...", flush=True)
            try:
                row = run_one(
                    task,
                    ablation_name=args.ablation,
                    gen_model=args.generator_model,
                    ref_model=args.reflector_model,
                )
            except Exception as e:
                row = {
                    "task_label": task.label(),
                    "env": task.env_name,
                    "seed": task.seed,
                    "error": f"{type(e).__name__}: {e}",
                    "final_score": 0.0,
                    "committed": False,
                }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            f.flush()
            print(
                f"      score={row.get('final_score', 0.0):.1f} "
                f"committed={row.get('committed')} turns={row.get('n_turns', 0)} "
                f"actions={row.get('n_actions', 0)}",
                flush=True,
            )

    scores = [float(r.get("final_score", 0.0)) for r in rows]
    induction_scores_available = [
        float(score)
        for r in rows
        for score in [_program_induction_score(r)]
        if score is not None
    ]
    induction_scores_all = [
        float(score) if score is not None else 0.0
        for r in rows
        for score in [_program_induction_score(r)]
    ]
    summary = {
        "run_id": args.run_id,
        "score_type": "official-env-compatible",
        "exact_paper_judge": args.judge_model == "DeepSeek-R1-0528-BF16",
        "generator_model": args.generator_model,
        "reflector_model": args.reflector_model,
        "judge_model": args.judge_model,
        "envs": envs,
        "seeds": seeds,
        "steps": args.steps,
        "free": args.free,
        "action_budget": args.action_budget,
        "difficulty": args.difficulty,
        "n": len(rows),
        "mean_score": st_stats.fmean(scores) if scores else 0.0,
        "mean_program_induction_score": st_stats.fmean(induction_scores_all) if induction_scores_all else None,
        "mean_program_induction_score_available": (
            st_stats.fmean(induction_scores_available) if induction_scores_available else None
        ),
        "n_program_induction": len(induction_scores_available),
        "program_induction_coverage": (
            len(induction_scores_available) / len(rows) if rows else 0.0
        ),
        "sd_score": st_stats.stdev(scores) if len(scores) > 1 else 0.0,
        "n_committed": sum(1 for r in rows if r.get("committed")),
        "wall_time_s": time.time() - t_start,
        "paths": {"run_log": str(run_log), "summary": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Summary ===")
    print(f"  mean_score={summary['mean_score']:.1f} N={summary['n']} committed={summary['n_committed']}/{summary['n']}")
    print(f"  run_log={run_log}")
    print(f"  summary={summary_path}")


if __name__ == "__main__":
    main()
