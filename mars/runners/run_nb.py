"""
Run MARS on NewtonBench (Tier-1 headline target).

Compares MARS-full / MARS-all-off / individual-ablation conditions on a
configurable subset of NewtonBench tasks. Scoring via the authors' own
`module.evaluate_law()` — directly comparable to the paper's SA metric.

Published baselines we target (avg SA % across 324 tasks):
  GPT-5: 75.9   Gemini-2.5-pro: 65.4   o4-mini: 47.8   DeepSeek-R1: 43.4

Hard-tier (where headroom is largest):
  GPT-5: 87.5   Gemini-2.5-pro: 69.4   o4-mini: 52.8   DeepSeek-R1: 36.8

ENV:
  MARS_NB_MODULES        comma-sep, default = first 4 modules (cost control)
  MARS_NB_DIFFICULTIES   comma-sep, default = "hard" (most headroom)
  MARS_NB_LAW_VERSIONS   comma-sep, default = "v0"
  MARS_NB_SYSTEMS        comma-sep, default = "vanilla_equation"
  MARS_NB_NOISE          comma-sep floats, default = "0.0"
  MARS_NB_TRIALS         per (mod, diff, lv) — default 1
  MARS_NB_BUDGET         rounds per episode (default 10)
  MARS_GENERATOR_MODEL   default openai/gpt-4o-mini
  MARS_REFLECTOR_MODEL   default openai/gpt-4o-mini
  MARS_NB_JUDGE_MODEL    default gpt41 (their key for gpt-4.1; in our env we
                         resolve via the same OpenRouter key)
  MARS_ABLATIONS         default "MARS-full,MARS-all-off"
"""

from __future__ import annotations

import json
import os
import signal
import statistics as st_stats
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

# CRITICAL: load API keys from .env.local with override=True BEFORE any LLM
# client is constructed. The shell may carry a stale/expired OPENAI_API_KEY;
# without this the Generator silently uses the bad key → empty completions →
# 0 actions → 0 score. (UH runner gets this via its adapter's import-time load.)
try:
    from dotenv import load_dotenv
    _env_path = _PROJ / "autodiscovery" / ".env.local"
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass

from mars.adapters.newtonbench_adapter import (
    NBTask,
    NewtonBenchAdapter,
    enumerate_nb_tasks,
)
from mars.agents.generator import Generator
from mars.agents.memory_selector import MemorySelector
from mars.agents.code_evolver import CodeEvolver
from mars.agents.reflector import Reflector
from mars.coordinator import Coordinator
from ols.core.abandon import FutilityDetector


ABLATIONS = [
    ("MARS-full",
     dict(use_reflector=True,  use_memory_selector=True,  use_futility_detector=True,
          use_code_evolver=False)),
    # MARS-SELF: Programmatic Completeness Gate (gate-only; no genetics module gen)
    ("MARS-self",
     dict(use_reflector=True,  use_memory_selector=True,  use_futility_detector=True,
          use_code_evolver=True)),
    ("MARS-no-ref",
     dict(use_reflector=False, use_memory_selector=True,  use_futility_detector=True,
          use_code_evolver=False)),
    ("MARS-no-mem",
     dict(use_reflector=True,  use_memory_selector=False, use_futility_detector=True,
          use_code_evolver=False)),
    ("MARS-no-fut",
     dict(use_reflector=True,  use_memory_selector=True,  use_futility_detector=False,
          use_code_evolver=False)),
    ("MARS-all-off",
     dict(use_reflector=False, use_memory_selector=False, use_futility_detector=False,
          use_code_evolver=False)),
]


def _episode(task: NBTask, ablation: dict, gen_model: str, ref_model: str,
             judge_model: str, budget: float = 10.0,
             auto_fit_in_results: bool = True) -> dict:
    ablation = dict(ablation)   # copy — don't mutate shared ABLATIONS dict
    adapter = NewtonBenchAdapter(task=task, budget=budget,
                                 judge_model=judge_model,
                                 auto_fit_in_results=auto_fit_in_results)
    gen = Generator(model=gen_model, max_tokens=1600)
    ref = Reflector(model=ref_model, max_tokens=300)
    sel = MemorySelector(k=4)

    # MARS-SELF gate-only mode for NewtonBench: max_modules=0 disables the
    # genetics-specific dynamic module generator; only the domain-agnostic
    # Programmatic Completeness Gate (rubric) operates here.
    use_ce = ablation.pop("use_code_evolver", False)
    evolver = CodeEvolver(model=ref_model, max_modules=0) if use_ce else None

    # NewtonBench-specific: FutilityDetector must NOT abandon "discover" sub-
    # goal during exploration. Claims are emitted only at terminal submit_law,
    # so the default theta_futile=0.30 would kill the agent after 3 rounds.
    # Set theta high enough to never fire on a single "discover" sub-goal.
    coord = Coordinator(
        adapter=adapter,
        generator=gen,
        reflector=ref,
        memory_selector=sel,
        futility=FutilityDetector(theta_futile=2.0, min_absolute_spend=1e9),
        max_turns_per_subgoal=int(budget) + 1,
        max_total_turns=int(budget) + 2,
        verbose=False,
        code_evolver=evolver,
        use_code_evolver=use_ce,
        **ablation,
    )
    t0 = time.time()
    rep = coord.run_episode()
    elapsed = time.time() - t0
    return {
        "task_label": task.label(),
        "module": task.module_name,
        "difficulty": task.difficulty,
        "law_version": task.law_version,
        "system": task.system,
        "noise_level": task.noise_level,
        "ablation": ablation,
        "primary": rep.primary,
        "SA": rep.score_dict.get("SA", 0.0),
        "numerical_accuracy": rep.score_dict.get("numerical_accuracy", 0.0),
        "rmsle": rep.score_dict.get("rmsle", float("nan")),
        "symbolic_equivalent": rep.score_dict.get("symbolic_equivalent", False),
        "submitted_preview": rep.score_dict.get("submitted", "")[:300],
        "judge_explain": rep.score_dict.get("explain", "")[:250],
        "n_turns": rep.n_turns,
        "n_actions": rep.n_actions,
        "n_claims_active": rep.n_claims_active,
        "n_generator_calls": rep.n_generator_calls,
        "n_reflector_calls": rep.n_reflector_calls,
        "verdict_counts": rep.verdict_counts,
        "code_evolver_stats": rep.code_evolver_stats,
        "wall_time_s": elapsed,
    }


def _parse_csv(env_name: str, default: str) -> list[str]:
    v = os.environ.get(env_name, default)
    return [x.strip() for x in v.split(",") if x.strip()]


class EpisodeTimeout(RuntimeError):
    pass


@contextmanager
def _episode_time_limit(seconds: int):
    if seconds <= 0:
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise EpisodeTimeout(f"episode exceeded timeout_s={seconds}")

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _load_checkpoint(path: Path) -> dict[tuple[str, str], dict]:
    completed: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        tag = str(row.get("ablation_tag", row.get("tag", "")))
        task_label = str(row.get("task_label", ""))
        if tag and task_label:
            completed[(tag, task_label)] = row
    return completed


def _append_checkpoint(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        f.flush()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    modules = _parse_csv("MARS_NB_MODULES",
                         "m0_gravity,m1_coulomb_force,m4_snell_law,m7_malus_law")
    difficulties = _parse_csv("MARS_NB_DIFFICULTIES", "hard")
    law_versions = _parse_csv("MARS_NB_LAW_VERSIONS", "v0")
    systems = _parse_csv("MARS_NB_SYSTEMS", "vanilla_equation")
    noise = [float(x) for x in _parse_csv("MARS_NB_NOISE", "0.0")]
    trials_per_combo = int(os.environ.get("MARS_NB_TRIALS", "1"))
    budget = float(os.environ.get("MARS_NB_BUDGET", "10"))
    gen_model = os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini")
    ref_model = os.environ.get("MARS_REFLECTOR_MODEL", "openai/gpt-4o-mini")
    judge_model = os.environ.get("MARS_NB_JUDGE_MODEL", "gpt41")
    ablations_env = os.environ.get("MARS_ABLATIONS", "MARS-full,MARS-all-off")
    chosen = [a for a in ABLATIONS if a[0] in {x.strip() for x in ablations_env.split(",")}]
    auto_fit = os.environ.get("MARS_NB_AUTOFIT", "1") not in ("0", "false", "False", "no")
    episode_timeout_s = int(float(os.environ.get("MARS_NB_EPISODE_TIMEOUT_S", "0")))

    tasks = enumerate_nb_tasks(
        modules=modules,
        difficulties=difficulties,
        law_versions=law_versions,
        systems=systems,
        noise_levels=noise,
        trials_per_combo=trials_per_combo,
    )

    suffix = f"_noautofit" if not auto_fit else ""
    run_label = os.environ.get("MARS_RUN_LABEL", "").strip()
    if run_label:
        suffix += f"_{run_label}"
    out = _PROJ / "lmw" / f"mars_nb_{gen_model.replace('/','-')}_{len(tasks)}t{suffix}.json"
    checkpoint_path = Path(
        os.environ.get("MARS_NB_CHECKPOINT_PATH", str(out.with_suffix(".jsonl")))
    )
    resume = os.environ.get("MARS_NB_RESUME", "1") not in ("0", "false", "False", "no")
    completed = _load_checkpoint(checkpoint_path) if resume else {}

    print(f"\n=== MARS × NewtonBench (SA via authors' judge) ===")
    print(f"  gen={gen_model}  ref={ref_model}  judge={judge_model}")
    print(f"  N={len(tasks)} tasks "
          f"(modules={modules}, difficulties={difficulties}, "
          f"law_versions={law_versions}, systems={systems}, "
          f"noise={noise}, trials={trials_per_combo})")
    print(f"  ablations: {[a[0] for a in chosen]}")
    print(f"  checkpoint={checkpoint_path} resume={resume} "
          f"episode_timeout_s={episode_timeout_s}\n")

    results: dict[str, list[dict]] = {a[0]: [] for a in chosen}
    for (tag, _task_label), row in completed.items():
        if tag in results:
            results[tag].append(row)
    t_start = time.time()
    total_eps = len(tasks) * len(chosen)
    done = 0

    for tag, abl in chosen:
        print(f"--- ablation: {tag} ---")
        for t in tasks:
            done += 1
            key = (tag, t.label())
            if key in completed:
                r = completed[key]
                print(f"  [{done}/{total_eps}] {t.label():<50}  "
                      f"SKIP checkpoint  SA={r.get('SA', 0):.2f}",
                      flush=True)
                continue
            try:
                with _episode_time_limit(episode_timeout_s):
                    r = _episode(t, abl, gen_model, ref_model, judge_model, budget,
                                 auto_fit_in_results=auto_fit)
            except EpisodeTimeout as e:
                r = {
                    "task_label": t.label(),
                    "module": t.module_name,
                    "difficulty": t.difficulty,
                    "law_version": t.law_version,
                    "system": t.system,
                    "noise_level": t.noise_level,
                    "error": str(e),
                    "timed_out": True,
                    "SA": 0.0,
                    "wall_time_s": float(episode_timeout_s),
                }
            except Exception as e:
                r = {"task_label": t.label(), "module": t.module_name,
                     "error": str(e)[:200], "SA": 0.0}
            r["ablation_tag"] = tag
            results[tag].append(r)
            _append_checkpoint(checkpoint_path, r)
            print(f"  [{done}/{total_eps}] {t.label():<50}  "
                  f"SA={r.get('SA', 0):.2f}  "
                  f"rmsle={r.get('rmsle', float('nan'))!s:.6}...  "
                  f"turns={r.get('n_turns', 0)}  "
                  f"({r.get('wall_time_s', 0):.0f}s)",
                  flush=True)

    print(f"\n=== Summary (wall {time.time()-t_start:.0f}s) ===")
    summary = {"generator_model": gen_model, "reflector_model": ref_model,
               "judge_model": judge_model, "budget": budget,
               "n_tasks": len(tasks), "ablations_run": [a[0] for a in chosen],
               "modules": modules, "difficulties": difficulties,
               "law_versions": law_versions, "systems": systems,
               "noise": noise, "trials_per_combo": trials_per_combo,
               "per_task": results}
    for tag, rows in results.items():
        scores = [r.get("SA", 0.0) for r in rows]
        nas = [r.get("numerical_accuracy", 0.0) for r in rows]
        m = st_stats.fmean(scores) if scores else 0.0
        sd = st_stats.stdev(scores) if len(scores) > 1 else 0.0
        na_m = st_stats.fmean(nas) if nas else 0.0
        n_sym = sum(1 for r in rows if r.get("symbolic_equivalent"))
        print(f"  {tag:<14}  SA.mean={m:.3f}  sd={sd:.3f}  "
              f"num_acc.mean={na_m:.3f}  "
              f"symbolic_match={n_sym}/{len(scores)}")

    if len(chosen) == 2 and all(len(results[a[0]]) == len(tasks) for a in chosen):
        a_name, b_name = chosen[0][0], chosen[1][0]
        paired = [(results[a_name][i].get("SA", 0.0),
                   results[b_name][i].get("SA", 0.0))
                  for i in range(len(tasks))]
        deltas = [p[0] - p[1] for p in paired]
        if deltas:
            m_d = st_stats.fmean(deltas)
            sd_d = st_stats.stdev(deltas) if len(deltas) > 1 else 0.0
            se_d = sd_d / (len(deltas) ** 0.5) if len(deltas) > 1 else 0.0
            wins = sum(1 for d in deltas if d > 0.05)
            losses = sum(1 for d in deltas if d < -0.05)
            ties = len(deltas) - wins - losses
            print(f"\n  Paired ΔSA ({a_name} − {b_name}) = "
                  f"{m_d:+.3f} ± {1.96*se_d:.3f} (95% CI z)  "
                  f"w/l/t={wins}/{losses}/{ties}")
            summary["paired_delta_SA_mean"] = m_d
            summary["paired_delta_SA_ci95z"] = 1.96 * se_d

    with open(out, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"\n[written {out}]")


if __name__ == "__main__":
    main()
