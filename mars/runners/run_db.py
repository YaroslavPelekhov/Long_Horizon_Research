"""
Run MARS on DiscoveryBench (convergent-validity vs OLS-v0.2 baseline).

H5 (pre-registered): MARS shows positive ΔHMS vs the OLS-v0.2 N=25 null on
the same DB-Real-train tasks with the same inner LLM (gpt-4o-mini). This
tests whether the multi-agent architecture closes the gap that OLS left.

ENV:
  MARS_DB_N              n tasks (default 5)
  MARS_DB_BUDGET         actions per episode (default 12)
  MARS_GENERATOR_MODEL   default openai/gpt-4o-mini
  MARS_REFLECTOR_MODEL   default openai/gpt-4o-mini
  OLS_DB_JUDGE_MODEL     default openai/gpt-4o
  MARS_ABLATIONS         default "MARS-full,MARS-all-off"
"""

from __future__ import annotations

import json
import os
import statistics as st_stats
import sys
import time
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from mars.adapters.discoverybench_adapter import (
    DBTask,
    DiscoveryBenchAdapter,
    load_db_tasks,
)
from mars.agents.generator import Generator
from mars.agents.memory_selector import MemorySelector
from mars.agents.reflector import Reflector
from mars.coordinator import Coordinator


ABLATIONS = [
    ("MARS-full",
     dict(use_reflector=True,  use_memory_selector=True,  use_futility_detector=True)),
    ("MARS-no-ref",
     dict(use_reflector=False, use_memory_selector=True,  use_futility_detector=True)),
    ("MARS-no-mem",
     dict(use_reflector=True,  use_memory_selector=False, use_futility_detector=True)),
    ("MARS-no-fut",
     dict(use_reflector=True,  use_memory_selector=True,  use_futility_detector=False)),
    ("MARS-all-off",
     dict(use_reflector=False, use_memory_selector=False, use_futility_detector=False)),
]


def _episode(task: DBTask, ablation: dict, gen_model: str, ref_model: str,
             judge_model: str, budget: float = 12.0) -> dict:
    adapter = DiscoveryBenchAdapter(task=task, budget=budget,
                                    judge_model=judge_model)
    gen = Generator(model=gen_model, max_tokens=900)
    ref = Reflector(model=ref_model, max_tokens=300)
    sel = MemorySelector(k=4)
    coord = Coordinator(
        adapter=adapter,
        generator=gen,
        reflector=ref,
        memory_selector=sel,
        max_turns_per_subgoal=4,
        max_total_turns=int(budget) + 4,
        verbose=False,
        **ablation,
    )
    t0 = time.time()
    rep = coord.run_episode()
    elapsed = time.time() - t0
    return {
        "task_id": task.task_id,
        "domain": task.domain,
        "query_type": task.query_type,
        "ablation": ablation,
        "primary": rep.primary,
        "HMS": rep.score_dict.get("HMS", 0.0),
        "submitted_preview": rep.score_dict.get("submitted", "")[:300],
        "judge_explain": rep.score_dict.get("judge_explain", "")[:250],
        "n_turns": rep.n_turns,
        "n_actions": rep.n_actions,
        "n_claims_active": rep.n_claims_active,
        "n_claims_retracted": rep.n_claims_retracted,
        "n_generator_calls": rep.n_generator_calls,
        "n_reflector_calls": rep.n_reflector_calls,
        "verdict_counts": rep.verdict_counts,
        "wall_time_s": elapsed,
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    repo = _PROJ / "discoverybench_repo"
    n_tasks = int(os.environ.get("MARS_DB_N", "5"))
    gen_model = os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini")
    ref_model = os.environ.get("MARS_REFLECTOR_MODEL", "openai/gpt-4o-mini")
    judge_model = os.environ.get("OLS_DB_JUDGE_MODEL", "openai/gpt-4o")
    budget = float(os.environ.get("MARS_DB_BUDGET", "12"))
    ablations_env = os.environ.get("MARS_ABLATIONS", "MARS-full,MARS-all-off")
    chosen = [a for a in ABLATIONS if a[0] in {x.strip() for x in ablations_env.split(",")}]

    tasks = load_db_tasks(repo, split="train", max_tasks=n_tasks)
    print(f"\n=== MARS × DiscoveryBench-Real-train (convergent validity vs OLS-v0.2) ===")
    print(f"  gen={gen_model}  ref={ref_model}  judge={judge_model}  budget={budget}")
    print(f"  N={len(tasks)} tasks; ablations: {[a[0] for a in chosen]}\n")

    results: dict[str, list[dict]] = {a[0]: [] for a in chosen}
    t_start = time.time()
    total_eps = len(tasks) * len(chosen)
    done = 0

    for tag, abl in chosen:
        print(f"--- ablation: {tag} ---")
        for t in tasks:
            done += 1
            try:
                r = _episode(t, abl, gen_model, ref_model, judge_model, budget)
            except Exception as e:
                r = {"task_id": t.task_id, "domain": t.domain,
                     "error": str(e)[:200], "HMS": 0.0}
            results[tag].append(r)
            print(f"  [{done}/{total_eps}] {t.task_id[:55]:<55}  "
                  f"HMS={r.get('HMS', 0):.2f}  "
                  f"turns={r.get('n_turns', 0)} "
                  f"claims={r.get('n_claims_active', 0)}+{r.get('n_claims_retracted', 0)}r  "
                  f"({r.get('wall_time_s', 0):.0f}s)",
                  flush=True)

    print(f"\n=== Summary (wall {time.time()-t_start:.0f}s) ===")
    summary = {"generator_model": gen_model, "reflector_model": ref_model,
               "judge_model": judge_model, "budget": budget,
               "n_tasks": len(tasks), "ablations_run": [a[0] for a in chosen],
               "per_task": results}
    for tag, rows in results.items():
        scores = [r.get("HMS", 0.0) for r in rows]
        m = st_stats.fmean(scores) if scores else 0.0
        sd = st_stats.stdev(scores) if len(scores) > 1 else 0.0
        print(f"  {tag:<14}  HMS.mean={m:.3f}  sd={sd:.3f}  N={len(scores)}")

    if len(chosen) == 2 and all(len(results[a[0]]) == len(tasks) for a in chosen):
        a_name, b_name = chosen[0][0], chosen[1][0]
        paired = [(results[a_name][i].get("HMS", 0.0),
                   results[b_name][i].get("HMS", 0.0))
                  for i in range(len(tasks))]
        deltas = [p[0] - p[1] for p in paired]
        if deltas:
            m_d = st_stats.fmean(deltas)
            sd_d = st_stats.stdev(deltas) if len(deltas) > 1 else 0.0
            se_d = sd_d / (len(deltas) ** 0.5)
            wins = sum(1 for d in deltas if d > 0.05)
            losses = sum(1 for d in deltas if d < -0.05)
            ties = len(deltas) - wins - losses
            print(f"\n  Paired ΔHMS ({a_name} − {b_name}) = "
                  f"{m_d:+.3f} ± {1.96*se_d:.3f} (95% CI z)  "
                  f"w/l/t={wins}/{losses}/{ties}")
            summary["paired_delta_HMS_mean"] = m_d
            summary["paired_delta_HMS_ci95z"] = 1.96 * se_d

    out = _PROJ / "lmw" / f"mars_db_{gen_model.replace('/','-')}_{len(tasks)}t.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n[written {out}]")


if __name__ == "__main__":
    main()
