"""
Run MARS on ScienceAgentBench (verified split, 102 tasks).

v0.1: scoring via LLM-judge (gpt-4o) — proxy metric, not published SAB
score. Used for architecture validation + paired comparison MARS-full vs
MARS-all-off.

ENV:
  MARS_SAB_N             number of tasks (default 5 for smoke)
  MARS_SAB_DOMAINS       comma-sep filter (e.g. "Computational Chemistry")
  MARS_GENERATOR_MODEL   default openai/gpt-4o-mini
  MARS_REFLECTOR_MODEL   default openai/gpt-4o-mini
  MARS_SAB_JUDGE_MODEL   default openai/gpt-4o
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

from mars.adapters.scienceagentbench_adapter import (
    SABTask,
    ScienceAgentBenchAdapter,
    load_sab_tasks,
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


def _episode(task: SABTask, ablation: dict, gen_model: str, ref_model: str,
             judge_model: str, budget: float = 4.0) -> dict:
    adapter = ScienceAgentBenchAdapter(task=task, budget=budget,
                                       judge_model=judge_model)
    gen = Generator(model=gen_model, max_tokens=2200)  # SAB programs are long
    ref = Reflector(model=ref_model, max_tokens=300)
    sel = MemorySelector(k=4)
    coord = Coordinator(
        adapter=adapter,
        generator=gen,
        reflector=ref,
        memory_selector=sel,
        max_turns_per_subgoal=int(budget) + 1,
        max_total_turns=int(budget) + 2,
        verbose=False,
        **ablation,
    )
    t0 = time.time()
    rep = coord.run_episode()
    elapsed = time.time() - t0
    return {
        "instance_id": task.instance_id,
        "domain": task.domain,
        "ablation": ablation,
        "primary": rep.primary,
        "score": rep.score_dict.get("score", 0.0),
        "n_submissions": rep.score_dict.get("n_submissions", 0),
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

    n_tasks = int(os.environ.get("MARS_SAB_N", "5"))
    domains_env = os.environ.get("MARS_SAB_DOMAINS", "")
    domains = [d.strip() for d in domains_env.split(",") if d.strip()] or None
    gen_model = os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini")
    ref_model = os.environ.get("MARS_REFLECTOR_MODEL", "openai/gpt-4o-mini")
    judge_model = os.environ.get("MARS_SAB_JUDGE_MODEL", "openai/gpt-4o")
    ablations_env = os.environ.get("MARS_ABLATIONS", "MARS-full,MARS-all-off")
    chosen = [a for a in ABLATIONS if a[0] in {x.strip() for x in ablations_env.split(",")}]

    tasks = load_sab_tasks(max_tasks=n_tasks, domains=domains)
    print(f"\n=== MARS × ScienceAgentBench (verified split, v0.1 LLM-judge) ===")
    print(f"  gen={gen_model}  ref={ref_model}  judge={judge_model}")
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
                r = _episode(t, abl, gen_model, ref_model, judge_model)
            except Exception as e:
                r = {"instance_id": t.instance_id, "domain": t.domain,
                     "error": str(e)[:200], "score": 0.0}
            results[tag].append(r)
            print(f"  [{done}/{total_eps}] id={t.instance_id} {t.domain[:25]:<25}  "
                  f"score={r.get('score', 0):.2f}  "
                  f"subs={r.get('n_submissions', 0)}  "
                  f"turns={r.get('n_turns', 0)}  "
                  f"({r.get('wall_time_s', 0):.0f}s)",
                  flush=True)

    print(f"\n=== Summary (wall {time.time()-t_start:.0f}s) ===")
    summary = {"generator_model": gen_model, "reflector_model": ref_model,
               "judge_model": judge_model, "n_tasks": len(tasks),
               "ablations_run": [a[0] for a in chosen], "per_task": results}
    for tag, rows in results.items():
        scores = [r.get("score", 0.0) for r in rows]
        m = st_stats.fmean(scores) if scores else 0.0
        sd = st_stats.stdev(scores) if len(scores) > 1 else 0.0
        print(f"  {tag:<14}  score.mean={m:.3f}  sd={sd:.3f}  N={len(scores)}")

    if len(chosen) == 2 and all(len(results[a[0]]) == len(tasks) for a in chosen):
        a_name, b_name = chosen[0][0], chosen[1][0]
        paired = [(results[a_name][i].get("score", 0.0),
                   results[b_name][i].get("score", 0.0))
                  for i in range(len(tasks))]
        deltas = [p[0] - p[1] for p in paired]
        if deltas:
            m_d = st_stats.fmean(deltas)
            sd_d = st_stats.stdev(deltas) if len(deltas) > 1 else 0.0
            se_d = sd_d / (len(deltas) ** 0.5)
            wins = sum(1 for d in deltas if d > 0.05)
            losses = sum(1 for d in deltas if d < -0.05)
            ties = len(deltas) - wins - losses
            print(f"\n  Paired Δ score ({a_name} − {b_name}) = "
                  f"{m_d:+.3f} ± {1.96*se_d:.3f} (95% CI z)")
            print(f"  per-task wins/losses/ties (|Δ|>0.05): "
                  f"{wins}/{losses}/{ties}")
            summary["paired_delta_score_mean"] = m_d
            summary["paired_delta_score_ci95z"] = 1.96 * se_d
            summary["paired_wins"] = wins
            summary["paired_losses"] = losses
            summary["paired_ties"] = ties

    out = _PROJ / "lmw" / f"mars_sab_{gen_model.replace('/','-')}_{len(tasks)}t.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n[written {out}]")


if __name__ == "__main__":
    main()
