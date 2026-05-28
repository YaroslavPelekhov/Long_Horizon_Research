"""
Run MARS on UltraHorizon — Alien Genetics Lab (long-horizon research target).

The agent discovers hidden genetic-inheritance rules of a triploid alien
organism by designing crosses and observing phenotypes. Scoring: 100-point
LLM rubric. This is a genuine long-horizon scientific-discovery task.

Published UltraHorizon human baseline: ~100/100.
OpenManus / GPT-4o baseline: significantly lower (paper TBD; see README).

ENV:
  MARS_UH_SEEDS          comma-sep int seeds, default "42,43,44"
  MARS_UH_BUDGET         crosses per episode, default 20
  MARS_UH_JUDGE_MODEL    scoring LLM, default "openai/gpt-4o"
  MARS_GENERATOR_MODEL   default "openai/gpt-4o-mini"
  MARS_REFLECTOR_MODEL   default "openai/gpt-4o-mini"
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

from mars.adapters.ultrahorizon_bio_adapter import UHBioTask, UltraHorizonBioAdapter
from mars.agents.generator import Generator
from mars.agents.memory_selector import MemorySelector
from mars.agents.reflector import Reflector
from mars.coordinator import Coordinator
from ols.core.abandon import FutilityDetector


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


def _parse_csv(env_name: str, default: str) -> list[str]:
    v = os.environ.get(env_name, default)
    return [x.strip() for x in v.split(",") if x.strip()]


def _episode(task: UHBioTask, ablation: dict,
             gen_model: str, ref_model: str) -> dict:
    adapter = UltraHorizonBioAdapter(task=task)
    gen = Generator(model=gen_model, max_tokens=2000)
    ref = Reflector(model=ref_model, max_tokens=500)
    sel = MemorySelector(k=6)
    # 4 explicit research phases (A–D) in handle().subdomains.
    # Give each phase up to 9 generator turns; 40 total provides a buffer.
    # Futility detector disabled — genetics requires patient accumulation.
    coord = Coordinator(
        adapter=adapter,
        generator=gen,
        reflector=ref,
        memory_selector=sel,
        futility=FutilityDetector(theta_futile=2.0, min_absolute_spend=1e9),
        max_turns_per_subgoal=9,
        max_total_turns=40,
        verbose=False,
        **ablation,
    )
    t0 = time.time()
    rep = coord.run_episode()
    elapsed = time.time() - t0
    return {
        "task_label": task.label(),
        "seed": task.seed,
        "budget": task.budget,
        "ablation": ablation,
        "primary": rep.primary,
        "final_score": rep.score_dict.get("final_score", 0.0),
        "judge_result": rep.score_dict.get("judge_result", {}),
        "n_crosses": rep.score_dict.get("n_crosses", 0),
        "submitted_preview": rep.score_dict.get("submitted", "")[:300],
        "n_turns": rep.n_turns,
        "n_actions": rep.n_actions,
        "n_claims_active": rep.n_claims_active,
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

    seeds = [int(s) for s in _parse_csv("MARS_UH_SEEDS", "42,43,44")]
    budget = int(os.environ.get("MARS_UH_BUDGET", "20"))
    judge_model = os.environ.get("MARS_UH_JUDGE_MODEL", "openai/gpt-4o")
    gen_model = os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini")
    ref_model = os.environ.get("MARS_REFLECTOR_MODEL", "openai/gpt-4o-mini")
    ablations_env = os.environ.get("MARS_ABLATIONS", "MARS-full,MARS-all-off")
    chosen = [a for a in ABLATIONS if a[0] in {x.strip() for x in ablations_env.split(",")}]

    tasks = [UHBioTask(seed=s, budget=budget, judge_model=judge_model) for s in seeds]

    print(f"\n=== MARS × UltraHorizon Genetics Lab (long-horizon scientific discovery) ===")
    print(f"  gen={gen_model}  ref={ref_model}  judge={judge_model}")
    print(f"  N={len(tasks)} episodes (seeds={seeds}), budget={budget} crosses/episode")
    print(f"  ablations: {[a[0] for a in chosen]}\n")

    results: dict[str, list[dict]] = {a[0]: [] for a in chosen}
    t_start = time.time()
    total_eps = len(tasks) * len(chosen)
    done = 0

    for tag, abl in chosen:
        print(f"--- ablation: {tag} ---")
        for t in tasks:
            done += 1
            try:
                r = _episode(t, abl, gen_model, ref_model)
            except Exception as e:
                r = {"task_label": t.label(), "seed": t.seed,
                     "error": str(e)[:200], "final_score": 0.0, "primary": 0.0}
            results[tag].append(r)
            print(f"  [{done}/{total_eps}] {t.label():<35}  "
                  f"score={r.get('final_score', 0):.1f}/100  "
                  f"crosses={r.get('n_crosses', 0)}  "
                  f"turns={r.get('n_turns', 0)}  "
                  f"({r.get('wall_time_s', 0):.0f}s)",
                  flush=True)

    print(f"\n=== Summary (wall {time.time()-t_start:.0f}s) ===")
    summary = {
        "generator_model": gen_model, "reflector_model": ref_model,
        "judge_model": judge_model, "budget": budget,
        "n_episodes": len(tasks), "seeds": seeds,
        "ablations_run": [a[0] for a in chosen],
        "per_episode": results,
    }

    for tag, rows in results.items():
        scores = [r.get("final_score", 0.0) for r in rows]
        m = st_stats.fmean(scores) if scores else 0.0
        sd = st_stats.stdev(scores) if len(scores) > 1 else 0.0
        print(f"  {tag:<14}  score.mean={m:.1f}  sd={sd:.1f}  N={len(scores)}")

    if len(chosen) == 2 and all(len(results[a[0]]) == len(tasks) for a in chosen):
        a_name, b_name = chosen[0][0], chosen[1][0]
        deltas = [results[a_name][i].get("final_score", 0.0) -
                  results[b_name][i].get("final_score", 0.0)
                  for i in range(len(tasks))]
        if deltas:
            m_d = st_stats.fmean(deltas)
            sd_d = st_stats.stdev(deltas) if len(deltas) > 1 else 0.0
            se_d = sd_d / (len(deltas) ** 0.5) if len(deltas) > 1 else 0.0
            wins = sum(1 for d in deltas if d > 2)
            losses = sum(1 for d in deltas if d < -2)
            ties = len(deltas) - wins - losses
            print(f"\n  Paired Δscore ({a_name} − {b_name}) = "
                  f"{m_d:+.1f} ± {1.96*se_d:.1f} (95% CI z)  "
                  f"w/l/t={wins}/{losses}/{ties}")
            summary["paired_delta_score_mean"] = m_d
            summary["paired_delta_score_ci95z"] = 1.96 * se_d

    out = _PROJ / "lmw" / f"mars_uh_bio_v2_{gen_model.replace('/', '-')}_{len(tasks)}eps.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"\n[written {out}]")


if __name__ == "__main__":
    main()
