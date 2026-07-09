"""
Run OLS overlay on DiscoveryBench-Real (train split, labeled).

The convergent-validity probe for OLS: same inner LLM (gpt-4o-mini by default)
with and without the OLS overlay, on the same set of DB-Real-train tasks.
Headline number is the paired Δ HMS = HMS(OLS-full) − HMS(OLS-all-off).

Each task is one episode. We pair task ↔ ablation so HMS deltas are computed
on the SAME task (paired design), maximizing power at small N. Judge model is
configurable (default openai/gpt-4o) — cheaper than the published
gpt-4-0125-preview but matches HMS-style facet scoring.

Outputs JSON to lmw/ols_db_<inner>_<judge>_<n>.json (lives next to the
other adapter dumps so metrics_multi.py can ingest later if we add a DB
table to the leaderboard).
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

from ols.adapters.discoverybench_adapter import (
    DBTask,
    DiscoveryBenchAdapter,
    load_db_tasks,
)
from ols.inner_agents.llm_inner import OLSInnerLLM
from ols.scaffold import OLSScaffold


# OLS-full enables the v0.2 load-bearing gate (require_claim_before_advance),
# which forces the inner agent to actually use the ClaimStore before
# advancing through the universal-research-process sub-goals seeded by the
# adapter. OLS-all-off is the clean no-overlay baseline (all three OLS
# modules off, gate off).
ABLATIONS = [
    ("OLS-full",
     dict(use_persistent_store=True, use_agenda_controller=True,
          use_futility_detector=True, require_claim_before_advance=True,
          min_claims_per_subgoal=1)),
    ("OLS-all-off",
     dict(use_persistent_store=False, use_agenda_controller=False,
          use_futility_detector=False, require_claim_before_advance=False)),
]


def _episode(task: DBTask, ablation: dict, model: str, budget: float,
             judge_model: str) -> dict:
    adapter = DiscoveryBenchAdapter(task=task, budget=budget,
                                    judge_model=judge_model)
    inner = OLSInnerLLM(model=model)
    scaffold = OLSScaffold(
        adapter=adapter,
        inner_agent=inner,
        max_total_inner_calls=int(budget) + 4,    # slack for terminal turns
        max_inner_calls_per_subgoal=4,            # 4 sub-goals × 4 turns = 16 max
        do_final_retest=False,                    # adapter has no oracle
        verbose=False,
        **ablation,
    )
    t0 = time.time()
    rep = scaffold.run_episode()
    elapsed = time.time() - t0
    return {
        "task_id": task.task_id,
        "domain": task.domain,
        "query_type": task.query_type,
        "ablation": ablation,
        "primary": rep.primary,
        "HMS": rep.score_dict.get("HMS", 0.0),
        "submitted": rep.score_dict.get("submitted", "")[:300],
        "judge_explain": rep.score_dict.get("judge_explain", "")[:300],
        "n_actions": rep.n_actions,
        "n_claims_active": rep.n_claims_active,
        "n_claims_retracted": rep.n_claims_retracted,
        "n_subgoals_done": rep.n_subgoals_done,
        "n_subgoals_abandoned": rep.n_subgoals_abandoned,
        "axis1_ratio": rep.axis1_ratio,
        "axis6_regret": rep.axis6_regret,
        "wall_time_s": elapsed,
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    repo = _PROJ / "discoverybench_repo"
    n_tasks = int(os.environ.get("OLS_DB_N", "20"))
    inner_model = os.environ.get("OLS_INNER_MODEL", "openai/gpt-4o-mini")
    judge_model = os.environ.get("OLS_DB_JUDGE_MODEL", "openai/gpt-4o")
    budget = float(os.environ.get("OLS_DB_BUDGET", "12"))
    ablations_env = os.environ.get("OLS_ABLATIONS", "OLS-full,OLS-all-off")
    chosen = [a for a in ABLATIONS if a[0] in {x.strip() for x in ablations_env.split(",")}]

    tasks = load_db_tasks(repo, split="train", max_tasks=n_tasks)
    print(f"\n=== OLS × DiscoveryBench-Real-train ===")
    print(f"  inner={inner_model}  judge={judge_model}  budget={budget} actions/episode")
    print(f"  tasks: {len(tasks)} (stratified by file order); "
          f"ablations: {[a[0] for a in chosen]}\n")

    results: dict[str, list[dict]] = {a[0]: [] for a in chosen}
    t_start = time.time()
    total_episodes = len(tasks) * len(chosen)
    done = 0

    for tag, abl in chosen:
        print(f"--- ablation: {tag} ---")
        for t in tasks:
            done += 1
            try:
                r = _episode(t, abl, inner_model, budget, judge_model)
            except Exception as e:
                r = {"task_id": t.task_id, "error": str(e)[:200], "HMS": 0.0}
            results[tag].append(r)
            print(f"  [{done}/{total_episodes}] {t.task_id[:60]:<60}  "
                  f"HMS={r.get('HMS', 0):.2f}  "
                  f"n_act={r.get('n_actions', 0):>2}  "
                  f"({r.get('wall_time_s', 0):.0f}s)  "
                  f"submitted={r.get('submitted', '')[:60]!r}",
                  flush=True)

    print(f"\n=== Summary (total wall {time.time()-t_start:.0f}s) ===")
    summary = {"inner_model": inner_model, "judge_model": judge_model,
               "budget": budget, "n_tasks": len(tasks),
               "ablations_run": [a[0] for a in chosen],
               "per_task": results}

    by_tag = {tag: [r.get("HMS", 0.0) for r in rows]
              for tag, rows in results.items()}
    for tag, scores in by_tag.items():
        m = st_stats.fmean(scores) if scores else 0.0
        sd = st_stats.stdev(scores) if len(scores) > 1 else 0.0
        print(f"  {tag:<14}  HMS.mean={m:.3f}  HMS.sd={sd:.3f}  N={len(scores)}")

    if len(chosen) == 2 and all(len(by_tag[a[0]]) == len(tasks) for a in chosen):
        a_name, b_name = chosen[0][0], chosen[1][0]
        paired = [(by_tag[a_name][i], by_tag[b_name][i]) for i in range(len(tasks))]
        deltas = [p[0] - p[1] for p in paired]
        m_d = st_stats.fmean(deltas)
        sd_d = st_stats.stdev(deltas) if len(deltas) > 1 else 0.0
        se_d = sd_d / (len(deltas) ** 0.5) if deltas else 0.0
        ci = 1.96 * se_d
        wins = sum(1 for d in deltas if d > 0.05)
        losses = sum(1 for d in deltas if d < -0.05)
        ties = len(deltas) - wins - losses
        print(f"\n  Paired Δ HMS ({a_name} − {b_name}) = {m_d:+.3f} ± {ci:.3f} (95% CI)")
        print(f"  per-task wins/losses/ties (|Δ|>0.05): {wins}/{losses}/{ties}")
        summary["paired_delta_HMS_mean"] = m_d
        summary["paired_delta_HMS_ci95"] = ci
        summary["paired_wins"] = wins
        summary["paired_losses"] = losses
        summary["paired_ties"] = ties

    out = _PROJ / "lmw" / f"ols_db_{inner_model.replace('/','-')}_{len(tasks)}t.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n[written {out}]")


if __name__ == "__main__":
    main()
