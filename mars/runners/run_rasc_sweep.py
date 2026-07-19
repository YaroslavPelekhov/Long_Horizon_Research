"""
RASC power-sweep — the self-configuring architecture across all 3 benchmarks,
multiple tasks each, paired against the fixed-architecture baselines already on
record.

Fixed-architecture reference numbers (gpt-4o-mini), from prior sweeps:
  UltraHorizon:  never-gate (MARS-full)=20.0 (5 seeds) | always-gate=87.0 (5 seeds)
  NewtonBench:   never-gate easy SA=0.667 / med=0.167 / hard=0.0
  DiscoveryBench:never-gate (MARS-full)=0.680 (6 tasks) | always-gate=0.570

RASC should self-route to: UH→completeness, NB→fit, DB→reasoning, reaching the
per-task optimum on each (i.e. ≈ the BEST fixed config per benchmark).

ENV:
  MARS_RASC_BENCH   csv subset of {uh,nb,db}   default "uh,nb,db"
  MARS_UH_SEEDS     default "42,43,44,45,46"
  MARS_NB_MODULES   default "m0_gravity,m9_hooke_law,m5_radioactive_decay"
  MARS_NB_DIFFS     default "easy,medium,hard"
  MARS_DB_N         default 6

Run:  python -m mars.runners.run_rasc_sweep
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from dotenv import load_dotenv
    _env = _PROJ / "autodiscovery" / ".env.local"
    if _env.exists():
        load_dotenv(_env, override=False)
except ImportError:
    pass

from mars.agents.generator import Generator
from mars.agents.reflector import Reflector
from mars.agents.memory_selector import MemorySelector
from mars.agents.code_evolver import CodeEvolver
from mars.rasc.controller import RASCController

GEN = os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini")
REF = os.environ.get("MARS_REFLECTOR_MODEL", "openai/gpt-4o-mini")


def _ctrl(adapter, gen_tok, ref_tok, k, mtps, mtt, ce=False):
    return RASCController(
        adapter=adapter,
        generator=Generator(model=GEN, max_tokens=gen_tok),
        reflector=Reflector(model=REF, max_tokens=ref_tok),
        memory_selector=MemorySelector(k=k),
        code_evolver_factory=(lambda: CodeEvolver(model=REF, max_modules=6)) if ce else None,
        max_turns_per_subgoal=mtps, max_total_turns=mtt,
    )


def sweep_uh():
    from mars.adapters.ultrahorizon_bio_adapter import UHBioTask, UltraHorizonBioAdapter
    seeds = [int(s) for s in os.environ.get("MARS_UH_SEEDS", "42,43,44,45,46").split(",")]
    rows = []
    print("--- RASC × UltraHorizon ---", flush=True)
    for sd in seeds:
        with contextlib.redirect_stdout(io.StringIO()):
            adapter = UltraHorizonBioAdapter(
                task=UHBioTask(seed=sd, budget=20, judge_model="openai/gpt-4o"))
        ctrl = _ctrl(adapter, 2000, 500, 6, 9, 40, ce=True)
        t0 = time.time()
        try:
            rep, mode, sig = ctrl.run_episode()
            sc = rep.score_dict.get("final_score", 0.0)
        except Exception as e:
            mode, sc = "ERR", 0.0
            print("   err", e)
        rows.append({"task": f"seed{sd}", "mode": mode, "score": sc})
        print(f"  seed{sd:<3} mode={mode:<12} score={sc:.1f}/100  ({time.time()-t0:.0f}s)", flush=True)
    return rows


def sweep_nb():
    from mars.adapters.newtonbench_adapter import NBTask, NewtonBenchAdapter
    mods = os.environ.get("MARS_NB_MODULES", "m0_gravity,m9_hooke_law,m5_radioactive_decay").split(",")
    diffs = os.environ.get("MARS_NB_DIFFS", "easy,medium,hard").split(",")
    rows = []
    print("--- RASC × NewtonBench ---", flush=True)
    for m in mods:
        for d in diffs:
            adapter = NewtonBenchAdapter(task=NBTask(m.strip(), d.strip(), "v0"),
                                         budget=10, judge_model="gpt41")
            ctrl = _ctrl(adapter, 1600, 300, 4, 11, 12, ce=False)
            t0 = time.time()
            try:
                rep, mode, sig = ctrl.run_episode()
                sa = rep.score_dict.get("SA", 0.0)
                na = rep.score_dict.get("numerical_accuracy", 0.0)
            except Exception as e:
                mode, sa, na = "ERR", 0.0, 0.0
                print("   err", e)
            rows.append({"task": f"{m.strip()}/{d.strip()}", "mode": mode, "score": sa, "num_acc": na})
            print(f"  {m.strip():<18}/{d.strip():<7} mode={mode:<5} SA={sa:.2f} num_acc={na:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    return rows


def sweep_db():
    from mars.adapters.discoverybench_adapter import load_db_tasks
    from ols.adapters.discoverybench_adapter import DiscoveryBenchAdapter
    n = int(os.environ.get("MARS_DB_N", "6"))
    tasks = load_db_tasks(_PROJ / "discoverybench_repo", split="train", max_tasks=n)
    rows = []
    print("--- RASC × DiscoveryBench ---", flush=True)
    for t in tasks:
        adapter = DiscoveryBenchAdapter(task=t, budget=12, judge_model="openai/gpt-4o")
        ctrl = _ctrl(adapter, 900, 300, 4, 4, 16, ce=False)
        t0 = time.time()
        try:
            rep, mode, sig = ctrl.run_episode()
            hms = rep.score_dict.get("HMS", 0.0)
        except Exception as e:
            mode, hms = "ERR", 0.0
            print("   err", e)
        rows.append({"task": t.task_id[-30:], "mode": mode, "score": hms})
        print(f"  {t.task_id[-34:]:<34} mode={mode:<11} HMS={hms:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    return rows


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    which = [b.strip() for b in os.environ.get("MARS_RASC_BENCH", "uh,nb,db").split(",")]
    print(f"\n=== RASC POWER-SWEEP (inner={GEN}) ===  benches={which}\n")
    t_start = time.time()
    out = {}

    if "uh" in which:
        rows = sweep_uh()
        out["uh"] = rows
        sm = [r["score"] for r in rows if r["mode"] != "ERR"]
        modes = {r["mode"] for r in rows}
        print(f"  UH RASC mean={st.fmean(sm):.1f}/100 (modes={modes})  "
              f"vs fixed never-gate=20.0, always-gate=87.0\n")

    if "nb" in which:
        rows = sweep_nb()
        out["nb"] = rows
        sa = [r["score"] for r in rows if r["mode"] != "ERR"]
        na = [r.get("num_acc", 0.0) for r in rows if r["mode"] != "ERR"]
        modes = {r["mode"] for r in rows}
        print(f"  NB RASC SA.mean={st.fmean(sa):.3f} num_acc.mean={st.fmean(na):.3f} "
              f"(modes={modes})\n")

    if "db" in which:
        rows = sweep_db()
        out["db"] = rows
        hm = [r["score"] for r in rows if r["mode"] != "ERR"]
        modes = {r["mode"] for r in rows}
        print(f"  DB RASC HMS.mean={st.fmean(hm):.3f} (modes={modes})  "
              f"vs fixed never-gate=0.680, always-gate=0.570\n")

    # routing accuracy
    expect = {"uh": "completeness", "nb": "fit", "db": "reasoning"}
    correct = total = 0
    for b, rows in out.items():
        for r in rows:
            total += 1
            if r["mode"] == expect[b]:
                correct += 1
    print(f"=== RASC self-diagnosis accuracy: {correct}/{total} correct "
          f"({100*correct/max(1,total):.0f}%) ===")
    print(f"=== wall {time.time()-t_start:.0f}s ===")

    outpath = _PROJ / "lmw" / f"rasc_sweep_{GEN.replace('/','-')}.json"
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"[written {outpath}]")


if __name__ == "__main__":
    main()
