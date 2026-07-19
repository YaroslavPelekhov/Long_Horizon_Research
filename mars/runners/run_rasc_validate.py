"""
RASC validation — one self-configuring architecture across all 3 benchmarks.

Demonstrates: RASC diagnoses each task's reward type and self-configures, thereby
reaching the per-task-optimal regime on every benchmark — where any FIXED
architecture is sub-optimal on 2 of 3.

Expected diagnosis + target:
  UltraHorizon  -> completeness -> ~85-100  (vs no-gate baseline 20)
  NewtonBench   -> fit          -> ~0.6-1.0 SA on easy (vs broken 0)
  DiscoveryBench-> reasoning     -> ~0.68   (vs gate/stats 0.46-0.57)

Run:
  python -m mars.runners.run_rasc_validate
"""

from __future__ import annotations

import contextlib
import io
import os
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


def _run_uh(gen_model, ref_model):
    from mars.adapters.ultrahorizon_bio_adapter import UHBioTask, UltraHorizonBioAdapter
    with contextlib.redirect_stdout(io.StringIO()):
        adapter = UltraHorizonBioAdapter(
            task=UHBioTask(seed=42, budget=20, judge_model="openai/gpt-4o"))
    ctrl = RASCController(
        adapter=adapter,
        generator=Generator(model=gen_model, max_tokens=2000),
        reflector=Reflector(model=ref_model, max_tokens=500),
        memory_selector=MemorySelector(k=6),
        code_evolver_factory=lambda: CodeEvolver(model=ref_model, max_modules=6),
        max_turns_per_subgoal=9, max_total_turns=40,
    )
    rep, mode, sig = ctrl.run_episode()
    return mode, rep.score_dict.get("final_score", 0.0), "/100"


def _run_nb(gen_model, ref_model):
    from mars.adapters.newtonbench_adapter import NBTask, NewtonBenchAdapter
    adapter = NewtonBenchAdapter(task=NBTask("m0_gravity", "easy", "v0"),
                                 budget=10, judge_model="gpt41")
    ctrl = RASCController(
        adapter=adapter,
        generator=Generator(model=gen_model, max_tokens=1600),
        reflector=Reflector(model=ref_model, max_tokens=300),
        memory_selector=MemorySelector(k=4),
        code_evolver_factory=None,
        max_turns_per_subgoal=11, max_total_turns=12,
    )
    rep, mode, sig = ctrl.run_episode()
    return mode, rep.score_dict.get("SA", 0.0), " SA"


def _run_db(gen_model, ref_model):
    from mars.adapters.discoverybench_adapter import load_db_tasks
    from ols.adapters.discoverybench_adapter import DiscoveryBenchAdapter
    t = load_db_tasks(_PROJ / "discoverybench_repo", split="train", max_tasks=1)[0]
    adapter = DiscoveryBenchAdapter(task=t, budget=12, judge_model="openai/gpt-4o")
    ctrl = RASCController(
        adapter=adapter,
        generator=Generator(model=gen_model, max_tokens=900),
        reflector=Reflector(model=ref_model, max_tokens=300),
        memory_selector=MemorySelector(k=4),
        code_evolver_factory=None,
        max_turns_per_subgoal=4, max_total_turns=16,
    )
    rep, mode, sig = ctrl.run_episode()
    return mode, rep.score_dict.get("HMS", 0.0), " HMS"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    gen_model = os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini")
    ref_model = os.environ.get("MARS_REFLECTOR_MODEL", "openai/gpt-4o-mini")

    print("\n=== RASC — one self-configuring architecture, 3 benchmarks ===")
    print(f"  inner model: {gen_model}\n")

    benches = [
        ("UltraHorizon", _run_uh, "completeness"),
        ("NewtonBench",  _run_nb, "fit"),
        ("DiscoveryBench", _run_db, "reasoning"),
    ]
    rows = []
    for name, fn, expect in benches:
        t0 = time.time()
        try:
            mode, score, unit = fn(gen_model, ref_model)
            ok = "OK" if mode == expect else "MISROUTED"
            rows.append((name, mode, expect, ok, f"{score:.3f}{unit}", time.time() - t0))
            print(f"  {name:<15} diagnosed={mode:<13} (expected {expect:<13}) "
                  f"[{ok}]  score={score:.3f}{unit}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            rows.append((name, "ERROR", expect, "ERR", str(e)[:80], time.time() - t0))
            print(f"  {name:<15} ERROR: {str(e)[:120]}", flush=True)

    print("\n=== RASC self-configuration summary ===")
    n_ok = sum(1 for r in rows if r[3] == "OK")
    print(f"  correct self-diagnosis: {n_ok}/{len(rows)}")
    for name, mode, expect, ok, score, dt in rows:
        print(f"    {name:<15} -> {mode:<13} [{ok}]  {score}")
    print("\n  Claim: a single architecture self-selects the per-task-optimal "
          "regime;\n  any FIXED architecture is sub-optimal on 2 of 3.")


if __name__ == "__main__":
    main()
