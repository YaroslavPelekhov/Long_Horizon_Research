"""Certified-Depth Accumulator — the universal "нарешка" engine, falsified clean.

THE claim of the whole project, isolated to one variable:

    A frozen weak model cannot get smarter per step. But if it DRILLS primitives
    (solves them, certifies them by EXECUTION, banks them as reusable building
    blocks), its REACHABLE depth grows — and the SAME mini then solves composite
    problems that neither cold-mini NOR cold-4o can solve in one shot.

This is not orchestration of the same data. New CERTIFIED content (verified
building blocks) enters the substrate, so round N+1 reasons over a larger
vocabulary than round 1. Certification (execution) is what makes accumulation
real instead of self-delusion.

Conditions on a held-out test set of COMPOSITE rules (sum of 2-4 primitives):
  - mini cold      : MDL engine, EMPTY library         (textbook, asked directly)
  - 4o   cold      : MDL engine, EMPTY library         (the strong model, raw)
  - mini + library : MDL engine, library DRILLED by mini itself (нарешка)

Headline if true: mini+library  >=  4o cold   AND   >  mini cold.
Plus the compounding curve: test accuracy vs number of banked primitives.

Run:
  python -m mars.runners.run_accumulator --run_id acc_v1 --overwrite
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_PROJ = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJ, _PROJ / "ultrahorizon_repo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=True)
except ImportError:
    pass

from mars.mdl.engine import MDLEngine, Library, Program, make_llm_proposer
from mars.mdl.tasks_composite import CompositeRuleTask, drill_task, PRIMITIVES

WEAK = "openai/gpt-4o-mini"
STRONG = "openai/gpt-4o"

# Vocabulary the model drills (one certified building block each).
DRILL_PRIMS = ["parity", "threshold", "modulo", "visit", "corner", "diag"]

# Held-out composite tests: novel COMBINATIONS of the drilled vocabulary.
# Depth 2-4 — cold discovery of the full sum is hard (p ~ 0); composition is easy.
TEST_COMPOSITES = [
    ("t_pt",    ["parity", "threshold"]),
    ("t_ptm",   ["parity", "threshold", "modulo"]),
    ("t_cdv",   ["corner", "diag", "visit"]),
    ("t_ptmv",  ["parity", "threshold", "modulo", "visit"]),
]

N_OBS = 24          # observations per task (enough to pin exact thresholds)
DRILL_K, DRILL_ROUNDS = 12, 4
TEST_K, TEST_ROUNDS = 10, 4


# ---------------------------------------------------------------------------
# Bank a certified solver as a renamed, callable building block.
# ---------------------------------------------------------------------------

def _rename_entry(code: str, new_name: str, entry: str = "f") -> str | None:
    """Rename the top-level `def entry(...)` to `def new_name(...)` so several
    banked solvers can coexist in the library preamble and be composed."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    found = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry:
            node.name = new_name
            found = True
            break
    if not found:
        return None
    try:
        return ast.unparse(tree)
    except Exception:
        return None


def bank_primitive(library: Library, prim_name: str, result) -> bool:
    """Bank a drilled primitive iff it is CERTIFIED (reproduces all obs)."""
    if result.best is None or result.exact_rate < 0.99:
        return False                       # certification gate: no proof, no bank
    renamed = _rename_entry(result.best.code, f"prim_{prim_name}")
    if renamed is None:
        return False
    library.add(Program(name=f"prim_{prim_name}", code=renamed,
                        fn=result.best.fn, bits=result.best.bits,
                        score=result.best.score))
    return True


def drill_until_certified(proposer, prim_name: str, base_seed: int,
                          k: int, rounds: int, max_tries: int = 4):
    """Practice a primitive on fresh data until it certifies (exact==1.0).
    Nарешка = drill until you have actually got it. Returns (result, n_tries)."""
    best_res = None
    for attempt in range(max_tries):
        dt = drill_task(prim_name, N_OBS, seed=base_seed + attempt * 1000)
        eng = MDLEngine(proposer, max_rounds=rounds, k_per_round=k, library=Library())
        res = eng.compress(dt)
        if best_res is None or res.exact_rate > best_res.exact_rate:
            best_res = res
        if res.exact_rate >= 0.99:
            return res, attempt + 1
    return best_res, max_tries


# ---------------------------------------------------------------------------
# Evaluate a set of test tasks with a given proposer + library snapshot.
# ---------------------------------------------------------------------------

def _snapshot(library: Library) -> Library:
    """A read-only copy so the engine's own promotion of a SOLVED test task
    cannot leak that answer back into the shared library or across test tasks."""
    snap = Library()
    snap._progs = dict(library._progs)
    return snap


def eval_test(proposer, library: Library, seed_base: int,
              k: int, rounds: int) -> tuple[float, float, dict]:
    """Returns (mean fractional exact, solve-rate of fully-cracked tasks, per-task)."""
    per = {}
    total = 0.0
    solved = 0
    for i, (name, prims) in enumerate(TEST_COMPOSITES):
        task = CompositeRuleTask.build(name, prims, N_OBS, seed=seed_base + 100 + i)
        engine = MDLEngine(proposer, max_rounds=rounds, k_per_round=k,
                           library=_snapshot(library))   # primitives only, no test leakage
        res = engine.compress(task)
        per[name] = round(res.exact_rate, 3)
        total += res.exact_rate
        if res.exact_rate >= 0.99:
            solved += 1
    n = len(TEST_COMPOSITES)
    return total / n, solved / n, per


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="acc_v1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = _PROJ / "lmw" / "accumulator" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; --overwrite to rerun: {out_path}")

    t0 = time.time()
    mini_drill = make_llm_proposer(WEAK, temperature=0.85)   # high temp: diverse candidates
    mini = make_llm_proposer(WEAK, temperature=0.4)          # low temp: stable test eval
    strong = make_llm_proposer(STRONG, temperature=0.4)

    summary: dict = {
        "run_id": args.run_id, "weak": WEAK, "strong": STRONG,
        "drill_prims": DRILL_PRIMS, "test": [t[0] for t in TEST_COMPOSITES],
        "config": {"n_obs": N_OBS, "drill_k": DRILL_K, "drill_rounds": DRILL_ROUNDS,
                   "test_k": TEST_K, "test_rounds": TEST_ROUNDS},
    }

    print("=== Certified-Depth Accumulator (нарешка) ===\n")

    # ---- Baseline 1: mini cold (empty library) ----
    print("--- mini COLD (empty library, asked directly) ---")
    mini_cold_mean, mini_cold_solve, mini_cold_per = eval_test(mini, Library(), args.seed, TEST_K, TEST_ROUNDS)
    print(f"  mini cold: solved={mini_cold_solve:.2f} mean_exact={mini_cold_mean:.3f}  {mini_cold_per}\n")
    summary["mini_cold"] = {"solve": round(mini_cold_solve, 3), "mean": round(mini_cold_mean, 3), "per": mini_cold_per}
    out_path.write_text(json.dumps(summary, indent=2))

    # ---- Baseline 2: 4o cold (empty library) ----
    print("--- 4o COLD (empty library, raw strong model) ---")
    strong_cold_mean, strong_cold_solve, strong_cold_per = eval_test(strong, Library(), args.seed, TEST_K, TEST_ROUNDS)
    print(f"  4o cold: solved={strong_cold_solve:.2f} mean_exact={strong_cold_mean:.3f}  {strong_cold_per}\n")
    summary["strong_cold"] = {"solve": round(strong_cold_solve, 3), "mean": round(strong_cold_mean, 3), "per": strong_cold_per}
    out_path.write_text(json.dumps(summary, indent=2))

    # ---- нарешка: drill primitives with mini, bank certified, curve as it grows ----
    print("--- нарешка: mini drills primitives, banks certified blocks ---")
    library = Library()
    curve = [{"banked": 0, "lib": [], "test_solve": round(mini_cold_solve, 3),
              "test_mean": round(mini_cold_mean, 3)}]
    drilled_log = []
    for j, prim in enumerate(DRILL_PRIMS):
        dres, tries = drill_until_certified(mini_drill, prim, args.seed + j * 7,
                                            DRILL_K, DRILL_ROUNDS)
        banked = bank_primitive(library, prim, dres)
        drilled_log.append({"prim": prim, "exact": round(dres.exact_rate, 3),
                            "tries": tries, "banked": banked})
        status = f"BANKED (in {tries} tries)" if banked else "FAILED-not-certified"
        print(f"  drill {prim:10} exact={dres.exact_rate:.2f} → {status}")
        # re-evaluate test with the growing library (compounding curve)
        tmean, tsolve, tper = eval_test(mini, library, args.seed, TEST_K, TEST_ROUNDS)
        curve.append({"banked": len(library.names()), "lib": library.names(),
                      "test_solve": round(tsolve, 3), "test_mean": round(tmean, 3),
                      "test_per": tper})
        print(f"     → test solved={tsolve:.2f} mean_exact={tmean:.3f}  {tper}", flush=True)
        summary["drilled"] = drilled_log
        summary["curve"] = curve
        out_path.write_text(json.dumps(summary, indent=2))

    # ---- Multi-seed averaged verdict (single seed is noisy across baselines) ----
    VERDICT_SEEDS = [args.seed, args.seed + 31, args.seed + 71]
    def avg_solve(proposer, lib):
        ss = [eval_test(proposer, lib, s, TEST_K, TEST_ROUNDS)[1] for s in VERDICT_SEEDS]
        return sum(ss) / len(ss), ss
    print("\n--- averaged over seeds", VERDICT_SEEDS, "---", flush=True)
    mini_cold_solve, mc_list = avg_solve(mini, Library())
    strong_cold_solve, sc_list = avg_solve(strong, Library())
    mini_lib_solve, ml_list = avg_solve(mini, library)
    mini_lib_mean = curve[-1]["test_mean"]
    summary["verdict_seeds"] = {"seeds": VERDICT_SEEDS,
                                "mini_cold": mc_list, "strong_cold": sc_list, "mini_lib": ml_list}

    # ---- Verdict (solve-rate = did it actually crack the whole composite) ----
    print("\n=== VERDICT (solve-rate: fraction of composites fully cracked, avg of seeds) ===")
    print(f"  mini cold      : solved={mini_cold_solve:.3f}   {mc_list}")
    print(f"  4o   cold      : solved={strong_cold_solve:.3f}   {sc_list}")
    print(f"  mini + library : solved={mini_lib_solve:.3f}   {ml_list}   (нарешка)")
    beats_cold = mini_lib_solve > mini_cold_solve + 1e-9
    beats_strong = mini_lib_solve >= strong_cold_solve - 1e-9
    print(f"\n  mini+lib > mini cold   : {'YES' if beats_cold else 'no'}")
    print(f"  mini+lib >= 4o cold    : {'YES' if beats_strong else 'no'}")
    thesis = beats_cold and beats_strong
    print(f"\n  THESIS (accumulation creates capability, beats strong): "
          f"{'HOLDS' if thesis else 'does not hold'}")

    print("\n=== COMPOUNDING CURVE (solve-rate vs banked vocabulary) ===")
    for pt in curve:
        bar = "█" * int(pt["test_solve"] * 30)
        print(f"  banked={pt['banked']:2d}  solve={pt['test_solve']:.2f}  {bar}")

    summary["verdict"] = {
        "mini_cold_solve": round(mini_cold_solve, 3),
        "strong_cold_solve": round(strong_cold_solve, 3),
        "mini_lib_solve": round(mini_lib_solve, 3),
        "beats_cold": beats_cold, "beats_strong": beats_strong,
        "thesis_holds": thesis,
    }
    summary["wall_time_s"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
