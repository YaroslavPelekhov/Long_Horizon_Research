"""CDA × DPSR: certified building blocks → typed-hole skeleton → measure coefficients.

The marriage of our two new methods:
  - CDA (accumulator) drills primitives and CERTIFIES them by execution.
  - DPSR composes the certified blocks into f(ctx) = sum(H[c_i]*prim_i(ctx))
    and INFERS the coefficients (subset selection) by measurement.

Claim: composites that DPSR could not crack from scratch (it had to guess the
additive structure) become solvable once the verified library supplies the
structure and DPSR only measures WHICH blocks and weights.

Run:
  python -m mars.runners.run_dpsr_cda --run_id dpsr_cda_v1 --overwrite
"""

from __future__ import annotations

import argparse
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
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=False)
except ImportError:
    pass

from mars.agents.base import make_openai_client
from mars.mdl.engine import Library, MDLEngine, make_llm_proposer
from mars.mdl.tasks_composite import PRIMITIVES, sample_contexts
from mars.induction.universal_cpi import CPIAdapter, Observation
from mars.induction.dpsr import DPSREngine
from mars.runners.run_accumulator import bank_primitive, drill_until_certified

WEAK = "openai/gpt-4o-mini"
DRILL_PRIMS = ["parity", "threshold", "modulo", "visit", "corner", "diag"]
TESTS = [
    ("pt",    ["parity", "threshold"]),
    ("ptm",   ["parity", "threshold", "modulo"]),
    ("cdv",   ["corner", "diag", "visit"]),
    ("ptmv",  ["parity", "threshold", "modulo", "visit"]),
]
N_OBS = 28


class CompositeCPIAdapter(CPIAdapter):
    name = "composite"

    def __init__(self, prims, seed):
        ctxs = sample_contexts(N_OBS, seed=seed)
        fns = [PRIMITIVES[p] for p in prims]
        self._obs = [Observation(inputs=c, target=sum(f(c) for f in fns), context={})
                     for c in ctxs]

    def interface_description(self) -> str:
        return ("Context dict with integer fields x,y (0-9), energy (0-30), steps (0-20), "
                "visit_count (1-6). Output: an integer score change. It may depend on parity "
                "(x+y)%2, an energy threshold, steps modulo, visit_count, corner, diagonal x==y, "
                "and is often a SUM of several such independent effects.")

    def signature_hint(self) -> str:
        return "def f(context):"

    def collect_observations(self):
        return self._obs

    def execute(self, program, obs):
        return program.fn(obs.inputs)

    def loss(self, prediction, obs):
        try:
            return 0.0 if int(prediction) == int(obs.target) else 1.0
        except Exception:
            return 1.0

    def render_report(self, winners, observations):
        return ""


def heldout_solved(programs, adapter, hold_obs) -> float:
    """Best program's exact rate on held-out (1.0 = fully cracked)."""
    best = 0.0
    for p in programs:
        ok = sum(1 for o in hold_obs if adapter.loss(adapter.execute(p, o), o) == 0.0)
        best = max(best, ok / len(hold_obs))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="dpsr_cda_v1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = _PROJ / "lmw" / "dpsr_cda" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; --overwrite: {out_path}")

    t0 = time.time()
    client = make_openai_client()
    drill_proposer = make_llm_proposer(WEAK, temperature=0.85)

    # ---- Phase 1: drill + certify primitives (CDA) ----
    print("=== CDA × DPSR ===\n--- drill + certify primitives ---")
    library = Library()
    for j, prim in enumerate(DRILL_PRIMS):
        dres, tries = drill_until_certified(drill_proposer, prim, args.seed + j * 7, 12, 4)
        banked = bank_primitive(library, prim, dres)
        print(f"  {prim:10} exact={dres.exact_rate:.2f} → {'BANKED' if banked else 'FAILED'} ({tries} tries)")
    blocks = [{"name": p.name, "code": p.code} for p in library._progs.values()]
    print(f"  library: {[b['name'] for b in blocks]}\n")

    summary = {"run_id": args.run_id, "weak": WEAK,
               "library": [b["name"] for b in blocks], "results": {}}

    # ---- Phase 2: DPSR on each composite, with vs without the library ----
    dpsr_cold = DPSREngine(model=WEAK, client=client, temperature=0.5, n_skeletons=4)
    dpsr_lib = DPSREngine(model=WEAK, client=client, temperature=0.5, n_skeletons=4,
                          building_blocks=blocks)

    print("--- DPSR on composites: cold (LLM skeletons) vs +library (CDA blocks) ---")
    for tid, prims in TESTS:
        ad = CompositeCPIAdapter(prims, seed=args.seed + 500)
        obs = ad.collect_observations()
        nh = max(1, int(len(obs) * 0.4))
        train, hold = obs[:-nh], obs[-nh:]

        cold = dpsr_cold.run(ad, train)
        cold_solve = heldout_solved(cold.programs, ad, hold) if cold.programs else 0.0

        lib = dpsr_lib.run(ad, train)
        lib_solve = heldout_solved(lib.programs, ad, hold) if lib.programs else 0.0
        # which coefficients DPSR turned on
        chosen = ""
        if lib.programs:
            line = [l for l in lib.programs[0].code.splitlines() if l.strip().startswith("H =")]
            chosen = line[0].strip()[:200] if line else ""

        summary["results"][tid] = {"prims": prims, "cold_solve": round(cold_solve, 3),
                                   "lib_solve": round(lib_solve, 3), "H": chosen}
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"  {tid:5} ({'+'.join(prims)}): cold={cold_solve:.2f}  +library={lib_solve:.2f}")
        if chosen:
            print(f"        {chosen}")

    # ---- Verdict ----
    n = len(TESTS)
    cold_mean = sum(r["cold_solve"] for r in summary["results"].values()) / n
    lib_mean = sum(r["lib_solve"] for r in summary["results"].values()) / n
    lib_solved = sum(1 for r in summary["results"].values() if r["lib_solve"] >= 0.99)
    print(f"\n=== VERDICT ===")
    print(f"  DPSR cold      : mean held-out solve = {cold_mean:.3f}")
    print(f"  DPSR + library : mean held-out solve = {lib_mean:.3f}  (fully solved {lib_solved}/{n})")
    print(f"  crossing helps : {'YES' if lib_mean > cold_mean + 1e-9 else 'no'}")
    summary["verdict"] = {"cold_mean": round(cold_mean, 3), "lib_mean": round(lib_mean, 3),
                          "lib_fully_solved": f"{lib_solved}/{n}",
                          "crossing_helps": lib_mean > cold_mean + 1e-9}
    summary["wall_time_s"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
