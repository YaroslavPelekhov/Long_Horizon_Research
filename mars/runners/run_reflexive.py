"""Reflexive meta-loop prototype — the system does science on ITSELF.

Inner solver = DPSR block-composition over a LIBRARY of certified primitives
(deterministic, no LLM). It is given a deliberate BLIND SPOT: the library starts
missing some primitives, so it systematically fails on tasks that need them.

Outer (reflexive) loop:
  1. run inner solver over a TRAIN task distribution -> failures
  2. diagnose: which single certified addition removes the most failures
     (compress the failure manifold = minimal change, max failure reduction)
  3. repair: add that primitive to the library
  4. ACCEPT iff held-out solve rate improves (the fix generalizes); else rollback
  5. update a self-model of discovered failure-laws; repeat

Demonstrates: self-diagnosis of systematic blind spots + verified self-repair +
accumulating self-model — and distractor primitives are NOT adopted because they
don't reduce held-out failure. No human injects the fix; the loop finds it.

Run:
  python -m mars.runners.run_reflexive --run_id reflexive_v1 --overwrite
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from mars.induction.dpsr import DPSREngine, DPSRTrace
from mars.mdl.tasks_composite import PRIMITIVES, sample_contexts
from mars.induction.universal_cpi import CPIAdapter, Observation

# Certified primitive pool: 6 REAL primitives + 4 distractors (also certified,
# but irrelevant to the task distribution — the loop must NOT adopt them).
BLOCKS: dict[str, str] = {
    "prim_parity":    "def prim_parity(context):\n    return 2 if (context['x']+context['y'])%2==0 else 0\n",
    "prim_threshold": "def prim_threshold(context):\n    return 3 if context['energy']>=15 else 0\n",
    "prim_modulo":    "def prim_modulo(context):\n    return 1 if context['steps']%3==0 else 0\n",
    "prim_visit":     "def prim_visit(context):\n    return 4 if context['visit_count']<=2 else 0\n",
    "prim_corner":    "def prim_corner(context):\n    return 5 if (context['x'] in (0,9) and context['y'] in (0,9)) else 0\n",
    "prim_diag":      "def prim_diag(context):\n    return 2 if context['x']==context['y'] else 0\n",
    # distractors (certified but not used by any task):
    "prim_parity_x":  "def prim_parity_x(context):\n    return 2 if context['x']%2==0 else 0\n",
    "prim_thr10":     "def prim_thr10(context):\n    return 3 if context['energy']>=10 else 0\n",
    "prim_mod2":      "def prim_mod2(context):\n    return 1 if context['steps']%2==0 else 0\n",
    "prim_xgt5":      "def prim_xgt5(context):\n    return 2 if context['x']>5 else 0\n",
}
REAL = ["prim_parity", "prim_threshold", "prim_modulo", "prim_visit", "prim_corner", "prim_diag"]
PRIM_OF = {  # block name -> data-generating fn (distractors included so active
    "prim_parity": PRIMITIVES["parity"], "prim_threshold": PRIMITIVES["threshold"],
    "prim_modulo": PRIMITIVES["modulo"], "prim_visit": PRIMITIVES["visit"],
    "prim_corner": PRIMITIVES["corner"], "prim_diag": PRIMITIVES["diag"],
    # probing can synthesize tasks for them too — but held-out (real tasks never
    # use them) gives no gain, so the gate rejects them):
    "prim_parity_x": lambda c: 2 if c["x"] % 2 == 0 else 0,
    "prim_thr10": lambda c: 3 if c["energy"] >= 10 else 0,
    "prim_mod2": lambda c: 1 if c["steps"] % 2 == 0 else 0,
    "prim_xgt5": lambda c: 2 if c["x"] > 5 else 0,
}
START_LIB = ["prim_parity", "prim_threshold", "prim_modulo"]   # blind to visit/corner/diag

N_OBS = 24
_ENG = DPSREngine(model="x", client=None)   # block path is LLM-free


class _Adapter(CPIAdapter):
    name = "reflexive_composite"
    def __init__(self, obs): self._obs = obs
    def interface_description(self): return "context dict -> int (sum of effects)"
    def signature_hint(self): return "def f(context):"
    def collect_observations(self): return self._obs
    def execute(self, p, o): return p.fn(o.inputs)
    def loss(self, pred, o):
        try: return 0.0 if int(pred) == int(o.target) else 1.0
        except Exception: return 1.0
    def render_report(self, w, o): return ""


def _make_task(prims, seed):
    ctxs = sample_contexts(N_OBS, seed=seed)
    fns = [PRIM_OF[p] for p in prims]
    return [Observation(inputs=c, target=sum(f(c) for f in fns), context={}) for c in ctxs]


def inner_solved(lib, prims, seed) -> bool:
    return inner_solved_obs(lib, _make_task(prims, seed))


def inner_solved_obs(lib, obs) -> bool:
    """DPSR block solver with `lib`: fit coefficients on train obs, test held-out."""
    k = int(len(obs) * 0.6)
    train, hold = obs[:k], obs[k:]
    if not hold:
        return False
    blocks = [{"name": n, "code": BLOCKS[n]} for n in lib]
    eng = DPSREngine(model="x", client=None, building_blocks=blocks)
    ad = _Adapter(train)
    skel = eng._block_skeleton(ad)
    if skel is None:
        return False
    H, _ = eng._close_skeleton(skel, ad, train, {}, DPSRTrace())
    # evaluate fitted H on held-out
    code = _set_block_H(skel.code, H)
    fn = _compile_fn(code, skel.entry)
    if fn is None:
        return False
    return all(int(fn(o.inputs)) == int(o.target) for o in hold)


def _set_block_H(code, H):
    import ast
    tree = ast.parse(code)
    newv = ast.parse(repr(H), mode="eval").body
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "H" for t in node.targets):
            node.value = newv
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _compile_fn(code, entry):
    import math
    ns = {"__builtins__": {"abs": abs, "int": int, "float": float, "sum": sum,
          "min": min, "max": max, "len": len, "range": range, "tuple": tuple}, "math": math}
    try:
        exec(compile(code, "<rfx>", "exec"), ns)
        return ns.get(entry)
    except Exception:
        return None


def solve_rate(lib, tasks) -> float:
    return sum(1 for prims, sd in tasks if inner_solved(lib, prims, sd)) / len(tasks)


def _probe_obs_for(c, known, seed, n=20):
    """Design a probe task BALANCED on candidate c's activation (half active, half
    not), so even a rare feature (e.g. corner ~4%) is exercised enough to reveal —
    and verify — a blind spot."""
    rng = random.Random(seed)
    fc, fk = PRIM_OF[c], PRIM_OF[known]
    act, inact, tries = [], [], 0
    half = n // 2
    while (len(act) < half or len(inact) < half) and tries < 40000:
        tries += 1
        ctx = {"x": rng.randint(0, 9), "y": rng.randint(0, 9),
               "energy": rng.randint(0, 30), "steps": rng.randint(0, 20),
               "visit_count": rng.randint(1, 6)}
        o = Observation(inputs=ctx, target=fc(ctx) + fk(ctx), context={})
        if fc(ctx) != 0 and len(act) < half:
            act.append(o)
        elif fc(ctx) == 0 and len(inact) < half:
            inact.append(o)
    if len(act) < 4:
        return []
    obs = act + inact
    rng.shuffle(obs)
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="reflexive_v1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    out_dir = _PROJ / "lmw" / "reflexive" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; --overwrite: {out_path}")

    rng = random.Random(args.seed)
    # task distribution: composites of 2-3 REAL primitives
    def gen_tasks(n, off):
        out = []
        for i in range(n):
            k = rng.choice([2, 3])
            prims = rng.sample(REAL, k)
            out.append((prims, args.seed + off + i))
        return out
    train_tasks = gen_tasks(18, 0)
    heldout_tasks = gen_tasks(18, 1000)

    t0 = time.time()
    lib = list(START_LIB)
    self_model: list[dict] = []
    curve = [{"round": 0, "lib": list(lib),
              "train_solve": round(solve_rate(lib, train_tasks), 3),
              "heldout_solve": round(solve_rate(lib, heldout_tasks), 3)}]

    print("=== Reflexive meta-loop: the system fixes its own blind spots ===")
    print(f"start library: {lib}")
    print(f"  round 0: train_solve={curve[0]['train_solve']:.2f}  heldout_solve={curve[0]['heldout_solve']:.2f}\n")

    rnd = 0
    while True:
        rnd += 1
        cur_hold = solve_rate(lib, heldout_tasks)
        failed = [(p, s) for (p, s) in train_tasks if not inner_solved(lib, p, s)]
        if not failed:
            print("no remaining failures — converged.")
            break
        # diagnose: candidate addition that removes the most TRAIN failures
        cands = [b for b in BLOCKS if b not in lib]
        scored = []
        for c in cands:
            lib2 = lib + [c]
            fixed = sum(1 for (p, s) in failed if inner_solved(lib2, p, s))
            scored.append((fixed, c))
        scored.sort(reverse=True)
        best_fixed, best = scored[0]
        if best_fixed == 0:
            print("no candidate reduces failures — stop.")
            break
        # VERIFY on held-out before accepting
        hold_with = solve_rate(lib + [best], heldout_tasks)
        accept = hold_with > cur_hold + 1e-9
        tag = "ACCEPT" if accept else "REJECT (no held-out gain)"
        print(f"round {rnd}: diagnosed blind-spot → +{best} "
              f"(fixes {best_fixed}/{len(failed)} train failures); "
              f"held-out {cur_hold:.2f}→{hold_with:.2f}  [{tag}]")
        if accept:
            lib.append(best)
            self_model.append({"round": rnd, "added": best,
                               "train_failures_fixed": best_fixed,
                               "heldout": round(hold_with, 3),
                               "law": f"was systematically failing tasks needing {best}"})
            curve.append({"round": rnd, "lib": list(lib),
                          "train_solve": round(solve_rate(lib, train_tasks), 3),
                          "heldout_solve": round(hold_with, 3)})
        else:
            # do not adopt; record the rejected candidate and stop if it was the best
            self_model.append({"round": rnd, "rejected": best, "reason": "no held-out gain"})
            break

    # ---- ACTIVE SELF-PROBING: design experiments that expose gaps the passive
    # distribution never exercised — but ADOPT only what improves the TRUE held-out
    # distribution. This correctly rejects distractors; rare-but-real features sit
    # at the data's detection floor (indistinguishable from distractors without
    # more real data) and are conservatively NOT adopted.
    print("\n--- active self-probing for remaining blind spots ---")
    cur_hold = solve_rate(lib, heldout_tasks)
    probed_gap, adopted, rejected = [], [], []
    for c in [b for b in BLOCKS if b not in lib]:
        known = rng.choice(lib)
        probe = _probe_obs_for(c, known, args.seed + 7000)
        if not probe or inner_solved_obs(lib, probe):
            continue                                  # no exposable gap
        probed_gap.append(c)
        hold_with = solve_rate(lib + [c], heldout_tasks)   # verify on TRUE distribution
        if hold_with > cur_hold + 1e-9:
            lib.append(c); adopted.append(c); cur_hold = hold_with
            self_model.append({"probe": c, "added": True, "heldout": round(hold_with, 3),
                               "law": f"active probing revealed verifiable blind spot needing {c}"})
        else:
            rejected.append(c)
    print(f"  designed experiments exposed gaps for: {probed_gap}")
    print(f"  adopted (verified gain on true dist): {adopted or 'NONE'}")
    print(f"  NOT adopted (no verifiable gain — distractor or rare-below-floor): {rejected}")

    final_hold = solve_rate(lib, heldout_tasks)
    print(f"\n=== RESULT ===")
    print(f"  start:  lib={START_LIB}  held-out solve={curve[0]['heldout_solve']:.2f}")
    print(f"  final:  lib={lib}")
    print(f"          held-out solve={final_hold:.2f}")
    added = [b for b in lib if b not in START_LIB]
    distractors_added = [a for a in added if a not in REAL]
    print(f"  self-discovered missing primitives: {added}")
    print(f"  distractors wrongly adopted: {distractors_added or 'NONE'}")
    print(f"  self-model (failure-laws):")
    for s in self_model:
        if s.get('law'):
            extra = (f"(fixed {s['train_failures_fixed']} train, held-out {s['heldout']})"
                     if 'train_failures_fixed' in s else f"(held-out {s.get('heldout','-')})")
            print(f"    - {s['law']}  {extra}")

    summary = {
        "run_id": args.run_id, "start_lib": START_LIB, "final_lib": lib,
        "start_heldout_solve": curve[0]["heldout_solve"], "final_heldout_solve": round(final_hold, 3),
        "added": added, "distractors_added": distractors_added,
        "self_model": self_model, "curve": curve,
        "verdict": {
            "improved": final_hold > curve[0]["heldout_solve"] + 1e-9,
            "no_distractors": len(distractors_added) == 0,
        },
        "wall_time_s": round(time.time() - t0, 2),
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
