"""Inference Invariants — "Noether for reasoning" (tail hypothesis #2), falsified clean.

Hypothesis: valid reasoning has CONSERVED QUANTITIES. The model can DISCOVER an
invariant f(state) that is constant along correct chains, VALIDATE it by
discrimination (constant on correct, breaks on corrupted), and use it as an
ORACLE-FREE per-step error localizer.

Falsifiable test: chains over K bins where total is conserved (transfer steps).
One step is corrupted (total jumps). Does a model-DISCOVERED + VALIDATED invariant
localize the corrupted step BETTER than asking the model directly "which step is
wrong?" If yes, a conserved quantity beats direct self-critique — the idea has legs.

  python -m mars.runners.run_inference_invariants --run_id inv_v1 --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=True)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client


# ---------------------------------------------------------------------------
# Domain: K bins, total conserved by transfer steps; optionally corrupt one step
# ---------------------------------------------------------------------------

def gen_chain(bins=4, steps=6, total=100, seed=0, corrupt_at=None):
    rng = random.Random(seed)
    state = [0] * bins
    for _ in range(total):
        state[rng.randrange(bins)] += 1
    chain = [list(state)]
    moves = []
    for t in range(steps):
        i, j = rng.sample(range(bins), 2)
        m = rng.randint(0, max(0, state[i]))
        state[i] -= m; state[j] += m
        s = list(state)
        if corrupt_at is not None and t == corrupt_at:        # break conservation here
            s[rng.randrange(bins)] += rng.choice([-7, -5, 5, 7])
        chain.append(s)
        moves.append((i, j, m))
        state = list(s)
    return chain  # list of states (state[0] is initial; step t produces chain[t+1])


def _chain_str(chain):
    return "\n".join(f"step {t}: {s}" for t, s in enumerate(chain))


# ---------------------------------------------------------------------------
# Noether: discover + validate an invariant, then localize
# ---------------------------------------------------------------------------

def discover_invariant(client, model, correct_chains):
    examples = "\n\n".join(_chain_str(c) for c in correct_chains[:3])
    raw = call_llm(client, model=model, system="Return only JSON.",
        user=(f"These state sequences are CORRECT (each 'state' is a list of numbers):\n\n{examples}\n\n"
              f"Find a QUANTITY that stays CONSTANT at every step of a correct sequence. "
              f"Write it as Python `def invariant(state):` returning a number.\n"
              f'Return JSON: {{"code":"def invariant(state):\\n    return ..."}}'),
        max_tokens=300, temperature=0.2)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        code = json.loads(t).get("code", "")
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: code = json.loads(t[i:j+1]).get("code", "")
        except Exception: code = ""
    return code


def _compile_inv(code):
    if not code:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    ns = {"__builtins__": {"sum": sum, "len": len, "abs": abs, "min": min, "max": max,
                           "range": range, "int": int, "float": float}}
    try:
        exec(compile(code, "<inv>", "exec"), ns)
    except Exception:
        return None
    return ns.get("invariant")


def validate_invariant(inv, correct, corrupt):
    """Constant on correct chains, breaks on corrupted ones -> discriminates."""
    def const_on(chain):
        try:
            vals = [inv(s) for s in chain]
            return all(abs(v - vals[0]) < 1e-9 for v in vals)
        except Exception:
            return None
    ok_correct = all(const_on(c) is True for c in correct)
    breaks_corrupt = all(const_on(c) is False for c in corrupt)
    return ok_correct and breaks_corrupt


def localize_with_invariant(inv, chain):
    """First step index where the invariant changes from its initial value."""
    try:
        base = inv(chain[0])
        for t in range(1, len(chain)):
            if abs(inv(chain[t]) - base) > 1e-9:
                return t - 1                      # the step (0-indexed) that produced chain[t]
    except Exception:
        return None
    return None


def localize_direct(client, model, chain):
    raw = call_llm(client, model=model, system="Answer with a single integer.",
        user=(f"This sequence of states should keep a hidden quantity constant, but ONE step "
              f"breaks it:\n\n{_chain_str(chain)}\n\n"
              f"Which step index (0..{len(chain)-2}) FIRST introduced the inconsistency? "
              f"Answer with just the integer."),
        max_tokens=8, temperature=0.0)
    try:
        return int("".join(ch for ch in raw if ch.isdigit() or ch == "-").split()[0]) \
            if any(c.isdigit() for c in raw) else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="inv_v1")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--n_test", type=int, default=12)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "inference_invariants" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()

    # 1. discover invariant from correct example chains
    correct = [gen_chain(steps=args.steps, seed=s) for s in range(3)]
    code = discover_invariant(client, args.model, correct)
    inv = _compile_inv(code)
    print(f"=== Inference Invariants (Noether for reasoning) — {args.model} ===\n")
    print(f"discovered invariant code:\n{code}\n")
    if inv is None:
        print("FALSIFIED at discovery: no compilable invariant proposed.")
        out_path.write_text(json.dumps({"status": "no_invariant", "code": code}, indent=2))
        return

    # 2. validate by discrimination (constant on correct, breaks on corrupt)
    val_correct = [gen_chain(steps=args.steps, seed=100 + s) for s in range(4)]
    val_corrupt = [gen_chain(steps=args.steps, seed=200 + s, corrupt_at=random.Random(s).randrange(args.steps))
                   for s in range(4)]
    valid = validate_invariant(inv, val_correct, val_corrupt)
    print(f"invariant validated (const on correct, breaks on corrupt): {valid}\n")

    # 3. localize injected errors on held-out corrupt chains: Noether vs direct critique
    n_ok_inv = n_ok_direct = 0
    rows = []
    for k in range(args.n_test):
        ca = random.Random(1000 + k).randrange(args.steps)
        chain = gen_chain(steps=args.steps, seed=1000 + k, corrupt_at=ca)
        li = localize_with_invariant(inv, chain) if valid else None
        ld = localize_direct(client, args.model, chain)
        ok_i = (li == ca); ok_d = (ld == ca)
        n_ok_inv += int(ok_i); n_ok_direct += int(ok_d)
        rows.append({"true": ca, "invariant": li, "direct": ld, "inv_ok": ok_i, "direct_ok": ok_d})
        print(f"  chain {k:2d}: true_err=step{ca}  invariant→{li} {'✓' if ok_i else '✗'}   "
              f"direct→{ld} {'✓' if ok_d else '✗'}")

    n = args.n_test
    print(f"\n=== RESULT ===")
    print(f"  localization accuracy — INVARIANT (Noether): {n_ok_inv}/{n} = {n_ok_inv/n:.2f}")
    print(f"  localization accuracy — DIRECT self-critique: {n_ok_direct}/{n} = {n_ok_direct/n:.2f}")
    verdict = ("HOLDS: conserved quantity beats direct self-critique"
               if n_ok_inv > n_ok_direct else "does not beat direct critique")
    print(f"  → {verdict}")
    out_path.write_text(json.dumps({
        "run_id": args.run_id, "model": args.model, "invariant_code": code,
        "validated": valid, "inv_acc": n_ok_inv / n, "direct_acc": n_ok_direct / n,
        "beats_direct": n_ok_inv > n_ok_direct, "rows": rows}, indent=2))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
