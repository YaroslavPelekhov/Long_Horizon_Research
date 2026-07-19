"""Inference Invariants — make-or-break generalization.

Two harder regimes than plain sum:
  --mode det : state = [a,b,c,d] (a 2x2), steps are determinant-PRESERVING row
               operations; the conserved quantity is a*d - b*c (NON-OBVIOUS).
               Tests whether the weak model can DISCOVER a non-trivial invariant.
  --mode nl  : narrative "transfer" chains rendered as PROSE with embedded counts;
               total is conserved; tests generalization to language.

Same falsifiable question: does a model-DISCOVERED + VALIDATED invariant localize
the corrupted step better than DIRECT self-critique?

  python -m mars.runners.run_inference_invariants_v2 --mode det --run_id inv_det
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
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=False)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client


# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------

def gen_det(steps, seed, corrupt_at=None):
    """2x2 [a,b,c,d]; det-preserving row ops; conserved = a*d - b*c."""
    rng = random.Random(seed)
    a, b, c, d = [rng.randint(1, 9) for _ in range(4)]
    chain = [[a, b, c, d]]
    for t in range(steps):
        k = rng.choice([-2, -1, 1, 2])
        if rng.random() < 0.5:
            c, d = c + k * a, d + k * b
        else:
            a, b = a + k * c, b + k * d
        s = [a, b, c, d]
        if corrupt_at == t:
            i = rng.randrange(4); s[i] += rng.choice([-3, -2, 2, 3]); a, b, c, d = s
        chain.append(list(s))
    return chain


def gen_nl(steps, seed, corrupt_at=None, total=60, bins=3):
    """Narrative transfer chain; total conserved; states embedded in prose."""
    rng = random.Random(seed)
    names = ["Tree A", "Tree B", "Tree C", "Tree D"][:bins]
    state = [0] * bins
    for _ in range(total):
        state[rng.randrange(bins)] += 1
    states = [list(state)]
    lines = [f"Start: " + ", ".join(f"{names[i]} has {state[i]} birds" for i in range(bins)) + "."]
    for t in range(steps):
        i, j = rng.sample(range(bins), 2)
        m = rng.randint(0, max(0, state[i]))
        state[i] -= m; state[j] += m
        s = list(state)
        if corrupt_at == t:
            s[rng.randrange(bins)] += rng.choice([-5, 5])
        states.append(list(s))
        lines.append(f"Step {t}: {m} birds fly from {names[i]} to {names[j]}. Now " +
                     ", ".join(f"{names[k]} has {s[k]} birds" for k in range(bins)) + ".")
        state = list(s)
    return states, "\n".join(lines)


def _chain_str(chain):
    return "\n".join(f"step {t}: {s}" for t, s in enumerate(chain))


# ---------------------------------------------------------------------------
# Discover / validate / localize
# ---------------------------------------------------------------------------

def discover_many(client, model, correct_chains, k=12):
    """Flood candidate invariants (incl. nonlinear/cross terms); validation selects."""
    examples = "\n\n".join(_chain_str(c) for c in correct_chains[:3])
    raw = call_llm(client, model=model, system="Return only JSON.",
        user=(f"CORRECT state sequences (state = list of numbers):\n\n{examples}\n\n"
              f"Propose {k} DIVERSE candidate conserved quantities — INCLUDE nonlinear / "
              f"cross terms (products like a*d, b*c, differences of products), not just linear "
              f"sums. Each: def invariant(state): return ...\n"
              f'Return JSON: {{"candidates":["def invariant(state):\\n    return ...", ...]}}'),
        max_tokens=1600, temperature=0.6)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        cands = json.loads(t).get("candidates", [])
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: cands = json.loads(t[i:j+1]).get("candidates", [])
        except Exception: cands = []
    return [c for c in cands if isinstance(c, str) and c.strip()]


def _compile(code):
    if not code:
        return None
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    ns = {"__builtins__": {"sum": sum, "len": len, "abs": abs, "min": min, "max": max,
                           "range": range, "int": int, "float": float, "pow": pow}}
    try:
        exec(compile(code, "<inv>", "exec"), ns)
    except Exception:
        return None
    return ns.get("invariant")


def validate(inv, correct, corrupt):
    def const(ch):
        try:
            vs = [inv(s) for s in ch]
            return all(abs(v - vs[0]) < 1e-9 for v in vs)
        except Exception:
            return None
    return all(const(c) is True for c in correct) and all(const(c) is False for c in corrupt)


def localize_inv(inv, chain):
    try:
        base = inv(chain[0])
        for t in range(1, len(chain)):
            if abs(inv(chain[t]) - base) > 1e-9:
                return t - 1
    except Exception:
        return None
    return None


def localize_direct(client, model, chain_repr, n_steps):
    raw = call_llm(client, model=model, system="Answer with a single integer.",
        user=(f"A hidden quantity should stay constant, but ONE step breaks it:\n\n{chain_repr}\n\n"
              f"Which step index (0..{n_steps-1}) FIRST introduced the inconsistency? Just the integer."),
        max_tokens=8, temperature=0.0)
    ds = "".join(ch if ch.isdigit() else " " for ch in raw).split()
    return int(ds[0]) if ds else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["det", "nl"], default="det")
    ap.add_argument("--run_id", default="inv_det")
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
    gen = (lambda **kw: gen_det(args.steps, **kw)) if args.mode == "det" else None

    # build correct/validation/test (det uses state lists; nl uses (states, prose))
    def states(seed, corrupt_at=None):
        if args.mode == "det":
            return gen_det(args.steps, seed, corrupt_at)
        return gen_nl(args.steps, seed, corrupt_at)[0]
    def prose(seed, corrupt_at):
        if args.mode == "det":
            return _chain_str(gen_det(args.steps, seed, corrupt_at))
        return gen_nl(args.steps, seed, corrupt_at)[1]

    correct = [states(s) for s in range(3)]
    cands = discover_many(client, args.model, correct)
    vcor = [states(100 + s) for s in range(4)]
    vcor_bad = [states(200 + s, random.Random(s).randrange(args.steps)) for s in range(4)]
    print(f"=== Inference Invariants v2 — mode={args.mode} — {args.model} ===\n")
    print(f"proposed {len(cands)} candidate invariants; validating each (discrimination)...")
    inv, code, valid = None, "", False
    for cc in cands:
        f = _compile(cc)
        if f is not None and validate(f, vcor, vcor_bad):
            inv, code, valid = f, cc, True
            break
    print(f"VALIDATED invariant found: {valid}")
    print(f"{code}\n")
    if inv is None:
        print("FALSIFIED: no candidate validated (discrimination on correct vs corrupt).")
        out_path.write_text(json.dumps({"mode": args.mode, "status": "no_validated_invariant",
                                        "candidates": cands}, indent=2)); return

    n_inv = n_dir = 0; rows = []
    for k in range(args.n_test):
        ca = random.Random(1000 + k).randrange(args.steps)
        st = states(1000 + k, ca)
        pr = prose(1000 + k, ca)
        li = localize_inv(inv, st) if valid else None
        ld = localize_direct(client, args.model, pr, args.steps)
        oi, od = (li == ca), (ld == ca)
        n_inv += oi; n_dir += od
        rows.append({"true": ca, "inv": li, "direct": ld, "inv_ok": oi, "dir_ok": od})
        print(f"  {k:2d}: true=step{ca}  inv→{li} {'✓' if oi else '✗'}   direct→{ld} {'✓' if od else '✗'}")

    n = args.n_test
    print(f"\n=== RESULT (mode={args.mode}) ===")
    print(f"  INVARIANT localization: {n_inv}/{n} = {n_inv/n:.2f}")
    print(f"  DIRECT   localization: {n_dir}/{n} = {n_dir/n:.2f}")
    print(f"  → {'HOLDS: invariant beats direct' if n_inv > n_dir else 'does NOT beat direct'}")
    out_path.write_text(json.dumps({"mode": args.mode, "model": args.model, "invariant_code": code,
                                    "validated": valid, "inv_acc": n_inv/n, "direct_acc": n_dir/n,
                                    "beats_direct": n_inv > n_dir, "rows": rows}, indent=2))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
