"""Manufactured invariants — impose a cheap algebraic redundancy where the weak
model is RELIABLE, turning structure-free reasoning into error-detectable form.

For arithmetic (+,-,*), residues mod m are a homomorphism: result % m must equal
(op(operands)) % m. This is an ARTIFICIAL invariant we IMPOSE (casting-out-nines).
Key: mod-9 via digit sums is single-digit arithmetic — a weak model is reliable
there even when it is unreliable on full computation. So we get reliable error
detection from an unreliable model (von Neumann: reliable-from-unreliable via
redundancy).

Compare error localization on GSM-style chains with one corrupted step:
  DIRECT     : mini holistic "which step is wrong?"
  RECOMPUTE  : mini rechecks each full step (the natural local check that tied before)
  MOD-MINI   : mini checks each step via casting-out-nines (mod 9 & mod 11)
  MOD-DET    : the manufactured invariant computed deterministically (upper bound)

  python -m mars.runners.run_manufactured_invariants --run_id mfi --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
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

_NOUNS = ["apples", "coins", "books", "marbles", "cookies", "tickets"]
_OPS = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("times", lambda a, b: a * b)]
_OPWORD = {"+": "plus", "-": "minus", "times": "times"}


def gen(steps, seed, corrupt_at):
    rng = random.Random(seed)
    rows = []
    for t in range(steps):
        op, fn = rng.choice(_OPS)
        a = rng.randint(6, 40)
        b = rng.randint(2, 9) if op == "times" else rng.randint(1, 30)
        res = fn(a, b)
        shown = res + (rng.choice([-4, -3, -2, 2, 3, 4]) if t == corrupt_at else 0)
        rows.append((a, op, b, shown, res))
    return rows


def prose(rows):
    out = []
    for t, (a, op, b, shown, _) in enumerate(rows):
        out.append(f"Step {t}: {a} {_OPWORD[op]} {b} gives {shown} {_NOUNS[t % len(_NOUNS)]}.")
    return "\n".join(out)


def _opval(a, op, b):
    return {"+": a + b, "-": a - b, "times": a * b}[op]


def mod_det_localize(rows):
    """Manufactured invariant, deterministic: first step failing residue mod 9 or 11."""
    for t, (a, op, b, shown, _) in enumerate(rows):
        true = _opval(a, op, b)
        if (shown - true) % 9 != 0 or (shown - true) % 11 != 0:
            return t
    return None


def direct_localize(client, model, p, n):
    raw = call_llm(client, model=model, system="Answer with a single integer.",
        user=(f"One step has an arithmetic mistake:\n\n{p}\n\nWhich step index (0..{n-1}) "
              f"is wrong? Just the integer."), max_tokens=8, temperature=0.0)
    ds = "".join(c if c.isdigit() else " " for c in raw).split()
    return int(ds[0]) if ds else None


def recompute_localize(client, model, rows):
    for t, (a, op, b, shown, _) in enumerate(rows):
        raw = call_llm(client, model=model, system="Answer yes or no.",
            user=f"Is this correct: {a} {_OPWORD[op]} {b} = {shown}? Answer yes or no.",
            max_tokens=4, temperature=0.0)
        if "no" in raw.strip().lower()[:4]:
            return t
    return None


def mod_mini_localize(client, model, rows):
    """Mini applies casting-out-nines (single-digit) per step — its reliable zone."""
    for t, (a, op, b, shown, _) in enumerate(rows):
        raw = call_llm(client, model=model, system="Answer yes or no.",
            user=(f"Use casting out nines (repeated digit sums, mod 9) to CHECK this without full "
                  f"computation: does  {a} {_OPWORD[op]} {b} = {shown}  pass the mod-9 digit-sum "
                  f"check? (digit-sum of {shown} must equal digit-sum of {a} {_OPWORD[op]} {b}.) "
                  f"Answer yes or no."),
            max_tokens=4, temperature=0.0)
        if "no" in raw.strip().lower()[:4]:
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="mfi")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--n_test", type=int, default=12)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "inference_invariants" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    acc = {"mod_det": 0, "mod_mini": 0, "recompute": 0, "direct": 0}
    rows_log = []
    print(f"=== Manufactured invariants (mod-9/11 redundancy) — {args.model} ===\n")
    for k in range(args.n_test):
        ca = random.Random(7000 + k).randrange(args.steps)
        rws = gen(args.steps, 7000 + k, ca)
        p = prose(rws)
        r = {"true": ca,
             "mod_det": mod_det_localize(rws),
             "mod_mini": mod_mini_localize(client, args.model, rws),
             "recompute": recompute_localize(client, args.model, rws),
             "direct": direct_localize(client, args.model, p, args.steps)}
        for key in acc:
            acc[key] += int(r[key] == ca)
        rows_log.append(r)
        print(f"  {k:2d} true=step{ca} | mod-det→{r['mod_det']} mod-mini→{r['mod_mini']} "
              f"recompute→{r['recompute']} direct→{r['direct']}")

    n = args.n_test
    print(f"\n=== RESULT (heterogeneous arithmetic, manufactured invariant) ===")
    for key in ["mod_det", "mod_mini", "recompute", "direct"]:
        print(f"  {key:10}: {acc[key]}/{n} = {acc[key]/n:.2f}")
    print(f"\n  mod-det = upper bound of the artificial invariant")
    print(f"  mod-mini vs recompute/direct = does the weak model exploit the cheap invariant?")
    out_path.write_text(json.dumps({"model": args.model, "n": n,
                                    "acc": {k: acc[k]/n for k in acc}, "rows": rows_log}, indent=2))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
