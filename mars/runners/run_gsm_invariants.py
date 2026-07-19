"""Decisive test: inference-invariant localization on HETEROGENEOUS arithmetic
reasoning (GSM8K-style), where there is NO global conserved quantity.

Each chain is a multi-step word-problem solution with heterogeneous ops
(+,-,*,%,//). One step's RESULT is corrupted (operands kept correct, so the error
is purely LOCAL at that step). Ground-truth error position is known.

Compare error LOCALIZATION:
  DIRECT  : one holistic call "which step is the first wrong one?"
  LOCAL   : the residual invariant when no global one exists — per-step check
            "does this step's result follow from its operands?" -> first failure.

If LOCAL >> DIRECT on heterogeneous reasoning, the inference-invariant principle
generalizes beyond conservation domains (honest cousin: process supervision).

  python -m mars.runners.run_gsm_invariants --run_id gsm_inv --model openai/gpt-4o-mini
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
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=False)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client

_NOUNS = ["apples", "coins", "books", "marbles", "cookies", "tickets", "pencils", "stamps"]
_OPS = [("+", lambda a, b: a + b, "{a} plus {b}"),
        ("-", lambda a, b: a - b, "{a} minus {b}"),
        ("*", lambda a, b: a * b, "{a} times {b}"),
        ("%", lambda a, b: a % b, "the remainder of {a} divided by {b}"),
        ("//", lambda a, b: a // b, "{a} divided by {b} (whole part)")]


def gen_problem(steps, seed, corrupt_at):
    rng = random.Random(seed)
    lines = []
    err_pos = corrupt_at
    for t in range(steps):
        noun = rng.choice(_NOUNS)
        op_sym, fn, phrase = rng.choice(_OPS)
        a = rng.randint(6, 40)
        b = rng.randint(2, 9) if op_sym in ("*", "%", "//") else rng.randint(1, 30)
        res = fn(a, b)
        shown = res
        if t == corrupt_at:
            shown = res + rng.choice([-4, -3, -2, 2, 3, 4])    # local error in the RESULT only
        lines.append(f"Step {t}: {phrase.format(a=a, b=b)} gives {shown} {noun}.")
    return "\n".join(lines)


def direct_localize(client, model, prose, n):
    raw = call_llm(client, model=model, system="Answer with a single integer.",
        user=(f"One step below has an arithmetic mistake (its result does not follow from its "
              f"numbers):\n\n{prose}\n\nWhich step index (0..{n-1}) is wrong? Just the integer."),
        max_tokens=8, temperature=0.0)
    ds = "".join(c if c.isdigit() else " " for c in raw).split()
    return int(ds[0]) if ds else None


def local_invariant_localize(client, model, prose, n):
    """Per-step local consistency check (the residual invariant); first failure."""
    steps = prose.split("\n")
    for t, line in enumerate(steps):
        raw = call_llm(client, model=model, system="Answer yes or no.",
            user=(f"Is the arithmetic in this single statement correct (does the stated result "
                  f"follow from the numbers and operation)?\n\n{line}\n\nAnswer yes or no."),
            max_tokens=4, temperature=0.0)
        if "no" in raw.strip().lower()[:4]:
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="gsm_inv")
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
    n_loc = n_dir = 0
    rows = []
    print(f"=== GSM-style heterogeneous reasoning — invariant vs direct — {args.model} ===\n")
    for k in range(args.n_test):
        ca = random.Random(5000 + k).randrange(args.steps)
        prose = gen_problem(args.steps, 5000 + k, ca)
        ld = direct_localize(client, args.model, prose, args.steps)
        li = local_invariant_localize(client, args.model, prose, args.steps)
        oi, od = (li == ca), (ld == ca)
        n_loc += oi; n_dir += od
        rows.append({"true": ca, "local_inv": li, "direct": ld, "inv_ok": oi, "dir_ok": od})
        print(f"  {k:2d}: true=step{ca}  local-inv→{li} {'✓' if oi else '✗'}   direct→{ld} {'✓' if od else '✗'}")

    n = args.n_test
    print(f"\n=== RESULT (heterogeneous, no global invariant) ===")
    print(f"  LOCAL-INVARIANT localization: {n_loc}/{n} = {n_loc/n:.2f}")
    print(f"  DIRECT (holistic) localization: {n_dir}/{n} = {n_dir/n:.2f}")
    print(f"  → {'HOLDS: per-step invariant beats holistic introspection' if n_loc > n_dir else 'does NOT beat holistic'}")
    out_path.write_text(json.dumps({"model": args.model, "local_acc": n_loc/n, "direct_acc": n_dir/n,
                                    "beats_direct": n_loc > n_dir, "rows": rows}, indent=2))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
