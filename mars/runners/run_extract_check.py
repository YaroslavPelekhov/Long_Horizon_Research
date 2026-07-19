"""Extract-then-check: weak model TRANSCRIBES (reliable zone), code VERIFIES.

Hypothesis (V/G asymmetry of transcription vs computation): a weak model is far
more reliable at EXTRACTING the arithmetic tuple (a, op, b, result) from natural
prose than at COMPUTING/verifying it. So: model extracts -> code checks each step
via the manufactured invariant (recompute / mod) -> reliable error localization
on free-form arithmetic reasoning.

Natural, VARIED phrasing (not a regex-trivial template) forces real extraction.
One step's result is corrupted (known position). Measure extraction fidelity and
localization vs direct holistic critique.

  python -m mars.runners.run_extract_check --run_id ext --model openai/gpt-4o-mini
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

_N = ["marbles", "apples", "coins", "books", "stickers", "berries", "cards"]
_PHR = {
    "+": ["had {a} {n}, then got {b} more, for {r} in total",
          "started with {a} {n} and gained {b}, ending at {r}",
          "combined {a} {n} with another {b}, making {r}",
          "{a} {n} plus {b} more comes to {r}"],
    "-": ["had {a} {n}, gave away {b}, leaving {r}",
          "took {b} from {a} {n}, so {r} remained",
          "{a} {n} minus {b} leaves {r}",
          "after losing {b} of the {a} {n}, {r} were left"],
    "*": ["packed {a} boxes of {b} {n}, which is {r}",
          "{a} groups with {b} {n} each totals {r}",
          "{a} times {b} {n} equals {r}",
          "{b} {n} in each of {a} bags makes {r}"],
}
_FN = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}


def gen(steps, seed, corrupt_at):
    rng = random.Random(seed)
    tuples, lines = [], []
    for t in range(steps):
        op = rng.choice(["+", "-", "*"])
        a = rng.randint(6, 40)
        b = rng.randint(2, 9) if op == "*" else rng.randint(1, 30)
        r = _FN[op](a, b)
        shown = r + (rng.choice([-4, -3, -2, 2, 3, 4]) if t == corrupt_at else 0)
        tuples.append((a, op, b, shown, r))
        n = rng.choice(_N)
        lines.append("She " + rng.choice(_PHR[op]).format(a=a, b=b, r=shown, n=n) + ".")
    return tuples, " ".join(f"({t}) {ln}" for t, ln in enumerate(lines))


def extract(client, model, prose, n):
    raw = call_llm(client, model=model, system="Return only JSON.",
        user=(f"Each numbered sentence states an arithmetic fact. EXTRACT (do NOT verify) the "
              f"two input numbers, the operation (+,-,*), and the stated result, exactly as "
              f"written.\n\n{prose}\n\n"
              f'Return JSON: {{"steps":[{{"a":int,"op":"+/-/*","b":int,"result":int}}]}} '
              f"with exactly {n} entries in order."),
        max_tokens=600, temperature=0.0)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        return json.loads(t).get("steps", [])
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: return json.loads(t[i:j+1]).get("steps", [])
        except Exception: return []


def code_check_localize(steps):
    for t, s in enumerate(steps):
        try:
            a, op, b, res = int(s["a"]), s["op"], int(s["b"]), int(s["result"])
            if _FN[op](a, b) != res:
                return t
        except Exception:
            continue
    return None


def direct(client, model, prose, n):
    raw = call_llm(client, model=model, system="Answer with a single integer.",
        user=f"One numbered sentence has an arithmetic error:\n\n{prose}\n\nWhich number (0..{n-1})? Just the integer.",
        max_tokens=8, temperature=0.0)
    ds = "".join(c if c.isdigit() else " " for c in raw).split()
    return int(ds[0]) if ds else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="ext")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--n_test", type=int, default=15)
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
    extract_fid = 0; extract_total = 0
    rows = []
    print(f"=== Extract-then-check (model transcribes, code verifies) — {args.model} ===\n")
    for k in range(args.n_test):
        ca = random.Random(8000 + k).randrange(args.steps)
        tuples, prose = gen(args.steps, 8000 + k, ca)
        steps = extract(client, args.model, prose, args.steps)
        # extraction fidelity: did model transcribe each (a,op,b,result) correctly?
        for t in range(min(len(steps), len(tuples))):
            a, op, b, shown, _ = tuples[t]
            try:
                ok = (int(steps[t]["a"]) == a and steps[t]["op"] == op and
                      int(steps[t]["b"]) == b and int(steps[t]["result"]) == shown)
            except Exception:
                ok = False
            extract_fid += int(ok); extract_total += 1
        li = code_check_localize(steps)
        ld = direct(client, args.model, prose, args.steps)
        oi, od = (li == ca), (ld == ca)
        n_loc += oi; n_dir += od
        rows.append({"true": ca, "extract_check": li, "direct": ld, "ec_ok": oi, "dir_ok": od})
        print(f"  {k:2d} true=step{ca} | extract+code→{li} {'✓' if oi else '✗'}   direct→{ld} {'✓' if od else '✗'}")

    n = args.n_test
    fid = extract_fid / extract_total if extract_total else 0
    print(f"\n=== RESULT (free-form arithmetic reasoning) ===")
    print(f"  extraction fidelity (tuple transcribed correctly): {fid:.2f}")
    print(f"  EXTRACT+CODE localization: {n_loc}/{n} = {n_loc/n:.2f}")
    print(f"  DIRECT holistic          : {n_dir}/{n} = {n_dir/n:.2f}")
    print(f"  → {'HOLDS: transcribe+code beats introspection' if n_loc > n_dir else 'does NOT beat introspection'}")
    out_path.write_text(json.dumps({"model": args.model, "extraction_fidelity": fid,
                                    "ec_acc": n_loc/n, "direct_acc": n_dir/n,
                                    "beats": n_loc > n_dir, "rows": rows}, indent=2))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
