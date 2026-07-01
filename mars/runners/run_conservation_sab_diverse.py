"""ScienceAgentBench — DECORRELATION angle: agreement across DIVERSE independent METHODS.

Challenges the "external oracle required" conclusion. Consensus failed on id71 because K samples
from one model shared a systematic bug (correlated errors). Here we DECORRELATE: force each of
the K programs to use a DELIBERATELY DIFFERENT method/algorithm/library. If deliberately-diverse
implementations AGREE on the output, that agreement is strong evidence of correctness (Condorcet /
independent-derivation logic) WITHOUT any external ground truth; if they disagree, abstain honestly.

Same universal engine (consensus path) — only the generator forces method diversity.

  python -m mars.runners.run_conservation_sab_diverse --run_id csabd_v1 --max_tasks 6 --k 3
"""
from __future__ import annotations

import argparse
import json
import re
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
from mars.induction import conservation_discovery as cd
from mars.runners.run_conservation_sab import _extract_code, _output_signature
from mars.runners.run_sab_amplification import (load_ready_tasks, build_prompt, run_program,
                                                official_eval_result, _SAB as _SAB_ROOT)

WEAK = "openai/gpt-4o-mini"

_METHODS = [
    "Use straightforward pandas/numpy vectorized operations.",
    "Use a DIFFERENT approach: explicit Python loops / manual aggregation, NOT the same library "
    "calls as a typical solution — arrive at the result by an independent route.",
    "Use scikit-learn / scipy primitives where applicable, or a matrix/linear-algebra formulation — "
    "a method structurally different from the other attempts.",
    "Use a groupby/split-apply-combine or SQL-like (pandasql-free, pure pandas) decomposition, "
    "structurally distinct from the others.",
]


def build_adapter(client, model, t, k):
    base = build_prompt(t)

    def generate_candidates(ctx):
        out = []
        for i in range(k):
            method = _METHODS[i % len(_METHODS)]
            prompt = (base + f"\n\nMETHOD CONSTRAINT (for independence): {method}\n"
                      f"Reach the SAME correct result by THIS method; do not copy a standard template.")
            raw = call_llm(client, model=model, system="You write correct, self-contained Python.",
                           user=prompt, max_tokens=1800, temperature=0.6)
            code = _extract_code(raw)
            if not code.strip():
                continue
            ok, err = run_program(code, t["output_fname"], timeout=150)
            key = _output_signature(t) if ok else None
            out.append(cd.Candidate(answer=code, answer_key=key, meta={"ran": ok, "method": i}))
        return out

    def score(code):
        run_program(code, t["output_fname"], timeout=150)
        sr, _ = official_eval_result(t)
        return float(sr)

    return cd.DiscoveryAdapter(generate_candidates=generate_candidates,
                               transforms=lambda c: [], score=score, noise_floor=lambda c: 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="csabd_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--max_tasks", type=int, default=6)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "conservation_sab_diverse" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    tasks = load_ready_tasks(args.max_tasks)
    results = {}; sr_list = []; n_ans = n_abs = 0
    print(f"=== ScienceAgentBench — DECORRELATION (agreement across diverse methods) — {args.model} ===\n")
    for t in tasks:
        adapter = build_adapter(client, args.model, t, args.k)
        sel = cd.select(adapter, {})
        if sel.abstained:
            n_abs += 1
            results[str(t["id"])] = {"status": "ABSTAIN", "reason": sel.reason, "domain": t["domain"]}
            print(f"  id={t['id']:>3} {t['domain'][:22]:<22} ABSTAIN ({sel.reason})"); continue
        sr = adapter.score(sel.answer)
        sr_list.append(sr); n_ans += 1
        results[str(t["id"])] = {"status": "ANSWER", "SR": sr, "reason": sel.reason, "domain": t["domain"]}
        print(f"  id={t['id']:>3} {t['domain'][:22]:<22} SR={sr:.0f} ({sel.reason})")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_ans + n_abs
    prec = sum(sr_list) / len(sr_list) if sr_list else 0.0
    forced = sum(sr_list) / n if n else 0.0
    print(f"\n=== RESULT (N={n}) ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  SR on answered = {prec:.2f}  |  forced (abstain=0) = {forced:.2f}")
    print(f"  (diverse-method agreement; vs same-method consensus SR_answered 0.50)")
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "SR_answered": round(prec, 3), "SR_all": round(forced, 3),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
