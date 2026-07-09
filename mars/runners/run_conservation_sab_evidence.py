"""ScienceAgentBench — EVIDENCE/PROPERTY-based selection (same principle as UH: verify against
the task's implied data-properties, not against peer agreement).

Consensus (peers) let "consensus-but-wrong" through (all K programs shared one bug). The fix,
matching UH evidence-selection: the MODEL authors executable PROPERTY CHECKS that any correct
output must satisfy (shape, columns, value ranges, counts, task constraints); each candidate
program is run and its output VERIFIED against the properties; we SELECT the program whose
output satisfies the most properties (V/G asymmetry: writing checks is easier than the analysis).
Abstain if nothing runs. Official eval scores the pick.

  python -m mars.runners.run_conservation_sab_evidence --run_id csabe_v1 --max_tasks 6 --k 3
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
from mars.induction.self_portfolio import _compile
from mars.runners.run_sab_amplification import (load_ready_tasks, build_prompt, run_program,
                                                official_eval_result, _SAB as _SAB_ROOT)

WEAK = "openai/gpt-4o-mini"


def _extract_code(raw: str) -> str:
    t = raw.strip()
    if "```" in t:
        m = re.search(r"```(?:python)?\n(.*?)```", t, re.DOTALL)
        if m:
            return m.group(1)
    return t


def author_checks(client, model, t, m=5):
    """Model authors executable property checks GROUNDED IN THE RAW INPUT DATA (external ground
    truth the model's systematic error can't corrupt) — re-derive a quantity from the inputs and
    compare to the program's output. Each: def check(path) -> bool."""
    raw = call_llm(client, model=model, system="Return only JSON.",
        user=(f"TASK: {t['task_inst']}\n\nDATASET (read from 'benchmark/datasets/...'):\n"
              f"{t['folder_tree'][:800]}\nPREVIEW:\n{t['preview'][:1000]}\n\n"
              f"The program writes its result to PATH ('{t['output_fname']}'). Write {m} STRICT "
              f"executable checks that verify the output AGAINST THE RAW INPUT DATA — each must "
              f"INDEPENDENTLY RE-DERIVE a quantity from the input files (read them with pandas) and "
              f"compare it to the output (e.g. output row count == #input rows after the described "
              f"filtering; predicted labels are a subset of input classes; an aggregate recomputed "
              f"from inputs matches the output within tolerance; ID columns align). Do NOT just "
              f"check the output in isolation — GROUND every check on the inputs. Each is "
              f"`def check(path):` returning True/False; use pandas as pd, numpy as np, os, json.\n"
              f'Return JSON: {{"checks":["def check(path):\\n    import pandas as pd, os\\n    ...\\n    return True", ...]}}'),
        max_tokens=1600, temperature=0.4)
    tx = raw.strip()
    if tx.startswith("```"):
        tx = "\n".join(tx.split("\n")[1:]).split("```")[0]
    try:
        arr = json.loads(tx).get("checks", [])
    except Exception:
        i, j = tx.find("{"), tx.rfind("}")
        try: arr = json.loads(tx[i:j+1]).get("checks", [])
        except Exception: arr = []
    return [c for c in arr if isinstance(c, str) and c.strip()][:m]


def run_check(code, path):
    fn = _compile(code, "check", {})
    if fn is None:
        return None
    try:
        return bool(fn(str(path)))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="csabe_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--max_tasks", type=int, default=6)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m_checks", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "conservation_sab_evidence" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    tasks = load_ready_tasks(args.max_tasks)
    results = {}; sr_list = []; n_ans = n_abs = 0
    print(f"=== ScienceAgentBench — EVIDENCE/property selection (data as verifier) — {args.model} ===\n")
    for t in tasks:
        prompt = build_prompt(t)
        checks = author_checks(client, args.model, t, args.m_checks)
        cands = []
        for i in range(args.k):
            raw = call_llm(client, model=args.model, system="You write correct, self-contained Python.",
                           user=prompt, max_tokens=1800, temperature=0.5 if i else 0.2)
            code = _extract_code(raw)
            if not code.strip():
                continue
            ok, err = run_program(code, t["output_fname"], timeout=150)
            support = 0.0
            if ok:
                outp = _SAB_ROOT / t["output_fname"]
                passed = [run_check(c, outp) for c in checks]
                passed = [p for p in passed if p is not None]
                support = (sum(1 for p in passed if p) / len(passed)) if passed else 0.0
            cands.append({"code": code, "ran": ok, "support": support})

        runnable = [c for c in cands if c["ran"]]
        if not runnable:
            n_abs += 1
            results[str(t["id"])] = {"status": "ABSTAIN", "reason": "no program ran", "domain": t["domain"]}
            print(f"  id={t['id']:>3} {t['domain'][:22]:<22} ABSTAIN (no program ran)"); continue
        chosen = max(runnable, key=lambda c: c["support"])
        # materialize chosen output and score officially
        run_program(chosen["code"], t["output_fname"], timeout=150)
        sr, _ = official_eval_result(t)
        sr_list.append(float(sr)); n_ans += 1
        results[str(t["id"])] = {"status": "ANSWER", "SR": sr, "support_of_selected": round(chosen["support"], 2),
                                 "supports": [round(c["support"], 2) for c in runnable], "domain": t["domain"]}
        print(f"  id={t['id']:>3} {t['domain'][:22]:<22} SR={sr:.0f} support={chosen['support']:.2f} "
              f"supports={[round(c['support'],2) for c in runnable]}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_ans + n_abs
    prec = sum(sr_list) / len(sr_list) if sr_list else 0.0
    forced = sum(sr_list) / n if n else 0.0
    print(f"\n=== RESULT (N={n}) ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  SR on answered = {prec:.2f}  |  forced (abstain=0) = {forced:.2f}")
    print(f"  (property-verified selection; vs consensus adapter earlier SR_answered 0.50)")
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "SR_answered": round(prec, 3), "SR_all": round(forced, 3),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
