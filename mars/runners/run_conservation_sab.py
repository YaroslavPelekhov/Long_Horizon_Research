"""ScienceAgentBench as a THIN ADAPTER over the SAME universal conservation engine.

Code-gen tasks are verifier-rich (execution + gold rubric), so here conservation = CONSENSUS
across independent implementations: the MODEL writes K programs, the shell runs each and reads
its numeric output signature, and the engine keeps the OUTCOME that is reproduced by a strict
majority of runs (idiosyncratic bugs don't agree); else abstain. Official eval scores the pick.

Same engine, same abstention logic — only the adapter (author programs / run / read output /
official score) is SAB-specific.

  python -m mars.runners.run_conservation_sab --run_id csab_v1 --max_tasks 8
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
_SAB = _PROJ / "scienceagentbench_repo"
for _p in (_PROJ,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=True)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.induction import conservation_discovery as cd
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


def _output_signature(t) -> str | None:
    """Read the program's output file and reduce it to a rounded numeric signature so that
    two implementations computing the same result get the SAME key. None if no output."""
    out_path = _SAB_ROOT / t["output_fname"]
    if not out_path.exists():
        return None
    try:
        data = out_path.read_bytes()
    except Exception:
        return None
    txt = data.decode("utf-8", errors="ignore")
    nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", txt)
    vals = []
    for s in nums[:200]:
        try:
            vals.append(round(float(s), 4))
        except Exception:
            pass
    if not vals:
        # non-numeric output (e.g. image) -> signature by size bucket
        return f"bytes~{len(data)//64}"
    return json.dumps(sorted(vals)[:50])


def build_adapter(client, model, t, k):
    prompt = build_prompt(t)

    def generate_candidates(ctx):
        out = []
        for i in range(k):
            raw = call_llm(client, model=model, system="You write correct, self-contained Python.",
                           user=prompt, max_tokens=1800, temperature=0.5 if i else 0.2)
            code = _extract_code(raw)
            if not code.strip():
                continue
            ok, err = run_program(code, t["output_fname"], timeout=150)
            key = _output_signature(t) if ok else None
            out.append(cd.Candidate(answer=code, answer_key=key,
                                    meta={"ran": ok, "err": err[:60] if err else ""}))
        return out

    def score(code):
        run_program(code, t["output_fname"], timeout=150)   # materialize output for official eval
        sr, _ = official_eval_result(t)
        return float(sr)

    return cd.DiscoveryAdapter(generate_candidates=generate_candidates,
                               transforms=lambda c: [], score=score, noise_floor=lambda c: 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="csab_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--max_tasks", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "conservation_sab" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    tasks = load_ready_tasks(args.max_tasks)
    results = {}; sr_list = []; n_ans = n_abs = 0
    print(f"=== ScienceAgentBench via universal conservation engine — {args.model} ===\n")
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
    print(f"  (consensus across K independent programs; same engine as NB/DB)")
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "SR_answered": round(prec, 3), "SR_all": round(forced, 3),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
