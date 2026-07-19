"""Cleanest amplification test — hypothesis generation on UH-Seq.

Here execution is PERFECTLY aligned with correctness: a rule hypothesis is a
program, and it is correct iff it exactly reproduces the observed transitions.
The task is also reachable by a weak model. This is exactly the regime where the
amplification thesis should hold:

    weak (gpt-4o-mini) + EVA   >=?   strong (gpt-4o) raw

Conditions (same 5 hidden rules, exact-match metric):
  A. weak_raw   : gpt-4o-mini writes a rule program (single shot)
  B. strong_raw : gpt-4o writes a rule program (single shot)
  C. weak_eva   : gpt-4o-mini + EVA; checks execute the candidate program on
                  observations and count exact matches — objective, aligned.

EVA is unchanged and task-agnostic. The executor just runs candidate programs
on observations. Metric: how many of the 5 hidden rules are recovered exactly.

Run:
  python -m mars.runners.run_eva_amplification_seq \\
    --run_id eva_seq_v1 --seeds 1,2,3 --difficulty easy \\
    --weak openai/gpt-4o-mini --strong openai/gpt-4o --overwrite
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJ, _PROJ / "ultrahorizon_repo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv
    _env = _PROJ / "autodiscovery" / ".env.local"
    if _env.exists():
        load_dotenv(_env, override=False)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client  # noqa: E402
from mars.amplify.eva import EVA, ExecResult  # noqa: E402


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# Observations for one rule slot
# ---------------------------------------------------------------------------

def collect_observations(seed: int, difficulty: str, steps: int = 5):
    import numpy as np
    from envs.common import Difficulty
    from envs.seq_env.env import SequenceExploreEnvironment

    random.seed(seed); np.random.seed(seed)
    SequenceExploreEnvironment.load_judge_config = lambda s: {}
    with contextlib.redirect_stdout(io.StringIO()):
        env = SequenceExploreEnvironment(difficulty=getattr(Difficulty, difficulty.upper()),
                                         required_steps=steps, free=False)
    defaults = [("ABCDE","EDCBA"),("EDCBA","ABCDE"),("AABCE","DDEAC"),
                ("BAEDC","CEBAD"),("AACDE","EABCD")]
    raw = []
    async def go():
        for m, v in defaults:
            with contextlib.redirect_stdout(io.StringIO()):
                r = await env.input_sequences(m, v)
            if r.get("success"):
                raw.append(r)
    _run_async(go())
    # Build per-slot observation lists
    slots = {s: [] for s in range(1, 6)}
    prev_main = ""
    for r in raw:
        outs = [str(t.get("sequence","")) for t in r.get("transformations",[])[1:6]]
        if len(outs) != 5:
            continue
        for slot in range(1, 6):
            cur = "" if slot == 1 else outs[slot-2]
            slots[slot].append({
                "current": cur, "target": outs[slot-1],
                "main": str(r.get("main_input","")), "vice": str(r.get("vice_input","")),
                "step_number": int(r.get("step_number",0)), "previous_main": prev_main,
            })
        prev_main = str(r.get("main_input",""))
    return slots


SEQ_TASK = """You are given observed transitions of a hidden string-transformation rule.
Each observation has: current (input string), main, vice, step_number, previous_main, and target (output).
Write a Python function that reproduces target from the inputs.

Signature EXACTLY:
def rule(current, context):
    # context has keys: main, vice, step_number, previous_main
    return <output string>

Allowed: ord, chr, len, range, zip, sorted, max, min, sum, str. Char position: ord(c)-ord('A'); char at i: chr(ord('A')+i%26).

OBSERVATIONS:
{obs}

Return ONLY the function code."""


def _obs_block(obs: list[dict], k: int = 5) -> str:
    return json.dumps([{kk: o[kk] for kk in ("current","main","vice","step_number","previous_main","target")}
                       for o in obs[:k]], ensure_ascii=False)


def make_executor(obs: list[dict]):
    def execute(code: str) -> ExecResult:
        ns: dict[str, Any] = {"__builtins__": __builtins__}
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, ns)
            return ExecResult(ok=True, value=ns.get("result"), stdout=buf.getvalue())
        except Exception as e:
            return ExecResult(ok=False, error=f"{type(e).__name__}: {e}", stdout=buf.getvalue())
    return execute


def eval_rule_code(code: str, obs: list[dict]) -> float:
    """Fraction of observations the rule reproduces exactly."""
    ns: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(_extract_code(code), ns)
    except Exception:
        return 0.0
    fn = ns.get("rule")
    if not callable(fn):
        return 0.0
    correct = 0
    for o in obs:
        try:
            out = fn(o["current"], {"main": o["main"], "vice": o["vice"],
                                    "step_number": o["step_number"], "previous_main": o["previous_main"]})
            correct += int(out == o["target"])
        except Exception:
            pass
    return correct / len(obs) if obs else 0.0


def _extract_code(s: str) -> str:
    s = s.strip()
    if "```" in s:
        import re
        m = re.search(r"```(?:python)?\n(.*?)```", s, re.DOTALL)
        if m:
            return m.group(1)
    return s


def raw_rule(client, model: str, obs: list[dict]) -> str:
    out = call_llm(client, model=model,
                   system="You write Python functions that reproduce hidden transformation rules.",
                   user=SEQ_TASK.format(obs=_obs_block(obs)), max_tokens=500, temperature=0.3)
    return _extract_code(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="eva_seq_smoke")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--weak", default="openai/gpt-4o-mini")
    ap.add_argument("--strong", default="openai/gpt-4o")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = _PROJ / "lmw" / "eva" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; pass --overwrite: {out_path}")

    client = make_openai_client()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []
    print(f"=== Amplification on UH-Seq (hypothesis generation) ===")
    print(f"weak={args.weak}  strong={args.strong}  metric: exact rules / 5\n")

    for seed in seeds:
        slots = collect_observations(seed, args.difficulty)
        w = s = e = 0
        for slot in range(1, 6):
            obs = slots[slot]
            if not obs:
                continue
            # A. weak raw
            cw = raw_rule(client, args.weak, obs); fw = eval_rule_code(cw, obs)
            # B. strong raw
            cs = raw_rule(client, args.strong, obs); fs = eval_rule_code(cs, obs)
            # C. weak + EVA
            eva = EVA(model=args.weak, k_candidates=args.k, max_rounds=args.rounds)
            contract = ("runs in a Python sandbox; CANDIDATE_ANSWER is a 'rule' function "
                        "as a code string. exec it, then call rule(current, context) on the "
                        "observation rows in the task and check it reproduces target exactly; "
                        "print PASS if all match else FAIL: <which row>.")
            res = eva.solve(SEQ_TASK.format(obs=_obs_block(obs)), make_executor(obs), exec_contract=contract)
            ce = res.answer; fe = eval_rule_code(ce, obs)
            w += int(fw == 1.0); s += int(fs == 1.0); e += int(fe == 1.0)
        print(f"  seed {seed}: weak_raw={w}/5  strong_raw={s}/5  weak+EVA={e}/5  "
              f"{'✓' if e >= s else '✗'}")
        rows.append({"seed": seed, "weak_raw": w, "strong_raw": s, "weak_eva": e})

    n = len(rows)
    mw = sum(r["weak_raw"] for r in rows)/n
    ms = sum(r["strong_raw"] for r in rows)/n
    me = sum(r["weak_eva"] for r in rows)/n
    summary = {
        "run_id": args.run_id, "score_type": "amplification-uhseq-hypothesis-gen",
        "weak": args.weak, "strong": args.strong, "n_seeds": n,
        "mean_weak_raw": round(mw,2), "mean_strong_raw": round(ms,2), "mean_weak_eva": round(me,2),
        "amplified": me >= ms, "results": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n=== MEAN (exact rules / 5) ===")
    print(f"  weak_raw   : {mw:.2f}")
    print(f"  strong_raw : {ms:.2f}")
    print(f"  weak+EVA   : {me:.2f}   {'>= strong ✓ THESIS HOLDS' if me >= ms else '< strong'}")
    print(f"summary → {out_path}")


if __name__ == "__main__":
    main()
