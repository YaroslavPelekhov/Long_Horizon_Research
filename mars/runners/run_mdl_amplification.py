"""Prove the amplification thesis with ONE principle (compression):

    weak (gpt-4o-mini) + MDL-engine   >=?   strong (gpt-4o) raw

on UH-Seq hidden-rule recovery (hypothesis generation, long-horizon: 5 linked
rules sharing one compression Library so building blocks transfer).

Conditions (same hidden rules, exact-recovery metric):
  A. weak_raw   : gpt-4o-mini writes a rule program (single shot)
  B. strong_raw : gpt-4o writes a rule program (single shot)
  C. weak_mdl   : gpt-4o-mini proposes; the bit-counter selects (MDLEngine).
                  One shared Library across the 5 rules (transfer).

The MDLEngine is the ONLY architecture. No task-specific logic, no rubric, no
answers. The same engine + a different CompressionTask runs any benchmark.

Run:
  python -m mars.runners.run_mdl_amplification \\
    --run_id mdl_seq_v1 --seeds 1,2,3,4,5 --difficulty easy \\
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
from mars.mdl.engine import MDLEngine, Library, make_llm_proposer, _compile  # noqa: E402
from mars.mdl.tasks_seq import SeqObs, SeqRuleTask  # noqa: E402


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def collect(seed: int, difficulty: str):
    import numpy as np
    from envs.common import Difficulty
    from envs.seq_env.env import SequenceExploreEnvironment

    random.seed(seed); np.random.seed(seed)
    SequenceExploreEnvironment.load_judge_config = lambda s: {}
    with contextlib.redirect_stdout(io.StringIO()):
        env = SequenceExploreEnvironment(difficulty=getattr(Difficulty, difficulty.upper()),
                                         required_steps=5, free=False)
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
    slots = {s: [] for s in range(1, 6)}
    prev = ""
    for r in raw:
        outs = [str(t.get("sequence","")) for t in r.get("transformations",[])[1:6]]
        if len(outs) != 5:
            continue
        for slot in range(1, 6):
            cur = "" if slot == 1 else outs[slot-2]
            slots[slot].append(SeqObs(current=cur, target=outs[slot-1],
                                      main=str(r.get("main_input","")), vice=str(r.get("vice_input","")),
                                      step_number=int(r.get("step_number",0)), previous_main=prev))
        prev = str(r.get("main_input",""))
    return slots


_RAW_PROMPT = """Observed transitions of a hidden string rule. Write a Python function reproducing target.
def f(current, context):  # context: main, vice, step_number, previous_main
    return <output string>
position of c: ord(c)-ord('A'); char at i: chr(ord('A')+i%26).

OBSERVATIONS:
{obs}

Return ONLY the function code for f."""


def _obs_json(obs: list[SeqObs], k: int = 5) -> str:
    return json.dumps([{"current": o.current, "main": o.main, "vice": o.vice,
                        "step_number": o.step_number, "previous_main": o.previous_main,
                        "target": o.target} for o in obs[:k]], ensure_ascii=False)


def _extract_code(s: str) -> str:
    s = s.strip()
    if "```" in s:
        import re
        m = re.search(r"```(?:python)?\n(.*?)```", s, re.DOTALL)
        if m:
            return m.group(1)
    return s


def _eval_code(code: str, obs: list[SeqObs]) -> float:
    ok, err, fn = _compile(code, "f")
    if not ok or fn is None:
        return 0.0
    c = 0
    for o in obs:
        try:
            if fn(o.current, {"main": o.main, "vice": o.vice,
                              "step_number": o.step_number, "previous_main": o.previous_main}) == o.target:
                c += 1
        except Exception:
            pass
    return c / len(obs) if obs else 0.0


def raw_solve(client, model: str, obs: list[SeqObs]) -> float:
    code = _extract_code(call_llm(client, model=model,
        system="You write Python functions that reproduce hidden transformation rules.",
        user=_RAW_PROMPT.format(obs=_obs_json(obs)), max_tokens=500, temperature=0.3))
    return _eval_code(code, obs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="mdl_seq_smoke")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--weak", default="openai/gpt-4o-mini")
    ap.add_argument("--strong", default="openai/gpt-4o")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = _PROJ / "lmw" / "mdl" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; pass --overwrite: {out_path}")

    client = make_openai_client()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []
    print(f"=== Compression amplification: weak+MDL vs strong (UH-Seq) ===")
    print(f"weak={args.weak}  strong={args.strong}  metric: exact rules / 5\n")

    for seed in seeds:
        slots = collect(seed, args.difficulty)
        # weak+MDL: ONE shared library across the 5 rules (long-horizon transfer)
        proposer = make_llm_proposer(args.weak)
        engine = MDLEngine(proposer, max_rounds=args.rounds, k_per_round=args.k, library=Library())
        w = s = e = 0
        per_rule = []
        for slot in range(1, 6):
            obs = slots[slot]
            if not obs:
                continue
            fw = raw_solve(client, args.weak, obs)
            fs = raw_solve(client, args.strong, obs)
            res = engine.compress(SeqRuleTask(slot, obs))
            fe = res.exact_rate
            w += int(fw == 1.0); s += int(fs == 1.0); e += int(fe >= 0.999)
            per_rule.append({"slot": slot, "weak_raw": fw, "strong_raw": fs,
                             "weak_mdl": fe, "ratio": res.compression_ratio,
                             "lib": engine.library.names()})
        print(f"  seed {seed}: weak_raw={w}/5  strong_raw={s}/5  weak+MDL={e}/5  "
              f"{'✓' if e >= s else '✗'}  (lib grew to {len(engine.library.names())})")
        rows.append({"seed": seed, "weak_raw": w, "strong_raw": s, "weak_mdl": e,
                     "per_rule": per_rule})

    n = len(rows)
    mw = sum(r["weak_raw"] for r in rows)/n
    ms = sum(r["strong_raw"] for r in rows)/n
    me = sum(r["weak_mdl"] for r in rows)/n
    summary = {
        "run_id": args.run_id, "score_type": "compression-amplification-uhseq",
        "principle": "shortest program reproducing observations; bits are the only judge",
        "weak": args.weak, "strong": args.strong, "n_seeds": n,
        "mean_weak_raw": round(mw,2), "mean_strong_raw": round(ms,2), "mean_weak_mdl": round(me,2),
        "amplified": me >= ms, "results": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n=== MEAN (exact rules / 5) ===")
    print(f"  weak_raw   : {mw:.2f}")
    print(f"  strong_raw : {ms:.2f}")
    print(f"  weak+MDL   : {me:.2f}   {'>= strong  THESIS HOLDS' if me >= ms else '< strong'}")
    print(f"summary -> {out_path}")


if __name__ == "__main__":
    main()
