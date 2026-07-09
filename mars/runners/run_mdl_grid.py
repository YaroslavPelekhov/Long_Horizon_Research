"""Compression amplification on UltraHorizon Grid (4th UH env → full coverage).

Same MDLEngine, thin GridLetterTask. Per letter A-E, discover the shortest
program mapping state (x,y,energy,steps,visit_count) -> Δscore. Execution is
aligned with correctness (exact integer match).

    weak (gpt-4o-mini) + MDL   >=?   strong (gpt-4o) raw

Run:
  python -m mars.runners.run_mdl_grid --run_id grid_v1 --seeds 1,2,3 \\
    --difficulty easy --weak openai/gpt-4o-mini --strong openai/gpt-4o \\
    --rounds 5 --k 12 --overwrite
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

_PROJ = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJ, _PROJ / "ultrahorizon_repo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv
    _env = _PROJ / "autodiscovery" / ".env.local"
    if _env.exists():
        load_dotenv(_env, override=True)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client  # noqa: E402
from mars.mdl.engine import MDLEngine, Library, make_llm_proposer, _compile  # noqa: E402
from mars.mdl.tasks_grid import GridObs, GridLetterTask  # noqa: E402


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _parse_pos(s: str):
    # "(x,y,letter)"
    inner = s.strip("()")
    parts = inner.split(",")
    return int(parts[0]), int(parts[1]), parts[2]


def collect(seed: int, difficulty: str, target_per_letter: int = 14):
    """Walk the grid, logging (state-at-effect -> Δscore) per letter, using
    resets for varied conditions. No knowledge of the hidden effects is used."""
    from envs.common import Difficulty
    from envs.grid_env.env import MysteryGridEnvironment

    random.seed(seed)
    MysteryGridEnvironment.load_judge_config = lambda s: {}
    with contextlib.redirect_stdout(io.StringIO()):
        env = MysteryGridEnvironment(difficulty=getattr(Difficulty, difficulty.upper()),
                                     required_steps=60, free=False)
    obs_by_letter: dict[str, list[GridObs]] = {L: [] for L in "ABCDE"}
    visited: dict[str, int] = {}
    dirs = ["up", "down", "left", "right"]

    async def walk():
        rounds = 0
        while min(len(v) for v in obs_by_letter.values()) < target_per_letter and rounds < 25:
            rounds += 1
            with contextlib.redirect_stdout(io.StringIO()):
                st = await env.get_current_state()
            # play a round until game_over
            for _ in range(40):
                # state before move
                cur = st.get("current_position") if isinstance(st, dict) else None
                try:
                    x0, y0, _l0 = _parse_pos(cur)
                except Exception:
                    break
                e0 = st.get("energy", 0); s0 = st.get("steps", 0); sc0 = st.get("score", 0)
                d = random.choice(dirs)
                with contextlib.redirect_stdout(io.StringIO()):
                    r = await env.move(d)
                if not r.get("success"):
                    # try other directions
                    moved = False
                    for d2 in dirs:
                        with contextlib.redirect_stdout(io.StringIO()):
                            r = await env.move(d2)
                        if r.get("success"):
                            moved = True; break
                    if not moved:
                        break
                x1, y1, letter = _parse_pos(r["position"])
                sc1 = r.get("score", sc0); s1 = r.get("steps", s0)
                delta = sc1 - sc0
                if letter in "ABCDE":
                    visited[letter] = visited.get(letter, 0) + 1
                    # energy at moment of effect = energy before - 1 (move cost)
                    obs_by_letter[letter].append(GridObs(
                        x=x1, y=y1, energy=e0 - 1, steps=s1,
                        visit_count=visited[letter], delta_score=delta))
                # refresh state for next iteration
                with contextlib.redirect_stdout(io.StringIO()):
                    st = await env.get_current_state()
                if not isinstance(st, dict) or st.get("game_over"):
                    break
            # reset for a fresh round (varied conditions)
            with contextlib.redirect_stdout(io.StringIO()):
                rr = await env.reset()
            if not rr.get("success"):
                break
        return env

    env = _run_async(walk())
    return obs_by_letter, env


_RAW = """Observed effects of stepping on a hidden-rule tile. Each observation gives the
state at the moment of stepping and the resulting score change.
def f(context):  # context: x, y, energy, steps, visit_count
    return <int score change>
Grid 10x10 (0..9). Effect may depend on position (parity x+y, corners/edges, x-y),
energy threshold, step modulo, or visit count.

OBSERVATIONS:
{obs}

Return ONLY the function code for f."""


def _obs_json(obs, k=8):
    return json.dumps([{"x": o.x, "y": o.y, "energy": o.energy, "steps": o.steps,
                        "visit_count": o.visit_count, "delta_score": o.delta_score}
                       for o in obs[:k]])


def _extract(s):
    s = s.strip()
    if "```" in s:
        import re
        m = re.search(r"```(?:python)?\n(.*?)```", s, re.DOTALL)
        if m: return m.group(1)
    return s


def _eval_code(code, obs):
    ok, err, fn = _compile(code, "f")
    if not ok or fn is None:
        return 0.0
    c = 0
    for o in obs:
        try:
            if int(fn({"x": o.x, "y": o.y, "energy": o.energy,
                       "steps": o.steps, "visit_count": o.visit_count})) == int(o.delta_score):
                c += 1
        except Exception:
            pass
    return c / len(obs) if obs else 0.0


def raw_solve(client, model, obs):
    code = _extract(call_llm(client, model=model,
        system="You write Python functions that reproduce hidden grid-tile effects.",
        user=_RAW.format(obs=_obs_json(obs)), max_tokens=400, temperature=0.3))
    return _eval_code(code, obs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="grid_smoke")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--weak", default="openai/gpt-4o-mini")
    ap.add_argument("--strong", default="openai/gpt-4o")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--k", type=int, default=12)
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
    print(f"=== Compression amplification on UH-Grid: weak+MDL vs strong ===")
    print(f"weak={args.weak} strong={args.strong} metric: exact letters / 5\n", flush=True)

    for seed in seeds:
        obs_by_letter, _env = collect(seed, args.difficulty)
        counts = {L: len(v) for L, v in obs_by_letter.items()}
        proposer = make_llm_proposer(args.weak)
        engine = MDLEngine(proposer, max_rounds=args.rounds, k_per_round=args.k, library=Library())
        w = s = e = 0
        per_letter = []
        for L in "ABCDE":
            obs = obs_by_letter[L]
            if len(obs) < 3:
                per_letter.append({"letter": L, "n_obs": len(obs), "skipped": True})
                continue
            fw = raw_solve(client, args.weak, obs)
            fs = raw_solve(client, args.strong, obs)
            res = engine.compress(GridLetterTask(L, obs))
            fe = res.exact_rate
            w += int(fw >= 0.999); s += int(fs >= 0.999); e += int(fe >= 0.999)
            per_letter.append({"letter": L, "n_obs": len(obs),
                               "weak_raw": fw, "strong_raw": fs, "weak_mdl": fe})
        print(f"  seed {seed}: weak_raw={w}/5 strong_raw={s}/5 weak+MDL={e}/5  "
              f"{'OK' if e>=s else '-'}  obs={counts}", flush=True)
        rows.append({"seed": seed, "weak_raw": w, "strong_raw": s, "weak_mdl": e,
                     "per_letter": per_letter})
        out_path.write_text(json.dumps({"partial": rows}, indent=2), encoding="utf-8")

    n = len(rows)
    mw = sum(r["weak_raw"] for r in rows)/n if n else 0
    ms = sum(r["strong_raw"] for r in rows)/n if n else 0
    me = sum(r["weak_mdl"] for r in rows)/n if n else 0
    summary = {
        "run_id": args.run_id, "score_type": "compression-amplification-uhgrid",
        "weak": args.weak, "strong": args.strong, "n_seeds": n,
        "mean_weak_raw": round(mw,2), "mean_strong_raw": round(ms,2), "mean_weak_mdl": round(me,2),
        "amplified": me >= ms, "results": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== MEAN (exact letters / 5) ===")
    print(f"  weak_raw {mw:.2f}  strong_raw {ms:.2f}  weak+MDL {me:.2f}   "
          f"{'>= strong THESIS HOLDS' if me>=ms else '< strong'}")
    print(f"summary -> {out_path}")


if __name__ == "__main__":
    main()
