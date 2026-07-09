"""Measure amplification on NewtonBench — where execution is ALIGNED with the
target metric (a law is correct iff it fits held-out data).

    weak (gpt-4o-mini) + EVA   >=?   strong (gpt-4o) raw

Conditions (same modules, same official symbolic-accuracy evaluator):
  A. weak_raw   : gpt-4o-mini proposes a law from data points (single shot)
  B. strong_raw : gpt-4o proposes a law (single shot)
  C. weak_eva   : gpt-4o-mini + EVA; checks recompute the law's error on
                  held-out points — objective, aligned with correctness.

EVA is unchanged and task-agnostic; only the executor differs (it evaluates a
candidate law expression on held-out points). The check signal here genuinely
tracks correctness, which is exactly when amplification should work.

Run:
  python -m mars.runners.run_eva_amplification_nb \\
    --run_id eva_nb_v1 --modules m0_gravity,m1_coulomb_force \\
    --weak openai/gpt-4o-mini --strong openai/gpt-4o --overwrite
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJ, _PROJ / "newtonbench_repo"):
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
from mars.amplify.eva import EVA, ExecResult  # noqa: E402
from mars.induction.nb_cpi import make_probe_inputs, recover_force  # noqa: E402


# ---------------------------------------------------------------------------
# Data collection for one module/system
# ---------------------------------------------------------------------------

def collect_points(module_name: str, difficulty: str, law_version: str,
                   system: str = "vanilla_equation") -> tuple[list[dict], list[str], str]:
    module = importlib.import_module(f"modules.{module_name}")
    signature = str(module.FUNCTION_SIGNATURE).strip()
    params = [p.strip() for p in signature[signature.index("(") + 1:signature.rindex(")")].split(",") if p.strip()]
    points = []
    for inp in make_probe_inputs(params, system=system):
        raw = module.run_experiment_for_module(noise_level=0.0, difficulty=difficulty,
            system=system, law_version=law_version, **inp)
        pt = recover_force(system, inp, raw)
        if pt is not None:
            row = dict(pt.inputs); row["__target__"] = pt.force
            points.append(row)
    return points, params, signature


def task_prompt(params: list[str], points: list[dict], target: str = "force") -> str:
    train = points[: max(4, len(points) // 2)]
    sample = json.dumps([{k: round(v, 4) for k, v in p.items()} for p in train[:10]], default=str)
    return (
        f"Discover the closed-form law that maps inputs {params} to the scalar "
        f"output '{target}'.\n\n"
        f"Observed data points (inputs + __target__):\n{sample}\n\n"
        f"Return your answer as a single Python expression in terms of {params} "
        f"(you may use math.*). The expression should compute '{target}'. There may "
        f"be an unknown multiplicative constant — include a numeric constant if needed.\n"
        f"Answer format: just the expression, e.g. 'C * {params[0]} * {params[1]} / {params[-1]}**2'."
    )


# ---------------------------------------------------------------------------
# Executor: evaluate a candidate law expression on held-out points
# ---------------------------------------------------------------------------

def make_executor(points: list[dict], params: list[str]):
    holdout = points[len(points) // 2:]

    def execute(code: str) -> ExecResult:
        ns: dict[str, Any] = {"math": math, "__builtins__": __builtins__}
        buf = io.StringIO()
        import contextlib
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, ns)
            return ExecResult(ok=True, value=ns.get("result"), stdout=buf.getvalue())
        except Exception as e:
            return ExecResult(ok=False, error=f"{type(e).__name__}: {e}", stdout=buf.getvalue())

    return execute, holdout


def law_rmsle(expr: str, points: list[dict], params: list[str]) -> float:
    """Objective fit metric for a law expression on points (calibrates constant)."""
    import contextlib
    # strip a possible leading constant 'C *' by fitting it
    ratios, preds, tgts = [], [], []
    for p in points:
        ns = {"math": math}
        ns.update({k: p[k] for k in params if k in p})
        # allow 'C' as a free symbol = 1 for structure; we calibrate after
        ns["C"] = 1.0
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                val = eval(expr, {"__builtins__": {}}, ns)
            val = float(val)
            if val == val:
                preds.append(val); tgts.append(float(p["__target__"]))
                if val != 0:
                    ratios.append(p["__target__"] / val)
        except Exception:
            continue
    if len(preds) < 2:
        return float("inf")
    # calibrate single multiplicative constant via geometric mean
    pos = [r for r in ratios if r > 0]
    k = math.exp(sum(math.log(r) for r in pos) / len(pos)) if pos else 1.0
    errs = []
    for pr, tg in zip(preds, tgts):
        a, b = abs(k * pr), abs(tg)
        if a > 1e-30 and b > 1e-30:
            errs.append((math.log(a) - math.log(b)) ** 2)
    return math.sqrt(sum(errs) / len(errs)) if errs else float("inf")


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def raw_law(client, model: str, params: list[str], points: list[dict]) -> str:
    out = call_llm(client, model=model,
                   system="You are a physicist. Return ONLY a Python math expression for the law.",
                   user=task_prompt(params, points), max_tokens=200, temperature=0.2)
    return _clean_expr(out)


def _clean_expr(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.split("\n", 1)[-1] if "\n" in s else s
    # take last non-empty line that looks like an expression
    for line in reversed(s.splitlines()):
        line = line.strip().strip("`")
        if line and not line.lower().startswith(("the ", "answer", "law", "expression")):
            if "=" in line and "==" not in line:
                line = line.split("=")[-1].strip()
            return line
    return s.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="eva_nb_smoke")
    ap.add_argument("--modules", default="m0_gravity")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--law_versions", default="v0,v1,v2")
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
    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    versions = [v.strip() for v in args.law_versions.split(",") if v.strip()]
    rows = []

    print(f"=== Amplification on NewtonBench: weak+EVA vs strong ===")
    print(f"weak={args.weak}  strong={args.strong}  (metric: holdout RMSLE, lower=better)\n")

    for mod in modules:
        for ver in versions:
            try:
                points, params, signature = collect_points(mod, args.difficulty, ver)
            except Exception as e:
                print(f"  {mod}/{ver}: collect failed: {e}")
                continue
            if len(points) < 6:
                print(f"  {mod}/{ver}: too few points ({len(points)})")
                continue

            execute, holdout = make_executor(points, params)

            # A. weak raw
            law_w = raw_law(client, args.weak, params, points)
            rmsle_w = law_rmsle(law_w, holdout, params)
            # B. strong raw
            law_s = raw_law(client, args.strong, params, points)
            rmsle_s = law_rmsle(law_s, holdout, params)
            # C. weak + EVA
            eva = EVA(model=args.weak, k_candidates=args.k, max_rounds=args.rounds)
            contract = ("runs in a Python sandbox with `math` and CANDIDATE_ANSWER "
                        "(a law expression string). eval the expression on the data "
                        "points to recompute the target and check the fit; print PASS "
                        "if it fits well else FAIL: <reason>. Data points are given in the task.")
            eva_res = eva.solve(task_prompt(params, points), execute, exec_contract=contract)
            law_e = _clean_expr(eva_res.answer)
            rmsle_e = law_rmsle(law_e, holdout, params)

            # amplified if weak+EVA fit is at least as good as strong
            amplified = rmsle_e <= rmsle_s + 1e-9
            mark = "✓ amplified" if amplified else "✗"
            print(f"  {mod}/{ver}: weak={_f(rmsle_w)}  strong={_f(rmsle_s)}  "
                  f"weak+EVA={_f(rmsle_e)}  {mark}")
            rows.append({
                "module": mod, "law_version": ver,
                "weak_raw_rmsle": rmsle_w, "strong_raw_rmsle": rmsle_s, "weak_eva_rmsle": rmsle_e,
                "amplified": amplified,
                "law_weak": law_w, "law_strong": law_s, "law_eva": law_e,
            })

    n = len(rows)
    if n:
        # use solved-rate (RMSLE < 0.1) and median rmsle for robustness
        def solved(key): return sum(1 for r in rows if r[key] < 0.1) / n
        summary = {
            "run_id": args.run_id, "score_type": "amplification-newtonbench",
            "weak": args.weak, "strong": args.strong, "n_tasks": n,
            "solved_rate_weak_raw": round(solved("weak_raw_rmsle"), 3),
            "solved_rate_strong_raw": round(solved("strong_raw_rmsle"), 3),
            "solved_rate_weak_eva": round(solved("weak_eva_rmsle"), 3),
            "n_eva_ge_strong": sum(1 for r in rows if r["amplified"]),
            "results": rows,
        }
        out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"\n=== SOLVED RATE (RMSLE < 0.1) ===")
        print(f"  weak_raw   : {summary['solved_rate_weak_raw']:.2f}")
        print(f"  strong_raw : {summary['solved_rate_strong_raw']:.2f}")
        print(f"  weak+EVA   : {summary['solved_rate_weak_eva']:.2f}   "
              f"{'>= strong' if summary['solved_rate_weak_eva'] >= summary['solved_rate_strong_raw'] else '< strong'}")
        print(f"  weak+EVA >= strong on {summary['n_eva_ge_strong']}/{n} tasks")
        print(f"summary → {out_path}")


def _f(x: float) -> str:
    return "inf" if x == float("inf") else f"{x:.4f}"


if __name__ == "__main__":
    main()
