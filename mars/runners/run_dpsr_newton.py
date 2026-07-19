"""DPSR universality test on a SECOND domain: numeric scientific-law induction.

If DPSR is universal, the SAME engine that closed coefficient holes on grid
composites should close EXPONENT/structure holes on physical laws — through the
same CPIAdapter contract, with no benchmark-specific code. The law's constant is
fit by the adapter's calibrate(); DPSR infers the exponents/variables by
measurement.

Compares DPSR (weak) vs raw one-shot weak law guessing, by held-out relative error.

Run:
  python -m mars.runners.run_dpsr_newton --run_id dpsr_nb_v1 --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_PROJ = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJ, _PROJ / "ultrahorizon_repo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=False)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.induction.cpi_adapters import NewtonAdapter
from mars.induction.dpsr import DPSREngine

WEAK = "openai/gpt-4o-mini"

# Hidden laws (the engine never sees these). (name, vars, fn, sampler)
LAWS = [
    ("gravity", ["m1", "m2", "r"],
     lambda d: 6.674e-11 * d["m1"] * d["m2"] / d["r"] ** 2,
     lambda rng: {"m1": rng.uniform(1, 1e3), "m2": rng.uniform(1, 1e3), "r": rng.uniform(1, 50)}),
    ("pendulum", ["L"],
     lambda d: 2 * math.pi * math.sqrt(d["L"] / 9.81),
     lambda rng: {"L": rng.uniform(0.1, 10)}),
    ("kinetic", ["m", "v"],
     lambda d: 0.5 * d["m"] * d["v"] ** 2,
     lambda rng: {"m": rng.uniform(1, 100), "v": rng.uniform(1, 50)}),
]


def make_points(law_fn, sampler, n, seed):
    rng = random.Random(seed)
    pts = []
    for _ in range(n):
        d = sampler(rng)
        d2 = dict(d)
        d2["__target__"] = law_fn(d)
        pts.append(d2)
    return pts


def rel_err(pred, tgt):
    try:
        return min(1.0, abs(float(pred) - float(tgt)) / (abs(float(tgt)) + 1e-12))
    except Exception:
        return 1.0


def raw_law(client, vars, examples):
    """One-shot weak baseline: ask for a closed-form law directly."""
    prompt = (
        f"Variables: {vars}. Data (inputs -> output):\n"
        + "\n".join(str(e) for e in examples[:8])
        + f"\n\nWrite ONE python function:\ndef law(inputs: dict) -> float\n"
        "Return ONLY the code."
    )
    try:
        raw = call_llm(client, model=WEAK, system="You induce physical laws.",
                       user=prompt, max_tokens=300, temperature=0.3)
        code = raw.strip()
        if code.startswith("```"):
            code = "\n".join(code.split("\n")[1:])
            code = code.split("```")[0]
        ns = {"math": math, "__builtins__": {"abs": abs, "float": float, "sum": sum,
              "min": min, "max": max, "pow": pow, "len": len, "range": range}}
        exec(compile(code, "<raw>", "exec"), ns)
        return ns.get("law")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="dpsr_nb_v1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    out_dir = _PROJ / "lmw" / "dpsr_newton" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; --overwrite: {out_path}")

    t0 = time.time()
    client = make_openai_client()
    dpsr = DPSREngine(model=WEAK, client=client, temperature=0.5, n_skeletons=4)
    summary = {"run_id": args.run_id, "weak": WEAK, "results": {}}

    print("=== DPSR universality: numeric scientific-law induction ===\n")
    for name, vars, law_fn, sampler in LAWS:
        pts = make_points(law_fn, sampler, 24, args.seed)
        # rename target into the adapter's expected key
        data = [{**{v: p[v] for v in vars}, "y": p["__target__"]} for p in pts]
        adapter = NewtonAdapter(data, var_names=vars, target_name="y")
        obs = adapter.collect_observations()
        nh = max(2, int(len(obs) * 0.4))
        train, hold = obs[:-nh], obs[-nh:]

        # DPSR
        res = dpsr.run(adapter, train)
        dpsr_err = 1.0
        for p in res.programs:
            errs = [rel_err(adapter.execute(p, o), o.target) for o in hold]
            dpsr_err = min(dpsr_err, sum(errs) / len(errs))

        # raw one-shot weak
        examples = [{**{v: o.inputs[v] for v in vars}, "y": o.target} for o in train]
        rawfn = raw_law(client, vars, examples)
        raw_err = 1.0
        if rawfn is not None:
            try:
                errs = [rel_err(rawfn(o.inputs), o.target) for o in hold]
                raw_err = sum(errs) / len(errs)
            except Exception:
                raw_err = 1.0

        summary["results"][name] = {"dpsr_relerr": round(dpsr_err, 4),
                                    "raw_relerr": round(raw_err, 4),
                                    "dpsr_better": dpsr_err < raw_err}
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"  {name:9}: DPSR rel-err={dpsr_err:.4f}   raw weak rel-err={raw_err:.4f}   "
              f"{'DPSR wins' if dpsr_err < raw_err else 'raw wins/tie'}")

    n = len(summary["results"])
    wins = sum(1 for r in summary["results"].values() if r["dpsr_better"])
    solved = sum(1 for r in summary["results"].values() if r["dpsr_relerr"] < 0.01)
    print(f"\n=== VERDICT ===")
    print(f"  DPSR beats raw weak on {wins}/{n} laws; DPSR near-exact (<1% err) on {solved}/{n}")
    summary["verdict"] = {"dpsr_wins": f"{wins}/{n}", "dpsr_near_exact": f"{solved}/{n}"}
    summary["wall_time_s"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
