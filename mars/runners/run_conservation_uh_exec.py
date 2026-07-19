"""UltraHorizon — EXECUTABLE ANALYSIS (offload the reasoning the model can't do to the
interpreter). The NB pattern applied to an agentic task.

The 87 ceiling is a hard ANALYTICAL limit: the weak model can't derive triploidy from breeding
data by prose reasoning, even when explicitly asked (V/G asymmetry fails for deep analytical
claims). The lever that worked on NewtonBench: don't have the model REASON about the data — have
it WRITE CODE that COMPUTES the answer, and let the interpreter (error-independent) run it. Here:
pool the structured experiment data from K rollouts (~K×3 crosses of offspring phenotypes),
have the model author `analyze(experiments)->dict` (cluster size_scores -> allele count -> ploidy;
dosage; dominance from intensities; lethal patterns), EXECUTE it, and write the final answer from
the computed results.

Tests whether executable analysis breaks the reasoning ceiling (>87) or whether the data budget /
model coding is the limit.

  python -m mars.runners.run_conservation_uh_exec --run_id cuhx_v1 --episodes 3 --k 3
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_PROJ = Path(__file__).resolve().parent.parent.parent
_UH = _PROJ / "ultrahorizon_repo"
for _p in (_PROJ, _UH):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=False)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.runners.run_uh_official import UHTask, run_one
from mars.runners.run_conservation_uh_evidence import judge_text

WEAK = "openai/gpt-4o-mini"


def _exec_analysis(code):
    """Compile analysis code with a FULL set of safe builtins (the NB fitter sandbox was too
    strict — blocked set/tuple/etc.). Allows numpy/math/statistics/collections."""
    import ast
    import math
    import statistics
    import collections
    import numpy as np
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    def _imp(name, *a, **k):
        if name.split(".")[0] in ("numpy", "math", "statistics", "collections", "itertools"):
            return __import__(name, *a, **k)
        raise ImportError(name)
    safe_names = ("abs min max pow sum len range float int str bool list dict set tuple frozenset "
                  "enumerate zip map filter any all sorted reversed round divmod isinstance "
                  "print").split()
    import builtins as _b
    bd = {n: getattr(_b, n) for n in safe_names if hasattr(_b, n)}
    bd["__import__"] = _imp
    ns = {"__builtins__": bd, "np": np, "math": math, "statistics": statistics,
          "collections": collections}
    try:
        exec(compile(code, "<analysis>", "exec"), ns)
    except Exception:
        return None
    return ns.get("analyze")


def collect_crosses(rep):
    out = []
    for h in rep.get("env_tool_history", []):
        if h.get("action") == "conduct_cross":
            r = h.get("result")
            if isinstance(r, dict) and r.get("success"):
                out.append({
                    "parent1_phenotype": r.get("parent1_phenotype"),
                    "parent2_phenotype": r.get("parent2_phenotype"),
                    "viable_offspring_count": r.get("viable_offspring_count"),
                    "lethal_offspring_count": r.get("lethal_offspring_count"),
                    "viability_rate": r.get("viability_rate"),
                    "offspring": [o.get("phenotype") for o in (r.get("offspring") or [])],
                })
    return out


def author_analysis(client, model, experiments):
    sample = json.dumps(experiments[:2], default=str)[:2500]
    prompt = (
        f"You are given `experiments`: a list of breeding crosses. Each has parent1_phenotype, "
        f"parent2_phenotype (dicts with body_size, size_score, body_color, color_intensity, "
        f"shell_shape, shell_hardness), viable_offspring_count, lethal_offspring_count, "
        f"viability_rate, and offspring (list of phenotype dicts). Example:\n{sample}\n\n"
        f"Write `def analyze(experiments):` using numpy that INFERS the genetics purely by COMPUTATION "
        f"and returns a dict. Compute, do not guess:\n"
        f"  - pool all offspring size_score values; cluster them (e.g. sort + gaps, or round) to find "
        f"the number of DISTINCT size levels and their values — a dosage trait with N distinct allele "
        f"values summed over C gene copies reveals ploidy C (e.g. 3 copies => triploid). Report the "
        f"distinct allele contributions and inferred number of gene copies per locus.\n"
        f"  - color: order body_color by mean color_intensity to get the dominance hierarchy.\n"
        f"  - shell: from lethal_offspring_count / viability_rate across crosses, infer which shell "
        f"combinations are lethal.\n"
        f"  - meiosis/viability: note if only one ploidy class is viable.\n"
        f"Return a dict with keys like 'ploidy', 'gene_copies_per_locus', 'size_allele_values', "
        f"'color_dominance', 'shell_lethal', 'notes'. Use only numpy + plain python.\n"
        f'Return JSON: {{"code":"def analyze(experiments):\\n    import numpy as np\\n    ..."}}'
    )
    raw = call_llm(client, model=model, system="Return only JSON.", user=prompt,
                   max_tokens=1400, temperature=0.3)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        return json.loads(t).get("code", "")
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: return json.loads(t[i:j+1]).get("code", "")
        except Exception: return ""


def run_analysis(code, experiments):
    fn = _exec_analysis(code)
    if fn is None:
        return None
    try:
        return fn(experiments)
    except Exception:
        return None


def write_answer(client, model, computed):
    prompt = (
        f"Computed genetic analysis results (from executed code on the breeding data):\n"
        f"{json.dumps(computed, default=str)[:3000]}\n\n"
        f"Write the FINAL answer for the genetics task, stating precisely: ploidy level (and gene "
        f"copies per locus), meiosis/segregation, viability constraints, body-size inheritance "
        f"(dosage + the allele values), color dominance hierarchy, and shell interactions incl. any "
        f"lethal combination. Use the COMPUTED numbers; be precise and complete."
    )
    return call_llm(client, model=model, system="Write a precise final report from the computed results.",
                    user=prompt, max_tokens=1100, temperature=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="cuhx_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--ref_model", default=WEAK)
    ap.add_argument("--env", default="bio")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--judge_model", default=WEAK)
    ap.add_argument("--ablation", default="MARS-full")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "conservation_uh_exec" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    results = {}; x_scores, best_scores = [], []
    print(f"=== UltraHorizon — EXECUTABLE ANALYSIS (offload reasoning to interpreter) — {args.model} ===\n")
    for ep in range(args.episodes):
        seed = 42 + ep
        task = UHTask(env_name=args.env, seed=seed, steps=args.steps, free=False,
                      difficulty=args.difficulty, judge_model=args.judge_model, action_budget=args.budget)
        experiments, scores = [], []
        for i in range(args.k):
            with contextlib.redirect_stdout(io.StringIO()):
                rep = run_one(task, ablation_name=args.ablation, gen_model=args.model, ref_model=args.ref_model)
            experiments += collect_crosses(rep)
            scores.append(float(rep.get("final_score", 0.0) or 0.0))

        computed, ran = None, False
        for _ in range(3):                       # small self-debug budget on the analysis code
            code = author_analysis(client, args.model, experiments)
            computed = run_analysis(code, experiments)
            if isinstance(computed, dict) and computed:
                ran = True; break
        if not ran:
            final = "Analysis code did not execute."
        else:
            final = write_answer(client, args.model, computed)
        x_score = judge_text(args.env, seed, args.difficulty, args.judge_model, final)

        best_s = max(scores)
        x_scores.append(x_score); best_scores.append(best_s)
        results[f"ep{ep}"] = {"exec_analysis": x_score, "best_rollout": best_s,
                              "rollout_scores": scores, "n_crosses": len(experiments), "code_ran": ran,
                              "computed": computed if ran else None}
        print(f"  ep{ep}: EXEC={x_score:.0f}  best_rollout={best_s:.0f}  rollouts={[int(s) for s in scores]}  "
              f"(crosses={len(experiments)}, code_ran={ran})")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = len(x_scores)
    mx = sum(x_scores) / n if n else 0.0
    mb = sum(best_scores) / n if n else 0.0
    print(f"\n=== RESULT (episodes={n}) ===")
    print(f"  EXECUTABLE-ANALYSIS = {mx:.1f}")
    print(f"  best-rollout ceiling = {mb:.1f}")
    verdict = "BREAKS the reasoning ceiling" if mx > mb + 1e-9 else ("matches" if abs(mx-mb) < 3 else "below")
    print(f"  → executable analysis {verdict}")
    out_path.write_text(json.dumps({"model": args.model, "episodes": n,
                                    "exec_analysis_mean": round(mx, 2), "best_rollout_mean": round(mb, 2),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
