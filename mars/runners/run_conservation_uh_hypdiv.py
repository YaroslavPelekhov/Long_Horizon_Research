"""UltraHorizon — HYPOTHESIS DIVERSIFICATION to break the generation support ceiling.

Key finding motivating this: temperature-diverse rollouts share the SAME conceptual blind spot
(0/4 ever recognize triploidy), so selection/union caps at the best single rollout (87). The
architecture's ceiling is the SUPPORT of the generator's distribution — insights never sampled
can't be selected. To break it WITHOUT injecting the answer: the model itself enumerates GENERAL
hypothesis classes for the domain (ploidy level, inheritance mode, lethal interactions, ...), then
is FORCED to analyze the pooled evidence UNDER EACH — turning spontaneous generation (hard) into
hypothesis-conditioned validation (easier, V/G asymmetry). Findings the data support are assembled.

Tests whether forced hypothesis diversification exceeds the best single rollout, or confirms a hard
model limit (model can't even validate the missing insight when prompted).

  python -m mars.runners.run_conservation_uh_hypdiv --run_id cuhh_v1 --episodes 2 --k 3
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
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=True)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.runners.run_uh_official import UHTask, run_one
from mars.runners.run_conservation_uh_evidence import extract_evidence, judge_text

WEAK = "openai/gpt-4o-mini"


def enumerate_hypotheses(client, model, task_hint, evidence, m=5):
    """Model authors GENERAL hypothesis CLASSES (not the answer) that a rigorous analysis must
    rule in/out for this KIND of system. Diversifies the hypothesis space it would skip."""
    ev = "\n".join(evidence[:40])[:4500]
    raw = call_llm(client, model=model, system="Return only JSON.",
        user=(f"Investigation domain: {task_hint}\n\nPooled experimental observations:\n{ev}\n\n"
              f"List {m} DISTINCT GENERAL hypothesis CLASSES that a rigorous analysis of THIS KIND of "
              f"system must explicitly test and rule in or out — the standard alternatives, not the "
              f"'default' assumption. Cover the axes a careful scientist checks (e.g. for a genetics "
              f"system: ploidy level incl. non-diploid; segregation/meiosis mode; dominance structure "
              f"incl. non-simple; viability/lethal interactions; dosage vs discrete). State each as a "
              f"QUESTION to resolve from data, WITHOUT assuming the answer.\n"
              f'Return JSON: {{"hypotheses":["<question 1>", "<question 2>", ...]}}'),
        max_tokens=600, temperature=0.5)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        return [str(h) for h in json.loads(t).get("hypotheses", []) if str(h).strip()][:m]
    except Exception:
        return []


def analyze_under(client, model, hypothesis, evidence):
    """Force analysis of the pooled data UNDER one hypothesis class; report only data-grounded
    conclusions (V/G asymmetry: resolving a posed question from data is easier than posing it)."""
    ev = "\n".join(evidence[:50])[:5500]
    return call_llm(client, model=model, system="Answer strictly from the data.",
        user=(f"Observations:\n{ev}\n\nQUESTION TO RESOLVE FROM THE DATA:\n{hypothesis}\n\n"
              f"Work it out numerically from the observations (count alleles/phenotypes per locus, "
              f"compare offspring ratios, check for missing/lethal classes, cluster quantitative "
              f"values). State the DATA-SUPPORTED answer to this question precisely, with the numbers "
              f"that justify it. If the data are insufficient, say so."),
        max_tokens=500, temperature=0.1)


def assemble(client, model, task_hint, findings):
    joined = "\n\n".join(f"- {f}" for f in findings)[:6000]
    return call_llm(client, model=model, system="Assemble a precise final report.",
        user=(f"Domain: {task_hint}\n\nData-grounded findings from systematic hypothesis testing:\n"
              f"{joined}\n\nAssemble the FINAL answer covering every inference target, keeping each "
              f"data-supported finding (including non-default ones like unusual ploidy or lethal "
              f"combinations) with its quantitative detail. Be precise and complete."),
        max_tokens=1200, temperature=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="cuhh_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--ref_model", default=WEAK)
    ap.add_argument("--env", default="bio")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--judge_model", default=WEAK)
    ap.add_argument("--ablation", default="MARS-full")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "conservation_uh_hypdiv" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    task_hint = "alien organism genetics: infer ploidy, meiosis, and per-trait inheritance rules"
    results = {}; hyp_scores, best_scores = [], []
    print(f"=== UltraHorizon — HYPOTHESIS DIVERSIFICATION (break the support ceiling) — {args.model} ===\n")
    for ep in range(args.episodes):
        seed = 42 + ep
        task = UHTask(env_name=args.env, seed=seed, steps=args.steps, free=False,
                      difficulty=args.difficulty, judge_model=args.judge_model, action_budget=args.budget)
        evidence, scores = [], []
        for i in range(args.k):
            with contextlib.redirect_stdout(io.StringIO()):
                rep = run_one(task, ablation_name=args.ablation, gen_model=args.model, ref_model=args.ref_model)
            evidence += extract_evidence(rep)
            scores.append(float(rep.get("final_score", 0.0) or 0.0))
        hyps = enumerate_hypotheses(client, args.model, task_hint, evidence)
        findings = [analyze_under(client, args.model, h, evidence) for h in hyps]
        final = assemble(client, args.model, task_hint, findings)
        hyp_score = judge_text(args.env, seed, args.difficulty, args.judge_model, final)

        best_s = max(scores)
        hyp_scores.append(hyp_score); best_scores.append(best_s)
        results[f"ep{ep}"] = {"hypdiv": hyp_score, "best_rollout": best_s,
                              "rollout_scores": scores, "n_hypotheses": len(hyps), "hypotheses": hyps}
        print(f"  ep{ep}: HYP-DIV={hyp_score:.0f}  best_rollout={best_s:.0f}  "
              f"rollouts={[int(s) for s in scores]}  ({len(hyps)} hyps tested)")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = len(hyp_scores)
    mh = sum(hyp_scores) / n if n else 0.0
    mb = sum(best_scores) / n if n else 0.0
    print(f"\n=== RESULT (episodes={n}) ===")
    print(f"  HYPOTHESIS-DIVERSIFIED = {mh:.1f}")
    print(f"  best-rollout (prior ceiling) = {mb:.1f}")
    verdict = "BREAKS the support ceiling" if mh > mb + 1e-9 else ("matches" if abs(mh-mb) < 3 else "below")
    print(f"  → hypothesis diversification {verdict}")
    out_path.write_text(json.dumps({"model": args.model, "episodes": n,
                                    "hypdiv_mean": round(mh, 2), "best_rollout_mean": round(mb, 2),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
