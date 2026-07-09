"""UltraHorizon — EVIDENCE-based SELECTION (the pivot that should WIN on UH).

Lessons that led here:
  * peer-consensus fails (rewards shared bias, discards rare-correct);
  * evidence RE-SYNTHESIS fails (routes back through the weak model's analysis, loses a good
    rollout's work — e.g. 2/3 rollouts scored 87 but re-synthesis got 57).

So we SELECT, we don't merge: run K rollouts, then VERIFY each rollout's final answer against
the POOLED experimental evidence and keep the best-supported one. This preserves the good
rollout's full answer and uses DATA (not peers) as the verifier — exploiting V/G asymmetry
(checking a claim against observations is easier than discovering it). A rare-but-correct
answer wins because the data back it, regardless of how many rollouts found it.

  python -m mars.runners.run_conservation_uh_select --run_id cuhs_v1 --episodes 3 --k 3
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
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
from mars.runners.run_uh_official import UHTask, run_one, UltraHorizonOfficialAdapter
from mars.runners.run_conservation_uh_evidence import extract_evidence, judge_text

WEAK = "openai/gpt-4o-mini"


def evidence_support(client, model, evidence, answer):
    """V/G asymmetry: score how well the POOLED observations SUPPORT this answer, [0,1].
    Validation is easier than generation, so a weak model can rank a correct answer above a
    wrong one even when it discovered the correct one only rarely."""
    ev = "\n".join(evidence[:60])[:6000]
    prompt = (
        f"POOLED EXPERIMENTAL OBSERVATIONS of one system:\n{ev}\n\n"
        f"CANDIDATE CONCLUSION:\n{answer[:1500]}\n\n"
        f"How well do the OBSERVATIONS support this conclusion? Check each quantitative claim "
        f"against the numbers (e.g. distinct size-score clusters, offspring ratios, lethal "
        f"combinations). Score 0.0 (contradicted) to 1.0 (fully data-supported). "
        f'Return JSON only: {{"support": <float>, "why": "<short>"}}'
    )
    raw = call_llm(client, model=model, system="Return only JSON.", user=prompt,
                   max_tokens=200, temperature=0.0)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        return float(json.loads(t).get("support", 0.0))
    except Exception:
        m = re.search(r'"support"\s*:\s*([0-9.]+)', t)
        return float(m.group(1)) if m else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="cuhs_v1")
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
    out_dir = _PROJ / "lmw" / "conservation_uh_select" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    results = {}
    sel_scores, mean_scores, best_scores = [], [], []
    print(f"=== UltraHorizon — EVIDENCE-based SELECTION (data as verifier) — {args.model} ===\n")
    for ep in range(args.episodes):
        seed = 42 + ep
        task = UHTask(env_name=args.env, seed=seed, steps=args.steps, free=False,
                      difficulty=args.difficulty, judge_model=args.judge_model, action_budget=args.budget)
        evidence, cands = [], []
        for i in range(args.k):
            with contextlib.redirect_stdout(io.StringIO()):
                rep = run_one(task, ablation_name=args.ablation, gen_model=args.model, ref_model=args.ref_model)
            evidence += extract_evidence(rep)
            cands.append({"answer": rep.get("submitted_preview", "") or "",
                          "score": float(rep.get("final_score", 0.0) or 0.0)})
        # verify each candidate against POOLED evidence; select best-supported
        for c in cands:
            c["support"] = evidence_support(client, args.model, evidence, c["answer"]) if c["answer"] else 0.0
        chosen = max(cands, key=lambda c: c["support"])
        sel_score = chosen["score"]

        scores = [c["score"] for c in cands]
        mean_s = sum(scores) / len(scores); best_s = max(scores)
        sel_scores.append(sel_score); mean_scores.append(mean_s); best_scores.append(best_s)
        results[f"ep{ep}"] = {"selected": sel_score, "support_of_selected": round(chosen["support"], 2),
                              "rollout_scores": scores,
                              "supports": [round(c["support"], 2) for c in cands],
                              "mean_rollout": round(mean_s, 1), "best_rollout": best_s}
        print(f"  ep{ep}: SELECTED={sel_score:.0f} (support={chosen['support']:.2f}) | "
              f"rollouts={[int(s) for s in scores]} supports={[round(c['support'],2) for c in cands]} "
              f"(mean={mean_s:.0f}, best={best_s:.0f})")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = len(sel_scores)
    ms = sum(sel_scores) / n if n else 0.0
    mm = sum(mean_scores) / n if n else 0.0
    mb = sum(best_scores) / n if n else 0.0
    print(f"\n=== RESULT (episodes={n}) ===")
    print(f"  EVIDENCE-SELECTED = {ms:.1f}")
    print(f"  single-rollout mean = {mm:.1f}   (baseline)")
    print(f"  best-rollout (oracle) = {mb:.1f}   (ceiling)")
    verdict = "BEATS" if ms > mm + 1e-9 else ("TIES" if abs(ms - mm) < 3 else "TRAILS")
    print(f"  → evidence-selection {verdict} the single-rollout baseline")
    out_path.write_text(json.dumps({"model": args.model, "episodes": n,
                                    "evidence_selected_mean": round(ms, 2),
                                    "single_rollout_mean": round(mm, 2),
                                    "best_rollout_mean": round(mb, 2),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
