"""UltraHorizon — CLAIM-LEVEL conservation (the right mode for open-ended prose).

Whole-answer consensus fails on UH: equivalent good answers phrase differently, and the
majority answer isn't the best. The fix keeps the SAME conservation principle but applies it
at the CLAIM level (like DiscoveryBench's quantity): run K independent rollouts, then keep only
the findings CORROBORATED BY A MAJORITY of rollouts (conserved sub-claims; idiosyncratic ones
dropped) and SYNTHESIZE the final answer from them. Universal — no domain knowledge; the model
does the extraction/synthesis, the shell enforces majority-conservation.

Reported against two honest baselines: the mean single rollout, and the best single rollout.

  python -m mars.runners.run_conservation_uh_claims --run_id cuhc_v1 --episodes 3 --k 3
"""
from __future__ import annotations

import argparse
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
from mars.runners.run_uh_official import UHTask, run_one, UltraHorizonOfficialAdapter

WEAK = "openai/gpt-4o-mini"


def synthesize_conserved(client, model, rollout_texts, k):
    """Claim-level conservation: keep ONLY findings corroborated by a majority of independent
    rollouts; drop idiosyncratic ones; synthesize the final answer from the conserved claims."""
    maj = k // 2 + 1
    reports = "\n\n".join(f"--- REPORT {i+1} ---\n{t[:1600]}" for i, t in enumerate(rollout_texts))
    prompt = (
        f"{k} INDEPENDENT investigation reports of the same system are below. Apply CONSERVATION: "
        f"a finding is trustworthy only if it is CORROBORATED BY AT LEAST {maj} of the {k} reports; "
        f"findings asserted by only one report are idiosyncratic and must be DROPPED.\n\n"
        f"{reports}\n\n"
        f"Write the FINAL answer using ONLY the majority-corroborated findings, stated precisely "
        f"(include quantitative values only if a majority agree on them). Do not add new claims."
    )
    return call_llm(client, model=model, system="You synthesize only majority-corroborated findings.",
                    user=prompt, max_tokens=1200, temperature=0.2)


def judge_text(env_name, seed, difficulty, judge_model, text):
    """Score an arbitrary final answer via a free=True env (no experiment gate; same ground
    truth by seed). Returns the official judge final_score."""
    task = UHTask(env_name=env_name, seed=seed, steps=1, free=True,
                  difficulty=difficulty, judge_model=judge_model, action_budget=3)
    ad = UltraHorizonOfficialAdapter(task)
    ad.execute("commit_final_result", {"content": text})
    sd = ad.score_episode([])
    return float(sd.get("final_score", 0.0) or 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="cuhc_v1")
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
    out_dir = _PROJ / "lmw" / "conservation_uh_claims" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    results = {}
    conserved_scores, mean_scores, best_scores = [], [], []
    print(f"=== UltraHorizon — CLAIM-LEVEL conservation — {args.model} ===\n")
    for ep in range(args.episodes):
        seed = 42 + ep
        task = UHTask(env_name=args.env, seed=seed, steps=args.steps, free=False,
                      difficulty=args.difficulty, judge_model=args.judge_model, action_budget=args.budget)
        texts, scores = [], []
        for i in range(args.k):
            rep = run_one(task, ablation_name=args.ablation, gen_model=args.model, ref_model=args.ref_model)
            texts.append(rep.get("submitted_preview", "") or "")
            scores.append(float(rep.get("final_score", 0.0) or 0.0))
        syn = synthesize_conserved(client, args.model, texts, args.k)
        syn_score = judge_text(args.env, seed, args.difficulty, args.judge_model, syn)

        mean_s = sum(scores) / len(scores)
        best_s = max(scores)
        conserved_scores.append(syn_score); mean_scores.append(mean_s); best_scores.append(best_s)
        results[f"ep{ep}"] = {"conserved": syn_score, "rollout_scores": scores,
                              "mean_rollout": round(mean_s, 1), "best_rollout": best_s}
        print(f"  ep{ep}: CONSERVED={syn_score:.0f}  | rollouts={[int(s) for s in scores]} "
              f"(mean={mean_s:.0f}, best={best_s:.0f})")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = len(conserved_scores)
    mc = sum(conserved_scores) / n if n else 0.0
    mm = sum(mean_scores) / n if n else 0.0
    mb = sum(best_scores) / n if n else 0.0
    print(f"\n=== RESULT (episodes={n}) ===")
    print(f"  CLAIM-CONSERVED synthesis = {mc:.1f}")
    print(f"  single-rollout mean       = {mm:.1f}   (baseline: take one rollout)")
    print(f"  best-rollout (oracle)     = {mb:.1f}   (ceiling)")
    verdict = "BEATS" if mc > mm + 1e-9 else ("TIES" if abs(mc - mm) < 3 else "TRAILS")
    print(f"  → claim-conservation {verdict} the single-rollout baseline")
    out_path.write_text(json.dumps({"model": args.model, "episodes": n,
                                    "claim_conserved_mean": round(mc, 2),
                                    "single_rollout_mean": round(mm, 2),
                                    "best_rollout_mean": round(mb, 2),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
