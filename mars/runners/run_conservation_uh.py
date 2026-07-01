"""UltraHorizon as a THIN ADAPTER over the SAME universal conservation engine.

Long-horizon agentic tasks are verifier-POOR at the judgment level, so conservation =
SELF-CONSISTENCY across independent rollouts: run K full episodes, canonicalize each final
conclusion to a short key, and the engine keeps the conclusion reproduced by a strict majority
(idiosyncratic dead-ends don't agree); else abstain. The selected rollout's official judge
score is reported.

Same engine, same consensus logic as the SAB adapter — only the candidate generator (a full
MARS episode) and the scorer (UH judge) are UH-specific. Episodes are expensive; defaults are
deliberately small.

  python -m mars.runners.run_conservation_uh --run_id cuh_v1 --episodes 1 --k 3
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
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=True)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.induction import conservation_discovery as cd
from mars.runners.run_uh_official import UHTask, run_one

WEAK = "openai/gpt-4o-mini"


def canonicalize(client, model, text):
    """Reduce a rich final conclusion to a short normalized key so independent rollouts that
    reached the SAME core finding cluster together. One cheap LLM call; no domain knowledge."""
    if not text or not text.strip():
        return None
    raw = call_llm(client, model=model, system="Output only a short normalized phrase.",
                   user=(f"Final conclusion of an investigation:\n{text[:1500]}\n\n"
                         f"State ONLY its single core finding as a normalized short phrase "
                         f"(<=8 words, lowercase, key entity + value/direction). No punctuation."),
                   max_tokens=30, temperature=0.0)
    return " ".join(raw.strip().lower().split())[:80] or None


def build_adapter(client, model, ref_model, task, k):
    def generate_candidates(ctx):
        out = []
        for i in range(k):
            rep = run_one(task, ablation_name="MARS-all-off", gen_model=model, ref_model=ref_model)
            submitted = rep.get("submitted", "") or ""
            score = float(rep.get("final_score", rep.get("primary", 0.0)) or 0.0)
            key = canonicalize(client, model, submitted)
            out.append(cd.Candidate(answer=submitted, answer_key=key,
                                    meta={"rollout": i, "self_score": score}))
        return out

    # score: judge score of the SELECTED conclusion (matched back from the rollouts we ran)
    def score(answer):
        return 0.0   # filled by main from the matching rollout's stored score

    return cd.DiscoveryAdapter(generate_candidates=generate_candidates,
                               transforms=lambda c: [], score=score, noise_floor=lambda c: 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="cuh_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--ref_model", default=WEAK)
    ap.add_argument("--env", default="bio")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--steps", type=int, default=3,
                    help="required experiments before commit is allowed (must be < budget)")
    ap.add_argument("--ablation", default="MARS-full")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--judge_model", default=WEAK)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "conservation_uh" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    results = {}; sc_list = []; n_ans = n_abs = 0
    print(f"=== UltraHorizon via universal conservation engine (consensus of rollouts) — {args.model} ===\n")
    for ep in range(args.episodes):
        task = UHTask(env_name=args.env, seed=42 + ep, steps=args.steps, free=False,
                      difficulty=args.difficulty, judge_model=args.judge_model, action_budget=args.budget)
        # generate K rollouts, keeping each rollout's own judge score for reporting
        rollouts = []
        for i in range(args.k):
            rep = run_one(task, ablation_name=args.ablation, gen_model=args.model, ref_model=args.ref_model)
            sub = rep.get("submitted_preview", "") or rep.get("submitted", "") or ""
            sc = float(rep.get("final_score", rep.get("primary", 0.0)) or 0.0)
            key = canonicalize(client, args.model, sub)
            rollouts.append({"answer": sub, "key": key, "score": sc})
            print(f"  ep{ep} rollout{i}: score={sc:.0f} key={key!r}")
        cands = [cd.Candidate(answer=r["answer"], answer_key=r["key"], meta={"score": r["score"]})
                 for r in rollouts]
        sel = cd.select(cd.DiscoveryAdapter(generate_candidates=lambda c: cands,
                        transforms=lambda c: [], score=lambda a: 0.0, noise_floor=lambda c: 0.0), {})
        if sel.abstained:
            n_abs += 1
            # honest baseline comparison: mean of rollout scores if we'd answered anyway
            results[f"ep{ep}"] = {"status": "ABSTAIN", "reason": sel.reason,
                                  "rollout_scores": [r["score"] for r in rollouts]}
            print(f"  ep{ep}: ABSTAIN ({sel.reason})\n"); continue
        chosen = next(r for r in rollouts if r["answer"] == sel.answer)
        sc_list.append(chosen["score"]); n_ans += 1
        results[f"ep{ep}"] = {"status": "ANSWER", "score": chosen["score"], "reason": sel.reason,
                              "rollout_scores": [r["score"] for r in rollouts]}
        print(f"  ep{ep}: ANSWER score={chosen['score']:.0f} ({sel.reason})\n")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_ans + n_abs
    prec = sum(sc_list) / len(sc_list) if sc_list else 0.0
    # honest reference: what a single random rollout would score on average
    all_scores = [s for r in results.values() for s in r.get("rollout_scores", [])]
    rand = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"=== RESULT (episodes={n}) ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  score on answered = {prec:.1f}  |  single-rollout mean = {rand:.1f}")
    print(f"  (consensus across K rollouts; same engine as NB/DB/SAB)")
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "score_answered": round(prec, 2), "single_rollout_mean": round(rand, 2),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
