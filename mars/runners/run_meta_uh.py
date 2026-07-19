"""META-SELECTOR on UltraHorizon — does the outer loop autonomously rediscover the NON-OBVIOUS
winner (evidence-select) that a human found only after 5 attempts?

This is the decisive test of the meta-loop's contribution: automating the per-benchmark method
research. The method space is the 4 aggregators over the SAME K rollouts (run once per seed):
  consensus | claim-majority | evidence-synthesis | evidence-selection
On PROBE seeds (known ground truth by construction) the meta-loop scores each aggregator and
picks the best; we then VALIDATE on HELD-OUT seeds that the auto-picked method also wins there.
If it auto-picks evidence-selection, the outer loop did the human's research by itself.

  python -m mars.runners.run_meta_uh --run_id muh_v1 --probe_seeds 42,43 --holdout_seeds 44,45
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from collections import Counter
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

from mars.agents.base import make_openai_client
from mars.runners.run_uh_official import UHTask, run_one
from mars.runners.run_conservation_uh import canonicalize
from mars.runners.run_conservation_uh_claims import synthesize_conserved
from mars.runners.run_conservation_uh_evidence import extract_evidence, synthesize_from_evidence, judge_text
from mars.runners.run_conservation_uh_select import evidence_support

WEAK = "openai/gpt-4o-mini"


def gather_rollouts(client, model, ref_model, env, seed, steps, budget, difficulty, judge_model, k, ablation):
    task = UHTask(env_name=env, seed=seed, steps=steps, free=False, difficulty=difficulty,
                  judge_model=judge_model, action_budget=budget)
    rolls, evidence = [], []
    for i in range(k):
        with contextlib.redirect_stdout(io.StringIO()):
            rep = run_one(task, ablation_name=ablation, gen_model=model, ref_model=ref_model)
        rolls.append({"answer": rep.get("submitted_preview", "") or "",
                      "score": float(rep.get("final_score", 0.0) or 0.0)})
        evidence += extract_evidence(rep)
    return rolls, evidence


# ---- the 4 aggregators (method space). Each returns a final answer TEXT. ----

def agg_consensus(client, model, rolls, evidence, k):
    keyed = [(canonicalize(client, model, r["answer"]), r["answer"]) for r in rolls if r["answer"]]
    ok = [(kk, a) for kk, a in keyed if kk]
    if not ok:
        return None
    counts = Counter(kk for kk, _ in ok)
    key, n = counts.most_common(1)[0]
    if n < 2 or n <= len(ok) / 2:
        return None                                  # no majority -> abstain
    return next(a for kk, a in ok if kk == key)


def agg_claims(client, model, rolls, evidence, k):
    return synthesize_conserved(client, model, [r["answer"] for r in rolls], k)


def agg_evidence_synth(client, model, rolls, evidence, k):
    return synthesize_from_evidence(client, model, evidence, [r["answer"] for r in rolls])


def agg_evidence_select(client, model, rolls, evidence, k):
    best, bs = None, -1.0
    for r in rolls:
        if not r["answer"]:
            continue
        s = evidence_support(client, model, evidence, r["answer"])
        if s > bs:
            bs, best = s, r["answer"]
    return best


AGGREGATORS = {"consensus": agg_consensus, "claims": agg_claims,
               "evidence_synth": agg_evidence_synth, "evidence_select": agg_evidence_select}


def score_seed(client, model, ref_model, args, seed):
    """Run K rollouts once, apply ALL aggregators, judge each -> {agg: score}."""
    rolls, evidence = gather_rollouts(client, model, ref_model, args.env, seed, args.steps,
                                      args.budget, args.difficulty, args.judge_model, args.k, args.ablation)
    scores = {}
    for name, fn in AGGREGATORS.items():
        ans = fn(client, model, rolls, evidence, args.k)
        sc = judge_text(args.env, seed, args.difficulty, args.judge_model, ans) if ans else 0.0
        scores[name] = sc
    scores["_best_rollout"] = max((r["score"] for r in rolls), default=0.0)
    scores["_single_mean"] = sum(r["score"] for r in rolls) / len(rolls) if rolls else 0.0
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="muh_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--ref_model", default=WEAK)
    ap.add_argument("--env", default="bio")
    ap.add_argument("--probe_seeds", default="42,43")
    ap.add_argument("--holdout_seeds", default="44,45")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--judge_model", default=WEAK)
    ap.add_argument("--ablation", default="MARS-full")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "meta_uh" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    probe = [int(s) for s in args.probe_seeds.split(",")]
    holdout = [int(s) for s in args.holdout_seeds.split(",")]
    print(f"=== META-SELECTOR on UltraHorizon — {args.model} ===")
    print(f"    method space: {list(AGGREGATORS)}")
    print(f"    probe seeds {probe} -> select; holdout seeds {holdout} -> validate\n")

    probe_scores = {a: [] for a in AGGREGATORS}
    print("  [PROBE] scoring aggregators on labeled probe seeds:")
    for s in probe:
        sc = score_seed(client, args.model, args.ref_model, args, s)
        for a in AGGREGATORS:
            probe_scores[a].append(sc[a])
        print(f"    seed{s}: " + " ".join(f"{a}={sc[a]:.0f}" for a in AGGREGATORS) +
              f"  (best_rollout={sc['_best_rollout']:.0f}, single_mean={sc['_single_mean']:.0f})", flush=True)
    probe_mean = {a: sum(v) / len(v) for a, v in probe_scores.items()}
    picked = max(probe_mean, key=probe_mean.get)
    print(f"\n  META-PICK (gold-free-on-real-task, chosen on probes): '{picked}'")
    print(f"    probe means: {json.dumps({a: round(probe_mean[a],1) for a in AGGREGATORS})}")

    print(f"\n  [HOLDOUT] validating on held-out seeds:")
    hold_scores = {a: [] for a in AGGREGATORS}
    for s in holdout:
        sc = score_seed(client, args.model, args.ref_model, args, s)
        for a in AGGREGATORS:
            hold_scores[a].append(sc[a])
        print(f"    seed{s}: " + " ".join(f"{a}={sc[a]:.0f}" for a in AGGREGATORS) +
              f"  (best_rollout={sc['_best_rollout']:.0f})", flush=True)
    hold_mean = {a: sum(v) / len(v) for a, v in hold_scores.items()}
    hold_best = max(hold_mean, key=hold_mean.get)
    validated = picked == hold_best

    print(f"\n=== META VERDICT ===")
    print(f"  auto-picked on probes: '{picked}'  |  actual best on holdout: '{hold_best}'")
    print(f"  holdout means: {json.dumps({a: round(hold_mean[a],1) for a in AGGREGATORS})}")
    print(f"  {'MATCH ✓ — meta-loop auto-rediscovered the non-obvious winner' if validated else 'MISMATCH ✗'}")
    out_path.write_text(json.dumps({"picked": picked, "holdout_best": hold_best, "validated": validated,
                                    "probe_mean": probe_mean, "holdout_mean": hold_mean,
                                    "probe_scores": probe_scores, "holdout_scores": hold_scores},
                                   indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
