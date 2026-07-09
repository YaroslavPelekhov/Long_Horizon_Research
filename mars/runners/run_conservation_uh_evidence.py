"""UltraHorizon — EVIDENCE conservation (conserve against the ENVIRONMENT, not against peers).

The pivot that makes conservation WIN on UH. Peer-consensus fails because it rewards shared
bias and discards rare-but-correct insight. Here we return to the NB/DB principle — conserve
against DATA — applied to an agentic task:

  1. Run K independent rollouts. Each gathers experimental EVIDENCE (crosses/queries) and
     proposes hypotheses. Collectively the K rollouts gather MORE experiments than any one.
  2. POOL all evidence + the UNION of all proposed hypotheses.
  3. Synthesize the final answer keeping ONLY claims the POOLED EVIDENCE SUPPORTS — a rare
     claim proposed by ONE rollout SURVIVES if the data backs it; a common claim is DROPPED if
     the data contradicts it. (V/G asymmetry: validating a claim against data is easier than
     discovering it.)
  4. Judge the synthesized answer.

Reported vs single-rollout mean and best-rollout ceiling.

  python -m mars.runners.run_conservation_uh_evidence --run_id cuhe_v1 --episodes 3 --k 3
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
from mars.runners.run_uh_official import UHTask, run_one, UltraHorizonOfficialAdapter

WEAK = "openai/gpt-4o-mini"


def extract_evidence(rep, max_events=40):
    """Pull the raw experimental observations (non-commit tool events) from a rollout."""
    lines = []
    for h in rep.get("env_tool_history", []):
        act = h.get("action")
        if act in (None, "commit_final_result"):
            continue
        res = h.get("result")
        rs = json.dumps(res, default=str) if not isinstance(res, str) else res
        lines.append(f"{act}({json.dumps(h.get('args', {}), default=str)}) -> {rs[:600]}")
        if len(lines) >= max_events:
            break
    return lines


def synthesize_from_evidence(client, model, evidence, hypotheses):
    ev = "\n".join(evidence[:60])[:6000]
    hyp = "\n\n".join(f"HYPOTHESIS {i+1}:\n{h[:900]}" for i, h in enumerate(hypotheses))[:4000]
    prompt = (
        f"POOLED EXPERIMENTAL OBSERVATIONS from several independent investigations of ONE system:\n"
        f"{ev}\n\n"
        f"PROPOSED HYPOTHESES (from those investigations; some may be wrong or incomplete):\n{hyp}\n\n"
        f"Produce the FINAL answer. CRITICAL rule (evidence conservation): include a claim ONLY if "
        f"the POOLED OBSERVATIONS support it — KEEP a claim even if just ONE hypothesis proposed it, "
        f"as long as the data back it; DROP a claim even if several proposed it, if the data do not "
        f"support it. Re-derive quantitative values (e.g. size groupings) directly from the numbers "
        f"in the observations. Cover all inference targets the task asks for."
    )
    return call_llm(client, model=model, system="You keep only claims the data support.",
                    user=prompt, max_tokens=1300, temperature=0.2)


def judge_text(env_name, seed, difficulty, judge_model, text):
    task = UHTask(env_name=env_name, seed=seed, steps=1, free=True,
                  difficulty=difficulty, judge_model=judge_model, action_budget=3)
    ad = UltraHorizonOfficialAdapter(task)
    with contextlib.redirect_stdout(io.StringIO()):
        ad.execute("commit_final_result", {"content": text})
        sd = ad.score_episode([])
    return float(sd.get("final_score", 0.0) or 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="cuhe_v1")
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
    out_dir = _PROJ / "lmw" / "conservation_uh_evidence" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    results = {}
    ev_scores, mean_scores, best_scores = [], [], []
    print(f"=== UltraHorizon — EVIDENCE conservation (conserve vs data, union of hypotheses) — {args.model} ===\n")
    for ep in range(args.episodes):
        seed = 42 + ep
        task = UHTask(env_name=args.env, seed=seed, steps=args.steps, free=False,
                      difficulty=args.difficulty, judge_model=args.judge_model, action_budget=args.budget)
        evidence, hypotheses, scores = [], [], []
        for i in range(args.k):
            with contextlib.redirect_stdout(io.StringIO()):
                rep = run_one(task, ablation_name=args.ablation, gen_model=args.model, ref_model=args.ref_model)
            evidence += extract_evidence(rep)
            hypotheses.append(rep.get("submitted_preview", "") or "")
            scores.append(float(rep.get("final_score", 0.0) or 0.0))
        syn = synthesize_from_evidence(client, args.model, evidence, hypotheses)
        syn_score = judge_text(args.env, seed, args.difficulty, args.judge_model, syn)

        mean_s = sum(scores) / len(scores); best_s = max(scores)
        ev_scores.append(syn_score); mean_scores.append(mean_s); best_scores.append(best_s)
        results[f"ep{ep}"] = {"evidence_conserved": syn_score, "rollout_scores": scores,
                              "mean_rollout": round(mean_s, 1), "best_rollout": best_s,
                              "n_evidence": len(evidence)}
        print(f"  ep{ep}: EVIDENCE={syn_score:.0f}  | rollouts={[int(s) for s in scores]} "
              f"(mean={mean_s:.0f}, best={best_s:.0f}, evidence_events={len(evidence)})")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = len(ev_scores)
    me = sum(ev_scores) / n if n else 0.0
    mm = sum(mean_scores) / n if n else 0.0
    mb = sum(best_scores) / n if n else 0.0
    print(f"\n=== RESULT (episodes={n}) ===")
    print(f"  EVIDENCE-CONSERVED synthesis = {me:.1f}")
    print(f"  single-rollout mean          = {mm:.1f}   (baseline)")
    print(f"  best-rollout (oracle)        = {mb:.1f}   (ceiling)")
    verdict = "BEATS" if me > mm + 1e-9 else ("TIES" if abs(me - mm) < 3 else "TRAILS")
    print(f"  → evidence-conservation {verdict} the single-rollout baseline")
    out_path.write_text(json.dumps({"model": args.model, "episodes": n,
                                    "evidence_conserved_mean": round(me, 2),
                                    "single_rollout_mean": round(mm, 2),
                                    "best_rollout_mean": round(mb, 2),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
