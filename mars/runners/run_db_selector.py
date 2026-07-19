"""Gold-free SELECTOR over the DiscoveryBench portfolio — the decisive test.

Per task we already have candidate hypotheses from 4 heterogeneous methods
(4o/mini data-first, question-grounded, type-routed) and their paper-judge HMS.
Oracle-max (best per task) = 32.89 > SOTA. Realized score depends on whether a
WEAK selector, WITHOUT the gold, can pick the right candidate.

Selector: mini reads the question + the candidate hypotheses and picks the one
that most directly/specifically answers it (selection among concrete candidates is
far easier than open verification). Realized HMS = HMS of the picked candidate.

Compare realized vs oracle (33) vs best-single (~15) vs random.

  python -m mars.runners.run_db_selector --run_id sel_v1 --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=False)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.runners.run_db_official_eval import load_official_real_tasks

UDR = _PROJ / "lmw" / "universal_discovery_real"


def _preds(run):
    """task_key -> pred_hypo from a predictions.jsonl"""
    p = UDR / run / "predictions.jsonl"
    out = {}
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            out[r["task_key"]] = r.get("pred_hypo", "")
    return out


def _rejudge(run):
    p = UDR / run / "rejudge_gpt-4-turbo.json"
    if p.exists():
        return {k: v["mean"] for k, v in json.load(open(p))["per_task"].items()}
    return {}


def _summary(path):
    d = json.load(open(path))
    res = d.get("results", {})
    return ({k: v.get("hypo", "") for k, v in res.items()},
            {k: v.get("HMS", 0.0) for k, v in res.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="sel_v1")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "db_selector" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    # assemble candidates: method -> (hypo_map, hms_map)
    cands = {
        "4o_datafirst": (_preds("db_real_4o_diverse"), _rejudge("db_real_4o_diverse")),
        "mini_datafirst": (_preds("db_real_dpsr_diverse"), _rejudge("db_real_dpsr_diverse")),
        "qground": _summary(_PROJ / "lmw/db_qground/qg_v1/summary.json"),
        "routed": _summary(_PROJ / "lmw/db_routed/dbr_v1/summary.json"),
    }
    methods = list(cands)
    keys = sorted(set().union(*[set(h) for h, _ in cands.values()]))

    tasks = {t.task_key: t for t in load_official_real_tasks(
        _PROJ / "discoverybench_repo", max_tasks=None, datasets=None, task_keys=set(keys))}

    client = make_openai_client()
    realized = oracle = best_single_sum = 0.0
    rng = random.Random(0)
    rand_sum = 0.0
    rows = []
    # best single method overall (for baseline)
    single_means = {m: sum((cands[m][1].get(k, 0) or 0) for k in keys) / len(keys) for m in methods}
    best_single = max(single_means, key=single_means.get)

    print(f"=== Gold-free portfolio selector — {args.model} ===\n")
    for k in keys:
        t = tasks.get(k)
        opts = []
        for m in methods:
            hypo = (cands[m][0].get(k, "") or "").strip()
            hms = cands[m][1].get(k, 0.0) or 0.0
            if hypo:
                opts.append((m, hypo, hms))
        if not opts:
            rows.append({"task": k, "picked": None, "hms": 0.0}); continue
        oracle += max(h for _, _, h in opts)
        rand_sum += rng.choice(opts)[2]
        # gold-free selection by mini
        listing = "\n".join(f"[{i}] {h[:240]}" for i, (_, h, _) in enumerate(opts))
        q = t.task.query if t else k
        raw = call_llm(client, model=args.model, system="Answer with a single integer index.",
            user=(f"Question: {q}\n\nCandidate hypotheses:\n{listing}\n\n"
                  f"Which candidate MOST directly and specifically answers the question with a "
                  f"concrete, grounded finding (right population, variables, and direction)? "
                  f"Reply with just the index."), max_tokens=6, temperature=0.0)
        ds = "".join(c if c.isdigit() else " " for c in raw).split()
        idx = int(ds[0]) if ds and int(ds[0]) < len(opts) else 0
        pm, ph, phms = opts[idx]
        realized += phms
        rows.append({"task": k, "picked": pm, "hms": phms,
                     "oracle": max(h for _, _, h in opts)})
        print(f"  {k:46} picked={pm:14} HMS={phms:5.1f}  (oracle={max(h for *_,h in opts):.0f})")

    n = len(keys)
    print(f"\n=== RESULT (N={n}) ===")
    print(f"  REALIZED (gold-free selector): {realized/n:.2f}")
    print(f"  oracle-max (upper bound)     : {oracle/n:.2f}")
    print(f"  best single method ({best_single}): {single_means[best_single]:.2f}")
    print(f"  random selection             : {rand_sum/n:.2f}")
    print(f"  published: standard ~15, Reflexion+Oracle 24.5")
    capture = (realized/n - single_means[best_single]) / (oracle/n - single_means[best_single] + 1e-9)
    print(f"  → captured {100*capture:.0f}% of the portfolio gap; "
          f"{'BEATS SOTA' if realized/n > 24.5 else 'below SOTA 24.5'}")
    out_path.write_text(json.dumps({"model": args.model, "realized": realized/n, "oracle": oracle/n,
                                    "best_single": single_means[best_single], "random": rand_sum/n,
                                    "rows": rows}, indent=2))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
