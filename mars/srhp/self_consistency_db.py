"""
Self-Consistency for DiscoveryBench — the Uncertainty-Driven Investigation (UDI)
principle applied to an open-ended reasoning task.

Why every prior scaffold lost to the 0.68 baseline: they CONSTRAINED the model
to one narrow structured relationship, losing the free-form flexibility that
earns partial credit on multi-faceted queries. UDI inverts this: do not
constrain — AMPLIFY. Sample k free-form hypotheses (preserving richness),
treat their spread as the model's UNCERTAINTY, and reduce it by synthesising a
consensus. Variance (the thing that makes the baseline occasionally pick the
wrong variable/direction) drops; flexibility is preserved.

Modes:
  consensus : generate k hypotheses → LLM synthesises the consensus one
  selfselect: generate k → LLM picks its own best
  verify    : consensus, then a single data-fit verifies the stated direction

Win condition: beat free-form MARS-full baseline HMS = 0.680.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from dotenv import load_dotenv
    _env = _PROJ / "autodiscovery" / ".env.local"
    if _env.exists():
        load_dotenv(_env, override=False)
except ImportError:
    pass

import numpy as np

from mars.adapters.discoverybench_adapter import load_db_tasks
from ols.adapters.discoverybench_adapter import DiscoveryBenchAdapter
from mars.agents.base import call_llm, make_openai_client


_GEN_SYS = (
    "You are a scientific analyst. Answer the discovery query with a single "
    "concise hypothesis (2-3 sentences) naming the key outcome and explanatory "
    "variables and the relationship (direction/magnitude). Be specific and "
    "grounded in the dataset's columns."
)

_CONSENSUS_SYS = (
    "You are given several independently-written candidate hypotheses answering "
    "the same discovery query. They reflect the analyst's uncertainty. "
    "Synthesise the SINGLE best consensus hypothesis: keep the variables and "
    "relationship that RECUR across candidates (high agreement = high "
    "confidence), drop idiosyncratic claims that appear only once. Output ONLY "
    "the final 2-3 sentence hypothesis."
)


def self_consistency_episode(adapter, client, model, k=5, mode="consensus"):
    query = adapter.get_query_text()
    cols = list(adapter.get_dataframes().get(
        list(adapter.get_dataframes())[0]).select_dtypes(include="number").columns) \
        if adapter.get_dataframes() else []
    col_desc = adapter.get_column_descriptions()
    col_block = "\n".join(f"  {c}: {col_desc.get(c,'')[:80]}" for c in cols[:30])
    user = f"DISCOVERY QUERY:\n{query}\n\nDATASET COLUMNS:\n{col_block}\n\nWrite the hypothesis."

    # 1. sample k free-form hypotheses (the model's uncertainty distribution)
    cands = []
    for _ in range(k):
        try:
            h = call_llm(client, model, _GEN_SYS, user, max_tokens=200, temperature=0.8)
            if h and h.strip():
                cands.append(h.strip())
        except Exception:
            continue
    if not cands:
        return ""
    if len(cands) == 1:
        return cands[0]

    # 2. reduce uncertainty → consensus
    numbered = "\n\n".join(f"Candidate {i+1}: {c}" for i, c in enumerate(cands))
    cuser = (f"DISCOVERY QUERY:\n{query}\n\nCANDIDATE HYPOTHESES:\n{numbered}\n\n"
             "Synthesise the consensus hypothesis now.")
    try:
        consensus = call_llm(client, model, _CONSENSUS_SYS, cuser,
                             max_tokens=220, temperature=0.2)
        return consensus.strip() if consensus else cands[0]
    except Exception:
        return cands[0]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    n = int(os.environ.get("MARS_DB_N", "6"))
    k = int(os.environ.get("MARS_SC_K", "5"))
    model = os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini")
    judge = os.environ.get("OLS_DB_JUDGE_MODEL", "openai/gpt-4o")
    tasks = load_db_tasks(_PROJ / "discoverybench_repo", split="train", max_tasks=n)
    client = make_openai_client()

    print(f"\n=== Self-Consistency (UDI) × DiscoveryBench  k={k} ===")
    print(f"  model={model} judge={judge}  N={len(tasks)}")
    print(f"  baseline MARS-full HMS = 0.680 (target to beat)\n")

    scores = []
    for i, t in enumerate(tasks):
        adapter = DiscoveryBenchAdapter(task=t, budget=12, judge_model=judge)
        try:
            hypo = self_consistency_episode(adapter, client, model, k=k)
            adapter._submitted_hypothesis = hypo
            sc = adapter.score_episode([], final_artifact=hypo)
            hms = sc.get("HMS", 0.0)
        except Exception as e:
            hms, hypo = 0.0, f"(err {e})"
        scores.append(hms)
        print(f"  [{i+1}/{len(tasks)}] {t.task_id[-28:]:<28} HMS={hms:.2f}  {str(hypo)[:90]}")

    import statistics as st
    m = st.fmean(scores) if scores else 0.0
    print(f"\n  Self-Consistency HMS.mean = {m:.3f}  (baseline 0.680)")
    print(f"  → {'BEATS baseline' if m > 0.680 else 'below/at baseline'}")


if __name__ == "__main__":
    main()
