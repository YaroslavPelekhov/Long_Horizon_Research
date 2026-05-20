"""
Benchmark-correctness check: are the axis-1 (-goal) and axis-6 (-abandon)
ablations measuring DISTINCT capabilities, or the same competence twice?

2x2 factorial on ConnectorAgent(choose_goals in {T,F}) x (can_abandon in
{T,F}), memory fixed True, RPS averaged over the L1 dev family. Report cell
means, the two main effects, and the interaction.

Reading: if both main effects are clearly non-zero AND the interaction is
small relative to them, the two ablations isolate separable capabilities
(the entanglement is benign and reportable). If the interaction dominates,
they are genuinely confounded and must be reframed. Deterministic, no API.
"""

from __future__ import annotations

import statistics as st

from agents import ConnectorAgent
from harness import run_single
from scm import Schema, make_structure

DEV = [Schema("d1", n_clusters=2, chain_len=1, n_latents=2, real_decoys=1),
       Schema("d2", n_clusters=3, chain_len=1, n_latents=2, real_decoys=1),
       Schema("d3", n_clusters=3, chain_len=2, n_latents=3, real_decoys=1),
       Schema("d4", n_clusters=4, chain_len=1, n_latents=2, real_decoys=2)]
SS, NS = [11, 22, 33], [1, 2]


def cell(goals: bool, abandon: bool) -> float:
    vals = []
    for sc in DEV:
        for ss in SS:
            stru = make_structure(ss, sc)
            for ns in NS:
                m = run_single(ConnectorAgent(True, goals, abandon), stru, ns)
                vals.append(m["RPS"])
    return st.fmean(vals)


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    TT = cell(True, True)     # goals on,  abandon on   (= Full)
    FT = cell(False, True)    # goals off, abandon on    (= -goal)
    TF = cell(True, False)    # goals on,  abandon off   (= -abandon)
    FF = cell(False, False)   # both off

    print("\n=== axis-1 / axis-6 entanglement: 2x2 factorial RPS "
          "(memory=True, L1 dev family) ===")
    print(f"            abandon=ON   abandon=OFF")
    print(f" goals=ON   {TT:>10.3f}   {TF:>10.3f}")
    print(f" goals=OFF  {FT:>10.3f}   {FF:>10.3f}")

    me_goals = ((TT + TF) / 2) - ((FT + FF) / 2)     # effect of having goals
    me_aband = ((TT + FT) / 2) - ((TF + FF) / 2)     # effect of abandonment
    inter = (TT - FT) - (TF - FF)                    # goals-effect difference
                                                     # across abandon levels
    print(f"\n main effect (goal-choice)      = {me_goals:+.3f}")
    print(f" main effect (abandonment)      = {me_aband:+.3f}")
    print(f" interaction (goals x abandon)  = {inter:+.3f}")
    big = max(abs(me_goals), abs(me_aband))
    ratio = abs(inter) / big if big else float('inf')
    print(f" |interaction| / max|main|      = {ratio:.2f}")
    if abs(me_goals) > 0.02 and abs(me_aband) > 0.02 and ratio < 0.5:
        verdict = ("SEPARABLE -- both ablations carry distinct, additive "
                   "capability signal; entanglement is benign and reportable")
    elif ratio >= 0.5:
        verdict = ("CONFOUNDED -- interaction dominates; -goal and -abandon "
                   "are not separable, must reframe as one combined axis")
    else:
        verdict = ("WEAK -- at least one ablation has little independent "
                   "effect on this family; reconsider that ablation")
    print(f" verdict: {verdict}")


if __name__ == "__main__":
    main()
