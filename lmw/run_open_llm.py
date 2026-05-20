"""
The key contamination-safe experiment: an LLM agent on Open-Ended LMW.

This is where axis-2 ACROSS stages can actually show: stage variables are
namespaced, so literal claim reuse does not transfer -- only an ABSTRACT
strategy can. LLM-mem carries a short self-written strategy memo across
stages; LLM-nomem resets it each stage. Question: does carrying an abstract
theory across the curriculum let the agent master more stages?

Synthetic + per-seed-infinite => contamination-immune; needs no private data.
Cheap: gpt-4o-mini, hard per-stage call cap. Run:
  set -a; . ../autodiscovery/.env.local; set +a
  ../autodiscovery/.venv/Scripts/python.exe run_open_llm.py
"""

from __future__ import annotations

import json
import os
import statistics as st

from claims import ClaimStore
from oracle import Oracle
from scm import SCM
from scorer import score
from world import BudgetExhausted, World
from openworld import (TIGHTNESS, MAX_STAGES, build_stage, _stage_view,
                       consistency_selftest)

MODEL = os.getenv("OPEN_LLM_MODEL", "openai/gpt-4o-mini")
CAP = int(os.getenv("OPEN_LLM_CAP", "12"))
SEEDS = list(range(1, 1 + int(os.getenv("OPEN_LLM_SEEDS", "3"))))
TAU = 0.15
_REASONING = ("gpt-5" in MODEL) or (MODEL.split("/")[-1][:1] == "o")

SYS = (
    "You are an autonomous research agent. Within a tight experiment budget, "
    "establish as many TRUE high-value facts as possible. observe(cluster) "
    "shows ASSOCIATION only; intervene(cause,target) reveals CAUSATION. A "
    "strongly-correlated pair can be NON-causal via a hidden common cause -> "
    "that is confounded(A,B,Z), assertable only after you have BOTH "
    "no_effect(A,B) (intervening A does not move B) AND a causal(T,B) for "
    "some other T. One cluster looks promising but every intervention is "
    "null: probe briefly, then abandon it. Claim grammar (exact): "
    "causal(A,B,+) | causal(A,B,-) | no_effect(A,B) | confounded(A,B,Z) | "
    "mean(V)~const. Reply EXACTLY ONE JSON object, no prose."
)
ACT = ('{"action":"observe","cluster":ID} | '
       '{"action":"intervene","cause":V,"target":V} | '
       '{"action":"claim","statement":S} | '
       '{"action":"abandon","cluster":ID} | {"action":"stop"}')


class _LLM:
    def __init__(self):
        from openai import OpenAI
        self.c = OpenAI()
        self.calls = 0

    def ask(self, sys_extra, user):
        self.calls += 1
        try:
            kw = {"model": MODEL,
                  "max_tokens": 1400 if _REASONING else 120,
                  "messages": [{"role": "system", "content": SYS + sys_extra},
                               {"role": "user", "content": user}]}
            if not _REASONING:
                kw["temperature"] = 0.2
            r = self.c.chat.completions.create(**kw)
            t = (r.choices[0].message.content or "").strip()
            if "```" in t:
                t = t.split("```")[1].lstrip("json").strip()
            return json.loads(t[t.find("{"):t.rfind("}") + 1])
        except Exception as e:
            print(f"[LLM] {e}")
            return None

    def memo(self, prev, transcript):
        self.calls += 1
        try:
            mkw = {"model": MODEL, "max_tokens": 1200 if _REASONING else 90}
            if not _REASONING:
                mkw["temperature"] = 0.3
            r = self.c.chat.completions.create(
                **mkw,
                messages=[{"role": "system", "content":
                           "Compress reusable STRATEGY (<=45 words) for future "
                           "unknown-but-similar tasks: how to find the real "
                           "trap among opaque clusters, what to abandon, the "
                           "confounded recipe. No task-specific variable names."},
                          {"role": "user", "content":
                           f"prev memo: {prev}\nthis run: {transcript[-700:]}"}])
            return (r.choices[0].message.content or "").strip()[:300]
        except Exception:
            return prev


def _prov(stmt, world):
    log = world.log
    s = stmt.replace(" ", "")
    def iv(c, t):
        return [e.eid for e in log if e.kind == "intervene"
                and c in e.interventions and t in e.measured]
    def ob(v):
        return [e.eid for e in log if e.kind == "observe" and v in e.measured]
    try:
        if s.startswith("causal(") or s.startswith("no_effect("):
            a, b = s[s.index("(") + 1:s.index(")")].split(",")[:2]
            return iv(a, b)[-2:]
        if s.startswith("confounded("):
            a, b, _ = s[s.index("(") + 1:s.index(")")].split(",")
            return (ob(a)[-1:] + iv(a, b)[-2:]) or iv(a, b)
        if s.startswith("mean("):
            return ob(s[s.index("(") + 1:s.index(")~")])[-1:]
    except Exception:
        return []
    return []


def _stage(llm, world, store, memo):
    clusters = dict(world._scm.structure.subdomains)
    ids = sorted(clusters)
    tr = []
    while world.budget_left > 1.0 and llm.calls < CAP:
        state = {"budget_left": round(world.budget_left, 1),
                 "clusters": {i: clusters[i] for i in ids},
                 "recent": tr[-5:]}
        extra = f"\nStrategy memo from prior tasks: {memo}" if memo else ""
        act = llm.ask(extra, f"STATE:{json.dumps(state)}\nACTIONS:{ACT}\n"
                             "Next single JSON action:")
        if not act or act.get("action") == "stop":
            break
        a = act.get("action")
        try:
            if a == "observe":
                vs = clusters.get(act.get("cluster"))
                if not vs:
                    tr.append("bad cluster"); continue
                e = world.observe(vs, n=8)
                cr = {f"{vs[i]}~{vs[j]}": round(e.corr(vs[i], vs[j]), 2)
                      for i in range(len(vs)) for j in range(i + 1, len(vs))}
                tr.append(f"obs {act['cluster']}: {cr}")
            elif a == "intervene":
                c, t = act.get("cause"), act.get("target")
                hi = world.intervene(c, +2.0, [t], n=6)
                lo = world.intervene(c, -2.0, [t], n=6)
                d = st.fmean(x[t] for x in hi.samples) - \
                    st.fmean(x[t] for x in lo.samples)
                tr.append(f"do({c})->{t}: d={round(d,2)} (|d|>~1 => causal)")
            elif a == "claim":
                stmt = str(act.get("statement", "")).replace(" ", "")
                store.assert_claim(stmt, 0.8, _prov(stmt, world),
                                   world.budget_total - world.budget_left)
                tr.append(f"claim {stmt}")
            elif a == "abandon":
                tr.append(f"abandon {act.get('cluster')}")
        except BudgetExhausted:
            break
    return "\n".join(tr)


def run(master_seed, carry_memo: bool):
    llm = _LLM()
    store = ClaimStore()
    memo = ""
    reached, total = 0, 0.0
    for k in range(MAX_STAGES):
        st_ = build_stage(master_seed, k)
        budget = round(Oracle(SCM(st_, 0)).reference_budget() * TIGHTNESS, 1)
        w = World(structure=st_, noise_seed=0, budget=budget)
        llm.calls = 0                                  # per-stage cap
        tlog = _stage(llm, w, store, memo if carry_memo else "")
        oc = Oracle(w._scm)
        rs = st_.schema.t_star if st_.schema.regime_shift else None
        sd = score(w, _stage_view(store, k), oc, regime_shift_at=rs)
        total += sd["RPS"]
        if carry_memo:
            memo = llm.memo(memo, tlog)
        if sd["RPS"] >= TAU:
            reached = k + 1
        else:
            break
    return reached, round(total, 3)


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    consistency_selftest()
    print(f"\n=== LLM on Open-Ended LMW (contamination-safe) | model={MODEL} "
          f"cap={CAP}/stage, {len(SEEDS)} seeds ===")
    print(f"{'variant':<12}{'avg_depth':>12}{'avg_total_RPS':>16}"
          f"{'depths':>14}")
    print("-" * 54)
    res, dump = {}, {"model": MODEL, "cap": CAP, "seeds": SEEDS}
    for nm, carry in (("LLM-mem", True), ("LLM-nomem", False)):
        per = []  # (seed, depth, total)
        for s in SEEDS:
            r, t = run(s, carry)
            per.append((s, r, t))
        ds = [p[1] for p in per]; ts = [p[2] for p in per]
        res[nm] = (st.fmean(ds), st.fmean(ts))
        dump[nm] = per
        print(f"{nm:<12}{st.fmean(ds):>12.2f}{st.fmean(ts):>16.3f}"
              f"{str(ds):>16}")
        print(f"   per-seed (seed,depth,total): {per}")
    import json as _j
    fn = f"open_llm_{MODEL.split('/')[-1]}_{len(SEEDS)}s.json"
    with open(fn, "w") as f:
        _j.dump(dump, f, indent=1)
    print(f" [written {fn}]")
    dm = res["LLM-mem"][0] - res["LLM-nomem"][0]
    tm = res["LLM-mem"][1] - res["LLM-nomem"][1]
    print(f"\n cross-stage axis-2 (carried strategy memo) value: "
          f"depth {dm:+.2f}, total_RPS {tm:+.3f}")
    print(" reading: >0 => carrying an abstract theory across the horizon "
          "helps a real agent go deeper (the project thesis, tested in the "
          "contamination-safe ceiling-free setting).")


if __name__ == "__main__":
    main()
