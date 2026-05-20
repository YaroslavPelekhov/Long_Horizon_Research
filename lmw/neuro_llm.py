"""
Step B on REAL data: an LLM agent against the cross-subject (LOSO) neuro
harness, behind the same domain-agnostic WorldSource.

Tests the project's core thesis on real data: does a persistent-theory
scaffold (axis 2) raise the cross-subject-replication RPS vs the same LLM with
that scaffold ablated, and vs scripted dummies?

The agent NEVER sees the held-out subject (that is the scorer's GT). Its only
signal is observe()'s per-feature Contrast incl. `stable` = the source's
honest cross-subject sign-consistency pre-registration on discovery subjects.

Cheap: gpt-4o-mini via OpenRouter, hard call cap. Run:
  set -a; . ../autodiscovery/.env.local; set +a
  ../autodiscovery/.venv/Scripts/python.exe neuro_llm.py
"""

from __future__ import annotations

import json
import os

from claims import AgendaItem, ClaimStore
from adapters.base import BudgetExhausted
from adapters.neuro import NeuroSource, SUBJECTS
from neuro_lab import NaiveAgent, StatAgent, score

SYS = (
    "You are an autonomous research agent exploring an unknown dataset via a "
    "fixed API. Goal: assert as many TRUE, high-value relationships as "
    "possible within a tight budget. observe(cluster) returns, per feature, an "
    "effect estimate with a boolean 'stable' = the effect is consistent on "
    "your discovery data (your only honest proxy for whether it will "
    "generalize). Only claim features with stable=true (claiming unstable "
    "ones is penalised). Some clusters are inert dead-ends (almost nothing "
    "stable, ~0 value) -> abandon them fast to save budget for productive "
    "clusters. Reply with EXACTLY ONE JSON object, no prose."
)
ACTIONS = ('{"action":"observe","cluster":ID} | '
           '{"action":"claim","feature":NAME,"sign":"+|-"} | '
           '{"action":"abandon","cluster":ID} | {"action":"stop"}')


class LLMNeuroAgent:
    def __init__(self, memory=True,
                 model=os.getenv("NEURO_LLM_MODEL", "openai/gpt-4o-mini"),
                 max_calls=int(os.getenv("NEURO_LLM_CALLS", "18"))):
        self.memory = memory
        self.model = model
        self.max_calls = max_calls
        self.calls = 0
        self.name = "LLM-mem" if memory else "LLM-nomem"
        from openai import OpenAI
        self._c = OpenAI()

    def _ask(self, user):
        self.calls += 1
        try:
            reasoning = "gpt-5" in self.model or self.model.split("/")[-1][:1] == "o"
            kw = {"model": self.model,
                  "max_tokens": int(os.getenv("NEURO_LLM_MAXTOK",
                                               "1400" if reasoning else "110")),
                  "messages": [{"role": "system", "content": SYS},
                               {"role": "user", "content": user}]}
            if not reasoning:
                kw["temperature"] = 0.2
            r = self._c.chat.completions.create(**kw)
            msg = r.choices[0].message.content
            if not msg:
                return None
            t = msg.strip()
            if "```" in t:
                t = t.split("```")[1].lstrip("json").strip()
            return json.loads(t[t.find("{"):t.rfind("}") + 1])
        except Exception as e:
            print(f"[LLM] {e}")
            return None

    def run(self, src, store: ClaimStore):
        clusters = src.clusters()
        ids = sorted(clusters)
        store.set_agenda([AgendaItem(c, "investigate", "llm") for c in ids],
                         src.budget_total - src.budget_left)
        # memory: persistent per-cluster stable features + status
        mem: dict[str, list[tuple[str, str, float]]] = {}
        status = {c: "new" for c in ids}
        last = None  # most recent observation (the only thing -nomem keeps)

        while src.budget_left >= 3.0 and self.calls < self.max_calls:
            if self.memory:
                view = {
                    "budget_left": round(src.budget_left, 1),
                    "clusters": {c: {"n": len(clusters[c]),
                                     "status": status[c]} for c in ids},
                    "stable_found": {c: [f"{n}{s}" for n, s, _ in v[:6]]
                                     for c, v in mem.items() if v},
                    "claims": [x.statement for x in store.active()][:12],
                }
            else:
                view = {"budget_left": round(src.budget_left, 1),
                        "clusters_ids": ids,
                        "last_observation": last}
            act = self._ask(f"STATE:{json.dumps(view,ensure_ascii=False)}\n"
                            f"ACTIONS:{ACTIONS}\nNext single JSON action:")
            if not act or act.get("action") == "stop":
                break
            a = act.get("action")
            try:
                if a == "observe":
                    cl = act.get("cluster")
                    if cl not in clusters:
                        continue
                    obs = src.observe(cl)
                    rows = sorted(
                        ((n, ("+" if c.delta > 0 else "-"),
                          round(abs(c.delta) / (c.se or 1e-9), 2), c.stable,
                          c.eid)
                         for n, c in obs.items()),
                        key=lambda r: -r[2])
                    stable = [(n, s, z) for (n, s, z, st, e) in rows if st]
                    mem[cl] = stable
                    status[cl] = "observed"
                    self._prov = {n: e for (n, s, z, st, e) in rows}
                    last = {"cluster": cl,
                            "stable": [f"{n}{s}(z{z})" for n, s, z in stable[:8]],
                            "n_stable": len(stable)}
                elif a == "claim":
                    f = act.get("feature"); sg = act.get("sign", "+")
                    eid = getattr(self, "_prov", {}).get(f)
                    prov = [eid] if eid is not None else []
                    stmt = f"stim_effect({f},{sg})"
                    if not any(x.statement == stmt and x.status == "active"
                               for x in store.claims):
                        store.assert_claim(stmt, 0.8, prov,
                                           src.budget_total - src.budget_left)
                elif a == "abandon":
                    cl = act.get("cluster")
                    status[cl] = "abandoned"
                    store.set_agenda([x for x in store.agenda
                                      if x.subdomain != cl],
                                     src.budget_total - src.budget_left)
            except BudgetExhausted:
                break


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    B = 45.0
    cols = ["RPS", "integrity_penalty", "axis1_agenda_ratio",
            "axis2_retention", "axis6_deadend_regret", "final_true_value",
            "oracle_value"]
    variants = {
        "Full(dummy)": lambda: StatAgent(True, True, True),
        "Naive":       NaiveAgent,
        "LLM-mem":     lambda: LLMNeuroAgent(memory=True),
        "LLM-nomem":   lambda: LLMNeuroAgent(memory=False),
    }
    print(f"\n=== Step B on REAL data: cross-subject LOSO, Smell_Betula "
          f"(4-fold avg) | LLM={os.getenv('NEURO_LLM_MODEL','openai/gpt-4o-mini')}"
          f" cap={os.getenv('NEURO_LLM_CALLS','18')} ===")
    head = f"{'variant':<12}" + "".join(f"{c:>20}" for c in cols)
    print(head); print("-" * len(head))
    res, calls = {}, {}
    for nm, fac in variants.items():
        acc, cc = {}, 0
        folds = SUBJECTS[:int(os.getenv("NEURO_LLM_FOLDS", str(len(SUBJECTS))))]
        for h in folds:
            src = NeuroSource(held_out=h, budget=B)
            store = ClaimStore()
            ag = fac()
            ag.run(src, store)
            cc += getattr(ag, "calls", 0)
            m = score(src, store)
            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + v / len(folds)
        res[nm] = {k: round(v, 4) for k, v in acc.items()}
        calls[nm] = cc
        print(f"{nm:<12}" + "".join(f"{res[nm][c]:>20}" for c in cols))
    lm, ln = res["LLM-mem"]["RPS"], res["LLM-nomem"]["RPS"]
    fd, nv = res["Full(dummy)"]["RPS"], res["Naive"]["RPS"]
    print(f"\n LLM memory-scaffold value (real data): {lm-ln:+.3f}  "
          f"(LLM-mem {lm} vs LLM-nomem {ln})")
    print(f" LLM-mem vs dummy Full {lm-fd:+.3f}   vs Naive {lm-nv:+.3f}")
    print(f" LLM calls total: mem={calls['LLM-mem']} nomem={calls['LLM-nomem']}"
          f"  (gpt-4o-mini)")
    print(f" thesis check (real): scaffold helps "
          f"{'YES' if lm > ln else 'NO'}; "
          f"real cross-subject signal exists "
          f"{'YES' if res['LLM-mem']['oracle_value'] > 1 else 'NO'}")


if __name__ == "__main__":
    main()
