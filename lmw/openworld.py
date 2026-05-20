"""
Open-Ended LMW v0 -- truth frozen, horizon moves.

The world is an infinite deterministic curriculum of progressively harder
stages. Stage k's structure is a pure function of (master_seed, k) (sha256
seed -> the validated v2 generator), variable names namespaced `s{k}.` so a
persistent theory across stages stays unambiguous. The agent advances k->k+1
ONLY if it mastered stage k (per-stage axis1 ratio >= tau, scored by the
penalty-robust v2 scorer against the full known SCM). Competence gates *when*
a stage is scored, never *what is true* -> consistency invariant holds, no L2
redux. Pure stdlib, deterministic, no API.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

from agents import ConnectorAgent
from claims import ClaimStore
from oracle import Oracle
from scm import SCM, Edge, Schema, Structure, make_structure
from scorer import score
from world import World

TIGHTNESS = 1.05
MAX_STAGES = 4


def _h(master_seed: int, k: int) -> int:
    return int(hashlib.sha256(f"{master_seed}:{k}".encode()).hexdigest()[:8], 16)


def _schema_k(k: int) -> Schema:
    """Monotone difficulty schedule (within v2 generator limits)."""
    return Schema(
        name=f"stage{k}",
        n_clusters=3 if k < 3 else 4,
        chain_len=1 + min(k, 3),
        n_latents=2 + (1 if k >= 2 else 0),
        real_decoys=1 + (1 if k >= 3 else 0),
        regime_shift=(k >= 2),
    )


def _namespace(st: Structure, k: int) -> Structure:
    p = lambda n: f"s{k}.{n}"
    return Structure(
        vars=[p(v) for v in st.vars],
        latents=[p(z) for z in st.latents],
        subdomains={cid: [p(v) for v in vs] for cid, vs in st.subdomains.items()},
        roles=dict(st.roles),
        edges=[Edge(p(e.src), p(e.dst), e.coeff, e.flips) for e in st.edges],
        spurious_pair=(p(st.spurious_pair[0]), p(st.spurious_pair[1])),
        true_cause=(p(st.true_cause[0]), p(st.true_cause[1])),
        simple_edges=[(p(c), p(e), s) for (c, e, s) in st.simple_edges],
        fake_const={cid: p(v) for cid, v in st.fake_const.items()},
        schema=st.schema,
        noise=st.noise,
    )


def build_stage(master_seed: int, k: int) -> Structure:
    """Pure function of (master_seed, k). Independent of agent / call order."""
    sc = _schema_k(k)
    st = _namespace(make_structure(_h(master_seed, k), sc), k)
    if sc.regime_shift:
        budget = Oracle(SCM(st, 0)).reference_budget() * TIGHTNESS
        st = _namespace(make_structure(_h(master_seed, k),
                                       replace(sc, t_star=round(0.55 * budget, 1))), k)
    return st


def _stage_view(store: ClaimStore, k: int) -> ClaimStore:
    v = ClaimStore()
    v.claims = [c for c in store.claims if f"s{k}." in c.statement]
    return v


def run_curriculum(make_agent, master_seed: int, persistent: bool,
                   tau: float = 0.15, max_stages: int = MAX_STAGES) -> dict:
    # Mastery gate uses the full validated metric (RPS) -- which already
    # integrates integrity + efficiency and is penalty-robust -- NOT a raw
    # sub-component like axis1 ratio (a low-RPS imposed-agenda agent could
    # otherwise pass the gate cheaply -> depth would not track competence).
    # tau-sensitivity is in the validity protocol (follow-up, not tuned here).
    store = ClaimStore()
    reached, total, per = 0, 0.0, []
    for k in range(max_stages):
        st = build_stage(master_seed, k)
        budget = round(Oracle(SCM(st, 0)).reference_budget() * TIGHTNESS, 1)
        world = World(structure=st, noise_seed=0, budget=budget)
        if not persistent:                       # -mem: theory wiped at horizon
            store = ClaimStore()
        make_agent().run(world, store)
        oc = Oracle(world._scm)
        rs = st.schema.t_star if st.schema.regime_shift else None
        sd = score(world, _stage_view(store, k), oc, regime_shift_at=rs)
        per.append((k, round(sd["RPS"], 3), round(sd["axis1_agenda_ratio"], 3)))
        total += sd["RPS"]
        if sd["RPS"] >= tau:
            reached = k + 1
        else:
            break
    return {"reached": reached, "total_rps": round(total, 3), "per": per}


def _sig(st: Structure):
    return (tuple(sorted(st.vars)),
            tuple(sorted(st.roles.items())),
            st.spurious_pair, st.true_cause,
            tuple(sorted((e.src, e.dst, round(e.coeff, 6), e.flips)
                         for e in st.edges)))


def consistency_selftest() -> None:
    """L2 insurance: stage-k truth must be identical regardless of repeated
    construction, and regardless of how many other stages were built first."""
    for seed in (1, 7, 42):
        for k in (0, 1, 2, 3):
            a = _sig(build_stage(seed, k))
            b = _sig(build_stage(seed, k))                 # rebuild
            # build later stages first, then k (order independence)
            _ = [build_stage(seed, j) for j in (3, 2, 1, 0)]
            c = _sig(build_stage(seed, k))
            if not (a == b == c):
                raise AssertionError(
                    f"CONSISTENCY VIOLATION at seed={seed} k={k} "
                    "-- truth depends on construction order (L2 redux). ABORT.")
    print("consistency_selftest: PASS "
          "(stage truth = pure fn of (seed,k); order-independent)")


if __name__ == "__main__":
    consistency_selftest()
    print(run_curriculum(lambda: ConnectorAgent(True, True, True),
                         master_seed=1, persistent=True))
