"""
Latent Mechanism World (LMW) v2 -- a FAMILY of schemas, not one shape.

A `Schema` parameterises the world shape:
  * n_clusters   : 2..4
  * chain_len    : length of the hidden causal chain t -> ... -> b (>=1)
  * n_latents    : 1.. (one confounds the trap; the rest power dead-ends)
  * real_decoys  : extra do-null vars hiding the true cause

Every generated instance keeps the invariants the metric depends on:
  TRAP cluster  : pair (a,b) correlated only via a latent (NO path a->b);
                  b's true root cause is `t`, reached through a chain of
                  `chain_len` hidden mediators, buried among `real_decoys`.
                  Facts: confounded(a,b,Z)=5, causal(t,b,netsign)=4,
                  no_effect(a,b)=3.
  FAKE cluster  : its own latent drives every var (looks just as promising)
                  but NO causal edges -> inert. Only mean(const)=1.
                  >=1 fake always exists and is placed BEFORE the trap in the
                  agent's traversal order (cluster ids; trap gets the last id)
                  so the dud-before-payoff dilemma holds for any schema.
  SIMPLE cluster: one real edge c->e. causal(c,e)=2.

Cluster ids are opaque ("C0"..); role is NOT encoded in the id (the agent
cannot tell trap from fake a priori -- nature hides the payoff behind a decoy).

Pure stdlib, deterministic given (structure_seed, noise_seed).
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Schema:
    name: str = "dev"
    n_clusters: int = 3
    chain_len: int = 1
    n_latents: int = 2
    real_decoys: int = 1
    regime_shift: bool = False
    t_star: float = 1e9          # set by harness relative to budget if used
    noise: float = 0.3

    def sig(self) -> tuple:
        return (self.n_clusters, self.chain_len, self.n_latents, self.real_decoys)


@dataclass
class Edge:
    src: str
    dst: str
    coeff: float
    flips: bool = False


@dataclass
class Structure:
    vars: list[str]
    latents: list[str]
    subdomains: dict[str, list[str]]      # opaque id -> vars
    roles: dict[str, str]                 # id -> 'trap'|'fake'|'simple'
    edges: list[Edge]
    spurious_pair: tuple[str, str]        # (a,b) in trap
    true_cause: tuple[str, str]           # (t,b) -- t is the chain root
    simple_edges: list[tuple[str, str, int]]
    fake_const: dict[str, str]            # fake id -> its const var
    schema: Schema
    noise: float

    def parents(self, v: str) -> list[Edge]:
        return [e for e in self.edges if e.dst == v]


def make_structure(structure_seed: int, schema: Schema | None = None) -> Structure:
    sc = schema or Schema()
    r = random.Random(structure_seed)
    nC = max(2, sc.n_clusters)

    # roles: exactly one trap; >=1 fake; rest simple
    n_fake = min(max(1, sc.n_latents - 1), nC - 1)
    roles_seq = ["fake"] * n_fake + ["simple"] * (nC - 1 - n_fake) + ["trap"]
    # ids C0..; trap LAST so sorted(ids) hits a fake first (dud before payoff)
    ids = [f"C{i}" for i in range(nC)]
    roles = {cid: role for cid, role in zip(ids, roles_seq)}
    trap_id = ids[-1]

    latents = [f"Z{i}" for i in range(max(1, sc.n_latents))]
    edges: list[Edge] = []
    subdomains: dict[str, list[str]] = {}
    fake_const: dict[str, str] = {}
    simple_edges: list[tuple[str, str, int]] = []

    vctr = 0

    def newvars(k: int) -> list[str]:
        nonlocal vctr
        out = [f"X{vctr + i}" for i in range(k)]
        vctr += k
        return out

    spurious_pair = true_cause = None
    fake_i = 0
    for cid in ids:
        role = roles[cid]
        if role == "trap":
            k = 2 + sc.chain_len + sc.real_decoys      # a,b + chain + decoys
            vs = newvars(k)
            r.shuffle(vs)
            a, b = vs[0], vs[1]
            chain = vs[2:2 + sc.chain_len]             # t = chain[0]
            t = chain[0]
            Z = latents[0]
            edges.append(Edge(Z, a, r.choice([-1, 1]) * r.uniform(1.2, 1.7)))
            edges.append(Edge(Z, b, r.choice([-1, 1]) * r.uniform(1.2, 1.7)))
            prev = chain[0]
            for nxt in chain[1:]:
                edges.append(Edge(prev, nxt,
                                  r.choice([-1, 1]) * r.uniform(1.0, 1.4)))
                prev = nxt
            edges.append(Edge(prev, b, r.choice([-1, 1]) * r.uniform(1.1, 1.5),
                              flips=sc.regime_shift))
            subdomains[cid] = vs
            spurious_pair, true_cause = (a, b), (t, b)
        elif role == "fake":
            lat = latents[(1 + fake_i) % len(latents)] if len(latents) > 1 \
                else latents[0]
            fake_i += 1
            vs = newvars(3)
            for v in vs:
                edges.append(Edge(lat, v,
                                  r.choice([-1, 1]) * r.uniform(1.2, 1.7)))
            subdomains[cid] = vs
            fake_const[cid] = r.choice(vs)
        else:  # simple
            vs = newvars(2)
            sign = r.choice([-1, 1])
            edges.append(Edge(vs[0], vs[1], sign * r.uniform(1.0, 1.5)))
            subdomains[cid] = vs
            simple_edges.append((vs[0], vs[1], sign))

    all_vars = [v for vs in subdomains.values() for v in vs]
    return Structure(
        vars=all_vars, latents=latents, subdomains=subdomains, roles=roles,
        edges=edges, spurious_pair=spurious_pair, true_cause=true_cause,
        simple_edges=simple_edges, fake_const=fake_const, schema=sc,
        noise=sc.noise,
    )


@dataclass
class SCM:
    structure: Structure
    noise_seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.noise_seed)
        self._order = self._topo_order()

    def _topo_order(self) -> list[str]:
        S = self.structure
        deps = {v: set() for v in S.vars}
        for ed in S.edges:
            if ed.src not in S.latents and ed.dst in deps:
                deps[ed.dst].add(ed.src)
        order, seen = [], set()
        while len(order) < len(S.vars):
            for v in S.vars:
                if v not in seen and deps[v] <= seen:
                    order.append(v); seen.add(v); break
            else:
                for v in S.vars:
                    if v not in seen:
                        order.append(v); seen.add(v)
        return order

    def _coeff(self, ed: Edge, budget_spent: float) -> float:
        if ed.flips and self.structure.schema.regime_shift \
           and budget_spent >= self.structure.schema.t_star:
            return -ed.coeff
        return ed.coeff

    def sample(self, interventions: dict[str, float], budget_spent: float) -> dict:
        S = self.structure
        nz = S.noise
        val: dict[str, float] = {lat: self._rng.gauss(0.0, 1.0)
                                 for lat in S.latents}
        consts = set(S.fake_const.values())
        for v in self._order:
            if v in interventions:
                val[v] = interventions[v]
                continue
            x = self._rng.gauss(2.0, 1.0) if v in consts else self._rng.gauss(0.0, 1.0)
            for ed in S.parents(v):
                x += self._coeff(ed, budget_spent) * val[ed.src]
            x += self._rng.gauss(0.0, nz)
            val[v] = x
        return {v: val[v] for v in S.vars}
