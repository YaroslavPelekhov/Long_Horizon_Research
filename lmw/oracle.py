"""
Oracle (v2): truth + D* derived from ANY schema's Structure.

Causality is PATH-based with a net sign (chains of length > 1 supported):
causal(A,B,s) is true iff a directed observable path A=>B exists and the
product of edge signs along it equals s (regime flip applied at t_star).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scm import SCM, Edge, Structure

OBS_U = 4.0
EFF_U = 12.0

_CAUSAL = re.compile(r"causal\(([^,]+),([^,]+),([+-])\)")
_NOEFF = re.compile(r"no_effect\(([^,]+),([^,]+)\)")
_CONF = re.compile(r"confounded\(([^,]+),([^,]+),Z\)")
_MEAN = re.compile(r"mean\(([^,]+)\)~const")


@dataclass
class Fact:
    statement: str
    subdomain: str
    value: float
    min_cost: float
    min_horizon: int


class Oracle:
    def __init__(self, scm: SCM):
        self.S: Structure = scm.structure
        self._scm = scm
        self._adj: dict[str, list[Edge]] = {}
        for e in self.S.edges:
            if e.src not in self.S.latents:
                self._adj.setdefault(e.src, []).append(e)

    # -- graph ----------------------------------------------------------
    def _path_sign(self, src: str, dst: str, budget: float):
        """Return net sign (+1/-1) of a directed path src=>dst, or None."""
        stack = [(src, 1)]
        seen = set()
        while stack:
            node, sgn = stack.pop()
            if node == dst:
                return sgn
            if node in seen:
                continue
            seen.add(node)
            for e in self._adj.get(node, []):
                c = e.coeff
                if e.flips and self.S.schema.regime_shift and budget >= self.S.schema.t_star:
                    c = -c
                stack.append((e.dst, sgn * (1 if c > 0 else -1)))
        return None

    def _has_path(self, a: str, b: str) -> bool:
        return self._path_sign(a, b, 0.0) is not None

    def _shared_latent(self, a: str, b: str) -> bool:
        for lat in self.S.latents:
            if any(e.src == lat and e.dst == a for e in self.S.edges) and \
               any(e.src == lat and e.dst == b for e in self.S.edges):
                return True
        return False

    def subdomain_of_var(self, v: str) -> str | None:
        for sd, members in self.S.subdomains.items():
            if v in members:
                return sd
        return None

    def fake_subdomains(self) -> list[str]:
        return [sd for sd, r in self.S.roles.items() if r == "fake"]

    def deadend_subdomain(self) -> str:
        return self.fake_subdomains()[0]

    # -- truth ----------------------------------------------------------
    def is_true(self, stmt: str, budget_spent: float = 0.0) -> bool:
        m = _CAUSAL.fullmatch(stmt)
        if m:
            A, B, sgn = m.group(1), m.group(2), 1 if m.group(3) == "+" else -1
            s = self._path_sign(A, B, budget_spent)
            return s is not None and s == sgn
        m = _NOEFF.fullmatch(stmt)
        if m:
            return not self._has_path(m.group(1), m.group(2))
        m = _CONF.fullmatch(stmt)
        if m:
            a, b = m.group(1), m.group(2)
            return (self._shared_latent(a, b)
                    and not self._has_path(a, b) and not self._has_path(b, a))
        m = _MEAN.fullmatch(stmt)
        if m:
            return m.group(1) in set(self.S.fake_const.values())
        return False

    # -- D* -------------------------------------------------------------
    def facts(self) -> list[Fact]:
        a, b = self.S.spurious_pair
        t, _ = self.S.true_cause
        cl = self.S.schema.chain_len
        nd = self.S.schema.real_decoys
        sd_trap = self.subdomain_of_var(b)
        tsign = "+" if self._path_sign(t, b, 0.0) == 1 else "-"
        fs = [
            Fact(f"confounded({a},{b},Z)", sd_trap, 5.0,
                 OBS_U + EFF_U + (nd + cl) * EFF_U, 2 + nd + cl),
            Fact(f"causal({t},{b},{tsign})", sd_trap, 4.0,
                 OBS_U + (nd + cl) * EFF_U, nd + cl),
            Fact(f"no_effect({a},{b})", sd_trap, 3.0, OBS_U + EFF_U, 2),
        ]
        for (c, e, sg) in self.S.simple_edges:
            fs.append(Fact(f"causal({c},{e},{'+' if sg == 1 else '-'})",
                           self.subdomain_of_var(c), 2.0, EFF_U, 1))
        for fid, cv in self.S.fake_const.items():
            fs.append(Fact(f"mean({cv})~const", fid, 1.0, OBS_U, 0))
        return fs

    def fact_value(self, stmt: str) -> float:
        for f in self.facts():
            if f.statement == stmt:
                return f.value
        return 0.0

    def max_value(self) -> float:
        return sum(f.value for f in self.facts())

    def line_value_curve(self, subdomain: str, budget: float) -> float:
        fs = sorted((f for f in self.facts() if f.subdomain == subdomain),
                    key=lambda f: f.min_cost)
        return sum(f.value for f in fs if budget >= f.min_cost)

    def deadend_point(self, subdomain: str | None = None) -> float:
        return OBS_U + EFF_U          # one obs + one diagnostic test = enough

    def optimal_value(self, budget: float) -> float:
        fs = sorted(self.facts(), key=lambda f: f.value / f.min_cost, reverse=True)
        spent = val = 0.0
        for f in fs:
            if spent + f.min_cost <= budget:
                spent += f.min_cost
                val += f.value
        return val

    def reference_budget(self) -> float:
        """Budget that just affords the GOOD path (crack trap + all simple +
        a bounded probe of each fake) and nothing wasteful -> keeps the tight-
        budget pressure constant across schemas."""
        cl, nd = self.S.schema.chain_len, self.S.schema.real_decoys
        trap = OBS_U + EFF_U + (nd + cl) * EFF_U
        simple = len(self.S.simple_edges) * EFF_U
        fakes = len(self.fake_subdomains()) * (OBS_U + 2 * EFF_U)
        return trap + simple + fakes
