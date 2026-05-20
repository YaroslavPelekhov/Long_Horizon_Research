"""
The experiment API the agent interacts with (surrogate-time, budget-metered).

"A month" is NOT wall-clock: it is a budget B of experiment-units plus K
agenda-decision points. observe() is cheap, intervene() is medium. Every
experiment is logged with an id so claims can cite provenance and the harness
can force a re-test (re-run the cited experiments and check the statistic holds).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from scm import SCM, Structure


C_OBS = 0.5    # cost per observational sample-batch unit
C_INT = 1.0    # cost per interventional sample-batch unit


@dataclass
class Experiment:
    eid: int
    kind: str                       # "observe" | "intervene"
    interventions: dict[str, float]
    measured: list[str]
    n: int
    budget_at: float
    samples: list[dict[str, float]] = field(default_factory=list)

    # -- cheap statistics the agent may use -------------------------------
    def mean(self, var: str) -> float:
        return statistics.fmean(s[var] for s in self.samples)

    def corr(self, a: str, b: str) -> float:
        xs = [s[a] for s in self.samples]
        ys = [s[b] for s in self.samples]
        try:
            return statistics.correlation(xs, ys)
        except statistics.StatisticsError:
            return 0.0


class BudgetExhausted(Exception):
    pass


class World:
    """Agent-facing environment. The agent must never touch SCM directly."""

    def __init__(self, structure: Structure, noise_seed: int = 0,
                 budget: float = 60.0, decision_every: int = 6):
        self._scm = SCM(structure=structure, noise_seed=noise_seed)
        self.budget_total = budget
        self.budget_spent = 0.0
        self.decision_every = decision_every
        self.log: list[Experiment] = []
        self._eid = 0

    # -- introspection ---------------------------------------------------
    @property
    def budget_left(self) -> float:
        return self.budget_total - self.budget_spent

    def at_decision_point(self) -> bool:
        return self._eid > 0 and self._eid % self.decision_every == 0

    # -- core API --------------------------------------------------------
    def _run(self, kind: str, interventions: dict[str, float],
             measured: list[str], n: int) -> Experiment:
        cost = (C_OBS if kind == "observe" else C_INT) * n
        if cost > self.budget_left:
            raise BudgetExhausted(f"need {cost}, have {self.budget_left:.2f}")
        exp = Experiment(self._eid, kind, dict(interventions), list(measured), n,
                         self.budget_spent)
        for _ in range(n):
            exp.samples.append(self._scm.sample(interventions, self.budget_spent))
        self.budget_spent += cost
        self.log.append(exp)
        self._eid += 1
        return exp

    def observe(self, measured: list[str], n: int = 8) -> Experiment:
        """Passive observation (cheap). Cannot reveal causality -- only association."""
        return self._run("observe", {}, measured, n)

    def intervene(self, var: str, value: float, measured: list[str],
                  n: int = 8) -> Experiment:
        """do(var=value): the only way to learn causal structure (medium cost)."""
        return self._run("intervene", {var: value}, measured, n)

    # -- re-test (used by the harness, not normally by the agent) --------
    def retest_experiment(self, exp: Experiment, n: int | None = None) -> Experiment:
        """Re-run an experiment's protocol at CURRENT budget (catches regime shift /
        irreproducible claims). Does not consume budget -- harness audit only."""
        n = n or exp.n
        replay = Experiment(-1, exp.kind, dict(exp.interventions), list(exp.measured),
                            n, self.budget_spent)
        for _ in range(n):
            replay.samples.append(self._scm.sample(exp.interventions, self.budget_spent))
        return replay
