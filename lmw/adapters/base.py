"""
WorldSource -- the domain-agnostic boundary.

The metric (RPS), the counterfactual harness, the agent, and ClaimStore only
ever talk to THIS interface. A data source (synthetic SCM, or the NeuroTrend
real recordings, or a future physics/bio source) is just a plugin behind it.
That is the structural anti-hardcoding guarantee: the scorer/agent never see
neuro-specific fields, only the generic methods below.

Claim grammar the agent emits (parsed by the source's truth oracle):
    stim_effect(<feature>,+|-)   feature shifts with the exogenous stimulus
    assoc(<featA>,<featB>,+|-)   two features co-vary

Ground truth is NOT a known SCM here: a claim is "true" iff it REPLICATES on a
held-out slice the agent never sees (out-of-sample). That is the whole point
of using real, uncontaminated data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Contrast:
    """Discovery-split statistic for one feature under the stimulus contrast."""
    delta: float          # mean(stimulus-on) - mean(stimulus-off)
    se: float             # pooled standard error
    eid: int              # experiment id (provenance)
    # split-half stability on the DISCOVERY data only (agent's own honest
    # guess at replicability; it must NOT peek at the held-out slice)
    stable: bool


class WorldSource(ABC):
    name: str

    # -- introspection --------------------------------------------------
    @abstractmethod
    def clusters(self) -> dict[str, list[str]]:
        """Investigable groups of variables (opaque ids -> feature names)."""

    @property
    @abstractmethod
    def budget_left(self) -> float: ...

    @property
    @abstractmethod
    def budget_total(self) -> float: ...

    # -- agent-facing experiment ---------------------------------------
    @abstractmethod
    def observe(self, cluster: str) -> dict[str, Contrast]:
        """Run the stimulus contrast for every feature in `cluster` on the
        DISCOVERY split. Costs budget. Raises BudgetExhausted when broke."""

    # -- ground truth (scorer/harness only; agent never calls these) ---
    @abstractmethod
    def replicates(self, statement: str) -> bool:
        """True iff the claimed relationship holds on the HELD-OUT slice."""

    @abstractmethod
    def value(self, statement: str) -> float:
        """Scientific value weight (neural stimulus effect > assoc > 0)."""

    @abstractmethod
    def max_value(self) -> float: ...

    @abstractmethod
    def oracle_value(self) -> float:
        """Total replicable value attainable (axis-1 normalizer)."""

    @abstractmethod
    def deadend_clusters(self) -> list[str]:
        """Clusters whose claims carry ~0 value (natural sunk-cost trap)."""

    @abstractmethod
    def provenance_ok(self, statement: str, prov: list[int]) -> bool: ...


class BudgetExhausted(Exception):
    pass
