"""
ResearchEnvAdapter — the only interface OLS sees of any benchmark.

This is the back-door-prevention seam: an adapter exposes a typed action space,
a budget, and a way to materialize sub-domains (or sub-questions, or sub-tasks).
It does NOT expose ground-truth structure of the environment, so OLS cannot
inadvertently learn benchmark-specific priors. The same OLSScaffold instance
should run unchanged against any concrete adapter.

Concrete adapters we ship:
  - LMWAdapter        — wraps lmw.world.World (5 SCMs)
  - DiscoveryBenchAdapter — wraps a DB-Real task (CSV + NL goal)

External adapters (for community submission, per protocol §8) implement this
ABC and the OLS overlay is portable to them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ols.core.types import (
    ActionSpec,
    Claim,
    ClaimVerdict,
    ExperimentResult,
)


@dataclass
class EnvHandle:
    """What an adapter hands the inner agent to describe the environment.

    Kept deliberately small and adapter-agnostic. Free-text `description` is
    the natural-language env brief; `subdomains` partitions the initial agenda;
    `actions` lists the typed actions the adapter supports.
    """
    description: str
    subdomains: list[tuple[str, str]]   # (sub_goal_id, question)
    actions: list[ActionSpec]
    budget_total: float


class ResearchEnvAdapter(ABC):
    """The thin, uniform surface OLS uses to drive any benchmark."""

    # -- environment info -----------------------------------------------------

    @abstractmethod
    def handle(self) -> EnvHandle:
        """Return the inner-agent-facing environment description."""

    @abstractmethod
    def budget_left(self) -> float:
        """Remaining budget. Adapter-defined units; OLS treats as opaque scalar."""

    def budget_spent(self) -> float:
        h = self.handle()
        return max(0.0, h.budget_total - self.budget_left())

    # -- execution ------------------------------------------------------------

    @abstractmethod
    def execute(self, action: str, args: dict) -> ExperimentResult:
        """Execute a typed action; return the experiment result.

        Adapter assigns a unique eid. Adapter is responsible for decrementing
        budget. Adapter may raise BudgetExhausted to signal the episode is over.
        """

    # -- ground truth (optional) ----------------------------------------------

    def verify_claim(self, claim: Claim) -> ClaimVerdict | None:
        """If the env has a ground-truth oracle, return a verdict.

        For LMW this dispatches to lmw.oracle.Oracle.is_true. For DiscoveryBench
        there is no per-claim oracle (only an end-of-episode LLM-judge), so this
        returns None and final scoring happens via score_episode().
        """
        return None

    # -- scoring --------------------------------------------------------------

    @abstractmethod
    def score_episode(
        self, claim_store_active: list[Claim], final_artifact: str | None = None
    ) -> dict:
        """Score the entire episode after OLS terminates.

        Returns a dict with at least one "primary" key (the headline metric for
        this env): RPS for LMW, HMS for DiscoveryBench. Additional diagnostic
        keys are adapter-specific and surfaced in the EpisodeReport.
        """


class BudgetExhausted(Exception):
    """Raised by adapter.execute() when no budget remains."""
