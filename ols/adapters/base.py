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
from dataclasses import dataclass, field
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


@dataclass
class RubricSection:
    """One mandatory section of a completeness rubric (Programmatic Completeness
    Gate, MARS-SELF).

    The CodeEvolver coverage guard uses these to MECHANICALLY block premature
    submission / advance / halt until every required section has been
    discovered. This is the domain-agnostic generalization of UltraHorizon's
    genetics-specific coverage guard.

    Detection: a section is "covered" when ANY of `keywords` appears (case-
    insensitively) in the concatenated active-claim text, OR when `predicate`
    (if given) returns True. `predicate` receives a context dict:
        {"claims_text": str, "adapter": ResearchEnvAdapter,
         "budget_left": float, "current_subgoal": str | None}

    Enforcement scope:
      - required_from_subgoal=None  → required from the very first turn
      - required_from_subgoal="X"   → required only once the episode has reached
                                       sub-goal X (by declaration order). A
                                       terminal-submit check (current_subgoal=
                                       None passed to the guard) treats ALL
                                       sections as required.
    """
    name: str
    keywords: list[str] = field(default_factory=list)
    hint: str = ""
    required_from_subgoal: str | None = None
    predicate: Any = None              # optional Callable[[dict], bool]
    blocks_submit: bool = True         # missing → block terminal submit_*


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

    # -- completeness gate (MARS-SELF, optional) ------------------------------

    def completeness_rubric(self) -> list["RubricSection"]:
        """Declare the mandatory sections of a complete answer for this env.

        Default: empty → the Programmatic Completeness Gate is a no-op. Adapters
        that benefit from MARS-SELF's mechanical completeness enforcement
        (UltraHorizon genetics, NewtonBench law discovery, and scientific
        programs) override this to return their required sections.
        """
        return []

    def submit_action_names(self) -> set[str]:
        """Names of terminal submit actions the completeness gate should guard.

        Default covers the common conventions across our adapters.
        """
        return {
            "submit_report", "submit_law", "submit_hypothesis",
            "submit_program", "submit", "finalize",
        }

    def get_dataframes(self) -> dict:
        """Expose tabular data for the MARS-SELF AutoStatAnalyzer.

        Default: {} → no auto-statistics. Tabular adapters (DiscoveryBench)
        override this to return {name: pandas.DataFrame} so the cognitive
        exoskeleton can auto-run correlation / group-mean analysis the small
        model would otherwise have to hand-write.
        """
        return {}

    # -- SRHP: Self-Refuting Hypothesis Programs (MARS v0.4 core) --------------
    # The single unifying idea: a hypothesis is an EXECUTABLE program that
    # predicts observations. The engine runs conjecture → discriminate → refute
    # → mutate; the LLM only does the creative leap. Adapters that support SRHP
    # implement the five hooks below. `supports_srhp()` defaults False.

    def supports_srhp(self) -> bool:
        return False

    def srhp_spec(self) -> dict:
        """Describe the hypothesis-program interface for this task.

        Return dict with:
          fn_name      : str   — the function the model must define
          signature    : str   — full `def fn_name(...):` line
          description  : str    — natural-language task brief
          input_keys   : list   — names of the function's parameters
          output_desc  : str    — what the function must return
        """
        return {}

    def srhp_candidate_experiments(self, n: int = 16) -> list[dict]:
        """A pool of candidate experiment input-dicts to choose among. The
        engine picks the one on which the current hypothesis population
        disagrees most (the discriminating experiment)."""
        return []

    def srhp_run(self, inputs: dict) -> dict:
        """Run ONE real experiment with the given inputs; return observed
        outputs as a dict. Adapter decrements budget here."""
        return {}

    def srhp_error(self, predicted, observed) -> float:
        """Prediction error of a program's output vs the observation. Generic
        default: symmetric log-distance for scalars / mean over shared keys.
        Lower is better; inf for unusable predictions."""
        import math
        def _num(x):
            try:
                v = float(x)
                return v if math.isfinite(v) else None
            except Exception:
                return None
        # dict outputs → mean error over shared numeric keys
        if isinstance(observed, dict) and isinstance(predicted, dict):
            errs = []
            for k in observed:
                pe = self.srhp_error(predicted.get(k), observed.get(k))
                if math.isfinite(pe):
                    errs.append(pe)
            return sum(errs) / len(errs) if errs else float("inf")
        po, pp = _num(observed), _num(predicted)
        if po is None or pp is None:
            return float("inf")
        # symmetric log-error (scale-free), falls back to abs for non-positive
        if po > 0 and pp > 0:
            return abs(math.log(pp) - math.log(po))
        return abs(pp - po) / (abs(po) + 1e-9)

    def srhp_finalize(self, program_code: str, fn_name: str) -> None:
        """Convert the winning hypothesis program into the benchmark's native
        submission (e.g. call submit_law / set the report). Called once at the
        end of the SRHP loop."""
        return None


class BudgetExhausted(Exception):
    """Raised by adapter.execute() when no budget remains."""
