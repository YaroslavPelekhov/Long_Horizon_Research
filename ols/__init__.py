"""
OLS — Outer-Loop Scaffold.

A domain-agnostic overlay that wraps any inner-loop research agent (LLM-driven,
MCTS-based, tree-search) and provides the three capabilities every system in
the 2024-2026 autonomous-science wave was missing per the field map of the
companion preprint (Long_Horizon_Research_Preprint_DRAFT.md, §2):

  axis 1: autonomous agenda formation + revision   → AgendaController
  axis 2: persistent revisable theory               → ClaimStore (portable)
  axis 6: planning under delayed reward + abandon   → FutilityDetector

OLS sits between an inner-agent and any ResearchEnvAdapter (LMW, DiscoveryBench,
future external benchmarks). The inner agent only ever sees a typed action
interface; OLS owns the persistent state, the agenda priority, and the abandon
policy. This is the *minimal* faithful instantiation of "outer-loop competence"
that does not encode benchmark-specific priors in its strategy code (cf. the
Scripted-domain / Scripted-blind distinction in §5.2 of the preprint).
"""

from ols.core.types import Claim, AgendaItem, ExperimentResult, ActionSpec, ClaimVerdict
from ols.core.claim_store import ClaimStore
from ols.core.agenda import AgendaController, SubGoalStatus
from ols.core.abandon import FutilityDetector
from ols.adapters.base import ResearchEnvAdapter
from ols.scaffold import OLSScaffold, EpisodeReport

__version__ = "0.1.0"

__all__ = [
    "Claim",
    "AgendaItem",
    "ExperimentResult",
    "ActionSpec",
    "ClaimVerdict",
    "ClaimStore",
    "AgendaController",
    "SubGoalStatus",
    "FutilityDetector",
    "ResearchEnvAdapter",
    "OLSScaffold",
    "EpisodeReport",
]
