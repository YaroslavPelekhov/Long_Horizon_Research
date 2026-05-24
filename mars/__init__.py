"""
MARS — Multi-Agent Research System.

A system that aims to beat published LLM baselines on:
  - MLAgentBench (Claude 3 Opus 37.5% target)
  - ScienceAgentBench (GPT-4o ReAct baseline)
  - DiscoveryBench (extends our OLS-v0.2 work)
  - MLRC-Bench (low-bar sanity check: gemini SoTA 9.3% gap closure)

Architecture (see MARS_SPEC_v0.md):
  Coordinator (non-LLM scheduler)
    Generator (LLM) — propose 1-2 actions/turn + claim, narrow context
    Reflector (LLM) — post-hoc critic (accept/require_evidence/retract/revise)
    MemorySelector (LLM or heuristic) — top-k claims for Generator context
    FutilityDetector (heuristic, reused from OLS)
    ClaimStore (reused from OLS)
    ResearchEnvAdapter (reused from OLS)

Critical design decisions from OLS ablations:
  - DROP AgendaController priority queue (showed harmful, +0.21 RPS on removal)
  - DROP "claim-gate" enforcement flag (~zero contribution alone)
  - DROP text-dump theory view (showed harmful, +0.24 RPS on removal)
  - KEEP FutilityDetector (only OLS module that worked as designed)
  - REPLACE single-LLM scaffold with Generator+Reflector dialogue
  - REPLACE "show all claims" with MemorySelector top-k
"""

from mars.agents.generator import Generator
from mars.agents.reflector import Reflector, ReflectorVerdict
from mars.agents.memory_selector import MemorySelector
from mars.coordinator import Coordinator, EpisodeReport

__version__ = "0.1.0"

__all__ = [
    "Generator",
    "Reflector",
    "ReflectorVerdict",
    "MemorySelector",
    "Coordinator",
    "EpisodeReport",
]
