"""Autonomous research loop for MARS.

This package is the top-level self-research layer: it scans benchmark papers,
local protocols, run summaries, and environment blockers, then proposes the
next falsifiable self-improvement experiment.
"""

from mars.autoresearch.core import ResearchCycle, ResearchState

__all__ = ["ResearchCycle", "ResearchState"]
