"""
DiscoveryBench adapter for MARS.

Thin re-export of the existing OLS DiscoveryBench adapter so MARS can drive
the same task surface. The OLS version already implements ResearchEnvAdapter
correctly (head / describe / query / submit_hypothesis actions, LLM-judge HMS
scoring). MARS layers Generator+Reflector+MemorySelector on top.

Used as Tier-2 convergent-validity check (H5): does the new multi-agent
architecture improve where OLS-v0.2 produced null at N=25?

NOTE: DiscoveryBench needs the cloned discoverybench_repo (see
ols.adapters.discoverybench_adapter.load_db_tasks). The same loader is
re-exported here as load_db_tasks for symmetry with the SAB adapter.
"""

from __future__ import annotations

# direct re-export — no rewrap needed. OLS's DiscoveryBenchAdapter already
# implements the same ResearchEnvAdapter ABC that MARS Coordinator uses.
from ols.adapters.discoverybench_adapter import (
    DBTask,
    DiscoveryBenchAdapter,
    load_db_tasks,
)

__all__ = ["DBTask", "DiscoveryBenchAdapter", "load_db_tasks"]
