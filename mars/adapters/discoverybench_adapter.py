"""DiscoveryBench observation and native-evaluator interface.

The retained loader and task surface are shared with the compatibility layer,
then exposed through the same adapter contract used by the RG-HLI runtime.
DiscoveryBench must be available at ``discoverybench_repo/``; the loader is
re-exported as :func:`load_db_tasks`.
"""

from __future__ import annotations

# Direct re-export: the compatibility adapter implements the runtime interface.
from ols.adapters.discoverybench_adapter import (
    DBTask,
    DiscoveryBenchAdapter,
    load_db_tasks,
)

__all__ = ["DBTask", "DiscoveryBenchAdapter", "load_db_tasks"]
