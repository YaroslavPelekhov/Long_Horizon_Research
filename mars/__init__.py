"""RG-HLI scientific-reasoning runtime.

The package exposes the proposal components and deterministic coordinator used
by the benchmark interfaces. The residual-guided hypothesis-language kernel
lives in :mod:`mars.induction`, while typed contracts, residuals, and operator
plans live in :mod:`mars.skills`.
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
