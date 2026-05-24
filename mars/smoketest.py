"""
MARS smoke test — exercises Coordinator end-to-end on LMW with a MOCK
Generator and a MOCK Reflector (no API calls, no $$). Catches wiring bugs
before paid bench runs.

Pass: 1 episode runs to completion across multiple ablation settings,
ClaimStore receives claims, no exceptions.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ / "lmw") not in sys.path:
    sys.path.insert(0, str(_PROJ / "lmw"))
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from ols.adapters.lmw_adapter import LMWAdapter

from mars.coordinator import Coordinator
from mars.agents.generator import Generator, GenContext, GenResponse
from mars.agents.reflector import Reflector, ReflectorContext, ReflectorResponse, ReflectorVerdict
from mars.agents.memory_selector import MemorySelector


class MockGenerator(Generator):
    """Deterministic generator for wiring tests — no API calls."""

    def __init__(self):
        # bypass parent __init__ (which calls OpenAI client)
        self.name = "MockGenerator"
        self.model = "mock"
        self.max_tokens = 0
        self.max_actions_per_turn = 2
        self.n_calls = 0
        self._turn = 0

    def propose(self, ctx: GenContext) -> GenResponse:
        self.n_calls += 1
        self._turn += 1
        import re
        vars_in_q = list(dict.fromkeys(re.findall(r"\bX\d+\b", ctx.sub_goal_question)))[:4]
        actions = []
        if vars_in_q and self._turn % 3 != 0:
            actions.append({"action": "observe",
                            "args": {"vars": vars_in_q, "n": 6}})
        if ctx.history_this_subgoal and len(vars_in_q) >= 2:
            actions.append({"action": "intervene",
                            "args": {"var": vars_in_q[0], "value": 2.0,
                                     "outcomes": [vars_in_q[1]], "n": 4}})
        claims = []
        if len(ctx.history_this_subgoal) >= 1 and len(vars_in_q) >= 2:
            claims.append({"op": "assert",
                           "statement": f"no_effect({vars_in_q[0]},{vars_in_q[1]})",
                           "confidence": 0.7})
        adv = len(ctx.history_this_subgoal) >= 2 and len(claims) > 0
        return GenResponse(actions=actions, claims=claims,
                           advance_subgoal=adv,
                           rationale=f"mock-turn-{self._turn}")


class MockReflector(Reflector):
    """Always accepts. For wiring test only."""

    def __init__(self):
        self.name = "MockReflector"
        self.model = "mock"
        self.max_tokens = 0
        self.n_calls = 0

    def review(self, ctx: ReflectorContext) -> ReflectorResponse:
        if ctx.last_actions or ctx.last_claims:
            self.n_calls += 1
        return ReflectorResponse(verdict=ReflectorVerdict.ACCEPT,
                                 reason="mock-accept")


def _build_lmw_world(seed=1):
    from scm import Schema, make_structure                                # type: ignore
    from world import World                                               # type: ignore
    sch = Schema(n_clusters=3, chain_len=2, n_latents=2,
                 real_decoys=1, regime_shift=False)
    structure = make_structure(seed, sch)
    return World(structure=structure, noise_seed=seed * 7919, budget=60.0)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("\n=== MARS smoke test (LMW adapter, mock Generator + Reflector) ===\n")
    abls = [
        ("MARS-ALL-ON",  dict(use_reflector=True,  use_memory_selector=True,  use_futility_detector=True)),
        ("MARS-no-ref",  dict(use_reflector=False, use_memory_selector=True,  use_futility_detector=True)),
        ("MARS-no-mem",  dict(use_reflector=True,  use_memory_selector=False, use_futility_detector=True)),
        ("MARS-no-fut",  dict(use_reflector=True,  use_memory_selector=True,  use_futility_detector=False)),
        ("MARS-ALL-OFF", dict(use_reflector=False, use_memory_selector=False, use_futility_detector=False)),
    ]
    for tag, abl in abls:
        world = _build_lmw_world(seed=1)
        adapter = LMWAdapter(world)
        gen = MockGenerator()
        ref = MockReflector()
        sel = MemorySelector(k=3)
        coord = Coordinator(
            adapter=adapter,
            generator=gen,
            reflector=ref,
            memory_selector=sel,
            seed=1,
            verbose=False,
            **abl,
        )
        rep = coord.run_episode()
        print(f"{tag:<13} RPS={rep.primary:+.3f} t={rep.n_turns:>2} "
              f"act={rep.n_actions:>2} cl={rep.n_claims_active}+{rep.n_claims_retracted}r "
              f"sg(done/aband)={rep.n_subgoals_done}/{rep.n_subgoals_abandoned} "
              f"verdicts={rep.verdict_counts} "
              f"g={rep.n_generator_calls} r={rep.n_reflector_calls} "
              f"wall={rep.wall_time_s:.2f}s")

    print("\nAll rows printed without exception → MARS wiring is sound.")


if __name__ == "__main__":
    main()
