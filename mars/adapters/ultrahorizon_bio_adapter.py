"""
MARS adapter for UltraHorizon — Alien Genetics Lab environment.

The agent must discover hidden genetic-inheritance rules by conducting
crosses between triploid alien organisms and observing offspring phenotypes.

Hidden rules to discover (100-point rubric):
  A. Fundamental (25 pts): triploidy, meiosis (1n/2n gametes), viability
  B. Body size (35 pts): dosage effect, allele ID, quantitative values
  C. Color (10 pts): dominance hierarchy, complete dominance
  D. Shell (30 pts): cyclic dominance, lethal combination

Actions the generator can call:
  conduct_cross(parent1_id, parent2_id, num_offspring)
  query_organisms(start_id, end_id)
  get_lab_status()
  submit_report(content)          ← terminal; triggers LLM judge

ENV vars consumed by run_uh_bio.py:
  MARS_UH_SEEDS       comma-sep int seeds (default "42,43,44")
  MARS_UH_BUDGET      crosses per episode (default 20)
  MARS_UH_JUDGE_MODEL LLM for scoring (default openai/gpt-4o)
  MARS_GENERATOR_MODEL
  MARS_REFLECTOR_MODEL
  MARS_ABLATIONS
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
_UH_REPO = _PROJ / "ultrahorizon_repo"
if str(_UH_REPO) not in sys.path:
    sys.path.insert(0, str(_UH_REPO))

# Load API keys from project .env.local (same pattern as newtonbench_adapter)
try:
    from dotenv import load_dotenv
    _env_path = _PROJ / "autodiscovery" / ".env.local"
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass

from ols.adapters.base import BudgetExhausted, EnvHandle, ResearchEnvAdapter
from ols.core.types import ActionSpec, Claim, ExperimentResult

# ── lazy import so repo needn't be present at import time ──────────────────
def _import_env():
    from envs.bio_env.env import GeneticsLabEnvironment  # noqa: PLC0415
    from envs.common import Difficulty                    # noqa: PLC0415
    return GeneticsLabEnvironment, Difficulty


def _run(coro):
    """Run an async coroutine from synchronous context."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result()


@dataclass
class UHBioTask:
    seed: int
    budget: int
    judge_model: str

    def label(self) -> str:
        return f"uh_bio_seed{self.seed}_b{self.budget}"


JUDGE_PROMPT = """
You are an evaluator. Compare the agent's submitted description of genetic
inheritance rules against the scoring rubric.

<Scoring rubric>
## A. Fundamental Understanding (25 points)
1. Triploidy recognition — organism is triploid (three alleles per locus). [15 pts, Medium]
2. Meiosis process — gametes are 1n and 2n due to unequal segregation. [5 pts, Easy]
3. Viability constraint — only triploid zygotes survive; other ploidy = lethal. [5 pts, Easy]

## B. Body Size Trait — Dosage Effect (35 points)
4. Inheritance rule — body size follows additive dosage effect. [5 pts, Easy]
5. Allele identification — three distinct size alleles exist. [10 pts, Medium]
6. Quantitative description — approximate values: ~200, ~50, ~10 (±20%). [20 pts, Hard]

## C. Color Trait — Dominance Hierarchy (10 points)
7. Dominance hierarchy — strict order Red(C1) > Blue(C2) > White(C3). [5 pts, Easy]
8. Complete dominance — phenotype determined solely by dominant allele. [5 pts, Easy]

## D. Shell Trait — Cyclic Interaction & Lethality (30 points)
9. Cyclic dominance — Spiky(H1) > Smooth(H2) > Ridged(H3) > Spiky. [10 pts, Medium]
10. Lethal combination — H1+H2+H3 together is lethal. [20 pts, Hard]
</Scoring rubric>

<Agent submitted report>
{content}
</Agent submitted report>

Evaluation guidelines:
- Award FULL score only if the point is clearly and completely correct.
- Equivalent wording is accepted (e.g., "3 gene copies" ≡ "triploid").
- Vague/incomplete descriptions = 0.

Respond ONLY with valid JSON (no markdown fences):
{{
  "final_score": <0-100>,
  "score_breakdown": [
    {{"criterion": "Triploidy recognition", "max_score": 15, "awarded_score": <int>, "comment": "..."}},
    {{"criterion": "Meiosis process (1n/2n gametes)", "max_score": 5, "awarded_score": <int>, "comment": "..."}},
    {{"criterion": "Viability constraint (only triploid survives)", "max_score": 5, "awarded_score": <int>, "comment": "..."}},
    {{"criterion": "Body size: dosage effect", "max_score": 5, "awarded_score": <int>, "comment": "..."}},
    {{"criterion": "Body size: allele identification", "max_score": 10, "awarded_score": <int>, "comment": "..."}},
    {{"criterion": "Body size: quantitative values", "max_score": 20, "awarded_score": <int>, "comment": "..."}},
    {{"criterion": "Color: dominance hierarchy", "max_score": 5, "awarded_score": <int>, "comment": "..."}},
    {{"criterion": "Color: complete dominance", "max_score": 5, "awarded_score": <int>, "comment": "..."}},
    {{"criterion": "Shell: cyclic dominance", "max_score": 10, "awarded_score": <int>, "comment": "..."}},
    {{"criterion": "Shell: lethal combination", "max_score": 20, "awarded_score": <int>, "comment": "..."}}
  ]
}}
"""


def _call_judge(content: str, judge_model: str) -> dict:
    """Call our OpenRouter/OpenAI judge to score the submitted report."""
    from mars.agents.base import call_llm, make_openai_client, parse_json_strict  # noqa: PLC0415
    client = make_openai_client()
    prompt = JUDGE_PROMPT.format(content=content)
    raw = call_llm(
        client, judge_model,
        system="You are a precise evaluator of genetic inheritance rules.",
        user=prompt,
        max_tokens=1400,
        temperature=0.0,
    )
    return parse_json_strict(raw) or {}


# ── adapter ────────────────────────────────────────────────────────────────

_ACTIONS = [
    ActionSpec(
        name="conduct_cross",
        arg_schema={
            "parent1_id": "int — ID of first parent organism",
            "parent2_id": "int — ID of second parent organism",
            "num_offspring": "int (1-50, default 10) — offspring to produce",
        },
        cost_estimate=1.0,
        description=(
            "Cross two organisms to produce offspring and observe their phenotypes. "
            "Returns phenotype distribution, viability rate, lethal count."
        ),
    ),
    ActionSpec(
        name="query_organisms",
        arg_schema={
            "start_id": "int — first organism ID to examine",
            "end_id": "int (optional) — last organism ID to examine",
        },
        cost_estimate=0.0,
        description=(
            "Examine organisms by ID range to see phenotype, generation, parents."
        ),
    ),
    ActionSpec(
        name="get_lab_status",
        arg_schema={},
        cost_estimate=0.0,
        description="Return current resource usage and experiment count. No args.",
    ),
    ActionSpec(
        name="submit_report",
        arg_schema={
            "content": "str — full formal report describing all discovered inheritance rules",
        },
        cost_estimate=0.0,
        description=(
            "TERMINAL. Submit your final genetic-inheritance report for LLM scoring. "
            "Must include: ploidy level, meiosis mechanism, body-size rule + allele "
            "quantitative values, color dominance order, shell cyclic dominance + "
            "lethal combination. Call only when confident."
        ),
    ),
]


class UltraHorizonBioAdapter(ResearchEnvAdapter):
    """MARS adapter wrapping the UltraHorizon GeneticsLab environment."""

    def __init__(self, task: UHBioTask):
        GeneticsLabEnvironment, Difficulty = _import_env()

        # Monkeypatch load_judge_config so the env doesn't need judge_config.yaml
        GeneticsLabEnvironment.load_judge_config = lambda self: {
            "model": "dummy", "base_url": "http://dummy", "api_key": "dummy"
        }

        self._env = GeneticsLabEnvironment(
            seed=task.seed,
            required_steps=task.budget,
            difficulty=Difficulty.HARD,
            free=True,          # no minimum-step requirement; MARS manages budget
        )
        self._task = task
        self._budget_total = float(task.budget)
        self._submission: str | None = None
        self._eid = 0

    # ── ResearchEnvAdapter interface ──────────────────────────────────────

    def handle(self) -> EnvHandle:
        return EnvHandle(
            description=(
                "ALIEN GENETICS LABORATORY\n"
                "You are a researcher studying alien triploid organisms. "
                "Each organism has 3 traits (body_size, color, shell_shape), "
                "each controlled by 3 alleles. "
                "Your goal is to design crosses, observe offspring phenotypes, "
                "and discover the hidden inheritance rules governing each trait. "
                "Initial organisms:\n"
                "  ID 1 (Line A): body_size S1×3, color C1×3, shell H1×3\n"
                "  ID 2 (Line B): body_size S1×S2×S2, color C2×3, shell H2×3\n"
                "  ID 3 (Line C): body_size S3×3, color C3×3, shell H3×3\n"
                "Submit your full formal report with submit_report when ready."
            ),
            subdomains=[
                ("discover_inheritance_rules",
                 "Discover all genetic inheritance rules: ploidy, meiosis mechanism, "
                 "body-size (alleles + values), color (dominance), shell (cyclic + lethal).")
            ],
            actions=_ACTIONS,
            budget_total=self._budget_total,
        )

    def budget_left(self) -> float:
        if self._submission is not None:
            return 0.0
        return max(0.0, self._budget_total - self._env.current_experiments)

    def execute(self, action: str, args: dict) -> ExperimentResult:
        if self.budget_left() <= 0 and action != "submit_report":
            raise BudgetExhausted("Genetics lab budget exhausted.")

        self._eid += 1
        obs = self._dispatch(action, args)
        obs_text = json.dumps(obs, default=str)[:3000]

        return ExperimentResult(
            eid=self._eid,
            action=action,
            args=args,
            cost=1.0 if action == "conduct_cross" else 0.0,
            raw=obs,
            summary={"observation": obs_text},
        )

    def score_episode(self, claim_store_active: list[Claim],
                      final_artifact: str | None = None) -> dict:
        content = self._submission or final_artifact or ""
        if not content.strip() and claim_store_active:
            # Agent ran out of budget without submitting — score its accumulated claims
            content = (
                "Preliminary findings from experimental evidence (no formal report submitted):\n"
                + "\n".join(f"- {c.statement}" for c in claim_store_active)
            )
        if not content.strip():
            return {"primary": 0.0, "final_score": 0.0, "judge_result": {},
                    "submitted": content, "n_crosses": self._env.current_experiments}

        judge_out = _call_judge(content, self._task.judge_model)
        final_score = float(judge_out.get("final_score", 0.0))
        return {
            "primary": final_score / 100.0,   # normalised 0-1
            "final_score": final_score,
            "judge_result": judge_out,
            "submitted": content[:400],
            "n_crosses": self._env.current_experiments,
        }

    # ── internals ────────────────────────────────────────────────────────

    def _dispatch(self, action: str, args: dict) -> Any:
        if action == "conduct_cross":
            return _run(self._env.conduct_cross(
                parent1_id=int(args.get("parent1_id", 1)),
                parent2_id=int(args.get("parent2_id", 2)),
                num_offspring=int(args.get("num_offspring", 10)),
            ))
        elif action == "query_organisms":
            start = int(args.get("start_id", 1))
            end = args.get("end_id")
            end = int(end) if end is not None else None
            return _run(self._env.query_organisms(start_id=start, end_id=end))
        elif action == "get_lab_status":
            return _run(self._env.get_lab_status())
        elif action == "submit_report":
            content = str(args.get("content", ""))
            self._submission = content
            self._env.committed = True
            return {"success": True, "message": "Report received. Episode will be scored."}
        else:
            return {"error": f"Unknown action: {action}"}
