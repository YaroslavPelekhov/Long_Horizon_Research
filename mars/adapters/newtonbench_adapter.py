"""
NewtonBench adapter for MARS.

NewtonBench (Chen et al., ICLR'26, arXiv 2510.07172) — 324 scientific-law-
discovery tasks across 12 physics domains × 3 difficulty tiers (easy/medium
/hard) × 3 law versions × system complexity (vanilla/simple/complex). Agent
discovers an unknown physical law via interactive experimentation.

Action space (verbatim from authors' agent harness):
  - run_experiment(experiments: list[dict])  — up to 20 input-parameter sets
    per round; environment returns the underlying-law output for each set
    (with optional noise injection); cost = 1 round.
  - submit_law(code: str)                     — submit a Python function
    with the module's FUNCTION_SIGNATURE; terminal, cost = 0.

Budget: 10 rounds (matches authors' default max_turns).

Scoring: re-uses the authors' own `module.evaluate_law()` which produces
exact_accuracy (the SA metric they report in the paper) + rmsle +
symbolic_equivalent (via their judge LLM). This gives directly-comparable
numbers to the published baselines.

Published baselines (average SA % across all 324 tasks):
  GPT-5: 75.9   Gemini-2.5-pro: 65.4   o4-mini: 47.8   DeepSeek-R1: 43.4
  Hard tier specifically:
  GPT-5: 87.5   Gemini-2.5-pro: 69.4   o4-mini: 52.8   DeepSeek-R1: 36.8

Repo: github.com/HKUST-KnowComp/NewtonBench (MIT). Clone alongside this
project as `newtonbench_repo/`.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ols.adapters.base import (
    BudgetExhausted,
    EnvHandle,
    ResearchEnvAdapter,
)
from ols.core.types import (
    ActionSpec,
    Claim,
    ClaimVerdict,
    ExperimentResult,
)


# 12 physics modules in NewtonBench
NB_MODULES = [
    "m0_gravity",
    "m1_coulomb_force",
    "m2_magnetic_force",
    "m3_fourier_law",
    "m4_snell_law",
    "m5_radioactive_decay",
    "m6_underdamped_harmonic",
    "m7_malus_law",
    "m8_sound_speed",
    "m9_kepler_third_law",
    "m10_kinetic_friction",
    "m11_centripetal_force",
]

# difficulty tiers
NB_DIFFICULTIES = ["easy", "medium", "hard"]

# law versions per difficulty
NB_LAW_VERSIONS = ["v0", "v1", "v2"]

# system complexity (vanilla = direct law evaluation; simple/complex = motion-
# simulation-based observation, which is harder)
NB_SYSTEMS = ["vanilla_equation"]   # v0.1: stick to vanilla; complex later


def _setup_nb_path() -> None:
    """Add the cloned newtonbench_repo to sys.path so we can `import modules.*`."""
    proj = Path(__file__).resolve().parent.parent.parent
    nb = proj / "newtonbench_repo"
    if not nb.exists():
        raise FileNotFoundError(
            f"Expected cloned NewtonBench at {nb}. Run "
            "`git clone https://github.com/HKUST-KnowComp/NewtonBench.git "
            "newtonbench_repo` in the project root."
        )
    sp = str(nb)
    if sp not in sys.path:
        sys.path.insert(0, sp)


@dataclass
class NBTask:
    module_name: str                    # one of NB_MODULES
    difficulty: str = "easy"            # easy / medium / hard
    law_version: str = "v0"             # v0 / v1 / v2
    system: str = "vanilla_equation"
    noise_level: float = 0.0
    trial_id: int = 0

    def label(self) -> str:
        return (f"{self.module_name}/{self.difficulty}/{self.law_version}/"
                f"{self.system}/n={self.noise_level}/t{self.trial_id}")


def enumerate_nb_tasks(
    modules: list[str] | None = None,
    difficulties: list[str] | None = None,
    law_versions: list[str] | None = None,
    systems: list[str] | None = None,
    noise_levels: list[float] | None = None,
    trials_per_combo: int = 1,
) -> list[NBTask]:
    """Enumerate task configurations. Default = all 12 × 3 × 3 = 108 vanilla-
    system tasks (×systems × noise_levels × trials = configurable)."""
    _setup_nb_path()
    mods = modules or NB_MODULES
    diffs = difficulties or NB_DIFFICULTIES
    lvs = law_versions or NB_LAW_VERSIONS
    sysz = systems or NB_SYSTEMS
    nls = noise_levels or [0.0]
    out: list[NBTask] = []
    for m in mods:
        for d in diffs:
            for lv in lvs:
                for sy in sysz:
                    for nl in nls:
                        for t in range(trials_per_combo):
                            out.append(NBTask(m, d, lv, sy, nl, t))
    return out


class NewtonBenchAdapter(ResearchEnvAdapter):
    """One NewtonBench task = one MARS episode."""

    def __init__(
        self,
        task: NBTask,
        budget: float = 10.0,
        judge_model: str | None = None,
    ):
        _setup_nb_path()
        # The judge_model name MUST be a key in NewtonBench's
        # api_source_mapping (see utils/call_llm_api.py). Default: gpt41
        # (gpt-4.1) which is in their mapping and on OpenRouter.
        self.task = task
        self._budget_total = float(budget)
        self._budget_spent = 0.0
        self._eid_next = 0
        self._submitted_law: str | None = None
        self._all_submitted_attempts: list[str] = []   # for fallback extraction
        self._judge_model = judge_model or os.environ.get(
            "MARS_NB_JUDGE_MODEL", "gpt41"
        )
        # lazy-import the physics module on demand
        self._module = importlib.import_module(f"modules.{task.module_name}")

    # -- handle ----

    def handle(self) -> EnvHandle:
        prompt = self._module.get_task_prompt(
            self.task.system, noise_level=self.task.noise_level
        )
        sig = self._module.FUNCTION_SIGNATURE
        desc = (
            "You are an AI research assistant tasked with discovering an "
            "unknown scientific law through interactive experimentation. The "
            "physical laws in this simulated universe may differ from those "
            "in our world.\n\n"
            f"TASK BRIEF FROM THE ENVIRONMENT:\n{prompt}\n\n"
            f"You must finish by submitting a Python function with signature:\n"
            f"    {sig}\n"
            f"that, when evaluated on test inputs, reproduces the underlying "
            f"law as closely as possible.\n\n"
            f"BUDGET: {int(self._budget_total)} experiment rounds. Per round you "
            f"may EITHER run a batch of experiments (up to 20 input sets, "
            f"`run_experiment`) OR submit the final law (`submit_law`, "
            f"terminal).\n\n"
            f"**CRITICAL RULES**:\n"
            f"  1. You MUST eventually call `submit_law` — that is the ONLY "
            f"action that yields a score. Running experiments without "
            f"submitting = SCORE 0.\n"
            f"  2. Use first 5-7 rounds to gather data via `run_experiment` "
            f"(batches of 5-15 inputs, varying parameters by orders of "
            f"magnitude). Use the LAST 1-2 rounds to call `submit_law`.\n"
            f"  3. If `budget_remaining` ≤ 3, STOP exploring and `submit_law` "
            f"NOW with your best guess.\n"
            f"  4. Your submitted code MUST be a single complete Python "
            f"function with the exact signature above, returning a float."
        )
        return EnvHandle(
            description=desc,
            subdomains=[(
                "discover",
                "Discover the underlying scientific law via interactive "
                "experimentation and submit it as a Python function."
            )],
            actions=[
                ActionSpec(
                    name="run_experiment",
                    arg_schema={
                        "experiments": (
                            "list[dict] — up to 20 input-parameter sets; each "
                            "dict provides the kwargs for one experiment "
                            "matching the function signature"
                        )
                    },
                    cost_estimate=1.0,
                    description=(
                        "Run a batch of experiments. Each input set in "
                        "`experiments` is evaluated under the unknown law and "
                        "the output is returned. Use this to gather data for "
                        "hypothesis testing. Costs 1 round regardless of "
                        "batch size."
                    ),
                ),
                ActionSpec(
                    name="submit_law",
                    arg_schema={
                        "code": (
                            "str — a complete Python function definition with "
                            "the exact signature shown above"
                        )
                    },
                    cost_estimate=0.0,
                    description=(
                        "Submit the final discovered law as a Python function. "
                        "This is the terminal action; once called, the "
                        "episode ends and the law is scored. Make sure your "
                        "function: (1) uses the exact signature, "
                        "(2) returns a single float, (3) handles edge cases "
                        "with sensible defaults."
                    ),
                ),
            ],
            budget_total=self._budget_total,
        )

    def budget_left(self) -> float:
        return max(0.0, self._budget_total - self._budget_spent)

    # -- execute ----

    def _next_eid(self) -> int:
        self._eid_next += 1
        return self._eid_next

    def execute(self, action: str, args: dict) -> ExperimentResult:
        if self.budget_left() <= 0:
            raise BudgetExhausted("NewtonBench budget exhausted")

        eid = self._next_eid()

        if action == "run_experiment":
            experiments = args.get("experiments") or []
            if not isinstance(experiments, list):
                experiments = []
            experiments = experiments[:20]                      # cap per round
            results: list[Any] = []
            for exp in experiments:
                if not isinstance(exp, dict):
                    results.append(f"error: each experiment must be a dict")
                    continue
                try:
                    r = self._module.run_experiment_for_module(
                        noise_level=self.task.noise_level,
                        difficulty=self.task.difficulty,
                        system=self.task.system,
                        law_version=self.task.law_version,
                        **exp,
                    )
                    if self.task.system == "vanilla_equation":
                        try:
                            r = "{:.15e}".format(float(r))
                        except (TypeError, ValueError):
                            pass
                    results.append(r)
                except Exception as e:
                    results.append(f"error: {type(e).__name__}: {str(e)[:120]}")
            self._budget_spent += 1.0
            summary = {
                "n_experiments": len(experiments),
                "outputs": results,
            }
            return ExperimentResult(
                eid=eid, action="run_experiment", args=args, cost=1.0,
                raw=None, summary=summary,
            )

        if action == "submit_law":
            code = str(args.get("code", "")).strip()
            # Strip wrapping <final_law>...</final_law> or code fences if the
            # Generator copied NewtonBench's XML-style tags from the task prompt
            import re as _re
            m = _re.search(r"<final_law>(.*?)</final_law>", code,
                           flags=_re.DOTALL | _re.IGNORECASE)
            if m:
                code = m.group(1).strip()
            if code.startswith("```"):
                code = code.strip("` \n")
                if code.lower().startswith("python"):
                    code = code[6:].lstrip("\n")
            self._submitted_law = code
            self._all_submitted_attempts.append(code)
            return ExperimentResult(
                eid=eid, action="submit_law", args={},
                cost=0.0, raw=None,
                summary={
                    "law_len": len(code),
                    "law_preview": code[:300],
                    "note": "Final law submitted — episode will end at next "
                            "Coordinator check.",
                },
            )

        # unknown action — burn no budget
        return ExperimentResult(
            eid=eid, action=action, args=args, cost=0.0,
            raw=None, summary={"error": f"unknown action: {action}"},
        )

    # -- ground-truth (no per-claim oracle; episode-level law eval) ----

    def verify_claim(self, claim: Claim) -> ClaimVerdict | None:
        return None

    # -- scoring ----

    def score_episode(self, claim_store_active, final_artifact=None) -> dict:
        if not self._submitted_law:
            return {
                "primary": 0.0, "SA": 0.0, "rmsle": float("nan"),
                "symbolic_equivalent": False,
                "submitted": "", "explain": "(no law submitted)",
            }

        # Load .env so their call_llm_api works for the symbolic-equiv judge
        try:
            from dotenv import load_dotenv
            proj = Path(__file__).resolve().parent.parent.parent
            env_path = proj / "autodiscovery" / ".env.local"
            if env_path.exists():
                load_dotenv(env_path, override=False)
        except Exception:
            pass

        try:
            param_desc = getattr(self._module, "PARAM_DESCRIPTION", "")
            ev = self._module.evaluate_law(
                self._submitted_law,
                param_description=param_desc,
                difficulty=self.task.difficulty,
                law_version=self.task.law_version,
                judge_model_name=self._judge_model,
            )
        except Exception as e:
            return {
                "primary": 0.0, "SA": 0.0, "rmsle": float("nan"),
                "symbolic_equivalent": False,
                "submitted": self._submitted_law[:500],
                "explain": f"eval error: {type(e).__name__}: {str(e)[:300]}",
            }

        sa = float(ev.get("exact_accuracy", 0.0))
        return {
            "primary": sa,
            "SA": sa,
            "rmsle": ev.get("rmsle", float("nan")),
            "symbolic_equivalent": bool(ev.get("symbolic_equivalent", False)),
            "symbolic_msg": str(ev.get("symbolic_msg", ""))[:300],
            "submitted": self._submitted_law[:500],
            "explain": str(ev.get("symbolic_msg", ""))[:300],
        }
