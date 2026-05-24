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
import io
import os
import sys
import threading
from contextlib import redirect_stdout
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


def _safe_python_exec(code: str, timeout_s: float = 8.0,
                      max_output_chars: int = 2000) -> str:
    """Execute Python code in a restricted namespace; return captured stdout.

    Allowed: numpy, scipy, math, statistics, json, regex. Blocked: subprocess,
    os, sys, eval, exec, import-statement of disallowed modules.

    This is a controlled science-discovery sandbox, NOT a fully hardened
    sandbox — assumes the inner LLM is non-adversarial. Worst-case bad code
    just times out or returns an error string.
    """
    code = (code or "").strip()
    if not code:
        return "(empty code)"
    # Lightweight blacklist
    for bad in ("subprocess", "os.system", "os.popen", "os.remove",
                "os.removedirs", "shutil", "__import__('os')",
                "__import__('subprocess')", "open(", "eval(", "exec("):
        if bad in code:
            return f"ERROR: disallowed token: {bad}"

    import numpy as np
    import math
    import statistics as st_stats
    try:
        from scipy import optimize as _sp_opt
        from scipy import stats as _sp_stats
        HAVE_SCIPY = True
    except Exception:
        HAVE_SCIPY = False

    ns: dict[str, Any] = {
        "np": np, "numpy": np, "math": math, "statistics": st_stats,
        "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
        "len": len, "range": range, "list": list, "dict": dict, "set": set,
        "tuple": tuple, "float": float, "int": int, "str": str, "bool": bool,
        "zip": zip, "enumerate": enumerate, "sorted": sorted, "print": print,
        "any": any, "all": all, "map": map, "filter": filter,
    }
    if HAVE_SCIPY:
        ns["optimize"] = _sp_opt
        ns["scipy_optimize"] = _sp_opt
        ns["scipy_stats"] = _sp_stats

    out_buf = io.StringIO()
    out: dict[str, Any] = {}

    def worker():
        try:
            with redirect_stdout(out_buf):
                exec(code, {"__builtins__": ns}, ns)
            out["ok"] = True
        except Exception as e:
            out["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        return f"(timeout > {timeout_s}s)"
    if "err" in out:
        captured = out_buf.getvalue()
        msg = f"ERROR: {out['err']}"
        if captured:
            msg += f"\n--- partial stdout ---\n{captured[:max_output_chars]}"
        return msg
    captured = out_buf.getvalue().strip()
    if not captured:
        return "(no stdout; remember to print() results)"
    if len(captured) > max_output_chars:
        captured = captured[:max_output_chars] + "...(truncated)"
    return captured


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
        self._action_log: list[dict] = []              # for synthesis fallback
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
            f"BUDGET: {int(self._budget_total)} rounds. Per round, ONE action: "
            f"`run_experiment` (gather data), `python_exec` (analyze data), or "
            f"`submit_law` (terminal, no cost). The available action set is "
            f"listed below.\n\n"
            f"**RECOMMENDED WORKFLOW** (this is a metaphysical-shift universe — "
            f"DO NOT assume standard textbook physics!):\n"
            f"  Rounds 1-3: `run_experiment` with input parameters varied by "
            f"orders of magnitude (e.g. distance ∈ [0.1, 1, 10, 100, 1000]) — "
            f"this lets you compute log-log slopes.\n"
            f"  Round 4: `python_exec` — do log-log linear regression on the "
            f"gathered data to find the scaling exponents. Example:\n"
            f"      import numpy as np\n"
            f"      r = np.array([...])  # distances you tried\n"
            f"      F = np.array([...])  # forces measured\n"
            f"      slope, intercept = np.polyfit(np.log(r), np.log(F), 1)\n"
            f"      print('exponent on r:', slope, 'log-intercept:', intercept)\n"
            f"  Rounds 5-7: more `run_experiment` to verify scaling on other "
            f"variables (mass, charge, etc.) — vary one at a time.\n"
            f"  Round 8: `python_exec` to fit the functional form against all "
            f"data with scipy.optimize.curve_fit.\n"
            f"  Round 9-10: `submit_law` with the discovered functional form.\n\n"
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
                    name="python_exec",
                    arg_schema={
                        "code": (
                            "str — arbitrary Python (numpy/scipy available as "
                            "`np`, `numpy`, `optimize`, `scipy_stats`). Use "
                            "print() to capture results. ~8s timeout."
                        )
                    },
                    cost_estimate=1.0,
                    description=(
                        "Run Python locally to ANALYZE experimental data you "
                        "already gathered. Strongly recommended for: log-log "
                        "regression to find scaling exponents, curve fitting "
                        "with scipy.optimize.curve_fit, computing slopes, "
                        "comparing functional forms numerically. Costs 1 "
                        "round (same as run_experiment). NO subprocess / os / "
                        "shutil / file I/O / open()."
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

        # Hard rule: when only 1 round left, only submit_law is allowed —
        # force the agent off exploration so it actually submits.
        if self.budget_left() <= 1.0 and action != "submit_law":
            eid = self._next_eid()
            return ExperimentResult(
                eid=eid, action=action, args=args, cost=0.0, raw=None,
                summary={
                    "error": (
                        "budget_left ≤ 1 — only `submit_law` is allowed now. "
                        "Submit your best current hypothesis IMMEDIATELY."
                    )
                },
            )

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
            self._action_log.append({
                "action": "run_experiment",
                "inputs": experiments,
                "outputs": results,
            })
            return ExperimentResult(
                eid=eid, action="run_experiment", args=args, cost=1.0,
                raw=None, summary=summary,
            )

        if action == "python_exec":
            code = str(args.get("code", "")).strip()
            # Strip <python>...</python> wrappers if Generator copied
            # NewtonBench's own tag style
            import re as _re
            m = _re.search(r"<python>(.*?)</python>", code,
                           flags=_re.DOTALL | _re.IGNORECASE)
            if m:
                code = m.group(1).strip()
            if code.startswith("```"):
                code = code.strip("` \n")
                if code.lower().startswith("python"):
                    code = code[6:].lstrip("\n")
            stdout = _safe_python_exec(code)
            self._budget_spent += 1.0
            return ExperimentResult(
                eid=eid, action="python_exec", args={"code_len": len(code)},
                cost=1.0, raw=None,
                summary={"code_preview": code[:300], "stdout": stdout},
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

    def _synthesize_final_law(self) -> str:
        """Last-resort LLM call: given the action log, emit a Python law."""
        try:
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL")
            if not base_url and key.startswith("sk-or-"):
                base_url = "https://openrouter.ai/api/v1"
            client = (OpenAI(base_url=base_url, api_key=key)
                      if base_url else OpenAI())
            model = os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini")
            sig = self._module.FUNCTION_SIGNATURE
            # compact log
            import json as _json
            log_lines = []
            for a in self._action_log[-12:]:
                if a["action"] == "run_experiment":
                    pairs = list(zip(a["inputs"][:6], a["outputs"][:6]))
                    log_lines.append(f"experiments: {_json.dumps(pairs)[:400]}")
            log_block = "\n".join(log_lines) or "(no experiments)"
            sys_p = (
                "You are a scientific-discovery assistant. The user gathered "
                "experimental data but did not submit a final law. Given the "
                "experimental log, return EXACTLY one Python function with "
                f"the signature `{sig}` that best fits the data. Reply with "
                "ONLY the function code, no markdown, no explanation."
            )
            usr = f"Experimental log:\n{log_block}\n\nReturn the function now."
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_p},
                          {"role": "user", "content": usr}],
                temperature=0.1, max_tokens=400,
            )
            txt = (r.choices[0].message.content or "").strip()
            if txt.startswith("```"):
                txt = txt.strip("` \n")
                if txt.lower().startswith("python"):
                    txt = txt[6:].lstrip("\n")
            # heuristic: keep from first 'def discovered_law' to end of indented block
            import re as _re
            m = _re.search(r"(def discovered_law\b.*)", txt, flags=_re.DOTALL)
            if m:
                txt = m.group(1)
            return txt.strip()
        except Exception:
            return ""

    def score_episode(self, claim_store_active, final_artifact=None) -> dict:
        # Fallback: if agent never submitted, synthesize a final law from the
        # action log via one extra LLM call. Mirrors NewtonBench's own
        # "force final submission" fallback.
        if not self._submitted_law and self._action_log:
            self._submitted_law = self._synthesize_final_law()
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
