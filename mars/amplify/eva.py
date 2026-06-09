"""EVA — Execution-Verified Amplification.

A task-agnostic wrapper around a WEAK model. The thesis:

    weak_model + EVA  >=  strong_model (raw)

on arbitrary tasks, with NO task-specific code. EVA never knows what benchmark
it is solving. It receives only a task prompt and one universal tool: execute().

Why it works
------------
Every existing amplifier (self-consistency, Tree-of-Thoughts, Reflexion, debate)
makes the MODEL both generate candidates AND judge them. A weak model is weak at
judging, so amplification stalls — it votes for its own plausible-but-wrong
outputs.

EVA moves the judgment off the model and onto EXECUTION:

    1. PROPOSE   — weak model emits K candidate answers
    2. CHECK     — weak model emits EXECUTABLE checks: code that any correct
                   answer must pass (tests, recomputations, refutations)
    3. EXECUTE   — checks run in a sandbox; the SANDBOX judges, not the model
    4. SELECT    — candidates ranked by objective check-pass signal
    5. REFINE    — failing candidates get the concrete failure as feedback

A weak model's weak judgment no longer matters, because judgment is delegated to
deterministic execution. The model only needs to (a) propose and (b) write
checks — and writing a check is often far easier than getting the answer right
(it is easier to write `assert is_sorted(out)` than to sort correctly).

This is the amplification analogue of P vs NP: verifying is easier than solving,
so we route the hard part (judgment) to cheap deterministic verification.

Task-agnostic contract
----------------------
EVA needs only:
  - task_prompt: str                 (what to solve)
  - execute(code: str) -> ExecResult (run code, return stdout/value/error)
The same EVA loop runs on math, code, data analysis, hypothesis search, etc.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from mars.agents.base import call_llm, make_openai_client


# ===========================================================================
# Execution interface — the ONLY tool. Universal.
# ===========================================================================

@dataclass
class ExecResult:
    ok: bool
    value: Any = None
    stdout: str = ""
    error: str = ""

    def brief(self, limit: int = 400) -> str:
        if not self.ok:
            return f"ERROR: {self.error[:limit]}"
        v = self.stdout or (json.dumps(self.value, default=str)[:limit] if self.value is not None else "")
        return v[:limit]


ExecFn = Callable[[str], ExecResult]


# ===========================================================================
# Records
# ===========================================================================

@dataclass
class Candidate:
    answer: str
    reasoning: str = ""
    checks_passed: int = 0
    checks_total: int = 0
    check_log: list[dict] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.checks_passed / self.checks_total if self.checks_total else 0.0


@dataclass
class EVAResult:
    answer: str
    best: Candidate
    rounds: int
    n_candidates: int
    n_checks: int
    wall_time_s: float
    trace: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "rounds": self.rounds,
            "n_candidates": self.n_candidates,
            "n_checks": self.n_checks,
            "best_pass_rate": self.best.pass_rate if self.best else 0.0,
            "wall_time_s": self.wall_time_s,
            "trace": self.trace,
        }


# ===========================================================================
# The amplifier
# ===========================================================================

class EVA:
    def __init__(
        self,
        *,
        model: str = "openai/gpt-4o-mini",     # the WEAK model being amplified
        k_candidates: int = 4,
        n_checks: int = 5,
        max_rounds: int = 3,
        temperature: float = 0.7,
    ):
        self.model = model
        self.k = k_candidates
        self.n_checks = n_checks
        self.max_rounds = max_rounds
        self.temperature = temperature
        self._client = None

    def _cl(self):
        if self._client is None:
            self._client = make_openai_client()
        return self._client

    def _llm(self, system: str, user: str, *, max_tokens: int = 1200, temp: float | None = None) -> str:
        return call_llm(self._cl(), model=self.model, system=system, user=user,
                        max_tokens=max_tokens, temperature=self.temperature if temp is None else temp)

    # ----- 1. PROPOSE ----------------------------------------------------
    def _propose(self, task: str, feedback: str) -> list[Candidate]:
        prompt = f"""TASK:
{task}
{feedback}

Produce {self.k} DIFFERENT candidate answers. Make them genuinely different
approaches, not rephrasings. For each, give a short reasoning then the answer.

Return ONLY JSON:
{{"candidates": [{{"reasoning": "...", "answer": "..."}}]}}"""
        raw = self._llm(
            "You solve tasks by proposing diverse candidate answers. Return only JSON.",
            prompt, max_tokens=2200,
        )
        items = _parse_list(raw, "candidates")
        cands = []
        for it in items[: self.k]:
            if isinstance(it, dict) and it.get("answer") is not None:
                cands.append(Candidate(answer=str(it["answer"]), reasoning=str(it.get("reasoning", ""))))
        return cands

    # ----- 2. CHECK (as executable code) ---------------------------------
    def _make_checks(self, task: str, candidates: list[Candidate], exec_contract: str) -> list[str]:
        cand_block = "\n".join(f"  Candidate {i+1}: {c.answer[:300]}" for i, c in enumerate(candidates))
        prompt = f"""TASK:
{task}

CANDIDATE ANSWERS:
{cand_block}

Write {self.n_checks} EXECUTABLE checks that distinguish correct from incorrect
answers. A check is Python code that {exec_contract}. Each check must:
- recompute, verify, or refute some property a CORRECT answer must satisfy
- print "PASS" or "FAIL: <reason>" (and the candidate index it concerns if relevant)
- be self-contained and deterministic

Writing a good check is often easier than solving the task — exploit that. Probe
the SPECIFIC ways these candidates could be wrong (boundary cases, recomputation,
consistency, units, counterexamples).

Return ONLY JSON: {{"checks": ["<python code>", ...]}}"""
        raw = self._llm(
            "You write executable verification code that judges candidate answers. Return only JSON.",
            prompt, max_tokens=2600, temp=0.5,
        )
        return [c for c in _parse_list(raw, "checks") if isinstance(c, str) and c.strip()]

    # ----- 3+4. EXECUTE checks & SELECT ----------------------------------
    def _evaluate(self, candidates: list[Candidate], checks: list[str],
                  execute: ExecFn) -> None:
        """Run each check; attribute PASS/FAIL to candidates.

        Checks reference candidates by index. We bind the candidate list into
        the execution namespace so a check can inspect any/all candidates. A
        check that errors is ignored (no signal), never counted as fail — we do
        not let a buggy check punish a candidate."""
        cand_payload = json.dumps([{"index": i + 1, "answer": c.answer}
                                   for i, c in enumerate(candidates)])
        for ci, c in enumerate(candidates):
            c.checks_total = 0
            c.checks_passed = 0
            c.check_log = []
        for check_code in checks:
            # Inject CANDIDATES and the index-under-test for per-candidate checks.
            for ci, c in enumerate(candidates):
                wrapped = (
                    f"CANDIDATES = {cand_payload}\n"
                    f"CANDIDATE_INDEX = {ci + 1}\n"
                    f"CANDIDATE_ANSWER = {json.dumps(c.answer)}\n"
                    + check_code
                )
                res = execute(wrapped)
                out = (res.stdout or "").strip()
                if not res.ok:
                    continue  # buggy check → no signal for this candidate
                verdict = _verdict(out)
                if verdict is None:
                    continue
                c.checks_total += 1
                if verdict:
                    c.checks_passed += 1
                c.check_log.append({"pass": verdict, "out": out[:160]})

    # ----- main loop -----------------------------------------------------
    def solve(self, task: str, execute: ExecFn, *, exec_contract: str = "") -> EVAResult:
        t0 = time.time()
        if not exec_contract:
            exec_contract = ("runs in a Python sandbox with access to a variable "
                             "CANDIDATE_ANSWER (str) and CANDIDATES (list of "
                             "{index, answer}); print PASS or FAIL")
        feedback = ""
        trace: list[dict] = []
        best: Candidate | None = None
        total_cands = 0
        total_checks = 0
        rounds = 0

        for rnd in range(self.max_rounds):
            rounds = rnd + 1
            cands = self._propose(task, feedback)
            if not cands:
                continue
            total_cands += len(cands)
            checks = self._make_checks(task, cands, exec_contract)
            total_checks += len(checks)
            if checks:
                self._evaluate(cands, checks, execute)
            cands.sort(key=lambda c: (c.pass_rate, c.checks_passed), reverse=True)
            round_best = cands[0]
            if best is None or round_best.pass_rate > best.pass_rate:
                best = round_best
            trace.append({
                "round": rounds,
                "n_candidates": len(cands),
                "n_checks": len(checks),
                "best_pass_rate": round_best.pass_rate,
                "best_answer": round_best.answer[:120],
            })
            # Stop early if a candidate passes everything (and there were checks)
            if round_best.checks_total > 0 and round_best.pass_rate == 1.0:
                break
            # Refine: feed the best candidate's failures back
            fails = [l["out"] for l in round_best.check_log if not l["pass"]]
            feedback = ("\nPREVIOUS BEST ANSWER: " + round_best.answer[:200] +
                        "\nIT FAILED THESE CHECKS:\n" + "\n".join(f"- {f}" for f in fails[:5]) +
                        "\nProduce better candidates that pass these checks." ) if fails else ""

        return EVAResult(
            answer=best.answer if best else "",
            best=best or Candidate(answer=""),
            rounds=rounds, n_candidates=total_cands, n_checks=total_checks,
            wall_time_s=time.time() - t0, trace=trace,
        )


# ===========================================================================
# Helpers
# ===========================================================================

def _verdict(out: str) -> bool | None:
    """Parse PASS/FAIL from check stdout. Last verdict line wins."""
    verdict = None
    for line in out.splitlines():
        s = line.strip().upper()
        if s.startswith("PASS"):
            verdict = True
        elif s.startswith("FAIL"):
            verdict = False
    if verdict is None:
        if "PASS" in out.upper() and "FAIL" not in out.upper():
            return True
        if "FAIL" in out.upper() and "PASS" not in out.upper():
            return False
    return verdict


def _parse_list(raw: str, key: str) -> list:
    if not raw:
        return []
    t = raw.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).lstrip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and isinstance(obj.get(key), list):
            return obj[key]
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    m = re.search(r'\{.*\}', t, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and isinstance(obj.get(key), list):
                return obj[key]
        except Exception:
            pass
    return []
