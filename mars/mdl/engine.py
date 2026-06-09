"""Compression-as-Cognition — ONE principle, one engine.

Principle
---------
Solving a task = finding the SHORTEST program that reproduces the observations.
A weak model only PROPOSES candidate programs. The single judge is the total
description length in bits:

    L(program, data) = bits(program) + bits(data | program)

where bits(data | program) is how many bits it costs to encode the observations
given the program (0 if the program reproduces them exactly; large if it errs).

Everything else is a consequence, not a separate mechanism:
  - progress compass  : accept a candidate iff it LOWERS total bits
  - decomposition     : reuse shrinks bits, so the search prefers modular code
  - rollback          : a candidate that doesn't lower bits is simply rejected
  - prohibition       : a constraint that rules out cases is a way to compress
  - transfer          : the library of accepted programs lowers bits on new tasks
  - Occam / simplicity : built into "shortest"

Why it can beat a larger model
------------------------------
A larger model maximizes likelihood — it favours plausible-but-complex text, i.e.
it hallucinates. Compression minimizes bits — Occam is intrinsic, so complex-but-
uninformative proposals are rejected automatically. The weak model supplies
variety; the bit-counter supplies judgment. Judgment is the part weak models are
bad at, and here it is exact and external.

Task-agnostic contract
----------------------
The engine needs only:
  - observations: list of items to compress
  - a way to RUN a candidate program on an observation and compare to it
Both are supplied by a tiny `CompressionTask` with no domain answers.
"""

from __future__ import annotations

import ast
import json
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


# ===========================================================================
# Description length — the ONLY judge
# ===========================================================================

def program_bits(code: str, library: "Library | None" = None) -> float:
    """Description length of a program in bits.

    We approximate Kolmogorov complexity by the compressed size of the source.
    Crucially, symbols already in the LIBRARY are (near) free to reference — this
    is what makes reuse/decomposition emerge: a program built from known parts is
    short. We use a token-ish count with a discount for library names.
    """
    # Normalize whitespace; count meaningful tokens.
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]", code)
    lib_names = set(library.names()) if library else set()
    bits = 0.0
    for t in toks:
        if t in lib_names:
            bits += 1.0          # referencing a known building block is cheap
        elif t.isidentifier():
            bits += 4.0          # a fresh name/op
        else:
            bits += 2.0          # punctuation/number
    return bits


def residual_bits(n_items: int, n_errors: int, per_item_bits: float) -> float:
    """Bits to encode the data GIVEN the program.

    If the program reproduces an item, it costs ~0 to encode. Each error costs
    the literal cost of the item (per_item_bits) plus a pointer to say "here is
    an exception". This is the classic two-part code (model + residual)."""
    if n_errors <= 0:
        return 0.0
    pointer = math.log2(n_items + 1)            # which item is an exception
    return n_errors * (pointer + per_item_bits)


@dataclass
class MDLScore:
    program_bits: float
    residual_bits: float
    n_items: int
    n_correct: int

    @property
    def total(self) -> float:
        return self.program_bits + self.residual_bits

    @property
    def exact_rate(self) -> float:
        return self.n_correct / self.n_items if self.n_items else 0.0


# ===========================================================================
# Library — accepted building blocks (this is where transfer/decomp live)
# ===========================================================================

class Library:
    def __init__(self):
        self._progs: dict[str, "Program"] = {}

    def names(self) -> list[str]:
        return list(self._progs.keys())

    def add(self, prog: "Program") -> None:
        self._progs[prog.name] = prog

    def source_preamble(self) -> str:
        """Source of all library programs, to prepend so new programs can call them."""
        return "\n\n".join(p.code for p in self._progs.values())

    def summary(self) -> list[dict]:
        return [{"name": p.name, "bits": p.bits} for p in self._progs.values()]


# ===========================================================================
# Program — a candidate compressor
# ===========================================================================

@dataclass
class Program:
    name: str
    code: str
    fn: Callable
    bits: float = 0.0
    score: MDLScore | None = None


# ===========================================================================
# Task contract — tiny, no domain answers
# ===========================================================================

class CompressionTask(ABC):
    """Binds the engine to a stream of observations and how to run a program.

    Contains NO answers: only what the observations ARE and how to execute a
    candidate against one observation."""

    name: str = "task"
    entry_point: str = "f"   # the function name a candidate must define

    @abstractmethod
    def observations(self) -> list[Any]:
        ...

    @abstractmethod
    def signature(self) -> str:
        """Exact signature the candidate must define, e.g. 'def f(x): ...'."""

    @abstractmethod
    def run(self, fn: Callable, obs: Any) -> Any:
        """Apply the candidate to one observation, return its prediction."""

    @abstractmethod
    def target(self, obs: Any) -> Any:
        """The observed value the prediction is compared against."""

    @abstractmethod
    def per_item_bits(self) -> float:
        """Literal cost (bits) to encode one observation verbatim — the cost the
        program must beat. E.g. for a length-L string over an A-alphabet:
        L * log2(A)."""

    def interface_hint(self) -> str:
        """Optional extra description for the proposer (types, available vars)."""
        return ""

    def render_observation(self, obs: Any) -> dict:
        """How one observation is shown to the proposer: a clean
        {"input": ..., "output": ...} dict. Override for typed observations so
        the model sees clean pairs, not object reprs."""
        return {"observation": obs}


# ===========================================================================
# Sandbox
# ===========================================================================

_SAFE = {
    "abs": abs, "all": all, "any": any, "bool": bool, "chr": chr, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "len": len, "list": list, "map": map, "max": max, "min": min, "ord": ord,
    "pow": pow, "range": range, "reversed": reversed, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "True": True, "False": False, "None": None,
}
_BAN = {"eval", "exec", "open", "compile", "__import__", "globals", "locals",
        "getattr", "setattr", "vars", "input"}


def _compile(code: str, entry: str, preamble: str = "") -> tuple[bool, str, Callable | None]:
    full = (preamble + "\n\n" + code) if preamble else code
    try:
        tree = ast.parse(full)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}", None
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""])
            if any(m.split(".")[0] != "math" for m in mods):
                return False, "import not allowed", None
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _BAN:
            return False, f"unsafe: {n.func.id}", None
        if isinstance(n, ast.Attribute) and n.attr.startswith("__"):
            return False, "unsafe attr", None
    ns: dict[str, Any] = {"__builtins__": _SAFE, "math": math}
    try:
        exec(compile(full, "<mdl>", "exec"), ns)
    except Exception as e:
        return False, f"ExecError: {e}", None
    fn = ns.get(entry)
    if not callable(fn):
        return False, f"no function '{entry}'", None
    return True, "", fn


# ===========================================================================
# Engine
# ===========================================================================

@dataclass
class MDLResult:
    task: str
    best: Program | None
    total_bits: float
    baseline_bits: float          # bits to store data verbatim (no program)
    compression_ratio: float      # baseline / total (higher = better)
    exact_rate: float
    rounds: int
    n_proposed: int
    n_valid: int
    accepted: list[dict]
    wall_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "best_program": self.best.code if self.best else None,
            "total_bits": round(self.total_bits, 1),
            "baseline_bits": round(self.baseline_bits, 1),
            "compression_ratio": round(self.compression_ratio, 3),
            "exact_rate": round(self.exact_rate, 3),
            "rounds": self.rounds,
            "n_proposed": self.n_proposed,
            "n_valid": self.n_valid,
            "accepted": self.accepted,
            "wall_time_s": round(self.wall_time_s, 1),
        }


class MDLEngine:
    """The weak model proposes; bits decide. One loop, any task."""

    def __init__(self, propose_fn: Callable[..., list[str]], *,
                 max_rounds: int = 4, k_per_round: int = 8,
                 library: Library | None = None):
        # propose_fn(task_prompt, n) -> list of candidate source strings
        self.propose_fn = propose_fn
        self.max_rounds = max_rounds
        self.k_per_round = k_per_round
        self.library = library or Library()

    def _score(self, fn: Callable, code: str, task: CompressionTask,
               obs: list[Any]) -> MDLScore:
        n_correct = 0
        for o in obs:
            try:
                pred = task.run(fn, o)
                if pred == task.target(o):
                    n_correct += 1
            except Exception:
                pass
        n = len(obs)
        pb = program_bits(code, self.library)
        rb = residual_bits(n, n - n_correct, task.per_item_bits())
        return MDLScore(program_bits=pb, residual_bits=rb, n_items=n, n_correct=n_correct)

    def compress(self, task: CompressionTask) -> MDLResult:
        t0 = time.time()
        obs = task.observations()
        n = len(obs)
        baseline = n * task.per_item_bits()      # store everything verbatim

        best: Program | None = None
        best_total = baseline
        accepted: list[dict] = []
        n_proposed = n_valid = 0
        feedback = ""
        rounds = 0

        for rnd in range(self.max_rounds):
            rounds = rnd + 1
            prompt = self._build_prompt(task, obs, best, feedback)
            candidates = self.propose_fn(prompt, self.k_per_round)
            n_proposed += len(candidates)
            preamble = self.library.source_preamble()

            round_best = None
            for code in candidates:
                ok, err, fn = _compile(code, task.entry_point, preamble)
                if not ok:
                    continue
                n_valid += 1
                score = self._score(fn, code, task, obs)
                prog = Program(name=f"p{rnd}_{n_valid}", code=code, fn=fn,
                               bits=score.program_bits, score=score)
                # The ONLY decision: does this lower total description length?
                if score.total < best_total - 1e-9:
                    best_total = score.total
                    best = prog
                    accepted.append({
                        "round": rounds, "total_bits": round(score.total, 1),
                        "exact_rate": round(score.exact_rate, 3),
                        "code": code[:160],
                    })
                if round_best is None or score.total < round_best.score.total:
                    round_best = prog

            # Stop if we've reached zero-residual (perfect compression)
            if best is not None and best.score and best.score.residual_bits == 0:
                break

            # Feedback for next round: show the current best and where it errs.
            if best is not None:
                feedback = (f"\nCURRENT SHORTEST PROGRAM ({best_total:.0f} bits, "
                            f"reproduces {best.score.exact_rate:.0%} of items):\n{best.code}\n"
                            f"Propose programs that are SHORTER or reproduce MORE items "
                            f"(lower total bits). Reuse library parts where possible.")
            elif round_best is not None:
                feedback = (f"\nBest attempt reproduced {round_best.score.exact_rate:.0%} "
                            f"but did not beat verbatim storage. Find real structure.")

        # If best is good enough, promote it to the library (transfer)
        if best is not None and best.score and best.score.exact_rate >= 0.99:
            promoted = Program(name=f"{task.name}_solver", code=best.code, fn=best.fn,
                               bits=best.bits, score=best.score)
            self.library.add(promoted)

        return MDLResult(
            task=task.name, best=best, total_bits=best_total, baseline_bits=baseline,
            compression_ratio=(baseline / best_total) if best_total > 0 else 1.0,
            exact_rate=best.score.exact_rate if best and best.score else 0.0,
            rounds=rounds, n_proposed=n_proposed, n_valid=n_valid,
            accepted=accepted, wall_time_s=time.time() - t0,
        )

    def _build_prompt(self, task: CompressionTask, obs: list[Any],
                      best: Program | None, feedback: str) -> str:
        lib = self.library.names()
        lib_block = (f"\nLIBRARY (call these freely; they are 'free' in bits):\n"
                     f"{self.library.source_preamble()[:1200]}\n" if lib else "")
        examples = json.dumps([task.render_observation(o) for o in obs[:8]],
                              default=str, ensure_ascii=False)[:2800]
        return f"""Find the SHORTEST program that reproduces ALL observations.
You are judged ONLY by total description length = program length + cost of the
items it fails to reproduce. Shorter program that fits more items always wins.

The function you define MUST use EXACTLY this signature (do not change argument
names or count):
{task.signature()}
{task.interface_hint()}
{lib_block}
OBSERVATIONS — each has an "input" and the "output" your function must return
for that input:
{examples}
{feedback}

Return ONLY JSON: {{"programs": ["<source defining {task.entry_point} with the EXACT signature above>", ...]}} (8 diverse, short candidates)"""


# ===========================================================================
# Helpers
# ===========================================================================

def _short(o: Any, limit: int = 200) -> Any:
    if isinstance(o, (int, float, bool)) or o is None:
        return o
    s = o if isinstance(o, str) else json.dumps(o, default=str, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"


def make_llm_proposer(model: str, temperature: float = 0.8):
    """A propose_fn backed by a (weak) LLM. Returns candidate source strings."""
    from mars.agents.base import call_llm, make_openai_client
    client = make_openai_client()

    def propose(prompt: str, n: int) -> list[str]:
        raw = call_llm(client, model=model,
                       system=("You compress observations into the shortest program. "
                               "Return only JSON with a 'programs' list of source strings."),
                       user=prompt, max_tokens=2600, temperature=temperature)
        return _parse_programs(raw)

    return propose


def _parse_programs(raw: str) -> list[str]:
    if not raw:
        return []
    t = raw.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).lstrip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and isinstance(obj.get("programs"), list):
            return [c for c in obj["programs"] if isinstance(c, str) and c.strip()]
        if isinstance(obj, list):
            return [c for c in obj if isinstance(c, str)]
    except Exception:
        pass
    m = re.search(r'\{.*\}', t, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and isinstance(obj.get("programs"), list):
                return [c for c in obj["programs"] if isinstance(c, str) and c.strip()]
        except Exception:
            pass
    return []
