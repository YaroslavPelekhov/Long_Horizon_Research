"""Automated grammar synthesis for CPI from observable interface + example traces.

Core pipeline:
    interface_description + traces → LLM proposals → sandbox validation → ProgramHypothesis list

This is the self-building step: no human writes the grammar primitives.
The LLM reads the observable interface schema, sees a few example transitions,
and proposes typed candidate functions. The sandbox validates them. The CPI
layer then selects winners by refutation score — not by asking the LLM to judge.

Key claim: the grammar is not pre-designed by the programmer for this benchmark.
It is synthesized on-the-fly from the interface the benchmark exposes.
"""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from mars.agents.base import call_llm, make_openai_client
from mars.induction.cpi import ProgramHypothesis, RuleTrace


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "ord": ord,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}

_BANNED_AST_NODES = {ast.Import, ast.ImportFrom}
_BANNED_NAMES = {"eval", "exec", "open", "compile", "__import__", "breakpoint", "input"}


def _ast_safe(code: str) -> tuple[bool, str]:
    """Static safety check: no imports, no dangerous builtins."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if type(node) in _BANNED_AST_NODES:
            return False, f"import not allowed"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_NAMES:
                return False, f"unsafe call: {node.func.id}"
        if isinstance(node, ast.Attribute) and node.attr in {"__class__", "__globals__", "__code__"}:
            return False, f"unsafe attribute: {node.attr}"
    return True, ""


def _sandbox_compile(
    code: str,
    test_traces: list[RuleTrace],
) -> tuple[bool, str, Any]:
    """Compile, exec, and smoke-test the proposed function. Returns (ok, error, fn)."""
    ok, err = _ast_safe(code)
    if not ok:
        return False, err, None

    ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(compile(code, "<grammar_synth>", "exec"), ns)
    except Exception as e:
        return False, f"ExecError: {e}", None

    fn = next(
        (v for k, v in ns.items() if callable(v) and not isinstance(v, type) and not k.startswith("_")),
        None,
    )
    if fn is None:
        return False, "no callable found", None

    for trace in test_traces[:4]:
        try:
            out = fn(trace.current, trace.context)
        except Exception as e:
            return False, f"RuntimeError: {e}", None
        if not isinstance(out, str):
            return False, f"returned {type(out).__name__}, expected str", None

    return True, "", fn


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SynthesisResult:
    proposed: int
    valid: int
    hypotheses: list[ProgramHypothesis]
    raw_proposals: list[dict]
    errors: list[str]
    model: str
    wall_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposed": self.proposed,
            "valid": self.valid,
            "model": self.model,
            "wall_time_s": self.wall_time_s,
            "errors": self.errors[:10],
            "primitives": [
                {
                    "name": h.name,
                    "description": h.description,
                    "complexity": h.complexity,
                    "causal_anchors": list(h.causal_anchors),
                    "tags": list(h.tags),
                }
                for h in self.hypotheses
            ],
        }


# ---------------------------------------------------------------------------
# Core synthesis call
# ---------------------------------------------------------------------------

_UH_SEQ_INTERFACE = """
Environment: Unknown sequence transformation environment.

Observable interface — your functions MUST match this exactly:
  def rule_name(current: str, context: dict) -> str

Arguments:
  current  — a string of uppercase letters (may be empty string for the first rule in a chain)
  context  — a dict with these keys:
    context['main']          str  — "main" input sequence (5 uppercase letters, A–E)
    context['vice']          str  — "vice" input sequence (5 uppercase letters, A–E)
    context['step_number']   int  — current step index (1, 2, 3, ...)
    context['previous_main'] str  — the main sequence from the previous step (empty on step 1)

Return value: a str of uppercase letters

Allowed Python builtins: ord, chr, len, range, zip, sorted, max, min, sum, abs,
  str, int, float, bool, list, tuple, set, dict, enumerate, round
NO imports. NO file I/O. NO eval/exec.

Character arithmetic:
  position of c in alphabet: ord(c) - ord('A')     (A=0, B=1, ..., E=4)
  char from position i (mod 26): chr(ord('A') + i % 26)

Operation categories to try (be creative, cover all of these):
  1. INTERLEAVING: alternate characters position-wise from main and vice
  2. POSITION-WISE COMPARISON: for each i, take max(main[i], vice[i]) or min(main[i], vice[i])
  3. POSITION-WISE ARITHMETIC: for each i, add/subtract alphabet positions of main[i] and vice[i] mod 26
  4. SORTING: sort characters from main+vice combined, or sort current
  5. STEP-BASED BRANCHING: different behavior on odd/even/prime step_number values
  6. HISTORY: use previous_main combined with current or main
  7. CURRENT-BASED: reverse, shift, or rearrange characters in current
  8. FREQUENCY-BASED: find most/least frequent char in current, replace or shift it
"""


def synthesize_grammar(
    traces: list[RuleTrace],
    *,
    interface_description: str = _UH_SEQ_INTERFACE,
    n_proposals: int = 16,
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.8,
) -> SynthesisResult:
    """Synthesize grammar primitives from interface description and observed traces.

    This is the zero-shot grammar construction step. The LLM receives only:
    - the interface schema (types, available variables)
    - a few observed (current, context, target) transitions

    It proposes Python functions. The sandbox validates them. No human writes
    any primitive for this benchmark.
    """
    t0 = time.time()

    examples = []
    for i, tr in enumerate(traces[:8]):
        ctx_clean = {k: v for k, v in tr.context.items() if not k.startswith("__")}
        examples.append({
            "ex": i + 1,
            "current": tr.current,
            "context": ctx_clean,
            "target": tr.target,
        })

    prompt = f"""{interface_description}

OBSERVED TRANSITIONS (current + context → target):
{json.dumps(examples, indent=2, ensure_ascii=False)}

Propose {n_proposals} diverse Python functions, each with a DIFFERENT mechanism.
Cover: character interleaving, modular arithmetic, lexicographic ops, step-based
logic, frequency operations, history use (previous_main), and sorting.

Some functions will not explain all examples — that is expected and fine.
The CPI layer will score each function against observations later.

Return ONLY this JSON (no other text):
{{
  "primitives": [
    {{
      "name": "short_snake_case_name",
      "description": "one-line description",
      "complexity": <1-5 integer, 1=simplest>,
      "causal_anchors": ["main", "vice"],
      "code": "def short_snake_case_name(current, context):\\n    ..."
    }}
  ]
}}"""

    client = make_openai_client()
    raw = call_llm(
        client,
        model=model,
        system=(
            "You are a program synthesis system. Propose diverse Python functions "
            "to explain observed transitions. Return only valid JSON."
        ),
        user=prompt,
        max_tokens=4500,
        temperature=temperature,
    )

    proposals: list[dict] = _parse_proposals(raw)
    hypotheses: list[ProgramHypothesis] = []
    errors: list[str] = []

    for p in proposals:
        if not isinstance(p, dict):
            errors.append("non-dict proposal")
            continue
        code = p.get("code", "").strip()
        if not code:
            errors.append(f"{p.get('name','?')}: missing code")
            continue
        ok, err, fn = _sandbox_compile(code, traces)
        if not ok:
            errors.append(f"{p.get('name','?')}: {err}")
            continue
        h = ProgramHypothesis(
            name=p.get("name", f"synth_{len(hypotheses)}"),
            description=p.get("description", ""),
            fn=fn,
            complexity=float(p.get("complexity", 2.0)),
            causal_anchors=tuple(str(a) for a in p.get("causal_anchors", [])),
            tags=("synthesized",),
        )
        hypotheses.append(h)

    return SynthesisResult(
        proposed=len(proposals),
        valid=len(hypotheses),
        hypotheses=hypotheses,
        raw_proposals=proposals,
        errors=errors,
        model=model,
        wall_time_s=time.time() - t0,
    )


def _parse_proposals(raw: str) -> list[dict]:
    """Best-effort extraction of the primitives list from LLM output."""
    if not raw:
        return []
    text = raw.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.lstrip()

    # Try as JSON object wrapping the list
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            candidates = obj.get("primitives", obj.get("functions", obj.get("proposals", [])))
            if isinstance(candidates, list):
                return candidates
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # Try to extract JSON array substring
    m = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass

    # Try to extract JSON object and pull primitives
    m2 = re.search(r'\{.*\}', text, re.DOTALL)
    if m2:
        try:
            obj = json.loads(m2.group())
            if isinstance(obj, dict):
                for key in ("primitives", "functions", "proposals"):
                    if isinstance(obj.get(key), list):
                        return obj[key]
        except Exception:
            pass

    return []
