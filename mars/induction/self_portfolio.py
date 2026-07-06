"""Self-Portfolio — a task-agnostic SHELL; the model authors everything.

The shell contains NO benchmark knowledge and NO hand-written strategy. Per task,
the model itself authors:
  - K SOLVER strategies   (code: solve(ctx) -> answer)        ← the "portfolio"
  - a VERIFIER            (code: verify(answer, ctx) -> float)  ← self-check
  - NEGATIVE controls     (plausibly-wrong answer variants)     ← to validate verifier

The shell only: executes code in a sandbox, computes the verifier's discrimination
MARGIN (does verify(answer) >> verify(negatives)?), and selects the answer whose
self-authored verifier is faithful (margin > 0); else abstains. Identical code
runs on any benchmark — the portfolio emerges from the model, not from us.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any, Callable

_SAFE = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate,
    "float": float, "int": int, "len": len, "list": list, "max": max, "min": min,
    "range": range, "round": round, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "set": set, "map": map, "filter": filter, "True": True, "False": False, "None": None,
}
_ALLOWED_IMPORTS = {"math", "statistics", "pandas", "numpy", "itertools"}


def _safe(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
            if any(m.split(".")[0] not in _ALLOWED_IMPORTS for m in names):
                return False
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {
                "eval", "exec", "open", "compile", "__import__", "globals", "getattr", "setattr"}:
            return False
        if isinstance(n, ast.Attribute) and n.attr.startswith("__"):
            return False
    return True


def _safe_import(name, *args, **kwargs):
    """Allow in-function imports of whitelisted modules only (model code often
    writes `import pandas as pd` inside its strategy)."""
    if name.split(".")[0] in _ALLOWED_IMPORTS:
        return __import__(name, *args, **kwargs)
    raise ImportError(f"import not allowed: {name}")


def _compile(code: str, fn_name: str, extra: dict) -> Callable | None:
    if not _safe(code):
        return None
    import math, statistics, itertools
    import numpy as np
    import pandas as pd
    builtins = dict(_SAFE); builtins["__import__"] = _safe_import
    ns: dict[str, Any] = {"__builtins__": builtins, "math": math, "statistics": statistics,
                          "itertools": itertools, "np": np, "pd": pd}
    ns.update(extra)
    try:
        exec(compile(code, "<self_portfolio>", "exec"), ns)
    except Exception:
        return None
    fn = ns.get(fn_name)
    return fn if callable(fn) else None


@dataclass
class PortfolioTrace:
    strategies: list[dict] = field(default_factory=list)
    selected: str | None = None
    margin: float = 0.0
    abstained: bool = False
    note: str = ""


class SelfPortfolioEngine:
    def __init__(self, *, model: str, client, k_strategies: int = 4,
                 k_negatives: int = 4, margin_thresh: float = 0.05):
        self.model = model
        self.client = client
        self.k = k_strategies
        self.kn = k_negatives
        self.thresh = margin_thresh

    def _llm_json(self, prompt: str, max_tokens=1800, temp=0.5):
        from mars.agents.base import call_llm
        raw = call_llm(self.client, model=self.model, system="Return only valid JSON.",
                       user=prompt, max_tokens=max_tokens, temperature=temp)
        t = raw.strip()
        if t.startswith("```"):
            t = "\n".join(t.split("\n")[1:]).split("```")[0]
        try:
            return json.loads(t)
        except Exception:
            i, j = t.find("{"), t.rfind("}")
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                return {}

    # --- model authors the portfolio ------------------------------------
    def _author_strategies(self, task: str, ctx_desc: str) -> list[dict]:
        obj = self._llm_json(
            f"TASK:\n{task}\n\nDATA INTERFACE (ctx given to your function):\n{ctx_desc}\n\n"
            f"Write {self.k} DIVERSE Python strategies that each solve the task differently. "
            f"Each defines `def solve(ctx):` returning the answer (a string). ctx has the keys "
            f"described above; pandas (pd) and numpy (np) are available. Invent genuinely "
            f"different approaches.\n"
            f'Return JSON: {{"strategies":[{{"name":"...","code":"def solve(ctx):\\n    ..."}}]}}')
        return [s for s in obj.get("strategies", []) if isinstance(s, dict) and s.get("code")][: self.k]

    def _author_verifier_for(self, task: str, answer: str, ctx_desc: str) -> str | None:
        """Claim-SPECIFIC verifier: a check that THIS particular answer implies about
        the data. The appropriate check-type emerges from the claim — not a fixed taxonomy."""
        obj = self._llm_json(
            f"TASK:\n{task}\n\nDATA INTERFACE:\n{ctx_desc}\n\n"
            f"CANDIDATE ANSWER:\n{answer}\n\n"
            f"Write `def verify(answer, ctx):` returning a float in [0,1] that checks, FROM THE "
            f"DATA in ctx, whether THIS SPECIFIC claim holds (its stated quantity/group/direction/"
            f"value). The check must be specific to what this answer asserts: a claim about a peak "
            f"checks the argmax; about a group difference checks that difference and its sign; about "
            f"an association checks sign-stability; etc. Correct claim -> high; wrong -> low.\n"
            f'Return JSON: {{"code":"def verify(answer, ctx):\\n    ..."}}')
        c = obj.get("code")
        return c if isinstance(c, str) and c.strip() else None

    def _author_negatives(self, task: str, answer: str) -> list[str]:
        obj = self._llm_json(
            f"TASK:\n{task}\n\nCANDIDATE ANSWER:\n{answer}\n\n"
            f"Produce {self.kn} PLAUSIBLE BUT WRONG variants of this answer (corrupt the key "
            f"variable/population/direction; keep them superficially similar). These are negative "
            f"controls to test a verifier.\n"
            f'Return JSON: {{"negatives":["...","..."]}}', max_tokens=600)
        return [str(n) for n in obj.get("negatives", []) if str(n).strip()][: self.kn]

    # --- shell: execute, per-candidate self-verify, select by margin ----
    def run(self, task: str, ctx: dict, ctx_desc: str) -> tuple[str | None, PortfolioTrace]:
        tr = PortfolioTrace()
        # 1. model-authored strategies -> candidate answers
        cands = []
        for s in self._author_strategies(task, ctx_desc):
            fn = _compile(s["code"], "solve", {})
            if fn is None:
                continue
            try:
                ans = fn(dict(ctx))
                if ans is not None and str(ans).strip():
                    cands.append((s.get("name", "strat"), str(ans)))
            except Exception:
                continue
        tr.strategies = [{"name": n, "answer": a[:120]} for n, a in cands]
        if not cands:
            tr.abstained = True; tr.note = "no strategy executed"
            return None, tr

        # 2. per-candidate: author a CLAIM-SPECIFIC verifier + negatives; margin
        scored = []
        for name, ans in cands:
            vcode = self._author_verifier_for(task, ans, ctx_desc)
            vfn = _compile(vcode, "verify", {}) if vcode else None
            if vfn is None:
                continue

            def vscore(a, _vfn=vfn):
                try:
                    return float(_vfn(a, dict(ctx)))
                except Exception:
                    return 0.0

            negs = self._author_negatives(task, ans)
            neg_scores = [vscore(n) for n in negs] or [0.0]
            self_score = vscore(ans)
            margin = self_score - max(neg_scores)        # is this verifier FAITHFUL for this claim?
            scored.append({"name": name, "answer": ans, "self": round(self_score, 3),
                           "margin": round(margin, 3)})

        tr.strategies = scored or tr.strategies
        # 3. keep candidates whose own verifier is faithful (discriminates) AND passed
        faithful = [c for c in scored if c["margin"] > self.thresh and c["self"] > 0.5]
        if not faithful:
            tr.abstained = True
            tr.note = "no candidate passed its own faithful verifier; abstain"
            return None, tr
        # 4. select the one with the strongest faithful verification
        best = max(faithful, key=lambda c: (c["margin"], c["self"]))
        tr.selected = best["name"]; tr.margin = best["margin"]
        tr.note = f"selected '{best['name']}' (self={best['self']}, margin={best['margin']})"
        return best["answer"], tr
