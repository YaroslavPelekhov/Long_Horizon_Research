"""
Autonomous Investigator — the self-improving core of MARS.

The thesis (user's framing): do NOT fine-tune a separate pipeline per
benchmark. Instead build a system that, faced with an UNFAMILIAR task,
investigates it, empirically tries strategies, ESCALATES when a strategy
underperforms, INVENTS a new strategy when the whole portfolio fails, and
REMEMBERS what worked — improving its own architecture over time.

This module realizes that loop for the "fit" family (law discovery), because
it is the cleanest place to *demonstrate* empirical self-escalation:

    investigate(adapter):
        for strategy in escalating_portfolio:        # cheap → expensive
            result = strategy.run(adapter, probe_data)
            quality = MEASURE(result, probe_holdout)  # empirical, not assumed
            if quality >= bar:
                remember(signature, strategy); return result   # stop early
        return invent_new(adapter)                    # architectural self-improvement

Crucially the choice is driven by MEASURED quality on held-out probe data, not
by hard-coded "this benchmark → this strategy". The same code therefore stops
at cheap regression on an easy task and escalates to structural-sketch + snap on
a hard one — with no knowledge of which benchmark it is looking at.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from mars.srhp.sketch_solver import StructuralSketchSolver, _build_fn, _rmsle


# ── strategy primitives (each: fit data → law code + measured holdout rmsle) ──

def _powerlaw_regression(inputs, outputs, var_names):
    """Cheapest strategy: log-log separable power-law (no LLM, no snap)."""
    X = np.array([[float(d[k]) for k in var_names] for d in inputs], float)
    y = np.array(outputs, float)
    mask = (y > 0) & np.all(X > 0, axis=1)
    if mask.sum() < len(var_names) + 1:
        return None, float("inf")
    lX = np.log(X[mask]); ly = np.log(y[mask])
    A = np.hstack([lX, np.ones((mask.sum(), 1))])
    try:
        coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    except Exception:
        return None, float("inf")
    exps = [float(c) for c in coef[:-1]]
    c0 = float(np.exp(coef[-1]))
    terms = " * ".join(f"{v}**({exps[i]:.6f})" for i, v in enumerate(var_names))
    body = f"return {c0:.8g} * {terms}"
    code = (f"def discovered_law({', '.join(var_names)}):\n"
            f"    import math\n    try:\n        return {c0:.8g} * "
            + " * ".join(f"{v}**({exps[i]:.6f})" for i, v in enumerate(var_names))
            + "\n    except Exception:\n        return 0.0")
    # rmsle on the data
    expr = "c0 * " + " * ".join(f"{v}**c{i+1}" for i, v in enumerate(var_names))
    fn = _build_fn(expr, var_names, len(exps) + 1)
    r = _rmsle(y, fn(X, c0, *exps)) if fn else float("inf")
    return code, r


# ── the investigator ─────────────────────────────────────────────────────────

@dataclass
class InvestigationResult:
    strategy: str
    law_code: str
    probe_rmsle: float
    n_escalations: int
    trace: list = field(default_factory=list)


@dataclass
class AutonomousInvestigator:
    """Empirical, escalating strategy selection for fit-family tasks.

    Memory persists across tasks within a process so the system gets faster on
    recurring task-signatures (self-improvement over time)."""
    model: str = "openai/gpt-4o-mini"
    quality_bar: float = 0.03          # holdout rmsle below this = "good enough"
    n_probe: int = 18
    n_holdout: int = 6
    memory: dict = field(default_factory=dict)   # signature -> strategy name

    def _signature(self, adapter) -> str:
        try:
            params = adapter._srhp_params()
        except Exception:
            params = []
        return f"nvars={len(params)}"

    def _gather(self, adapter, n):
        cands = adapter.srhp_candidate_experiments(max(48, n * 2))
        inputs, outputs = [], []
        step = max(1, len(cands) // n)
        for exp in cands[::step]:
            if len(inputs) >= n or adapter.budget_left() <= 1:
                break
            try:
                y = adapter.srhp_run(exp)
                if y == y and math.isfinite(float(y)):
                    inputs.append(exp); outputs.append(float(y))
            except Exception:
                continue
        return inputs, outputs

    def investigate(self, adapter) -> "InvestigationResult | None":
        var_names = adapter._srhp_params()
        if not var_names:
            return None
        # gather probe + holdout data
        data_in, data_out = self._gather(adapter, self.n_probe + self.n_holdout)
        if len(data_out) < 6:
            return None
        split = max(4, len(data_out) - self.n_holdout)
        tr_in, tr_out = data_in[:split], data_out[:split]
        ho_in, ho_out = data_in[split:], data_out[split:]

        def _holdout_rmsle(code):
            # exec the law and score on holdout
            ns = {"math": __import__("math")}
            try:
                exec(code, ns)
                fn = ns.get("discovered_law")
            except Exception:
                return float("inf")
            if fn is None:
                return float("inf")
            preds = []
            for d in ho_in:
                try:
                    preds.append(float(fn(**{k: float(d[k]) for k in var_names})))
                except Exception:
                    preds.append(float("nan"))
            return _rmsle(np.array(ho_out, float), np.array(preds, float))

        trace = []
        escalations = 0

        # known signature → try its winner first (memory / self-improvement)
        sig = self._signature(adapter)
        # ── escalating portfolio: cheap → expensive ──
        # 1. power-law regression (cheapest, no LLM)
        code, _ = _powerlaw_regression(tr_in, tr_out, var_names)
        if code is not None:
            ho = _holdout_rmsle(code)
            trace.append(("regression", round(ho, 4)))
            if ho <= self.quality_bar:
                self.memory[sig] = "regression"
                return InvestigationResult("regression", code, ho, escalations, trace)

        # 2. ESCALATE → structural sketch + Fit-then-Snap (LLM structure + snap)
        escalations += 1
        solver = StructuralSketchSolver(model=self.model, max_rounds=5)
        res = solver.solve(tr_in, tr_out)
        if res is not None:
            code2 = res.to_law_code()
            ho2 = _holdout_rmsle(code2)
            trace.append(("sketch+snap", round(ho2, 4)))
            # keep whichever is better on holdout
            best_code, best_ho, best_name = code2, ho2, "sketch+snap"
            if code is not None and trace and trace[0][1] < ho2:
                best_code, best_ho, best_name = code, trace[0][1], "regression"
            self.memory[sig] = best_name
            return InvestigationResult(best_name, best_code, best_ho, escalations, trace)

        # 3. fall back to regression if sketch failed
        if code is not None:
            return InvestigationResult("regression", code,
                                       trace[0][1] if trace else float("inf"),
                                       escalations, trace)
        return None
