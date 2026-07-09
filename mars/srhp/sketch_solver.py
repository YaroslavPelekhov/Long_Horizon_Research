"""
StructuralSketchSolver — the hard-task cracker for RASC-fit escalation.

Division of labour, made precise:
  - the LLM proposes the STRUCTURE of a law as a parametric SKETCH: a Python
    expression in the input variables with named free constants c0, c1, …
    (and free choice of sin / cos / exp / log / power / ratio / threshold).
  - scipy.optimize.curve_fit fills the holes (fits c0…cK) against the data the
    episode already gathered — no extra experiment budget.
  - refutation: if the best sketch still mispredicts, the worst residuals are
    shown back and the LLM proposes a STRUCTURALLY different sketch.

This is what plain regression (a single fixed power-law basis) cannot do: it
searches OVER functional forms. It is triggered only when the fit-probe says the
law is non-trivial (moderate R²), so it costs LLM calls only on hard tasks.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from mars.agents.base import call_llm, make_openai_client, parse_json_strict


_SKETCH_SYS = (
    "You discover the STRUCTURE of a hidden scientific law. You are given "
    "experimental data (inputs -> output). Propose ONE parametric formula as a "
    "Python expression. Use the input variable names, free constants named "
    "c0, c1, c2, ... (these are FITTED by a numeric solver — you do NOT pick "
    "their values), and any of: +, -, *, /, **, np.sin, np.cos, np.exp, np.log, "
    "np.sqrt, np.abs, np.tanh.\n\n"
    "The universe's physics may be NON-standard (shifted exponents, unusual "
    "trig/exponential structure). Fit the DATA, not textbook formulas.\n\n"
    "Return EXACTLY one JSON object:\n"
    '{"expr": "<python expression in the variables and c0..cK>",'
    ' "n_consts": <int>,'
    ' "init": [<initial guess per constant>],'
    ' "note": "<one line on the mechanism>"}\n'
    "Rules: every c0..c{n_consts-1} must appear in expr; use np. for functions; "
    "angles may be in radians OR degrees — if a variable looks like an angle, "
    "consider np.sin(c*var) to absorb unit scaling. Reply ONLY the JSON."
)

_REVISE_SYS = (
    "Your previous sketch fit poorly. Propose a STRUCTURALLY DIFFERENT formula "
    "(change the functional form: additive<->multiplicative, add/remove a trig "
    "or exponential term, a ratio, a threshold via np.tanh, a different power "
    "structure) — not just different constants. Same JSON format. Reply ONLY JSON."
)


def _build_fn(expr: str, var_names: list[str], n_consts: int):
    """Compile expr into f(X, *consts) where X is an (N, n_vars) array.
    Returns callable or None."""
    # safety: only allow a small token set
    if re.search(r"(import|__|open|eval|exec|lambda|;)", expr):
        return None
    safe = {"np": np, "__builtins__": {}}

    def f(X, *consts):
        local = dict(safe)
        for i, name in enumerate(var_names):
            local[name] = X[:, i]
        for j in range(n_consts):
            local[f"c{j}"] = consts[j]
        try:
            with np.errstate(all="ignore"):
                out = eval(expr, {"__builtins__": {}}, local)  # noqa: S307
        except Exception:
            return np.full(X.shape[0], np.nan)
        out = np.asarray(out, dtype=float)
        if out.shape != (X.shape[0],):
            out = np.full(X.shape[0], np.nan)
        # clip pathological overflow to keep curve_fit / rmsle finite
        out = np.where(np.isfinite(out), out, np.nan)
        return np.clip(out, -1e300, 1e300)

    return f


def _rmsle(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("inf")
    a = np.abs(y_true[m]); b = np.abs(y_pred[m])
    return float(np.sqrt(np.mean((np.log1p(b) - np.log1p(a)) ** 2)))


@dataclass
class SketchResult:
    expr: str
    consts: list
    rmsle: float
    var_names: list
    fn_name: str = "discovered_law"

    def to_law_code(self) -> str:
        """Emit a `def discovered_law(<params>): return <expr with fitted consts>`."""
        expr = self.expr
        for j, c in enumerate(self.consts):
            expr = re.sub(rf"\bc{j}\b", repr(float(c)), expr)
        # convert np.* to math.* not needed — evaluator imports numpy; but the NB
        # evaluator runs scalar calls, so use math-friendly: replace np. with math.
        expr_scalar = expr.replace("np.", "math.")
        params = ", ".join(self.var_names)
        return (f"def {self.fn_name}({params}):\n"
                f"    import math\n"
                f"    try:\n"
                f"        return {expr_scalar}\n"
                f"    except Exception:\n"
                f"        return 0.0")


@dataclass
class StructuralSketchSolver:
    model: str = "openai/gpt-4o-mini"
    max_rounds: int = 4
    max_tokens: int = 320

    def __post_init__(self):
        self._client = make_openai_client()
        self.n_sketches = 0
        self.n_fitted = 0

    def _fit_sketch(self, sk: dict, X, y, var_names):
        from scipy.optimize import curve_fit
        expr = str(sk.get("expr", "")).strip()
        nC = int(sk.get("n_consts", 0))
        if not expr or nC <= 0 or nC > 12:
            return None
        fn = _build_fn(expr, var_names, nC)
        if fn is None:
            return None
        init = sk.get("init") or [1.0] * nC
        if len(init) != nC:
            init = [1.0] * nC
        try:
            popt, _ = curve_fit(fn, X, y, p0=init, maxfev=4000)
        except Exception:
            # fall back: keep init guesses
            try:
                popt = np.array(init, float)
                _ = fn(X, *popt)
            except Exception:
                return None
        pred = fn(X, *popt)
        r = _rmsle(y, pred)
        if not math.isfinite(r):
            return None
        self.n_fitted += 1
        return SketchResult(expr=expr, consts=list(popt), rmsle=r, var_names=var_names)

    def _snap_and_refit(self, res: "SketchResult", X, y) -> "SketchResult":
        """Exact-symbolic recovery: snap EXPONENT constants (those after `**`) to
        the nearest simple value (integer / n·½ / n·⅓ / small rational), refit
        the remaining multiplicative constants, and keep the snapped law if it
        does not worsen the fit. When the snapped exponents hit the true law,
        holdout error collapses → exact symbolic form recovered.

        This is the missing link between a good numeric fit and a SYMBOLIC match
        (binary SA): curve_fit gives 1.97, the true law is 2 — snapping recovers
        the exact form.
        """
        from scipy.optimize import curve_fit
        var_names = res.var_names
        # which constants are exponents?  match  ** c<idx>  (with optional spaces)
        exp_idxs = sorted({int(m) for m in re.findall(r"\*\*\s*c(\d+)", res.expr)})
        if not exp_idxs:
            return res
        n = len(res.consts)
        fn = _build_fn(res.expr, var_names, n)
        if fn is None:
            return res

        def _simple_snaps(v: float):
            cands = set()
            for q in (1, 2, 3, 4):
                cands.add(round(v * q) / q)        # nearest 1/q multiple
            # also small integers nearby
            cands.add(float(round(v)))
            return {c for c in cands if abs(c) <= 12}

        # snap each exponent independently to its nearest simple value
        snapped = list(res.consts)
        for i in exp_idxs:
            if i < n:
                v = float(snapped[i])
                opts = _simple_snaps(v)
                if opts:
                    snapped[i] = min(opts, key=lambda s: abs(s - v))

        non_exp = [i for i in range(n) if i not in exp_idxs]

        # refit the multiplicative (non-exponent) constants with exponents fixed
        if non_exp:
            def f_refit(Xd, *free):
                full = list(snapped)
                for k, idx in enumerate(non_exp):
                    full[idx] = free[k]
                return fn(Xd, *full)
            p0 = [snapped[i] for i in non_exp]
            try:
                popt, _ = curve_fit(f_refit, X, y, p0=p0, maxfev=4000)
                for k, idx in enumerate(non_exp):
                    snapped[idx] = float(popt[k])
            except Exception:
                pass

        pred = fn(X, *snapped)
        r_snap = _rmsle(y, pred)
        # accept the snapped law if it is no worse than ~1.3× the raw fit
        # (a true snap collapses error; a wrong snap inflates it and is rejected)
        if math.isfinite(r_snap) and r_snap <= max(res.rmsle * 1.3, res.rmsle + 0.02):
            return SketchResult(expr=res.expr, consts=snapped, rmsle=r_snap,
                                var_names=var_names)
        return res

    @staticmethod
    def _snap_val(v: float) -> float:
        cands = set()
        for q in (1, 2, 3, 4):
            cands.add(round(v * q) / q)
        cands.add(float(round(v)))
        cands = {c for c in cands if abs(c) <= 12}
        return min(cands, key=lambda s: abs(s - v)) if cands else v

    def _seed_powerlaw(self, X, y, var_names) -> "SketchResult | None":
        """Deterministic separable power-law via robust log-log least squares,
        with exponents snapped to exact rationals. Catches gravity / coulomb /
        hooke / fourier-type laws reliably — no curve_fit fragility, no reliance
        on the LLM guessing the multiplicative-power structure."""
        mask = (y > 0) & np.all(X > 0, axis=1)
        if mask.sum() < len(var_names) + 1:
            return None
        lX = np.log(X[mask]); ly = np.log(y[mask])
        A = np.hstack([lX, np.ones((mask.sum(), 1))])
        try:
            coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
        except Exception:
            return None
        exps = [self._snap_val(float(c)) for c in coef[:-1]]
        # refit the constant with snapped exponents: log(y) - sum(e_i log x_i) = log c0
        resid = ly - lX @ np.array(exps)
        c0 = float(np.exp(np.mean(resid)))
        consts = [c0] + exps
        # expr: c0 * v0**c1 * v1**c2 * ...
        terms = " * ".join(f"{v}**c{i+1}" for i, v in enumerate(var_names))
        expr = f"c0 * {terms}"
        fn = _build_fn(expr, var_names, len(consts))
        if fn is None:
            return None
        r = _rmsle(y, fn(X, *consts))
        if not math.isfinite(r):
            return None
        return SketchResult(expr=expr, consts=consts, rmsle=r, var_names=var_names)

    def _seed_trig(self, X, y, var_names) -> "SketchResult | None":
        """Invented primitive for laws with an ANGLE variable: fit
        log(y) = log(c0) + a·log|sin(angle)| (or cos) + Σ bᵢ·log(varᵢ), then snap
        the trig exponent a to a simple value (1, 2 → sin/sin², cos/cos²).
        Recovers Malus (cos²), Snell (sin)-type laws that power-law seeds miss."""
        # detect angle-like variables (by name or by value range ⊂ [0, 2π] or [0,180])
        ANGLE = ("theta", "angle", "phi", "alpha", "incid", "polar", "tilt")
        ang_idx = []
        for i, v in enumerate(var_names):
            vl = v.lower()
            col = X[:, i]
            finite = col[np.isfinite(col)]
            rng_ok = finite.size and finite.min() >= 0 and finite.max() <= 6.5
            if any(h in vl for h in ANGLE) or rng_ok:
                ang_idx.append(i)
        if not ang_idx:
            return None
        others = [i for i in range(len(var_names)) if i not in ang_idx]
        av = var_names[ang_idx[0]]
        best = None
        # angles sampled in [0, π/2] → sin/cos > 0, no abs needed (keeps the law
        # valid in both vector (np.) and scalar (math.) execution after np→math).
        for trig_name in ("sin", "cos"):
            t = (np.sin if trig_name == "sin" else np.cos)(X[:, ang_idx[0]])
            mask = (y > 0) & (t > 1e-9)
            for oi in others:
                mask = mask & (X[:, oi] > 0)
            if mask.sum() < len(others) + 2:
                continue
            cols = [np.log(t[mask])]
            for oi in others:
                cols.append(np.log(X[mask, oi]))
            A = np.column_stack(cols + [np.ones(int(mask.sum()))])
            try:
                coef, *_ = np.linalg.lstsq(A, np.log(y[mask]), rcond=None)
            except Exception:
                continue
            a_trig = self._snap_val(float(coef[0]))
            o_exps = [self._snap_val(float(c)) for c in coef[1:-1]]
            c0 = float(np.exp(coef[-1]))
            terms = [f"np.{trig_name}({av})**({a_trig})"]
            for k, oi in enumerate(others):
                terms.append(f"{var_names[oi]}**({o_exps[k]})")
            expr = "c0 * " + " * ".join(terms)
            fn = _build_fn(expr, var_names, 1)
            if fn is None:
                continue
            r = _rmsle(y, fn(X, c0))
            if math.isfinite(r) and (best is None or r < best.rmsle):
                best = SketchResult(expr=expr, consts=[c0], rmsle=r,
                                    var_names=var_names)
        return best

    def _seed_exp(self, X, y, var_names) -> "SketchResult | None":
        """Deterministic exponential-decay/growth seed: log(y) linear in each var.
        Catches radioactive-decay-type laws (N0 * exp(-λ t))."""
        mask = (y > 0)
        if mask.sum() < len(var_names) + 1:
            return None
        Xm = X[mask]; ly = np.log(y[mask])
        A = np.hstack([Xm, np.ones((mask.sum(), 1))])
        try:
            coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
        except Exception:
            return None
        rates = [float(c) for c in coef[:-1]]
        c0 = float(np.exp(coef[-1]))
        consts = [c0] + rates
        terms = " + ".join(f"c{i+1}*{v}" for i, v in enumerate(var_names))
        expr = f"c0 * np.exp({terms})"
        fn = _build_fn(expr, var_names, len(consts))
        if fn is None:
            return None
        r = _rmsle(y, fn(X, *consts))
        if not math.isfinite(r):
            return None
        return SketchResult(expr=expr, consts=consts, rmsle=r, var_names=var_names)

    def solve(self, inputs: list[dict], outputs: list[float]) -> "SketchResult | None":
        """Search functional forms; return the best-fitting SketchResult."""
        if len(outputs) < 4:
            return None
        var_names = list(inputs[0].keys())
        try:
            X = np.array([[float(d[k]) for k in var_names] for d in inputs], float)
            y = np.array(outputs, float)
        except Exception:
            return None
        m = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        X, y = X[m], y[m]
        if len(y) < 4:
            return None

        # data summary for the prompt
        def _fmt_rows(limit=12):
            rows = []
            for i in range(min(limit, len(y))):
                d = {var_names[j]: round(float(X[i, j]), 4) for j in range(len(var_names))}
                rows.append(f"  {json.dumps(d)} -> {float(y[i]):.6g}")
            return "\n".join(rows)

        best: SketchResult | None = None
        # deterministic seeds (robust, snapped) — the "invented primitives"
        # portfolio: power-law, trig (sin/cos^k for angles), exp-decay.
        for seed_fn in (self._seed_powerlaw, self._seed_trig, self._seed_exp):
            try:
                sd = seed_fn(X, y, var_names)
            except Exception:
                sd = None
            if sd is not None and (best is None or sd.rmsle < best.rmsle):
                best = sd
        # if a deterministic seed already nails it, we can skip LLM rounds
        if best is not None and best.rmsle < 0.02:
            return best

        tried = []
        sysp = _SKETCH_SYS
        for rnd in range(self.max_rounds):
            prev = ""
            if best is not None:
                # show worst residuals to drive structural change
                from numpy import argsort
                fn = _build_fn(best.expr, var_names, len(best.consts))
                resid_txt = ""
                if fn is not None:
                    pred = fn(X, *best.consts)
                    err = np.abs(np.log1p(np.abs(pred)) - np.log1p(np.abs(y)))
                    order = argsort(-err)[:3]
                    resid_txt = "\n".join(
                        f"  worst: {json.dumps({var_names[j]: round(float(X[i,j]),3) for j in range(len(var_names))})}"
                        f" obs={float(y[i]):.4g} pred={float(pred[i]):.4g}"
                        for i in order)
                prev = (f"\nBest so far: expr={best.expr} rmsle={best.rmsle:.4f}\n"
                        f"{resid_txt}\n")
            user = (f"VARIABLES: {var_names}\nDATA (inputs -> output):\n{_fmt_rows()}\n"
                    f"{prev}\nPropose the sketch JSON now.")
            try:
                raw = call_llm(self._client, self.model, sysp, user,
                               max_tokens=self.max_tokens, temperature=0.4)
                sk = parse_json_strict(raw) or {}
            except Exception:
                sk = {}
            self.n_sketches += 1
            res = self._fit_sketch(sk, X, y, var_names)
            if res is not None:
                # exact-symbolic recovery: snap exponents to simple values
                res = self._snap_and_refit(res, X, y)
            tried.append((sk.get("expr", ""), res.rmsle if res else float("inf")))
            if res is not None and (best is None or res.rmsle < best.rmsle):
                best = res
            if best is not None and best.rmsle < 0.02:
                break
            sysp = _REVISE_SYS
        return best

    def stats(self) -> dict:
        return {"n_sketches": self.n_sketches, "n_fitted": self.n_fitted}
