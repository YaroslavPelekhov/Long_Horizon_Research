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
    RubricSection,
)
from ols.core.types import (
    ActionSpec,
    Claim,
    ClaimVerdict,
    ExperimentResult,
)


# 12 physics modules in NewtonBench (actual folder names)
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
    "m9_hooke_law",
    "m10_be_distribution",
    "m11_heat_transfer",
]

# difficulty tiers
NB_DIFFICULTIES = ["easy", "medium", "hard"]

# law versions per difficulty
NB_LAW_VERSIONS = ["v0", "v1", "v2"]

# system complexity (vanilla = direct law evaluation; simple/complex = motion-
# simulation-based observation, which is harder)
NB_SYSTEMS = ["vanilla_equation", "simple_system", "complex_system"]


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

    # ---- Symbolic-regression helpers (Angle 1: tool > scaffold) ----
    # These functions implement what gpt-4o-mini struggles to do by reasoning
    # alone: log-log regression to find power-law exponents from data.

    def log_log_fit(x_array, y_array):
        """Fit log(y) = a*log(x) + b. Returns dict with slope, intercept, r2.
        For F = C * x^a, slope=a, exp(intercept)=C. If r2 > ~0.98 → x is a
        single power-law contributor; lower r2 → non-trivial functional form."""
        x = np.asarray(x_array, dtype=float)
        y = np.asarray(y_array, dtype=float)
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return {"error": "need >=2 positive finite points",
                    "n_used": int(mask.sum())}
        lx = np.log(x[mask]); ly = np.log(y[mask])
        a, b = np.polyfit(lx, ly, 1)
        pred = a * lx + b
        ss_res = float(((ly - pred) ** 2).sum())
        ss_tot = float(((ly - ly.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        return {"slope_exponent": float(a), "intercept_log": float(b),
                "constant_C": float(np.exp(b)), "r2": float(r2),
                "n_used": int(mask.sum())}

    def fit_separable_powerlaw(data_dict, target_key):
        """Multi-variate log-log linear regression.
            log(target) = c0 + sum_i a_i * log(var_i)
        Best when the true law is F = C * prod_i var_i^a_i.
        Returns dict with exponents per variable, constant C, and R^2.

        data_dict: {var_name: array_like}, including the target.
        target_key: name of dependent var (e.g. 'force').
        """
        target = np.asarray(data_dict[target_key], dtype=float)
        feat_names = [k for k in data_dict if k != target_key]
        if not feat_names:
            return {"error": "no feature columns"}
        # mask out non-positive
        mask = (target > 0) & np.isfinite(target)
        feats = []
        for name in feat_names:
            arr = np.asarray(data_dict[name], dtype=float)
            mask = mask & (arr > 0) & np.isfinite(arr)
            feats.append(arr)
        if mask.sum() < len(feat_names) + 1:
            return {"error": f"need >={len(feat_names)+1} pos points, have {int(mask.sum())}"}
        log_t = np.log(target[mask])
        log_f = np.stack([np.log(f[mask]) for f in feats], axis=1)
        # design matrix [log_f | 1]
        X = np.hstack([log_f, np.ones((mask.sum(), 1))])
        coef, residuals, rank, sv = np.linalg.lstsq(X, log_t, rcond=None)
        a_vec = coef[:-1]
        c0 = coef[-1]
        pred = X @ coef
        ss_res = float(((log_t - pred) ** 2).sum())
        ss_tot = float(((log_t - log_t.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        return {
            "exponents": {name: float(a_vec[i]) for i, name in enumerate(feat_names)},
            "constant_C": float(np.exp(c0)),
            "r2": float(r2),
            "n_used": int(mask.sum()),
            "interpretation": (
                "F = " + f"{np.exp(c0):.6g}" + " * "
                + " * ".join(
                    f"{name}^{a_vec[i]:.4f}" for i, name in enumerate(feat_names)
                )
            ),
        }

    def fit_with_sum_basis(x1_array, x2_array, y_array,
                           operations=("sum", "product", "sum_squared", "product_squared")):
        """Try several feature constructions for binary input (e.g. m1, m2),
        do log-log fit of y vs each basis. Returns ranked list by R^2.

        Useful when the law might be of form F ~ (m1+m2)^k, (m1*m2)^k, etc.
        """
        x1 = np.asarray(x1_array, dtype=float)
        x2 = np.asarray(x2_array, dtype=float)
        y = np.asarray(y_array, dtype=float)
        bases = {
            "sum": x1 + x2,
            "product": x1 * x2,
            "sum_squared": (x1 + x2) ** 2,
            "product_squared": (x1 * x2) ** 2,
            "sq_sum": x1 ** 2 + x2 ** 2,
            "sq_product": (x1 ** 2) * (x2 ** 2),
            "max": np.maximum(x1, x2),
            "min": np.minimum(x1, x2),
        }
        results = []
        for name, basis in bases.items():
            if name not in operations and not any(name.startswith(op) for op in operations):
                continue
            fit = log_log_fit(basis, y)
            fit["basis"] = name
            results.append(fit)
        results.sort(key=lambda d: d.get("r2", -1), reverse=True)
        return results

    def discover_law_auto(data_dict, target_key, variable_names=None):
        """All-in-one law discovery: try every plausible basis and return
        a ranked list of candidate functional forms by R^2 in log space.

        data_dict: {var: array}, including target_key
        target_key: name of dependent variable (e.g. 'force')
        variable_names: list of input var names (default: all keys except target)

        Returns: sorted list of dicts, each with:
          - form: human-readable formula string
          - r2: goodness-of-fit in log space
          - python_body: a `return ...` Python expression you can paste into
                         the submit_law function body (uses the variable names)
        Pick the top entry if its R^2 is close to 1.0. If multiple forms tie,
        prefer the simpler one (lower complexity).
        """
        candidates = []
        target = np.asarray(data_dict[target_key], dtype=float)
        if variable_names is None:
            variable_names = [k for k in data_dict if k != target_key]

        # --- Candidate A: full separable power-law ---
        sp = fit_separable_powerlaw(data_dict, target_key)
        if "exponents" in sp:
            exps = sp["exponents"]
            form_parts = [f"{name}^{exps[name]:.4f}" for name in variable_names]
            form = f"{sp['constant_C']:.6g} * " + " * ".join(form_parts)
            body = f"return {sp['constant_C']:.6g}" + "".join(
                f" * {n}**{exps[n]:.6f}" for n in variable_names
            )
            candidates.append({"form": form, "r2": sp.get("r2", -1),
                               "python_body": body, "kind": "separable"})

        # --- Candidate B+: sum-based for binary inputs (m1, m2 style) ---
        # Try first two non-target variables as the candidate "binary" inputs.
        non_target = variable_names
        if len(non_target) >= 2:
            v1, v2 = non_target[0], non_target[1]
            extra = non_target[2:]
            x1 = np.asarray(data_dict[v1], dtype=float)
            x2 = np.asarray(data_dict[v2], dtype=float)
            bases = {
                "sum":          x1 + x2,
                "sum_squared":  (x1 + x2) ** 2,
                "sq_sum":       x1 ** 2 + x2 ** 2,
                "product":      x1 * x2,
                "product_squared": (x1 * x2) ** 2,
                "sq_product":   (x1 ** 2) * (x2 ** 2),
            }
            for basis_name, basis_arr in bases.items():
                if not np.all((basis_arr > 0) & np.isfinite(basis_arr)):
                    continue
                # if there are extra vars (e.g. distance), regress against
                # log(basis) and log(extra_i)
                stacked = [np.log(basis_arr)]
                for e in extra:
                    stacked.append(np.log(np.maximum(
                        np.asarray(data_dict[e], dtype=float), 1e-30)))
                stacked.append(np.ones_like(basis_arr))
                X = np.stack(stacked, axis=1)
                mask = (target > 0) & np.isfinite(target) & np.all(np.isfinite(X), axis=1)
                if mask.sum() < X.shape[1] + 1:
                    continue
                coef, *_ = np.linalg.lstsq(X[mask], np.log(target[mask]), rcond=None)
                pred = X[mask] @ coef
                lt = np.log(target[mask])
                ss_res = float(((lt - pred) ** 2).sum())
                ss_tot = float(((lt - lt.mean()) ** 2).sum())
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
                exp_basis = float(coef[0])
                extra_exps = [float(c) for c in coef[1:-1]]
                C = float(np.exp(coef[-1]))
                form_parts = [f"({_render_basis(v1, v2, basis_name)})^{exp_basis:.4f}"]
                form_parts += [f"{e}^{extra_exps[i]:.4f}" for i, e in enumerate(extra)]
                form = f"{C:.6g} * " + " * ".join(form_parts)
                py_basis = _render_basis_py(v1, v2, basis_name)
                body = f"return {C:.6g} * ({py_basis})**{exp_basis:.6f}"
                for i, e in enumerate(extra):
                    body += f" * {e}**{extra_exps[i]:.6f}"
                candidates.append({"form": form, "r2": r2,
                                   "python_body": body,
                                   "kind": f"basis:{basis_name}"})

        candidates.sort(key=lambda d: d.get("r2", -1), reverse=True)
        return candidates[:8]

    def _render_basis(v1, v2, basis_name):
        if basis_name == "sum":           return f"{v1}+{v2}"
        if basis_name == "sum_squared":   return f"({v1}+{v2})^2"
        if basis_name == "sq_sum":        return f"{v1}^2+{v2}^2"
        if basis_name == "product":       return f"{v1}*{v2}"
        if basis_name == "product_squared": return f"({v1}*{v2})^2"
        if basis_name == "sq_product":    return f"{v1}^2*{v2}^2"
        return basis_name

    def _render_basis_py(v1, v2, basis_name):
        if basis_name == "sum":           return f"({v1}+{v2})"
        if basis_name == "sum_squared":   return f"({v1}+{v2})**2"
        if basis_name == "sq_sum":        return f"({v1}**2+{v2}**2)"
        if basis_name == "product":       return f"({v1}*{v2})"
        if basis_name == "product_squared": return f"({v1}*{v2})**2"
        if basis_name == "sq_product":    return f"({v1}**2 * {v2}**2)"
        return basis_name

    ns["log_log_fit"] = log_log_fit
    ns["fit_separable_powerlaw"] = fit_separable_powerlaw
    ns["fit_with_sum_basis"] = fit_with_sum_basis
    ns["discover_law_auto"] = discover_law_auto

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


# ===== Module-level SR helpers (also exposed inside _safe_python_exec) =====
# Duplicated as standalone module-level functions so the adapter can call them
# directly (the inner versions live inside _safe_python_exec's namespace).

def _log_log_fit(x_array, y_array) -> dict:
    import numpy as np
    x = np.asarray(x_array, dtype=float)
    y = np.asarray(y_array, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return {"error": "need >=2 positive finite points", "n_used": int(mask.sum())}
    lx = np.log(x[mask]); ly = np.log(y[mask])
    a, b = np.polyfit(lx, ly, 1)
    pred = a * lx + b
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return {"slope_exponent": float(a), "constant_C": float(np.exp(b)),
            "r2": float(r2), "n_used": int(mask.sum())}


def _fit_separable_powerlaw(data_dict, target_key) -> dict:
    import numpy as np
    target = np.asarray(data_dict[target_key], dtype=float)
    feat_names = [k for k in data_dict if k != target_key]
    if not feat_names:
        return {"error": "no feature columns"}
    mask = (target > 0) & np.isfinite(target)
    feats = []
    for name in feat_names:
        arr = np.asarray(data_dict[name], dtype=float)
        mask = mask & (arr > 0) & np.isfinite(arr)
        feats.append(arr)
    if mask.sum() < len(feat_names) + 1:
        return {"error": f"need >={len(feat_names)+1} pos points",
                "n_used": int(mask.sum())}
    log_t = np.log(target[mask])
    log_f = np.stack([np.log(f[mask]) for f in feats], axis=1)
    X = np.hstack([log_f, np.ones((mask.sum(), 1))])
    coef, *_ = np.linalg.lstsq(X, log_t, rcond=None)
    a_vec = coef[:-1]; c0 = coef[-1]
    pred = X @ coef
    ss_res = float(((log_t - pred) ** 2).sum())
    ss_tot = float(((log_t - log_t.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return {"exponents": {name: float(a_vec[i]) for i, name in enumerate(feat_names)},
            "constant_C": float(np.exp(c0)), "r2": float(r2),
            "n_used": int(mask.sum())}


def _render_basis_py_mod(v1, v2, basis_name):
    if basis_name == "sum":             return f"({v1}+{v2})"
    if basis_name == "sum_squared":     return f"({v1}+{v2})**2"
    if basis_name == "sq_sum":          return f"({v1}**2+{v2}**2)"
    if basis_name == "product":         return f"({v1}*{v2})"
    if basis_name == "product_squared": return f"({v1}*{v2})**2"
    if basis_name == "sq_product":      return f"({v1}**2 * {v2}**2)"
    return basis_name


def _angle_like(arr) -> str | None:
    """Heuristic: variable looks like an angle. Returns its likely UNIT:
       'rad'  if range fits inside [0, 2π]
       'deg'  if range fits inside [-360, 360] but exceeds π (i.e. larger
              than a radian interpretation could plausibly justify)
       None   otherwise.
    """
    import numpy as np
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return None
    mn, mx = float(a.min()), float(a.max())
    span = mx - mn
    if span < 0.05:
        return None
    # Radians range: ≤ 2π
    if (mn >= -0.01) and (mx <= 6.5):
        return "rad"
    # Degrees range: typical angles 0..180 (Snell, Malus) or up to 360
    if (mn >= -360.0) and (mx <= 360.0) and (mx > 4.0):
        return "deg"
    return None


def _fit_alt_basis(data_dict, target_key, variable_names, alt_var, alt_kind,
                   angle_unit="rad"):
    """Fit log(y) = a*g(alt_var) + sum_other a_i*log(other_var_i) + c, where
    g is one of:
      - 'exp':  use raw alt_var (so y = C * exp(a*alt_var) * prod ...)
      - 'sin':  use log(sin(alt_var))     # alt_var assumed in `angle_unit`
      - 'cos':  use log(cos(alt_var))
      - 'sin2': use log(sin(alt_var)^2)
      - 'cos2': use log(cos(alt_var)^2)
    `angle_unit` ∈ {'rad', 'deg'} — for trig kinds; converts input via
    np.radians if 'deg'. Each call produces a python_body that ALSO bakes
    in the unit conversion explicitly so the resulting law runs correctly
    against the oracle's native unit.
    Returns dict with form, r2, python_body, or None on failure."""
    import numpy as np
    target = np.asarray(data_dict[target_key], dtype=float)
    others = [v for v in variable_names if v != alt_var]
    alt = np.asarray(data_dict[alt_var], dtype=float)

    # Apply unit-aware angle conversion for trig kinds
    if alt_kind in ("sin", "cos", "sin2", "cos2"):
        if angle_unit == "deg":
            alt_calc = np.radians(alt)
        else:
            alt_calc = alt
    else:
        alt_calc = alt

    # Build alt feature
    if alt_kind == "exp":
        alt_feat = alt_calc
    elif alt_kind in ("sin", "sin2"):
        s = np.sin(alt_calc)
        mask_pos = s > 1e-12
        with np.errstate(invalid="ignore", divide="ignore"):
            alt_feat = np.where(mask_pos, np.log(s), np.nan)
        if alt_kind == "sin2":
            alt_feat = 2.0 * alt_feat
    elif alt_kind in ("cos", "cos2"):
        c = np.cos(alt_calc)
        mask_pos = c > 1e-12
        with np.errstate(invalid="ignore", divide="ignore"):
            alt_feat = np.where(mask_pos, np.log(c), np.nan)
        if alt_kind == "cos2":
            alt_feat = 2.0 * alt_feat
    else:
        return None

    stacked = [alt_feat]
    for v in others:
        stacked.append(np.log(np.maximum(
            np.asarray(data_dict[v], dtype=float), 1e-30)))
    stacked.append(np.ones_like(alt_feat))
    X = np.stack(stacked, axis=1)
    mask = (target > 0) & np.isfinite(target) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < X.shape[1] + 1:
        return None
    log_t = np.log(target[mask])
    coef, *_ = np.linalg.lstsq(X[mask], log_t, rcond=None)
    pred = X[mask] @ coef
    ss_res = float(((log_t - pred) ** 2).sum())
    ss_tot = float(((log_t - log_t.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    a_alt = float(coef[0])
    other_exps = [float(c) for c in coef[1:-1]]
    C = float(np.exp(coef[-1]))

    # Render python body — bake in the unit conversion explicitly when needed
    var_expr = alt_var if angle_unit == "rad" else f"math.radians({alt_var})"
    if alt_kind == "exp":
        body = f"return {C:.6g} * (2.718281828)**({a_alt:.6f}*{alt_var})"
        form = f"{C:.6g} * exp({a_alt:.4f}*{alt_var})"
    elif alt_kind == "sin":
        body = f"return {C:.6g} * abs(math.sin({var_expr}))**({a_alt:.6f})"
        form = f"{C:.6g} * sin({alt_var}[{angle_unit}])^{a_alt:.4f}"
    elif alt_kind == "sin2":
        body = f"return {C:.6g} * (math.sin({var_expr}))**({2*a_alt:.6f})"
        form = f"{C:.6g} * sin({alt_var}[{angle_unit}])^{2*a_alt:.4f}"
    elif alt_kind == "cos":
        body = f"return {C:.6g} * abs(math.cos({var_expr}))**({a_alt:.6f})"
        form = f"{C:.6g} * cos({alt_var}[{angle_unit}])^{a_alt:.4f}"
    elif alt_kind == "cos2":
        body = f"return {C:.6g} * (math.cos({var_expr}))**({2*a_alt:.6f})"
        form = f"{C:.6g} * cos({alt_var}[{angle_unit}])^{2*a_alt:.4f}"
    for i, v in enumerate(others):
        body += f" * {v}**{other_exps[i]:.6f}"
        form += f" * {v}^{other_exps[i]:.4f}"
    suffix = "" if angle_unit == "rad" else "_deg"
    return {"kind": f"{alt_kind}_on_{alt_var}{suffix}", "form": form, "r2": r2,
            "python_body": body}


def _discover_law_auto(data_dict, target_key="force", variable_names=None) -> list[dict]:
    """Try multiple bases; return ranked candidates by R^2."""
    import numpy as np
    if variable_names is None:
        variable_names = [k for k in data_dict if k != target_key]
    target = np.asarray(data_dict[target_key], dtype=float)
    out: list[dict] = []

    # Candidate A: separable power-law
    sp = _fit_separable_powerlaw(data_dict, target_key)
    if "exponents" in sp:
        exps = sp["exponents"]
        body = f"return {sp['constant_C']:.6g}" + "".join(
            f" * {n}**{exps[n]:.6f}" for n in variable_names
        )
        form = f"{sp['constant_C']:.6g}" + "".join(
            f" * {n}^{exps[n]:.4f}" for n in variable_names
        )
        out.append({"kind": "separable", "form": form, "r2": sp["r2"],
                    "python_body": body})

    # Candidates B+: binary-input bases (first 2 vars as m1, m2)
    non_target = variable_names
    if len(non_target) >= 2:
        v1, v2 = non_target[0], non_target[1]
        extras = non_target[2:]
        x1 = np.asarray(data_dict[v1], dtype=float)
        x2 = np.asarray(data_dict[v2], dtype=float)
        bases = {
            "sum": x1 + x2, "sum_squared": (x1 + x2) ** 2,
            "sq_sum": x1 ** 2 + x2 ** 2,
            "product": x1 * x2, "product_squared": (x1 * x2) ** 2,
            "sq_product": (x1 ** 2) * (x2 ** 2),
        }
        for bname, barr in bases.items():
            valid = (barr > 0) & np.isfinite(barr)
            if not valid.all():
                continue
            stacked = [np.log(barr)]
            for e in extras:
                stacked.append(np.log(np.maximum(
                    np.asarray(data_dict[e], dtype=float), 1e-30)))
            stacked.append(np.ones_like(barr))
            X = np.stack(stacked, axis=1)
            mask = (target > 0) & np.isfinite(target) & np.all(np.isfinite(X), axis=1)
            if mask.sum() < X.shape[1] + 1:
                continue
            coef, *_ = np.linalg.lstsq(X[mask], np.log(target[mask]), rcond=None)
            pred = X[mask] @ coef
            lt = np.log(target[mask])
            ss_res = float(((lt - pred) ** 2).sum())
            ss_tot = float(((lt - lt.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
            exp_basis = float(coef[0])
            extra_exps = [float(c) for c in coef[1:-1]]
            C = float(np.exp(coef[-1]))
            py_basis = _render_basis_py_mod(v1, v2, bname)
            body = f"return {C:.6g} * ({py_basis})**{exp_basis:.6f}"
            form = f"{C:.6g} * ({v1} {bname} {v2})^{exp_basis:.4f}"
            for i, e in enumerate(extras):
                body += f" * {e}**{extra_exps[i]:.6f}"
                form += f" * {e}^{extra_exps[i]:.4f}"
            out.append({"kind": f"basis:{bname}", "form": form,
                        "r2": r2, "python_body": body})

    # Candidates C+: alternative bases (exp/sin/cos/sin²/cos²) per variable
    # — for laws like Snell, Malus, radioactive decay, Boltzmann distribution.
    # For trig: try BOTH unit interpretations (rad / deg) when the variable's
    # range is ambiguous; pick the one with higher R^2 implicitly by ranking.
    for v in variable_names:
        arr = np.asarray(data_dict[v], dtype=float)
        # Always try exp (works for time/dose variables)
        c = _fit_alt_basis(data_dict, target_key, variable_names, v, "exp")
        if c is not None:
            out.append(c)
        # Try trig only if variable looks like an angle
        unit = _angle_like(arr)
        if unit is not None:
            for kind in ("sin", "cos", "sin2", "cos2"):
                # Always try the inferred unit first
                c = _fit_alt_basis(data_dict, target_key, variable_names,
                                   v, kind, angle_unit=unit)
                if c is not None:
                    out.append(c)
                # If the inferred unit was rad but range allows it, also
                # try deg interpretation (sometimes a small radians-range
                # value is actually degrees). Conversely if inferred deg,
                # also try rad. The right one wins on R^2.
                other_unit = "deg" if unit == "rad" else "rad"
                # Skip the other unit only if it would produce out-of-domain
                # values (radians > 2π or degrees < 0)
                if (other_unit == "deg" and float(arr[np.isfinite(arr)].max()) > 0.05) \
                   or (other_unit == "rad" and float(arr[np.isfinite(arr)].max()) <= 6.5):
                    c = _fit_alt_basis(data_dict, target_key, variable_names,
                                       v, kind, angle_unit=other_unit)
                    if c is not None:
                        out.append(c)

    out.sort(key=lambda d: d.get("r2", -1), reverse=True)
    return out


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
    """Enumerate task configurations. Default = official 12 x 3 x 3 x 3 = 324
    task grid (x noise_levels x trials = configurable)."""
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
        auto_fit_in_results: bool = True,    # toggle: surface auto-fit
                                              # candidates in every batch
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
        # Accumulated experimental data — used by auto-fit on every batch
        self._all_inputs: list[dict[str, float]] = []
        self._all_outputs: list[float] = []
        self._auto_fit_in_results = bool(auto_fit_in_results)
        self._verbose_override = bool(os.environ.get("MARS_NB_VERBOSE_OVERRIDE", "0") not in ("0", "false", "no"))

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
            f"**RECOMMENDED WORKFLOW — DO NOT GUESS STANDARD NEWTON'S LAW!**\n"
            f"The universe has metaphysical-shifted physics. Newton's "
            f"`F = G*m1*m2/r^2` will be WRONG. Use the symbolic-regression "
            f"tool to FIND the actual law from data. The fastest path:\n\n"
            f"  Rounds 1-3: `run_experiment` to collect ~15-25 data points "
            f"spanning orders of magnitude for each input variable.\n"
            f"  Round 4: ONE `python_exec` call:\n\n"
            f"      data = {{\n"
            f"          'mass1':   [...],   # from your experiments\n"
            f"          'mass2':   [...],\n"
            f"          'distance':[...],\n"
            f"          'force':   [...],\n"
            f"      }}\n"
            f"      candidates = discover_law_auto(data, target_key='force')\n"
            f"      for c in candidates[:5]:\n"
            f"          print(f\"R2={{c['r2']:.4f}} {{c['kind']}}: F = {{c['form']}}\")\n"
            f"          print(f\"  body: {{c['python_body']}}\")\n\n"
            f"  Round 5+: `submit_law` using the python_body of the BEST candidate "
            f"(highest R^2, ideally > 0.999). Just paste the body into:\n\n"
            f"      def discovered_law(mass1, mass2, distance):\n"
            f"          <PASTE python_body HERE>\n\n"
            f"  If multiple candidates have R^2 > 0.99, prefer the SIMPLER one "
            f"(e.g. prefer 'sum_squared' over 'separable' if both fit).\n"
            f"  Note: numpy is preloaded as `np` — DO NOT `import numpy`.\n\n"
            f"  Rounds 1-3: `run_experiment` to sweep each input variable across "
            f"orders of magnitude (e.g. distance ∈ [0.1, 1, 10, 100, 1000]) — "
            f"vary ONE variable at a time, keep others fixed.\n"
            f"  Round 4: `python_exec` — call the BUILT-IN symbolic-regression "
            f"helpers to extract scaling exponents from your data. You DON'T need "
            f"to write polyfit yourself, just use these:\n\n"
            f"      # log-log fit of single variable (returns slope=exponent, R^2)\n"
            f"      r = [0.1, 1.0, 10.0, 100.0]\n"
            f"      F = [F1, F2, F3, F4]  # measured forces\n"
            f"      print(log_log_fit(r, F))\n"
            f"      #  → {{'slope_exponent': -1.5, 'constant_C': 6.67e-5, 'r2': 0.99}}\n\n"
            f"      # multi-variate power-law fit (best when F = C * v1^a * v2^b * v3^c)\n"
            f"      data = {{'mass1': m1_list, 'mass2': m2_list, 'distance': r_list,\n"
            f"              'force': F_list}}\n"
            f"      print(fit_separable_powerlaw(data, target_key='force'))\n"
            f"      #  → {{'exponents': {{'mass1': 1.0, 'mass2': 1.0, 'distance': -1.5}},\n"
            f"      #      'constant_C': 6.67e-5, 'r2': 0.9995}}\n\n"
            f"      # If the law might involve (m1+m2) or (m1^2+m2^2), try:\n"
            f"      print(fit_with_sum_basis(m1_list, m2_list, F_list))\n"
            f"      #  → ranked list by R^2 of bases: 'sum_squared', 'product', ...\n\n"
            f"  Rounds 5-7: refine — run additional `run_experiment` to verify the "
            f"top-R² form on held-out inputs.\n"
            f"  Rounds 8-9: `submit_law` with the discovered functional form. The "
            f"function body should use the exponents and constant from the fit.\n\n"
            f"**HIGH R² (> 0.99) means you've likely found the right form.** Lower "
            f"R² means try a different basis (additive vs multiplicative).\n\n"
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

    # -- completeness gate (MARS-SELF) --------------------------------------

    def completeness_rubric(self) -> list[RubricSection]:
        """Programmatic Completeness Gate for law discovery (MARS-SELF).

        Blocks the canonical gpt-4o-mini failure mode: submitting memorised
        textbook physics WITHOUT gathering data. Two predicate-driven sections:

          1. sufficient_data — must have ≥8 accumulated finite experiments
             spanning the input space before a law can be submitted.
          2. fit_grounded   — the built-in symbolic-regression auto-fit must
             have found a candidate with R² ≥ 0.90 on the accumulated data;
             this guarantees the agent is submitting an empirically-grounded
             form rather than a guess.

        Both use the adapter's live state via the predicate context.
        """
        def _has_enough_data(ctx) -> bool:
            ad = ctx.get("adapter")
            return bool(ad is not None and len(getattr(ad, "_all_outputs", [])) >= 8)

        def _fit_is_grounded(ctx) -> bool:
            ad = ctx.get("adapter")
            if ad is None:
                return False
            try:
                cands = ad._auto_fit_summary(top_k=3)
                if isinstance(cands, list) and cands:
                    return float(cands[0].get("r2", 0.0)) >= 0.90
            except Exception:
                return False
            return False

        return [
            RubricSection(
                name="sufficient_data",
                hint=("gather more experimental data — run_experiment across "
                      "orders of magnitude (need ≥8 finite data points)"),
                predicate=_has_enough_data,
            ),
            RubricSection(
                name="fit_grounded",
                hint=("ground the law in data — run python_exec / inspect the "
                      "auto_discovery_top_candidates; the best fit must reach "
                      "R² ≥ 0.90 before submitting"),
                predicate=_fit_is_grounded,
            ),
        ]

    def submit_action_names(self) -> set[str]:
        return {"submit_law"}

    # -- SRHP hooks (hypothesis = the law program itself) -------------------

    def supports_srhp(self) -> bool:
        return True

    def _srhp_params(self) -> list[str]:
        sig = str(self._module.FUNCTION_SIGNATURE).strip()
        try:
            inside = sig[sig.index("(") + 1: sig.rindex(")")]
            return [p.strip() for p in inside.split(",") if p.strip()]
        except Exception:
            return []

    def srhp_spec(self) -> dict:
        params = self._srhp_params()
        # Use a SHORT brief — the full task prompt contains a "DO NOT GUESS"
        # workflow that makes the model refuse to conjecture. SRHP wants guesses.
        try:
            pdesc = str(getattr(self._module, "PARAM_DESCRIPTION", "") or "")[:500]
        except Exception:
            pdesc = ""
        return {
            "fn_name": "discovered_law",
            "signature": str(self._module.FUNCTION_SIGNATURE).strip(),
            "input_keys": params,
            "output_desc": ("a single float — the law's output for the given "
                            "inputs (the underlying physics may be non-standard)"),
            "description": (
                f"Discover the hidden physical law mapping inputs "
                f"({', '.join(params)}) to one numeric output. The universe's "
                f"physics may differ from textbook formulas — fit the observed "
                f"data. Parameters: {pdesc}"
            ),
        }

    def srhp_candidate_experiments(self, n: int = 16) -> list[dict]:
        import numpy as np
        params = self._srhp_params()
        if not params:
            return []
        # angle-like variables need a domain-appropriate range — sampling an
        # angle in [0.1, 100] makes sin/cos garbage and breaks trig-law fitting.
        ANGLE_HINTS = ("theta", "angle", "phi", "alpha", "incid", "polar",
                       "tilt", "deg", "rad")
        def _is_angle(p):
            pl = p.lower()
            return any(h in pl for h in ANGLE_HINTS)

        angle_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]      # radians 0..~85°
        scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
        defaults = {q: (0.7 if _is_angle(q) else 1.0) for q in params}
        cands: list[dict] = []
        # one-variable-at-a-time sweeps with per-variable appropriate ranges
        for p in params:
            grid = angle_vals if _is_angle(p) else scales
            for s in grid:
                exp = dict(defaults)
                exp[p] = float(s)
                cands.append(exp)
        # random combinations (angles in [0.05, 1.55] rad, others log-uniform)
        rng = np.random.default_rng(12345 + self.task.trial_id)
        for _ in range(max(n, 8)):
            row = {}
            for p in params:
                if _is_angle(p):
                    row[p] = float(rng.uniform(0.05, 1.55))
                else:
                    row[p] = float(np.exp(rng.uniform(np.log(0.1), np.log(100.0))))
            cands.append(row)
        return cands

    def srhp_run(self, inputs: dict) -> float:
        # spend one round (matches run_experiment cost)
        self._budget_spent += 1.0
        try:
            r = self._module.run_experiment_for_module(
                noise_level=self.task.noise_level,
                difficulty=self.task.difficulty,
                system=self.task.system,
                law_version=self.task.law_version,
                **{k: float(v) for k, v in inputs.items()},
            )
            y = float(r)
            # accumulate for the tool-sovereignty fallback in score_episode
            self._all_inputs.append({k: float(v) for k, v in inputs.items()})
            self._all_outputs.append(y)
            return y
        except Exception:
            return float("nan")

    def srhp_finalize(self, program_code: str, fn_name: str) -> None:
        if not program_code:
            return
        # ensure the function is named discovered_law for the evaluator
        code = program_code
        if fn_name != "discovered_law":
            code = code.replace(f"def {fn_name}", "def discovered_law", 1)
        self._submitted_law = code
        self._all_submitted_attempts.append(code)

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
            # accumulate inputs/outputs across rounds for the auto-fit hook
            import math as _math
            for exp, r_val in zip(experiments, results):
                try:
                    if isinstance(r_val, str) and r_val.startswith("error"):
                        continue
                    y = float(r_val)
                    if not _math.isfinite(y):
                        continue
                    self._all_inputs.append({k: float(v) for k, v in exp.items()
                                              if isinstance(v, (int, float))})
                    self._all_outputs.append(y)
                except (TypeError, ValueError):
                    continue
            summary = {
                "n_experiments": len(experiments),
                "outputs": results,
            }
            if self._auto_fit_in_results:
                summary["auto_discovery_top_candidates"] = self._auto_fit_summary()
                summary["_hint"] = (
                    "auto_discovery_top_candidates was computed across ALL "
                    "experiments accumulated so far. If the top candidate has "
                    "R^2 > 0.99, copy its 'python_body' verbatim into "
                    "submit_law. DO NOT substitute textbook Newton — the "
                    "ground-truth law in this universe is non-standard."
                )
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

    # -- auto-replace: tool sovereignty when R^2 > 0.99 ---------------------

    def _maybe_auto_replace_submission(self, r2_threshold: float = 0.99) -> str | None:
        """If the top auto-fit candidate fits the collected data with R^2
        above threshold AND its predictions beat the agent's submitted law on
        the collected data, return the auto-fit's python_body (to override
        the agent's submission). Otherwise return None.

        This is documented in the paper as 'tool sovereignty': when a
        symbolic-regression tool has mathematical certainty (R^2 > 0.99 on
        accumulated experiments), we let the tool override LLM-bias toward
        memorized textbook physics.
        """
        if len(self._all_outputs) < 4:
            return None
        try:
            import numpy as np
            import math as _math
            cands = self._auto_fit_summary(top_k=12)
            if not isinstance(cands, list) or not cands:
                return None
            # Filter to candidates that fit very well in log space
            good = [c for c in cands if c.get("r2", 0.0) >= r2_threshold]
            if not good:
                return None
            # Among good candidates, prefer the one with lowest REAL-SPACE
            # RMSLE on accumulated data. This breaks ties between R^2-near-1
            # candidates (e.g. separable vs sum_squared when both are near
            # perfect in log space, but sum_squared is the true form).
            sig = self._module.FUNCTION_SIGNATURE.strip()
            try:
                argstr = sig[sig.index("(") + 1:sig.rindex(")")]
                params = [p.strip() for p in argstr.split(",") if p.strip()]
            except Exception:
                return None
            ys = np.array(self._all_outputs, dtype=float)

            def _real_rmsle(body: str) -> float:
                code = f"def __c({', '.join(params)}):\n    {body}"
                import math as _math
                ns: dict = {"__builtins__": {"abs": abs, "min": min,
                                              "max": max, "sum": sum,
                                              "round": round, "pow": pow},
                            "math": _math}
                try:
                    exec(code, ns)
                    fn = ns["__c"]
                except Exception:
                    return float("inf")
                preds = []
                for inp in self._all_inputs:
                    try:
                        ad = {p: inp.get(p, 0.0) for p in params}
                        preds.append(fn(**ad))
                    except Exception:
                        preds.append(float("nan"))
                pr = np.array(preds, dtype=float)
                m = np.isfinite(pr) & (pr > 0) & np.isfinite(ys) & (ys > 0)
                if m.sum() < 3:
                    return float("inf")
                return float(np.sqrt(
                    ((np.log(pr[m]) - np.log(ys[m])) ** 2).mean()
                ))

            # Score each good candidate by real-space RMSLE
            scored: list[tuple[float, dict]] = []
            for c in good:
                rmsle = _real_rmsle(c["python_body"])
                scored.append((rmsle, c))
            scored.sort(key=lambda t: t[0])
            top = scored[0][1]
            best_rmsle = scored[0][0]
            if best_rmsle >= 0.5:
                return None
            # set top.r2 to actual R^2 from log-space (already in c["r2"])
            # Build the auto-fit law as a Python function and predict on
            # the accumulated data
            tool_body = top["python_body"]
            tool_rmsle = best_rmsle
            # Compute agent's RMSLE on the same accumulated data
            agent_rmsle = float("inf")
            if self._submitted_law:
                import math as _math
                agent_ns: dict = {"__builtins__": {"abs": abs, "min": min,
                                                    "max": max, "sum": sum,
                                                    "round": round, "pow": pow},
                                  "math": _math}
                try:
                    exec(self._submitted_law, agent_ns)
                    agent_fn = agent_ns.get("discovered_law")
                    if agent_fn is not None:
                        agent_pred = []
                        for inp in self._all_inputs:
                            try:
                                ad = {p: inp.get(p, 0.0) for p in params}
                                agent_pred.append(agent_fn(**ad))
                            except Exception:
                                agent_pred.append(float("nan"))
                        ap = np.array(agent_pred, dtype=float)
                        am = np.isfinite(ap) & (ap > 0) & np.isfinite(ys) & (ys > 0)
                        if am.sum() >= 3:
                            agent_rmsle = float(np.sqrt(
                                ((np.log(ap[am]) - np.log(ys[am])) ** 2).mean()
                            ))
                except Exception:
                    pass
            # Override conditions (any one is sufficient):
            #   (i)  tool clearly wins: tool_rmsle < 0.5 AND < 0.5x agent
            #   (ii) tool is excellent (rmsle<0.1) while agent is poor (>0.5)
            #   (iii) agent's submission is the memorised-Newton failure mode
            #        (RMSLE > 1.0) AND tool's R^2 > 0.99 with reasonable rmsle.
            do_override = False
            reason = ""
            if tool_rmsle < 0.5 and tool_rmsle < agent_rmsle * 0.5:
                do_override = True; reason = f"clear win (tool {tool_rmsle:.3f} vs agent {agent_rmsle:.3f})"
            elif tool_rmsle < 0.1 and agent_rmsle > 0.5:
                do_override = True; reason = f"excellent tool {tool_rmsle:.3f}"
            elif agent_rmsle > 1.0 and top.get("r2", 0.0) > 0.99 and tool_rmsle < 1.0:
                do_override = True; reason = f"agent regress, tool R2={top['r2']:.4f}"
            if do_override:
                if self._verbose_override:
                    print(f"[NB tool override] {reason}: replacing agent law "
                          f"with: {tool_body[:120]}")
                tool_code = (f"def __tool_law({', '.join(params)}):\n"
                             f"    {tool_body}")
                return tool_code.replace("__tool_law", "discovered_law")
            return None
        except Exception:
            return None

    # -- auto-fit hook (runs after each batch; surfaces top candidates) ----

    def _auto_fit_summary(self, top_k: int = 4) -> list[dict] | str:
        """Run discover_law_auto on all accumulated experimental data; return
        a compact list of top candidate functional forms. This is mandatorily
        shown to the agent after every run_experiment so it can't fall back
        to textbook physics if the auto-fit found something better."""
        if len(self._all_outputs) < 3:
            return f"(need >=3 finite points, have {len(self._all_outputs)})"
        try:
            import numpy as np
            # Build column-oriented data dict from accumulated rows
            keys: set[str] = set()
            for row in self._all_inputs:
                keys.update(row.keys())
            data: dict[str, list[float]] = {k: [] for k in keys}
            ys: list[float] = []
            for row, y in zip(self._all_inputs, self._all_outputs):
                if all(k in row for k in keys):
                    for k in keys:
                        data[k].append(row[k])
                    ys.append(y)
            if len(ys) < 3:
                return f"(need >=3 complete rows, have {len(ys)})"
            data_np = {k: np.array(v, dtype=float) for k, v in data.items()}
            data_np["force"] = np.array(ys, dtype=float)
            # Use the local helpers from _safe_python_exec; re-implement here
            # to avoid re-entry into the threaded exec wrapper.
            cands = _discover_law_auto(data_np, target_key="force",
                                       variable_names=sorted(keys))
            return cands[:top_k]
        except Exception as e:
            return f"(auto-fit error: {type(e).__name__}: {str(e)[:120]})"

    # -- ground-truth (no per-claim oracle; episode-level law eval) ----

    def verify_claim(self, claim: Claim) -> ClaimVerdict | None:
        return None

    # -- scoring ----

    def _synthesize_final_law(self) -> str:
        """Last-resort LLM call: given the action log, emit a Python law."""
        try:
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
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
        if final_artifact:
            self._submitted_law = str(final_artifact).strip()

        # Fallback: if agent never submitted, synthesize a final law from the
        # action log via one extra LLM call. Mirrors NewtonBench's own
        # "force final submission" fallback.
        if not self._submitted_law and self._action_log:
            self._submitted_law = self._synthesize_final_law()

        # TOOL SOVEREIGNTY (Angle 1 mechanism): if the SR-tool has
        # mathematical certainty on the accumulated experimental data
        # (R^2 > 0.99) AND its predictions beat the agent's submitted law
        # by ≥2× in RMSLE, override the agent's submission with the tool's
        # auto-fit. Reported in paper as a distinct "tool override" event
        # alongside agent-native submissions.
        self._was_tool_override = False
        override = self._maybe_auto_replace_submission(r2_threshold=0.99)
        if override:
            self._submitted_law = override
            self._was_tool_override = True

        if not self._submitted_law:
            return {
                "primary": 0.0, "SA": 0.0, "rmsle": float("nan"),
                "symbolic_equivalent": False, "tool_override": False,
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
        rmsle = ev.get("rmsle", float("nan"))
        # NB's SA is BINARY symbolic equivalence — brutal for small models by
        # design. RMSLE (numerical fit error) is the continuous metric where a
        # forced-data-gathering scaffold + tool sovereignty actually show up.
        # numerical_accuracy = exp(-rmsle) ∈ (0,1]: 1.0 = perfect fit, decays
        # with log-error. Reported alongside SA so small-model gains are visible.
        try:
            import math as _m
            na = float(_m.exp(-rmsle)) if rmsle == rmsle else 0.0  # NaN check
        except Exception:
            na = 0.0
        return {
            "primary": sa,
            "SA": sa,
            "numerical_accuracy": na,
            "rmsle": rmsle,
            "symbolic_equivalent": bool(ev.get("symbolic_equivalent", False)),
            "symbolic_msg": str(ev.get("symbolic_msg", ""))[:300],
            "tool_override": bool(getattr(self, "_was_tool_override", False)),
            "submitted": self._submitted_law[:500],
            "explain": str(ev.get("symbolic_msg", ""))[:300],
        }
