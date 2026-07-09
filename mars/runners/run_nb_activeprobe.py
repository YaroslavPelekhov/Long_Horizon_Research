"""NewtonBench: Universal Active Probing — controlled experiments → exact law recovery.

Core idea: vary one variable at a time (controlled experiment), detect functional form
from probe data using universal math (log-log slope, exp fit, trig residuals), assemble
law, verify by held-out rmsle. NO model prior used — data-driven only.

Why this beats fitter/selfdebug: counterfactual laws break model priors (e.g. gravity
with no mass term). Controlled probing has no prior; log-log slope reads the ACTUAL
exponent from data, not from "gravity = G*m1*m2/r^2" knowledge.

Universal shell: works on ANY parametric law in any domain where experiments can be run.

  python -m mars.runners.run_nb_activeprobe --run_id nbap_v1 --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
_PROJ = Path(__file__).resolve().parent.parent.parent
_NB = _PROJ / "newtonbench_repo"
for _p in (_PROJ, _NB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=True)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.induction.operator_genome import (
    MeasurementTrace,
    induce_univariate_operators,
    summarize_operator_genome,
)
from mars.induction.self_induced_language import (
    CoordinateProgram,
    NumericTrace,
    SelfInducedLanguage,
    _phase_coordinates,
)
from mars.runners.run_nb_selfverify import _params, collect
from mars.runners.run_nb_fitters import _law_from_source, _rmsle
from mars.induction.nb_cpi import NBForcePoint, fit_law_candidates

WEAK = "openai/gpt-4o-mini"
MODULES = [
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
EXP_ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0]
EXPONENT_GRID = [
    -4.0,
    -3.0,
    -math.e,
    -2.6,
    -2.5,
    -2.0,
    -1.5,
    -1.3,
    -1.25,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.3,
    1.5,
    1.75,
    2.0,
    2.5,
    2.6,
    math.e,
    3.0,
    4.0,
]
TRIG_TRANSFORMS = {
    "sin": lambda x: np.sin(x),
    "cos": lambda x: np.cos(x),
    "sin2": lambda x: np.sin(x) ** 2,
    "cos2": lambda x: np.cos(x) ** 2,
    "sin2x": lambda x: np.sin(2 * x),
    "cos2x": lambda x: np.cos(2 * x),
    "1+sin2x": lambda x: 1 + np.sin(2 * x),
    "1+cos2x": lambda x: 1 + np.cos(2 * x),
    "1+sinx": lambda x: 1 + np.sin(x),
    "1+cosx": lambda x: 1 + np.cos(x),
}


def snap_exponent(value: float, *, tolerance: float = 0.03) -> float:
    """Close a measured exponent hole against a small scientific constants grid."""
    best = min(EXPONENT_GRID, key=lambda g: abs(float(g) - float(value)))
    return float(best) if abs(float(best) - float(value)) <= tolerance else float(value)


_ALIAS_CACHE = {}


def _experiment_aliases(module, n: int) -> list[str]:
    """Infer observable API kwarg names from the experiment function.

    NewtonBench sometimes evaluates `discovered_law(gamma, T, M)` but the
    experiment API exposes `adiabatic_index, temperature, molar_mass`.  This
    is interface mechanics, not law knowledge; extracting it prevents probes
    from accidentally hitting defaults and measuring a constant environment.
    """
    key = (getattr(module, "__name__", str(module)), n)
    if key in _ALIAS_CACHE:
        return list(_ALIAS_CACHE[key])
    try:
        src = inspect.getsource(module.run_experiment_for_module)
    except Exception:
        _ALIAS_CACHE[key] = []
        return []
    aliases = []
    for match in re.finditer(r"kwargs\.get\(\s*['\"]([^'\"]+)['\"]", src):
        name = match.group(1)
        if name not in aliases:
            aliases.append(name)
    # The vanilla branch appears before simple/complex branches, so the first
    # n names are the scalar-law observable inputs when an alias mismatch exists.
    aliases = aliases[:n] if len(aliases) >= n else []
    _ALIAS_CACHE[key] = aliases
    return list(aliases)


def _translate_kwargs(module, inp):
    names = list(inp.keys())
    aliases = _experiment_aliases(module, len(names))
    if not aliases or aliases == names:
        return dict(inp)
    out = dict(inp)
    for src, dst in zip(names, aliases):
        out[dst] = inp[src]
    return out


def run_exp(module, inp, difficulty="easy", law_version="v0", system="vanilla_equation"):
    try:
        kwargs = _translate_kwargs(module, inp)
        y = float(module.run_experiment_for_module(
            noise_level=0.0, difficulty=difficulty,
            system=system, law_version=law_version, **kwargs))
        return y if math.isfinite(y) else None
    except Exception:
        return None


def collect_active(module, params, difficulty="easy", law_version="v0",
                   system="vanilla_equation", n=24, seed=11):
    import random
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        inp = {p: round(rng.uniform(0.5, 5.0), 3) for p in params}
        y = run_exp(module, inp, difficulty=difficulty,
                    law_version=law_version, system=system)
        if y is not None and y == y:
            rows.append({**inp, "_y": y})
    return rows


def collect_stress_holdout(module, params, rows, *, difficulty="easy",
                           law_version="v0", system="vanilla_equation",
                           n=80, seed=29):
    """Generate verifier rows that stress likely hidden coordinates.

    Passive held-out rows can be too local.  This verifier expands angle-like
    coordinates while sampling the remaining variables from already valid
    observations, so local coordinate surrogates are refuted.
    """

    import random
    rng = random.Random(seed)
    if not any(_angle_like_param(p) for p in params) or not rows:
        return []
    values = {
        p: [float(r[p]) for r in rows if p in r and math.isfinite(float(r[p]))]
        for p in params
    }
    out = []
    for _ in range(n):
        inp = {}
        for p in params:
            if _angle_like_param(p):
                low = p.lower()
                if "theta" in low or "phase" in low:
                    inp[p] = rng.uniform(0.05, math.pi / 2 - 0.02)
                else:
                    inp[p] = rng.uniform(1.0, 80.0)
            else:
                vals = values.get(p) or [1.0]
                inp[p] = rng.choice(vals)
        y = run_exp(module, inp, difficulty=difficulty,
                    law_version=law_version, system=system)
        if y is not None and y == y and math.isfinite(y):
            out.append({**inp, "_y": y})
    return out


def _median_anchor(rows, params):
    anchor = {}
    for p in params:
        vals = [float(r[p]) for r in rows if p in r and float(r[p]) > 0 and math.isfinite(float(r[p]))]
        anchor[p] = float(np.median(vals)) if vals else 1.0
    return anchor


def _intervention_values(rows, target, n=15):
    vals = [float(r[target]) for r in rows if target in r and float(r[target]) > 0 and math.isfinite(float(r[target]))]
    if len(vals) >= 3:
        lo = max(1e-6, float(np.quantile(vals, 0.10)) * 0.5)
        hi = max(lo * 1.1, float(np.quantile(vals, 0.90)) * 2.0)
        return np.logspace(math.log10(lo), math.log10(hi), n)
    return np.logspace(-1.5, 1.5, n)


def probe_param(module, params, target, n=15, base=1.0, *, anchor=None,
                x_values=None, difficulty="easy", law_version="v0",
                system="vanilla_equation"):
    xs = np.asarray(x_values if x_values is not None else np.logspace(-1.5, 1.5, n), dtype=float)
    ys = []
    if anchor is None:
        anchor = {p: float(base) for p in params}
    for x in xs:
        inp = {p: float(anchor.get(p, base)) for p in params}
        inp[target] = float(x)
        y = run_exp(module, inp, difficulty=difficulty, law_version=law_version, system=system)
        if y is not None:
            ys.append((float(x), y))
    if not ys:
        return np.array([]), np.array([])
    return np.array([r[0] for r in ys]), np.array([r[1] for r in ys])


def detect_constant(xs, ys, tol=0.02):
    """True if y barely varies: normalized std < tol."""
    mu = np.mean(np.abs(ys))
    if mu < 1e-15:
        return True, np.mean(ys)
    return np.std(ys) / mu < tol, np.mean(ys)


def detect_power(xs, ys, tol=0.005):
    valid = (xs > 0) & (ys > 0)
    if valid.sum() < 4:
        return None
    lx, ly = np.log(xs[valid]), np.log(ys[valid])
    slope, ic = np.polyfit(lx, ly, 1)
    resid = np.std(ly - (slope * lx + ic))
    if resid < tol:
        snapped = snap_exponent(float(slope))
        return {"form": "power", "exp": snapped, "raw_exp": slope, "resid": resid}
    return None


def detect_exp(xs, ys, tol=0.005):
    """Detect y = C * exp(b * x^alpha). Try multiple alphas."""
    valid = ys > 0
    if valid.sum() < 4:
        return None
    best = None
    for alpha in EXP_ALPHAS:
        z = xs[valid] ** alpha
        logy = np.log(ys[valid])
        slope, ic = np.polyfit(z, logy, 1)
        resid = np.std(logy - (slope * z + ic))
        if best is None or resid < best["resid"]:
            best = {"form": "exp", "rate": float(slope), "alpha": float(alpha),
                    "C": float(np.exp(ic)), "resid": float(resid)}
    if best and best["resid"] < tol:
        return best
    return None


def detect_trig(xs, ys, tol=0.01):
    """Try trig transforms of x, fit power law in transform(x)."""
    best = None
    for name, fn in TRIG_TRANSFORMS.items():
        try:
            tx = fn(xs)
            valid = np.isfinite(tx) & np.isfinite(ys) & (np.abs(tx) > 1e-9) & (np.abs(ys) > 1e-9)
            if valid.sum() < 4:
                continue
            # Direct linear fit (for forms like y = C * f(x))
            slope, ic = np.polyfit(tx[valid], ys[valid], 1)
            resid_lin = np.std(ys[valid] - (slope * tx[valid] + ic)) / (np.mean(np.abs(ys[valid])) + 1e-12)
            # Power fit (for y = C * f(x)^p)
            pos = valid & (tx > 0) & (ys > 0)
            if pos.sum() >= 4:
                slope_p, ic_p = np.polyfit(np.log(tx[pos]), np.log(ys[pos]), 1)
                resid_p = np.std(np.log(ys[pos]) - (slope_p * np.log(tx[pos]) + ic_p))
            else:
                slope_p, ic_p, resid_p = 0, 0, 999
            r = min(resid_lin, resid_p)
            if best is None or r < best["resid"]:
                if resid_lin < resid_p:
                    best = {"form": "trig", "fn": name, "mode": "linear",
                            "slope": float(slope), "ic": float(ic), "resid": float(r)}
                else:
                    best = {"form": "trig", "fn": name, "mode": "power",
                            "exp": float(round(slope_p * 4) / 4), "C": float(np.exp(ic_p)), "resid": float(r)}
        except Exception:
            continue
    if best and best["resid"] < tol:
        return best
    return None


def analyze_param(module, params, target, *, anchor=None, x_values=None,
                  difficulty="easy", law_version="v0", system="vanilla_equation",
                  enable_operator_charts=True):
    xs, ys = probe_param(module, params, target, anchor=anchor, x_values=x_values,
                         difficulty=difficulty,
                         law_version=law_version, system=system)
    def with_trace(form):
        if isinstance(form, dict):
            out = dict(form)
            out.setdefault("xs", xs.tolist())
            out.setdefault("ys", ys.tolist())
            return out
        return form

    if len(xs) < 4:
        return {"form": "unknown", "resid": 999}
    is_const, const_val = detect_constant(xs, ys)
    if is_const:
        return with_trace({"form": "constant", "value": const_val, "resid": 0.0})
    r_pow = detect_power(xs, ys)
    if r_pow:
        return with_trace(r_pow)
    r_exp = detect_exp(xs, ys)
    if r_exp:
        return with_trace(r_exp)
    r_trig = detect_trig(xs, ys)
    if r_trig:
        return with_trace(r_trig)
    if enable_operator_charts:
        ops = induce_univariate_operators(
            MeasurementTrace(
                name=f"{target}_probe",
                xs=tuple(float(x) for x in xs),
                ys=tuple(float(y) for y in ys),
                context={
                    "target": target,
                    "params": list(params),
                    "difficulty": difficulty,
                    "law_version": law_version,
                    "system": system,
                },
            )
        )
        if ops:
            return {
                "form": "operator_proposal",
                "resid": 999,
                "operator_genome": summarize_operator_genome(ops, limit=4),
                "xs": xs.tolist(),
                "ys": ys.tolist(),
            }
    return {"form": "unknown", "xs": xs.tolist(), "ys": ys.tolist(), "resid": 999}


def detect_joint_exp(module, params, exp_params, power_forms, *, difficulty="easy",
                     law_version="v0", system="vanilla_equation"):
    """Detect N = C * prod(x_i^p_i) * exp(-prod(x_j^a_j)) interaction.
    When multiple params show independent exp form, their exponents likely interact
    as a product inside the exp: exp(-lambda^a * t^b). Fit jointly via log(-log)."""
    import itertools
    grid = [0.3, 0.6, 1.0, 2.0, 4.0]
    data = []
    for combo in itertools.product(grid, repeat=len(exp_params)):
        inp = {p: 1.0 for p in params}
        for p, v in zip(exp_params, combo):
            inp[p] = float(v)
        y = run_exp(module, inp, difficulty=difficulty, law_version=law_version, system=system)
        if y is None or y <= 0 or not math.isfinite(y):
            continue
        # Divide out known power factors
        y_adj = y
        for p, pf in power_forms.items():
            if p not in exp_params and pf["form"] == "power":
                y_adj /= inp[p] ** pf["exp"]
        if y_adj <= 0:
            continue
        nll = -math.log(y_adj)
        if nll <= 0:
            continue
        data.append((inp, nll))
    if len(data) < 8:
        return None
    # Fit: log(nll) = sum(a_i * log(x_i)) + const  (each x_i in exp_params)
    X = np.array([[math.log(max(inp[p], 1e-9)) for p in exp_params] + [1.0]
                  for inp, _ in data])
    Y = np.array([math.log(nll) for _, nll in data])
    try:
        coefs, resid, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    except Exception:
        return None
    alphas = coefs[:-1]
    log_c = coefs[-1]
    # Residual quality
    Y_pred = X @ coefs
    r = np.std(Y - Y_pred)
    if r > 0.01:
        return None
    alphas_snapped = [snap_exponent(float(a)) for a in alphas]
    c_int = math.exp(log_c)
    return {"form": "joint_exp", "params": exp_params,
            "alphas": alphas_snapped, "raw_alphas": alphas.tolist(),
            "c_int": c_int, "resid": float(r)}


def _build_joint_exp_source(params, param_forms, entry, sig):
    """Build law source when joint exponential interaction is detected."""
    ji = param_forms.get("__joint__")
    if ji is None:
        return None
    exp_params = ji["params"]
    alphas = ji["alphas"]
    c_int = ji["c_int"]
    # Build: exp(-c_int * x0^a0 * x1^a1 * ...)
    parts = []
    for p, a in zip(exp_params, alphas):
        parts.append(f"({p} ** {a})" if a != 1.0 else p)
    inner = " * ".join(parts)
    exp_expr = f"math.exp(-{c_int} * {inner})"
    # Power law parts
    power_parts = []
    for p, pf in param_forms.items():
        if p in ("__joint__",) or pf["form"] in ("joint_exp_member", "joint_exp"):
            continue
        if pf["form"] == "power":
            power_parts.append(f"({p} ** {pf['exp']})")
        elif pf["form"] == "constant":
            pass
    power_expr = " * ".join(power_parts) if power_parts else "1.0"
    # Calibrate C: at all params=1, N = C * 1^... * exp(-c_int * 1*1) = C * exp(-c_int)
    # Calibrate by taking several measurements
    return f"import math\n\n{sig}\n    return {power_expr} * {exp_expr}\n"


def calibrate_const(module, params, param_forms, n=20, base=1.0, *, difficulty="easy",
                    law_version="v0", system="vanilla_equation"):
    """Calibrate overall multiplicative constant by comparing predictions to measurements."""
    ratios = []
    for _ in range(n):
        inp = {p: round(0.3 + np.random.random() * 4, 3) for p in params}
        y = run_exp(module, inp, difficulty=difficulty, law_version=law_version, system=system)
        if y is None or y == 0 or not math.isfinite(y):
            continue
        pred = 1.0
        ok = True
        for p, pf in param_forms.items():
            x = inp[p]
            f = pf["form"]
            if f == "constant":
                pass
            elif f == "power":
                pred *= x ** pf["exp"]
            elif f == "exp":
                pred *= math.exp(pf["rate"] * (x ** pf["alpha"]))
            elif f == "trig":
                fn = TRIG_TRANSFORMS[pf["fn"]]
                tx = float(fn(np.array([x]))[0])
                if pf["mode"] == "linear":
                    pred *= pf["slope"] * tx + pf["ic"]
                else:
                    pred *= pf["C"] * abs(tx) ** pf["exp"]
            else:
                ok = False; break
        if ok and pred != 0 and math.isfinite(pred) and pred > 0 and y > 0:
            ratios.append(math.log(y) - math.log(abs(pred)))
    if not ratios:
        return None
    return math.exp(np.median(ratios))


def build_law_source(params, param_forms, const, entry, sig):
    """Build Python source for the discovered law."""
    terms = []
    for p, pf in param_forms.items():
        if p == "__joint__" or pf["form"] in ("joint_exp_member",):
            continue
        f = pf["form"]
        if f == "constant":
            pass
        elif f == "power":
            e = pf["exp"]
            terms.append(f"({p} ** {e})")
        elif f == "exp":
            alpha = pf["alpha"]
            rate = pf["rate"]
            x_expr = f"{p}" if alpha == 1.0 else f"({p} ** {alpha})"
            terms.append(f"math.exp({rate} * {x_expr})")
        elif f == "trig":
            fn_name = pf["fn"]
            # Build the trig expression
            trig_map = {
                "sin": f"math.sin({p})",
                "cos": f"math.cos({p})",
                "sin2": f"(math.sin({p}) ** 2)",
                "cos2": f"(math.cos({p}) ** 2)",
                "sin2x": f"math.sin(2 * {p})",
                "cos2x": f"math.cos(2 * {p})",
                "1+sin2x": f"(1 + math.sin(2 * {p}))",
                "1+cos2x": f"(1 + math.cos(2 * {p}))",
                "1+sinx": f"(1 + math.sin({p}))",
                "1+cosx": f"(1 + math.cos({p}))",
            }
            tx_expr = trig_map.get(fn_name, f"math.sin({p})")
            if pf["mode"] == "linear":
                terms.append(f"({pf['slope']} * {tx_expr} + {pf['ic']})")
            else:
                terms.append(f"({pf['C']} * abs({tx_expr}) ** {pf['exp']})")
        else:
            return None
    body = " * ".join(terms) if terms else "1.0"
    src = f"import math\n\n{sig}\n    return {const} * {body}\n"
    return src


def _angle_like_param(name: str) -> bool:
    low = str(name).lower()
    return any(token in low for token in ("angle", "theta", "phase"))


def _finite_basis_ratios(rows, params, basis_fn):
    ratios = []
    for r in rows:
        try:
            b = float(basis_fn(r))
            y = float(r["_y"])
        except Exception:
            continue
        if math.isfinite(b) and math.isfinite(y) and abs(b) > 1e-12:
            ratios.append(y / b)
    return ratios


def _fit_const_to_target(rows, target_fn, basis_fn):
    ratios = []
    for r in rows:
        try:
            b = float(basis_fn(r))
            z = float(target_fn(r["_y"]))
        except Exception:
            continue
        if math.isfinite(b) and math.isfinite(z) and abs(b) > 1e-12:
            ratios.append(z / b)
    if len(ratios) < 4:
        return None
    return float(np.median(ratios))


def _fit_inverse_coordinate_loglinear(rows, params, angle, coord: CoordinateProgram):
    if coord.target_fn is None:
        return None
    X = []
    Y = []
    for r in rows:
        try:
            z = float(coord.target_fn(float(r["_y"])))
            c = abs(_coordinate_value(coord, float(r[angle])))
            row = [math.log(max(abs(float(r[p])), 1e-12)) for p in params if p != angle]
            row.append(math.log(max(c, 1e-12)))
        except Exception:
            continue
        if not math.isfinite(z) or z <= 1e-12 or not all(math.isfinite(v) for v in row):
            continue
        X.append(row + [1.0])
        Y.append(math.log(z))
    if len(X) < max(5, len(params) + 2):
        return None
    try:
        coefs, *_ = np.linalg.lstsq(np.asarray(X, dtype=float), np.asarray(Y, dtype=float), rcond=None)
    except Exception:
        return None
    pred = np.asarray(X, dtype=float) @ coefs
    resid = float(np.std(np.asarray(Y, dtype=float) - pred))
    if not math.isfinite(resid) or resid > 0.05:
        return None
    non_angle_params = [p for p in params if p != angle]
    param_exps = {p: snap_exponent(float(e), tolerance=0.08) for p, e in zip(non_angle_params, coefs[:-2])}
    coord_exp = snap_exponent(float(coefs[-2]), tolerance=0.08)
    const = float(math.exp(coefs[-1]))
    return const, param_exps, coord_exp, resid


def _fit_inverse_coordinate_loglinear_pair(
    rows,
    params,
    angle,
    target_coord: CoordinateProgram,
    input_coord: CoordinateProgram,
):
    if target_coord.target_fn is None:
        return None
    X = []
    Y = []
    for r in rows:
        try:
            z = float(target_coord.target_fn(float(r["_y"])))
            c = abs(_coordinate_value(input_coord, float(r[angle])))
            row = [math.log(max(abs(float(r[p])), 1e-12)) for p in params if p != angle]
            row.append(math.log(max(c, 1e-12)))
        except Exception:
            continue
        if not math.isfinite(z) or z <= 1e-12 or not all(math.isfinite(v) for v in row):
            continue
        X.append(row + [1.0])
        Y.append(math.log(z))
    if len(X) < max(5, len(params) + 2):
        return None
    try:
        coefs, *_ = np.linalg.lstsq(np.asarray(X, dtype=float), np.asarray(Y, dtype=float), rcond=None)
    except Exception:
        return None
    pred = np.asarray(X, dtype=float) @ coefs
    resid = float(np.std(np.asarray(Y, dtype=float) - pred))
    if not math.isfinite(resid) or resid > 0.05:
        return None
    non_angle_params = [p for p in params if p != angle]
    param_exps = {p: snap_exponent(float(e), tolerance=0.08) for p, e in zip(non_angle_params, coefs[:-2])}
    coord_exp = snap_exponent(float(coefs[-2]), tolerance=0.08)
    const = float(math.exp(coefs[-1]))
    return const, param_exps, coord_exp, resid


def _coordinate_expr(program: CoordinateProgram, p: str) -> str:
    return program.python_expr.format(x=p)


def _coordinate_value(program: CoordinateProgram, x: float) -> float:
    return float(program.fn(float(x)))


def _coordinate_candidates_for_param(param: str, param_forms) -> tuple[CoordinateProgram, ...]:
    if _angle_like_param(param):
        base = tuple(_phase_coordinates("deg"))
    else:
        base = ()
    if not isinstance(param_forms, dict):
        return base
    pf = param_forms.get(param)
    if not isinstance(pf, dict):
        return base
    xs = pf.get("xs") or []
    ys = pf.get("ys") or []
    if len(xs) < 4 or len(ys) < 4:
        return base
    induced = SelfInducedLanguage(max_depth=1, max_programs=48).induce_coordinate_language(
        NumericTrace(name=f"{param}_coordinate_trace", x=tuple(map(float, xs)), y=tuple(map(float, ys))),
        variable_name=param,
    )
    merged: list[CoordinateProgram] = []
    seen: set[str] = set()
    for coord in (*base, *induced):
        if coord.name in seen:
            continue
        seen.add(coord.name)
        merged.append(coord)
    return tuple(merged)


def _pow_expr(expr: str, exp: float) -> str:
    if exp == 1.0:
        return f"({expr})"
    return f"(({expr}) ** {exp})"


def fit_chart_candidates(module, params, tr, ho, sig, entry, *, difficulty="easy",
                         law_version="v0", system="vanilla_equation",
                         param_forms=None):
    """Universal coordinate recombination from self-induced proposals.

    This is deliberately not a Snell/Malus branch.  The runner receives
    coordinate programs born from residual traces and keeps only candidates that
    verify on held-out rows.
    """
    angle_params = [p for p in params if _angle_like_param(p)]
    if not angle_params:
        return None, 9.9, "no_chart"
    non_angles = [p for p in params if p not in angle_params]
    best = (None, 9.9, "chart_fail", 9)
    trig_powers = [1.0, 2.0, math.e, -1.0, -2.0]
    non_angle_exps = [0.0, 1.0, 2.0, 2.5, 3.0, -1.0, -2.0, 0.5]
    coords_by_angle = {
        angle: _coordinate_candidates_for_param(angle, param_forms)
        for angle in angle_params
    }
    coords_by_angle = {k: v for k, v in coords_by_angle.items() if v}
    if not coords_by_angle:
        return None, 9.9, "no_self_induced_coordinate"

    def try_candidate(src, method, priority=5):
        nonlocal best
        law = _law_from_source(src, entry)
        if law is None:
            return
        rm = _rmsle(law, ho, params)
        if rm < best[1] - 1e-4 or (abs(rm - best[1]) <= 1e-4 and priority < best[3]):
            best = (src, rm, method, priority)

    # Direct coordinate: y = C * product(non_angle^a) * coord(angle)^p
    for angle, coordinates in coords_by_angle.items():
        for coord in coordinates:
            for tp in trig_powers:
                for exps in _small_exp_product(non_angles, non_angle_exps, max_terms=2):
                    def basis_fn(r, angle=angle, coord=coord, tp=tp, exps=exps):
                        val = _coordinate_value(coord, float(r[angle]))
                        if abs(val) < 1e-12:
                            return float("nan")
                        if tp != int(tp) and val < 0:
                            return float("nan")
                        b = val ** tp
                        for p, e in exps.items():
                            if e != 0.0:
                                b *= float(r[p]) ** e
                        return b

                    ratios = _finite_basis_ratios(tr, params, basis_fn)
                    if len(ratios) < 4:
                        continue
                    const = float(np.median(ratios))
                    terms = [str(const)]
                    for p, e in exps.items():
                        if e != 0.0:
                            terms.append(f"({p} ** {e})")
                    terms.append(_pow_expr(_coordinate_expr(coord, angle), tp))
                    src = f"import math\n\n{sig}\n    return " + " * ".join(terms) + "\n"
                    try_candidate(src, f"self_induced_coordinate_direct:{coord.name}", priority=3)

    # Inverse coordinate: output = target_coord^{-1}(
    #   C * product(non_angle^a) * input_coord(angle)^p
    # ).
    # Target and input coordinates are intentionally allowed to differ.  This
    # closes laws such as cos(output)=C*sin(input)*ratio without naming Snell.
    for angle, coordinates in coords_by_angle.items():
        for target_coord in coordinates:
            if target_coord.inverse_template is None or target_coord.target_fn is None:
                continue
            for input_coord in coordinates:
                fitted = _fit_inverse_coordinate_loglinear_pair(tr, params, angle, target_coord, input_coord)
                if fitted is not None:
                    const, param_exps, coord_exp, resid = fitted
                    terms = [str(const)]
                    for p, e in param_exps.items():
                        if abs(e) > 1e-12:
                            terms.append(f"({p} ** {e})")
                    terms.append(_pow_expr(_coordinate_expr(input_coord, angle), coord_exp))
                    inner = " * ".join(terms)
                    rendered = target_coord.inverse_template.format(inner="INNER")
                    src = f"import math\n\n{sig}\n    INNER = {inner}\n    return {rendered}\n"
                    try_candidate(
                        src,
                        (
                            "self_induced_coordinate_inverse_refit:"
                            f"{target_coord.name}<-{input_coord.name}:resid={resid:.4g}"
                        ),
                        priority=0,
                    )
            fitted = _fit_inverse_coordinate_loglinear(tr, params, angle, target_coord)
            if fitted is not None:
                const, param_exps, coord_exp, resid = fitted
                terms = [str(const)]
                for p, e in param_exps.items():
                    if abs(e) > 1e-12:
                        terms.append(f"({p} ** {e})")
                terms.append(_pow_expr(_coordinate_expr(target_coord, angle), coord_exp))
                inner = " * ".join(terms)
                rendered = target_coord.inverse_template.format(inner="INNER")
                src = f"import math\n\n{sig}\n    INNER = {inner}\n    return {rendered}\n"
                try_candidate(
                    src,
                    f"self_induced_coordinate_inverse_refit:{target_coord.name}:resid={resid:.4g}",
                    priority=0,
                )
            for exps in _small_exp_product(non_angles, non_angle_exps, max_terms=2):
                def basis_fn(r, angle=angle, coord=target_coord, exps=exps):
                    b = _coordinate_value(coord, float(r[angle]))
                    for p, e in exps.items():
                        if e != 0.0:
                            b *= float(r[p]) ** e
                    return b

                const = _fit_const_to_target(tr, target_coord.target_fn, basis_fn)
                if const is None:
                    continue
                terms = [str(const)]
                for p, e in exps.items():
                    if e != 0.0:
                        terms.append(f"({p} ** {e})")
                terms.append(_coordinate_expr(target_coord, angle))
                inner = " * ".join(terms)
                rendered = target_coord.inverse_template.format(inner="INNER")
                src = f"import math\n\n{sig}\n    INNER = {inner}\n    return {rendered}\n"
                try_candidate(src, f"self_induced_coordinate_inverse:{target_coord.name}", priority=1)

    return best[:3]


def _small_exp_product(params, exps, max_terms=2):
    """Small generic exponent products without exploding the search."""
    if not params:
        yield {}
        return
    # one active variable or all variables with exponent 1 covers many causal
    # ratios/products while keeping this chart birth cheap.
    yield {p: 0.0 for p in params}
    for p in params:
        for e in exps:
            if e != 0.0:
                yield {q: (e if q == p else 0.0) for q in params}
    if len(params) <= max_terms:
        for e in [1.0, -1.0, 2.0, -2.0, 2.5]:
            yield {p: e for p in params}
        # Mixed products/ratios are the minimal inner language for many
        # inverse-coordinate laws: inv(C * x^a * y^b * chart(z)).
        # This is universal basis search, not a benchmark-specific rule.
        import itertools
        mixed_exps = [1.0, -1.0, 2.0, -2.0, 2.5, -2.5, 0.5, -0.5, 0.0]
        seen = set()
        for combo in itertools.product(mixed_exps, repeat=len(params)):
            if all(e == 0.0 for e in combo):
                continue
            if sum(e != 0.0 for e in combo) > max_terms:
                continue
            key = tuple(combo)
            if key in seen:
                continue
            seen.add(key)
            yield {p: e for p, e in zip(params, combo)}


def ask_model_law(client, model, params, probe_summary, sig, entry):
    """Fallback: ask model to propose law from probe summary. Returns law source or None."""
    prompt = (f"Signature: {sig}\n\n"
              f"Controlled experiment results (vary one param, hold others at 1.0):\n"
              f"{probe_summary}\n\n"
              f"Based on the data above, write the law function. Use only math module (no numpy). "
              f"Return JSON: {{\"source\": \"{entry}(...):\\n    import math\\n    return <expr>\"}} "
              f"with ACTUAL fitted numeric constants baked in.")
    raw = call_llm(client, model=model, system="Return only JSON.", user=prompt,
                   max_tokens=600, temperature=0.2)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        return json.loads(t).get("source")
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try:
            return json.loads(t[i:j+1]).get("source")
        except Exception:
            return None


def discover_law(client, model, module, params, sig, entry, rows, difficulty="easy",
                 law_version="v0", system="vanilla_equation",
                 enable_operator_charts=True):
    """Full active probing pipeline. Returns (law_src, held_out_rmsle, method)."""
    discover_law.last_operator_trace = []
    np.random.seed(42)
    tr = rows[:int(len(rows) * 0.6)]
    ho = rows[int(len(rows) * 0.6):]
    stress_ho = collect_stress_holdout(
        module,
        params,
        rows,
        difficulty=difficulty,
        law_version=law_version,
        system=system,
    )
    stress_mid = len(stress_ho) // 2
    coordinate_tr = tr + stress_ho[:stress_mid] if stress_ho else tr
    coordinate_ho = ho + stress_ho[stress_mid:] if stress_ho else ho

    # Phase 1: probe each parameter independently
    param_forms = {}
    probe_log = []
    anchor = _median_anchor(rows, params)
    for p in params:
        pf = analyze_param(module, params, p, anchor=anchor,
                           x_values=_intervention_values(rows, p),
                           difficulty=difficulty,
                           law_version=law_version, system=system,
                           enable_operator_charts=enable_operator_charts)
        param_forms[p] = pf
        probe_log.append(f"  {p}: {pf}")

    print(f"    probe results:")
    for line in probe_log:
        print(f"    {line}")
    discover_law.last_operator_trace = [
        {
            "param": p,
            "operators": pf.get("operator_genome", []),
        }
        for p, pf in param_forms.items()
        if isinstance(pf, dict) and pf.get("operator_genome")
    ]

    all_known = all(pf["form"] != "unknown" for pf in param_forms.values())

    # All-constant law → abstain (module returns constant, no scientific content)
    if all(pf["form"] == "constant" for pf in param_forms.values()):
        print("    → all params constant, abstain")
        return None, 9.9, "all_constant"

    # Detect multi-exp interaction (e.g. exp(-lambda * t^1.5))
    exp_params = [p for p, pf in param_forms.items() if pf["form"] == "exp"]
    if len(exp_params) >= 2:
        power_forms = {p: pf for p, pf in param_forms.items() if pf["form"] == "power"}
        ji = detect_joint_exp(module, params, exp_params, power_forms,
                              difficulty=difficulty, law_version=law_version, system=system)
        if ji:
            print(f"    → joint exp detected: {exp_params} alphas={ji['alphas']} resid={ji['resid']:.5f}")
            # Replace independent exp forms with joint form
            for p in exp_params:
                param_forms[p] = {"form": "joint_exp_member"}  # placeholder
            param_forms["__joint__"] = ji
            all_known = True

    if all_known and not any(pf["form"] == "unknown" for pf in param_forms.values()):
        # Phase 2: calibrate constant
        # For joint_exp, use specialized assembly
        if "__joint__" in param_forms:
            src = _build_joint_exp_source(params, param_forms, entry, sig)
            if src:
                law = _law_from_source(src, entry)
                if law:
                    rm = _rmsle(law, ho, params)
                    print(f"    → joint-exp law rmsle={rm:.4f}")
                    if rm < 0.1:
                        return src, rm, "joint_probe"
        const = calibrate_const(module, params, param_forms, difficulty=difficulty,
                                law_version=law_version, system=system)
        if const is None:
            print("    → constant calibration failed")
        else:
            # Phase 3: build law source
            src = build_law_source(params, param_forms, const, entry, sig)
            if src is None:
                print("    → law assembly failed")
            else:
                # Phase 4: verify
                law = _law_from_source(src, entry)
                if law is None:
                    print(f"    → law compile failed\n    src: {src[:200]}")
                else:
                    rm = _rmsle(law, ho, params)
                    print(f"    → assembled law rmsle={rm:.4f}")
                    if rm < 0.5:
                        if enable_operator_charts and any(_angle_like_param(p) for p in params) and rm > 1e-6:
                            csrc, crm, cmethod = fit_chart_candidates(
                                module, params, coordinate_tr, coordinate_ho, sig, entry,
                                difficulty=difficulty, law_version=law_version, system=system,
                                param_forms=param_forms)
                            if csrc is not None and crm <= rm:
                                print(f"    → operator chart improves symbolic candidate rmsle={crm:.4f} [{cmethod}]")
                                return csrc, crm, cmethod
                        return src, rm, "probe"
                    print("    → rmsle too high, falling back to operator/model")

    if enable_operator_charts:
        try:
            src, rm, method = fit_chart_candidates(
                module, params, coordinate_tr, coordinate_ho, sig, entry,
                difficulty=difficulty, law_version=law_version, system=system,
                param_forms=param_forms)
            if src is not None:
                print(f"    → operator chart candidate rmsle={rm:.4f} [{method}]")
                if rm < 0.1:
                    return src, rm, method
        except Exception as exc:
            print(f"    → operator chart fallback failed: {type(exc).__name__}: {exc}")

    # Phase 4b: multivariate recombination probe.  One-at-a-time probes fail
    # when a variable participates in a compact joint basis such as (x+y)^p or
    # x*y.  Fit a small universal basis library on ordinary observations and
    # verify on held-out rows; no module answers are used.
    try:
        points = [
            NBForcePoint(
                inputs={p: float(r[p]) for p in params},
                force=float(r["_y"]),
                raw_output=float(r["_y"]),
                analyzer="active_scalar",
            )
            for r in tr
            if "_y" in r and float(r["_y"]) > 0
        ]
        cands = fit_law_candidates(points, params)
        for cand in cands[:5]:
            src = cand.function_code(sig)
            law = _law_from_source(src, entry)
            if law is None:
                continue
            rm = _rmsle(law, ho, params)
            print(f"    → multivariate candidate {cand.kind} rmsle={rm:.4f}")
            if rm < 0.1:
                return src, rm, "multivariate_probe"
    except Exception as exc:
        print(f"    → multivariate fallback failed: {type(exc).__name__}: {exc}")

    # Phase 5 (fallback): give model the probe data, ask it to propose law
    probe_summary = "\n".join(probe_log)
    src = ask_model_law(client, model, params, probe_summary, sig, entry)
    if src is None:
        return None, 9.9, "model_fail"
    law = _law_from_source(src, entry)
    if law is None:
        return None, 9.9, "model_compile_fail"
    rm = _rmsle(law, ho, params)
    print(f"    → model-proposed law rmsle={rm:.4f}")
    return src, rm, "model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="nbap_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--difficulties", default="")
    ap.add_argument("--law_versions", default="v0")
    ap.add_argument("--system", default="vanilla_equation")
    ap.add_argument(
        "--disable_operator_charts",
        action="store_true",
        help="Ablation: skip residual-born/operator-chart recombination layer.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "nb_activeprobe" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    difficulties = [x.strip() for x in (args.difficulties or args.difficulty).split(",") if x.strip()]
    law_versions = [x.strip() for x in args.law_versions.split(",") if x.strip()]
    modules = [x.strip() for x in args.modules.split(",") if x.strip()]

    results = {}
    rows_out = []
    sa_list = []
    n_ans = n_abs = 0

    print(f"=== NewtonBench: Universal Active Probing — {args.model} ===\n")
    if args.disable_operator_charts:
        print("ablation=disable_operator_charts\n")
    for mod_name in modules:
        for difficulty in difficulties:
            for law_version in law_versions:
                module = importlib.import_module(f"modules.{mod_name}")
                sig = str(module.FUNCTION_SIGNATURE).strip()
                entry = sig[4:sig.index("(")].strip()
                params = _params(sig)
                rows = collect_active(module, params, difficulty=difficulty,
                                      law_version=law_version, system=args.system)
                key = f"{mod_name}/{difficulty}/{law_version}/{args.system}"
                if len(rows) < 10:
                    row = {"module": mod_name, "difficulty": difficulty,
                           "law_version": law_version, "system": args.system,
                           "status": "no_data", "SA": 0.0}
                    results[key] = row
                    rows_out.append(row)
                    n_abs += 1
                    continue

                print(f"  {key}:")
                src, rm, method = discover_law(
                    client, args.model, module, params, sig, entry, rows,
                    difficulty=difficulty, law_version=law_version, system=args.system,
                    enable_operator_charts=not args.disable_operator_charts)
                operator_trace = getattr(discover_law, "last_operator_trace", [])

                if src is None or rm > 0.5:
                    row = {"module": mod_name, "difficulty": difficulty,
                           "law_version": law_version, "system": args.system,
                           "status": "ABSTAIN", "SA": 0.0,
                           "best_rmsle": round(rm, 3), "method": method,
                           "operator_trace": operator_trace}
                    results[key] = row
                    rows_out.append(row)
                    n_abs += 1
                    print(f"      → ABSTAIN (rmsle={rm:.3f})")
                    continue

                try:
                    ev = module.evaluate_law(
                        src,
                        param_description=getattr(module, "PARAM_DESCRIPTION", ""),
                        difficulty=difficulty, law_version=law_version, judge_model_name="gpt41")
                    sa = float(ev.get("exact_accuracy", 0.0))
                except Exception as exc:
                    ev = {"eval_error": f"{type(exc).__name__}: {exc}"}
                    sa = 0.0

                sa_list.append(sa)
                n_ans += 1
                law_snip = src.split("return", 1)[-1].strip()[:100]
                row = {
                    "module": mod_name,
                    "difficulty": difficulty,
                    "law_version": law_version,
                    "system": args.system,
                    "status": "ANSWER",
                    "SA": sa,
                    "held_out_rmsle": round(rm, 4),
                    "method": method,
                    "operator_trace": operator_trace,
                    "law": law_snip,
                    "symbolic_equivalent": bool(ev.get("symbolic_equivalent", False)),
                    "rmsle": ev.get("rmsle"),
                }
                results[key] = row
                rows_out.append(row)
                print(f"      → SA={sa:.2f} rmsle={rm:.4f} [{method}] | {law_snip[:50]}")
                out_path.write_text(json.dumps({"results": results, "rows": rows_out}, indent=2, default=str))

    n = n_ans + n_abs
    mean_ans = sum(sa_list) / len(sa_list) if sa_list else 0.0
    mean_all = sum(sa_list) / n if n else 0.0
    print(f"\n=== RESULT ({args.difficulty}) ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  SA on answered = {mean_ans:.2f}  |  SA over all = {mean_all:.2f}")
    print(f"  self-debug baseline: SA_all=0.17 | non-thinking GPT-4.1 baseline: ~0.06")
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "SA_answered": round(mean_ans, 3), "SA_all": round(mean_all, 3),
                                    "modules": modules, "difficulties": difficulties,
                                    "law_versions": law_versions, "system": args.system,
                                    "disable_operator_charts": bool(args.disable_operator_charts),
                                    "n": n, "rows": rows_out, "results": results},
                                   indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
