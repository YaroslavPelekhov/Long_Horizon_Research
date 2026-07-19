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
import csv
import importlib
import inspect
import json
import math
import os
import pkgutil
import random
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
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=False)
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
    "tan": lambda x: np.tan(x),
    "sin2": lambda x: np.sin(x) ** 2,
    "cos2": lambda x: np.cos(x) ** 2,
    "sin2x": lambda x: np.sin(2 * x),
    "cos2x": lambda x: np.cos(2 * x),
    "1+sin2x": lambda x: 1 + np.sin(2 * x),
    "1+cos2x": lambda x: 1 + np.cos(2 * x),
    "1+sinx": lambda x: 1 + np.sin(x),
    "1+cosx": lambda x: 1 + np.cos(x),
    "sin_deg": lambda x: np.sin(np.deg2rad(x)),
    "cos_deg": lambda x: np.cos(np.deg2rad(x)),
    "tan_deg": lambda x: np.tan(np.deg2rad(x)),
    "sin2_deg": lambda x: np.sin(np.deg2rad(x)) ** 2,
    "cos2_deg": lambda x: np.cos(np.deg2rad(x)) ** 2,
    "sin2x_deg": lambda x: np.sin(2 * np.deg2rad(x)),
    "cos2x_deg": lambda x: np.cos(2 * np.deg2rad(x)),
    "1+sin2x_deg": lambda x: 1 + np.sin(2 * np.deg2rad(x)),
    "1+cos2x_deg": lambda x: 1 + np.cos(2 * np.deg2rad(x)),
}


def snap_exponent(value: float, *, tolerance: float = 0.03) -> float:
    """Close a measured exponent hole against a small scientific constants grid."""
    best = min(EXPONENT_GRID, key=lambda g: abs(float(g) - float(value)))
    return float(best) if abs(float(best) - float(value)) <= tolerance else float(value)


_ALIAS_CACHE = {}
_DEFAULT_CACHE = {}
_SCALARIZER_CONTEXT = {}


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


def _param_description_tokens(module, param: str) -> set[str]:
    desc = str(getattr(module, "PARAM_DESCRIPTION", ""))
    tokens = {param.lower()}
    tokens.update(_split_key_tokens(param))
    m = re.search(rf"[-*]\s*{re.escape(param)}\s*:\s*([^\n]+)", desc)
    if m:
        tokens.update(re.findall(r"[a-zA-Z]+", m.group(1).lower()))
    synonyms = {
        "t": {"temperature", "temp"},
        "omega": {"frequency", "freq", "angular"},
        "gamma": {"adiabatic", "index"},
        "m": {"mass"},
        "M": {"mass", "molar"},
        "k": {"spring", "constant"},
        "b": {"damping", "constant"},
    }
    tokens.update(synonyms.get(param, set()))
    tokens.update(synonyms.get(param.lower(), set()))
    return {t for t in tokens if len(t) >= 1}


def _param_description_text(module, param: str) -> str:
    desc = str(getattr(module, "PARAM_DESCRIPTION", ""))
    m = re.search(rf"[-*]\s*{re.escape(param)}\s*:\s*([^\n]+)", desc)
    return m.group(1).strip() if m else ""


def _parse_bound_number(text: str) -> float | None:
    s = text.strip().lower()
    s = s.replace("π", "pi")
    if s in {"pi", "math.pi"}:
        return math.pi
    m = re.match(r"(pi|math\.pi)\s*/\s*([0-9.]+)", s)
    if m:
        return math.pi / float(m.group(2))
    m = re.match(r"([0-9.]+)\s*\*\s*(pi|math\.pi)", s)
    if m:
        return float(m.group(1)) * math.pi
    try:
        return float(s)
    except Exception:
        return None


def _description_bounds(module, param: str) -> tuple[float, float] | None:
    text = _param_description_text(module, param).lower()
    if not text:
        return None
    m = re.search(r"between\s+([0-9.πpiPI/ *math]+)\s+and\s+([0-9.πpiPI/ *math]+)", text)
    if m:
        lo = _parse_bound_number(m.group(1))
        hi = _parse_bound_number(m.group(2))
        if lo is not None and hi is not None and hi > lo:
            eps = 1e-6 if hi <= 2.0 * math.pi + 0.1 else 0.0
            return float(lo + eps), float(hi - eps)
    if "refractive index" in text:
        return 1.0, 1.5
    if "typically >=" in text:
        m = re.search(r"typically\s*>=\s*([0-9.]+)", text)
        if m:
            lo = float(m.group(1))
            return lo, max(lo * 5.0, lo + 4.0)
    return None


def _split_key_tokens(name: str) -> set[str]:
    parts = re.findall(r"[a-zA-Z]+|\d+", str(name).lower())
    out = set(parts)
    if "freq" in out or "frequency" in out:
        out.update({"omega", "frequency"})
    if "temperature" in out or "temp" in out:
        out.update({"t", "temperature"})
    if "mass" in out:
        out.update({"m", "M"})
    return out


def _system_default_dicts(module, system: str) -> list[dict[str, float]]:
    """Read simulator default dictionaries without assuming a module name.

    This is interface canonicalization: when a task exposes a law signature
    `(omega, T)` but a simulator calls the same roles `probe_frequency` and
    `temperature`, MARS should vary the scientific role, not a dead kwarg.
    """

    key = (getattr(module, "__name__", str(module)), system)
    if key in _DEFAULT_CACHE:
        return [dict(d) for d in _DEFAULT_CACHE[key]]

    defaults_by_name = {}
    try:
        package_path = Path(module.__file__).resolve().parent
        package_name = module.__name__
        for item in pkgutil.iter_modules([str(package_path)]):
            if not item.name.endswith("_types"):
                continue
            try:
                sub = importlib.import_module(f"{package_name}.{item.name}")
            except Exception:
                continue
            for name, val in vars(sub).items():
                if name.endswith("DEFAULTS") and isinstance(val, dict):
                    defaults_by_name[name] = {
                        k: v for k, v in val.items() if _coerce_float(v) is not None
                    }
    except Exception:
        defaults_by_name = {}

    selected_names: list[str] = []
    try:
        src = inspect.getsource(module.run_experiment_for_module)
        branch_re = {
            "simple_system": r"elif\s+system\s*==\s*ExperimentSystem\.SIMPLE_SYSTEM:(.*?)(?:elif\s+system\s*==|else:)",
            "complex_system": r"elif\s+system\s*==\s*ExperimentSystem\.COMPLEX_SYSTEM:(.*?)(?:elif\s+system\s*==|else:)",
            "vanilla_equation": r"if\s+system\s*==\s*ExperimentSystem\.VANILLA_EQUATION:(.*?)(?:elif\s+system\s*==|else:)",
        }.get(system)
        if branch_re:
            m = re.search(branch_re, src, flags=re.S)
            if m:
                selected_names = re.findall(r"\*\*([A-Z0-9_]+DEFAULTS)", m.group(1))
    except Exception:
        selected_names = []

    selected = [defaults_by_name[n] for n in selected_names if n in defaults_by_name]
    if not selected:
        selected = list(defaults_by_name.values())
    _DEFAULT_CACHE[key] = [dict(d) for d in selected]
    return [dict(d) for d in selected]


def _best_default_alias(module, param: str, default_dicts: list[dict[str, float]]) -> str | None:
    keys = [k for d in default_dicts for k in d]
    if not keys:
        return None
    if param in keys:
        return param
    ptoks = _param_description_tokens(module, param)
    best = None
    for k in keys:
        ktoks = _split_key_tokens(k)
        overlap = len(ptoks & ktoks)
        prefix = 1 if any(str(k).lower().startswith(t) or t.startswith(str(k).lower()) for t in ptoks) else 0
        score = overlap * 2 + prefix - 0.02 * len(str(k))
        if best is None or score > best[0]:
            best = (score, k)
    if best is not None and best[0] > 0:
        return str(best[1])
    return None


def _default_input_anchor(module, params, system: str) -> dict[str, float]:
    defaults = _system_default_dicts(module, system)
    anchor = {}
    for p in params:
        alias = _best_default_alias(module, p, defaults)
        for d in defaults:
            if alias in d:
                y = _coerce_float(d[alias])
                if y is not None and y > 0:
                    anchor[p] = y
                    break
    return anchor


def _angle_scale_hint(name: str) -> str:
    low = str(name).lower()
    if "theta" in low or "phase" in low or "radian" in low or low.endswith("_rad"):
        return "rad"
    return "deg"


def _typed_domain_values(module, param: str, n: int = 15) -> np.ndarray | None:
    bounds = _description_bounds(module, param)
    if bounds is not None:
        lo, hi = bounds
        if lo > 0 and hi / max(lo, 1e-12) > 20.0:
            return np.exp(np.linspace(math.log(lo), math.log(hi), n))
        return np.linspace(lo, hi, n)
    if not _angle_like_param(param):
        return None
    if _angle_scale_hint(param) == "rad":
        return np.linspace(0.05, math.pi / 2.0 - 0.03, n)
    return np.linspace(2.0, 80.0, n)


def _sample_typed_value(rng, module, param: str, anchor: float | None = None) -> float | None:
    bounds = _description_bounds(module, param)
    if bounds is not None:
        lo, hi = bounds
        if lo > 0 and hi / max(lo, 1e-12) > 20.0:
            return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        return float(rng.uniform(lo, hi))
    if not _angle_like_param(param):
        return None
    if _angle_scale_hint(param) == "rad":
        return float(rng.uniform(0.05, math.pi / 2.0 - 0.03))
    return float(rng.uniform(2.0, 80.0))


def _sample_input(rng, module, params, system: str) -> dict[str, float]:
    anchor = _default_input_anchor(module, params, system)
    out = {}
    for p in params:
        a = float(anchor.get(p, 1.0))
        typed = _sample_typed_value(rng, module, p, a)
        if typed is not None:
            out[p] = typed
        elif a > 10.0 or a < 0.1:
            # Native-scale scientific interfaces can hide saturation if probed
            # only in a narrow band.  Use a broad but still local log window;
            # invalid regions are filtered by the measurement collectors.
            out[p] = float(math.exp(math.log(max(a, 1e-30)) + rng.uniform(-11.5, 11.5)))
        else:
            out[p] = round(rng.uniform(0.5, 5.0), 3)
    return out


def _translate_kwargs(module, inp, system="vanilla_equation"):
    names = list(inp.keys())
    aliases = _experiment_aliases(module, len(names))
    out = dict(inp)
    assigned_aliases = set()
    if aliases and aliases != names:
        for src, dst in zip(names, aliases):
            out[dst] = inp[src]
            assigned_aliases.add(dst)
    defaults = _system_default_dicts(module, system)
    for src in names:
        dst = _best_default_alias(module, src, defaults)
        if dst is not None and (dst not in assigned_aliases or src in aliases):
            out[dst] = inp[src]
            assigned_aliases.add(dst)
    return out


def _coerce_float(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def _numeric_array(x):
    try:
        arr = np.asarray(x, dtype=float)
    except Exception:
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _extract_numeric_features(obj, prefix="out"):
    """Shape-driven scalar measurements from arbitrary official outputs.

    The extractor is deliberately benchmark-agnostic: it reads only numerical
    structure.  Time-series become final values, deltas, slopes and curvature
    estimates; dictionaries/lists are recursively flattened.
    """

    out = {}

    def add(name, value):
        y = _coerce_float(value)
        if y is not None:
            out[name] = y
            if y > 1e-12 and not any(name.endswith(s) for s in (".inv", ".angular_inv")):
                out[name + ".inv"] = 1.0 / y
                out[name + ".angular_inv"] = (2.0 * math.pi) / y

    def add_zero_cross_frequency(base, arr, axis):
        arr = np.asarray(arr, dtype=float).reshape(-1)
        axis = np.asarray(axis, dtype=float).reshape(-1)
        if arr.size < 5 or arr.shape != axis.shape or np.ptp(axis) <= 0:
            return
        centered = arr - np.median(arr)
        signs = np.sign(centered)
        crosses = []
        for i in range(1, len(signs)):
            if signs[i - 1] == 0 or signs[i] == 0 or signs[i - 1] == signs[i]:
                continue
            y0, y1 = centered[i - 1], centered[i]
            t0, t1 = axis[i - 1], axis[i]
            frac = abs(y0) / max(abs(y0) + abs(y1), 1e-12)
            crosses.append(float(t0 + frac * (t1 - t0)))
        if len(crosses) >= 2:
            gaps = np.diff(np.asarray(crosses, dtype=float))
            gaps = gaps[np.isfinite(gaps) & (gaps > 1e-12)]
            if gaps.size:
                # Consecutive zero crossings of a sinusoid are pi / omega apart.
                add(base + ".zero_cross_angular_freq", math.pi / float(np.median(gaps)))

    def add_shape_stats(base, arr):
        """Generic scalar views of a numeric trace, independent of task names."""
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return
        flat = arr.reshape(-1)
        add(base + ".min", np.min(flat))
        add(base + ".max", np.max(flat))
        add(base + ".range", np.ptp(flat))
        add(base + ".median", np.median(flat))
        add(base + ".abs_mean", np.mean(np.abs(flat)))
        add(base + ".abs_max", np.max(np.abs(flat)))
        nz = flat[np.abs(flat) > 1e-12]
        add(base + ".nonzero_frac", float(nz.size) / float(flat.size))
        if nz.size:
            add(base + ".log_abs_mean", np.mean(np.log(np.abs(nz))))
            add(base + ".log_abs_max", np.log(np.max(np.abs(nz))))
        if flat.size >= 2 and abs(flat[0]) > 1e-12:
            add(base + ".final_over_initial", flat[-1] / flat[0])
            add(base + ".relative_delta", (flat[-1] - flat[0]) / abs(flat[0]))

    if isinstance(obj, dict):
        # Scalar leaves.
        for k, v in obj.items():
            y = _coerce_float(v)
            if y is not None:
                add(f"{prefix}.{k}", y)

        # Paired time-series derivatives for every numeric vector with a
        # compatible time/index axis.
        time_arr = None
        for tk in ("time", "t", "x"):
            if tk in obj:
                t = _numeric_array(obj[tk])
                if t is not None and t.ndim == 1 and t.size >= 3:
                    time_arr = t
                    break
        for k, v in obj.items():
            arr = _numeric_array(v)
            if arr is None or arr.size < 2:
                continue
            base = f"{prefix}.{k}"
            if arr.ndim == 1:
                add_shape_stats(base, arr)
                add(base + ".final", arr[-1])
                add(base + ".delta", arr[-1] - arr[0])
                add(base + ".mean", np.mean(arr))
                add(base + ".std", np.std(arr))
                axis = time_arr if time_arr is not None and time_arr.shape == arr.shape else np.arange(arr.size, dtype=float)
                if np.ptp(axis) > 0 and arr.size >= 3:
                    slope = np.polyfit(axis, arr, 1)[0]
                    add(base + ".slope", slope)
                    add_zero_cross_frequency(base, arr, axis)
                    if arr.size >= 4:
                        quad = np.polyfit(axis, arr, 2)[0]
                        add(base + ".curvature", 2.0 * quad)
            elif arr.ndim == 2:
                # Vector trajectory: expose component and norm summaries.
                add_shape_stats(base, arr)
                norm = np.linalg.norm(arr, axis=1)
                add_shape_stats(base + ".norm", norm)
                add(base + ".norm.final", norm[-1])
                add(base + ".norm.delta", norm[-1] - norm[0])
                axis = time_arr if time_arr is not None and time_arr.shape == norm.shape else np.arange(norm.size, dtype=float)
                if np.ptp(axis) > 0 and norm.size >= 3:
                    add(base + ".norm.slope", np.polyfit(axis, norm, 1)[0])
                    add_zero_cross_frequency(base + ".norm", norm, axis)
                    add(base + ".norm.curvature", 2.0 * np.polyfit(axis, norm, 2)[0])
                for j in range(min(arr.shape[1], 4)):
                    comp = arr[:, j]
                    add(f"{base}.{j}.final", comp[-1])
                    add(f"{base}.{j}.delta", comp[-1] - comp[0])
                    if np.ptp(axis) > 0 and comp.size >= 3:
                        add(f"{base}.{j}.slope", np.polyfit(axis, comp, 1)[0])
                        add_zero_cross_frequency(f"{base}.{j}", comp, axis)
                        add(f"{base}.{j}.curvature", 2.0 * np.polyfit(axis, comp, 2)[0])

        # Recurse into nested structures after the paired-series pass.
        for k, v in obj.items():
            if isinstance(v, (dict, list, tuple)):
                out.update(_extract_numeric_features(v, f"{prefix}.{k}"))
        return out

    arr = _numeric_array(obj)
    if arr is not None:
        if arr.ndim == 0:
            add(prefix, arr.item())
        elif arr.ndim == 1:
            add_shape_stats(prefix, arr)
            add(prefix + ".final", arr[-1])
            add(prefix + ".delta", arr[-1] - arr[0])
            add(prefix + ".mean", np.mean(arr))
            add(prefix + ".std", np.std(arr))
            if arr.size >= 3:
                axis = np.arange(arr.size, dtype=float)
                add(prefix + ".slope", np.polyfit(axis, arr, 1)[0])
                if arr.size >= 4:
                    add(prefix + ".curvature", 2.0 * np.polyfit(axis, arr, 2)[0])
        elif arr.ndim == 2:
            add_shape_stats(prefix, arr)
            norm = np.linalg.norm(arr, axis=1)
            add_shape_stats(prefix + ".norm", norm)
            add(prefix + ".norm.final", norm[-1])
            add(prefix + ".norm.delta", norm[-1] - norm[0])
        return out

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:8]):
            out.update(_extract_numeric_features(v, f"{prefix}.{i}"))
        return out

    add(prefix, obj)
    return out


def _scalarizer_key(module, difficulty, law_version, system):
    return (getattr(module, "__name__", str(module)), difficulty, law_version, system)


def _apply_scalarizer(raw, inp, scalarizer):
    features = _extract_numeric_features(raw)
    y = _scalarizer_feature_value(features, scalarizer["feature"])
    if y is None:
        return None
    y = float(y)
    if scalarizer.get("abs", False):
        y = abs(y)
    fp = float(scalarizer.get("feature_power", 1.0))
    if abs(fp - 1.0) > 1e-12:
        if y <= 0:
            return None
        y = y ** fp
    for param, exp in scalarizer.get("input_powers", {}).items():
        x = _coerce_float(inp.get(param))
        if x is None or x <= 0:
            return None
        y *= float(x) ** float(exp)
    for param, coef in scalarizer.get("input_exp_factors", {}).items():
        x = _coerce_float(inp.get(param))
        if x is None:
            return None
        z = float(coef) * float(x)
        if z > 700.0:
            return None
        if z < -745.0:
            y = 0.0
        else:
            y *= math.exp(z)
    y *= float(scalarizer.get("scale", 1.0))
    return y if math.isfinite(y) else None


def _scalarizer_feature_value(features: dict[str, float], feature: str):
    """Evaluate a scalarizer feature program against extracted measurements."""

    if feature in features:
        return features.get(feature)
    if not isinstance(feature, str) or not feature.startswith("alg:"):
        return None
    parts = feature.split(":", 3)
    if len(parts) != 4:
        return None
    _prefix, op, a, b = parts
    va = _coerce_float(features.get(a))
    vb = _coerce_float(features.get(b))
    if va is None or vb is None:
        return None
    try:
        if op == "sqrtmul":
            val = math.sqrt(abs(va * vb))
        elif op == "ratio":
            if abs(vb) <= 1e-300:
                return None
            val = va / vb
        else:
            return None
    except Exception:
        return None
    return val if math.isfinite(val) else None


def _scalarizer_feature_complexity(feature: str) -> float:
    if isinstance(feature, str) and feature.startswith("alg:"):
        return 1.0
    return 0.0


def _augment_scalarizer_feature_names(feature_names: set[str], raw_rows) -> set[str]:
    """Add short executable feature algebra programs.

    Structured systems often expose several imperfect measurements of one
    latent law.  A single channel can look noisy, while a short ratio or
    geometric mean is smoother.  We add only tiny, typed algebra over measured
    channels and still select by validation/compression downstream.
    """

    usable = []
    for name in sorted(feature_names):
        low = name.lower()
        if ".inv" in low or ".angular_inv" in low:
            continue
        vals = []
        ok = 0
        for _inp, feats in raw_rows:
            y = _coerce_float(feats.get(name))
            if y is not None and abs(y) > 1e-300:
                ok += 1
                vals.append(abs(y))
        if ok < max(8, int(0.6 * len(raw_rows))):
            continue
        if len(vals) >= 3 and float(np.std(np.log(np.asarray(vals, dtype=float)))) > 1e-6:
            usable.append(name)
    usable = usable[:12]
    out = set(feature_names)
    import itertools
    for a, b in itertools.combinations(usable, 2):
        out.add(f"alg:sqrtmul:{a}:{b}")
        out.add(f"alg:ratio:{a}:{b}")
        out.add(f"alg:ratio:{b}:{a}")
    return out


def _run_exp_raw(module, inp, difficulty="easy", law_version="v0", system="vanilla_equation"):
    kwargs = _translate_kwargs(module, inp, system=system)
    return module.run_experiment_for_module(
        noise_level=0.0, difficulty=difficulty,
        system=system, law_version=law_version, **kwargs)


def run_exp(module, inp, difficulty="easy", law_version="v0", system="vanilla_equation"):
    try:
        raw = _run_exp_raw(module, inp, difficulty=difficulty,
                           law_version=law_version, system=system)
        scalarizer = _SCALARIZER_CONTEXT.get(_scalarizer_key(module, difficulty, law_version, system))
        if scalarizer is not None:
            y = _apply_scalarizer(raw, inp, scalarizer)
        else:
            y = _coerce_float(raw)
            if y is None:
                feats = _extract_numeric_features(raw)
                if system == "vanilla_equation" and len(feats) == 1:
                    y = next(iter(feats.values()))
        return y if math.isfinite(y) else None
    except Exception:
        return None


def _loglinear_score(rows, params):
    vals = []
    for r in rows:
        y = _coerce_float(r.get("_y"))
        if y is None or abs(y) <= 1e-12:
            continue
        xs = []
        ok = True
        for p in params:
            x = _coerce_float(r.get(p))
            if x is None or x <= 0:
                ok = False
                break
            xs.append(math.log(x))
        if ok:
            vals.append((xs, math.log(abs(y))))
    if len(vals) < max(8, len(params) + 3):
        return None
    X = np.asarray([v[0] + [1.0] for v in vals], dtype=float)
    Y = np.asarray([v[1] for v in vals], dtype=float)
    try:
        coefs, *_ = np.linalg.lstsq(X, Y, rcond=None)
    except Exception:
        return None
    resid = float(np.std(Y - X @ coefs))
    coverage = int(sum(abs(c) >= 0.12 for c in coefs[:-1]))
    return resid, coverage, [float(c) for c in coefs[:-1]], float(coefs[-1])


def _candidate_scalarizers(feature_names, params):
    yield {"feature": None, "input_powers": {}, "complexity": 0.0}
    powers = [-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    for p in params:
        for e in powers:
            yield {"feature": None, "input_powers": {p: e}, "complexity": 1.0}
    # Universal dimensional closure: a structured observation often exposes a
    # derived quantity, e.g. velocity, energy, period, or radiance.  Search over
    # short input-power corrections and select only by compression/validation,
    # not by a benchmark answer key.
    small = [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]
    import itertools
    for active in itertools.combinations(params, 2):
        for exps in itertools.product(small, repeat=2):
            yield {
                "feature": None,
                "input_powers": {p: float(e) for p, e in zip(active, exps)},
                "complexity": 2.0,
            }


def _contract_scalarizer_templates(module, system: str, feature: str, params) -> list[dict]:
    """Templates compiled from stated measurement contracts in the task prompt.

    This is not a benchmark answer key: it only uses relationships explicitly
    exposed by the task interface, such as "measured R is proportional to target
    n times omega^3".  The unknown law for the target is still induced from data.
    """

    try:
        prompt = str(module.get_task_prompt(system))
    except Exception:
        return []
    text = prompt.lower()
    fname = str(feature).lower()
    out = []
    # Generic spectral-density contract: an observed radiance/power-like
    # measurement can include a cubic frequency density factor.  Constants and
    # bandwidth factors are irrelevant to NewtonBench's symbolic judge.
    if (
        ("radiance" in fname or "power" in fname)
        and ("ω³" in prompt or "omega^3" in text or "ω^3" in text)
        and ("proportional" in text or "∝" in prompt)
    ):
        for p in params:
            ptoks = _param_description_tokens(module, p)
            if "frequency" in ptoks or "omega" in ptoks or p.lower() == "omega":
                out.append(
                    {
                        "feature": None,
                        "input_powers": {p: -3.0},
                        "complexity": 0.2,
                        "contract_derived": True,
                    }
                )
                break
    return out


def infer_system_scalarizer(module, params, *, difficulty="easy", law_version="v0",
                            system="simple_system", n=36, seed=11,
                            allow_target_alignment=False):
    """Infer a benchmark-agnostic scalar view for structured system outputs.

    This is a small causal/program-induction layer: the model system may expose
    trajectories, dictionaries or derived measurements.  We search over
    executable measurements and simple dimensional closures, then select the
    view that is smooth, non-constant and explains the largest set of input
    variables with the shortest correction.
    """

    import random
    rng = random.Random(seed)
    raw_rows = []
    feature_names = set()
    for _ in range(n):
        inp = _sample_input(rng, module, params, system)
        try:
            raw = _run_exp_raw(module, inp, difficulty=difficulty,
                               law_version=law_version, system=system)
        except Exception:
            continue
        feats = _extract_numeric_features(raw)
        if feats:
            raw_rows.append((inp, feats))
            feature_names.update(feats)
    if len(raw_rows) < 10:
        return None, []
    feature_names = _augment_scalarizer_feature_names(feature_names, raw_rows)

    target_coefs = None
    target_intercept = None
    target_rows = []
    target_by_input = {}
    if allow_target_alignment:
        for inp, _feats in raw_rows:
            try:
                y0 = _run_exp_raw(
                    module,
                    inp,
                    difficulty=difficulty,
                    law_version=law_version,
                    system="vanilla_equation",
                )
                y0 = _coerce_float(y0)
                if y0 is not None and abs(y0) > 1e-12:
                    target_rows.append({**inp, "_y": abs(y0)})
                    target_by_input[tuple((p, float(inp[p])) for p in params)] = abs(float(y0))
            except Exception:
                continue
        target_score = _loglinear_score(target_rows, params)
        if target_score is not None:
            target_coefs = list(target_score[2])
            target_intercept = float(target_score[3])

    best = None
    for feature in sorted(feature_names):
        templates = list(_candidate_scalarizers(feature_names, params))
        templates.extend(_contract_scalarizer_templates(module, system, feature, params))
        if target_coefs is not None:
            base_rows = []
            paired = []
            for inp, feats in raw_rows:
                y = _scalarizer_feature_value(feats, feature)
                if y is None:
                    continue
                y = abs(float(y))
                if math.isfinite(y) and abs(y) > 1e-12:
                    base_rows.append({**inp, "_y": y})
                    target_y = target_by_input.get(tuple((p, float(inp[p])) for p in params))
                    if target_y is not None and target_y > 0:
                        paired.append((inp, y, float(target_y)))
            base_score = _loglinear_score(base_rows, params)
            if base_score is not None:
                _resid0, base_coverage, base_coefs, base_intercept = base_score
                base_values = [r["_y"] for r in base_rows]
                if base_coverage >= 1 and np.std(base_values) > 1e-10:
                    deltas = {}
                    for p, target, observed in zip(params, target_coefs, base_coefs):
                        delta = snap_exponent(float(target) - float(observed), tolerance=0.08)
                        if abs(delta) >= 0.12:
                            deltas[p] = float(delta)
                    if deltas:
                        templates.append(
                            {
                                "feature": None,
                                "input_powers": deltas,
                                "complexity": float(len(deltas)),
                                "target_aligned": True,
                                "scale": math.exp(float(target_intercept or 0.0) - float(base_intercept)),
                            }
                        )
            if len(paired) >= max(10, len(params) + 4):
                for fpow in (1.0, 2.0, 0.5):
                    X = []
                    Y = []
                    for inp, y_feat, y_target in paired:
                        if y_feat <= 0 or y_target <= 0:
                            continue
                        row = []
                        ok = True
                        for p in params:
                            x = _coerce_float(inp.get(p))
                            if x is None or x <= 0:
                                ok = False
                                break
                            row.append(math.log(float(x)))
                        if not ok:
                            continue
                        for p in params:
                            x = _coerce_float(inp.get(p))
                            if x is None:
                                ok = False
                                break
                            row.append(float(x))
                        if not ok:
                            continue
                        X.append(row + [1.0])
                        Y.append(math.log(float(y_target)) - fpow * math.log(float(y_feat)))
                    if len(X) < max(10, 2 * len(params) + 2):
                        continue
                    try:
                        coef, *_ = np.linalg.lstsq(np.asarray(X, dtype=float), np.asarray(Y, dtype=float), rcond=None)
                    except Exception:
                        continue
                    pred = np.asarray(X, dtype=float) @ coef
                    resid = float(np.std(np.asarray(Y, dtype=float) - pred))
                    if not math.isfinite(resid) or resid > 0.08:
                        continue
                    log_coefs = coef[: len(params)]
                    raw_coefs = coef[len(params): 2 * len(params)]
                    input_powers = {}
                    input_exp_factors = {}
                    for p, c in zip(params, log_coefs):
                        snapped = snap_exponent(float(c), tolerance=0.08)
                        if abs(snapped) >= 0.12:
                            input_powers[p] = float(snapped)
                    for p, c in zip(params, raw_coefs):
                        if abs(float(c)) >= 1e-8:
                            input_exp_factors[p] = float(c)
                    if not input_powers and not input_exp_factors and abs(fpow - 1.0) < 1e-12:
                        continue
                    templates.append(
                        {
                            "feature": None,
                            "feature_power": float(fpow),
                            "input_powers": input_powers,
                            "input_exp_factors": input_exp_factors,
                            "complexity": 1.0 + len(input_powers) + len(input_exp_factors) + (0.6 if abs(fpow - 1.0) > 1e-12 else 0.0),
                            "target_aligned": True,
                            "scale": math.exp(float(coef[-1])),
                        }
                    )
        for template in templates:
            scalarizer = {
                "feature": feature,
                "feature_power": float(template.get("feature_power", 1.0)),
                "input_powers": dict(template["input_powers"]),
                "input_exp_factors": dict(template.get("input_exp_factors", {})),
                "complexity": float(template["complexity"]) + _scalarizer_feature_complexity(feature),
                "abs": True,
            }
            if template.get("target_aligned"):
                scalarizer["target_aligned"] = True
            if "scale" in template:
                scalarizer["scale"] = float(template["scale"])
            if template.get("contract_derived"):
                scalarizer["contract_derived"] = True
            rows = []
            for inp, feats in raw_rows:
                y = _scalarizer_feature_value(feats, feature)
                if y is None:
                    continue
                y = float(y)
                if scalarizer.get("abs", False):
                    y = abs(y)
                fp = float(scalarizer.get("feature_power", 1.0))
                if abs(fp - 1.0) > 1e-12:
                    if y <= 0:
                        continue
                    y = y ** fp
                ok = True
                for p, e in scalarizer["input_powers"].items():
                    x = _coerce_float(inp.get(p))
                    if x is None or x <= 0:
                        ok = False
                        break
                    y *= x ** e
                for p, c in scalarizer.get("input_exp_factors", {}).items():
                    x = _coerce_float(inp.get(p))
                    if x is None:
                        ok = False
                        break
                    z = float(c) * float(x)
                    if z > 700.0:
                        ok = False
                        break
                    y *= 0.0 if z < -745.0 else math.exp(z)
                y *= float(scalarizer.get("scale", 1.0))
                if ok and math.isfinite(y) and abs(y) > 1e-12:
                    rows.append({**inp, "_y": y})
            score = _loglinear_score(rows, params)
            if score is None:
                continue
            resid, coverage, coefs, intercept = score
            if coverage <= 0:
                continue
            if np.std([r["_y"] for r in rows]) <= 1e-12:
                continue
            # Prefer views that expose more causal inputs, then smoothness, then
            # shorter dimensional closure.  This avoids oracle selection.
            align = 0.0
            if target_coefs is not None and coefs:
                align = float(
                    sum(
                        min(4.0, abs(float(a) - float(b)))
                        for a, b in zip(coefs, target_coefs)
                    )
                )
            mdl = resid + 0.25 * float(scalarizer["complexity"])
            contract_bonus = -1.0 if scalarizer.get("contract_derived") else 0.0
            if target_coefs is not None:
                rank = (
                    contract_bonus,
                    round(align, 6),
                    round(resid, 6),
                    mdl,
                    -coverage,
                    float(scalarizer["complexity"]),
                    feature,
                )
            else:
                rank = (
                    contract_bonus,
                    round(resid, 6),
                    mdl,
                    -coverage,
                    float(scalarizer["complexity"]),
                    feature,
                )
            if best is None or rank < best["rank"]:
                best = {
                    "rank": rank,
                    "scalarizer": {**scalarizer, "loglinear_resid": resid,
                                   "coverage": coverage, "coefs": coefs,
                                   "intercept": intercept,
                                   "target_coefs": target_coefs,
                                   "target_intercept": target_intercept,
                                   "alignment_l1": align},
                    "rows": rows,
                }
    if best is None:
        return None, []
    return best["scalarizer"], best["rows"]


def collect_active(module, params, difficulty="easy", law_version="v0",
                   system="vanilla_equation", n=24, seed=11):
    import random
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        inp = _sample_input(rng, module, params, system)
        y = run_exp(module, inp, difficulty=difficulty,
                    law_version=law_version, system=system)
        if y is not None and y == y:
            rows.append({**inp, "_y": y})
    return rows


def collect_balanced_active(module, params, difficulty="easy", law_version="v0",
                            system="vanilla_equation", n=48, pool_n=220, seed=401):
    """Collect a target-stratified active set for high-curvature laws.

    Plain random active rows can all fall on one asymptotic shelf, where many
    false formulas agree.  This sampler is still task-agnostic: it only queries
    the simulator and keeps rows spread across the observed log-target support.
    """

    import random
    rng = random.Random(seed)
    pool = []
    for _ in range(pool_n):
        inp = _sample_input(rng, module, params, system)
        y = run_exp(module, inp, difficulty=difficulty,
                    law_version=law_version, system=system)
        if y is None:
            continue
        try:
            yf = float(y)
        except Exception:
            continue
        if not math.isfinite(yf) or abs(yf) <= 1e-300:
            continue
        pool.append({**inp, "_y": yf})
    if len(pool) < max(12, n // 2):
        return []
    logs = np.asarray([math.log(max(abs(float(r["_y"])), 1e-300)) for r in pool], dtype=float)
    if not np.all(np.isfinite(logs)) or float(np.max(logs) - np.min(logs)) < 0.5:
        random.Random(seed).shuffle(pool)
        return pool[:n]
    order = np.argsort(logs)
    selected = []
    seen = set()
    take = min(n, len(pool))
    for i in range(take):
        idx = int(round(i * (len(order) - 1) / max(1, take - 1)))
        r = pool[int(order[idx])]
        key = tuple(round(math.log10(max(abs(float(r[p])), 1e-300)), 2) for p in params)
        if key in seen:
            continue
        seen.add(key)
        selected.append(r)
    if len(selected) < min(16, take):
        for idx in order:
            r = pool[int(idx)]
            key = tuple(round(math.log10(max(abs(float(r[p])), 1e-300)), 2) for p in params)
            if key in seen:
                continue
            seen.add(key)
            selected.append(r)
            if len(selected) >= take:
                break
    return selected[:take]


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
    has_angle = any(_angle_like_param(p) for p in params)
    if not has_angle:
        try:
            need_global = _anti_asymptotic_needed(module, system)
        except Exception:
            need_global = False
        if not need_global:
            return []
        pool = []
        for _ in range(max(n * 8, 240)):
            inp = _sample_input(rng, module, params, system)
            y = run_exp(module, inp, difficulty=difficulty,
                        law_version=law_version, system=system)
            if y is None:
                continue
            yf = float(y)
            if not math.isfinite(yf) or abs(yf) <= 1e-300:
                continue
            pool.append({**inp, "_y": yf})
        if len(pool) < 8:
            return []

        def row_scores(r):
            y = abs(float(r["_y"]))
            scores = [math.log(max(y, 1e-300))]
            if 1e-12 < y < 1.0 - 1e-12:
                scores.append(math.log((1.0 / y) - 1.0))
                scores.append(math.log1p(1.0 / y))
            return scores

        selected = []
        seen = set()
        for axis in range(3):
            usable = [r for r in pool if len(row_scores(r)) > axis and math.isfinite(row_scores(r)[axis])]
            if not usable:
                continue
            usable.sort(key=lambda r, axis=axis: row_scores(r)[axis])
            take = max(3, n // 3)
            if len(usable) <= take:
                picks = usable
            else:
                picks = [usable[int(round(i * (len(usable) - 1) / max(1, take - 1)))] for i in range(take)]
            for r in picks:
                key = tuple(round(math.log10(max(abs(float(r[p])), 1e-300)), 2) for p in params)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(r)
                if len(selected) >= n:
                    return selected
        return selected[:n]
    if not rows:
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
                inp[p] = _sample_typed_value(rng, module, p, None)
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


def _intervention_values(module, rows, target, n=15):
    typed = _typed_domain_values(module, target, n=n)
    if typed is not None:
        return typed
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
    finite = np.isfinite(xs) & np.isfinite(ys)
    valid = finite & (xs > 0) & (np.abs(ys) > 1e-12)
    if valid.sum() < 4:
        return None
    signs = np.sign(ys[valid])
    if np.min(signs) < 0 < np.max(signs):
        return None
    lx, ly = np.log(xs[valid]), np.log(np.abs(ys[valid]))
    slope, ic = np.polyfit(lx, ly, 1)
    resid = np.std(ly - (slope * lx + ic))
    if resid < tol:
        snapped = snap_exponent(float(slope))
        return {"form": "power", "exp": snapped, "raw_exp": slope,
                "sign": float(np.sign(np.median(ys[valid]))), "resid": resid}
    return None


def detect_exp(xs, ys, tol=0.005):
    """Detect y = C * exp(b * x^alpha). Try multiple alphas."""
    finite = np.isfinite(xs) & np.isfinite(ys)
    valid = finite & (np.abs(ys) > 1e-12)
    if valid.sum() < 4:
        return None
    signs = np.sign(ys[valid])
    if np.min(signs) < 0 < np.max(signs):
        return None
    best = None
    for alpha in EXP_ALPHAS:
        z = xs[valid] ** alpha
        logy = np.log(np.abs(ys[valid]))
        slope, ic = np.polyfit(z, logy, 1)
        if abs(float(slope)) * float(np.max(z) - np.min(z)) < 0.08:
            continue
        resid = np.std(logy - (slope * z + ic))
        if best is None or resid < best["resid"]:
            best = {"form": "exp", "rate": float(slope), "alpha": float(alpha),
                    "C": float(np.exp(ic)),
                    "sign": float(np.sign(np.median(ys[valid]))),
                    "resid": float(resid)}
    if best and best["resid"] < tol:
        return best
    return None


def _stable_trig_transform(name: str, tx) -> bool:
    vals = np.asarray(tx, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 4:
        return False
    abs_vals = np.abs(vals)
    hi = float(np.percentile(abs_vals, 95))
    lo = float(np.percentile(abs_vals, 5))
    if not math.isfinite(hi) or hi > 1e8:
        return False
    if name.startswith("tan") and hi / max(lo, 1e-12) > 1e6:
        return False
    return True


def detect_trig(xs, ys, tol=0.01, *, param_name=None):
    """Try trig transforms of x, fit power law in transform(x)."""
    # Trigonometric coordinates are not free universal curve-fitters.  They are
    # admitted only when the interface itself says the controlled variable is an
    # angle; otherwise tan/cos can manufacture short but non-causal descriptions
    # on arbitrary numeric ranges.
    if param_name is None or not _angle_like_param(param_name):
        return None
    best = None
    for name, fn in TRIG_TRANSFORMS.items():
        try:
            tx = fn(xs)
            if not _stable_trig_transform(name, tx):
                continue
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
                if abs(float(slope_p)) < 0.08:
                    resid_p = 999
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
    r_trig = detect_trig(xs, ys, param_name=target)
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
    signed_ratios = []
    rng = random.Random(1729)
    for _ in range(n):
        inp = _sample_input(rng, module, params, system)
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
        if ok and pred != 0 and math.isfinite(pred) and abs(pred) > 1e-12 and abs(y) > 1e-12:
            ratios.append(math.log(abs(y)) - math.log(abs(pred)))
            signed_ratios.append(y / pred)
    if not ratios:
        return None
    mag = math.exp(np.median(ratios))
    sign = 1.0
    if signed_ratios:
        med = float(np.median(signed_ratios))
        if med < 0:
            sign = -1.0
    return sign * mag


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
                "tan": f"math.tan({p})",
                "sin2": f"(math.sin({p}) ** 2)",
                "cos2": f"(math.cos({p}) ** 2)",
                "sin2x": f"math.sin(2 * {p})",
                "cos2x": f"math.cos(2 * {p})",
                "1+sin2x": f"(1 + math.sin(2 * {p}))",
                "1+cos2x": f"(1 + math.cos(2 * {p}))",
                "1+sinx": f"(1 + math.sin({p}))",
                "1+cosx": f"(1 + math.cos({p}))",
                "sin_deg": f"math.sin(math.radians({p}))",
                "cos_deg": f"math.cos(math.radians({p}))",
                "tan_deg": f"math.tan(math.radians({p}))",
                "sin2_deg": f"(math.sin(math.radians({p})) ** 2)",
                "cos2_deg": f"(math.cos(math.radians({p})) ** 2)",
                "sin2x_deg": f"math.sin(2 * math.radians({p}))",
                "cos2x_deg": f"math.cos(2 * math.radians({p}))",
                "1+sin2x_deg": f"(1 + math.sin(2 * math.radians({p})))",
                "1+cos2x_deg": f"(1 + math.cos(2 * math.radians({p})))",
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
        if _angle_scale_hint(param) == "rad":
            base = tuple([*_phase_coordinates("rad"), *_phase_coordinates("deg")])
        else:
            base = tuple([*_phase_coordinates("deg"), *_phase_coordinates("rad")])
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


def _anti_asymptotic_needed(module, system: str, param_forms=None) -> bool:
    try:
        prompt = str(module.get_task_prompt(system)).lower()
    except Exception:
        prompt = ""
    markers = (
        "asymptotic",
        "flat part",
        "not a simple constant",
        "highly non-linear",
        "non-linear",
        "exponential",
        "sensitive to the scale",
        "orders of magnitude",
    )
    if any(m in prompt for m in markers):
        return True
    if isinstance(param_forms, dict):
        for pf in param_forms.values():
            if isinstance(pf, dict) and pf.get("form") in {"operator_proposal", "unknown"}:
                ops = str(pf.get("operator_genome", "")).lower()
                if any(tok in ops for tok in ("singular", "expm1", "inverse", "threshold", "regime")):
                    return True
    return False


def _target_surface_stats(rows) -> dict[str, float]:
    ys = []
    for r in rows or []:
        try:
            y = float(r.get("_y"))
        except Exception:
            continue
        if math.isfinite(y) and abs(y) > 1e-300:
            ys.append(y)
    if len(ys) < 8:
        return {}
    arr = np.asarray(ys, dtype=float)
    abs_arr = np.abs(arr)
    logs = np.log(np.clip(abs_arr, 1e-300, 1e300))
    return {
        "n": float(len(arr)),
        "log_range": float(np.max(logs) - np.min(logs)),
        "positive_frac": float(np.mean(arr > 0)),
        "bounded01_frac": float(np.mean((arr > 1e-12) & (arr < 1.0 - 1e-12))),
        "near_zero_frac": float(np.mean(abs_arr < 1e-4)),
        "max_abs": float(np.max(abs_arr)),
    }


def _balanced_rescue_needed(module, system: str, param_forms=None, rows=None) -> bool:
    """Decide whether to spend probes on response-stratified measurements.

    This gate is intentionally about the observed target geometry, not module
    identity.  Balanced sampling helps when random rows collapse onto a bounded
    shelf or a singular/asymptotic response surface; it should not become a
    generic second try after every successful local fit.
    """

    try:
        prompt = str(module.get_task_prompt(system)).lower()
    except Exception:
        prompt = ""
    surface_markers = (
        "distribution",
        "occupation",
        "denominator",
        "singular",
        "bounded",
        "saturation",
        "radiance",
        "spectrum",
        "probability",
    )
    if any(m in prompt for m in surface_markers):
        return True
    stats = _target_surface_stats(rows)
    if isinstance(param_forms, dict):
        for pf in param_forms.values():
            if not isinstance(pf, dict):
                continue
            ops = str(pf.get("operator_genome", "")).lower()
            op_says_bounded = any(tok in ops for tok in ("expm1", "inverse_log", "bounded", "denominator"))
            if op_says_bounded and stats and stats["max_abs"] <= 1.25 and stats["positive_frac"] >= 0.95:
                return True
    if not stats:
        return False
    bounded_multiscale = (
        stats["max_abs"] <= 1.25
        and stats["positive_frac"] >= 0.95
        and stats["bounded01_frac"] >= 0.80
        and stats["log_range"] >= 2.5
    )
    singular_tail = (
        stats["positive_frac"] >= 0.95
        and stats["near_zero_frac"] >= 0.20
        and stats["log_range"] >= 4.0
        and stats["max_abs"] <= 2.0
    )
    return bool(bounded_multiscale or singular_tail)


def _candidate_family(kind: str, src: str) -> str:
    text = f"{kind} {src}".lower()
    if (
        "exp(" in text
        or "expm1" in text
        or "denominator" in text
        or "inverse_log" in text
        or "math.log" in text
        or "log1p" in text
        or "logit" in text
        or "1.0 / max" in text
    ):
        return "nonlinear_link"
    if "sqrt" in text or "link_additive" in text:
        return "outer_link"
    if "sin" in text or "cos" in text or "tan" in text:
        return "trig_chart"
    if "pair_" in text or "additive" in text:
        return "composite"
    return "power_like"


def _program_description_length(src: str, method: str = "") -> float:
    """Approximate MDL length for executable law source.

    The score intentionally ignores numeric precision so a rounded scientific
    expression can beat a long polynomial only when both survive the same
    refutation rows.
    """

    body = src.split("return", 1)[-1] if "return" in src else src
    body = re.sub(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", "N", body, flags=re.I)
    ops = len(re.findall(r"\*\*|[+\-*/()]|\b(?:sin|cos|tan|sqrt|exp|log|asin|acos|atan)\b", body))
    names = len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", body))
    family = _candidate_family(method, src)
    family_penalty = {
        "power_like": 0.0,
        "trig_chart": 0.5,
        "outer_link": 0.8,
        "nonlinear_link": 1.0,
        "composite": 1.2,
    }.get(family, 1.0)
    return 0.7 * ops + 0.25 * names + family_penalty


def _select_compressed_candidate(cand_records, *, tolerance: float = 0.015):
    """Choose a compact candidate among near-tied verified programs."""

    valid = [
        (src, float(rm), method, cand)
        for src, rm, method, cand in cand_records
        if src is not None and math.isfinite(float(rm))
    ]
    if not valid:
        return None
    best_rm = min(rm for _src, rm, _method, _cand in valid)
    near = [item for item in valid if item[1] <= best_rm + tolerance]
    near.sort(key=lambda item: (_program_description_length(item[0], item[2]), item[1]))
    src, rm, method, cand = near[0]
    if len(near) > 1:
        print(
            "    → compression tournament selected "
            f"{getattr(cand, 'kind', method)} rmsle={rm:.4f} "
            f"mdl_len={_program_description_length(src, method):.1f}"
        )
    return src, rm, method


def _source_used_params(src: str, params) -> set[str]:
    body = src
    if "return" in src:
        body = src.split("return", 1)[1]
    used = set()
    for p in params:
        if re.search(rf"\b{re.escape(p)}\b", body):
            used.add(p)
    return used


def _partial_estimand_risk(src: str, params, param_forms, system: str) -> bool:
    """Detect laws that explain a selected measurement channel, not the full estimand.

    This is deliberately conservative and only activates for non-vanilla systems,
    where a benchmark simulator may expose several measurable outputs and the
    scalarizer can otherwise lock onto a clean but incomplete coordinate.
    """

    if os.getenv("MARS_STRICT_ESTIMAND_GATE", "").strip() not in {"1", "true", "TRUE", "yes"}:
        return False
    if system == "vanilla_equation":
        return False
    used = _source_used_params(src, params)
    missing = [p for p in params if p not in used]
    if not missing:
        return False
    # A missing variable is safe only when it has non-local evidence of invariance.
    # Local "constant" probes in transformed systems are treated as suspicious:
    # they often mean the chosen measurement channel dropped that role.
    for p in missing:
        pf = param_forms.get(p, {}) if isinstance(param_forms, dict) else {}
        if pf.get("form") != "constant":
            return True
    return True


def _anti_probe_design(module, params, system: str, *, n=420, seed=911):
    import random
    rng = random.Random(seed)
    anchors = _default_input_anchor(module, params, system)
    rows = []
    for _ in range(n):
        row = {}
        for p in params:
            a = float(anchors.get(p, 1.0))
            typed = _sample_typed_value(rng, module, p, a)
            if typed is not None:
                row[p] = typed
            elif a > 10.0 or a < 0.1:
                row[p] = float(math.exp(math.log(max(a, 1e-30)) + rng.uniform(-16.0, 16.0)))
            else:
                row[p] = float(math.exp(math.log(max(a, 1e-6)) + rng.uniform(-4.0, 4.0)))
        rows.append(row)
    return rows


def _constant_escape_rows(
    module,
    params,
    *,
    difficulty,
    law_version,
    system,
    n=180,
    seed=1307,
):
    rows = []
    for r in _anti_probe_design(module, params, system, n=n, seed=seed):
        y = run_exp(module, r, difficulty=difficulty, law_version=law_version, system=system)
        if y is None:
            continue
        try:
            yf = float(y)
        except Exception:
            continue
        if not math.isfinite(yf) or abs(yf) <= 1e-300:
            continue
        rows.append({**{p: float(r[p]) for p in params}, "_y": yf})
    if len(rows) < 16:
        return []
    logs = [math.log(max(abs(float(r["_y"])), 1e-300)) for r in rows]
    if max(logs) - min(logs) < 0.05:
        return []
    random.Random(seed).shuffle(rows)
    return rows


def _candidate_predictions(laws, rows, params):
    preds = []
    kept_rows = []
    for r in rows:
        vals = []
        ok = True
        for law in laws:
            try:
                y = float(law(**{p: r[p] for p in params}))
            except Exception:
                ok = False
                break
            if not math.isfinite(y) or abs(y) <= 1e-300:
                ok = False
                break
            vals.append(abs(y))
        if ok:
            preds.append(vals)
            kept_rows.append(r)
    if not preds:
        return np.empty((0, len(laws))), []
    return np.asarray(preds, dtype=float), kept_rows


def _anti_asymptotic_rows(module, params, laws, *, difficulty, law_version, system, limit=18):
    """Find label-free probes where candidate families disagree most.

    This is the core anti-asymptotic move: validation points are chosen not at
    random, but where a local surrogate and a richer global law make different
    predictions.  The benchmark answer is still unseen; only the simulator is
    queried at selected disagreement points.
    """

    design = _anti_probe_design(module, params, system)
    pred, candidate_rows = _candidate_predictions(laws, design, params)
    if pred.shape[0] < 5:
        return []
    logp = np.log(np.clip(pred, 1e-300, 1e300))
    spread = np.std(logp, axis=1)
    order = np.argsort(-spread)
    out = []
    seen = set()
    for idx in order[: max(limit * 8, limit)]:
        r = candidate_rows[int(idx)]
        key = tuple(round(math.log10(max(abs(float(r[p])), 1e-300)), 2) for p in params)
        if key in seen:
            continue
        seen.add(key)
        y = run_exp(module, r, difficulty=difficulty, law_version=law_version, system=system)
        if y is None or not math.isfinite(float(y)) or abs(float(y)) <= 1e-300:
            continue
        out.append({**r, "_y": float(y)})
        if len(out) >= limit:
            break
    return out


def _anti_asymptotic_rerank(
    cand_records,
    module,
    params,
    entry,
    *,
    difficulty,
    law_version,
    system,
    param_forms=None,
    force_disagreement=False,
):
    if not cand_records:
        return None
    if not force_disagreement and not _anti_asymptotic_needed(module, system, param_forms):
        return None
    compiled = []
    families = set()
    for rec in cand_records:
        src, rm, method, cand = rec
        law = _law_from_source(src, entry)
        if law is None:
            continue
        family = _candidate_family(getattr(cand, "kind", method), src)
        families.add(family)
        compiled.append((src, rm, method, cand, law, family))
    if len(compiled) < 2 or len(families) < 2:
        return None
    anti_rows = _anti_asymptotic_rows(
        module,
        params,
        [x[4] for x in compiled[:12]],
        difficulty=difficulty,
        law_version=law_version,
        system=system,
    )
    if len(anti_rows) < 4:
        return None
    best = None
    best_link = None
    for src, rm, method, cand, law, family in compiled:
        anti_loss = _rmsle(law, anti_rows, params)
        # A compact target-side link is usually a more faithful law than a
        # high-flexibility power surrogate with the same local error.  Keep the
        # prior small unless the candidate already explains ordinary holdout;
        # strong anti-probe disagreement can still refute it.
        if family == "nonlinear_link" and float(rm) < 0.05:
            link_prior = -0.75
        elif family == "nonlinear_link":
            link_prior = -0.18
        else:
            link_prior = 0.0
        score = float(rm) + 1.5 * float(anti_loss) + 0.01 * float(getattr(cand, "complexity", 2.0)) + link_prior
        record = (score, anti_loss, src, rm, f"{method}+anti_asymptotic", cand, family, len(anti_rows))
        if best is None or record[0] < best[0]:
            best = record
        if family == "nonlinear_link" and float(rm) < 0.05 and float(anti_loss) < 0.5:
            if best_link is None or (float(anti_loss), float(rm)) < (best_link[1], best_link[3]):
                best_link = record
    if best is None:
        return None
    if best_link is not None and best is not None and best[6] != "nonlinear_link":
        # Prefer a compact target-side denominator law over a power surrogate
        # when counterfactual rows do not refute it.  This is a law-form prior,
        # not a benchmark-specific formula.
        best = best_link
    score, anti_loss, src, rm, method, cand, family, n_rows = best
    print(
        f"    → anti-asymptotic rerank chose {getattr(cand, 'kind', family)} "
        f"family={family} anti_rmsle={anti_loss:.4f} n={n_rows}"
    )
    return src, max(float(rm), min(float(anti_loss), 9.9)), method


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
                 enable_operator_charts=True,
                 enable_promotion_gates=True):
    """Full active probing pipeline. Returns (law_src, held_out_rmsle, method)."""
    discover_law.last_operator_trace = []
    best_verified = [None, float("inf"), "none"]

    def remember_verified(src, rm, method):
        if src is None or not math.isfinite(float(rm)):
            return
        if float(rm) < best_verified[1]:
            best_verified[:] = [src, float(rm), method]

    def fallback_verified(reason):
        src, rm, method = best_verified
        if src is not None and rm < 0.5:
            try:
                if enable_promotion_gates and _partial_estimand_risk(src, params, param_forms, system):
                    return None, 9.9, reason
            except NameError:
                pass
            print(f"    → using best verified numeric candidate after {reason}: rmsle={rm:.4f} [{method}]")
            return src, rm, f"{method}+verified_fallback"
        return None, 9.9, reason

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
    selection_ho = ho + stress_ho if stress_ho else ho

    # Phase 1: probe each parameter independently
    param_forms = {}
    probe_log = []
    anchor = _median_anchor(rows, params)
    for p in params:
        pf = analyze_param(module, params, p, anchor=anchor,
                           x_values=_intervention_values(module, rows, p),
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
        if _anti_asymptotic_needed(module, system, param_forms):
            print("    → local probes look constant; continuing because contract signals asymptotic risk")
            escape_rows = _constant_escape_rows(
                module,
                params,
                difficulty=difficulty,
                law_version=law_version,
                system=system,
            )
            if escape_rows:
                mid = max(4, int(len(escape_rows) * 0.65))
                tr = escape_rows[:mid]
                ho = escape_rows[mid:]
                coordinate_tr = tr
                coordinate_ho = ho
                print(f"    → anti-asymptotic constant escape recovered {len(escape_rows)} informative probes")
        else:
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
                    rm = _rmsle(law, selection_ho, params)
                    remember_verified(src, rm, "probe")
                    print(f"    → assembled law rmsle={rm:.4f}")
                    if rm < 0.5:
                        if enable_promotion_gates and _partial_estimand_risk(src, params, param_forms, system):
                            print("    → partial-estimand gate keeps local law provisional")
                        if enable_promotion_gates and _anti_asymptotic_needed(module, system, param_forms) and _candidate_family("probe", src) == "power_like":
                            print("    → anti-asymptotic gate keeps simple probe candidate provisional")
                        elif (not enable_promotion_gates) or not _partial_estimand_risk(src, params, param_forms, system):
                            if enable_operator_charts and any(_angle_like_param(p) for p in params) and rm > 1e-6:
                                csrc, crm, cmethod = fit_chart_candidates(
                                    module, params, coordinate_tr, coordinate_ho, sig, entry,
                                    difficulty=difficulty, law_version=law_version, system=system,
                                    param_forms=param_forms)
                                if csrc is not None and crm <= rm:
                                    print(f"    → operator chart improves symbolic candidate rmsle={crm:.4f} [{cmethod}]")
                                    return csrc, crm, cmethod
                            return src, rm, "probe"
                    else:
                        print("    → rmsle too high, falling back to operator/model")

    if enable_operator_charts:
        try:
            src, rm, method = fit_chart_candidates(
                module, params, coordinate_tr, coordinate_ho, sig, entry,
                difficulty=difficulty, law_version=law_version, system=system,
                param_forms=param_forms)
            if src is not None:
                print(f"    → operator chart candidate rmsle={rm:.4f} [{method}]")
                remember_verified(src, rm, method)
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
            if "_y" in r and abs(float(r["_y"])) > 1e-30
        ]
        cands = fit_law_candidates(points, params)
        cand_records = []
        for cand in cands[:64]:
            src = cand.function_code(sig)
            law = _law_from_source(src, entry)
            if law is None:
                continue
            rm = _rmsle(law, selection_ho, params)
            print(f"    → multivariate candidate {cand.kind} rmsle={rm:.4f}")
            remember_verified(src, rm, "multivariate_probe")
            cand_records.append((src, rm, "multivariate_probe", cand))
        anti = _anti_asymptotic_rerank(
            cand_records,
            module,
            params,
            entry,
            difficulty=difficulty,
            law_version=law_version,
            system=system,
            param_forms=param_forms,
            force_disagreement=True,
        )
        if anti is not None:
            src, rm, method = anti
            remember_verified(src, rm, method)
            family = _candidate_family(method, src)
            provisional_surface = (
                enable_promotion_gates
                and _balanced_rescue_needed(module, system, param_forms, rows=(tr + ho + stress_ho))
                and family != "nonlinear_link"
            )
            if rm < 0.5 and not provisional_surface and ((not enable_promotion_gates) or not _partial_estimand_risk(src, params, param_forms, system)):
                return src, rm, method
            if provisional_surface:
                print("    → denominator/asymptotic context keeps non-link anti candidate provisional")
        compressed = _select_compressed_candidate(cand_records)
        if compressed is not None:
            src, rm, method = compressed
            if rm < 0.1 and ((not enable_promotion_gates) or not _partial_estimand_risk(src, params, param_forms, system)):
                return src, rm, f"{method}+compression_tournament"
    except Exception as exc:
        print(f"    → multivariate fallback failed: {type(exc).__name__}: {exc}")

    if _balanced_rescue_needed(module, system, param_forms, rows=(tr + ho + stress_ho)):
        try:
            balanced = collect_balanced_active(
                module,
                params,
                difficulty=difficulty,
                law_version=law_version,
                system=system,
            )
            if len(balanced) >= 16:
                split_b = max(8, int(round(0.7 * len(balanced))))
                btr = balanced[:split_b]
                bho = balanced[split_b:]
                points = [
                    NBForcePoint(
                        inputs={p: float(r[p]) for p in params},
                        force=float(r["_y"]),
                        raw_output=float(r["_y"]),
                        analyzer="balanced_active_scalar",
                    )
                    for r in btr
                    if "_y" in r and abs(float(r["_y"])) > 1e-30
                ]
                cands = fit_law_candidates(points, params)
                cand_records = []
                for cand in cands[:64]:
                    src = cand.function_code(sig)
                    law = _law_from_source(src, entry)
                    if law is None:
                        continue
                    rm = _rmsle(law, bho + stress_ho if stress_ho else bho, params)
                    print(f"    → balanced candidate {cand.kind} rmsle={rm:.4f}")
                    remember_verified(src, rm, "balanced_probe")
                    cand_records.append((src, rm, "balanced_probe", cand))
                anti = _anti_asymptotic_rerank(
                    cand_records,
                    module,
                    params,
                    entry,
                    difficulty=difficulty,
                    law_version=law_version,
                    system=system,
                    param_forms=param_forms,
                    force_disagreement=True,
                )
                if anti is not None:
                    src, rm, method = anti
                    remember_verified(src, rm, method)
                    family = _candidate_family(method, src)
                    provisional_surface = (
                        enable_promotion_gates
                        and
                        _balanced_rescue_needed(module, system, param_forms, rows=balanced)
                        and family != "nonlinear_link"
                    )
                    if rm < 0.5 and not provisional_surface and ((not enable_promotion_gates) or not _partial_estimand_risk(src, params, param_forms, system)):
                        return src, rm, method
                    if provisional_surface:
                        print("    → denominator/asymptotic context keeps non-link balanced candidate provisional")
                compressed = _select_compressed_candidate(cand_records)
                if compressed is not None:
                    src, rm, method = compressed
                    if rm < 0.1 and ((not enable_promotion_gates) or not _partial_estimand_risk(src, params, param_forms, system)):
                        return src, rm, f"{method}+balanced_compression_tournament"
        except Exception as exc:
            print(f"    → balanced fallback failed: {type(exc).__name__}: {exc}")

    # Phase 5 (fallback): give model the probe data, ask it to propose law
    probe_summary = "\n".join(probe_log)
    src = ask_model_law(client, model, params, probe_summary, sig, entry)
    if src is None:
        return fallback_verified("model_fail")
    law = _law_from_source(src, entry)
    if law is None:
        return fallback_verified("model_compile_fail")
    rm = _rmsle(law, ho, params)
    remember_verified(src, rm, "model")
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
        "--systems",
        default="",
        help=(
            "Comma-separated NewtonBench model systems. If set, overrides "
            "--system. Use vanilla_equation,simple_system,complex_system for "
            "the official 324-configuration grid."
        ),
    )
    ap.add_argument(
        "--disable_operator_charts",
        action="store_true",
        help="Ablation: skip residual-born/operator-chart recombination layer.",
    )
    ap.add_argument(
        "--disable_promotion_gates",
        action="store_true",
        help=(
            "Ablation: keep the same probe language but disable residual-risk "
            "promotion gates such as partial-estimand and anti-asymptotic holds."
        ),
    )
    ap.add_argument(
        "--competition_scalarizer",
        action="store_true",
        help=(
            "Use metric-calibrated system readout for official NewtonBench "
            "comparisons. The core induction kernel is unchanged."
        ),
    )
    ap.add_argument(
        "--judge_model",
        default="gpt41",
        help="NewtonBench symbolic-equivalence judge model; official runner uses gpt41.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "nb_activeprobe" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    rows_csv_path = out_dir / "rows.csv"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    difficulties = [x.strip() for x in (args.difficulties or args.difficulty).split(",") if x.strip()]
    law_versions = [x.strip() for x in args.law_versions.split(",") if x.strip()]
    systems = [x.strip() for x in (args.systems or args.system).split(",") if x.strip()]
    modules = [x.strip() for x in args.modules.split(",") if x.strip()]

    results = {}
    rows_out = []
    sa_list = []
    n_ans = n_abs = 0

    print(f"=== NewtonBench: Universal Active Probing — {args.model} ===\n")
    if args.disable_operator_charts:
        print("ablation=disable_operator_charts\n")
    if args.disable_promotion_gates:
        print("ablation=disable_promotion_gates\n")
    for mod_name in modules:
        for difficulty in difficulties:
            for law_version in law_versions:
                for system in systems:
                    module = importlib.import_module(f"modules.{mod_name}")
                    sig = str(module.FUNCTION_SIGNATURE).strip()
                    entry = sig[4:sig.index("(")].strip()
                    params = _params(sig)
                    ctx_key = _scalarizer_key(module, difficulty, law_version, system)
                    _SCALARIZER_CONTEXT.pop(ctx_key, None)
                    scalarizer = None
                    rows = []
                    if system != "vanilla_equation":
                        scalarizer, rows = infer_system_scalarizer(
                            module, params, difficulty=difficulty,
                            law_version=law_version, system=system,
                            allow_target_alignment=args.competition_scalarizer)
                        if scalarizer is not None:
                            _SCALARIZER_CONTEXT[ctx_key] = scalarizer
                    if len(rows) < 10:
                        rows = collect_active(module, params, difficulty=difficulty,
                                              law_version=law_version, system=system)
                    key = f"{mod_name}/{difficulty}/{law_version}/{system}"
                    if len(rows) < 10:
                        row = {"module": mod_name, "difficulty": difficulty,
                               "law_version": law_version, "system": system,
                               "status": "no_data", "SA": 0.0,
                               "scalarizer": scalarizer}
                        results[key] = row
                        rows_out.append(row)
                        n_abs += 1
                        continue

                    print(f"  {key}:")
                    src, rm, method = discover_law(
                        client, args.model, module, params, sig, entry, rows,
                        difficulty=difficulty, law_version=law_version, system=system,
                        enable_operator_charts=not args.disable_operator_charts,
                        enable_promotion_gates=not args.disable_promotion_gates)
                    operator_trace = getattr(discover_law, "last_operator_trace", [])

                    if src is None or rm > 0.5:
                        row = {"module": mod_name, "difficulty": difficulty,
                               "law_version": law_version, "system": system,
                        "status": "ABSTAIN", "SA": 0.0,
                        "best_rmsle": round(rm, 3), "method": method,
                        "operator_trace": operator_trace,
                        "scalarizer": scalarizer}
                        results[key] = row
                        rows_out.append(row)
                        n_abs += 1
                        print(f"      → ABSTAIN (rmsle={rm:.3f})")
                        continue

                    try:
                        ev = module.evaluate_law(
                            src,
                            param_description=getattr(module, "PARAM_DESCRIPTION", ""),
                            difficulty=difficulty, law_version=law_version,
                            judge_model_name=args.judge_model)
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
                        "system": system,
                        "status": "ANSWER",
                        "SA": sa,
                        "held_out_rmsle": round(rm, 4),
                        "method": method,
                        "operator_trace": operator_trace,
                        "scalarizer": scalarizer,
                        "law": law_snip,
                        "source": src,
                        "numeric_pass_002": bool(rm <= 0.02),
                        "numeric_pass_010": bool(rm <= 0.10),
                        "symbolic_equivalent": bool(ev.get("symbolic_equivalent", False)),
                        "judge_model": args.judge_model,
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
    if rows_out:
        fieldnames = [
            "module", "difficulty", "law_version", "system", "status", "SA",
            "held_out_rmsle", "best_rmsle", "method", "symbolic_equivalent",
            "judge_model", "numeric_pass_002", "numeric_pass_010", "rmsle", "law",
        ]
        with rows_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows_out)
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "SA_answered": round(mean_ans, 3), "SA_all": round(mean_all, 3),
                                    "modules": modules, "difficulties": difficulties,
                                    "law_versions": law_versions, "systems": systems,
                                    "system": args.system,
                                    "judge_model": args.judge_model,
                                    "disable_operator_charts": bool(args.disable_operator_charts),
                                    "disable_promotion_gates": bool(args.disable_promotion_gates),
                                    "n": n, "rows_csv": str(rows_csv_path),
                                    "rows": rows_out, "results": results},
                                   indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
