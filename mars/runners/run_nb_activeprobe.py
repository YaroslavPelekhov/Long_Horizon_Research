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
import json
import math
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
from mars.runners.run_nb_selfverify import _params, collect
from mars.runners.run_nb_fitters import _law_from_source, _rmsle

WEAK = "openai/gpt-4o-mini"
MODULES = ["m0_gravity", "m1_coulomb_force", "m9_hooke_law",
           "m4_snell_law", "m5_radioactive_decay", "m7_malus_law"]
EXP_ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0]
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


def run_exp(module, inp, difficulty="easy"):
    try:
        y = float(module.run_experiment_for_module(
            noise_level=0.0, difficulty=difficulty,
            system="vanilla_equation", law_version="v0", **inp))
        return y if math.isfinite(y) else None
    except Exception:
        return None


def probe_param(module, params, target, n=15, base=1.0):
    xs = np.logspace(-1.5, 1.5, n)
    ys = []
    for x in xs:
        inp = {p: base for p in params}
        inp[target] = float(x)
        y = run_exp(module, inp)
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
        snapped = round(slope * 4) / 4   # snap to nearest 0.25
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


def analyze_param(module, params, target):
    xs, ys = probe_param(module, params, target)
    if len(xs) < 4:
        return {"form": "unknown", "resid": 999}
    is_const, const_val = detect_constant(xs, ys)
    if is_const:
        return {"form": "constant", "value": const_val, "resid": 0.0}
    r_pow = detect_power(xs, ys)
    if r_pow:
        return r_pow
    r_exp = detect_exp(xs, ys)
    if r_exp:
        return r_exp
    r_trig = detect_trig(xs, ys)
    if r_trig:
        return r_trig
    return {"form": "unknown", "xs": xs.tolist(), "ys": ys.tolist(), "resid": 999}


def detect_joint_exp(module, params, exp_params, power_forms):
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
        y = run_exp(module, inp)
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
    # Snap alphas to nearest 0.25
    alphas_snapped = [round(a * 4) / 4 for a in alphas]
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


def calibrate_const(module, params, param_forms, n=20, base=1.0):
    """Calibrate overall multiplicative constant by comparing predictions to measurements."""
    ratios = []
    for _ in range(n):
        inp = {p: round(0.3 + np.random.random() * 4, 3) for p in params}
        y = run_exp(module, inp)
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


def discover_law(client, model, module, params, sig, entry, rows, difficulty="easy"):
    """Full active probing pipeline. Returns (law_src, held_out_rmsle, method)."""
    np.random.seed(42)
    tr = rows[:int(len(rows) * 0.6)]
    ho = rows[int(len(rows) * 0.6):]

    # Phase 1: probe each parameter independently
    param_forms = {}
    probe_log = []
    for p in params:
        pf = analyze_param(module, params, p)
        param_forms[p] = pf
        probe_log.append(f"  {p}: {pf}")

    print(f"    probe results:")
    for line in probe_log:
        print(f"    {line}")

    all_known = all(pf["form"] != "unknown" for pf in param_forms.values())

    # All-constant law → abstain (module returns constant, no scientific content)
    if all(pf["form"] == "constant" for pf in param_forms.values()):
        print("    → all params constant, abstain")
        return None, 9.9, "all_constant"

    # Detect multi-exp interaction (e.g. exp(-lambda * t^1.5))
    exp_params = [p for p, pf in param_forms.items() if pf["form"] == "exp"]
    if len(exp_params) >= 2:
        power_forms = {p: pf for p, pf in param_forms.items() if pf["form"] == "power"}
        ji = detect_joint_exp(module, params, exp_params, power_forms)
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
        const = calibrate_const(module, params, param_forms)
        if const is None:
            print("    → constant calibration failed")
            return None, 9.9, "calib_fail"

        # Phase 3: build law source
        src = build_law_source(params, param_forms, const, entry, sig)
        if src is None:
            print("    → law assembly failed")
            return None, 9.9, "assembly_fail"

        # Phase 4: verify
        law = _law_from_source(src, entry)
        if law is None:
            print(f"    → law compile failed\n    src: {src[:200]}")
            return None, 9.9, "compile_fail"
        rm = _rmsle(law, ho, params)
        print(f"    → assembled law rmsle={rm:.4f}")
        if rm < 0.5:
            return src, rm, "probe"
        print("    → rmsle too high, falling back to model")

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
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "nb_activeprobe" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    results = {}
    sa_list = []
    n_ans = n_abs = 0

    print(f"=== NewtonBench: Universal Active Probing — {args.model} ===\n")
    for mod_name in args.modules.split(","):
        module = importlib.import_module(f"modules.{mod_name}")
        sig = str(module.FUNCTION_SIGNATURE).strip()
        entry = sig[4:sig.index("(")].strip()
        params = _params(sig)
        rows = collect(module, params, difficulty=args.difficulty)
        if len(rows) < 10:
            results[mod_name] = {"status": "no_data"}; n_abs += 1; continue

        print(f"  {mod_name}:")
        src, rm, method = discover_law(client, args.model, module, params, sig, entry, rows, args.difficulty)

        if src is None or rm > 0.5:
            results[mod_name] = {"status": "ABSTAIN", "best_rmsle": round(rm, 3), "method": method}
            n_abs += 1
            print(f"      → ABSTAIN (rmsle={rm:.3f})")
            continue

        try:
            ev = module.evaluate_law(
                src,
                param_description=getattr(module, "PARAM_DESCRIPTION", ""),
                difficulty=args.difficulty, law_version="v0", judge_model_name="gpt41")
            sa = float(ev.get("exact_accuracy", 0.0))
        except Exception:
            sa = 0.0

        sa_list.append(sa); n_ans += 1
        law_snip = src.split("return", 1)[-1].strip()[:70]
        results[mod_name] = {"status": "ANSWER", "SA": sa, "held_out_rmsle": round(rm, 4),
                             "method": method, "law": law_snip}
        print(f"      → SA={sa:.2f} rmsle={rm:.4f} [{method}] | {law_snip[:50]}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_ans + n_abs
    mean_ans = sum(sa_list) / len(sa_list) if sa_list else 0.0
    mean_all = sum(sa_list) / n if n else 0.0
    print(f"\n=== RESULT ({args.difficulty}) ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  SA on answered = {mean_ans:.2f}  |  SA over all = {mean_all:.2f}")
    print(f"  self-debug baseline: SA_all=0.17 | non-thinking GPT-4.1 baseline: ~0.06")
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "SA_answered": round(mean_ans, 3), "SA_all": round(mean_all, 3),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
