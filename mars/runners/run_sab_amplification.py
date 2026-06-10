"""ScienceAgentBench — execution-verified amplification on CODE.

This is the cleanest test of the thesis: a generated program is correct iff it
runs and its output passes the official eval script. Execution is PERFECTLY
aligned with the metric, so amplification should show clearly.

    weak (gpt-4o-mini) + EVA-style amplification   >=?   strong (gpt-4o) raw

Conditions (same tasks, official SAB success metric):
  A. weak_raw   : gpt-4o-mini writes the program (single shot)
  B. strong_raw : gpt-4o writes the program (single shot)
  C. weak_amp   : gpt-4o-mini proposes K programs; we RUN each (the run is the
                  judge: did it execute and produce the required output?), keep
                  survivors, and refine failures with the real error trace.
                  The official eval script is used ONLY for final scoring, never
                  inside the loop (no oracle leakage).

No task-specific logic: the amplifier only knows "run this program, did it
produce the output file without error". The same loop works for any code task.

Run:
  python -m mars.runners.run_sab_amplification --run_id sab_amp_v1 \\
    --max_tasks 12 --weak openai/gpt-4o-mini --strong openai/gpt-4o \\
    --k 3 --rounds 2 --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
_SAB = _PROJ / "scienceagentbench_repo"
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from dotenv import load_dotenv
    _env = _PROJ / "autodiscovery" / ".env.local"
    if _env.exists():
        load_dotenv(_env, override=True)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client  # noqa: E402

_AVAILABLE = {"pandas", "numpy", "sklearn", "scipy", "matplotlib", "torch",
              "seaborn", "statsmodels", "networkx", "PIL", "skimage"}
_UNAVAILABLE_KW = ["geopandas", "rdkit", "deepchem", "rasterio", "shapely",
                   "cartopy", "osmnx", "pyproj", "fiona", "tensorflow", "dgl",
                   "biopython", "Bio", "mne", "oggm", "pymatgen", "xarray", "netCDF4"]


def load_ready_tasks(max_tasks: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("osunlp/ScienceAgentBench", split="verified")
    ready = []
    for t in ds:
        folder = t["dataset_folder_tree"].split("\n")[0].replace("|-- ", "").strip().rstrip("/")
        gold = t["gold_program_name"]
        ev = t["eval_script_name"]
        out = t["output_fname"]
        if not (_SAB / "benchmark/datasets" / folder).is_dir():
            continue
        if not (_SAB / "benchmark/gold_programs" / gold).is_file():
            continue
        eval_path = _SAB / "benchmark/eval_programs" / ev
        if not eval_path.is_file():
            continue
        # skip tasks needing libs we don't have (scan gold program imports)
        gold_src = (_SAB / "benchmark/gold_programs" / gold).read_text(errors="ignore")
        eval_src = eval_path.read_text(errors="ignore")
        if any(kw in gold_src or kw in eval_src for kw in _UNAVAILABLE_KW):
            continue
        # eval must reference a gold_results file that exists
        ready.append({
            "id": t["instance_id"], "domain": t["domain"] or "",
            "task_inst": t["task_inst"] or "", "domain_knowledge": (t["domain_knowledge"] or "")[:800],
            "folder_tree": t["dataset_folder_tree"] or "", "preview": (t["dataset_preview"] or "")[:1500],
            "output_fname": out, "eval_script": ev, "gold": gold,
        })
        if len(ready) >= max_tasks:
            break
    return ready


def build_prompt(t: dict) -> str:
    return (
        f"Write a complete, self-contained Python program for this scientific task.\n\n"
        f"TASK: {t['task_inst']}\n\n"
        f"DOMAIN KNOWLEDGE: {t['domain_knowledge']}\n\n"
        f"DATASET FILES (relative to working dir, under benchmark/datasets/):\n{t['folder_tree']}\n\n"
        f"DATA PREVIEW:\n{t['preview']}\n\n"
        f"REQUIREMENTS:\n"
        f"- The program runs from the repository root. Read inputs from "
        f"'benchmark/datasets/...'.\n"
        f"- It MUST save its result to EXACTLY this path: '{t['output_fname']}' "
        f"(create parent dirs).\n"
        f"- Use only: pandas, numpy, scikit-learn, scipy, matplotlib, torch.\n"
        f"- Self-contained, no CLI args, runnable as `python program.py`.\n"
    )


def _strip_code(s: str) -> str:
    s = s.strip()
    if "```" in s:
        import re
        m = re.search(r"```(?:python)?\n(.*?)```", s, re.DOTALL)
        if m:
            return m.group(1)
    return s


def run_program(code: str, output_fname: str, timeout: int = 150) -> tuple[bool, str]:
    """Run a candidate program from repo root in an isolated copy of cwd.
    Returns (produced_output_without_error, error_text). This RUN is the judge
    used inside amplification (not the official eval)."""
    # ensure pred_results/ exists (official harness provides it)
    (_SAB / "pred_results").mkdir(exist_ok=True)
    # write program to repo root temp file; clear stale output
    out_path = _SAB / output_fname
    if out_path.exists():
        try:
            out_path.unlink()
        except Exception:
            pass
    prog_file = _SAB / f"_cand_{os.getpid()}_{int(time.time()*1000)%100000}.py"
    prog_file.write_text(code, encoding="utf-8")
    try:
        res = subprocess.run([sys.executable, prog_file.name],
                             cwd=str(_SAB), capture_output=True, text=True, timeout=timeout)
        err = (res.stderr or "")[-1500:]
        produced = out_path.exists()
        ok = produced and res.returncode == 0
        return ok, ("" if ok else err or f"no output at {output_fname}")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        try:
            prog_file.unlink()
        except Exception:
            pass


def official_success(t: dict) -> int:
    """Run the official eval script; return 1 on success, 0 otherwise."""
    eval_path = _SAB / "benchmark/eval_programs" / t["eval_script"]
    try:
        res = subprocess.run([sys.executable, str(eval_path.name)],
                             cwd=str(_SAB / "benchmark/eval_programs") if False else str(_SAB),
                             capture_output=True, text=True, timeout=120)
        out = (res.stdout or "").strip()
        # eval prints a tuple like "(1, '{...}')"
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("("):
                try:
                    val = eval(line, {"__builtins__": {}}, {})
                    if isinstance(val, tuple):
                        return int(val[0])
                    return int(val)
                except Exception:
                    continue
        return 0
    except Exception:
        return 0


def gen_raw(client, model: str, t: dict) -> str:
    out = call_llm(client, model=model,
                   system="You are an expert scientific Python programmer. Return only code.",
                   user=build_prompt(t), max_tokens=2000, temperature=0.2)
    return _strip_code(out)


def amplify(client, model: str, t: dict, k: int, rounds: int) -> tuple[str, dict]:
    """Execution-verified amplification: propose K, run each (run = judge),
    keep a survivor that produced output; refine failures with real errors."""
    feedback = ""
    best_code = ""
    trace = []
    for rnd in range(rounds):
        prompt = build_prompt(t) + feedback + (
            "\nReturn ONLY JSON: {\"programs\": [\"<full python program>\", ...]} "
            f"with {k} diverse, complete candidate programs."
        )
        raw = call_llm(client, model=model,
                       system="You write complete scientific Python programs. Return only JSON.",
                       user=prompt, max_tokens=4000, temperature=0.6)
        cands = _parse_programs(raw)
        errs = []
        for code in cands[:k]:
            ok, err = run_program(code, t["output_fname"])
            if ok:
                return code, {"round": rnd + 1, "status": "executed", "trace": trace}
            errs.append(err)
            if not best_code:
                best_code = code
        trace.append({"round": rnd + 1, "n_cands": len(cands), "errors": [e[:120] for e in errs[:k]]})
        # refine with the most informative error
        if errs:
            feedback = ("\nYOUR PREVIOUS PROGRAMS FAILED. Real errors:\n" +
                        "\n".join(f"- {e[:200]}" for e in errs[:k]) +
                        "\nFix these specific errors. Ensure the output file is created.")
    return best_code, {"round": rounds, "status": "no_exec", "trace": trace}


def _parse_programs(raw: str) -> list[str]:
    if not raw:
        return []
    t = raw.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).lstrip()
    import re
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
    # fallback: single program in code fence
    return [_strip_code(raw)]


def score_condition(client, t: dict, code: str) -> tuple[int, int]:
    """Return (VER, SR): VER=program executed and produced output;
    SR=official eval passed. VER is what execution-verification can target;
    SR additionally requires scientific correctness (used only for reporting)."""
    if not code:
        return 0, 0
    ok, _ = run_program(code, t["output_fname"])
    if not ok:
        return 0, 0
    return 1, official_success(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="sab_amp_smoke")
    ap.add_argument("--max_tasks", type=int, default=10)
    ap.add_argument("--weak", default="openai/gpt-4o-mini")
    ap.add_argument("--strong", default="openai/gpt-4o")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = _PROJ / "lmw" / "sab_amp" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; pass --overwrite: {out_path}")

    client = make_openai_client()
    tasks = load_ready_tasks(args.max_tasks)
    print(f"=== SAB execution-verified amplification: weak+amp vs strong ===")
    print(f"weak={args.weak} strong={args.strong}  tasks={len(tasks)}\n", flush=True)

    rows = []
    for i, t in enumerate(tasks):
        cw = gen_raw(client, args.weak, t); vw, sw = score_condition(client, t, cw)
        cs = gen_raw(client, args.strong, t); vs, ss = score_condition(client, t, cs)
        ca, info = amplify(client, args.weak, t, args.k, args.rounds); va, sa = score_condition(client, t, ca)
        print(f"  [{i+1}/{len(tasks)}] id={t['id']:3d} {t['domain'][:16]:<16} "
              f"VER w/s/amp={vw}/{vs}/{va}  SR w/s/amp={sw}/{ss}/{sa}", flush=True)
        rows.append({"id": t["id"], "domain": t["domain"],
                     "VER_weak": vw, "VER_strong": vs, "VER_amp": va,
                     "SR_weak": sw, "SR_strong": ss, "SR_amp": sa, "amp_status": info["status"]})
        out_path.write_text(json.dumps({"partial": rows}, indent=2), encoding="utf-8")

    n = len(rows)
    def m(key): return sum(r[key] for r in rows)/n if n else 0
    summary = {
        "run_id": args.run_id, "score_type": "sab-execution-verified-amplification",
        "weak": args.weak, "strong": args.strong, "n_tasks": n,
        "VER_weak_raw": round(m("VER_weak"),3), "VER_strong_raw": round(m("VER_strong"),3),
        "VER_weak_amp": round(m("VER_amp"),3),
        "SR_weak_raw": round(m("SR_weak"),3), "SR_strong_raw": round(m("SR_strong"),3),
        "SR_weak_amp": round(m("SR_amp"),3),
        "VER_amplified_vs_strong": m("VER_amp") >= m("VER_strong"),
        "results": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== VER (valid execution rate — aligned with execution signal) ===")
    print(f"  weak_raw {m('VER_weak'):.2f}  strong_raw {m('VER_strong'):.2f}  "
          f"weak+amp {m('VER_amp'):.2f}   {'>= strong' if m('VER_amp')>=m('VER_strong') else '< strong'}")
    print(f"=== SR (success rate — needs scientific correctness, harder) ===")
    print(f"  weak_raw {m('SR_weak'):.2f}  strong_raw {m('SR_strong'):.2f}  weak+amp {m('SR_amp'):.2f}")
    print(f"summary -> {out_path}")


if __name__ == "__main__":
    main()
