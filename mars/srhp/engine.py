"""
SRHP — Self-Refuting Hypothesis Programs.

The single unifying idea of MARS v0.4: a scientific hypothesis is an
EXECUTABLE PROGRAM that predicts observations from experiment inputs. The
engine runs Popper's conjecture-and-refutation as code:

    1. CONJECTURE   — the LLM writes a hypothesis-program h: inputs → prediction
    2. DISCRIMINATE — pick the experiment on which the current population of
                      hypotheses DISAGREES most (maximum-information, not
                      confirmation)
    3. REFUTE       — run that experiment; measure each hypothesis's prediction
                      error against the observation
    4. MUTATE       — drop the worst; if the best still mispredicts, ask the LLM
                      for a STRUCTURALLY DIFFERENT mechanism (new functional
                      form), seeded with where it failed most

The small model does ONLY the creative steps (conjecture, mutate). Everything
mechanical — choosing experiments, executing programs, scoring, selection — is
the deterministic exoskeleton. Completeness is intrinsic: a program that
predicts all observations must encode the full hidden rule (no partial reports).

Adapters expose the task via five hooks (see ResearchEnvAdapter.srhp_*):
    srhp_spec, srhp_candidate_experiments, srhp_run, srhp_error, srhp_finalize
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field

from mars.agents.base import call_llm, make_openai_client, parse_json_strict


# ── sandbox (allows math + numpy + statistics; blocks I/O / imports) ──────────

def _safe_ns() -> dict:
    ns: dict = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "filter": filter, "float": float,
        "int": int, "isinstance": isinstance, "len": len, "list": list,
        "map": map, "max": max, "min": min, "pow": pow, "range": range,
        "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
        "tuple": tuple, "zip": zip, "True": True, "False": False, "None": None,
        "__import__": None, "eval": None, "exec": None, "compile": None,
        "open": None, "input": None,
    }
    g: dict = {"__builtins__": ns, "math": math, "statistics": statistics}
    try:
        import numpy as np
        g["np"] = np
        g["numpy"] = np
    except Exception:
        pass
    return g


def _strip_fences(code: str) -> str:
    code = (code or "").strip()
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


def _compile_program(code: str, fn_name: str):
    """Exec code in a sandbox; return the callable or None."""
    g = _safe_ns()
    try:
        exec(compile(code, "<srhp>", "exec"), g)  # noqa: S102
        fn = g.get(fn_name)
        return fn if callable(fn) else None
    except Exception:
        return None


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class HypothesisProgram:
    code: str
    fn_name: str
    origin: str = "conjecture"          # or "mutation"
    errors: list[float] = field(default_factory=list)   # per-experiment error
    _fn: object = None
    alive: bool = True

    def compile(self) -> bool:
        self._fn = _compile_program(self.code, self.fn_name)
        return self._fn is not None

    def predict(self, inputs: dict):
        """Run the program on one input-dict; return its prediction or None."""
        if self._fn is None:
            return None
        try:
            return self._fn(**inputs)
        except Exception:
            try:
                # positional fallback
                return self._fn(*list(inputs.values()))
            except Exception:
                return None

    @property
    def mean_error(self) -> float:
        finite = [e for e in self.errors if math.isfinite(e)]
        if not finite:
            return float("inf")
        # penalise programs that errored on some experiments
        miss = len(self.errors) - len(finite)
        return (sum(finite) / len(finite)) + 0.5 * miss


@dataclass
class SRHPReport:
    primary: float
    score_dict: dict
    n_experiments: int
    n_conjectures: int
    n_mutations: int
    n_programs: int
    best_error: float
    best_code: str
    population_trace: list = field(default_factory=list)


# ── prompts ──────────────────────────────────────────────────────────────────

_CONJ_SYS = (
    "You are a scientific hypothesis generator. A hypothesis here is an "
    "EXECUTABLE Python function that PREDICTS the experiment's observed output "
    "from its inputs. Your function encodes a guess about the hidden rule.\n\n"
    "STRICT RULES:\n"
    "  1. Define EXACTLY the function signature given — same name, same params.\n"
    "  2. Return the prediction in the form described (a number, or a dict).\n"
    "  3. Use ONLY built-ins + `math`, `numpy as np`, `statistics` (all already "
    "in scope — DO NOT write any import statement).\n"
    "  4. NO imports, eval, exec, open, file/network I/O — they are BLOCKED.\n"
    "  5. Wrap risky math in try/except and return a sensible default on error.\n"
    "  6. The hidden rule may be NON-standard — do NOT assume textbook formulas. "
    "Fit the DATA you are given.\n"
    "  7. ALWAYS output a function. If data is sparse or empty, propose your "
    "single best INITIAL guess (e.g. a simple power law / linear form) — it will "
    "be refuted and refined. NEVER return an error, JSON, or explanation.\n"
    "Output ONLY raw Python source for the single function — no prose, no fences."
)

_MUT_SYS = (
    "You are a scientific hypothesis REVISER applying conjecture-and-refutation. "
    "The current best hypothesis-program still mispredicts the data. Propose a "
    "STRUCTURALLY DIFFERENT mechanism — a different FUNCTIONAL FORM (e.g. switch "
    "additive↔multiplicative, add a threshold/saturation, a trig or exponential "
    "term, an interaction) — NOT a small parameter tweak.\n\n"
    "Same strict rules: exact signature; only math/numpy/statistics in scope; no "
    "imports/eval/exec/open; return the described output; fit the DATA.\n"
    "Output ONLY raw Python source for the single revised function."
)


def _fmt_data(data: list[tuple[dict, object]], limit: int = 14) -> str:
    rows = []
    for inp, obs in data[-limit:]:
        try:
            rows.append(f"  {json.dumps(inp, default=str)[:160]} -> "
                        f"{json.dumps(obs, default=str)[:120]}")
        except Exception:
            rows.append(f"  {str(inp)[:160]} -> {str(obs)[:120]}")
    return "\n".join(rows) if rows else "  (no experiments yet)"


def _fmt_population(pop: list[HypothesisProgram], limit: int = 3) -> str:
    alive = [p for p in pop if p.alive]
    alive.sort(key=lambda p: p.mean_error)
    out = []
    for p in alive[:limit]:
        err = p.mean_error
        es = f"{err:.4f}" if math.isfinite(err) else "inf"
        out.append(f"  [mean_err={es}] {p.code.strip()[:240]}")
    return "\n".join(out) if out else "  (none yet)"


# ── engine ───────────────────────────────────────────────────────────────────

@dataclass
class SRHPEngine:
    adapter: object
    model: str = "openai/gpt-4o-mini"
    min_population: int = 2
    max_population: int = 4
    candidate_pool: int = 16
    error_tol: float = 0.05            # "good enough" mean error → stop mutating
    n_seed_experiments: int = 3        # bootstrap data before first conjecture
    max_tokens: int = 700
    verbose: bool = False

    def __post_init__(self):
        self._client = make_openai_client()
        self.n_conjectures = 0
        self.n_mutations = 0

    # -- LLM steps ----

    def _gen_program(self, spec: dict, data: list, population: list,
                     mutate_from: "HypothesisProgram | None" = None) -> "HypothesisProgram | None":
        fn_name = spec.get("fn_name", "predict")
        sig = spec.get("signature", f"def {fn_name}(...):")
        desc = spec.get("description", "")
        out_desc = spec.get("output_desc", "a numeric prediction")

        base = (
            f"TASK:\n{desc[:1400]}\n\n"
            f"FUNCTION SIGNATURE (define exactly this):\n  {sig}\n"
            f"It must return: {out_desc}\n\n"
            f"ACCUMULATED EXPERIMENTS (inputs -> observed):\n{_fmt_data(data)}\n\n"
            f"CURRENT HYPOTHESES (code + mean error, lower better):\n"
            f"{_fmt_population(population)}\n"
        )
        if mutate_from is not None:
            # find this program's worst experiment for a targeted refutation hint
            worst = None
            worst_e = -1.0
            for (inp, obs) in data:
                pred = mutate_from.predict(inp)
                e = self.adapter.srhp_error(pred, obs)
                if math.isfinite(e) and e > worst_e:
                    worst_e, worst = e, (inp, obs, pred)
            hint = ""
            if worst is not None:
                hint = (f"\nWORST MISPREDICTION of the current best:\n"
                        f"  inputs={json.dumps(worst[0], default=str)[:150]}\n"
                        f"  observed={json.dumps(worst[1], default=str)[:100]}  "
                        f"predicted={json.dumps(worst[2], default=str)[:100]}\n"
                        f"Change the FORM so this case is explained.")
            user = base + hint + "\n\nWrite the structurally-different function now:"
            sysp = _MUT_SYS
        else:
            user = base + "\n\nWrite the hypothesis function now:"
            sysp = _CONJ_SYS

        try:
            raw = call_llm(self._client, self.model, sysp, user,
                           max_tokens=self.max_tokens, temperature=0.5)
        except Exception:
            return None
        code = _strip_fences(raw)
        # salvage a function definition embedded in prose/JSON
        if code and f"def {fn_name}" not in code:
            m = re.search(rf"(def\s+{re.escape(fn_name)}\b.*)", code, flags=re.DOTALL)
            if m:
                code = m.group(1)
        if not code or f"def {fn_name}" not in code:
            return None
        prog = HypothesisProgram(
            code=code, fn_name=fn_name,
            origin=("mutation" if mutate_from is not None else "conjecture"),
        )
        if not prog.compile():
            return None
        # must produce a finite numeric prediction on at least one input
        if data:
            ok = False
            for inp, _ in data[:4]:
                v = _scalarize(prog.predict(inp))
                if v is not None and math.isfinite(v):
                    ok = True
                    break
            if not ok:
                return None
        if mutate_from is not None:
            self.n_mutations += 1
        else:
            self.n_conjectures += 1
        return prog

    def _gen_retry(self, spec, data, population, mutate_from=None, tries=3):
        for _ in range(tries):
            p = self._gen_program(spec, data, population, mutate_from=mutate_from)
            if p is not None:
                return p
        return None

    # -- mechanical steps ----

    def _discriminating(self, candidates: list[dict],
                        population: list, tried: set) -> dict | None:
        """Pick the candidate experiment maximising disagreement (spread of
        predictions) across the living population. Falls back to first untried."""
        alive = [p for p in population if p.alive and p._fn is not None]
        best_exp = None
        best_spread = -1.0
        for exp in candidates:
            key = json.dumps(exp, sort_keys=True, default=str)
            if key in tried:
                continue
            preds = []
            for p in alive:
                pr = p.predict(exp)
                v = _scalarize(pr)
                if v is not None and math.isfinite(v):
                    preds.append(v)
            if len(preds) >= 2:
                # coefficient-of-variation style spread (scale-free)
                m = statistics.fmean(preds)
                sd = statistics.pstdev(preds)
                spread = sd / (abs(m) + 1e-9)
            else:
                spread = 0.0
            if best_exp is None or spread > best_spread:
                best_spread, best_exp = spread, exp
        if best_exp is None:
            for exp in candidates:
                key = json.dumps(exp, sort_keys=True, default=str)
                if key not in tried:
                    return exp
        return best_exp

    def _prune(self, population: list):
        alive = [p for p in population if p.alive]
        if len(alive) <= self.max_population:
            return
        alive.sort(key=lambda p: p.mean_error)
        for p in alive[self.max_population:]:
            p.alive = False

    # -- main loop ----

    def run(self) -> SRHPReport:
        adapter = self.adapter
        spec = adapter.srhp_spec()
        candidates = list(adapter.srhp_candidate_experiments(self.candidate_pool))
        population: list[HypothesisProgram] = []
        data: list[tuple[dict, object]] = []
        tried: set = set()
        trace: list = []

        # SEED phase: run a few diverse experiments so the model has data to
        # form its first conjecture (models refuse to hypothesise from nothing).
        n_seed = min(self.n_seed_experiments, max(0, int(adapter.budget_left()) - 1))
        seed_cands = candidates[:max(1, len(candidates))]
        # spread the seeds across the candidate pool for diversity
        if seed_cands and n_seed > 0:
            step = max(1, len(seed_cands) // n_seed)
            picks = seed_cands[::step][:n_seed]
            for exp in picks:
                if adapter.budget_left() <= 1:
                    break
                try:
                    observed = adapter.srhp_run(exp)
                except Exception:
                    break
                data.append((exp, observed))
                tried.add(json.dumps(exp, sort_keys=True, default=str))

        # first conjecture, now grounded in seed data
        p0 = self._gen_retry(spec, data, population)
        if p0 is not None:
            for inp, obs in data:
                p0.errors.append(adapter.srhp_error(p0.predict(inp), obs))
            population.append(p0)

        while adapter.budget_left() > 0:
            # 1. ensure enough living hypotheses to discriminate
            alive = [p for p in population if p.alive]
            if len(alive) < self.min_population:
                pn = self._gen_retry(spec, data, population)
                if pn is not None:
                    population.append(pn)
                    # backfill its errors on existing data
                    for inp, obs in data:
                        pn.errors.append(adapter.srhp_error(pn.predict(inp), obs))

            # 2. discriminating experiment
            if not candidates:
                candidates = list(adapter.srhp_candidate_experiments(self.candidate_pool))
            exp = self._discriminating(candidates, population, tried)
            if exp is None:
                # refresh pool; if still none, stop
                candidates = list(adapter.srhp_candidate_experiments(self.candidate_pool))
                exp = self._discriminating(candidates, population, tried)
                if exp is None:
                    break
            tried.add(json.dumps(exp, sort_keys=True, default=str))

            # 3. refute — run real experiment
            if adapter.budget_left() <= 0:
                break
            try:
                observed = adapter.srhp_run(exp)
            except Exception:
                break
            data.append((exp, observed))

            # 4. score every living program on the new observation
            for p in population:
                if p.alive:
                    p.errors.append(adapter.srhp_error(p.predict(exp), observed))

            # 5. prune + mutate-on-surprise
            self._prune(population)
            living = [p for p in population if p.alive]
            if living:
                best = min(living, key=lambda p: p.mean_error)
                if best.mean_error > self.error_tol and adapter.budget_left() > 0:
                    mut = self._gen_retry(spec, data, population, mutate_from=best, tries=2)
                    if mut is not None:
                        for inp, obs in data:
                            mut.errors.append(adapter.srhp_error(mut.predict(inp), obs))
                        population.append(mut)
                        self._prune(population)

            if self.verbose:
                living = [p for p in population if p.alive]
                be = min((p.mean_error for p in living), default=float("inf"))
                print(f"[SRHP] exp={len(data)} pop={len(living)} "
                      f"best_err={be:.4f} budget={adapter.budget_left():.0f}")
            trace.append({
                "n_exp": len(data),
                "best_err": min((p.mean_error for p in population if p.alive),
                                default=float("inf")),
                "pop": len([p for p in population if p.alive]),
            })

        # finalize: narrate / submit the best program
        living = [p for p in population if p.alive] or population
        best = min(living, key=lambda p: p.mean_error) if living else None
        best_code = best.code if best else ""
        best_err = best.mean_error if best else float("inf")
        try:
            adapter.srhp_finalize(best_code, spec.get("fn_name", "predict"))
        except Exception:
            pass
        score = adapter.score_episode([], final_artifact=None)
        primary = score.get("primary", 0.0)

        return SRHPReport(
            primary=primary,
            score_dict=score,
            n_experiments=len(data),
            n_conjectures=self.n_conjectures,
            n_mutations=self.n_mutations,
            n_programs=len(population),
            best_error=best_err,
            best_code=best_code[:600],
            population_trace=trace,
        )


def _scalarize(pred) -> "float | None":
    """Reduce a prediction (number or dict of numbers) to one scalar for the
    discrimination spread computation."""
    if pred is None:
        return None
    try:
        if isinstance(pred, dict):
            vals = [float(v) for v in pred.values()
                    if isinstance(v, (int, float))]
            return statistics.fmean(vals) if vals else None
        return float(pred)
    except Exception:
        return None
