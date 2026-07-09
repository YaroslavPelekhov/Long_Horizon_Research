"""DPSR — Bayesian Typed-Hole Hypothesis Induction.

A pre-proposal induction layer for UniversalCPI. The novelty is not that we
verify hypotheses, but that we change the UNIT OF GENERATION:

    not:  weak LLM -> a complete hypothesis (p ~ 0 for hard cases)
    but:  weak LLM -> a hypothesis SKELETON with typed holes
          engine   -> infers each hole's value by executable micro-measurements
          engine   -> recombine + verify; on failure, blame and reopen holes

Formally, a skeleton is a program with typed holes  θ = {θ1..θk}  exposed as a
top-level dict `H`. The engine maintains a posterior  p(θ_i | D)  where the
likelihood of a candidate value is exp(-loss) measured by EXECUTING the
instantiated program on observations (via the adapter). Holes are closed in
order of uncertainty × downstream impact; a closed skeleton is recombined and
verified; blame on failure reopens only the guilty hole.

Fully universal: the engine only uses the CPIAdapter contract
(signature_hint / interface_description / sandbox_globals / execute / loss /
calibrate). The skeleton and per-hole probes are PROPOSED by the LLM from the
generic interface; the hole VALUES are inferred from data, never guessed.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from mars.agents.base import call_llm
from mars.induction.universal_cpi import (
    CPIAdapter,
    HypothesisProgram,
    Observation,
    _ast_safe,
    _SAFE_BUILTINS,
)


# ===========================================================================
# Entities
# ===========================================================================

@dataclass
class HolePosterior:
    """Distribution over candidate values for one typed hole."""
    probs: dict[Any, float] = field(default_factory=dict)

    def entropy(self) -> float:
        if not self.probs:
            return 1.0
        h = -sum(p * math.log(p + 1e-12) for p in self.probs.values() if p > 0)
        # normalize by max entropy (uniform over the candidates)
        hmax = math.log(len(self.probs)) if len(self.probs) > 1 else 1.0
        return h / hmax if hmax > 0 else 0.0

    def argmax(self) -> Any:
        return max(self.probs.items(), key=lambda kv: kv[1])[0] if self.probs else None

    def peak(self) -> float:
        return max(self.probs.values()) if self.probs else 0.0


@dataclass
class TypedHole:
    name: str
    htype: str                       # "numeric" | "keys" | "category" | "value"
    candidates: list[Any] = field(default_factory=list)
    posterior: HolePosterior = field(default_factory=HolePosterior)
    frozen: bool = False
    probe_code: str = ""             # optional LLM-proposed measurement program


@dataclass
class HypothesisSkeleton:
    code: str                        # source defining H = {...} + the entry fn
    entry: str                       # entry function name (from signature_hint)
    holes: dict[str, TypedHole]
    default_H: dict[str, Any]


@dataclass
class DPSRTrace:
    steps: list[dict] = field(default_factory=list)


@dataclass
class DPSRResult:
    programs: list[HypothesisProgram]
    trace: DPSRTrace
    errors: list[str]


# ===========================================================================
# Skeleton compilation (compile once, vary H between evaluations)
# ===========================================================================

def _compile_skeleton(code: str, entry: str, sg: dict[str, Any] | None):
    """Compile skeleton; return (fn, namespace) so H can be mutated per eval."""
    ok, err = _ast_safe(code)
    if not ok:
        return None, None, err
    ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    import itertools as _itertools
    import math as _math
    import statistics as _stats
    ns["math"], ns["statistics"], ns["itertools"] = _math, _stats, _itertools
    if sg:
        ns.update(sg)
    try:
        exec(compile(code, "<dpsr_skeleton>", "exec"), ns)
    except Exception as e:
        return None, None, f"ExecError: {e}"
    fn = ns.get(entry)
    if not callable(fn):
        return None, None, f"entry '{entry}' not found"
    if not isinstance(ns.get("H"), dict):
        return None, None, "skeleton has no top-level dict H of holes"
    return fn, ns, ""


def _entry_name(signature_hint: str) -> str:
    """Extract the function name from a signature hint like 'def law(inputs):'."""
    m = signature_hint.strip()
    if m.startswith("def "):
        return m[4:].split("(")[0].strip()
    return "f"


def _entry_and_args(signature_hint: str) -> tuple[str, list[str]]:
    """('def predict(p1, p2, context) -> dict') -> ('predict', ['p1','p2','context'])."""
    entry = _entry_name(signature_hint)
    s = signature_hint
    try:
        inside = s[s.index("(") + 1: s.index(")")]
    except ValueError:
        return entry, ["context"]
    args = []
    for part in inside.split(","):
        nm = part.split(":")[0].split("=")[0].strip()
        if nm and nm not in ("self", "*", "/"):
            args.append(nm)
    return entry, args or ["context"]


def _func_name(code: str) -> str | None:
    """First top-level function name in a block of source."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


# ===========================================================================
# Engine
# ===========================================================================

class DPSREngine:
    def __init__(self, *, model: str, client, temperature: float = 0.5,
                 max_sweeps: int = 3, max_candidates: int = 18, tau: float = 0.25,
                 n_skeletons: int = 3, building_blocks: list[dict] | None = None):
        self.model = model
        self.client = client
        self.temperature = temperature
        self.max_sweeps = max_sweeps
        self.max_candidates = max_candidates
        self.tau = tau                 # posterior temperature: p ∝ exp(-loss/tau)
        self.n_skeletons = n_skeletons
        # CDA × DPSR: certified building blocks (e.g. drilled prim_* programs)
        # composed into a coefficient-hole skeleton. Structure from the verified
        # library; coefficients (subset selection) inferred by measurement.
        self.building_blocks = building_blocks or []

    # ------------------------------------------------------------------
    def run(self, adapter: CPIAdapter, train_obs: list[Observation]) -> DPSRResult:
        trace = DPSRTrace()
        errors: list[str] = []
        if not train_obs:
            return DPSRResult([], trace, ["no train observations"])

        skeletons: list[HypothesisSkeleton] = []
        block_skel = self._block_skeleton(adapter)        # CDA × DPSR
        if block_skel is not None:
            skeletons.append(block_skel)
        skeletons.extend(self._propose_skeletons(adapter, train_obs))
        if not skeletons:
            return DPSRResult([], trace, ["DPSR: no valid skeleton proposed"])

        sg = adapter.sandbox_globals()
        # Close holes on EACH candidate skeleton; keep the best by MDL
        # (train loss first, then fewer holes = simpler). A weak LLM tends to
        # over-propose a kitchen-sink skeleton; competing simpler shapes win
        # when they fit, which is the Occam pressure we want.
        scored: list[tuple[float, int, HypothesisSkeleton, dict]] = []
        for skel in skeletons:
            closed = self._close_skeleton(skel, adapter, train_obs, sg, trace)
            if closed is None:
                continue
            final_H, final_loss = closed
            scored.append((final_loss, len(skel.holes), skel, final_H))
        if not scored:
            return DPSRResult([], trace, ["DPSR: no skeleton compiled/closed"])
        scored.sort(key=lambda t: (round(t[0], 6), t[1]))
        best_loss, _, best_skel, best_H = scored[0]
        trace.steps.append({"selected_skeleton": best_skel.entry,
                            "train_loss": round(best_loss, 4),
                            "n_skeletons": len(scored)})

        programs: list[HypothesisProgram] = []
        for tag, Hval in self._final_candidates(best_skel, best_H):
            prog = self._instantiate(best_skel, Hval, adapter, train_obs, sg, tag)
            if prog is not None:
                programs.append(prog)
        if not programs:
            errors.append("DPSR: recombined program failed to compile")
        return DPSRResult(programs, trace, errors)

    def _close_skeleton(self, skel: HypothesisSkeleton, adapter: CPIAdapter,
                        train_obs: list[Observation], sg: dict,
                        trace: DPSRTrace) -> tuple[dict, float] | None:
        """Infer all hole values for one skeleton by executable measurement."""
        fn, ns, cerr = _compile_skeleton(skel.code, skel.entry, sg)
        if fn is None:
            trace.steps.append({"skeleton_error": cerr[:120]})
            return None
        for hole in skel.holes.values():
            hole.frozen = False
            hole.posterior = HolePosterior()
            hole.candidates = self._candidates_for_hole(hole, train_obs, sg)[: self.max_candidates]
            dv = skel.default_H.get(hole.name)
            if dv is not None and dv not in hole.candidates:
                hole.candidates.append(dv)

        cur_H = dict(skel.default_H)
        for sweep in range(self.max_sweeps):
            for hname in self._hole_order(skel, cur_H, fn, ns, adapter, train_obs):
                hole = skel.holes[hname]
                if hole.frozen:
                    continue
                post, best_val, best_loss, joint_H = self._measure_hole(
                    hole, cur_H, fn, ns, adapter, train_obs, skel=skel)
                hole.posterior = post
                cur_H = joint_H
                if post.peak() >= 0.85:
                    hole.frozen = True
            if all(h.frozen for h in skel.holes.values()):
                break

        final_H = cur_H
        final_loss = self._eval_H(final_H, fn, ns, adapter, train_obs)
        if final_loss > 1e-9:                      # blame → reopen the guilty hole
            blamed = self._blame(skel, final_H, fn, ns, adapter, train_obs)
            if blamed is not None:
                skel.holes[blamed].frozen = False
                _, _, best_loss, joint_H = self._measure_hole(
                    skel.holes[blamed], final_H, fn, ns, adapter, train_obs,
                    skel=skel, widen=True)
                if best_loss < final_loss:
                    final_H, final_loss = joint_H, best_loss
        return final_H, final_loss

    # ------------------------------------------------------------------
    # Skeleton proposal (LLM)
    # ------------------------------------------------------------------
    def _block_skeleton(self, adapter: CPIAdapter) -> HypothesisSkeleton | None:
        """CDA × DPSR: compose certified building blocks into a coefficient-hole
        skeleton  f(args) = sum( H[c_i] * block_i(args) ). The structure comes
        from the verified library; DPSR infers WHICH blocks (coefficient 0/1) and
        their weights by measurement. Universal: blocks share the task signature."""
        if not self.building_blocks:
            return None
        entry, args = _entry_and_args(adapter.signature_hint())
        argcall = ", ".join(args) if args else ""
        block_srcs: list[str] = []
        lines: list[str] = []
        holes: dict[str, TypedHole] = {}
        default_H: dict[str, Any] = {}
        seen: set[str] = set()
        for b in self.building_blocks:
            code = str(b.get("code", "")).strip()
            fname = _func_name(code)
            if not fname or fname in seen:
                continue
            seen.add(fname)
            block_srcs.append(code)
            cname = f"c_{fname}"
            lines.append(f"    total = total + H[{cname!r}] * ({fname}({argcall}))")
            holes[cname] = TypedHole(name=cname, htype="coef")
            default_H[cname] = 1
        if not lines:
            return None
        code = (
            "\n\n".join(block_srcs)
            + "\n\n"
            + f"H = {default_H!r}\n"
            + f"def {entry}({', '.join(args)}):\n    total = 0\n"
            + "\n".join(lines)
            + "\n    return total\n"
        )
        return HypothesisSkeleton(code=code, entry=entry, holes=holes, default_H=default_H)

    def _propose_skeletons(self, adapter: CPIAdapter,
                           train_obs: list[Observation]) -> list[HypothesisSkeleton]:
        entry = _entry_name(adapter.signature_hint())
        examples = [{
            "inputs": _short(o.inputs, 240),
            "context": {k: _short(v, 120) for k, v in o.context.items() if not k.startswith("__")},
            "target": _short(o.target, 160),
        } for o in train_obs[:8]]
        n = int(self.n_skeletons)
        prompt = f"""{adapter.interface_description()}

FUNCTION SIGNATURE (must match exactly):
{adapter.signature_hint()}

OBSERVED EVIDENCE:
{json.dumps(examples, indent=2, ensure_ascii=False, default=str)[:3000]}

Do NOT give the final answer. Write {n} DIVERSE HYPOTHESIS SKELETONS: the SHAPE
of the rule with its unknown values left as typed HOLES. Each skeleton exposes
every hole in a single top-level dict named H, and the function reads values
from H. The system will INFER the hole values from data by measurement, then
keep whichever skeleton fits best with the FEWEST holes — so:

- make the FIRST skeleton the SIMPLEST plausible shape (few holes),
- make later ones progressively richer (e.g. a SUM of several effects),
- do NOT add an effect unless the evidence suggests it; spurious terms lose.

Example shape (illustrative only):
H = {{"var": "energy", "threshold": 15, "A": 1, "B": 0}}
def {entry}(...):
    ...
    return H["A"] if (... >= H["threshold"]) else H["B"]

Hole types: "keys" (name of an input/context field), "numeric" (a constant:
threshold, coefficient, exponent), "category" (an observed discrete output
value), "value" (other literal).

Optionally give a PROBE per hole: def probe(observations) returning a SHORT list
of candidate values measured from data. observations is a list of dicts with
keys 'inputs','context','target'.

Return ONLY JSON:
{{
  "skeletons": [
    {{
      "skeleton": "H = {{...}}\\n{adapter.signature_hint().split('->')[0].strip()}...:\\n    ...",
      "entry": "{entry}",
      "holes": [{{"name": "...", "type": "keys|numeric|category|value"}}],
      "probes": {{"hole_name": "def probe(observations):\\n    return [...]"}}
    }}
  ]
}}"""
        try:
            raw = call_llm(self.client, model=self.model,
                           system="You design typed-hole hypothesis skeletons. Return only valid JSON.",
                           user=prompt, max_tokens=3200, temperature=self.temperature)
        except Exception:
            return []
        obj = _parse_json(raw)
        items = []
        if isinstance(obj, dict):
            items = obj.get("skeletons") or ([obj] if obj.get("skeleton") else [])
        elif isinstance(obj, list):
            items = obj
        out: list[HypothesisSkeleton] = []
        for it in items[: n]:
            skel = self._parse_skeleton(it, entry)
            if skel is not None:
                out.append(skel)
        return out

    def _parse_skeleton(self, obj: Any, entry: str) -> HypothesisSkeleton | None:
        if not isinstance(obj, dict):
            return None
        code = str(obj.get("skeleton", "")).strip()
        entry = str(obj.get("entry", entry)).strip() or entry
        if not code or "H" not in code:
            return None
        default_H = _extract_H(code)
        if not default_H:
            return None
        holes: dict[str, TypedHole] = {}
        for h in obj.get("holes", []):
            if not isinstance(h, dict):
                continue
            name = str(h.get("name", "")).strip()
            if name and name in default_H:
                holes[name] = TypedHole(name=name, htype=str(h.get("type", "value")))
        for k in default_H:
            if k not in holes:
                holes[k] = TypedHole(name=k, htype=_infer_type(default_H[k]))
        probes = obj.get("probes", {}) or {}
        for name, code_str in probes.items():
            if name in holes and isinstance(code_str, str):
                holes[name].probe_code = code_str.strip()
        return HypothesisSkeleton(code=code, entry=entry, holes=holes, default_H=default_H)

    # ------------------------------------------------------------------
    # Candidate generation: probe (LLM) → else generic from observations
    # ------------------------------------------------------------------
    def _candidates_for_hole(self, hole: TypedHole, train_obs: list[Observation],
                             sg: dict[str, Any]) -> list[Any]:
        if hole.htype == "coef":
            return [0, 1, 2, -1, 3]          # subset-selection / small weights
        cands: list[Any] = []
        if hole.probe_code:
            cands = self._run_probe(hole.probe_code, train_obs, sg)
        if cands:
            return _dedup(cands)
        # Generic fallback by type.
        if hole.htype == "keys":
            keys = set()
            for o in train_obs:
                if isinstance(o.inputs, dict):
                    keys.update(o.inputs.keys())
                keys.update(k for k in o.context if not k.startswith("__"))
            return list(keys)
        if hole.htype == "category":
            vals = []
            for o in train_obs:
                t = o.target
                if isinstance(t, dict):
                    vals.extend(str(v) for v in t.values())
                else:
                    vals.append(t)
            return _dedup(vals)
        if hole.htype == "numeric":
            targets, inputs = [], []
            for o in train_obs:
                if isinstance(o.target, (int, float)) and not isinstance(o.target, bool):
                    targets.append(float(o.target))
                src = o.inputs if isinstance(o.inputs, dict) else {}
                for v in list(src.values()) + list(o.context.values()):
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        inputs.append(float(v))
            tvals = sorted(set(targets))                 # output-constant holes (A,B)
            ivals = sorted(set(inputs))                  # threshold/breakpoint holes
            # midpoints between consecutive input values = natural thresholds
            mids = [(ivals[i] + ivals[i + 1]) / 2 for i in range(len(ivals) - 1)]
            # small ints/halves cover exponents/powers/small constants; midpoints
            # cover thresholds; targets cover larger output constants. Priority
            # order so the [:max_candidates] cap keeps coverage of all kinds.
            small = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
            return _dedup(small + tvals + mids + ivals)
        return []

    def _run_probe(self, code: str, train_obs: list[Observation],
                   sg: dict[str, Any]) -> list[Any]:
        ok, _ = _ast_safe(code)
        if not ok:
            return []
        ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
        import math as _math
        import statistics as _stats
        ns["math"], ns["statistics"] = _math, _stats
        if sg:
            ns.update(sg)
        try:
            exec(compile(code, "<dpsr_probe>", "exec"), ns)
            probe = ns.get("probe")
            if not callable(probe):
                return []
            rows = [{"inputs": o.inputs, "context": o.context, "target": o.target}
                    for o in train_obs]
            out = probe(rows)
            return list(out) if isinstance(out, (list, tuple)) else []
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Measurement: posterior over a hole by executed loss
    # ------------------------------------------------------------------
    def _measure_hole(self, hole: TypedHole, cur_H: dict, fn, ns, adapter,
                      train_obs, skel=None, widen: bool = False):
        """Posterior over a hole by executed loss. For each candidate value we
        GREEDILY REFIT the dependent (non-frozen) holes — this resolves the
        chicken-and-egg between structural holes (var/threshold) and output
        holes (A/B), per the dependency-graph design. Returns the best JOINT H."""
        cands = hole.candidates
        if widen and len(cands) < self.max_candidates:
            cands = _dedup(cands + self._candidates_for_hole(hole, train_obs,
                                                             adapter.sandbox_globals()))
        losses: dict[Any, float] = {}
        best_joint: dict | None = None
        best_loss = 2.0
        for v in cands[: self.max_candidates]:
            H = dict(cur_H)
            H[hole.name] = v
            if skel is not None:
                H = self._greedy_refit(skel, H, fix={hole.name}, fn=fn, ns=ns,
                                       adapter=adapter, train_obs=train_obs)
            loss = self._eval_H(H, fn, ns, adapter, train_obs)
            losses[v] = loss
            if loss < best_loss:
                best_loss, best_joint = loss, H
        if not losses:
            return HolePosterior(), None, 1.0, dict(cur_H)
        weights = {v: math.exp(-l / max(self.tau, 1e-6)) for v, l in losses.items()}
        z = sum(weights.values()) or 1.0
        post = HolePosterior({v: w / z for v, w in weights.items()})
        best_val = min(losses.items(), key=lambda kv: kv[1])[0]
        return post, best_val, losses[best_val], (best_joint or dict(cur_H))

    def _greedy_refit(self, skel, H, *, fix, fn, ns, adapter, train_obs,
                      passes: int = 2) -> dict:
        """Set each non-fixed, non-frozen hole to its loss-min candidate. Several
        passes let interacting additive holes co-adapt (needed for sum-shaped
        skeletons where structural and output holes depend on each other)."""
        H = dict(H)
        for _ in range(passes):
            changed = False
            for name, hole in skel.holes.items():
                if name in fix or hole.frozen or not hole.candidates:
                    continue
                best_v, best_l = H.get(name), 2.0
                for v in hole.candidates[: self.max_candidates]:
                    cand = dict(H)
                    cand[name] = v
                    l = self._eval_H(cand, fn, ns, adapter, train_obs)
                    if l < best_l:
                        best_l, best_v = l, v
                if best_v != H.get(name):
                    changed = True
                H[name] = best_v
            if not changed:
                break
        return H

    def _hole_order(self, skel, cur_H, fn, ns, adapter, train_obs) -> list[str]:
        """Order holes by uncertainty × downstream impact (info-gain heuristic)."""
        scored = []
        for name, hole in skel.holes.items():
            if hole.frozen:
                continue
            unc = hole.posterior.entropy() if hole.posterior.probs else 1.0
            impact = max(1, len(hole.candidates))
            scored.append((unc * math.log(impact + 1), name))
        scored.sort(reverse=True)
        return [n for _, n in scored]

    # ------------------------------------------------------------------
    # Evaluation / recombination / blame
    # ------------------------------------------------------------------
    def _eval_H(self, H: dict, fn, ns, adapter, train_obs) -> float:
        ns["H"] = H
        prog = _FnProg(fn)
        # Calibrate free constants (e.g. a law's multiplicative constant) for THIS
        # hole assignment before scoring — otherwise structures that differ only by
        # a constant are indistinguishable and exponent holes can't be measured.
        try:
            prog = adapter.calibrate(prog, train_obs)
        except Exception:
            pass
        total, n = 0.0, 0
        for o in train_obs:
            try:
                pred = adapter.execute(prog, o)
                total += float(adapter.loss(pred, o))
            except Exception:
                total += 1.0
            n += 1
        return total / n if n else 1.0

    def _blame(self, skel, final_H, fn, ns, adapter, train_obs) -> str | None:
        """Hole whose perturbation most changes residual = most likely guilty."""
        base = self._eval_H(final_H, fn, ns, adapter, train_obs)
        best_name, best_delta = None, -1.0
        for name, hole in skel.holes.items():
            alt = [c for c in hole.candidates if c != final_H.get(name)]
            if not alt:
                continue
            H = dict(final_H)
            H[name] = alt[0]
            delta = abs(self._eval_H(H, fn, ns, adapter, train_obs) - base)
            if delta > best_delta:
                best_delta, best_name = delta, name
        return best_name

    def _final_candidates(self, skel, final_H):
        yield "dpsr_inferred", dict(final_H)
        # runner-up: flip the highest-entropy hole to its 2nd-best value
        worst = max(skel.holes.values(),
                    key=lambda h: h.posterior.entropy() if h.posterior.probs else 0.0,
                    default=None)
        if worst and worst.posterior.probs and len(worst.posterior.probs) > 1:
            ranked = sorted(worst.posterior.probs.items(), key=lambda kv: -kv[1])
            second = ranked[1][0]
            if second != final_H.get(worst.name):
                alt = dict(final_H)
                alt[worst.name] = second
                yield "dpsr_runnerup", alt

    def _instantiate(self, skel, H, adapter, train_obs, sg, tag) -> HypothesisProgram | None:
        code = _set_H(skel.code, H)
        if code is None:
            return None
        fn, ns, err = _compile_skeleton(code, skel.entry, sg)
        if fn is None:
            return None
        prog = HypothesisProgram(
            name=f"{tag}_{skel.entry}",
            description=f"DPSR typed-hole induction; holes inferred by measurement: {_short(H, 160)}",
            code=code,
            fn=fn,
            complexity=1.4,
            tags=("dpsr", tag),
        )
        try:
            return adapter.calibrate(prog, train_obs)
        except Exception:
            return prog


# ===========================================================================
# Helpers
# ===========================================================================

class _FnProg:
    """Minimal shim so adapter.execute (which expects program.fn) can run a raw fn."""
    def __init__(self, fn):
        self.fn = fn
        self._const = 1.0


def _extract_H(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "H":
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        return {}
    return {}


def _set_H(code: str, H: dict) -> str | None:
    """Replace the top-level H = {...} assignment with the inferred values."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    new_val = ast.parse(repr(H), mode="eval").body
    found = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "H":
                    node.value = new_val
                    found = True
    if not found:
        return None
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return None


def _infer_type(v: Any) -> str:
    if isinstance(v, bool):
        return "value"
    if isinstance(v, (int, float)):
        return "numeric"
    if isinstance(v, str):
        return "value"
    return "value"


def _dedup(xs: list[Any]) -> list[Any]:
    out, seen = [], set()
    for x in xs:
        # Only keep hashable scalar-ish candidates; a hole value must be a
        # literal we can place into H (str/num/bool/None). Skip the rest.
        if not isinstance(x, (str, int, float, bool)) and x is not None:
            continue
        k = (type(x).__name__, x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _short(o: Any, n: int = 100) -> Any:
    if isinstance(o, (int, float, bool)) or o is None:
        return o
    s = o if isinstance(o, str) else json.dumps(o, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


def _parse_json(raw: str) -> Any:
    if not raw:
        return None
    t = raw.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).lstrip()
    try:
        return json.loads(t)
    except Exception:
        # last resort: grab the outermost {...}
        i, j = t.find("{"), t.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                return None
    return None
