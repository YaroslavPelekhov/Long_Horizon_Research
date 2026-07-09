"""
RASC — Reward-Aware Self-Configuring architecture.

The architectural novelty: most science-agent architectures are FIXED — the
same pipeline runs on every task. RASC instead DIAGNOSES, at the start of each
episode, what kind of answer the task rewards, and then RECONFIGURES its own
pipeline to match. It is one architecture that becomes a different agent per
task.

Motivation (established empirically across 5 scaffold mechanisms): the three
benchmarks reward three different things, and any fixed strategy is wrong on
two of them —

    reward type     winning strategy            scaffold that HURTS here
    -----------     ----------------            ------------------------
    FIT             numeric optimiser / SR       completeness gate, free-form
    REASONING       the model's own reasoning    fit-optimisation, gate (friction)
    COMPLETENESS    mechanical completeness      none (it's the whole point)

RASC diagnoses the reward type from (a) interface signals the adapter exposes
and (b) a cheap empirical fit-probe whose data is NOT wasted (it seeds the
real run), then dispatches to the matching configuration:

    COMPLETENESS  → Coordinator + Programmatic Completeness Gate (MARS-SELF)
    FIT           → Coordinator (plain) — the adapter's numeric machinery
                    + the probe data do the work
    REASONING     → Coordinator (plain) — no scaffold, let the model reason

The claim is not "beat every benchmark's ceiling" but: ONE self-configuring
architecture reaches the per-task-optimal configuration on every benchmark,
whereas any fixed architecture is sub-optimal on 2 of 3.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from mars.coordinator import Coordinator
from ols.core.abandon import FutilityDetector


@dataclass
class _SimpleReport:
    """Lightweight report when fit-mode is handled by the Investigator instead
    of the full Coordinator loop."""
    primary: float
    score_dict: dict
    n_turns: int = 0
    n_actions: int = 0


# ── empirical fit-probe ──────────────────────────────────────────────────────

def _probe_fit_r2(adapter, n_probe: int = 4) -> "float | None":
    """Run a few SRHP experiments and fit a separable power-law in log-space;
    return R² (the data persists in the adapter, so it is not wasted). None if
    the adapter doesn't support probing or there isn't enough signal."""
    try:
        import numpy as np
    except Exception:
        return None
    if not getattr(adapter, "supports_srhp", lambda: False)():
        return None
    try:
        cands = adapter.srhp_candidate_experiments(24)
    except Exception:
        return None
    if not cands:
        return None
    # spread probe points across the candidate pool
    step = max(1, len(cands) // n_probe)
    picks = cands[::step][:n_probe]
    inputs, outputs = [], []
    for exp in picks:
        if adapter.budget_left() <= 2:
            break
        try:
            y = adapter.srhp_run(exp)
            yv = float(y)
            if math.isfinite(yv):
                inputs.append(exp)
                outputs.append(yv)
        except Exception:
            continue
    if len(outputs) < 3:
        return None
    keys = list(inputs[0].keys())
    try:
        X = np.array([[float(d[k]) for k in keys] for d in inputs], float)
        y = np.array(outputs, float)
        m = (y > 0) & np.all(X > 0, axis=1)
        if m.sum() < 3:
            # try linear fit instead of log-log
            A = np.hstack([X, np.ones((len(X), 1))])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
        else:
            lX = np.log(X[m]); ly = np.log(y[m])
            A = np.hstack([lX, np.ones((m.sum(), 1))])
            coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
            pred = A @ coef
            ly_ = ly
            ss_res = float(((ly_ - pred) ** 2).sum())
            ss_tot = float(((ly_ - ly_.mean()) ** 2).sum())
            return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    except Exception:
        return None


# ── diagnosis ────────────────────────────────────────────────────────────────

def diagnose_reward_type(adapter, do_probe: bool = True) -> tuple[str, dict]:
    """Return ('completeness' | 'fit' | 'reasoning', signals).

    Decision order:
      1. rubric with ≥3 required sections → COMPLETENESS (multi-part artifact)
      2. supports SRHP + empirical fit-probe R² ≥ 0.6 → FIT (clean optimiser wins)
      3. exposes dataframes → REASONING (query-relevant reasoning; fit misleads)
      4. else → REASONING (safe default — minimal scaffold)
    """
    signals: dict = {}
    try:
        rubric = adapter.completeness_rubric()
    except Exception:
        rubric = []
    # count only keyword sections (multi-part artifact signal); predicate-only
    # rubrics (NewtonBench) are data-gathering hints, not artifact sections.
    kw_sections = [s for s in rubric if getattr(s, "keywords", None)]
    signals["n_rubric_keyword_sections"] = len(kw_sections)
    signals["n_rubric_sections"] = len(rubric)

    try:
        has_df = bool(adapter.get_dataframes())
    except Exception:
        has_df = False
    signals["has_dataframes"] = has_df

    supports_srhp = bool(getattr(adapter, "supports_srhp", lambda: False)())
    signals["supports_srhp"] = supports_srhp

    # 1. completeness — a multi-section required artifact
    if len(kw_sections) >= 3:
        signals["decision_reason"] = f"{len(kw_sections)} keyword artifact sections"
        return "completeness", signals

    # 2. fit — empirical probe
    fit_r2 = None
    if supports_srhp and do_probe:
        fit_r2 = _probe_fit_r2(adapter)
    signals["fit_probe_r2"] = fit_r2
    if fit_r2 is not None and fit_r2 >= 0.6:
        signals["decision_reason"] = f"fit-probe R2={fit_r2:.3f} ≥ 0.6"
        return "fit", signals
    if supports_srhp:
        # numeric interface but ambiguous probe → still fit (optimiser path)
        signals["decision_reason"] = "numeric experiment interface"
        return "fit", signals

    # 3. reasoning — tabular query task
    if has_df:
        signals["decision_reason"] = "tabular data + query (reasoning)"
        return "reasoning", signals

    signals["decision_reason"] = "default reasoning"
    return "reasoning", signals


# ── controller ───────────────────────────────────────────────────────────────

@dataclass
class RASCController:
    adapter: object
    generator: object
    reflector: object
    memory_selector: object
    code_evolver_factory: object = None   # callable() -> CodeEvolver (for completeness)
    max_turns_per_subgoal: int = 9
    max_total_turns: int = 40
    futility_theta: float = 2.0
    use_fit_investigator: bool = True     # fit-mode → AutonomousInvestigator
    investigator: object = None           # shared across episodes (memory)
    verbose: bool = False

    def run_episode(self):
        mode, signals = diagnose_reward_type(self.adapter)
        if self.verbose:
            print(f"[RASC] diagnosed reward-type = {mode}  ({signals.get('decision_reason')})")

        # FIT mode → hand off to the Autonomous Investigator (empirical escalation:
        # regression → sketch+snap → invented primitive), which is far stronger
        # than the agent-loop at clean numeric law recovery.
        if (mode == "fit" and self.use_fit_investigator
                and getattr(self.adapter, "supports_srhp", lambda: False)()):
            try:
                from mars.rasc.investigator import AutonomousInvestigator
                if self.investigator is None:
                    self.investigator = AutonomousInvestigator(
                        model=getattr(self.reflector, "model", "openai/gpt-4o-mini"))
                res = self.investigator.investigate(self.adapter)
                if res is not None:
                    self.adapter.srhp_finalize(res.law_code, "discovered_law")
                    score = self.adapter.score_episode([], final_artifact=None)
                    score = dict(score)
                    score["rasc_mode"] = mode
                    score["rasc_signals"] = signals
                    score["investigator_strategy"] = res.strategy
                    score["investigator_escalations"] = res.n_escalations
                    rep = _SimpleReport(primary=score.get("primary", 0.0),
                                        score_dict=score)
                    return rep, mode, signals
            except Exception:
                pass  # fall through to the Coordinator path

        use_ce = (mode == "completeness")
        evolver = None
        if use_ce and self.code_evolver_factory is not None:
            try:
                evolver = self.code_evolver_factory()
            except Exception:
                evolver = None

        coord = Coordinator(
            adapter=self.adapter,
            generator=self.generator,
            reflector=self.reflector,
            memory_selector=self.memory_selector,
            futility=FutilityDetector(theta_futile=self.futility_theta,
                                      min_absolute_spend=1e9),
            max_turns_per_subgoal=self.max_turns_per_subgoal,
            max_total_turns=self.max_total_turns,
            verbose=self.verbose,
            code_evolver=evolver,
            use_code_evolver=use_ce,
            use_reflector=True,
            use_memory_selector=True,
            use_futility_detector=(mode != "fit"),  # fit needs patient accumulation
        )
        rep = coord.run_episode()
        # attach diagnosis to the report
        try:
            rep.score_dict = dict(rep.score_dict)
            rep.score_dict["rasc_mode"] = mode
            rep.score_dict["rasc_signals"] = signals
        except Exception:
            pass
        return rep, mode, signals
