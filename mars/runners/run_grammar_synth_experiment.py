"""Grammar synthesis experiment: automated vs hand-seeded CPI on UH-Seq.

Experimental design
-------------------
This is the key experiment for the SB-CPI novelty claim.

  Run A — synthesized grammar (zero human-written primitives):
    1. Collect N_INIT observations from the environment.
    2. Build interface description from the observable schema only.
    3. Ask LLM to propose candidate primitive functions (no examples of correct rules).
    4. Sandbox-validate proposals → ProgramHypothesis list.
    5. Run CPI (MDL + refutation) on ALL observations → winner per rule slot.

  Run B — hand-seeded grammar (control, current baseline):
    Same observations, per-slot hand-designed library, same CPI scorer.

The comparison answers:
  "Can a system with zero hand-designed primitives identify the same mechanisms
   as one with a curated library, given only the observable interface?"

If synthesized grammar matches (loss=0.0) on K/5 rules vs control 5/5,
this is the empirical evidence for the self-building claim.

Usage
-----
python -m mars.runners.run_grammar_synth_experiment \\
    --run_id grammar_synth_s42_easy \\
    --seed 42 --difficulty easy --n_init 2 --steps 5 \\
    --synth_model openai/gpt-4o-mini --n_proposals 16 --overwrite
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
_UH_REPO = _PROJ / "ultrahorizon_repo"
for _p in (_PROJ, _UH_REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv
    _env = _PROJ / "autodiscovery" / ".env.local"
    if _env.exists():
        load_dotenv(_env, override=False)
except ImportError:
    pass

from mars.induction.cpi import rank_hypotheses  # noqa: E402
from mars.induction.grammar_synthesizer import _UH_SEQ_INTERFACE, synthesize_grammar  # noqa: E402
from mars.induction.grammar_refiner import (  # noqa: E402
    explain_signals,
    extract_slot_signals,
    synthesize_per_slot,
)
from mars.induction.uh_seq_inductor import UHSeqProgramInductor  # noqa: E402


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _judge_config(judge_model: str) -> dict[str, str]:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if key.startswith("sk-or-") and not base_url:
        base_url = "https://openrouter.ai/api/v1"
    return {"model": judge_model, "base_url": base_url, "api_key": key}


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415
    from envs.common import Difficulty  # noqa: PLC0415
    from envs.seq_env.env import SequenceExploreEnvironment  # noqa: PLC0415

    random.seed(args.seed)
    np.random.seed(args.seed)
    SequenceExploreEnvironment.load_judge_config = lambda _self: _judge_config("openai/gpt-4o")
    difficulty = getattr(Difficulty, args.difficulty.upper())

    with contextlib.redirect_stdout(io.StringIO()):
        env = SequenceExploreEnvironment(difficulty=difficulty, required_steps=args.steps, free=False)

    # ------------------------------------------------------------------ #
    # 1. Collect all observations                                          #
    # ------------------------------------------------------------------ #
    inductor = UHSeqProgramInductor()
    t_collect = time.time()
    for _ in range(args.steps):
        main, vice = inductor.select_next_pair()
        with contextlib.redirect_stdout(io.StringIO()):
            result = _run_async(env.input_sequences(main, vice))
        if not result.get("success"):
            break
        inductor.add_result(result)
    collect_time = time.time() - t_collect

    n_obs = len(inductor.observations)
    if n_obs == 0:
        raise RuntimeError("no observations collected; check environment setup")

    # ------------------------------------------------------------------ #
    # 2. Build flat trace set for grammar synthesis                        #
    #    Use only first N_INIT observations, all 5 rule slots mixed.      #
    # ------------------------------------------------------------------ #
    n_init = min(args.n_init, n_obs)
    init_obs = inductor.observations[:n_init]

    # Mixed traces: (current, context, target) across all slots and init obs
    # The synthesizer does NOT know which slot each trace belongs to — it sees
    # only the raw observable interface.
    synth_traces = [
        obs.trace_for_rule(slot)
        for obs in init_obs
        for slot in range(1, 6)
    ]

    # All traces per slot (used for CPI scoring after synthesis)
    all_traces_by_slot = {
        slot: inductor.traces(slot)
        for slot in range(1, 6)
    }

    # ------------------------------------------------------------------ #
    # 3. Synthesize grammar                                                #
    # ------------------------------------------------------------------ #
    print(f"\n[grammar_synth] synthesizing {args.n_proposals} proposals "
          f"from {n_init} observations × 5 slots = {len(synth_traces)} traces "
          f"(model={args.synth_model})")

    synthesis = synthesize_grammar(
        synth_traces,
        interface_description=_UH_SEQ_INTERFACE,
        n_proposals=args.n_proposals,
        model=args.synth_model,
        temperature=0.8,
    )

    print(f"[grammar_synth] proposed={synthesis.proposed} valid={synthesis.valid} "
          f"time={synthesis.wall_time_s:.1f}s")
    for h in synthesis.hypotheses:
        print(f"  + {h.name}: {h.description[:60]}")
    if synthesis.errors:
        print(f"  errors ({len(synthesis.errors)}): {synthesis.errors[:5]}")

    # ------------------------------------------------------------------ #
    # 4a. Per-slot synthesis (hard mode)                                  #
    #     Each slot gets its own signal-enriched interface description.   #
    # ------------------------------------------------------------------ #
    per_slot_syntheses: dict[int, Any] = {}
    per_slot_synth_total_valid = 0
    if args.per_slot:
        print("\n[grammar_synth] per-slot synthesis mode")
        for slot in range(1, 6):
            slot_traces = all_traces_by_slot[slot]
            init_slot_traces = slot_traces[:n_init]
            signals = extract_slot_signals(slot, init_slot_traces)
            print(f"  slot {slot}: {explain_signals(signals)}")
            slot_synth = synthesize_per_slot(
                slot,
                init_slot_traces,
                model=args.synth_model,
                n_proposals=args.n_proposals,
                temperature=0.75,
            )
            per_slot_syntheses[slot] = slot_synth
            per_slot_synth_total_valid += slot_synth.valid
            print(f"    proposed={slot_synth.proposed} valid={slot_synth.valid}")
            for h in slot_synth.hypotheses[:3]:
                print(f"    + {h.name}: {h.description[:55]}")

        # Merge all per-slot hypotheses into one universal library
        all_per_slot_hypotheses = []
        for slot_synth in per_slot_syntheses.values():
            all_per_slot_hypotheses.extend(slot_synth.hypotheses)
        # Deduplicate by name
        seen_names: set[str] = set()
        merged: list[Any] = []
        for h in all_per_slot_hypotheses:
            if h.name not in seen_names:
                seen_names.add(h.name)
                merged.append(h)
        synthesis = type(synthesis)(
            proposed=sum(s.proposed for s in per_slot_syntheses.values()),
            valid=len(merged),
            hypotheses=merged,
            raw_proposals=[],
            errors=[],
            model=args.synth_model,
            wall_time_s=sum(s.wall_time_s for s in per_slot_syntheses.values()),
        )
        print(f"\n[grammar_synth] per-slot merged: {len(merged)} unique hypotheses")

    # ------------------------------------------------------------------ #
    # 4b. CPI with synthesized grammar (Run A)                            #
    # ------------------------------------------------------------------ #
    synth_per_slot: dict[str, Any] = {}
    synth_exact = 0
    if synthesis.hypotheses:
        for slot in range(1, 6):
            traces = all_traces_by_slot[slot]
            ranked = rank_hypotheses(synthesis.hypotheses, traces, complexity_weight=0.01)
            winner_h, winner_s = ranked[0]
            exact = winner_s.loss_mean == 0.0
            synth_exact += int(exact)
            synth_per_slot[f"rule_{slot}"] = {
                "winner": winner_h.name,
                "description": winner_h.description,
                "loss_mean": winner_s.loss_mean,
                "exact_rate": winner_s.exact_rate,
                "mdl_score": winner_s.mdl_score,
                "exact": exact,
                "top3": [
                    {"name": h.name, "loss_mean": s.loss_mean, "exact": s.loss_mean == 0.0}
                    for h, s in ranked[:3]
                ],
            }
    else:
        for slot in range(1, 6):
            synth_per_slot[f"rule_{slot}"] = {"winner": None, "exact": False, "loss_mean": 1.0}

    # ------------------------------------------------------------------ #
    # 5. CPI with hand-seeded grammar (Run B / control)                   #
    # ------------------------------------------------------------------ #
    hand_per_slot: dict[str, Any] = {}
    hand_exact = 0
    for slot in range(1, 6):
        ranked = inductor.ranked(slot)
        winner_h, winner_s = ranked[0]
        exact = winner_s.loss_mean == 0.0
        hand_exact += int(exact)
        hand_per_slot[f"rule_{slot}"] = {
            "winner": winner_h.name,
            "description": winner_h.description,
            "loss_mean": winner_s.loss_mean,
            "exact": exact,
        }

    # ------------------------------------------------------------------ #
    # 6. Refinement pass — counterexample-conditioned grammar extension    #
    #    For each unsolved rule slot, show its specific failing traces     #
    #    and ask for additional targeted primitives. This is the          #
    #    "failure trace → new primitive class" loop from SB-CPI spec.    #
    # ------------------------------------------------------------------ #
    unsolved = [slot for slot in range(1, 6) if not synth_per_slot.get(f"rule_{slot}", {}).get("exact")]
    refinement: Any = None
    if unsolved and synthesis.hypotheses and args.refinement_pass:
        refinement_traces = [
            obs.trace_for_rule(slot)
            for obs in inductor.observations[:n_init]
            for slot in unsolved
        ]
        failing_examples = [
            {
                "rule_slot": slot,
                "winner_so_far": synth_per_slot[f"rule_{slot}"]["winner"],
                "loss_mean": synth_per_slot[f"rule_{slot}"]["loss_mean"],
            }
            for slot in unsolved
        ]
        refinement_hint = (
            _UH_SEQ_INTERFACE
            + f"\n\nREFINEMENT CONTEXT — these {len(unsolved)} rule slot(s) were NOT solved "
            f"by the initial grammar:\n{json.dumps(failing_examples, indent=2)}\n"
            "Propose functions that SPECIFICALLY target these failing traces. "
            "Focus on position-wise character comparison (max/min per position), "
            "character-wise difference, and any pattern the initial grammar missed."
        )
        print(f"\n[grammar_synth] refinement pass: {len(unsolved)} unsolved slots "
              f"→ {len(refinement_traces)} targeted traces")
        refinement = synthesize_grammar(
            refinement_traces,
            interface_description=refinement_hint,
            n_proposals=args.n_proposals,
            model=args.synth_model,
            temperature=0.9,
        )
        print(f"[grammar_synth] refinement: proposed={refinement.proposed} valid={refinement.valid}")
        for h in refinement.hypotheses:
            print(f"  + {h.name}: {h.description[:60]}")

        # Re-run CPI with combined grammar on unsolved slots
        combined = synthesis.hypotheses + refinement.hypotheses
        for slot in unsolved:
            traces = all_traces_by_slot[slot]
            ranked = rank_hypotheses(combined, traces, complexity_weight=0.01)
            winner_h, winner_s = ranked[0]
            exact = winner_s.loss_mean == 0.0
            old_exact = synth_per_slot[f"rule_{slot}"]["exact"]
            synth_per_slot[f"rule_{slot}"].update({
                "winner": winner_h.name,
                "description": winner_h.description,
                "loss_mean": winner_s.loss_mean,
                "exact_rate": winner_s.exact_rate,
                "exact": exact,
                "refined": True,
                "improved": (not old_exact) and exact,
            })
            if exact and not old_exact:
                synth_exact += 1
                print(f"  rule_{slot}: FIXED by refinement → {winner_h.name} (loss=0.0)")

    # ------------------------------------------------------------------ #
    # 7. Assemble result                                                   #
    # ------------------------------------------------------------------ #
    result: dict[str, Any] = {
        "run_id": args.run_id,
        "score_type": "grammar-synthesis-experiment",
        "seed": args.seed,
        "difficulty": args.difficulty,
        "n_steps": n_obs,
        "n_init_for_synthesis": n_init,
        "synth_model": args.synth_model,
        "n_proposals_requested": args.n_proposals,
        # Key metrics
        "synthesized_grammar_rules_exact": synth_exact,
        "hand_seeded_grammar_rules_exact": hand_exact,
        "synthesized_grammar_proposed": synthesis.proposed,
        "synthesized_grammar_valid": synthesis.valid,
        # Per-slot breakdown
        "run_A_synthesized": synth_per_slot,
        "run_B_hand_seeded": hand_per_slot,
        # Grammar detail
        "synthesis": synthesis.to_dict(),
        # Timing
        "refinement_pass": args.refinement_pass,
        "refinement": refinement.to_dict() if refinement is not None else None,
        "collect_time_s": collect_time,
        "wall_time_s": collect_time + synthesis.wall_time_s + (refinement.wall_time_s if refinement else 0.0),
    }
    return result


def _print_report(row: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("GRAMMAR SYNTHESIS EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"seed={row['seed']} difficulty={row['difficulty']} "
          f"steps={row['n_steps']} n_init={row['n_init_for_synthesis']}")
    print(f"synth_model={row['synth_model']} "
          f"proposed={row['synthesized_grammar_proposed']} "
          f"valid={row['synthesized_grammar_valid']}")
    print()
    print(f"{'Rule':<8} {'Run A (synthesized)':<38} {'Run B (hand-seeded)':<38}")
    print("-" * 84)
    synth_rules = row.get("run_A_synthesized", {})
    hand_rules = row.get("run_B_hand_seeded", {})
    for slot in range(1, 6):
        key = f"rule_{slot}"
        s = synth_rules.get(key, {})
        h = hand_rules.get(key, {})
        s_name = (s.get("winner") or "—")[:20]
        h_name = (h.get("winner") or "—")[:20]
        s_mark = "✓" if s.get("exact") else "✗"
        h_mark = "✓" if h.get("exact") else "✗"
        s_loss = f"loss={s.get('loss_mean', 1.0):.3f}"
        print(f"rule_{slot}   {s_mark} {s_name:<20} {s_loss}   {h_mark} {h_name}")
    print("-" * 84)
    print(f"TOTAL    {row['synthesized_grammar_rules_exact']}/5 exact (synthesized)"
          f"          {row['hand_seeded_grammar_rules_exact']}/5 exact (hand-seeded)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grammar synthesis experiment vs hand-seeded CPI")
    parser.add_argument("--run_id", default=os.environ.get("MARS_GRAMMAR_SYNTH_RUN_ID", "grammar_synth_smoke"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--steps", type=int, default=5,
                        help="Total observations to collect (all used for CPI scoring)")
    parser.add_argument("--n_init", type=int, default=2,
                        help="Observations shown to grammar synthesizer (must be ≤ steps)")
    parser.add_argument("--synth_model", default=os.environ.get("MARS_SYNTH_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--n_proposals", type=int, default=16)
    parser.add_argument("--refinement_pass", action="store_true",
                        help="After initial synthesis, do a targeted pass for unsolved rule slots")
    parser.add_argument("--per_slot", action="store_true",
                        help="Synthesize per rule slot with signal-extracted interface (for hard rules)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = _PROJ / "lmw" / "grammar_synth" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"summary exists; pass --overwrite: {out_path}")

    row = run_experiment(args)
    out_path.write_text(
        json.dumps(row, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    _print_report(row)
    print(f"summary → {out_path}")


if __name__ == "__main__":
    main()
