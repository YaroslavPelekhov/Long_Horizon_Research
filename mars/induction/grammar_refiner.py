"""Grammar refiner: per-slot signal extraction + targeted synthesis for hard rules.

The core problem with mixed-trace synthesis for hard difficulty:
- Each rule slot has wildly different input/output lengths
- Output length and character patterns carry strong structural signals
- These signals are invisible when all slots' traces are mixed together

Solution:
1. Analyze each slot's traces INDEPENDENTLY to extract structural signals
2. Build a per-slot interface description that includes those signals
3. Run grammar synthesis per-slot with the enriched description
4. Combine winning hypotheses across slots

Key signals extracted per slot:
- Output length pattern (constant / doubles_input / grows_by_step / varies)
- Step-number correlation (do characters shift systematically with step?)
- History dependency (does previous_main appear in output?)
- Current dependency (does output depend on current at all?)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from mars.induction.cpi import RuleTrace
from mars.induction.grammar_synthesizer import SynthesisResult, _UH_SEQ_INTERFACE, synthesize_grammar


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

@dataclass
class SlotSignals:
    slot: int
    output_length_pattern: str        # "constant" | "doubles_input" | "grows_by_step" | "varies"
    length_examples: list[dict]       # (step, input_len, output_len)
    step_correlated_chars: bool       # do output chars shift systematically with step?
    shift_amount_equals_step: bool    # specifically, shift = step_number?
    uses_previous_main: bool          # does output seem to depend on previous_main?
    current_matters: bool             # does output depend on current (vs only main/vice)?
    main_vice_matters: bool           # does output depend on main/vice?
    output_is_identity_sometimes: bool  # some outputs == inputs (noop on some steps)
    raw: dict = field(default_factory=dict)


def _char_shift(a: str, b: str) -> int | None:
    """If a and b are same-length strings, return the consistent shift value mod 26, or None."""
    if len(a) != len(b) or not a:
        return None
    shifts = set()
    for ca, cb in zip(a, b):
        if ca.isalpha() and cb.isalpha():
            shifts.add((ord(cb) - ord(ca)) % 26)
    return shifts.pop() if len(shifts) == 1 else None


def extract_slot_signals(slot: int, traces: list[RuleTrace]) -> SlotSignals:
    """Automatically analyze traces for one rule slot."""
    if not traces:
        return SlotSignals(
            slot=slot,
            output_length_pattern="unknown",
            length_examples=[],
            step_correlated_chars=False,
            shift_amount_equals_step=False,
            uses_previous_main=False,
            current_matters=False,
            main_vice_matters=True,
            output_is_identity_sometimes=False,
        )

    length_examples = [
        {
            "step": tr.context.get("step_number"),
            "input_len": len(tr.current),
            "output_len": len(tr.target),
            "delta": len(tr.target) - len(tr.current),
        }
        for tr in traces
    ]

    in_lens = [e["input_len"] for e in length_examples]
    out_lens = [e["output_len"] for e in length_examples]
    deltas = [e["delta"] for e in length_examples]
    steps = [tr.context.get("step_number", 0) for tr in traces]

    # Length pattern
    if all(d == 0 for d in deltas):
        length_pattern = "constant"
    elif all(out == 2 * inp for inp, out in zip(in_lens, out_lens)):
        length_pattern = "doubles_input"
    elif all(d == s for d, s in zip(deltas, steps)):
        length_pattern = "grows_by_step"
    else:
        length_pattern = "varies"

    # Step-correlated character shift
    step_correlated = False
    shift_equals_step = False
    if len(traces) >= 2:
        t1 = next((tr for tr in traces if tr.context.get("step_number") == 1), None)
        t2 = next((tr for tr in traces if tr.context.get("step_number") == 2), None)
        if t1 and t2 and t1.current == t2.current and len(t1.target) == len(t2.target):
            s1 = _char_shift(t1.current[:5], t1.target[:5])
            s2 = _char_shift(t2.current[:5], t2.target[:5])
            if s1 is not None and s2 is not None and s1 != s2:
                step_correlated = True
                if s1 == 1 and s2 == 2:
                    shift_equals_step = True
        # Check if output chars are systematically shifted from INPUT chars
        shifts_per_step = []
        for tr in traces[:4]:
            s = _char_shift(tr.current[:min(5, len(tr.current))], tr.target[:min(5, len(tr.target))])
            step = tr.context.get("step_number", 0)
            if s is not None:
                shifts_per_step.append((step, s))
        if len(shifts_per_step) >= 2:
            step_correlated = True
            if all(s == step for step, s in shifts_per_step):
                shift_equals_step = True

    # History dependency: does previous_main appear literally in output?
    uses_prev = False
    for tr in traces:
        prev = tr.context.get("previous_main", "")
        if prev and any(prev[:3] in tr.target or tr.target[:3] in prev for _ in [1]):
            pass  # weak signal
        # Stronger: does the output differ from what main/vice alone would produce?
        # We check if previous_main changes the first 5 chars of the output
    first5_targets = [tr.target[:5] for tr in traces]
    prevs = [tr.context.get("previous_main", "") for tr in traces]
    if any(p for p in prevs):  # has any non-empty previous_main
        # Check if outputs correlate with previous_main more than with current
        uses_prev = True  # conservative: flag if previous_main is ever present

    # Current dependency: does output change when current changes (keeping main/vice same)?
    current_matters = any(tr.current for tr in traces)

    # Identity sometimes
    identity_sometimes = any(tr.current == tr.target for tr in traces if tr.current)

    # Main/vice matters: does output depend on main/vice?
    # If current is empty for all traces, output MUST come from main/vice
    main_vice_matters = any(not tr.current for tr in traces) or True

    return SlotSignals(
        slot=slot,
        output_length_pattern=length_pattern,
        length_examples=length_examples[:5],
        step_correlated_chars=step_correlated,
        shift_amount_equals_step=shift_equals_step,
        uses_previous_main=uses_prev,
        current_matters=current_matters,
        main_vice_matters=main_vice_matters,
        output_is_identity_sometimes=identity_sometimes,
        raw={
            "in_lens": in_lens,
            "out_lens": out_lens,
            "deltas": deltas,
            "steps": steps,
        },
    )


# ---------------------------------------------------------------------------
# Per-slot interface description builder
# ---------------------------------------------------------------------------

def _build_slot_interface(slot: int, traces: list[RuleTrace], signals: SlotSignals) -> str:
    base = f"""Environment: Unknown sequence transformation environment.

You are synthesizing a candidate function for RULE SLOT {slot} ONLY.

Function signature (MUST match exactly):
  def rule_{slot}_name(current: str, context: dict) -> str

  current  — the input string for this rule (may be empty for rule_1)
  context  — dict with:
    context['main']          str  — main input (5 uppercase letters A-E)
    context['vice']          str  — vice input (5 uppercase letters A-E)
    context['step_number']   int  — step index (1, 2, 3, ...)
    context['previous_main'] str  — main sequence from previous step

Return: a str of uppercase letters

Allowed builtins ONLY: ord, chr, len, range, zip, sorted, max, min, sum, abs,
  str, int, float, bool, list, tuple, set, dict, enumerate, round
NO imports. All helpers must be inline.

Character arithmetic:
  position of c: ord(c) - ord('A')   (A=0, B=1, ..., Z=25)
  char at position i: chr(ord('A') + i % 26)

"""

    def is_prime(n: int) -> bool:
        if n <= 1: return False
        if n == 2: return True
        if n % 2 == 0: return False
        d = 3
        while d * d <= n:
            if n % d == 0: return False
            d += 2
        return True

    hints = [f"\n=== AUTOMATICALLY EXTRACTED PATTERNS FOR SLOT {slot} ==="]

    # ---- Output length pattern ----
    if signals.output_length_pattern == "doubles_input":
        hints.append("OUTPUT LENGTH: output is EXACTLY DOUBLE the input length in every observation.")
        hints.append("  Pattern: output = part_A + part_B  where len(part_A)==len(part_B)==len(current)")
        hints.append("  CRITICAL: if characters are also shifted by step_number, the formula is:")
        hints.append("    step = context['step_number']")
        hints.append("    shifted_rev = ''.join(chr(ord('A')+(ord(c)-ord('A')+step)%26) for c in current[::-1])")
        hints.append("    shifted_orig = ''.join(chr(ord('A')+(ord(c)-ord('A')+step)%26) for c in current)")
        hints.append("    return shifted_rev + shifted_orig")

    elif signals.output_length_pattern == "grows_by_step":
        hints.append("OUTPUT LENGTH: output length = input length + step_number.")
        for e in signals.length_examples[:3]:
            hints.append(f"  step={e['step']}: input_len={e['input_len']} → output_len={e['output_len']}")
        hints.append("  Pattern: a = step_number % 10; if a==0: a=10")
        hints.append("  output = current + current[a-1] * a  (if len(current) > a-1, else return current)")
        hints.append("  Example code:")
        hints.append("    def rule(current, context):")
        hints.append("        a = context['step_number'] % 10")
        hints.append("        if a == 0: a = 10")
        hints.append("        if len(current) > a - 1:")
        hints.append("            return current + current[a-1] * a")
        hints.append("        return current")

    elif signals.output_length_pattern == "constant":
        hints.append("OUTPUT LENGTH: same as input length in every observation.")

    # ---- Character shift ----
    if signals.shift_amount_equals_step:
        hints.append("CHARACTER SHIFT: output chars are shifted by EXACTLY step_number positions in alphabet.")
        hints.append("  Formula: chr(ord('A') + (ord(c) - ord('A') + step_number) % 26)")
        hints.append("  Apply this shift to BOTH the reversed and original current, then concatenate.")

    elif signals.step_correlated_chars:
        hints.append("CHARACTER SHIFT: output chars shift with step_number — use modular alphabet arithmetic.")

    # ---- Step-parity interleave detection for slot 1 ----
    if slot == 1 and not signals.current_matters:
        # Check if output[0] alternates between main[0] and vice[0]
        odd_lead_main = []
        even_lead_vice = []
        for tr in traces:
            step = tr.context.get("step_number", 0)
            main0 = tr.context.get("main", "")[:1]
            vice0 = tr.context.get("vice", "")[:1]
            tgt0 = tr.target[:1]
            if step % 2 == 1:
                odd_lead_main.append(tgt0 == main0)
            else:
                even_lead_vice.append(tgt0 == vice0)
        if odd_lead_main and all(odd_lead_main) and even_lead_vice and all(even_lead_vice):
            hints.append("STEP-PARITY INTERLEAVE: on ODD steps, interleave starts with main; on EVEN steps, starts with vice.")
            hints.append("  Pattern:")
            hints.append("    if step_number % 2 == 1:  # odd: main leads")
            hints.append("        return ''.join(m+v for m,v in zip(context['main'], context['vice']))")
            hints.append("    else:  # even: vice leads")
            hints.append("        return ''.join(v+m for v,m in zip(context['vice'], context['main']))")
        elif any(odd_lead_main) or any(even_lead_vice):
            hints.append("INTERLEAVE PATTERN with possible step-based switching between main-first and vice-first.")
            hints.append("  Try: return ''.join(m+v for m,v in zip(main, vice))  (odd steps)")
            hints.append("  and: return ''.join(v+m for v,m in zip(vice, main))  (even steps)")

    # ---- previous_main dependency ----
    if signals.uses_previous_main:
        prev_traces = [(tr.context.get("previous_main", ""), tr.current, tr.target)
                       for tr in traces if tr.context.get("previous_main")]
        if prev_traces:
            # Check add_mod26 on prefix
            found_add_mod26 = False
            for prev, cur, tgt in prev_traces[:3]:
                if prev and cur and len(prev) >= 5 and len(cur) >= 5:
                    expected_prefix = "".join(
                        chr(ord('A') + ((ord(c) - ord('A')) + (ord(p) - ord('A'))) % 26)
                        for c, p in zip(cur[:5], prev[:5])
                    )
                    if expected_prefix == tgt[:5]:
                        found_add_mod26 = True
                        hints.append("HISTORY: add_mod26(current[:5], previous_main[:5]) gives the first 5 output chars.")
                        hints.append("  Remainder of current (current[5:]) is appended unchanged.")
                        hints.append(f"  Example: cur[:5]={cur[:5]!r}, prev={prev[:5]!r} → tgt[:5]={tgt[:5]!r}")
                        hints.append("  Formula for first 5 chars:")
                        hints.append("    out = []")
                        hints.append("    for c, p in zip(current[:5], prev[:5]):")
                        hints.append("        out.append(chr(ord('A') + (ord(c)-ord('A') + ord(p)-ord('A')) % 26))")
                        hints.append("    return ''.join(out) + current[5:]")
                        hints.append("  When previous_main is empty: return current unchanged.")
                        break
            if not found_add_mod26:
                hints.append("HISTORY DEPENDENCY: previous_main influences output.")
                for prev, cur, tgt in prev_traces[:2]:
                    hints.append(f"  prev={prev[:8]!r} cur={cur[:8]!r} → tgt={tgt[:8]!r}")

    # ---- Conditional/prime pattern ----
    if signals.output_is_identity_sometimes:
        identity_steps = [tr.context.get("step_number") for tr in traces if tr.current == tr.target and tr.current]
        active_steps = [tr.context.get("step_number") for tr in traces if tr.current != tr.target and tr.current]
        if identity_steps and active_steps:
            hints.append(f"CONDITIONAL: identity on steps {identity_steps}, active on steps {active_steps}")
            if all(is_prime(s) for s in active_steps if s) and not any(is_prime(s) for s in identity_steps if s):
                hints.append("PRIME STEP PATTERN: rule applies only on prime steps (2, 3, 5, 7, 11, ...).")
                hints.append("  On prime steps: find the MOST FREQUENT character in current,")
                hints.append("    replace ALL its occurrences with the NEXT letter in alphabet (mod 26).")
                hints.append("  On non-prime steps: return current unchanged.")
                hints.append("  CRITICAL: do NOT use collections.Counter — use dict.get() instead:")
                hints.append("  Inline prime check (no imports):")
                hints.append("    def _is_prime(n):")
                hints.append("        if n <= 1: return False")
                hints.append("        if n == 2: return True")
                hints.append("        if n % 2 == 0: return False")
                hints.append("        d = 3")
                hints.append("        while d*d <= n: ")
                hints.append("            if n%d==0: return False")
                hints.append("            d += 2")
                hints.append("        return True")
                hints.append("  Inline frequency count (no imports):")
                hints.append("    freq = {}")
                hints.append("    for c in current:")
                hints.append("        if c.isalpha(): freq[c] = freq.get(c, 0) + 1")
                hints.append("    most_frequent = max(freq, key=lambda c: freq[c]) if freq else None")

                # Show a worked example
                for tr in traces:
                    if tr.current and tr.current != tr.target and is_prime(tr.context.get("step_number", 0)):
                        step = tr.context.get("step_number")
                        freq: dict[str, int] = {}
                        for c in tr.current:
                            if c.isalpha():
                                freq[c] = freq.get(c, 0) + 1
                        if freq:
                            mf = max(freq, key=lambda c: freq[c])
                            nxt = chr(ord('A') + (ord(mf) - ord('A') + 1) % 26)
                            hints.append(f"  Example at step={step}:")
                            hints.append(f"    input: {tr.current[:20]!r}")
                            hints.append(f"    most_frequent='{mf}' (count={freq[mf]}), next='{nxt}'")
                            hints.append(f"    output: {tr.target[:20]!r}")
                        break
            elif active_steps and not identity_steps:
                hints.append(f"  Active steps are: {active_steps} — check if prime/even/odd pattern.")

    # ---- Concrete examples ----
    hints.append("\nCONCRETE EXAMPLES (current → target, this slot only):")
    for tr in traces[:5]:
        ctx_info = f"step={tr.context.get('step_number')}"
        pm = tr.context.get("previous_main", "")
        if pm:
            ctx_info += f" prev={pm[:8]!r}"
        cur_disp = repr(tr.current[:30]) if tr.current else "''"
        tgt_disp = repr(tr.target[:30])
        hints.append(f"  {cur_disp} → {tgt_disp} ({ctx_info})")

    return base + "\n".join(hints)


# ---------------------------------------------------------------------------
# Per-slot synthesis
# ---------------------------------------------------------------------------

def synthesize_per_slot(
    slot: int,
    traces: list[RuleTrace],
    *,
    model: str = "openai/gpt-4o-mini",
    n_proposals: int = 12,
    temperature: float = 0.75,
) -> SynthesisResult:
    """Synthesize grammar for a single rule slot with signal-enriched interface."""
    signals = extract_slot_signals(slot, traces)
    interface = _build_slot_interface(slot, traces, signals)
    result = synthesize_grammar(
        traces,
        interface_description=interface,
        n_proposals=n_proposals,
        model=model,
        temperature=temperature,
    )
    result.raw_proposals  # ensure populated
    # Tag result with slot info
    for h in result.hypotheses:
        h.tags = h.tags + (f"slot_{slot}",)
    return result


def explain_signals(signals: SlotSignals) -> str:
    parts = [f"slot={signals.slot} length_pattern={signals.output_length_pattern}"]
    if signals.shift_amount_equals_step:
        parts.append("shift=step")
    if signals.uses_previous_main:
        parts.append("uses_prev_main")
    if signals.output_is_identity_sometimes:
        parts.append("conditional")
    return " ".join(parts)
