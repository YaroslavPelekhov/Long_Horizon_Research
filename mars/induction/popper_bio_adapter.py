"""Popperian Bio adapter — inheritance rules as PROHIBITIONS.

An event is one (parent1, parent2, offspring) triple drawn from a cross.
A hypothesis is a predicate allows(event) -> bool. It FORBIDS a region of
plausible (parent -> offspring) combinations.

Examples of prohibitions the engine can discover from data alone:
  - "if either parent is red, offspring is never white" (dominance as a ban)
  - "offspring shell never equals a type absent from both parents" (no novelty)
  - "viable offspring never appears when viability_rate < threshold for combo X"
    (lethality as a ban on existence)

Empirical content = fraction of plausible (parent-pair, offspring) combinations
the prohibition forbids. Real viable offspring must never be forbidden.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
from collections import Counter
from typing import Any


@contextlib.contextmanager
def _silence():
    with contextlib.redirect_stdout(io.StringIO()):
        yield

from mars.agents.base import call_llm, make_openai_client
from mars.induction.popperian_cpi import PopperAdapter, Prohibition, ProhibitionScore


class PopperBioAdapter(PopperAdapter):
    name = "uh_bio_popper"

    def __init__(self, cross_records: list[dict], judge_model: str = "openai/gpt-4o",
                 env: Any = None, run_async: Any = None):
        self.cross_records = cross_records
        self.judge_model = judge_model
        self.env = env                # if set, enables ACTIVE severe testing
        self._run_async = run_async   # callable to run a coroutine synchronously
        self._real: list[dict] | None = None
        self.severe_observations: list[dict] = []  # viability findings from active tests
        # observed feature vocabularies (data-grounded, no rubric)
        self._colors: list[str] = []
        self._shells: list[str] = []
        self._size_levels: list[str] = []
        self._size_range: tuple[float, float] = (0.0, 1.0)

    def interface_description(self) -> str:
        self._build_vocab()
        return (
            "Domain: hidden inheritance rules of an organism.\n"
            "An EVENT is one parent1 x parent2 -> offspring observation:\n"
            "  event['parent1'], event['parent2'], event['offspring'] are dicts with\n"
            "    body_color (str), shell_shape (str), body_size (str), size_score (float).\n"
            "  event['viability_rate'] (float) = fraction of the cross's fertilizations\n"
            "    that produced viable offspring (low rate => many zygotes died).\n\n"
            f"Observed body_color values: {self._colors}\n"
            f"Observed shell_shape values: {self._shells}\n"
            f"Observed body_size values: {self._size_levels}\n"
            f"Observed size_score range: {self._size_range[0]:.1f} to {self._size_range[1]:.1f}\n\n"
            "A prohibition forbids some parent->offspring combinations. The strongest\n"
            "honest laws forbid the most while never contradicting real viable offspring."
        )

    def predicate_signature(self) -> str:
        return ("def allows(event) -> bool\n"
                "  # event has parent1, parent2, offspring (each: body_color, shell_shape,\n"
                "  #   body_size, size_score) and viability_rate.\n"
                "  # return False where the offspring is FORBIDDEN given the parents.")

    def _build_vocab(self) -> None:
        if self._colors:
            return
        colors, shells, sizes, scores = set(), set(), set(), []
        for cr in self.cross_records:
            for key in ("parent1_phenotype", "parent2_phenotype"):
                p = cr.get(key, {})
                if p.get("body_color"): colors.add(p["body_color"])
                if p.get("shell_shape"): shells.add(p["shell_shape"])
                if p.get("body_size"): sizes.add(p["body_size"])
                if p.get("size_score"): scores.append(p["size_score"])
            for o in cr.get("offspring", []):
                ph = o.get("phenotype", {})
                if ph.get("body_color"): colors.add(ph["body_color"])
                if ph.get("shell_shape"): shells.add(ph["shell_shape"])
                if ph.get("body_size"): sizes.add(ph["body_size"])
                if ph.get("size_score"): scores.append(ph["size_score"])
        self._colors = sorted(colors)
        self._shells = sorted(shells)
        self._size_levels = sorted(sizes)
        self._size_range = (min(scores), max(scores)) if scores else (0.0, 1.0)

    def real_events(self) -> list[dict]:
        if self._real is not None:
            return self._real
        self._build_vocab()
        events = []
        for cr in self.cross_records:
            p1 = cr.get("parent1_phenotype", {})
            p2 = cr.get("parent2_phenotype", {})
            vr = cr.get("viability_rate", 1.0)
            for o in cr.get("offspring", []):
                ph = o.get("phenotype", {})
                if ph:
                    events.append({
                        "parent1": _clean(p1), "parent2": _clean(p2),
                        "offspring": _clean(ph), "viability_rate": vr,
                    })
        self._real = events
        return events

    def plausible_events(self, n: int) -> list[dict]:
        """Plausible = real parent pairs x all observed-value offspring combos.
        Content is measured against this grounded space, so a prohibition only
        earns content for forbidding combinations that COULD plausibly occur."""
        self._build_vocab()
        real = self.real_events()
        if not real:
            return []
        parent_pairs = [(e["parent1"], e["parent2"], e["viability_rate"]) for e in real]
        # de-dup parent pairs
        seen = set()
        uniq_pairs = []
        for p1, p2, vr in parent_pairs:
            key = (p1.get("body_color"), p1.get("shell_shape"),
                   p2.get("body_color"), p2.get("shell_shape"))
            if key not in seen:
                seen.add(key)
                uniq_pairs.append((p1, p2, vr))

        lo, hi = self._size_range
        size_buckets = [lo, (lo + hi) / 2, hi] if hi > lo else [lo]
        combos = list(itertools.product(self._colors, self._shells,
                                        self._size_levels or ["?"], size_buckets))
        events = []
        import random
        rng = random.Random(0)
        while len(events) < n and uniq_pairs and combos:
            p1, p2, vr = rng.choice(uniq_pairs)
            c, sh, sz, score = rng.choice(combos)
            events.append({
                "parent1": p1, "parent2": p2,
                "offspring": {"body_color": c, "shell_shape": sh,
                              "body_size": sz, "size_score": score},
                "viability_rate": vr,
            })
        return events

    # ----- ACTIVE severe testing -----------------------------------------

    def supports_active(self) -> bool:
        return self.env is not None and self._run_async is not None

    def action_space(self) -> str:
        """Expose ONE atomic action (cross two organisms) and the current
        organisms available to act on. No strategy — the engine plans."""
        if self.env is None:
            return ""
        # List a compact, diverse set of available organisms (id + phenotype)
        items = []
        for oid, o in list(self.env.organisms.items()):
            ph = o.get("phenotype", {})
            items.append({"id": oid, "color": ph.get("body_color"),
                          "shell": ph.get("shell_shape"), "size": ph.get("body_size")})
        # Keep a representative, capped sample (founders + variety of offspring)
        founders = [it for it in items if it["id"] <= 10]
        rest = [it for it in items if it["id"] > 10]
        # diversify rest by (color, shell)
        seen, diverse = set(), []
        for it in rest:
            key = (it["color"], it["shell"])
            if key not in seen:
                seen.add(key); diverse.append(it)
        shown = founders + diverse[:20]
        return (
            'ATOMIC ACTION (the only action; returns offspring + viability_rate):\n'
            '{"action": "cross", "p1": <organism id>, "p2": <organism id>, "n": <int>}\n'
            'Offspring produced by a cross become NEW organisms with fresh ids that\n'
            'you can use as parents in later actions (this is how you build lineages\n'
            'to reach allele combinations not present in any single current organism).\n'
            'Newly created offspring ids continue from the current maximum id.\n\n'
            f'CURRENT ORGANISMS (id: color/shell/size):\n' +
            "\n".join(f'  {it["id"]}: {it["color"]}/{it["shell"]}/{it["size"]}' for it in shown)
        )

    def execute_action(self, action: dict) -> list[dict]:
        """Execute ONE atomic cross. No strategy — just translate to env API."""
        if not self.supports_active() or action.get("action") != "cross":
            return []
        p1, p2 = action.get("p1"), action.get("p2")
        n = int(action.get("n", 12))
        if p1 is None or p2 is None:
            return []

        async def _do():
            if len(self.env.organisms) > 150:
                with _silence():
                    await self.env.remove_organisms(
                        [o for o in list(self.env.organisms.keys()) if o > 10])
            with _silence():
                return await self.env.conduct_cross(p1, p2, n)

        cr = self._run_async(_do())
        if not isinstance(cr, dict) or not cr.get("success"):
            return []
        self.cross_records.append(cr)
        self.severe_observations.append({
            "p1": p1, "p2": p2,
            "p1_shell": cr.get("parent1_phenotype", {}).get("shell_shape"),
            "p2_shell": cr.get("parent2_phenotype", {}).get("shell_shape"),
            "viability_rate": cr.get("viability_rate", 1.0),
        })
        p1p = cr.get("parent1_phenotype", {})
        p2p = cr.get("parent2_phenotype", {})
        vr = cr.get("viability_rate", 1.0)
        new_events = []
        for o in cr.get("offspring", []):
            ph = o.get("phenotype", {})
            if ph:
                new_events.append({"parent1": _clean(p1p), "parent2": _clean(p2p),
                                   "offspring": _clean(ph), "viability_rate": vr})
        self._real = None
        return new_events

    def verbalize(self, survivors: list[tuple[Prohibition, ProhibitionScore]]) -> str:
        """Compose a genetics report FROM the surviving prohibitions.
        The LLM sees only discovered bans + raw facts — no rubric."""
        if not survivors:
            return "No surviving inheritance prohibitions discovered."
        bans = []
        for p, s in survivors[:10]:
            bans.append(f"- FORBIDDEN (content {s.content:.2f}): {p.description}")
        facts = _bio_facts(self.cross_records)
        # Severe-testing findings: viability under deliberately-created conflicts
        severe = ""
        if self.severe_observations:
            vias = [so["viability_rate"] for so in self.severe_observations]
            base = max(vias) if vias else 1.0  # best-case viability as reference
            lines = [f"\nACTIVE SEVERE-TEST FINDINGS (engine-designed crosses; "
                     f"best observed viability ~{base:.2f}):"]
            for so in self.severe_observations:
                collapse = so["viability_rate"] < base * 0.6
                lines.append(
                    f"- cross of {so.get('p1_shell')}-type x {so.get('p2_shell')}-type "
                    f"→ viability {so['viability_rate']:.2f}"
                    + ("  <-- COLLAPSE: this allele combination appears LETHAL"
                       if collapse else "")
                )
            severe = "\n".join(lines)
        client = make_openai_client()
        prompt = (
            "You are a geneticist. Through experiments you have established a set of\n"
            "PROHIBITIONS — combinations that NEVER occur among viable offspring.\n"
            "Each prohibition survived all data without a single counterexample.\n\n"
            f"ESTABLISHED PROHIBITIONS (laws of what cannot happen):\n" + "\n".join(bans) + "\n\n"
            f"RAW EXPERIMENTAL FACTS:\n{facts}\n{severe}\n\n"
            "Translate these prohibitions into a formal inheritance report. A ban like\n"
            "'if a parent is red, offspring is never white' IS a dominance statement\n"
            "(red dominates white). A ban tied to low viability IS a lethal-combination\n"
            "statement. Cover: genetic architecture / ploidy (infer from viability if\n"
            "fertilizations die), body size mechanism and approximate per-allele values,\n"
            "color dominance ordering, shell mechanism, and any lethal combination.\n"
            "Ground every claim in the prohibitions and facts above."
        )
        return call_llm(
            client, model=self.judge_model,
            system="You write rigorous genetics reports from established prohibitions.",
            user=prompt, max_tokens=900, temperature=0.3,
        )


# ---------------------------------------------------------------------------

def _clean(p: dict) -> dict:
    return {
        "body_color": p.get("body_color"),
        "shell_shape": p.get("shell_shape"),
        "body_size": p.get("body_size"),
        "size_score": p.get("size_score"),
    }


def _bio_facts(cross_records: list[dict]) -> str:
    lines = []
    via = [cr.get("viability_rate", 1.0) for cr in cross_records]
    if via:
        lines.append(f"- viability across {len(via)} crosses: min={min(via):.2f}, "
                     f"max={max(via):.2f}, mean={sum(via)/len(via):.2f} "
                     f"(rates below 1.0 mean many fertilizations died)")
    sizes = [o.get("phenotype", {}).get("size_score")
             for cr in cross_records for o in cr.get("offspring", [])
             if o.get("phenotype", {}).get("size_score")]
    if sizes:
        lines.append(f"- offspring size_score range: {min(sizes):.1f} to {max(sizes):.1f}")
    for cr in cross_records[:1]:
        for k in ("parent1_phenotype", "parent2_phenotype"):
            p = cr.get(k, {})
            if p:
                lines.append(f"- founder example: color={p.get('body_color')}, "
                             f"shell={p.get('shell_shape')}, size={p.get('size_score')}")
    for cr in cross_records[:10]:
        p1 = cr.get("parent1_phenotype", {}); p2 = cr.get("parent2_phenotype", {})
        offs = cr.get("offspring", [])
        if not offs:
            continue
        c = Counter(o.get("phenotype", {}).get("body_color") for o in offs)
        sh = Counter(o.get("phenotype", {}).get("shell_shape") for o in offs)
        lines.append(
            f"- {p1.get('body_color')}/{p1.get('shell_shape')} x "
            f"{p2.get('body_color')}/{p2.get('shell_shape')} -> "
            f"colors {dict(c)}, shells {dict(sh)}, viability {cr.get('viability_rate',1.0):.2f}"
        )
    return "\n".join(lines)
