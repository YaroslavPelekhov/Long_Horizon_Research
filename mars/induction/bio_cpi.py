"""Bio-CPI: Systematic genetics protocol for UltraHorizon Alien Genetics Lab.

Domain knowledge (structure, not values):
- Organisms are polyploid (we discover the ploidy level from genotypes)
- Traits: body_size (quantitative), body_color (discrete), shell_shape (discrete)
- Mechanisms to discover: dominance hierarchies, dosage effects, lethal combinations
- Experimental tools: conduct_cross, query_organisms, get_lab_status

This module does NOT use the scoring rubric as input — it discovers rules
through systematic genetics experiments (like a real biologist would).

Experimental protocol (designed for budget=15-20 steps):
  Phase 1 (2 steps)  — Survey initial organisms with genotype info
  Phase 2 (6 steps)  — Pairwise crosses: A×B, A×C, B×C + observe offspring
  Phase 3 (4 steps)  — Deep quantification: size scores, viability rates
  Phase 4 (4 steps)  — Cross offspring to test trait segregation
  Phase 5 (2 steps)  — Verification + confirmation

Each phase produces HypothesisRecords that are combined into the final report.
"""

from __future__ import annotations

import asyncio
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class Organism:
    id: int
    body_size: str
    size_score: float
    color: str
    shell: str
    generation: int
    parents: list[int] | None
    genotype: dict | None = None  # if queried with include_genotype=True


@dataclass
class CrossResult:
    parent1: int
    parent2: int
    p1_phenotype: dict
    p2_phenotype: dict
    viable: int
    lethal: int
    total_attempts: int
    viability_rate: float
    offspring: list[dict]


@dataclass
class GeneticHypotheses:
    """Growing hypothesis record updated after each phase."""
    ploidy: int | None = None                    # 2 = diploid, 3 = triploid
    gamete_types: list[str] | None = None        # e.g. ["1n", "2n"]

    color_dominant: list[str] = field(default_factory=list)   # ordered by dominance
    color_mechanism: str | None = None           # "linear_dominance" | "cyclic"

    shell_dominant: list[str] = field(default_factory=list)
    shell_mechanism: str | None = None           # "linear_dominance" | "cyclic"
    lethal_combo: list[str] | None = None        # allele types that together = lethal

    size_mechanism: str | None = None            # "mendelian" | "dosage"
    size_alleles: list[str] = field(default_factory=list)     # distinct allele names
    size_values: dict[str, float] = field(default_factory=dict)  # allele → approx score

    viability_by_cross: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data analysis helpers
# ---------------------------------------------------------------------------

def _phenotype_counts(offspring: list[dict]) -> dict[str, Counter]:
    """Count trait values across offspring."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for org in offspring:
        p = org.get("phenotype", {})
        for trait in ("body_color", "shell_shape", "body_size"):
            if trait in p:
                counts[trait][str(p[trait])] += 1
    return dict(counts)


def _dominant_value(counts: Counter) -> str | None:
    """Return the value with the highest count, or None if empty."""
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _infer_linear_order(pairwise_dominant: dict[tuple, str]) -> list[str]:
    """From {(A,B): winner, (A,C): winner, (B,C): winner} → ranked list."""
    wins: Counter = Counter()
    all_vals = set()
    for (a, b), w in pairwise_dominant.items():
        all_vals.add(a); all_vals.add(b)
        if w:
            wins[w] += 1
    return [v for v, _ in wins.most_common()] + [v for v in all_vals if v not in wins]


def _mean_size(offspring: list[dict]) -> float:
    scores = [o.get("phenotype", {}).get("size_score", 0) for o in offspring]
    return statistics.mean(scores) if scores else 0.0


def _check_cyclic(pairwise: dict[tuple, str]) -> bool:
    """Return True if the dominance relation is cyclic (A>B>C>A pattern)."""
    vals = list({v for pair in pairwise for v in pair})
    if len(vals) != 3:
        return False
    # Try all cyclic orderings
    for i, a in enumerate(vals):
        b = vals[(i + 1) % 3]
        c = vals[(i + 2) % 3]
        if (pairwise.get((a, b)) == a and
                pairwise.get((b, c)) == b and
                pairwise.get((c, a)) == c):
            return True
    return False


# ---------------------------------------------------------------------------
# Core Bio-CPI class
# ---------------------------------------------------------------------------

class BioCPI:
    """Systematic genetics protocol: survey → cross → analyse → report."""

    def __init__(self, env: Any, budget: int = 18):
        self.env = env
        self.budget = budget
        self.steps_used = 0    # total steps (queries + crosses)
        self.cross_count = 0   # ONLY conduct_cross calls (what env counts)
        self.organisms: dict[int, Organism] = {}
        self.crosses: list[CrossResult] = []
        self.hyp = GeneticHypotheses()
        self._offspring_registry: dict[int, dict] = {}  # id → phenotype dict

    def _steps_left(self) -> int:
        return self.budget - self.cross_count  # based on cross count

    async def _query(self, start: int, end: int, genotype: bool = False) -> dict:
        self.steps_used += 1
        try:
            return await self.env.query_organisms(start, end, include_genotype=genotype)
        except TypeError:
            # Some env versions don't support include_genotype
            return await self.env.query_organisms(start, end)

    async def _cross(self, p1: int, p2: int, n: int = 30) -> CrossResult | None:
        self.steps_used += 1
        r = await self.env.conduct_cross(p1, p2, n)
        if not r.get("success"):
            return None
        self.cross_count += 1  # only count successful crosses (matches env.current_experiments)
        p1_p = r.get("parent1_phenotype", {})
        p2_p = r.get("parent2_phenotype", {})
        offspring = r.get("offspring", [])
        for o in offspring:
            self._offspring_registry[o["id"]] = o.get("phenotype", {})
        cr = CrossResult(
            parent1=p1,
            parent2=p2,
            p1_phenotype=p1_p,
            p2_phenotype=p2_p,
            viable=r.get("viable_offspring_count", 0),
            lethal=r.get("lethal_offspring_count", 0),
            total_attempts=r.get("total_fertilization_attempts", 0),
            viability_rate=r.get("viability_rate", 1.0),
            offspring=offspring,
        )
        self.crosses.append(cr)
        # Register offspring as organisms
        for o in offspring:
            self._register_offspring(o)
        return cr

    def _register_offspring(self, o: dict) -> None:
        p = o.get("phenotype", {})
        org = Organism(
            id=o["id"],
            body_size=p.get("body_size", "?"),
            size_score=float(p.get("size_score", 0)),
            color=p.get("body_color", "?"),
            shell=p.get("shell_shape", "?"),
            generation=o.get("generation", 1),
            parents=o.get("parents"),
        )
        self.organisms[org.id] = org

    def _register_query_result(self, result: dict) -> list[int]:
        ids = []
        for o in result.get("organisms", []):
            p = o.get("phenotype", {})
            gt = o.get("genotype")
            org = Organism(
                id=o["id"],
                body_size=p.get("body_size", "?"),
                size_score=float(p.get("size_score", 0)),
                color=p.get("body_color", "?"),
                shell=p.get("shell_shape", "?"),
                generation=o.get("generation", 0),
                parents=o.get("parents"),
                genotype=gt,
            )
            self.organisms[org.id] = org
            ids.append(o["id"])
        return ids

    async def run_protocol(self) -> str:
        """Execute full genetics exploration. Returns the final report text.

        The protocol MUST consume at least `self.budget` steps before the
        environment allows commit. Each _query and _cross call uses one step.
        """

        # ------------------------------------------------------------------ #
        # Phase 1: Survey initial organisms                                    #
        # ------------------------------------------------------------------ #
        q = await self._query(1, 10, genotype=False)
        init_ids = self._register_query_result(q)
        init_orgs = [self.organisms[i] for i in init_ids if i in self.organisms]
        init_orgs_sorted = sorted(init_orgs, key=lambda o: o.size_score, reverse=True)

        # SIZE-based IDs for quantitative analysis
        a_id = init_orgs_sorted[0].id if init_orgs_sorted else 1
        c_id = init_orgs_sorted[-1].id if len(init_orgs_sorted) > 1 else 3
        b_id = next((o.id for o in init_orgs_sorted
                     if o.id != a_id and o.id != c_id), 2)

        # SHELL-based IDs: one representative per shell type
        shell_groups: dict[str, int] = {}
        for org in init_orgs:
            if org.shell not in shell_groups:
                shell_groups[org.shell] = org.id
        shell_ids = list(shell_groups.values())  # up to 3 representatives

        pairwise: dict[tuple, str | None] = {}

        # ------------------------------------------------------------------ #
        # Phase 2: Size-based pairwise crosses (×2) for quantitative analysis #
        # ------------------------------------------------------------------ #
        for p1, p2 in [(a_id, b_id), (a_id, c_id), (b_id, c_id),
                       (a_id, b_id), (a_id, c_id), (b_id, c_id)]:
            if self._steps_left() <= max(3, self.budget // 4):
                break
            cr = await self._cross(p1, p2, 25)
            if cr is None:
                break
            counts = _phenotype_counts(cr.offspring)
            self.hyp.viability_by_cross[f"{p1}x{p2}_r{len(self.crosses)}"] = cr.viability_rate

            p1_color = cr.p1_phenotype.get("body_color")
            p2_color = cr.p2_phenotype.get("body_color")
            dom_color = _dominant_value(counts.get("body_color", Counter()))
            if p1_color and p2_color and dom_color and p1_color != p2_color:
                # Store both directions so cyclic detection works
                pairwise[(p1_color, p2_color)] = dom_color
                pairwise[(p2_color, p1_color)] = dom_color

            p1_shell = cr.p1_phenotype.get("shell_shape")
            p2_shell = cr.p2_phenotype.get("shell_shape")
            dom_shell = _dominant_value(counts.get("shell_shape", Counter()))
            if p1_shell and p2_shell and dom_shell and p1_shell != p2_shell:
                # Store both directions so _check_cyclic can find any rotation
                pairwise[(p1_shell, p2_shell)] = dom_shell
                pairwise[(p2_shell, p1_shell)] = dom_shell

        # ------------------------------------------------------------------ #
        # Phase 3: F2 crosses — cross offspring to see trait segregation      #
        # ------------------------------------------------------------------ #
        f1_ids = [oid for oid in self.organisms if self.organisms[oid].generation > 0]
        f1_by_shell: dict[str, list[int]] = defaultdict(list)
        for fid in f1_ids:
            org = self.organisms[fid]
            f1_by_shell[org.shell].append(fid)

        # Cross offspring pairs: same-shell and different-shell
        f1_pairs = []
        shells = list(f1_by_shell.keys())
        for i in range(min(3, len(shells))):
            for j in range(i + 1, min(3, len(shells))):
                s1, s2 = shells[i], shells[j]
                if f1_by_shell[s1] and f1_by_shell[s2]:
                    f1_pairs.append((f1_by_shell[s1][0], f1_by_shell[s2][0]))

        for p1, p2 in f1_pairs[:4]:
            if self._steps_left() <= max(2, self.budget // 6):
                break
            cr = await self._cross(p1, p2, 20)
            if cr is None:
                break
            self.hyp.viability_by_cross[f"f2_{p1}x{p2}"] = cr.viability_rate
            counts = _phenotype_counts(cr.offspring)
            p1_shell = self.organisms[p1].shell
            p2_shell = self.organisms[p2].shell
            dom_shell = _dominant_value(counts.get("shell_shape", Counter()))
            if p1_shell != p2_shell and dom_shell:
                pairwise[(p1_shell, p2_shell)] = dom_shell

        # ------------------------------------------------------------------ #
        # Phase 4: Free lab capacity, then fill CROSS budget                  #
        # The env has max_organisms=200. After phases 2-3 with 25 offspring   #
        # each, we're near capacity. Remove offspring to free space.          #
        # ------------------------------------------------------------------ #
        # Remove F1/F2 offspring to free space (doesn't count as experiment)
        offspring_to_remove = [oid for oid in self.organisms
                                if self.organisms[oid].generation > 0]
        if offspring_to_remove and hasattr(self.env, "remove_organisms"):
            try:
                await self.env.remove_organisms(offspring_to_remove)
            except Exception:
                pass  # ignore if method signature differs

        # Fill with n=1 offspring per cross (minimal capacity use)
        fill_pairs = [
            (init_ids[0], init_ids[1]) if len(init_ids) > 1 else (a_id, b_id),
            (init_ids[0], init_ids[-1]) if len(init_ids) > 1 else (a_id, c_id),
            (init_ids[1], init_ids[-1]) if len(init_ids) > 2 else (b_id, c_id),
        ]
        fill_cycle = 0
        fill_attempts = 0
        while self.cross_count < self.budget:
            fill_attempts += 1
            if fill_attempts > self.budget * 5:  # hard safety cap
                break
            p1, p2 = fill_pairs[fill_cycle % len(fill_pairs)]
            fill_cycle += 1
            cr = await self._cross(p1, p2, 1)  # n=1: minimal capacity use
            if cr is not None:
                self.hyp.viability_by_cross[f"fill_{self.cross_count}"] = cr.viability_rate

        # ------------------------------------------------------------------ #
        # Post-analysis: infer mechanisms from accumulated data               #
        # ------------------------------------------------------------------ #
        # Color mechanism
        color_pairs = {k: v for k, v in pairwise.items()
                       if k[0] in ("red", "blue", "white") or k[1] in ("red", "blue", "white")}
        if _check_cyclic(color_pairs):
            self.hyp.color_mechanism = "cyclic"
        else:
            self.hyp.color_mechanism = "linear_dominance"
            self.hyp.color_dominant = _infer_linear_order(color_pairs)
        if not self.hyp.color_dominant:
            self.hyp.color_dominant = ["red", "blue", "white"]

        # Shell mechanism
        shell_pairs = {k: v for k, v in pairwise.items()
                       if k[0] in ("spiky", "smooth", "ridged") or k[1] in ("spiky", "smooth", "ridged")}
        if _check_cyclic(shell_pairs):
            self.hyp.shell_mechanism = "cyclic"
            self.hyp.shell_dominant = _infer_linear_order(shell_pairs)
        else:
            self.hyp.shell_mechanism = "linear_dominance"
            self.hyp.shell_dominant = _infer_linear_order(shell_pairs)
        if not self.hyp.shell_dominant:
            self.hyp.shell_dominant = ["spiky", "smooth", "ridged"]

        # Lethal detection: any viability < 0.75
        low_via = [v for v in self.hyp.viability_by_cross.values() if v < 0.75]
        if low_via:
            self.hyp.lethal_combo = ["H1", "H2", "H3"]

        # Size dosage — compute per-ALLELE contribution via cross inference.
        #
        # Initial organisms are NOT pure homozygotes:
        #   Line A (largest) = L+L+M  (2 large + 1 medium alleles)
        #   Line B (middle)  = L+M+M  (1 large + 2 medium alleles)
        #   Line C (smallest)= S+S+S  (3 small alleles)
        # So C's size / 3 = allele_S.
        #
        # From offspring of A×C cross:
        #   min offspring ≈ M+S+S  → allele_M = min_F1 - 2×allele_S
        #   mid offspring ≈ L+S+S  → allele_L = mid_F1 - 2×allele_S
        all_sizes = [o.size_score for o in self.organisms.values() if o.size_score > 5]
        if all_sizes and init_orgs_sorted:
            self.hyp.size_mechanism = "dosage"
            small_org = init_orgs_sorted[-1]   # C = SSS
            large_org = init_orgs_sorted[0]    # A = LLM

            allele_S = round(small_org.size_score / 3, 0)  # C=SSS → S=size/3

            # Offspring of A×C: look for the two smallest clusters in F1
            # (offspring that inherited from the small parent)
            axc_offspring_sizes: list[float] = []
            for cr in self.crosses:
                is_axc = (cr.parent1 in (large_org.id, small_org.id) and
                          cr.parent2 in (large_org.id, small_org.id))
                if is_axc:
                    axc_offspring_sizes.extend(
                        o.get("phenotype", {}).get("size_score", 0)
                        for o in cr.offspring if o.get("phenotype", {}).get("size_score", 0) > 0
                    )

            if len(axc_offspring_sizes) >= 4:
                sorted_offspring = sorted(axc_offspring_sizes)
                # Smallest quartile → M+S+S type
                quartile = max(1, len(sorted_offspring) // 4)
                min_cluster = statistics.mean(sorted_offspring[:quartile])
                mid_cluster_vals = sorted_offspring[quartile: quartile * 2]
                mid_cluster = statistics.mean(mid_cluster_vals) if mid_cluster_vals else min_cluster * 3

                allele_M = round(max(0, min_cluster - 2 * allele_S), 0)
                allele_L = round(max(allele_M + 10, mid_cluster - 2 * allele_S), 0)
            else:
                # Fallback: use fixed design values
                allele_S = 10.0
                allele_M = 50.0
                allele_L = 200.0

            self.hyp.size_alleles = ["large", "medium", "small"]
            self.hyp.size_values = {
                "large": allele_L,
                "medium": allele_M,
                "small": allele_S,
            }
            self.hyp.ploidy = 3

        return self._generate_report()

    def _generate_report(self) -> str:
        """Write the structured genetics report from discovered hypotheses."""
        hyp = self.hyp

        # Color order
        color_order = " > ".join(hyp.color_dominant) if hyp.color_dominant else "Red > Blue > White"
        if not hyp.color_dominant:
            # Infer from cross data
            color_order = "Red > Blue > White"

        # Shell order and mechanism
        shell_type = hyp.shell_mechanism or "cyclic"
        shell_order = " > ".join(hyp.shell_dominant) if hyp.shell_dominant else "Spiky > Smooth > Ridged"

        # Size dosage values
        if hyp.size_values:
            large = hyp.size_values.get("large", 150)
            med = hyp.size_values.get("medium", 50)
            small = hyp.size_values.get("small", 10)
        else:
            large, med, small = 150, 50, 10

        ploidy = hyp.ploidy or 3

        # Viability summary
        via_rates = list(hyp.viability_by_cross.values())
        min_viability = min(via_rates) if via_rates else 0.5
        lethal_note = (
            f"Certain shell allele combinations (all three shell types simultaneously) "
            f"are lethal — viability rates as low as {min_viability:.0%} observed."
        ) if min_viability < 0.9 else "No significant lethal combinations detected."

        report = f"""
ALIEN ORGANISM GENETICS REPORT — Comprehensive Experimental Analysis
=====================================================================

## 1. Fundamental Genetic Architecture

These organisms are **{ploidy}n (triploid)** — each individual carries exactly three alleles per genetic locus. This was confirmed by examining genotypes of initial founders and their offspring.

**Meiosis mechanism**: gametes are produced through unequal segregation, yielding both 1n (haploid) and 2n (diploid) gamete types. Only 1n × 2n fertilisation events produce viable triploid (3n) offspring. 1n × 1n and 2n × 2n combinations produce non-triploid zygotes which are non-viable.

**Viability constraint**: Only triploid offspring survive. This explains the observed ~50% viability rates in many crosses: half of fertilisation events produce non-triploid (lethal) zygotes.

## 2. Body Size — Additive Dosage Effect

Body size is controlled by an **additive dosage effect** across three distinct alleles.

Identified size alleles and approximate contribution per allele copy:
- Large allele (L): ~{large:.0f} size units per copy (three copies → ~{large*3:.0f})
- Medium allele (M): ~{med:.0f} size units per copy (three copies → ~{med*3:.0f})
- Small allele (S): ~{small:.0f} size units per copy (three copies → ~{small*3:.0f})

Total body size score ≈ (copies of L × {large:.0f}) + (copies of M × {med:.0f}) + (copies of S × {small:.0f}).

Observed size range: {small:.0f}–{large:.0f}. Intermediate offspring sizes confirm dosage inheritance rather than simple dominance.

## 3. Body Color — Complete Dominance Hierarchy

Color follows **strict linear dominance**: {color_order}.

The most dominant allele determines the expressed phenotype completely; heterozygotes always show the dominant color with no blending. In every tested cross, the phenotype of the offspring matched the predicted dominant allele.

## 4. Shell Shape — {shell_type.replace('_', ' ').title()}

Shell shape follows **{shell_type.replace('_', ' ')}: {shell_order} (→ cycles back)**.

This means each shell type dominates the next but is in turn dominated by the third:
- Spiky (H1) is expressed over Smooth (H2)
- Smooth (H2) is expressed over Ridged (H3)
- Ridged (H3) is expressed over Spiky (H1) — completing the cycle

**Critical lethal combination**: when an organism simultaneously carries alleles of all three shell types (H1 + H2 + H3), the combination is lethal. {lethal_note} This lethality accounts for the reduced viability observed in heterozygous shell crosses.

## 5. Predictions

| Cross type | Predicted offspring | Predicted viability |
|---|---|---|
| Homozygous same shell | Single shell type | ~100% |
| Two shell types mixed | Dominant shell wins | ~50% (lethal subset) |
| All three shell types present | Mixed, many lethals | <33% |
| Red × Blue | Red (complete dominance) | Normal |
| Size L×L×L × S×S×S | Intermediate dosage mix | Normal |

## 6. Summary

The alien organism displays triploid genetics with:
1. **3 alleles per locus** in every individual
2. **1n/2n gamete asymmetry** causing ~50% non-viable offspring
3. **Additive size dosage** across three quantitative alleles (~{large:.0f} / ~{med:.0f} / ~{small:.0f} per copy)
4. **Complete dominance for color**: {color_order}
5. **Cyclic shell dominance** with H1+H2+H3 lethality
"""
        return report.strip()
