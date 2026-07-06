"""
KGMemorySelector — Knowledge-Graph-based replacement for MemorySelector.

Instead of scoring claims by token-overlap + recency, this module:
  1. Extracts structured triplets (subject, relation, object) from each new
     claim via a single batched LLM call.
  2. Stores them in a TripletGraph (adapted from kg4code/graph-dev).
  3. At retrieval time, runs BFS from entities found in the current sub-goal
     question and returns the k most-relevant triplets as synthetic Claims.

Triplet format (kg4code convention):
    [subject_str, object_str, {"label": relation_str}]
    e.g. ["S1", "200_units", {"label": "contributes_size"}]

Drop-in interface:
    select(claims, sub_goal_question, budget_total, current_sub_goal) -> list[Claim]
    render_for_log(picked) -> str

Ablation flag: set use_kg_memory=True in Coordinator (added in runner).
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from mars.agents.base import call_llm, make_openai_client, parse_json_strict
from ols.core.types import Claim


# ── lightweight TripletGraph (core subset from kg4code/graphs/parent_graph.py) ──

class TripletGraph:
    """
    In-memory knowledge graph of [subject, object, {label: relation}] triplets.
    Adapted from kg4code graph-dev (Aloriosa/kg4code, branch graph-dev).
    """

    def __init__(self):
        self.triplets: list[list] = []
        self.items: list[str] = []

    def _str(self, triplet) -> str:
        return f"{triplet[0]}, {triplet[2]['label']}, {triplet[1]}"

    def _clear(self, triplet):
        return [str(triplet[0]).lower().strip(),
                str(triplet[1]).lower().strip(),
                {"label": str(triplet[2].get("label", "")).lower().strip()}]

    def add_triplets(self, triplets: list) -> None:
        for t in triplets:
            t = self._clear(t)
            if t not in self.triplets:
                self.triplets.append(t)
            if t[0] not in self.items:
                self.items.append(t[0])
            if t[1] not in self.items:
                self.items.append(t[1])

    def get_all_triplets_str(self) -> list[str]:
        return [self._str(t) for t in self.triplets]

    def get_associated_triplets(self, items: list[str], steps: int = 2) -> list[str]:
        """BFS: starting from items, walk steps hops and collect all triplets."""
        frontier = {s.lower().strip() for s in items}
        seen_triplets: list[str] = []

        for _ in range(steps):
            next_frontier: set[str] = set()
            for t in self.triplets:
                for entity in frontier:
                    if entity == t[0] or entity == t[1]:
                        t_str = self._str(t)
                        if t_str not in seen_triplets:
                            seen_triplets.append(t_str)
                        next_frontier.add(t[1] if entity == t[0] else t[0])
                        break
            frontier = next_frontier - {s for s in next_frontier if s == "itself"}

        return seen_triplets

    def size(self) -> int:
        return len(self.triplets)


# ── extraction prompt ──────────────────────────────────────────────────────────

_EXTRACT_SYS = (
    "You extract knowledge-graph triplets from scientific observations. "
    "Return ONLY valid JSON, no markdown.\n"
    "Format: {\"triplets\": [[\"subject\", \"object\", \"relation\"], ...]}\n"
    "Rules:\n"
    "- Use short, stable entity names (e.g. S1, H1H2, triploidy, meiosis)\n"
    "- Relations: contributes_size, dominant_over, cyclic_dominates, is_lethal, "
    "  produces_gamete, viable_ploidy, allele_value, has_phenotype, causes\n"
    "- Max 5 triplets per observation\n"
    "- If nothing structured can be extracted, return {\"triplets\": []}"
)

_EXTRACT_USER_TMPL = (
    "Extract triplets from these observations (one per numbered line):\n\n"
    "{observations}\n\n"
    "Return JSON: {{\"triplets\": [...]}}"
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*")

# Known genetics entities for entity extraction from sub-goal text
_GENETICS_ENTITIES = {
    "s1", "s2", "s3",
    "c1", "c2", "c3",
    "h1", "h2", "h3",
    "triploidy", "triploid", "meiosis", "gamete", "viability",
    "body_size", "color", "shell",
    "additive", "dominance", "lethal", "cyclic",
    "allele", "locus", "ploidy",
}


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(s or "") if len(t) > 1}


# ── KGMemorySelector ──────────────────────────────────────────────────────────

@dataclass
class KGMemorySelector:
    """
    Drop-in replacement for MemorySelector using a TripletGraph backend.

    Usage in run_uh_bio.py:
        from mars.agents.kg_memory_selector import KGMemorySelector
        sel = KGMemorySelector(k=8)
        coord = Coordinator(..., memory_selector=sel, ...)
    """
    k: int = 8
    model: str = "openai/gpt-4o-mini"
    bfs_steps: int = 2
    # fallback weights when graph is empty
    w_recency: float = 0.4
    w_overlap: float = 0.4
    w_confidence: float = 0.2

    def __post_init__(self):
        self.graph = TripletGraph()
        self._client = make_openai_client()
        # Track which claim objects have been processed (by id())
        self._processed_ids: set[int] = set()
        # Cache: claim statement -> extracted triplets (avoid re-extracting)
        self._extraction_cache: dict[str, list] = {}

    # ── core: sync new claims to graph ────────────────────────────────────────

    def _batch_extract(self, statements: list[str]) -> dict[str, list]:
        """
        Single LLM call to extract triplets from a batch of statements.
        Returns {statement: [triplets]} mapping.
        """
        if not statements:
            return {}

        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(statements))
        prompt = _EXTRACT_USER_TMPL.format(observations=numbered)

        try:
            raw = call_llm(
                self._client, self.model,
                system=_EXTRACT_SYS,
                user=prompt,
                max_tokens=400,
                temperature=0.0,
            )
            obj = parse_json_strict(raw) or {}
            all_triplets = obj.get("triplets", [])
        except Exception:
            all_triplets = []

        # The LLM returns a flat list for all statements combined — that's fine,
        # we add them all to the graph regardless of which statement they came from.
        valid = []
        for t in all_triplets:
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                valid.append([str(t[0]), str(t[1]), {"label": str(t[2])}])

        # Return all triplets mapped to first statement for cache purposes
        result = {s: [] for s in statements}
        if statements:
            result[statements[0]] = valid  # all go in under first key
        return result

    def _sync(self, claims: list[Claim]) -> None:
        """Incrementally add new active claims to the graph."""
        new_statements: list[str] = []
        new_claim_ids: list[int] = []

        for c in claims:
            cid = id(c)
            if cid in self._processed_ids:
                continue
            if c.status != "active":
                self._processed_ids.add(cid)
                continue
            if c.statement in self._extraction_cache:
                # Already extracted — add cached triplets and mark done
                self.graph.add_triplets(self._extraction_cache[c.statement])
                self._processed_ids.add(cid)
                continue
            new_statements.append(c.statement)
            new_claim_ids.append(cid)

        if not new_statements:
            return

        # Batch extract
        result_map = self._batch_extract(new_statements)
        # All triplets come back under statements[0] key
        all_new_triplets = result_map.get(new_statements[0], []) if new_statements else []
        self.graph.add_triplets(all_new_triplets)

        # Cache and mark processed
        for stmt, cid in zip(new_statements, new_claim_ids):
            self._extraction_cache[stmt] = all_new_triplets  # shared ref — ok
            self._processed_ids.add(cid)

    # ── entity extraction from sub-goal text ──────────────────────────────────

    def _extract_entities(self, text: str) -> list[str]:
        """
        Extract query entities from sub-goal text.
        Combines known genetics domain entities with generic short tokens.
        """
        toks = _tokens(text)
        # Priority: known domain entities first
        known = [t for t in toks if t in _GENETICS_ENTITIES]
        # Then short tokens that look like allele names (S1, C2, H3, etc.)
        allele_like = [t for t in toks if re.match(r'^[a-z]\d$', t)]
        # Generic: 3-8 char tokens likely to be entities
        generic = [t for t in toks if 3 <= len(t) <= 8 and t not in known]
        combined = list(dict.fromkeys(known + allele_like + generic))
        return combined[:15]

    # ── fallback: token-overlap scoring (when graph empty) ────────────────────

    def _fallback_select(
        self,
        claims: list[Claim],
        sub_goal_question: str,
        budget_total: float,
        current_sub_goal: str | None,
    ) -> list[Claim]:
        active = [c for c in claims if c.status == "active"]
        if not active:
            return []
        q_tokens = _tokens(sub_goal_question)
        B = max(1.0, budget_total)
        scored: list[tuple[float, Claim]] = []
        for c in active:
            recency = math.exp(-max(0.0, (B - c.budget_stamp)) / B)
            c_tokens = _tokens(c.statement)
            overlap = (len(q_tokens & c_tokens) / max(1, len(q_tokens))
                       if q_tokens else 0.0)
            bonus = 0.05 if current_sub_goal and c.sub_goal == current_sub_goal else 0.0
            score = (self.w_recency * recency
                     + self.w_overlap * overlap
                     + self.w_confidence * c.confidence
                     + bonus)
            scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[: self.k]]

    # ── public interface (same as MemorySelector) ─────────────────────────────

    def select(
        self,
        claims: list[Claim],
        sub_goal_question: str,
        budget_total: float,
        current_sub_goal: str | None = None,
    ) -> list[Claim]:
        # 1. Sync new claims to graph
        self._sync(claims)

        # 2. If graph is empty, fall back to token-overlap
        if self.graph.size() == 0:
            return self._fallback_select(claims, sub_goal_question,
                                         budget_total, current_sub_goal)

        # 3. Extract entities from sub-goal
        entities = self._extract_entities(sub_goal_question)

        # 4. BFS retrieval
        relevant = self.graph.get_associated_triplets(entities, steps=self.bfs_steps)

        if not relevant:
            # BFS found nothing — fall back
            return self._fallback_select(claims, sub_goal_question,
                                         budget_total, current_sub_goal)

        # 5. Deduplicate and cap at k
        seen: set[str] = set()
        unique: list[str] = []
        for t_str in relevant:
            if t_str not in seen:
                seen.add(t_str)
                unique.append(t_str)
            if len(unique) >= self.k:
                break

        # 6. Wrap as synthetic Claim objects (statement = triplet string)
        synthetic: list[Claim] = []
        for t_str in unique:
            synthetic.append(Claim(
                statement=f"[KG] {t_str}",
                confidence=0.85,
                provenance=[],
                budget_stamp=budget_total,
                sub_goal=current_sub_goal,
                status="active",
            ))
        return synthetic

    def render_for_log(self, picked: list[Claim]) -> str:
        if not picked:
            return "(empty)"
        return "; ".join(c.statement[:80] for c in picked)

    def stats(self) -> dict:
        return {
            "graph_triplets": self.graph.size(),
            "graph_entities": len(self.graph.items),
            "processed_claims": len(self._processed_ids),
            "cache_size": len(self._extraction_cache),
        }
