"""
CodeEvolver — within-episode cognitive module synthesizer.

The MARS-SELF / Cognitive Exoskeleton: a gpt-4o-mini agent that generates
Python analysis functions during the episode.  Each function is sandboxed,
tested, and registered in a live library.  On every subsequent turn, all
registered functions run against the new observation result and inject
findings as synthetic Claims (prefix "[CE:fn_name]").

Architecture (three module types):
  Analytical  — parse_cross_result, allele_freq_from_offspring, ratio_3class
  Detector    — detect_dominance, detect_cyclic, lethal_combo_test
  Reasoner    — gap_detector, score_hypothesis, phase_completion_check

Recursive composition: later-generated modules may reference findings
from earlier ones via the ClaimStore (they share the same "[CE:...]"
prefix so MemorySelector surfaces them together).

Hypothesis H_SELF: gpt-4o-mini + MARS-SELF ≈ gpt-4o without MARS-SELF
on long-horizon scientific discovery (UltraHorizon Bio, NewtonBench).

Integration:
    evolver = CodeEvolver(model="openai/gpt-4o-mini", max_modules=10)
    # Coordinator calls after each action:
    findings = evolver.process_observation(obs_result, sub_goal)
    # Each finding → store.assert_claim(statement=finding, confidence=0.75, ...)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from mars.agents.base import call_llm, make_openai_client, parse_json_strict


# ── safe execution sandbox ────────────────────────────────────────────────────

_SAFE_BUILTINS: dict = {
    "abs": abs, "all": all, "any": any, "bool": bool,
    "dict": dict, "enumerate": enumerate, "filter": filter,
    "float": float, "frozenset": frozenset,
    "hasattr": hasattr, "int": int, "isinstance": isinstance,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "print": print, "range": range, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "type": type, "zip": zip,
    "None": None, "True": True, "False": False,
    # block dangerous builtins
    "__import__": None, "eval": None, "exec": None, "compile": None,
    "open": None, "input": None, "breakpoint": None,
}

_SAFE_GLOBALS: dict = {
    "__builtins__": _SAFE_BUILTINS,
    "math": math,
    "re": re,
    "json": json,
}


def _sandbox_exec(code: str, fn_name: str) -> "callable | None":
    """Execute code in restricted namespace. Return function or None."""
    ns: dict = dict(_SAFE_GLOBALS)
    try:
        exec(compile(code, "<ce_module>", "exec"), ns)  # noqa: S102
        fn = ns.get(fn_name)
        if callable(fn):
            return fn
    except Exception:
        pass
    return None


def _sanitize_fn_name(raw: str) -> str:
    """Turn an arbitrary LLM-produced string into a valid Python identifier."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", str(raw))[:48].rstrip("_")
    if not name or not name[0].isalpha():
        name = "ce_" + name
    return name or "ce_fn"


# ── gap detection prompts ─────────────────────────────────────────────────────

_GAP_SYS = (
    "You are the CodeEvolver gap detector for a triploid-alien-genetics agent.\n"
    "Given one observation result and a sub-goal label, decide whether a\n"
    "NEW specialized Python analysis function would extract additional,\n"
    "non-trivial insight beyond what is already visible in the raw data.\n\n"
    "Gap types (pick exactly one):\n"
    "  ALLELE_FREQ   — compute allele frequencies from offspring phenotype counts\n"
    "  DOMINANCE     — test if one phenotype dominates (>50 % of offspring)\n"
    "  LETHAL        — detect missing expected phenotype classes (lethal combos)\n"
    "  CYCLIC        — detect A>B>C>A cyclic dominance from multiple crosses\n"
    "  VIABILITY     — offspring count vs expected ratio (1:1, 2:1, 3:1, etc.)\n"
    "  SIZE_ADDITIVE — decompose body size into additive allele contributions\n"
    "  RATIO_CHECK   — simple ratio of offspring classes vs Mendelian expectation\n"
    "  OTHER         — any other structured quantitative computation\n\n"
    "Return EXACTLY one JSON object:\n"
    '{"has_gap": true|false,\n'
    ' "gap_type": "<one of the types above>",\n'
    ' "fn_name": "<snake_case_function_name>",\n'
    ' "description": "<one sentence: what this function should compute>"}\n\n'
    "Return has_gap=false when:\n"
    "  - result is text-only (no numeric phenotype data)\n"
    "  - an existing module already covers this computation\n"
    "  - the function would trivially duplicate a raw result field\n"
    "IMPORTANT: fn_name must be unique — do NOT reuse names in 'existing_modules'."
)

_GAP_USER_TMPL = (
    "Sub-goal: {sub_goal}\n\n"
    "Observation result (truncated):\n{obs_result}\n\n"
    "NOTE — obs['summary']['observation'] is a JSON string, not a plain dict.\n"
    "To reach the cross data a generated function must do:\n"
    "  raw = obs.get('summary', {{}}).get('observation', '{{}}')\n"
    "  data = json.loads(raw) if isinstance(raw, str) else raw\n"
    "Key fields in `data`: success, parent1_phenotype, parent2_phenotype,\n"
    "  viable_offspring_count, lethal_offspring_count, viability_rate,\n"
    "  offspring (list of {{id, phenotype: {{body_size, size_score, body_color, shell_shape}}}})\n\n"
    "Existing modules (do NOT duplicate): {existing}\n\n"
    "Is a new specialized analysis function warranted? Reply JSON."
)


# ── module generation prompts ─────────────────────────────────────────────────

_GEN_SYS = (
    "You are CodeEvolver, a Python code synthesizer for a genetics-discovery agent.\n"
    "Generate ONE Python function that analyzes a genetics experiment observation.\n\n"
    "STRICT RULES:\n"
    "  1. Function signature: def {fn_name}(obs: dict) -> str | None\n"
    "  2. The obs dict structure is:\n"
    "       obs['eid']     = experiment id (int)\n"
    "       obs['action']  = 'conduct_cross'\n"
    "       obs['args']    = {'parent1_id': int, 'parent2_id': int}\n"
    "       obs['cost']    = 1.0\n"
    "       obs['summary'] = {'observation': <json_string>}\n"
    "     To extract the cross result, ALWAYS use this exact pattern at the top:\n"
    "       raw = obs.get('summary', {}).get('observation', '{}')\n"
    "       data = json.loads(raw) if isinstance(raw, str) else raw\n"
    "     Then access fields on `data`:\n"
    "       data.get('offspring', [])            # list of offspring dicts\n"
    "       data.get('viable_offspring_count')   # int\n"
    "       data.get('lethal_offspring_count')   # int\n"
    "       data.get('viability_rate')           # float\n"
    "       data.get('parent1_phenotype')        # dict with body_size, body_color, shell_shape\n"
    "       data.get('parent2_phenotype')        # dict with body_size, body_color, shell_shape\n"
    "     Each offspring: {'id': int, 'phenotype': {'body_size': str, 'size_score': int,\n"
    "                       'body_color': str, 'shell_shape': str}}\n"
    "     NOTE: `json` is already in scope — do NOT write `import json`\n"
    "  3. Return a SHORT descriptive string (≤150 chars) with the finding, or None\n"
    "  4. Use ONLY built-ins + math, re, json (already imported — no `import` needed)\n"
    "  5. NO import statements — they are BLOCKED and will raise TypeError\n"
    "  6. NO eval(), exec(), open(), input() — BLOCKED\n"
    "  7. Wrap ALL logic in try/except BaseException and return None on any error\n"
    "  8. Be defensive: check all dict keys with .get(); cast to float carefully\n\n"
    "Output ONLY raw Python source — no markdown fences, no docstrings required."
)

_GEN_USER_TMPL = (
    "Gap type: {gap_type}\n"
    "Function name: {fn_name}\n"
    "Task: {description}\n\n"
    "Sample observation (for key inspection):\n{obs_sample}\n\n"
    "Write `{fn_name}(obs)` now:"
)


# ── CodeEvolver ────────────────────────────────────────────────────────────────

@dataclass
class CodeEvolver:
    """
    Within-episode cognitive module synthesizer (MARS-SELF).

    Usage::

        evolver = CodeEvolver(model="openai/gpt-4o-mini", max_modules=10)

        # inside Coordinator, after each action result:
        findings = evolver.process_observation(obs_result, cur_sg)
        for f in findings:
            store.assert_claim(statement=f, confidence=0.75, ...)
    """

    model: str = "openai/gpt-4o-mini"
    max_modules: int = 6           # cap on auto-generated functions
    max_retries: int = 2           # code-gen retries per gap
    gap_check_interval: int = 4    # check for new gap every N observations
    max_gap_checks: int = 8        # hard cap on LLM gap-detection calls / episode
    max_findings_per_obs: int = 3  # cap findings asserted per observation

    # populated during episode (not constructor args — use field())
    library: dict = field(default_factory=dict)          # fn_name → callable
    module_docs: dict = field(default_factory=dict)      # fn_name → source snippet
    failed_attempts: set = field(default_factory=set)    # fn_names that failed
    _seen_findings: set = field(default_factory=set)     # dedupe emitted findings
    _stats_done: bool = False                            # AutoStatAnalyzer ran?

    # counters
    n_obs_processed: int = 0
    n_modules_generated: int = 0
    n_gap_checks: int = 0
    n_findings_total: int = 0

    def __post_init__(self) -> None:
        self._client = make_openai_client()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _detect_gap(self, obs_result: dict, sub_goal: str) -> "dict | None":
        """Ask LLM whether a new module is warranted. Returns gap dict or None."""
        existing = list(self.library) or ["(none yet)"]
        obs_str = json.dumps(obs_result, default=str)[:700]
        user = _GAP_USER_TMPL.format(
            sub_goal=sub_goal,
            obs_result=obs_str,
            existing=", ".join(existing),
        )
        try:
            raw = call_llm(
                self._client, self.model, _GAP_SYS, user,
                max_tokens=160, temperature=0.0,
            )
            obj = parse_json_strict(raw) or {}
            if obj.get("has_gap") and obj.get("fn_name"):
                return obj
        except Exception:
            pass
        return None

    def _generate_module(
        self, gap: dict, fn_name: str, obs_result: dict
    ) -> "str | None":
        """Generate Python source for `fn_name`. Returns code string or None."""
        obs_sample = json.dumps(obs_result, default=str)[:450]
        user = _GEN_USER_TMPL.format(
            gap_type=gap.get("gap_type", "OTHER"),
            fn_name=fn_name,
            description=gap.get("description", "analyze the observation"),
            obs_sample=obs_sample,
        )
        try:
            code = call_llm(
                self._client, self.model,
                _GEN_SYS.replace("{fn_name}", fn_name),
                user,
                max_tokens=520, temperature=0.2,
            )
            # strip markdown fences if LLM wrapped them
            code = re.sub(r"^```(?:python)?\s*\n?", "", code.strip())
            code = re.sub(r"\n?```\s*$", "", code.strip())
            return code.strip() or None
        except Exception:
            return None

    def _test_and_register(
        self, fn_name: str, code: str, obs_result: dict
    ) -> bool:
        """Sandbox-exec code; smoke-test against obs_result. Register on success."""
        fn = _sandbox_exec(code, fn_name)
        if fn is None:
            return False
        # smoke test: must not throw, must return str/None/scalar
        try:
            out = fn(obs_result)
            if out is not None and not isinstance(out, (str, int, float, bool)):
                return False
        except Exception:
            return False
        self.library[fn_name] = fn
        self.module_docs[fn_name] = code[:250]
        self.n_modules_generated += 1
        return True

    def _run_library(self, obs_result: dict) -> list[str]:
        """Run every registered module on obs_result; return NOVEL findings only.

        Deduplicates against previously-emitted findings (a module run on the
        same data produces the same string — we don't want to flood the claim
        store) and caps the number of new findings per observation.
        """
        findings: list[str] = []
        for fn_name, fn in list(self.library.items()):
            if len(findings) >= self.max_findings_per_obs:
                break
            try:
                out = fn(obs_result)
                if out and str(out).strip().lower() not in ("none", ""):
                    finding = f"[CE:{fn_name}] {str(out)[:150]}"
                    if finding not in self._seen_findings:
                        self._seen_findings.add(finding)
                        findings.append(finding)
            except Exception:
                pass
        return findings

    # ── public API ────────────────────────────────────────────────────────────

    def process_observation(
        self, obs_result: dict, sub_goal: str
    ) -> list[str]:
        """
        Main Coordinator hook — call after every action result.

        1. Run existing modules → collect findings.
        2. Every ``gap_check_interval`` observations: detect gap → generate +
           register a new module → immediately run it on current obs.

        Returns list of finding strings to be asserted as Claims (confidence=0.75).
        """
        self.n_obs_processed += 1
        findings: list[str] = []

        # --- step 1: run registered library ---
        findings.extend(self._run_library(obs_result))

        # --- step 2: maybe grow the library ---
        # Throttled: only while under both the module cap AND the per-episode
        # gap-check (LLM-call) budget, and only on the check schedule.
        can_grow = (len(self.library) < self.max_modules
                    and self.n_gap_checks < self.max_gap_checks)
        on_schedule = (self.n_obs_processed % self.gap_check_interval == 0)

        if can_grow and on_schedule:
            self.n_gap_checks += 1
            gap = self._detect_gap(obs_result, sub_goal)

            if gap:
                fn_name = _sanitize_fn_name(gap.get("fn_name", "analysis_fn"))

                if fn_name not in self.library and fn_name not in self.failed_attempts:
                    generated = False
                    for _ in range(self.max_retries):
                        code = self._generate_module(gap, fn_name, obs_result)
                        if code and self._test_and_register(fn_name, code, obs_result):
                            # immediately run new module on current data
                            try:
                                out = self.library[fn_name](obs_result)
                                if out and str(out).strip():
                                    finding = f"[CE:{fn_name}] {str(out)[:150]}"
                                    if finding not in self._seen_findings:
                                        self._seen_findings.add(finding)
                                        findings.append(finding)
                            except Exception:
                                pass
                            generated = True
                            break

                    if not generated:
                        self.failed_attempts.add(fn_name)

        self.n_findings_total += len(findings)
        return findings

    def check_report_coverage(
        self, claims: list, budget_remaining: float,
        current_subgoal: str | None = None,
        rubric: list | None = None,
        subgoal_order: list | None = None,
        adapter=None,
    ) -> list[str]:
        """
        Domain-agnostic Programmatic Completeness Gate (PCG).

        Given a `rubric` (list of RubricSection declared by the adapter) and the
        ordered list of sub-goal ids, return a "[CE:coverage_guard] ..." warning
        when any REQUIRED section is still undiscovered.  Empty rubric → no-op.

        A section is required when:
          - current_subgoal is None (terminal-submit check) → ALL sections, OR
          - the episode has reached section.required_from_subgoal by order.

        A section is covered when ANY of its keywords appears in active-claim
        text, OR its `predicate(ctx)` returns True.

        Only fires when budget_remaining > 3 (too late to act otherwise).
        """
        if budget_remaining <= 3 or not rubric:
            return []

        subgoal_order = subgoal_order or []
        # index of current subgoal for ordering comparisons
        if current_subgoal is None:
            cur_idx = len(subgoal_order)        # terminal: past everything
        elif current_subgoal in subgoal_order:
            cur_idx = subgoal_order.index(current_subgoal)
        else:
            cur_idx = len(subgoal_order)        # unknown → strict

        all_text = " ".join(
            getattr(c, "statement", str(c)).lower()
            for c in claims
            if getattr(c, "status", "active") == "active"
        )
        ctx = {
            "claims_text": all_text,
            "adapter": adapter,
            "budget_left": budget_remaining,
            "current_subgoal": current_subgoal,
        }

        missing: list[str] = []
        for sec in rubric:
            # Is this section required at the current point in the episode?
            req_from = getattr(sec, "required_from_subgoal", None)
            if current_subgoal is not None and req_from is not None:
                req_idx = (subgoal_order.index(req_from)
                           if req_from in subgoal_order else 0)
                if cur_idx < req_idx:
                    continue                      # not required yet
            # Covered?
            covered = False
            kws = getattr(sec, "keywords", []) or []
            if kws and any(kw.lower() in all_text for kw in kws):
                covered = True
            pred = getattr(sec, "predicate", None)
            if not covered and callable(pred):
                try:
                    covered = bool(pred(ctx))
                except Exception:
                    covered = False
            if not covered:
                hint = getattr(sec, "hint", "") or getattr(sec, "name", "section")
                missing.append(hint)

        if missing:
            bullet = "; ".join(missing)
            return [f"[CE:coverage_guard] INCOMPLETE — still must establish: {bullet[:300]}"]
        return []

    def scaffold_data_analysis(
        self, adapter, query_text: str = "",
    ) -> list[str]:
        """
        AutoStatAnalyzer (MARS-SELF) — auto-run statistics the small model
        struggles to hand-write.

        If the adapter exposes tabular data via `get_dataframes()`, compute the
        strongest pairwise correlations among numeric columns (prioritising
        columns whose names appear in the discovery query) and return them as
        `[CE:stat]` finding strings. Runs ONCE per episode (cached).

        This is the exoskeleton component for DiscoveryBench: it grounds the
        agent's hypothesis in real coefficients rather than relying on the
        small model to author correct pandas/regression code itself.
        """
        if self._stats_done:
            return []
        get_dfs = getattr(adapter, "get_dataframes", None)
        if not callable(get_dfs):
            return []
        try:
            dfs = get_dfs()
        except Exception:
            return []
        if not dfs:
            return []
        self._stats_done = True

        try:
            import pandas as pd
        except Exception:
            return []

        # --- query-relevance: which columns does the discovery query target? ---
        # Match query tokens against each column's name AND its natural-language
        # description (column names are often opaque codes like BAMM_speciation,
        # so the description is what actually connects to the query).
        q_text = (query_text or "")
        try:
            gq = getattr(adapter, "get_query_text", None)
            if callable(gq):
                q_text = (gq() or "") + " " + q_text
        except Exception:
            pass
        q_tokens = {t.lower() for t in re.findall(r"[A-Za-z]\w{2,}", q_text)}
        # drop ultra-common words that would match everything
        _STOP = {"the", "and", "for", "with", "this", "that", "are", "was",
                 "between", "does", "how", "what", "which", "value", "values",
                 "data", "dataset", "variable", "variables", "relationship",
                 "discovery", "query", "hypothesis", "task", "domain"}
        q_tokens -= _STOP

        col_desc: dict = {}
        try:
            gcd = getattr(adapter, "get_column_descriptions", None)
            if callable(gcd):
                col_desc = gcd() or {}
        except Exception:
            col_desc = {}

        def _relevance(col: str) -> int:
            """How strongly does this column connect to the query?"""
            score = 0
            cl = col.lower()
            if cl in q_tokens:
                score += 2
            # token overlap between column name parts and query
            for part in re.split(r"[_\W]+", cl):
                if len(part) > 2 and part in q_tokens:
                    score += 1
            # description overlap
            desc = col_desc.get(col, "").lower()
            d_tokens = {t for t in re.findall(r"[A-Za-z]\w{2,}", desc)} - _STOP
            score += len(d_tokens & q_tokens)
            return score

        findings: list[str] = []
        for name, df in dfs.items():
            try:
                num = df.select_dtypes(include="number")
                if num.shape[1] < 2 or num.shape[0] < 3:
                    continue
                corr = num.corr(numeric_only=True)
                cols = list(corr.columns)
                rel = {c: _relevance(c) for c in cols}
                any_relevant = any(v > 0 for v in rel.values())
                pairs: list[tuple] = []
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        r = corr.iloc[i, j]
                        if pd.notna(r):
                            pair_rel = rel[cols[i]] + rel[cols[j]]
                            pairs.append((pair_rel, abs(float(r)),
                                          cols[i], cols[j], float(r)))
                # If the query maps to specific columns, surface ONLY pairs that
                # involve a relevant column (avoids flooding with redundant
                # high-correlation derived metrics). Otherwise fall back to top|r|.
                if any_relevant:
                    pairs = [p for p in pairs if p[0] > 0]
                pairs.sort(key=lambda t: (-t[0], -t[1]))
                n_rows = int(num.shape[0])
                for pair_rel, absr, a, b, r in pairs[:3]:
                    if absr >= 0.20:
                        direction = ("positive" if r > 0 else "negative")
                        finding = (f"[CE:stat] corr({a}, {b}) = {r:+.3f} "
                                   f"({direction}, n={n_rows})")
                        if finding not in self._seen_findings:
                            self._seen_findings.add(finding)
                            findings.append(finding)
            except Exception:
                continue

        self.n_findings_total += len(findings)
        return findings[:4]

    def stats(self) -> dict:
        return {
            "n_modules": len(self.library),
            "module_names": list(self.library),
            "n_gap_checks": self.n_gap_checks,
            "n_obs_processed": self.n_obs_processed,
            "n_findings_total": self.n_findings_total,
            "n_modules_generated": self.n_modules_generated,
            "stats_done": self._stats_done,
            "failed_attempts": sorted(self.failed_attempts),
        }

    def library_summary(self) -> str:
        if not self.library:
            return "(no modules generated yet)"
        parts = [f"{n}: {doc[:55]}..." for n, doc in self.module_docs.items()]
        return "; ".join(parts)
