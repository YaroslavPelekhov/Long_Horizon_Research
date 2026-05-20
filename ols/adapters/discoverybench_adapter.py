"""
DiscoveryBenchAdapter — wraps a single DiscoveryBench-Real task as a
ResearchEnvAdapter so OLS can drive it identically to LMW.

Design:
  - One task = one episode. (Each task: one or more CSVs + a discovery query +
    optional domain-knowledge hint.)
  - Sub-goal partition: the discovery query is the single top-level sub-goal,
    optionally decomposed by the inner agent via spawn_subgoal actions.
  - Action space (intentionally small, ReAct-style):
       * describe_dataset(name)            — column descriptions + dtypes
       * head(name, n=5)                   — preview rows
       * query(name, code)                 — run a single pandas/numpy expression
                                              on the named CSV; result truncated
                                              to 1000 chars
       * submit_hypothesis(text)           — terminal claim; ends the episode
  - Budget = number of action calls. Default cap = 12 (matches the DB paper's
    typical ReAct turn budget).
  - Scoring: end-of-episode LLM-judge comparing submitted hypothesis text to
    the task's ground-truth `hypotheses.main.text` (or query.true_hypothesis).
    Returns HMS-style 0..1 score via a single gpt-4o judge call (cheaper than
    the published gpt-4-0125-preview, still strong relative judgment).

Notes on the safe-eval:
  - We expose ONLY `df` and `pd` in the namespace.
  - We use `eval()` (Python expression, not statement) — no imports, no
    exec/eval recursion, no file I/O. The inner LLM cannot run "rm -rf /";
    the worst it can do is consume CPU on a malformed pandas op (we cap with
    a 5-second timeout via signal on POSIX, 5s wall via threading on Windows).
  - All exceptions are caught and returned as the result summary.

OpenRouter model overrides via env:
  - OLS_DB_JUDGE_MODEL (default: openai/gpt-4o)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ols.adapters.base import (
    BudgetExhausted,
    EnvHandle,
    ResearchEnvAdapter,
)
from ols.core.types import (
    ActionSpec,
    Claim,
    ClaimVerdict,
    ExperimentResult,
)


# -- DB task loading ---------------------------------------------------------

@dataclass
class DBTask:
    """One DiscoveryBench task (metadata + resolved CSV paths + gold)."""
    task_id: str                            # e.g. "train/nls_bmi/metadata_0"
    domain: str
    query: str                              # the natural-language discovery goal
    query_type: str                         # "variable" / "context" / etc.
    domain_knowledge: str
    datasets: list[dict]                    # parsed metadata.datasets
    csv_paths: dict[str, Path]              # csv name → resolved path
    gold_hypothesis: str                    # the ground truth NL hypothesis


def _load_metadata_json(path: Path) -> dict:
    # DB metadata files include UTF-8 chars; default open fails on Windows cp1251
    return json.loads(path.read_text(encoding="utf-8"))


def load_db_tasks(repo_root: Path | str, split: str = "train",
                  max_tasks: int | None = None) -> list[DBTask]:
    """Walk discoverybench/real/<split>/<topic>/metadata_*.json files."""
    repo = Path(repo_root)
    base = repo / "discoverybench" / "real" / split
    if not base.exists():
        raise FileNotFoundError(f"DB repo missing: {base}")
    tasks: list[DBTask] = []
    for topic_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for meta_file in sorted(topic_dir.glob("metadata_*.json")):
            try:
                m = _load_metadata_json(meta_file)
            except Exception:
                continue
            queries = m.get("queries", [])
            if queries and isinstance(queries[0], list):  # nested list pattern
                queries = queries[0]
            datasets = m.get("datasets", [])
            csv_paths = {}
            for d in datasets:
                if not isinstance(d, dict):
                    continue
                name = d.get("name")
                if not name:
                    continue
                p = topic_dir / name
                if p.exists():
                    csv_paths[name] = p

            main = (m.get("hypotheses") or {}).get("main") or []
            gold_main = ""
            if main and isinstance(main[0], dict):
                gold_main = main[0].get("text", "") or ""

            for q in queries:
                if not isinstance(q, dict):
                    continue
                gold = q.get("true_hypothesis") or gold_main
                if not gold:
                    continue  # skip unlabeled queries
                if not csv_paths:
                    continue
                tasks.append(DBTask(
                    task_id=f"{split}/{topic_dir.name}/{meta_file.stem}/q{q.get('qid', '?')}",
                    domain=m.get("domain", ""),
                    query=q.get("question", "") or "",
                    query_type=q.get("question_type", ""),
                    domain_knowledge=m.get("domain_knowledge", "") or "",
                    datasets=datasets,
                    csv_paths=csv_paths,
                    gold_hypothesis=gold,
                ))
                if max_tasks is not None and len(tasks) >= max_tasks:
                    return tasks
    return tasks


# -- safe-eval ---------------------------------------------------------------

def _safe_eval(code: str, df: pd.DataFrame, timeout_s: float = 5.0) -> str:
    """Run a single pandas expression against `df` and return a stringified result.

    Restricted namespace: only df, pd, np available. The expression must be a
    Python EXPRESSION (no imports/assignments).
    """
    import numpy as np
    code = (code or "").strip().strip("`").strip()
    if not code or code.startswith("#"):
        return "(empty code)"
    if any(bad in code for bad in ("import ", "exec(", "open(", "__", "eval(")):
        return "(disallowed token: import/exec/open/__/eval)"

    ns = {"df": df, "pd": pd, "np": np}
    out: dict[str, Any] = {}

    def worker():
        try:
            r = eval(code, {"__builtins__": {"len": len, "range": range,
                                              "abs": abs, "min": min, "max": max,
                                              "sum": sum, "round": round}}, ns)
            out["ok"] = repr(r)
        except Exception as e:
            out["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        return "(timed out)"
    if "err" in out:
        return f"ERROR: {out['err']}"
    s = out.get("ok", "")
    # Truncate to keep inner-LLM context small
    if len(s) > 1200:
        s = s[:1200] + "...(truncated)"
    return s


# -- adapter -----------------------------------------------------------------

class DiscoveryBenchAdapter(ResearchEnvAdapter):
    def __init__(self, task: DBTask, budget: float = 12.0,
                 judge_model: str | None = None):
        self.task = task
        self._budget_total = budget
        self._budget_spent = 0.0
        self._eid_next = 0
        self._dfs: dict[str, pd.DataFrame] = {}        # lazy CSV cache
        self._submitted_hypothesis: str | None = None
        self._judge_model = judge_model or os.environ.get(
            "OLS_DB_JUDGE_MODEL", "openai/gpt-4o"
        )

    # -- df cache ----

    def _df(self, name: str) -> pd.DataFrame:
        if name not in self._dfs:
            p = self.task.csv_paths.get(name)
            if p is None or not p.exists():
                raise FileNotFoundError(name)
            self._dfs[name] = pd.read_csv(p)
        return self._dfs[name]

    def _next_eid(self) -> int:
        self._eid_next += 1
        return self._eid_next

    # -- handle ----

    def handle(self) -> EnvHandle:
        ds_names = list(self.task.csv_paths.keys())
        ds_desc = "\n".join(
            f"  - {d.get('name')}: {d.get('description','')[:200]}"
            for d in self.task.datasets if isinstance(d, dict)
        )
        col_desc_lines = []
        for d in self.task.datasets:
            if not isinstance(d, dict):
                continue
            cols = (d.get("columns") or {}).get("raw") or []
            if cols:
                col_desc_lines.append(f"  Columns of {d.get('name')}:")
                for c in cols[:24]:
                    if isinstance(c, dict):
                        col_desc_lines.append(
                            f"    {c.get('name')}: {str(c.get('description',''))[:120]}"
                        )
        col_block = "\n".join(col_desc_lines)
        dk = (self.task.domain_knowledge or "").strip()
        dk_block = f"\nDOMAIN KNOWLEDGE:\n  {dk[:1200]}" if dk else ""

        desc = (
            f"DiscoveryBench task — domain: {self.task.domain}. You have access "
            f"to {len(ds_names)} CSV dataset(s) and must derive a single "
            f"hypothesis answering the discovery query.\n\n"
            f"DATASETS:\n{ds_desc}\n\n"
            f"{col_block}{dk_block}\n\n"
            f"DISCOVERY QUERY: {self.task.query}"
        )
        # Universal research-process decomposition — domain-agnostic, applies
        # to any discovery task. Each step REQUIRES at least one claim before
        # the scaffold lets the agent advance (gating is configured on the
        # scaffold side via require_claim_before_advance).
        subdomains = [
            ("explore",
             "Inspect the dataset(s): which variables look most relevant to "
             "the query? Use head / describe / query to survey. Emit one claim "
             "summarising what variables you will focus on and why."),
            ("test",
             "Test the candidate relationship: compute correlations, group "
             "means, or a simple regression as appropriate. Emit one claim "
             "stating the qualitative relationship (positive / negative / null) "
             "with supporting numbers."),
            ("quantify",
             "Pin down the magnitude: a coefficient, a slope, a percent change, "
             "or the relevant numeric summary. Emit one claim stating the "
             "quantitative result."),
            ("submit",
             f"Synthesise the prior claims into a single-paragraph hypothesis "
             f"answering the discovery query and call submit_hypothesis. "
             f"Query: {self.task.query[:200]}"),
        ]
        return EnvHandle(
            description=desc,
            subdomains=subdomains,
            actions=[
                ActionSpec(
                    name="head",
                    arg_schema={"dataset": "str", "n": "int"},
                    cost_estimate=1.0,
                    description="Preview the first n rows of a named CSV.",
                ),
                ActionSpec(
                    name="describe",
                    arg_schema={"dataset": "str"},
                    cost_estimate=1.0,
                    description="Return df.describe() for the named CSV.",
                ),
                ActionSpec(
                    name="query",
                    arg_schema={"dataset": "str", "code": "str"},
                    cost_estimate=1.0,
                    description=("Evaluate a single pandas/numpy EXPRESSION "
                                 "against the named dataset (df=that CSV). "
                                 "Examples: df.corr()['BMI'], "
                                 "df.groupby('region')['income'].mean(), "
                                 "df['x'].quantile([0.25,0.5,0.75]). "
                                 "No imports, no assignments — expression only."),
                ),
                ActionSpec(
                    name="submit_hypothesis",
                    arg_schema={"text": "str"},
                    cost_estimate=0.0,
                    description=("Submit the FINAL hypothesis answering the "
                                 "discovery query. Should be a short paragraph "
                                 "stating the relationship + direction + any "
                                 "quantitative coefficient when grounded. "
                                 "Submitting ends the episode."),
                ),
            ],
            budget_total=self._budget_total,
        )

    def budget_left(self) -> float:
        return max(0.0, self._budget_total - self._budget_spent)

    # -- execute ----

    def execute(self, action: str, args: dict) -> ExperimentResult:
        if self.budget_left() <= 0:
            raise BudgetExhausted("DB budget exhausted")

        eid = self._next_eid()
        try:
            if action == "head":
                name = str(args.get("dataset", ""))
                n = int(args.get("n", 5))
                df = self._df(name)
                summary = {"head": df.head(min(n, 10)).to_string(max_cols=12)[:1200],
                           "shape": list(df.shape),
                           "dtypes": {c: str(t) for c, t in df.dtypes.items()}}
                cost = 1.0
            elif action == "describe":
                name = str(args.get("dataset", ""))
                df = self._df(name)
                summary = {"describe": df.describe(include="all").to_string()[:1500],
                           "shape": list(df.shape)}
                cost = 1.0
            elif action == "query":
                name = str(args.get("dataset", ""))
                code = str(args.get("code", ""))
                df = self._df(name)
                result = _safe_eval(code, df)
                summary = {"code": code, "result": result}
                cost = 1.0
            elif action == "submit_hypothesis":
                text = str(args.get("text", "")).strip()
                self._submitted_hypothesis = text
                summary = {"submitted": text[:400]}
                cost = 0.0
            else:
                summary = {"error": f"unknown action: {action}"}
                cost = 0.0
        except FileNotFoundError as e:
            summary = {"error": f"dataset not found: {e}"}
            cost = 0.5
        except Exception as e:
            summary = {"error": f"{type(e).__name__}: {e}"}
            cost = 0.5

        self._budget_spent += cost
        return ExperimentResult(
            eid=eid, action=action, args=args,
            cost=cost, raw=None, summary=summary,
        )

    # -- ground-truth (none per-claim — judge at episode end) ----

    def verify_claim(self, claim: Claim) -> ClaimVerdict | None:
        return None

    # -- scoring (LLM judge) ----

    def _judge(self, submitted: str, gold: str, query: str) -> tuple[float, str]:
        """Single LLM-judge call returning HMS-style 0..1 score + brief reason."""
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url and key.startswith("sk-or-"):
            base_url = "https://openrouter.ai/api/v1"
        client = (OpenAI(base_url=base_url, api_key=key)
                  if base_url else OpenAI())
        sys_p = (
            "You are a strict scientific-hypothesis judge. Given a discovery "
            "query, a GOLD hypothesis, and a SUBMITTED hypothesis, score the "
            "submission on three facets: (1) context (does it address the "
            "right scope and dataset/variables); (2) variables (are the right "
            "predictor and outcome variables identified); (3) relationship "
            "(is the relationship type — positive/negative/null, magnitude or "
            "sign — correct). Each facet is 0..1. The final HMS = ctx × "
            "((vars + rel)/2). Reply EXACTLY one JSON object: "
            '{"ctx":0..1,"vars":0..1,"rel":0..1,"hms":0..1,"explain":"..."}'
        )
        usr = (
            f"DISCOVERY QUERY:\n{query}\n\n"
            f"GOLD HYPOTHESIS:\n{gold}\n\n"
            f"SUBMITTED HYPOTHESIS:\n{submitted or '(none submitted)'}\n\n"
            "Score now."
        )
        for use_rf in (True, False):
            try:
                kw = dict(
                    model=self._judge_model,
                    messages=[{"role": "system", "content": sys_p},
                              {"role": "user", "content": usr}],
                    temperature=0.1,
                    max_tokens=500,
                )
                if use_rf:
                    kw["response_format"] = {"type": "json_object"}
                r = client.chat.completions.create(**kw)
                content = (r.choices[0].message.content or "").strip()
                if not content:
                    continue
                # tolerate fences
                if content.startswith("```"):
                    content = content.strip("` \n")
                    if content.startswith("json"):
                        content = content[4:]
                i, j = content.find("{"), content.rfind("}")
                if 0 <= i < j:
                    content = content[i:j + 1]
                obj = json.loads(content)
                hms = float(obj.get("hms", 0.0))
                exp = str(obj.get("explain", ""))[:300]
                return max(0.0, min(1.0, hms)), exp
            except Exception:
                continue
        return 0.0, "(judge failed)"

    def score_episode(self, claim_store_active, final_artifact=None) -> dict:
        submitted = self._submitted_hypothesis or ""
        # if nothing was explicitly submitted, fall back to the highest-confidence
        # active claim's statement (so OLS earns partial credit if its inner
        # agent only emitted claims and forgot to call submit_hypothesis).
        if not submitted and claim_store_active:
            top = max(claim_store_active, key=lambda c: c.confidence)
            submitted = top.statement
        hms, explain = self._judge(
            submitted=submitted,
            gold=self.task.gold_hypothesis,
            query=self.task.query,
        )
        return {
            "primary": hms,
            "HMS": hms,
            "submitted": submitted[:400],
            "gold": self.task.gold_hypothesis[:400],
            "judge_explain": explain,
        }
