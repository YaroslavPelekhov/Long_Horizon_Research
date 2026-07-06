"""
ScienceAgentBench adapter for MARS.

ScienceAgentBench (Chen et al., ICLR'25, arXiv 2410.05080) — 102 tasks in
4 disciplines (Computational Chemistry, Geographical IS, Bioinformatics,
Psychology/Cognitive Science). Each task = (instruction, dataset folder
tree, dataset preview, domain knowledge) → agent must output a single
Python program file.

v0.1 (this file) — task loading via HuggingFace dataset (input-only data,
password-protected zip not required); MARS produces submit_program(code);
scoring via LLM-judge that reads the code + task description and rates
(0..1) likely correctness. This is a proxy metric — NOT the published
ScienceAgentBench score; for that we need their docker harness + zip.

v0.2 (deferred) — integrate their docker eval harness with the password-
protected benchmark zip for real execution-based scoring.

Action space:
  - read_dataset_preview(): returns the preview already in env description
    (provided for API symmetry; agent rarely needs it)
  - submit_program(code: str): the program; terminal action

Budget: 4 program submissions per task (one for first attempt + 3 debug
iterations). Each submission costs 1 unit.

env vars:
  MARS_SAB_JUDGE_MODEL   default: openai/gpt-4o
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class SABTask:
    instance_id: int
    domain: str
    github: str
    task_inst: str
    domain_knowledge: str
    dataset_folder_tree: str
    dataset_preview: str
    output_fname: str
    subtask_categories: str
    src_file_or_path: str = ""
    gold_program_name: str = ""
    eval_script_name: str = ""

    @classmethod
    def from_hf_row(cls, row: dict) -> "SABTask":
        return cls(
            instance_id=int(row.get("instance_id", -1)),
            domain=str(row.get("domain", "")),
            github=str(row.get("github_name", "")),
            task_inst=str(row.get("task_inst", "")),
            domain_knowledge=str(row.get("domain_knowledge", "") or ""),
            dataset_folder_tree=str(row.get("dataset_folder_tree", "") or ""),
            dataset_preview=str(row.get("dataset_preview", "") or ""),
            output_fname=str(row.get("output_fname", "") or ""),
            subtask_categories=str(row.get("subtask_categories", "") or ""),
            src_file_or_path=str(row.get("src_file_or_path", "") or ""),
            gold_program_name=str(row.get("gold_program_name", "") or ""),
            eval_script_name=str(row.get("eval_script_name", "") or ""),
        )


def load_sab_tasks(max_tasks: int | None = None,
                   domains: list[str] | None = None) -> list[SABTask]:
    """Load tasks from the public HuggingFace mirror (verified split, 102 tasks)."""
    from datasets import load_dataset
    ds = load_dataset("osunlp/ScienceAgentBench", split="verified")
    out: list[SABTask] = []
    for row in ds:
        if domains and row["domain"] not in domains:
            continue
        out.append(SABTask.from_hf_row(row))
        if max_tasks is not None and len(out) >= max_tasks:
            break
    return out


class ScienceAgentBenchAdapter(ResearchEnvAdapter):
    """One task = one MARS episode. Scoring via LLM-judge (v0.1)."""

    def __init__(self, task: SABTask, budget: float = 4.0,
                 judge_model: str | None = None):
        self.task = task
        self._budget_total = float(budget)
        self._budget_spent = 0.0
        self._eid_next = 0
        self._submitted_program: str | None = None
        self._submissions: list[dict] = []
        self._judge_model = judge_model or os.environ.get(
            "MARS_SAB_JUDGE_MODEL", "openai/gpt-4o"
        )

    # -- handle ----

    def handle(self) -> EnvHandle:
        dk = (self.task.domain_knowledge or "").strip()
        dk_block = f"\nDOMAIN KNOWLEDGE:\n{dk[:1500]}\n" if dk else ""
        first_tree_line = (self.task.dataset_folder_tree or "").splitlines()[0:1]
        dataset_rel = ""
        if first_tree_line:
            # SAB official prompt computes:
            # args.datasets_path + example["dataset_folder_tree"].split("\n")[0][4:]
            # with args.datasets_path defaulting to "benchmark/datasets/".
            dataset_rel = first_tree_line[0][4:].strip()
        dataset_path = f"benchmark/datasets/{dataset_rel}" if dataset_rel else "benchmark/datasets/"
        desc = (
            f"ScienceAgentBench task — domain: {self.task.domain}, "
            f"github: {self.task.github}, subtasks: {self.task.subtask_categories}.\n\n"
            f"TASK INSTRUCTION:\n{self.task.task_inst}\n\n"
            f"DATASET PATH:\n{dataset_path}\n\n"
            f"DATASET FOLDER TREE:\n{self.task.dataset_folder_tree[:1500]}\n\n"
            f"DATASET PREVIEW:\n{self.task.dataset_preview[:2000]}\n"
            f"{dk_block}\n"
            f"OUTPUT FILENAME (your program must save its result here): "
            f"{self.task.output_fname}\n\n"
            f"You will produce one Python program file that, when executed in a "
            f"sandbox from the repository root, can access the dataset at "
            f"`{dataset_path}` and must write output to `{self.task.output_fname}`. "
            f"Use paths exactly in this style, e.g. `benchmark/datasets/...`, "
            f"not only the bare dataset folder name.\n\n"
            f"UNIVERSAL SELF-GROUNDING RULES:\n"
            f"- First build a small manifest from the listed dataset files; never invent file names.\n"
            f"- Create output parent directories before saving.\n"
            f"- Probe file schema/shape/value ranges before choosing the transformation.\n"
            f"- Category maps must be total: unseen values must not crash the program.\n"
            f"- For rasters or large arrays, prefer streaming/windowed processing and avoid full reads when possible.\n"
            f"- Keep execution under the official timeout; compute the minimal artifact required by the task.\n\n"
            f"You may submit up to {int(self._budget_total)} "
            f"program revisions; each submission costs 1 budget unit. The final "
            f"submitted program will be scored."
        )
        return EnvHandle(
            description=desc,
            subdomains=[(
                "solve",
                f"Write a complete, self-contained Python program that solves "
                f"the task: {self.task.task_inst[:300]}"
            )],
            actions=[
                ActionSpec(
                    name="submit_program",
                    arg_schema={"code": "str (complete Python program)"},
                    cost_estimate=1.0,
                    description=(
                        "Submit a complete Python program (no partials, no "
                        "interactive commands like !pip). The previously-submitted "
                        "program is replaced. After your final submission OR when "
                        "budget is exhausted, the program will be scored."
                    ),
                ),
            ],
            budget_total=self._budget_total,
        )

    def budget_left(self) -> float:
        return max(0.0, self._budget_total - self._budget_spent)

    # -- execute ----

    def execute(self, action: str, args: dict) -> ExperimentResult:
        if self.budget_left() <= 0:
            raise BudgetExhausted("SAB budget exhausted")

        self._eid_next += 1
        if action == "submit_program":
            code = str(args.get("code", "")).strip()
            self._submitted_program = code
            self._submissions.append({"eid": self._eid_next, "code_len": len(code)})
            self._budget_spent += 1.0
            summary = {
                "submission_n": len(self._submissions),
                "code_len": len(code),
                "code_preview": code[:500],
            }
            return ExperimentResult(
                eid=self._eid_next, action="submit_program", args={},
                cost=1.0, raw=None, summary=summary,
            )
        return ExperimentResult(
            eid=self._eid_next, action=action, args=args, cost=0.0,
            raw=None, summary={"error": f"unknown action: {action}"},
        )

    # -- ground-truth (no per-claim oracle; episode-level judge) ----

    def verify_claim(self, claim: Claim) -> ClaimVerdict | None:
        return None

    # -- scoring ----

    def _judge(self, code: str) -> tuple[float, str]:
        """LLM-judge that rates 0..1 likely correctness of submitted code
        against the task description. NOT the published SAB metric; proxy
        for v0.1 architecture validation."""
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url and key.startswith("sk-or-"):
            base_url = "https://openrouter.ai/api/v1"
        client = (OpenAI(base_url=base_url, api_key=key)
                  if base_url else OpenAI())

        sys_p = (
            "You are a strict code-judge for a scientific-discovery benchmark. "
            "Given a TASK INSTRUCTION, the DATASET PREVIEW the agent saw, and "
            "the SUBMITTED PROGRAM, rate the program on four facets (each 0..1):\n"
            "  - load: does it correctly load/parse the dataset?\n"
            "  - method: does it apply a reasonable method for the requested task?\n"
            "  - output: does it write to the requested output filename in the "
            "    expected format?\n"
            "  - completeness: is it a self-contained executable program (no "
            "    placeholders, no missing imports)?\n"
            "Final SCORE = mean of the four. Reply EXACTLY one JSON object:\n"
            '{"load":0..1, "method":0..1, "output":0..1, "completeness":0..1,\n'
            ' "score":0..1, "explain":"..."}'
        )
        usr = (
            f"TASK INSTRUCTION:\n{self.task.task_inst}\n\n"
            f"DATASET FOLDER TREE:\n{self.task.dataset_folder_tree[:600]}\n\n"
            f"DATASET PREVIEW:\n{self.task.dataset_preview[:1000]}\n\n"
            f"OUTPUT FILENAME EXPECTED: {self.task.output_fname}\n\n"
            f"SUBMITTED PROGRAM:\n```python\n{code[:4000]}\n```\n\n"
            "Score now."
        )
        for use_rf in (True, False):
            try:
                kw = dict(
                    model=self._judge_model,
                    messages=[{"role": "system", "content": sys_p},
                              {"role": "user", "content": usr}],
                    temperature=0.1, max_tokens=500,
                )
                if use_rf:
                    kw["response_format"] = {"type": "json_object"}
                r = client.chat.completions.create(**kw)
                content = (r.choices[0].message.content or "").strip()
                if not content:
                    continue
                if content.startswith("```"):
                    content = content.strip("` \n")
                    if content.startswith("json"):
                        content = content[4:].lstrip()
                i, j = content.find("{"), content.rfind("}")
                if 0 <= i < j:
                    content = content[i:j + 1]
                obj = json.loads(content)
                s = float(obj.get("score", 0.0))
                ex = str(obj.get("explain", ""))[:300]
                return max(0.0, min(1.0, s)), ex
            except Exception:
                continue
        return 0.0, "(judge failed)"

    def score_episode(self, claim_store_active, final_artifact=None) -> dict:
        code = str(final_artifact).strip() if final_artifact else (self._submitted_program or "")
        if not code and claim_store_active:
            # fallback: agent emitted claims but no program — use highest-conf
            # claim as a stand-in (will score very low)
            top = max(claim_store_active, key=lambda c: c.confidence)
            code = top.statement
        if not code:
            return {"primary": 0.0, "score": 0.0, "submitted": "",
                    "judge_explain": "(no program submitted)",
                    "n_submissions": len(self._submissions)}
        score, explain = self._judge(code)
        return {
            "primary": score,
            "score": score,
            "submitted": code[:500],
            "judge_explain": explain,
            "n_submissions": len(self._submissions),
        }
