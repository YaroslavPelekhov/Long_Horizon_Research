"""
Run MARS on DiscoveryBench with the official HMS evaluator.

This runner is intentionally separate from `run_db.py`:

  * `run_db.py` is a cheap dev loop over Real-train with a proxy judge.
  * this file reads the official answer key, exports predictions, and calls
    DiscoveryBench's `eval.new_eval.run_eval_gold_vs_gen_NL_hypo_workflow`.

Outputs:

  lmw/db_official/<run_id>/predictions.jsonl
  lmw/db_official/<run_id>/official_eval.jsonl
  lmw/db_official/<run_id>/eval_trace.log
  lmw/db_official/<run_id>/summary.json

For an exact paper-compatible HMS run, keep the default judge
`gpt-4-1106-preview`. If using OpenRouter, set OPENAI_BASE_URL in the
environment or .env.local; the runner will not print secrets.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import statistics as st_stats
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from dotenv import load_dotenv

    _env_path = _PROJ / "autodiscovery" / ".env.local"
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass

from mars.adapters.discoverybench_adapter import DBTask, DiscoveryBenchAdapter  # noqa: E402
from mars.agents.code_evolver import CodeEvolver  # noqa: E402
from mars.agents.base import call_llm, make_openai_client, parse_json_strict  # noqa: E402
from mars.agents.generator import Generator  # noqa: E402
from mars.agents.memory_selector import MemorySelector  # noqa: E402
from mars.agents.reflector import Reflector  # noqa: E402
from mars.coordinator import Coordinator  # noqa: E402


ABLATIONS = {
    "MARS-full": dict(
        use_reflector=True,
        use_memory_selector=True,
        use_futility_detector=True,
        use_code_evolver=False,
    ),
    "MARS-self": dict(
        use_reflector=True,
        use_memory_selector=True,
        use_futility_detector=True,
        use_code_evolver=True,
    ),
    "MARS-no-ref": dict(
        use_reflector=False,
        use_memory_selector=True,
        use_futility_detector=True,
        use_code_evolver=False,
    ),
    "MARS-no-mem": dict(
        use_reflector=True,
        use_memory_selector=False,
        use_futility_detector=True,
        use_code_evolver=False,
    ),
    "MARS-no-fut": dict(
        use_reflector=True,
        use_memory_selector=True,
        use_futility_detector=False,
        use_code_evolver=False,
    ),
    "MARS-all-off": dict(
        use_reflector=False,
        use_memory_selector=False,
        use_futility_detector=False,
        use_code_evolver=False,
    ),
}


@dataclass
class OfficialDBTask:
    task: DBTask
    dataset_name: str
    metadata_id: int
    query_id: int
    metadata_path: Path
    metadata_type: str
    gold_workflow: str = ""

    @property
    def task_key(self) -> str:
        return f"{self.dataset_name}:{self.metadata_id}:{self.query_id}"


class OfficialDiscoveryBenchAdapter(DiscoveryBenchAdapter):
    """DiscoveryBench adapter variant for export plus official evaluator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_submission_full = ""

    def handle(self):
        h = super().handle()
        h.description += (
            "\n\nOFFICIAL DISCOVERYBENCH OUTPUT FORMAT:\n"
            "When you are ready, call submit_hypothesis with a concise final "
            "answer in this form:\n"
            "HYPOTHESIS: <one scientific hypothesis answering the query>\n"
            "WORKFLOW SUMMARY: <short bullet-like summary of the data operations "
            "and statistical checks you used>\n"
            "Keep the hypothesis specific: include context, variables, direction, "
            "magnitude, dates/groups, or statistical significance when supported. "
            "Avoid extra controls, proxy variables, or broad indicator lists unless "
            "the query explicitly asks for them; HMS rewards matching the minimal "
            "gold context, variables, and relation."
        )
        for action in h.actions:
            if action.name == "submit_hypothesis":
                action.description += (
                    " Include both HYPOTHESIS and WORKFLOW SUMMARY sections."
                )
        return h

    def score_episode(self, claim_store_active, final_artifact=None) -> dict:
        submitted = self._submitted_hypothesis or ""
        if not submitted and claim_store_active:
            top = max(claim_store_active, key=lambda c: c.confidence)
            submitted = top.statement
        self.final_submission_full = submitted
        return {
            "primary": 0.0,
            "HMS": 0.0,
            "submitted": submitted,
            "judge_explain": "official DiscoveryBench export: proxy judge skipped",
        }


def _ensure_openai_env() -> None:
    """Make OpenAI-compatible clients work with either OpenAI or OpenRouter env."""
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    key = os.environ.get("OPENAI_API_KEY", "")
    if key.startswith("sk-or-") and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"


def _metadata_queries(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    queries = metadata.get("queries", [])
    if queries and isinstance(queries[0], list):
        queries = queries[0]
    return [q for q in queries if isinstance(q, dict)]


def _resolve_query(metadata: dict[str, Any], query_id: int) -> dict[str, Any] | None:
    queries = _metadata_queries(metadata)
    for q in queries:
        try:
            if int(q.get("qid")) == int(query_id):
                return q
        except Exception:
            pass
    if 0 <= int(query_id) < len(queries):
        return queries[int(query_id)]
    return None


def _resolve_files(topic_dir: Path, metadata: dict[str, Any]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in metadata.get("datasets", []):
        if not isinstance(d, dict):
            continue
        name = d.get("name")
        if not name:
            continue
        p = topic_dir / str(name)
        if p.exists():
            out[str(name)] = p
    return out


def load_official_real_tasks(
    repo: Path,
    *,
    max_tasks: int | None,
    datasets: set[str] | None,
    task_keys: set[str] | None,
    start_index: int = 0,
) -> list[OfficialDBTask]:
    answer_key = repo / "eval" / "answer_key_real.csv"
    base = repo / "discoverybench" / "real" / "test"
    if not answer_key.exists():
        raise FileNotFoundError(answer_key)
    rows: list[OfficialDBTask] = []
    with answer_key.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            dataset_name = str(raw["dataset"])
            metadata_id = int(raw["metadataid"])
            query_id = int(raw["query_id"])
            key = f"{dataset_name}:{metadata_id}:{query_id}"
            if datasets and dataset_name not in datasets:
                continue
            if task_keys and key not in task_keys:
                continue
            topic_dir = base / dataset_name
            metadata_path = topic_dir / f"metadata_{metadata_id}.json"
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            query_obj = _resolve_query(metadata, query_id)
            if not query_obj:
                continue
            files = _resolve_files(topic_dir, metadata)
            if not files:
                continue
            task = DBTask(
                task_id=f"test/{dataset_name}/metadata_{metadata_id}/q{query_id}",
                domain=str(metadata.get("domain", "")),
                query=str(query_obj.get("question", "") or ""),
                query_type=str(query_obj.get("question_type", "") or ""),
                domain_knowledge=str(metadata.get("domain_knowledge", "") or ""),
                datasets=metadata.get("datasets", []),
                csv_paths=files,
                gold_hypothesis=str(raw["gold_hypo"]),
            )
            rows.append(
                OfficialDBTask(
                    task=task,
                    dataset_name=dataset_name,
                    metadata_id=metadata_id,
                    query_id=query_id,
                    metadata_path=metadata_path,
                    metadata_type="real",
                    gold_workflow="",
                )
            )
    if start_index:
        rows = rows[start_index:]
    if max_tasks is not None:
        rows = rows[:max_tasks]
    return rows


def _split_submission(text: str) -> tuple[str, str]:
    text = (text or "").strip()
    if not text:
        return "", ""
    pattern = re.compile(
        r"(?:^|\n|\b)\s*(?:WORKFLOW\s+SUMMARY|WORKFLOW|ANALYSIS\s+WORKFLOW)\s*:?\s*",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        hypo = text[: match.start()].strip()
        workflow = text[match.end() :].strip()
    else:
        hypo = text
        workflow = ""
    hypo = re.sub(r"^\s*(?:HYPOTHESIS|SCIENTIFIC\s+HYPOTHESIS|FINAL\s+ANSWER)\s*:?\s*", "", hypo, flags=re.I)
    workflow = re.sub(r"^\s*(?:SUMMARY)\s*:?\s*", "", workflow, flags=re.I)
    return hypo.strip(), workflow.strip()


def _synthesize_final_submission(
    official_task: OfficialDBTask,
    *,
    active_claims: list[Any],
    model: str,
) -> tuple[str, str, str]:
    """Create a final answer when the episode ended before terminal submit.

    The finalizer is deliberately benchmark-agnostic: it sees only the query,
    dataset metadata, and active claims produced during the episode. It never
    sees the answer key.
    """
    claims = [
        {
            "statement": getattr(c, "statement", str(c)),
            "confidence": getattr(c, "confidence", None),
            "sub_goal": getattr(c, "sub_goal", None),
        }
        for c in active_claims
    ]
    datasets = []
    for d in official_task.task.datasets:
        if not isinstance(d, dict):
            continue
        cols = (d.get("columns") or {}).get("raw") or []
        datasets.append(
            {
                "name": d.get("name"),
                "description": d.get("description"),
                "columns": [
                    {"name": c.get("name"), "description": c.get("description")}
                    for c in cols[:40]
                    if isinstance(c, dict)
                ],
            }
        )
    sys_p = (
        "You are a scientific result finalizer. Synthesize a concise final "
        "DiscoveryBench-style answer from the query, dataset metadata, and "
        "active claims only. Do not invent numbers unless a claim supports "
        "them. Return exactly one JSON object with keys hypothesis and workflow."
    )
    usr = json.dumps(
        {
            "query": official_task.task.query,
            "domain": official_task.task.domain,
            "query_type": official_task.task.query_type,
            "datasets": datasets,
            "active_claims": claims,
            "requirements": (
                "hypothesis: one natural-language scientific hypothesis that "
                "answers the query; include context, variables, direction, and "
                "magnitude/significance if supported. Keep it minimal: do not "
                "mention controls, proxy variables, or extra indicators unless "
                "the query explicitly asks for them. workflow: short summary of "
                "the analysis steps implied by the claims."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )
    client = make_openai_client()
    raw = call_llm(client, model, sys_p, usr, max_tokens=700)
    obj = parse_json_strict(raw) or {}
    hypo = str(obj.get("hypothesis") or "").strip()
    workflow = str(obj.get("workflow") or "").strip()
    if not hypo:
        hypo = " ".join(c["statement"] for c in claims[:3] if c.get("statement")).strip()
    if not workflow:
        workflow = "Synthesized the final hypothesis from active episode claims."
    return hypo, workflow, raw


def _run_mars_episode(
    official_task: OfficialDBTask,
    *,
    ablation_name: str,
    gen_model: str,
    ref_model: str,
    budget: float,
) -> tuple[dict[str, Any], str, str]:
    ablation = dict(ABLATIONS[ablation_name])
    use_ce = bool(ablation.pop("use_code_evolver", False))
    adapter = OfficialDiscoveryBenchAdapter(
        task=official_task.task,
        budget=budget,
        judge_model="official-export-no-proxy",
    )
    gen = Generator(model=gen_model, max_tokens=1100)
    ref = Reflector(model=ref_model, max_tokens=300)
    sel = MemorySelector(k=4)
    evolver = CodeEvolver(model=ref_model, max_modules=0) if use_ce else None
    coord = Coordinator(
        adapter=adapter,
        generator=gen,
        reflector=ref,
        memory_selector=sel,
        max_turns_per_subgoal=4,
        max_total_turns=int(budget) + 4,
        verbose=False,
        code_evolver=evolver,
        use_code_evolver=use_ce,
        **ablation,
    )
    t0 = time.time()
    rep = coord.run_episode()
    elapsed = time.time() - t0
    full_submission = adapter.final_submission_full or rep.score_dict.get("submitted", "")
    pred_hypo, pred_workflow = _split_submission(full_submission)
    explicit_submission = bool(adapter._submitted_hypothesis)
    finalizer_raw = ""
    finalizer_used = False
    if not explicit_submission:
        active_claims = coord.last_store.active() if coord.last_store is not None else []
        pred_hypo, pred_workflow, finalizer_raw = _synthesize_final_submission(
            official_task,
            active_claims=active_claims,
            model=gen_model,
        )
        full_submission = (
            f"HYPOTHESIS: {pred_hypo}\nWORKFLOW SUMMARY: {pred_workflow}"
        )
        finalizer_used = True
    row = {
        "task_key": official_task.task_key,
        "task_id": official_task.task.task_id,
        "dataset": official_task.dataset_name,
        "metadata_id": official_task.metadata_id,
        "query_id": official_task.query_id,
        "domain": official_task.task.domain,
        "query_type": official_task.task.query_type,
        "query": official_task.task.query,
        "gold_hypo": official_task.task.gold_hypothesis,
        "metadata_path": str(official_task.metadata_path),
        "ablation": ablation_name,
        "generator_model": gen_model,
        "reflector_model": ref_model,
        "budget": budget,
        "n_turns": rep.n_turns,
        "n_actions": rep.n_actions,
        "n_claims_active": rep.n_claims_active,
        "n_claims_retracted": rep.n_claims_retracted,
        "n_generator_calls": rep.n_generator_calls,
        "n_reflector_calls": rep.n_reflector_calls,
        "verdict_counts": rep.verdict_counts,
        "wall_time_s": elapsed,
        "pred_hypo": pred_hypo,
        "pred_workflow": pred_workflow,
        "raw_submission": full_submission,
        "explicit_submit_hypothesis": explicit_submission,
        "finalizer_used": finalizer_used,
        "finalizer_raw": finalizer_raw,
        "history": rep.history,
    }
    return row, pred_hypo, pred_workflow


def _run_official_eval(
    official_task: OfficialDBTask,
    *,
    pred_hypo: str,
    pred_workflow: str,
    judge_model: str,
    trace_f,
) -> dict[str, Any]:
    _ensure_openai_env()
    db_repo = _PROJ / "discoverybench_repo"
    if str(db_repo) not in sys.path:
        sys.path.insert(0, str(db_repo))
    from eval.new_eval import run_eval_gold_vs_gen_NL_hypo_workflow  # noqa: E402

    metadata = json.loads(official_task.metadata_path.read_text(encoding="utf-8"))
    trace_f.write(f"\n\n===== {official_task.task_key} =====\n")
    trace_f.flush()
    with contextlib.redirect_stdout(trace_f):
        result = run_eval_gold_vs_gen_NL_hypo_workflow(
            query=official_task.task.query,
            gold_hypo=official_task.task.gold_hypothesis,
            gold_workflow=official_task.gold_workflow,
            gen_hypo=pred_hypo,
            gen_workflow=pred_workflow,
            dataset_meta=metadata,
            llm_used=judge_model,
            dataset_type=official_task.metadata_type,
            use_column_metadata=True,
        )
    return result


def _jsonl_write(f, row: dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=os.environ.get("MARS_DB_RUN_ID", "mars_db_official_smoke"))
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--max_tasks", type=int, default=int(os.environ.get("MARS_DB_N", "1")))
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--datasets", default=os.environ.get("MARS_DB_DATASETS", ""))
    parser.add_argument("--task_keys", nargs="*", default=None)
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default=os.environ.get("MARS_ABLATION", "MARS-full"))
    parser.add_argument("--generator_model", default=os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--reflector_model", default=os.environ.get("MARS_REFLECTOR_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--budget", type=float, default=float(os.environ.get("MARS_DB_BUDGET", "12")))
    parser.add_argument(
        "--judge_model",
        default=os.environ.get("MARS_DB_OFFICIAL_JUDGE_MODEL", "gpt-4-1106-preview"),
        help="DiscoveryBench HMS judge. Default matches discovery_eval.py.",
    )
    parser.add_argument("--skip_official_eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    max_tasks = args.max_tasks if args.max_tasks > 0 else None
    dataset_filter = {x.strip() for x in args.datasets.split(",") if x.strip()} or None
    task_key_filter = {x.strip() for x in args.task_keys or [] if x.strip()} or None

    out_dir = Path(args.out_dir) if args.out_dir else _PROJ / "lmw" / "db_official" / args.run_id
    pred_path = out_dir / "predictions.jsonl"
    eval_path = out_dir / "official_eval.jsonl"
    trace_path = out_dir / "eval_trace.log"
    summary_path = out_dir / "summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in [pred_path, eval_path, trace_path, summary_path]:
        if p.exists() and not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {p}")
    if args.overwrite:
        for p in [pred_path, eval_path, trace_path, summary_path]:
            if p.exists():
                p.unlink()

    repo = _PROJ / "discoverybench_repo"
    tasks = load_official_real_tasks(
        repo,
        max_tasks=max_tasks,
        datasets=dataset_filter,
        task_keys=task_key_filter,
        start_index=args.start_index,
    )
    if not tasks:
        raise SystemExit("no DiscoveryBench official tasks selected")

    print("\n=== MARS -> DiscoveryBench official HMS runner ===")
    print(f"  run_id={args.run_id}")
    print(f"  out_dir={out_dir}")
    print(f"  gen={args.generator_model}  ref={args.reflector_model}")
    print(f"  ablation={args.ablation}  budget={args.budget}")
    print(f"  judge={args.judge_model}  official_eval={not args.skip_official_eval}")
    print(f"  N={len(tasks)} tasks\n")

    pred_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    t_start = time.time()
    with pred_path.open("w", encoding="utf-8") as pred_f, eval_path.open("w", encoding="utf-8") as eval_f, trace_path.open("w", encoding="utf-8") as trace_f:
        for i, task in enumerate(tasks, 1):
            print(f"  [{i}/{len(tasks)}] {task.task_key} ...", flush=True)
            try:
                pred_row, pred_hypo, pred_workflow = _run_mars_episode(
                    task,
                    ablation_name=args.ablation,
                    gen_model=args.generator_model,
                    ref_model=args.reflector_model,
                    budget=args.budget,
                )
                _jsonl_write(pred_f, pred_row)
                pred_rows.append(pred_row)

                eval_row: dict[str, Any] = {
                    "task_key": task.task_key,
                    "dataset": task.dataset_name,
                    "metadata_id": task.metadata_id,
                    "query_id": task.query_id,
                    "judge_model": args.judge_model,
                    "official_eval_skipped": bool(args.skip_official_eval),
                }
                if not args.skip_official_eval:
                    try:
                        eval_result = _run_official_eval(
                            task,
                            pred_hypo=pred_hypo,
                            pred_workflow=pred_workflow,
                            judge_model=args.judge_model,
                            trace_f=trace_f,
                        )
                        eval_row["eval_result"] = eval_result
                        eval_row["final_score"] = float(eval_result.get("final_score", 0.0))
                        eval_row["HMS_100"] = 100.0 * eval_row["final_score"]
                    except Exception as e:
                        eval_row["error"] = f"{type(e).__name__}: {e}"
                        eval_row["final_score"] = 0.0
                        eval_row["HMS_100"] = 0.0
                _jsonl_write(eval_f, eval_row)
                eval_rows.append(eval_row)
                score = eval_row.get("HMS_100")
                score_s = "skip" if args.skip_official_eval else f"{score:.1f}"
                print(
                    f"      HMS={score_s} turns={pred_row['n_turns']} "
                    f"actions={pred_row['n_actions']} ({pred_row['wall_time_s']:.0f}s)",
                    flush=True,
                )
            except Exception as e:
                err_row = {
                    "task_key": task.task_key,
                    "dataset": task.dataset_name,
                    "metadata_id": task.metadata_id,
                    "query_id": task.query_id,
                    "error": f"{type(e).__name__}: {e}",
                }
                _jsonl_write(pred_f, err_row)
                pred_rows.append(err_row)
                print(f"      ERROR {err_row['error']}", flush=True)

    scores = [
        float(r.get("final_score", 0.0))
        for r in eval_rows
        if not r.get("error") and not r.get("official_eval_skipped")
    ]
    summary = {
        "run_id": args.run_id,
        "generator_model": args.generator_model,
        "reflector_model": args.reflector_model,
        "ablation": args.ablation,
        "budget": args.budget,
        "judge_model": args.judge_model,
        "n_tasks_selected": len(tasks),
        "n_predictions": len(pred_rows),
        "n_official_eval_success": len(scores),
        "official_eval_skipped": bool(args.skip_official_eval),
        "HMS_mean_0_1": st_stats.fmean(scores) if scores else 0.0,
        "HMS_mean_100": 100.0 * st_stats.fmean(scores) if scores else 0.0,
        "HMS_sd_100": 100.0 * st_stats.stdev(scores) if len(scores) > 1 else 0.0,
        "wall_time_s": time.time() - t_start,
        "paths": {
            "predictions": str(pred_path),
            "official_eval": str(eval_path),
            "eval_trace": str(trace_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Summary ===")
    print(f"  predictions={pred_path}")
    print(f"  official_eval={eval_path}")
    print(f"  HMS.mean={summary['HMS_mean_100']:.1f}  N={summary['n_official_eval_success']}")
    print(f"  summary={summary_path}")


if __name__ == "__main__":
    main()
