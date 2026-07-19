"""Zero-shot DiscoveryBench: synthesize measurement operators from task schema.

Compares two conditions on the same task × same HMS judge:
  A. zero_shot:   db_grammar_synthesizer proposes pandas operators from
                  task description + dataset schema (no hand-written operators)
  B. hand_cpi:    existing db_cpi.py with hand-written operators (control)

Usage
-----
python -m mars.runners.run_db_zero_shot \\
    --run_id db_zeroshot_archaeology_v1 \\
    --task_dir discoverybench_repo/discoverybench/real/test/archaeology \\
    --metadata_file metadata_1.json \\
    --synth_model openai/gpt-4o --judge_model openai/gpt-4o \\
    --n_proposals 12 --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
_DB_REPO = _PROJ / "discoverybench_repo"
for _p in (_PROJ, _DB_REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv
    _env = _PROJ / "autodiscovery" / ".env.local"
    if _env.exists():
        load_dotenv(_env, override=False)
except ImportError:
    pass

import pandas as pd  # noqa: E402

from mars.agents.base import call_llm, make_openai_client  # noqa: E402
from mars.induction.db_grammar_synthesizer import (  # noqa: E402
    DBSynthesisResult,
    extract_column_descriptions,
    extract_schema_description,
    run_operators,
    synthesize_db_grammar,
)
from mars.induction.qd_cpi import classify_interface, solve as qd_solve  # noqa: E402


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_task(task_dir: Path, metadata_file: str) -> dict[str, Any]:
    meta = json.load(open(task_dir / metadata_file))
    datasets = []
    for ds_meta in meta.get("datasets", []):
        csv_path = task_dir / ds_meta["name"]
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                datasets.append({"meta": ds_meta, "df": df, "path": str(csv_path)})
            except Exception as e:
                datasets.append({"meta": ds_meta, "df": None, "error": str(e)})
    return {"meta": meta, "datasets": datasets}


def get_main_question(meta: dict) -> str:
    queries = meta.get("queries", [])
    if queries and queries[0]:
        return queries[0][0].get("question", "")
    return ""


def get_domain_knowledge(meta: dict) -> str:
    return meta.get("domain_knowledge", "")[:800]


# ---------------------------------------------------------------------------
# HMS judge call (reuse from db_cpi.py pattern)
# ---------------------------------------------------------------------------

def _hms_judge(
    client: Any,
    model: str,
    question: str,
    hypothesis_text: str,
    ground_truth_hyp: str | None = None,
) -> dict[str, Any]:
    """Call HMS judge to score a generated hypothesis."""
    judge_prompt = f"""You are evaluating a scientific discovery hypothesis.

Question: {question}

Generated hypothesis:
{hypothesis_text}

Rate this hypothesis on these criteria (0.0 to 1.0 each):
1. context_match: Does it correctly identify the relevant context/domain?
2. variable_match: Does it identify the correct variables?
3. relation_correct: Is the stated relationship/finding correct?

Return JSON: {{"context_match": 0.0, "variable_match": 0.0, "relation_correct": 0.0, "reasoning": "..."}}"""

    raw = call_llm(
        client,
        model=model,
        system="You evaluate scientific discovery hypotheses. Return only valid JSON.",
        user=judge_prompt,
        max_tokens=600,
        temperature=0.0,
    )

    try:
        scores = json.loads(raw)
    except Exception:
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        scores = json.loads(m.group()) if m else {}

    ctx = float(scores.get("context_match", 0.0))
    var = float(scores.get("variable_match", 0.0))
    rel = float(scores.get("relation_correct", 0.0))
    hms = (ctx + var + rel) / 3.0

    return {
        "hms_0_1": hms,
        "hms_100": hms * 100,
        "context_match": ctx,
        "variable_match": var,
        "relation_correct": rel,
        "reasoning": scores.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Zero-shot rendering
# ---------------------------------------------------------------------------

def render_zero_shot_hypothesis(
    client: Any,
    model: str,
    question: str,
    domain_knowledge: str,
    evidence_list: list[dict],
) -> str:
    """Synthesize a natural-language hypothesis from evidence strings."""
    evidence_text = "\n".join(
        f"- [{e['operator']}] {e['evidence']}"
        for e in evidence_list[:8]
        if e.get("evidence")
    )

    prompt = f"""Based on the following evidence extracted from a scientific dataset, write a concise hypothesis answering the research question.

Research question: {question}

Domain context: {domain_knowledge[:400]}

Evidence from data analysis:
{evidence_text}

Write a 1-3 sentence hypothesis that directly answers the question using the evidence above.
Be specific: name variables, time periods, or groups if relevant."""

    return call_llm(
        client,
        model=model,
        system="You write concise, evidence-based scientific hypotheses.",
        user=prompt,
        max_tokens=300,
        temperature=0.2,
    )


# ---------------------------------------------------------------------------
# Hand-written CPI control (simplified from db_cpi.py)
# ---------------------------------------------------------------------------

def run_hand_cpi(
    client: Any,
    model: str,
    question: str,
    domain_knowledge: str,
    df: pd.DataFrame,
    schema_desc: str,
) -> dict[str, Any]:
    """Control: LLM-only baseline — no synthesized operators, just schema + question."""
    # This is the "LLM only" baseline: show schema to LLM, ask for hypothesis directly
    prompt = f"""You are a scientist analyzing a dataset to answer a research question.

Question: {question}

Domain context: {domain_knowledge[:400]}

Dataset schema and sample:
{schema_desc[:1200]}

Based on this schema and your knowledge, write a 1-2 sentence hypothesis answering the question.
Be specific: name variables, time periods, or values if possible."""

    hypothesis = call_llm(
        client,
        model=model,
        system="You write evidence-based scientific hypotheses from dataset schemas.",
        user=prompt,
        max_tokens=250,
        temperature=0.2,
    )
    return {"hypothesis": hypothesis, "evidence": ["schema-only reasoning"], "method": "llm_schema_only"}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    task_dir = Path(args.task_dir)
    task = load_task(task_dir, args.metadata_file)
    meta = task["meta"]
    question = get_main_question(meta)
    domain_knowledge = get_domain_knowledge(meta)

    print(f"\nTask: {task_dir.name}/{args.metadata_file}")
    print(f"Question: {question}")

    # Load and clean ALL datasets; select the most relevant for the question
    question_lower = question.lower()
    all_dfs: list[tuple[str, pd.DataFrame, str]] = []  # (name, df, description)
    for ds in task["datasets"]:
        if ds.get("df") is None:
            continue
        ds_df = ds["df"].copy()
        # Fix European comma-decimal notation
        for col in ds_df.columns:
            if ds_df[col].dtype == object:
                try:
                    converted = ds_df[col].str.replace(",", ".", regex=False).astype(float)
                    ds_df[col] = converted
                except Exception:
                    pass
        all_dfs.append((ds["meta"]["name"], ds_df, ds["meta"].get("description", "")))

    # Score by column+description relevance to question
    q_words = set(question_lower.replace("?", "").split())
    best_score = -1
    df = None
    df_name = None
    for name, ds_df, desc in all_dfs:
        col_text = (" ".join(ds_df.columns) + " " + desc).lower()
        score = sum(1 for w in q_words if len(w) > 4 and w in col_text)
        # Bonus for description length (more informative metadata)
        score += len(desc) * 0.0001
        if score > best_score:
            best_score = score
            df = ds_df
            df_name = name

    if df is None and all_dfs:
        df_name, df, _ = all_dfs[-1]  # last = usually the main analysis table

    # Extract column descriptions from metadata for the selected dataset
    col_descs: dict[str, str] = {}
    for ds in task["datasets"]:
        if ds["meta"]["name"] == df_name:
            col_descs = extract_column_descriptions(ds["meta"])
            break

    # Show synthesizer ALL dataset schemas (with column descriptions) to pick from
    all_schemas_parts = []
    for name, d, _ in all_dfs:
        ds_col_descs: dict[str, str] = {}
        for ds in task["datasets"]:
            if ds["meta"]["name"] == name:
                ds_col_descs = extract_column_descriptions(ds["meta"])
                break
        all_schemas_parts.append(
            f"=== Dataset: {name} ===\n{extract_schema_description(d, column_descriptions=ds_col_descs)[:700]}"
        )
    all_schemas = "\n\n".join(all_schemas_parts) if len(all_schemas_parts) > 1 else ""

    if df is None:
        raise RuntimeError(f"No readable datasets in {task_dir}")

    print(f"Dataset: {df_name} shape={df.shape}")
    schema_desc = extract_schema_description(df, column_descriptions=col_descs)

    client = make_openai_client()
    t0 = time.time()

    # ------------------------------------------------------------------ #
    # Run A: zero-shot grammar synthesis                                   #
    # ------------------------------------------------------------------ #
    print(f"\n[zero_shot] synthesizing {args.n_proposals} operators (model={args.synth_model})")
    task_description = (
        f"{question}\n\nDomain context: {domain_knowledge[:400]}"
        + (f"\n\nAll available datasets:\n{all_schemas[:1500]}" if all_schemas else "")
    )

    synthesis: DBSynthesisResult = synthesize_db_grammar(
        task_description,
        df,
        column_descriptions=col_descs,
        model=args.synth_model,
        n_proposals=args.n_proposals,
        temperature=0.7,
    )

    print(f"[zero_shot] proposed={synthesis.proposed} valid={synthesis.valid} time={synthesis.wall_time_s:.1f}s")
    for op in synthesis.operators:
        print(f"  + {op.name}: {op.description[:60]}")
    if synthesis.errors:
        print(f"  errors: {synthesis.errors[:4]}")

    evidence_list = run_operators(synthesis.operators, df)
    print(f"[zero_shot] evidence_pieces={len(evidence_list)}")
    for ev in evidence_list[:4]:
        print(f"  [{ev['operator']}] {str(ev.get('evidence',''))[:80]}")

    zero_shot_hyp = render_zero_shot_hypothesis(
        client, args.synth_model, question, domain_knowledge, evidence_list
    )
    print(f"\n[zero_shot] hypothesis: {zero_shot_hyp[:200]}")

    zero_shot_score = _hms_judge(client, args.judge_model, question, zero_shot_hyp)
    print(f"[zero_shot] HMS={zero_shot_score['hms_100']:.1f}/100")

    # ------------------------------------------------------------------ #
    # Run A2: QD-CPI (question decomposition chain)                       #
    # ------------------------------------------------------------------ #
    interface_type = classify_interface(task_description)
    print(f"\n[qd_cpi] interface_type={interface_type} — running reasoning chain")
    qd_chain = qd_solve(
        question,
        domain_knowledge,
        df,
        column_descriptions=col_descs,
        model=args.synth_model,
    )
    print(f"[qd_cpi] steps={len(qd_chain.steps)} answer={qd_chain.final_answer[:120]}")
    for s in qd_chain.steps:
        status = "✓" if s.verified else "✗"
        print(f"  {status} step_{s.step_id} ({s.step_type}): result={str(s.result)[:60]}")

    qd_score = _hms_judge(client, args.judge_model, question, qd_chain.final_answer)
    print(f"[qd_cpi] HMS={qd_score['hms_100']:.1f}/100")

    # ------------------------------------------------------------------ #
    # Run B: LLM schema-only (control)                                    #
    # ------------------------------------------------------------------ #
    print(f"\n[hand_cpi] running control")
    hand_result = run_hand_cpi(client, args.synth_model, question, domain_knowledge, df, schema_desc)
    print(f"[hand_cpi] hypothesis: {hand_result['hypothesis'][:200]}")

    hand_score = _hms_judge(client, args.judge_model, question, hand_result["hypothesis"])
    print(f"[hand_cpi] HMS={hand_score['hms_100']:.1f}/100")

    # ------------------------------------------------------------------ #
    # Result                                                               #
    # ------------------------------------------------------------------ #
    return {
        "run_id": args.run_id,
        "score_type": "db-zero-shot-vs-hand-cpi",
        "task_dir": str(task_dir),
        "metadata_file": args.metadata_file,
        "question": question,
        "dataset": df_name,
        "synth_model": args.synth_model,
        "judge_model": args.judge_model,
        "n_proposals": args.n_proposals,
        "interface_type": interface_type,
        "qd_cpi": {
            "n_steps": len(qd_chain.steps),
            "n_verified": sum(1 for s in qd_chain.steps if s.verified),
            "final_answer": qd_chain.final_answer,
            "hms_100": qd_score["hms_100"],
            "scores": qd_score,
            "chain": qd_chain.to_dict(),
        },
        "zero_shot": {
            "proposed": synthesis.proposed,
            "valid": synthesis.valid,
            "evidence_pieces": len(evidence_list),
            "hypothesis": zero_shot_hyp,
            "hms_100": zero_shot_score["hms_100"],
            "scores": zero_shot_score,
            "synthesis": synthesis.to_dict(),
        },
        "hand_cpi": {
            "hypothesis": hand_result["hypothesis"],
            "hms_100": hand_score["hms_100"],
            "scores": hand_score,
        },
        "wall_time_s": time.time() - t0,
    }


def _print_report(row: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("DB ZERO-SHOT EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"task: {Path(row['task_dir']).name}/{row['metadata_file']}")
    print(f"question: {row['question']}")
    print()
    z = row["zero_shot"]
    h = row["hand_cpi"]
    qd = row.get("qd_cpi", {})
    print(f"{'Condition':<20} {'HMS':>8}  {'Answer (truncated)'}")
    print("-" * 65)
    print(f"{'qd_cpi':<20} {qd.get('hms_100',0):>7.1f}  {qd.get('final_answer','')[:50]}")
    print(f"{'zero_shot':<20} {z['hms_100']:>7.1f}  {z['hypothesis'][:50]}")
    print(f"{'hand_cpi':<20} {h['hms_100']:>7.1f}  {h['hypothesis'][:50]}")
    print()
    print(f"zero_shot operators: {z['valid']} valid, {z['evidence_pieces']} evidence pieces")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=os.environ.get("MARS_DB_ZS_RUN_ID", "db_zeroshot_smoke"))
    parser.add_argument("--task_dir", default="discoverybench_repo/discoverybench/real/test/archaeology")
    parser.add_argument("--metadata_file", default="metadata_1.json")
    parser.add_argument("--synth_model", default=os.environ.get("MARS_SYNTH_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--judge_model", default=os.environ.get("MARS_JUDGE_MODEL", "openai/gpt-4o"))
    parser.add_argument("--n_proposals", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = _PROJ / "lmw" / "db_zero_shot" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; pass --overwrite: {out_path}")

    row = run_experiment(args)
    out_path.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    _print_report(row)
    print(f"summary → {out_path}")


if __name__ == "__main__":
    main()
