"""External non-MARS DiscoveryBench baselines.

This runner intentionally avoids UniversalCPI/RG-HLI modules. It produces
plain LLM hypotheses and scores them with the same official DiscoveryBench HMS
evaluator used by the main paper pipeline.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

_PROJ = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJ, _PROJ / "discoverybench_repo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv

    env_path = _PROJ / "autodiscovery" / ".env.local"
    if env_path.exists():
        load_dotenv(env_path, override=False)
except ImportError:
    pass

from mars.agents.base import make_openai_client, parse_json_strict  # noqa: E402
from mars.runners.run_db_official_eval import (  # noqa: E402
    _run_official_eval,
    load_official_real_tasks,
)


def _read_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) == 1:
        only = str(df.columns[0])
        if "\t" in only:
            df = pd.read_csv(path, sep="\t")
    return df


def _safe_head(df: pd.DataFrame, n: int = 5) -> str:
    with pd.option_context("display.max_columns", 24, "display.width", 160):
        return df.head(n).to_string(index=False)


def _stats_block(df: pd.DataFrame) -> str:
    parts: list[str] = []
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        try:
            parts.append("describe:\n" + numeric.describe().round(4).to_string())
        except Exception:
            pass
        try:
            corr = numeric.corr(numeric_only=True).abs()
            pairs: list[tuple[float, str, str]] = []
            cols = list(corr.columns)
            for i, a in enumerate(cols):
                for b in cols[i + 1:]:
                    val = corr.loc[a, b]
                    if pd.notna(val):
                        pairs.append((float(val), str(a), str(b)))
            pairs.sort(reverse=True)
            if pairs:
                top = [f"{a} ~ {b}: |r|={v:.3f}" for v, a, b in pairs[:8]]
                parts.append("top absolute correlations:\n" + "\n".join(top))
        except Exception:
            pass
    return "\n\n".join(parts)


def _code_report_block(df: pd.DataFrame, *, max_lines: int = 80) -> str:
    """Small executed-analysis report for a CodeAct-style external baseline."""
    lines: list[str] = []
    numeric = list(df.select_dtypes(include="number").columns)
    non_numeric = [c for c in df.columns if c not in numeric]
    if numeric:
        desc = df[numeric].describe().round(4).transpose()
        lines.append("numeric profile:")
        lines.extend(desc.to_string().splitlines()[: min(12, len(desc) + 1)])
    if len(numeric) >= 2:
        corr = df[numeric].corr(numeric_only=True)
        pairs: list[tuple[float, float, str, str]] = []
        for i, a in enumerate(numeric):
            for b in numeric[i + 1:]:
                val = corr.loc[a, b]
                if pd.notna(val):
                    pairs.append((abs(float(val)), float(val), str(a), str(b)))
        pairs.sort(reverse=True)
        if pairs:
            lines.append("strongest numeric correlations:")
            for _, val, a, b in pairs[:10]:
                lines.append(f"- {a} vs {b}: r={val:.3f}")
    time_like = [
        c for c in df.columns
        if any(tok in str(c).lower() for tok in ("year", "date", "time", "month"))
    ]
    if time_like and numeric:
        tcol = time_like[0]
        tmp = df[[tcol, *numeric]].copy()
        try:
            tmp = tmp.sort_values(tcol)
            first = tmp.head(max(1, min(5, len(tmp))))
            last = tmp.tail(max(1, min(5, len(tmp))))
            lines.append(f"first-vs-last numeric shifts by {tcol}:")
            for col in numeric[:12]:
                a = pd.to_numeric(first[col], errors="coerce").mean()
                b = pd.to_numeric(last[col], errors="coerce").mean()
                if pd.notna(a) and pd.notna(b):
                    lines.append(f"- {col}: first_mean={a:.4g}, last_mean={b:.4g}, delta={b-a:.4g}")
        except Exception:
            pass
    for cat in non_numeric[:6]:
        nunique = df[cat].nunique(dropna=True)
        if not 1 < nunique <= 30 or not numeric:
            continue
        vc = df[cat].astype(str).value_counts().head(6)
        lines.append(f"group counts for {cat}: " + "; ".join(f"{k}={v}" for k, v in vc.items()))
        for num in numeric[:5]:
            try:
                means = (
                    df.groupby(cat, dropna=True)[num]
                    .mean(numeric_only=True)
                    .dropna()
                    .sort_values()
                )
            except Exception:
                continue
            if len(means) >= 2:
                lo, hi = means.index[0], means.index[-1]
                lines.append(
                    f"- {num} by {cat}: lowest={lo}:{means.iloc[0]:.4g}, "
                    f"highest={hi}:{means.iloc[-1]:.4g}"
                )
    if not lines:
        return ""
    return "\n".join(lines[:max_lines])


def _task_context(task: Any, *, method: str) -> str:
    metadata = json.loads(task.metadata_path.read_text(encoding="utf-8"))
    lines = [
        f"Question: {task.task.query}",
        f"Domain knowledge: {str(task.task.domain_knowledge or '')[:1500]}",
        "Datasets:",
    ]
    for name, csv_path in task.task.csv_paths.items():
        try:
            df = _read_df(Path(csv_path))
        except Exception as exc:
            lines.append(f"\n[{name}] load error: {type(exc).__name__}: {exc}")
            continue
        ds_meta = next(
            (d for d in metadata.get("datasets", []) if isinstance(d, dict) and d.get("name") == name),
            {},
        )
        col_desc = ds_meta.get("columns", ds_meta.get("column_descriptions", {}))
        lines.append(f"\n[{name}] rows={len(df)} columns={list(df.columns)}")
        if col_desc:
            lines.append(f"column descriptions: {json.dumps(col_desc, ensure_ascii=False)[:1200]}")
        lines.append("head:\n" + _safe_head(df))
        if method in {"stats", "react", "selfdebug", "codeact"}:
            stats = _stats_block(df)
            if stats:
                lines.append(stats[:2500])
        if method == "codeact":
            code_report = _code_report_block(df)
            if code_report:
                lines.append("executed pandas analysis report:\n" + code_report[:4500])
    return "\n".join(lines)


def _chat_json(
    client: Any,
    model: str,
    *,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], str]:
    last_error = ""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw = (resp.choices[0].message.content or "").strip()
            return parse_json_strict(raw) or {}, raw
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2.0 * (attempt + 1))
    return {}, json.dumps({"error": last_error}, ensure_ascii=False)


def _extract_answer(data: dict[str, Any], raw: str) -> tuple[str, str]:
    hypo = str(data.get("hypothesis") or data.get("final_hypothesis") or "").strip()
    workflow = str(data.get("workflow") or data.get("reasoning") or data.get("trace") or "").strip()
    if not hypo:
        hypo = raw[:1200]
    return hypo, workflow


def _call_direct(client: Any, model: str, context: str, *, method: str) -> tuple[str, str, str]:
    system = (
        "You are a scientific data analyst. Answer the benchmark question with a "
        "specific, testable hypothesis grounded only in the provided tables. "
        "Return strict JSON with keys hypothesis, workflow, reasoning."
    )
    user = (
        f"Baseline method: {method}\n\n{context}\n\n"
        "Produce the best concise DiscoveryBench answer. The hypothesis should "
        "name the context, variables, direction/effect, and relevant groups or time ranges when supported."
    )
    data, raw = _chat_json(
        client,
        model=model,
        system=system,
        user=user,
        temperature=0.2,
        max_tokens=900,
    )
    hypo, workflow = _extract_answer(data, raw)
    return hypo, workflow, raw


def _call_react(client: Any, model: str, context: str) -> tuple[str, str, str]:
    system = (
        "You are a scientific data analyst using the ReAct pattern. You cannot call "
        "external tools, but you must explicitly alternate Thought and Check steps "
        "over the provided table evidence before giving the final hypothesis. "
        "Return strict JSON with keys trace, hypothesis, workflow."
    )
    user = (
        f"{context}\n\n"
        "Use 4-6 concise ReAct steps. Each step must contain: Thought, Check, "
        "Observation. Checks must reference concrete columns, groups, dates, or "
        "statistics visible in the provided context. Then return one final "
        "DiscoveryBench hypothesis and a compact workflow summary."
    )
    data, raw = _chat_json(
        client,
        model=model,
        system=system,
        user=user,
        temperature=0.25,
        max_tokens=1300,
    )
    hypo, workflow = _extract_answer(data, raw)
    trace = str(data.get("trace") or "").strip()
    if trace:
        workflow = (trace + "\n\n" + workflow).strip()
    return hypo, workflow, raw


def _call_selfdebug(client: Any, model: str, context: str) -> tuple[str, str, str]:
    draft_hypo, draft_workflow, draft_raw = _call_react(client, model, context)
    system = (
        "You are a strict scientific reviewer. Find flaws in a proposed "
        "DiscoveryBench answer using only the provided task context. Return strict "
        "JSON with keys critique, revised_hypothesis, revised_workflow."
    )
    user = (
        f"{context}\n\n"
        "DRAFT HYPOTHESIS:\n"
        f"{draft_hypo}\n\n"
        "DRAFT WORKFLOW:\n"
        f"{draft_workflow}\n\n"
        "Debug the draft. Penalize unsupported variables, wrong direction, overly "
        "broad claims, missing time/group constraints, and claims not grounded in "
        "the visible data. Produce the best revised answer."
    )
    data, raw = _chat_json(
        client,
        model=model,
        system=system,
        user=user,
        temperature=0.15,
        max_tokens=1100,
    )
    hypo = str(data.get("revised_hypothesis") or data.get("hypothesis") or "").strip()
    workflow = str(data.get("revised_workflow") or data.get("workflow") or "").strip()
    critique = str(data.get("critique") or "").strip()
    if not hypo:
        hypo = draft_hypo
    if critique:
        workflow = (f"Self-debug critique: {critique}\n\n{workflow or draft_workflow}").strip()
    raw_bundle = json.dumps(
        {"draft": draft_raw[:3000], "revision": raw[:3000]},
        ensure_ascii=False,
    )
    return hypo, workflow, raw_bundle


def _call_codeact(client: Any, model: str, context: str) -> tuple[str, str, str]:
    system = (
        "You are a code-assisted scientific analyst. The prompt includes an "
        "EXECUTED pandas analysis report computed from the provided tables. Use "
        "that executed evidence, not generic intuition, to answer. Return strict "
        "JSON with keys hypothesis, workflow, evidence."
    )
    user = (
        f"{context}\n\n"
        "Formulate one concise DiscoveryBench hypothesis grounded in the executed "
        "analysis report. The workflow must cite the concrete computations used, "
        "such as correlations, group contrasts, first-vs-last shifts, or summary "
        "statistics. If the executed report is insufficient, make the narrowest "
        "supported claim."
    )
    data, raw = _chat_json(
        client,
        model=model,
        system=system,
        user=user,
        temperature=0.15,
        max_tokens=1100,
    )
    hypo, workflow = _extract_answer(data, raw)
    evidence = str(data.get("evidence") or "").strip()
    if evidence:
        workflow = (workflow + "\n\nEvidence: " + evidence).strip()
    return hypo, workflow, raw


def _call_baseline(client: Any, model: str, context: str, *, method: str) -> tuple[str, str, str]:
    if method in {"direct", "stats"}:
        return _call_direct(client, model, context, method=method)
    if method == "react":
        return _call_react(client, model, context)
    if method == "selfdebug":
        return _call_selfdebug(client, model, context)
    if method == "codeact":
        return _call_codeact(client, model, context)
    raise ValueError(f"unknown method: {method}")


def _write_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="discovery_external_baseline")
    ap.add_argument("--method", choices=["direct", "stats", "react", "selfdebug", "codeact"], default="direct")
    ap.add_argument("--max_tasks", type=int, default=10)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--model", default=os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini"))
    ap.add_argument("--judge_model", default=os.environ.get("MARS_DB_OFFICIAL_JUDGE_MODEL", "openai/gpt-4o"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    datasets = {x.strip() for x in args.datasets.split(",") if x.strip()} or None
    tasks = load_official_real_tasks(
        _PROJ / "discoverybench_repo",
        max_tasks=args.max_tasks if args.max_tasks > 0 else None,
        datasets=datasets,
        task_keys=None,
        start_index=args.start_index,
    )
    out_dir = _PROJ / "lmw" / "discovery_external" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    eval_path = out_dir / "official_eval.jsonl"
    trace_path = out_dir / "eval_trace.log"
    summary_path = out_dir / "summary.json"
    for path in (pred_path, eval_path, trace_path, summary_path):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"output exists: {path}")
        if path.exists():
            path.unlink()

    client = make_openai_client()
    eval_rows: list[dict[str, Any]] = []
    started = time.time()
    print(f"=== DiscoveryBench external baseline: {args.method} ===")
    print(f"N={len(tasks)} model={args.model} judge={args.judge_model}")
    with pred_path.open("w", encoding="utf-8") as pred_f, eval_path.open("w", encoding="utf-8") as eval_f, trace_path.open("w", encoding="utf-8") as trace_f:
        for i, task in enumerate(tasks, 1):
            context = _task_context(task, method=args.method)
            hypo, workflow, raw = _call_baseline(client, args.model, context, method=args.method)
            pred = {
                "task_key": task.task_key,
                "dataset": task.dataset_name,
                "metadata_id": task.metadata_id,
                "query_id": task.query_id,
                "method": args.method,
                "pred_hypo": hypo,
                "pred_workflow": workflow,
                "raw_response": raw[:3000],
            }
            _write_jsonl(pred_f, pred)
            row = {
                "task_key": task.task_key,
                "dataset": task.dataset_name,
                "metadata_id": task.metadata_id,
                "query_id": task.query_id,
                "method": args.method,
                "judge_model": args.judge_model,
            }
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    ev = _run_official_eval(
                        task,
                        pred_hypo=hypo,
                        pred_workflow=workflow,
                        judge_model=args.judge_model,
                        trace_f=trace_f,
                    )
                row.update(ev)
                row["HMS_100"] = 100.0 * float(ev.get("final_score", 0.0))
            except Exception as exc:
                row.update({"error": f"{type(exc).__name__}: {exc}", "HMS_100": 0.0})
            eval_rows.append(row)
            _write_jsonl(eval_f, row)
            print(f"[{i}/{len(tasks)}] {task.task_key} HMS={row.get('HMS_100', 0):.1f}")

    scores = [float(r.get("HMS_100", 0.0)) for r in eval_rows]
    summary = {
        "run_id": args.run_id,
        "method": args.method,
        "model": args.model,
        "judge_model": args.judge_model,
        "n": len(eval_rows),
        "mean_HMS": float(st.mean(scores)) if scores else 0.0,
        "errors": sum(1 for r in eval_rows if "error" in r),
        "wall_time_s": time.time() - started,
        "prediction_path": str(pred_path),
        "official_eval_path": str(eval_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary → {summary_path}")
    print(f"HMS.mean={summary['mean_HMS']:.2f} N={summary['n']} errors={summary['errors']}")


if __name__ == "__main__":
    main()
