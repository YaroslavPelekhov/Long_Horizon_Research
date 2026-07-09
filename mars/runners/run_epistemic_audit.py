"""Audit MARS result files for universal epistemic certificates.

This runner is intentionally benchmark-agnostic.  It reads existing JSON
artifacts, extracts `supermetrics.epistemic_certificate`, and writes a compact
JSON + Markdown table that separates transferable operators from local patches.

Example:
  python -m mars.runners.run_epistemic_audit \\
      --run_id audit_smoke \\
      --paths lmw/universal_cpi/epistemic_smoke_20260705/summary.json \\
              lmw/universal_cpi/epistemic_smoke_20260705/cpi_results.json \\
      --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from mars.skills.supermetrics import aggregate_supermetrics


_PROJ = Path(__file__).resolve().parent.parent.parent


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_supermetric_rows(payload: Any) -> Iterable[dict[str, Any]]:
    """Yield supermetric rows from common MARS result shapes."""

    if isinstance(payload, Mapping):
        if isinstance(payload.get("supermetrics"), Mapping):
            yield dict(payload["supermetrics"])
            return
        aggregate = payload.get("aggregate_internal_supermetrics")
        if isinstance(aggregate, Mapping):
            for row in (aggregate.get("benchmarks") or {}).values():
                if isinstance(row, Mapping):
                    yield dict(row)
        for key in ("results", "rows"):
            rows = payload.get(key)
            if isinstance(rows, list):
                for item in rows:
                    if isinstance(item, Mapping):
                        yield from _iter_supermetric_rows(item)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_supermetric_rows(item)


def _certificate_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for path in paths:
        payload = _load_json(path)
        for idx, sm in enumerate(_iter_supermetric_rows(payload)):
            cert = sm.get("epistemic_certificate")
            if not isinstance(cert, Mapping):
                continue
            fingerprint = cert.get("fingerprint") or {}
            key = (
                str(sm.get("benchmark", "")),
                str(fingerprint.get("observable_hash", idx)),
                str(sm.get("best_name", "")),
                str(cert.get("winner_family", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source": str(path),
                    "benchmark": sm.get("benchmark", ""),
                    "best_name": sm.get("best_name", ""),
                    "status": cert.get("status", "rejected"),
                    "accepted": bool(cert.get("accepted", False)),
                    "interface_family": fingerprint.get("interface_family", "unknown"),
                    "n_observations": fingerprint.get("n_observations", 0),
                    "universal_score": cert.get("universal_score", 0.0),
                    "compression_gain": cert.get("compression_gain", 0.0),
                    "leakage_penalty": cert.get("leakage_penalty", 0.0),
                    "transfer_support": cert.get("transfer_support", 0.0),
                    "best_loss": cert.get("best_loss", 1.0),
                    "winner_family": cert.get("winner_family", ""),
                    "rejected_reasons": cert.get("rejected_reasons", []),
                }
            )
    return rows


def _markdown_table(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    lines = [
        "# MARS Epistemic Certificate Audit",
        "",
        "This table audits whether saved MARS results look like transferable "
        "epistemic operators rather than benchmark-specific patches.",
        "",
        "## Aggregate",
        "",
        "```json",
        json.dumps(aggregate.get("epistemic", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Certificates",
        "",
        "| status | benchmark | family | winner | loss | compression | leakage | transfer |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {status} | {benchmark} | {family} | {winner} | {loss:.4f} | "
            "{compression:.3f} | {leakage:.3f} | {transfer:.3f} |".format(
                status=row["status"],
                benchmark=str(row.get("benchmark", ""))[:40],
                family=str(row.get("interface_family", ""))[:32],
                winner=str(row.get("best_name", ""))[:48].replace("|", "/"),
                loss=_as_float(row.get("best_loss"), 1.0),
                compression=_as_float(row.get("compression_gain"), 0.0),
                leakage=_as_float(row.get("leakage_penalty"), 0.0),
                transfer=_as_float(row.get("transfer_support"), 0.0),
            )
        )
    return "\n".join(lines) + "\n"


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="epistemic_audit")
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = [Path(p) if Path(p).is_absolute() else _PROJ / p for p in args.paths]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing input files: {missing}")

    out_dir = _PROJ / "lmw" / "epistemic_audit" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "summary.json"
    out_md = out_dir / "audit.md"
    if (out_json.exists() or out_md.exists()) and not args.overwrite:
        raise SystemExit(f"exists; pass --overwrite: {out_dir}")

    rows = _certificate_rows(paths)
    aggregate = aggregate_supermetrics(
        [
            {
                "benchmark": row.get("benchmark", ""),
                "universal_score": row.get("universal_score", 0.0),
                "epistemic_certificate": {
                    "status": row.get("status"),
                    "universal_score": row.get("universal_score"),
                    "compression_gain": row.get("compression_gain"),
                    "leakage_penalty": row.get("leakage_penalty"),
                    "transfer_support": row.get("transfer_support"),
                    "fingerprint": {"interface_family": row.get("interface_family")},
                },
            }
            for row in rows
        ]
    )
    summary = {
        "run_id": args.run_id,
        "n_sources": len(paths),
        "sources": [str(p) for p in paths],
        "n_certificates": len(rows),
        "aggregate": aggregate,
        "rows": rows,
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    out_md.write_text(_markdown_table(rows, aggregate), encoding="utf-8")

    epistemic = aggregate.get("epistemic", {})
    print(f"certificates={len(rows)} accepted={epistemic.get('accepted', 0)} rejected={epistemic.get('rejected', 0)}")
    print(f"summary → {out_json}")
    print(f"markdown → {out_md}")


if __name__ == "__main__":
    main()
