"""Persistent registry for self-written analysis modules.

Generated code should not patch the core repository directly.  A promoted
program is written as a versioned module under ``lmw/self_modules`` only after
it passes a syntax/safety check and a residual-gain gate.  Future runs can load
the registry as a scored library of reusable partial programs.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT = _PROJECT_ROOT / "lmw" / "self_modules"
_BANNED_CALLS = {
    "eval",
    "exec",
    "open",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}
_BANNED_ATTR_PREFIX = "__"
_ALLOWED_IMPORT_ROOTS = {"math", "statistics", "itertools"}


@dataclass(frozen=True)
class SelfModuleRecord:
    namespace: str
    name: str
    path: str
    description: str
    score: dict[str, Any]
    source_hash: str
    created_at: float


@dataclass(frozen=True)
class TrustedSelfModuleRecord:
    namespace: str
    source_hash: str
    name: str
    path: str
    description: str
    evidence_count: int
    validation_keys: tuple[str, ...]
    mean_loss: float
    mean_exact_rate: float
    promoted_at: float


def _safe_slug(raw: str, fallback: str = "module") -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw).strip()).strip("_").lower()
    if not slug or not slug[0].isalpha():
        slug = f"{fallback}_{slug}".strip("_")
    return slug[:80] or fallback


def _source_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def validate_self_module_source(code: str) -> tuple[bool, str]:
    """Static safety check for generated modules."""

    if not code or not code.strip():
        return False, "empty source"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    has_function = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            has_function = True
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORT_ROOTS:
                    return False, f"import not allowed: {alias.name}"
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                return False, f"import not allowed: {node.module}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_CALLS:
                return False, f"unsafe call: {node.func.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith(_BANNED_ATTR_PREFIX):
            return False, f"unsafe attribute: {node.attr}"
    if not has_function:
        return False, "no function defined"
    return True, ""


class SelfModuleRegistry:
    """Write and load generated modules after promotion gates pass."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else _DEFAULT_ROOT

    def _namespace_dir(self, namespace: str) -> Path:
        ns = _safe_slug(namespace, "namespace")
        return self.root / ns

    def load(self, namespace: str | None = None) -> list[SelfModuleRecord]:
        records: list[SelfModuleRecord] = []
        roots = [self._namespace_dir(namespace)] if namespace else [
            p for p in self.root.glob("*") if p.is_dir()
        ]
        for ns_dir in roots:
            manifest = ns_dir / "manifest.jsonl"
            if not manifest.exists():
                continue
            for line in manifest.read_text(encoding="utf-8").splitlines():
                try:
                    data = json.loads(line)
                    records.append(SelfModuleRecord(**data))
                except Exception:
                    continue
        return records

    def load_trusted(self, namespace: str | None = None) -> list[TrustedSelfModuleRecord]:
        records: list[TrustedSelfModuleRecord] = []
        roots = [self._namespace_dir(namespace)] if namespace else [
            p for p in self.root.glob("*") if p.is_dir()
        ]
        for ns_dir in roots:
            manifest = ns_dir / "trusted_manifest.jsonl"
            if not manifest.exists():
                continue
            for line in manifest.read_text(encoding="utf-8").splitlines():
                try:
                    data = json.loads(line)
                    if isinstance(data.get("validation_keys"), list):
                        data["validation_keys"] = tuple(data["validation_keys"])
                    records.append(TrustedSelfModuleRecord(**data))
                except Exception:
                    continue
        return records

    def promote(
        self,
        *,
        namespace: str,
        name: str,
        code: str,
        description: str,
        score: dict[str, Any],
        min_exact_rate: float = 0.5,
        max_loss_mean: float = 0.12,
    ) -> SelfModuleRecord | None:
        """Persist a generated module if it passes safety and score gates."""

        try:
            exact_rate = float(score.get("exact_rate", 0.0))
            loss_mean = float(score.get("loss_mean", 1.0))
        except Exception:
            return None
        if exact_rate < min_exact_rate or loss_mean > max_loss_mean:
            return None
        ok, reason = validate_self_module_source(code)
        if not ok:
            return None

        ns_dir = self._namespace_dir(namespace)
        ns_dir.mkdir(parents=True, exist_ok=True)
        (ns_dir / "__init__.py").touch()

        h = _source_hash(code)
        mod_name = f"{_safe_slug(name, 'module')}_{h}"
        path = ns_dir / f"{mod_name}.py"
        header = (
            "# Auto-generated by MARS SelfModuleRegistry.\n"
            "# Do not edit by hand unless promoting into core code.\n\n"
        )
        path.write_text(header + code.strip() + "\n", encoding="utf-8")

        record = SelfModuleRecord(
            namespace=_safe_slug(namespace, "namespace"),
            name=mod_name,
            path=str(path),
            description=description,
            score=dict(score),
            source_hash=h,
            created_at=time.time(),
        )
        manifest = ns_dir / "manifest.jsonl"
        with manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
        return record

    def trusted_candidates(
        self,
        namespace: str,
        *,
        min_validation_keys: int = 2,
        min_mean_exact_rate: float = 0.5,
        max_mean_loss: float = 0.12,
    ) -> list[TrustedSelfModuleRecord]:
        """Return modules that pass cross-task/seed promotion gates.

        Records are grouped by source hash.  A group is trusted only when it has
        evidence on at least ``min_validation_keys`` distinct validation keys and
        its aggregate score still passes the residual gate.
        """

        grouped: dict[str, list[SelfModuleRecord]] = {}
        for rec in self.load(namespace):
            grouped.setdefault(rec.source_hash, []).append(rec)
        trusted: list[TrustedSelfModuleRecord] = []
        for source_hash, records in grouped.items():
            validation_keys = {
                str(r.score.get("validation_key") or r.score.get("task_key") or "")
                for r in records
            }
            validation_keys.discard("")
            if len(validation_keys) < min_validation_keys:
                continue
            losses = []
            exacts = []
            for rec in records:
                try:
                    losses.append(float(rec.score.get("loss_mean", 1.0)))
                    exacts.append(float(rec.score.get("exact_rate", 0.0)))
                except Exception:
                    continue
            if not losses or not exacts:
                continue
            mean_loss = sum(losses) / len(losses)
            mean_exact = sum(exacts) / len(exacts)
            if mean_exact < min_mean_exact_rate or mean_loss > max_mean_loss:
                continue
            best = sorted(
                records,
                key=lambda r: (
                    float(r.score.get("loss_mean", 1.0)),
                    -float(r.score.get("exact_rate", 0.0)),
                    r.created_at,
                ),
            )[0]
            trusted.append(
                TrustedSelfModuleRecord(
                    namespace=best.namespace,
                    source_hash=source_hash,
                    name=best.name,
                    path=best.path,
                    description=best.description,
                    evidence_count=len(records),
                    validation_keys=tuple(sorted(validation_keys)),
                    mean_loss=mean_loss,
                    mean_exact_rate=mean_exact,
                    promoted_at=time.time(),
                )
            )
        trusted.sort(key=lambda r: (r.mean_loss, -r.mean_exact_rate, r.name))
        return trusted

    def refresh_trusted_manifest(
        self,
        namespace: str,
        *,
        min_validation_keys: int = 2,
        min_mean_exact_rate: float = 0.5,
        max_mean_loss: float = 0.12,
    ) -> list[TrustedSelfModuleRecord]:
        """Recompute and persist the trusted-skill manifest for a namespace."""

        trusted = self.trusted_candidates(
            namespace,
            min_validation_keys=min_validation_keys,
            min_mean_exact_rate=min_mean_exact_rate,
            max_mean_loss=max_mean_loss,
        )
        ns_dir = self._namespace_dir(namespace)
        ns_dir.mkdir(parents=True, exist_ok=True)
        manifest = ns_dir / "trusted_manifest.jsonl"
        with manifest.open("w", encoding="utf-8") as f:
            for rec in trusted:
                payload = {
                    **rec.__dict__,
                    "validation_keys": list(rec.validation_keys),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return trusted
