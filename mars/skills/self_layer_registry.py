"""Persistent registry for self-written MARS layers.

A self-module is a candidate hypothesis program.  A self-layer is one level
higher: it transforms a generic task context into additional candidate program
sources, signals, or residual hints.  Layers are still sandboxed and only become
trusted after cross-key validation, so they extend the architecture without
patching core repository files.
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
_DEFAULT_ROOT = _PROJECT_ROOT / "lmw" / "self_layers"
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
class SelfLayerRecord:
    namespace: str
    name: str
    path: str
    layer_type: str
    contract: dict[str, Any]
    description: str
    score: dict[str, Any]
    source_hash: str
    created_at: float


@dataclass(frozen=True)
class TrustedSelfLayerRecord:
    namespace: str
    source_hash: str
    name: str
    path: str
    layer_type: str
    contract: dict[str, Any]
    description: str
    evidence_count: int
    validation_keys: tuple[str, ...]
    mean_gain: float
    mean_loss: float
    promoted_at: float


def _safe_slug(raw: str, fallback: str = "layer") -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw).strip()).strip("_").lower()
    if not slug or not slug[0].isalpha():
        slug = f"{fallback}_{slug}".strip("_")
    return slug[:80] or fallback


def _source_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def validate_self_layer_source(code: str) -> tuple[bool, str]:
    """Static safety and contract check for generated layer modules."""

    if not code or not code.strip():
        return False, "empty source"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    has_build_layer = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_layer":
            has_build_layer = True
        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef)):
            return False, "async functions/classes are not allowed in self layers"
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
    if not has_build_layer:
        return False, "required function missing: build_layer(context: dict) -> dict"
    return True, ""


class SelfLayerRegistry:
    """Write and load generated architecture layers after promotion gates."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else _DEFAULT_ROOT

    def _namespace_dir(self, namespace: str) -> Path:
        return self.root / _safe_slug(namespace, "namespace")

    def load(self, namespace: str | None = None) -> list[SelfLayerRecord]:
        records: list[SelfLayerRecord] = []
        roots = [self._namespace_dir(namespace)] if namespace else [
            p for p in self.root.glob("*") if p.is_dir()
        ]
        for ns_dir in roots:
            manifest = ns_dir / "manifest.jsonl"
            if not manifest.exists():
                continue
            for line in manifest.read_text(encoding="utf-8").splitlines():
                try:
                    records.append(SelfLayerRecord(**json.loads(line)))
                except Exception:
                    continue
        return records

    def load_trusted(self, namespace: str | None = None) -> list[TrustedSelfLayerRecord]:
        records: list[TrustedSelfLayerRecord] = []
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
                    records.append(TrustedSelfLayerRecord(**data))
                except Exception:
                    continue
        return records

    def promote(
        self,
        *,
        namespace: str,
        name: str,
        layer_type: str,
        code: str,
        contract: dict[str, Any],
        description: str,
        score: dict[str, Any],
        min_gain: float = 0.05,
        max_loss_mean: float = 0.12,
    ) -> SelfLayerRecord | None:
        try:
            gain = float(score.get("gain", 0.0))
            loss_mean = float(score.get("loss_mean", 1.0))
        except Exception:
            return None
        if gain < min_gain or loss_mean > max_loss_mean:
            return None
        ok, _reason = validate_self_layer_source(code)
        if not ok:
            return None

        ns_dir = self._namespace_dir(namespace)
        ns_dir.mkdir(parents=True, exist_ok=True)
        (ns_dir / "__init__.py").touch()

        h = _source_hash(code)
        mod_name = f"{_safe_slug(name, 'layer')}_{h}"
        path = ns_dir / f"{mod_name}.py"
        header = (
            "# Auto-generated by MARS SelfLayerRegistry.\n"
            "# This file is loaded as a sandboxed architecture layer.\n\n"
        )
        path.write_text(header + code.strip() + "\n", encoding="utf-8")

        record = SelfLayerRecord(
            namespace=_safe_slug(namespace, "namespace"),
            name=mod_name,
            path=str(path),
            layer_type=_safe_slug(layer_type, "generic"),
            contract=dict(contract),
            description=description,
            score=dict(score),
            source_hash=h,
            created_at=time.time(),
        )
        with (ns_dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
        return record

    def trusted_candidates(
        self,
        namespace: str,
        *,
        min_validation_keys: int = 2,
        min_mean_gain: float = 0.05,
        max_mean_loss: float = 0.12,
    ) -> list[TrustedSelfLayerRecord]:
        grouped: dict[str, list[SelfLayerRecord]] = {}
        for rec in self.load(namespace):
            grouped.setdefault(rec.source_hash, []).append(rec)
        trusted: list[TrustedSelfLayerRecord] = []
        for source_hash, records in grouped.items():
            validation_keys = {
                str(r.score.get("validation_key") or r.score.get("task_key") or "")
                for r in records
            }
            validation_keys.discard("")
            if len(validation_keys) < min_validation_keys:
                continue
            gains: list[float] = []
            losses: list[float] = []
            for rec in records:
                try:
                    gains.append(float(rec.score.get("gain", 0.0)))
                    losses.append(float(rec.score.get("loss_mean", 1.0)))
                except Exception:
                    continue
            if not losses:
                continue
            mean_gain = sum(gains) / len(gains) if gains else 0.0
            mean_loss = sum(losses) / len(losses)
            if mean_gain < min_mean_gain or mean_loss > max_mean_loss:
                continue
            best = sorted(
                records,
                key=lambda r: (
                    float(r.score.get("loss_mean", 1.0)),
                    -float(r.score.get("gain", 0.0)),
                    r.created_at,
                ),
            )[0]
            trusted.append(
                TrustedSelfLayerRecord(
                    namespace=best.namespace,
                    source_hash=source_hash,
                    name=best.name,
                    path=best.path,
                    layer_type=best.layer_type,
                    contract=best.contract,
                    description=best.description,
                    evidence_count=len(records),
                    validation_keys=tuple(sorted(validation_keys)),
                    mean_gain=mean_gain,
                    mean_loss=mean_loss,
                    promoted_at=time.time(),
                )
            )
        trusted.sort(key=lambda r: (r.mean_loss, -r.mean_gain, r.name))
        return trusted

    def refresh_trusted_manifest(
        self,
        namespace: str,
        *,
        min_validation_keys: int = 2,
        min_mean_gain: float = 0.05,
        max_mean_loss: float = 0.12,
    ) -> list[TrustedSelfLayerRecord]:
        trusted = self.trusted_candidates(
            namespace,
            min_validation_keys=min_validation_keys,
            min_mean_gain=min_mean_gain,
            max_mean_loss=max_mean_loss,
        )
        ns_dir = self._namespace_dir(namespace)
        ns_dir.mkdir(parents=True, exist_ok=True)
        with (ns_dir / "trusted_manifest.jsonl").open("w", encoding="utf-8") as f:
            for rec in trusted:
                payload = {
                    **rec.__dict__,
                    "validation_keys": list(rec.validation_keys),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return trusted
