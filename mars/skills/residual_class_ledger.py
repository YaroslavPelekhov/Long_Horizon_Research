"""Cross-task, typed residual memory for hypothesis-language induction.

The ledger deliberately records interface-level failure geometry rather than
task identifiers, data values, column names, or answers.  It gives the core a
small persistent substrate on which it can induce a reusable operator only
after a residual family recurs across independent source tasks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any


class ResidualClassLedger:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.proposals_path = self.root / "proposals.jsonl"
        self.outcomes_path = self.root / "outcomes.jsonl"

    @staticmethod
    def _class_id(fingerprint: dict[str, Any]) -> str:
        # Residual magnitude and sample count describe an observation of a
        # class, not the class identity itself.  Including them fragments a
        # recurring numeric interface into artificial ``low``/``mixed`` loss
        # buckets and prevents a shared operator from ever reaching its
        # recurrence threshold.  Class membership is therefore defined by the
        # typed interface geometry; the full fingerprint remains stored with
        # every event for later audit.
        identity = {
            key: value
            for key, value in fingerprint.items()
            if key not in {"loss_band", "support_bucket"}
        }
        raw = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def observe(self, *, validation_key: str, fingerprint: dict[str, Any]) -> tuple[str, int]:
        class_id = self._class_id(fingerprint)
        existing = self.events()
        keys = {
            str(row.get("validation_key", ""))
            for row in existing
            if row.get("class_id") == class_id
        }
        if validation_key not in keys:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "class_id": class_id,
                    "validation_key": validation_key,
                    "fingerprint": fingerprint,
                    "created_at": time.time(),
                }, sort_keys=True) + "\n")
            keys.add(validation_key)
        return class_id, len(keys)

    def events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def claim(self, class_id: str) -> bool:
        # A semantic rejection is useful feedback, not a permanent ban on the
        # residual class.  Only a successfully quarantined operator consumes
        # the class; later evidence may produce a better proposal.
        latest_outcome: str | None = None
        if self.outcomes_path.exists():
            for line in reversed(self.outcomes_path.read_text(encoding="utf-8").splitlines()):
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("class_id") == class_id:
                    latest_outcome = str(row.get("status", ""))
                    break
        if latest_outcome is not None:
            return latest_outcome != "quarantined"
        if self.proposals_path.exists():
            for line in self.proposals_path.read_text(encoding="utf-8").splitlines():
                try:
                    if json.loads(line).get("class_id") == class_id:
                        return False
                except Exception:
                    continue
        with self.proposals_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"class_id": class_id, "created_at": time.time()}) + "\n")
        return True

    def record_outcome(self, class_id: str, status: str, detail: str = "") -> None:
        with self.outcomes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "class_id": class_id,
                "status": status,
                "detail": detail[:500],
                "created_at": time.time(),
            }) + "\n")

    def latest_outcome(self, class_id: str) -> dict[str, Any] | None:
        if not self.outcomes_path.exists():
            return None
        for line in reversed(self.outcomes_path.read_text(encoding="utf-8").splitlines()):
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("class_id") == class_id:
                return row
        return None
