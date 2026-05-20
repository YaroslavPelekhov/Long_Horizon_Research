"""
NeuroSource v2 -- CROSS-SUBJECT (leave-one-subject-out) real-data plugin.

v1 lesson: within-session 4-vs-4 segment split on ONE subject is dominated by
habituation/non-stationarity -> no signal an agent can honestly find. Correct
unit = MULTI-SUBJECT: discover on N-1 subjects, a claim is TRUE iff it
replicates on the HELD-OUT subject. That tests genuinely generalizable
stimulus->physiology structure (what a real "finding" means).

Smell_Betula protocol #2591, 4 subjects (private NeuroTrend, uncontaminated).
Per subject: qEEG feature matrix `EEG--*.csv` (~1 Hz; COGN_LOAD/verdicts
dropped) + stimulus windows from `85*_EEG_NTrend_1.csv` (odor vs reference).
ACC_EEG / RX_ / ART kept on purpose as the natural motion/quality dead-end.

Still behind the domain-agnostic WorldSource interface (anti-hardcoding).
"""

from __future__ import annotations

import csv
import glob
import statistics as st
from pathlib import Path

from adapters.base import BudgetExhausted, Contrast, WorldSource

_SMELL = Path(__file__).resolve().parent.parent.parent / "smell"
EEG_BANDS = {"TH", "A1", "A2", "B1", "B2"}
SUBJECTS = ["Лавреньев", "Малышев", "Томилов", "Трушникова"]
OBS_COST = 3.0


def _is_mot(f: str) -> bool:
    return f.startswith("RX_") or f in ("ART", "ACC_EEG", "ACC_X", "ACC_Y",
                                        "ACC_Z")


def _is_neural(f: str) -> bool:
    p = f.split("_")
    return len(p) >= 2 and p[1] in EEG_BANDS and not _is_mot(f)


def _lobe(f: str) -> str:
    c = f.split("_")[0]
    return {"F": "FRONTAL", "P": "PARIETAL", "O": "OCCIPITAL",
            "C": "CENTRAL", "T": "TEMPORAL"}.get(c[:1], "OTHER")


def _load_subject(folder: Path):
    """Return (feature_header, rows[list[float]], label[list[str]]).
    label[i] in {'odor','ref','none'} for second i (1 Hz assumption)."""
    feat_csv = None
    for p in folder.glob("EEG--*.csv"):
        if "NTrend" not in p.name:
            feat_csv = p
            break
    ntrend = next(iter(folder.glob("85*_EEG_NTrend_1.csv")), None)
    if feat_csv is None or ntrend is None:
        raise FileNotFoundError(folder.name)

    rows = list(csv.reader(open(feat_csv, encoding="utf-8", newline=""),
                           delimiter="\t"))
    raw_h = rows[0]
    keep = [i for i, h in enumerate(raw_h) if h != "COGN_LOAD"]
    header = [raw_h[i] for i in keep]
    data = []
    for r in rows[1:]:
        try:
            data.append([float(r[i]) for i in keep])
        except (ValueError, IndexError):
            continue
    n = len(data)
    label = ["none"] * n
    nt = list(csv.reader(open(ntrend, encoding="utf-8", newline=""),
                         delimiter="\t"))
    for r in nt[1:]:
        if len(r) < 6:
            continue
        name = str(r[2])
        try:
            t0, t1 = int(float(r[4])), int(float(r[5]))
        except (ValueError, TypeError):
            continue
        is_ref = ("опорный" in name) or ("опорн" in name)
        kind = "ref" if is_ref else "odor"
        for t in range(t0, min(t1, n)):
            label[t] = kind
    return header, data, label


class NeuroSource(WorldSource):
    """One LOSO fold: discover on all subjects except `held_out`."""

    def __init__(self, held_out: str, budget: float = 45.0):
        self.name = f"Smell-LOSO[{held_out}]"
        self.held = held_out
        self.budget_total_ = budget
        self.spent = 0.0
        self._eid = 0
        self.log: list[tuple[int, str]] = []

        self.subj = {}
        for s in SUBJECTS:
            try:
                self.subj[s] = _load_subject(_SMELL / s)
            except FileNotFoundError:
                pass
        # common feature set across subjects
        common = None
        for s, (h, _, _) in self.subj.items():
            common = set(h) if common is None else (common & set(h))
        self.feats = sorted(common or [])
        self.disc_subj = [s for s in self.subj if s != held_out]

    # -- helpers --------------------------------------------------------
    def _vals(self, subj: str, f: str, kind: str) -> list[float]:
        h, data, lab = self.subj[subj]
        j = h.index(f)
        return [data[t][j] for t in range(len(data)) if lab[t] == kind]

    def _contrast_subj(self, subj: str, f: str):
        a = self._vals(subj, f, "odor"); b = self._vals(subj, f, "ref")
        if len(a) < 8 or len(b) < 8:
            return None
        d = st.fmean(a) - st.fmean(b)
        se = (st.pvariance(a) / len(a) + st.pvariance(b) / len(b)) ** 0.5 or 1e-9
        return d, se

    def _pooled_disc(self, f: str):
        """Mean effect over discovery subjects + cross-subject consistency."""
        ds = []
        for s in self.disc_subj:
            c = self._contrast_subj(s, f)
            if c:
                ds.append(c[0])
        if len(ds) < max(2, len(self.disc_subj) - 1):
            return None
        m = st.fmean(ds)
        sd = st.pstdev(ds) if len(ds) > 1 else 1e9
        # stable = consistent sign across discovery subjects (agent's honest
        # cross-subject pre-registration; never touches the held-out subject)
        stable = all((x > 0) == (m > 0) for x in ds) and abs(m) > sd
        return m, sd, stable

    # -- WorldSource ----------------------------------------------------
    def clusters(self) -> dict[str, list[str]]:
        c: dict[str, list[str]] = {}
        for f in self.feats:
            if _is_mot(f):
                c.setdefault("A_MOT", []).append(f)
            elif _is_neural(f):
                c.setdefault(_lobe(f), []).append(f)
        return c

    @property
    def budget_left(self) -> float:
        return self.budget_total_ - self.spent

    @property
    def budget_total(self) -> float:
        return self.budget_total_

    def observe(self, cluster: str) -> dict[str, Contrast]:
        if self.budget_left < OBS_COST:
            raise BudgetExhausted()
        self.spent += OBS_COST
        eid = self._eid; self._eid += 1
        self.log.append((eid, cluster))
        out = {}
        for f in self.clusters().get(cluster, []):
            pd = self._pooled_disc(f)
            if pd is None:
                continue
            m, sd, stable = pd
            out[f] = Contrast(delta=m, se=(sd or 1e-9), eid=eid, stable=stable)
        return out

    # -- ground truth: replication on the HELD-OUT subject --------------
    def replicates(self, statement: str) -> bool:
        s = statement.replace(" ", "")
        if not s.startswith("stim_effect(") or not s.endswith(")"):
            return False
        body = s[len("stim_effect("):-1]
        if "," not in body:
            return False
        f, sg = body.rsplit(",", 1)
        if self.held not in self.subj or f not in self.feats:
            return False
        c = self._contrast_subj(self.held, f)
        if not c:
            return False
        d, se = c
        return abs(d) > 2 * se and (1 if d > 0 else -1) == (1 if sg == "+" else -1)

    def value(self, statement: str) -> float:
        s = statement.replace(" ", "")
        if not s.startswith("stim_effect(") or not s.endswith(")"):
            return 0.0
        body = s[len("stim_effect("):-1]
        if "," not in body:
            return 0.0
        f = body.rsplit(",", 1)[0]
        if _is_mot(f):
            return 0.0
        return 3.0 if _is_neural(f) else 0.0

    def max_value(self) -> float:
        return self.oracle_value()

    def oracle_value(self) -> float:
        v = 0.0
        for f in self.feats:
            if _is_neural(f):
                if (self.replicates(f"stim_effect({f},+)")
                        or self.replicates(f"stim_effect({f},-)")):
                    v += 3.0
        return v or 1.0

    def deadend_clusters(self) -> list[str]:
        return ["A_MOT"] if "A_MOT" in self.clusters() else []

    def provenance_ok(self, statement: str, prov: list[int]) -> bool:
        if not prov:
            return False
        seen = {e for e, _ in self.log}
        return all(p in seen for p in prov)
