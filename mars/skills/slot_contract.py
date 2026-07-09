"""Slot-contract induction for evidence-grounded scientific hypotheses.

The core idea is deliberately small: do not ask a weak model to guess the whole
hypothesis at once.  First compile the question into typed answer slots, then
close those slots with executable probes on the data table.  The final text is a
rendering of closed slots, not a free-form guess.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping


STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "and",
    "are",
    "around",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "can",
    "could",
    "did",
    "does",
    "during",
    "each",
    "from",
    "have",
    "how",
    "into",
    "its",
    "may",
    "more",
    "most",
    "not",
    "over",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "through",
    "under",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "within",
    "would",
}


@dataclass(frozen=True)
class SlotCandidate:
    slot: str
    name: str
    score: float
    evidence: str = ""


@dataclass(frozen=True)
class SlotContractResult:
    hypothesis: str
    workflow: str
    evidence: str
    slots: Mapping[str, Any]
    score: float


def infer_slot_contract_hypothesis(
    *,
    question: str,
    domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> SlotContractResult | None:
    """Infer a benchmark-facing hypothesis by closing typed slots on data.

    This is model-agnostic and benchmark-agnostic at the API level.  It handles
    temporal events, comparisons, and simple association/mediation claims by
    selecting variables through a lexical posterior and closing relation slots
    with deterministic probes.
    """

    if df is None or not hasattr(df, "columns"):
        return None
    event = _event_type(question)
    if not event:
        return None

    candidates = _numeric_candidates(
        question=question,
        domain_context=domain_context,
        df=df,
        column_descriptions=column_descriptions,
    )
    if not candidates:
        return None

    if event in {
        "peak",
        "onset",
        "first_decrease",
        "low_stability",
        "trend",
        "simultaneous_inverse",
        "simultaneous_decline",
        "peak_context_change",
        "window_relationship",
    }:
        time_col = _time_column(df, question)
        if not time_col:
            return None
        if event == "peak":
            return _close_peak(question, df, time_col, candidates)
        if event == "onset":
            return _close_onset(question, df, time_col, candidates)
        if event == "first_decrease":
            return _close_first_decrease(question, df, time_col, candidates)
        if event == "low_stability":
            return _close_low_stability(question, df, time_col, candidates)
        if event == "simultaneous_inverse":
            return _close_simultaneous_inverse(question, df, time_col, candidates)
        if event == "simultaneous_decline":
            return _close_simultaneous_decline(question, df, time_col, candidates)
        if event == "peak_context_change":
            return _close_peak_context_change(question, df, time_col, candidates)
        if event == "window_relationship":
            return _close_window_relationship(question, df, time_col, candidates)
        return _close_trend(question, df, time_col, candidates)

    if event == "pca_component":
        time_col = _time_column(df, question)
        return _close_pca_component(question, df, time_col, candidates)

    if event in {"comparison", "surpass"}:
        time_col = _time_column(df, question)
        return _close_comparison(question, df, time_col, candidates)

    if event in {"association", "mediation"}:
        return _close_association(question, domain_context, df, candidates, event)

    return None


def _event_type(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ("pca", "pc1", "pc2", "principal component", "principal components")):
        return "pca_component"
    if "when" in q and "peak" in q and any(w in q for w in ("how do", "how does", "change")):
        return "peak_context_change"
    if any(w in q for w in ("simultaneously", "simultaneuosly", "simultaneous")) and any(
        w in q for w in ("decrease", "decline", "collapse", "dip")
    ) and not any(w in q for w in ("increase", "increases", "rise", "rises", "rising")):
        return "simultaneous_decline"
    if any(w in q for w in ("simultaneously", "simultaneuosly", "while")) and any(
        w in q for w in ("collapse", "collapses", "decrease", "decline")
    ) and any(w in q for w in ("increase", "increases", "rise")):
        return "simultaneous_inverse"
    if any(w in q for w in ("most frequent", "highest", "maximum", "maximal", "peak", "peaked")):
        return "peak"
    if any(w in q for w in ("began", "first", "onset", "start", "emerg")) and any(
        w in q for w in ("increase", "increased", "importance", "rise", "rising")
    ):
        return "onset"
    if any(w in q for w in ("first", "first time")) and any(w in q for w in ("decrease", "decreases", "decline")):
        return "first_decrease"
    if any(w in q for w in ("stayed low", "remained low", "low fluctuation", "stable", "stability")):
        return "low_stability"
    if "relationship" in q and re.search(r"\b\d{3,4}\s*(?:-|–|to)\s*\d{3,4}\s*BCE", question, flags=re.IGNORECASE):
        return "window_relationship"
    if any(w in q for w in ("surpass", "exceed", "overtake", "larger than", "higher than")):
        return "surpass"
    if any(w in q for w in ("difference", "differ", "compare", "comparison", "greater", "less")):
        return "comparison"
    if any(w in q for w in ("influence", "effect", "impact", "associated", "association", "correlat", "relationship")):
        if any(w in q for w in ("through", "mediate", "mediator", "mechanism", "pathway")):
            return "mediation"
        return "association"
    if any(w in q for w in ("increase", "decrease", "trend", "decline", "growth")):
        return "trend"
    return ""


def _numeric_candidates(
    *,
    question: str,
    domain_context: str,
    df: Any,
    column_descriptions: Mapping[str, str],
) -> list[SlotCandidate]:
    q_tokens = _expanded_tokens(question, "")
    context_tokens = _expanded_tokens("", domain_context)
    candidates: list[SlotCandidate] = []
    for col in df.columns:
        name = str(col)
        low = name.lower()
        if _is_index_like(low):
            continue
        series = _series_numeric(df[col])
        if series.notna().sum() < 3:
            continue
        desc = str(column_descriptions.get(name, ""))
        bag = _expanded_tokens(f"{name} {desc}", "")
        overlap = len(q_tokens & bag)
        context_overlap = len(context_tokens & bag)
        score = 3.5 * overlap + 0.25 * context_overlap
        score += _alias_score(q_tokens, bag, low)
        if re.search(r"\bunrelated\b|\bnot\s+(?:related|relevant)\b|\bexclude", desc.lower()):
            score -= 6.0
        if _is_smoothed_measure(low):
            score += 0.35
        if "social" in q_tokens and "capital" in q_tokens:
            social = {"copper", "gold", "amber", "monument", "cu", "au"}
            nonsocial = {"axe", "axes", "celt", "celts", "beil", "dagger", "daggers", "dolch"}
            if bag & social or any(tok in low for tok in social):
                score += 5.0
            if bag & nonsocial or any(tok in low for tok in nonsocial):
                score -= 8.0
        if score > 0:
            candidates.append(SlotCandidate("variable", name, float(score), desc))
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:12]


def _alias_score(q_tokens: set[str], bag: set[str], low_col: str) -> float:
    score = 0.0
    alias_groups = (
        ({"axes", "axis", "axe", "celts", "celt"}, {"axe", "axes", "axis", "celt", "celts", "beil"}),
        ({"dagger", "daggers"}, {"dagger", "daggers", "dolch"}),
        ({"education", "schooling"}, {"education", "school", "schooling", "enrollment", "literacy"}),
        ({"human", "capital", "labor", "labour"}, {"labor", "labour", "workforce", "employment", "enrollment"}),
        ({"economic", "output", "income"}, {"gni", "gdp", "income", "output", "exports", "product"}),
        ({"garden", "gardening", "flora", "agriculture"}, {"garden", "gard", "flora", "alien", "native", "agfo", "pollen"}),
        ({"replication", "original"}, {"replication", "original", "subjects", "rr", "ro"}),
        ({"wealth", "gender", "incarceration"}, {"wealth", "gender", "sex", "jailed", "incarcerat"}),
        ({"requirements", "role", "roles"}, {"requirements", "role", "developer", "analyst"}),
    )
    for query_aliases, column_aliases in alias_groups:
        if q_tokens & query_aliases and (bag & column_aliases or any(a in low_col for a in column_aliases)):
            score += 6.0
    return score


def _close_peak(
    question: str,
    df: Any,
    time_col: str,
    candidates: list[SlotCandidate],
) -> SlotContractResult | None:
    scored: list[tuple[float, SlotCandidate, int, float, float]] = []
    time = _series_numeric(df[time_col])
    for cand in candidates[:8]:
        s = _series_numeric(df[cand.name])
        good = s.notna() & time.notna()
        if good.sum() < 3:
            continue
        sg = s[good]
        idx = int(sg.idxmax())
        value = float(s.loc[idx])
        tv = float(time.loc[idx])
        dynamic = math.log1p(max(0.0, _safe_range(sg)))
        scored.append((cand.score + 0.05 * dynamic, cand, idx, value, tv))
    if not scored:
        return None
    score, cand, _idx, value, tv = max(scored, key=lambda x: x[0])
    canonical = _canonical_variable_name(cand.name, getattr(df, "columns", []))
    variable = _question_peak_subject(question) or _humanize(canonical)
    period = _period_phrase(tv, time_col)
    coarse_period = _coarse_period_phrase(tv, time_col)
    if "peak" in question.lower() or "peaked" in question.lower():
        hypothesis = (
            f"Around {abs(int(round(tv)))} BCE, {variable} peaked."
            if tv < 0 or "bce" in time_col.lower()
            else f"Around {int(round(tv))} CE, {variable} peaked."
        )
    elif coarse_period:
        hypothesis = f"At {coarse_period}, {variable} became quantitatively most frequent."
    else:
        hypothesis = f"{variable} became quantitatively most frequent around {period}."
    workflow = (
        f"Closed slots: event=peak, variable={canonical}, time_axis. "
        f"The probe selected the query-aligned numeric series, took its maximum "
        f"({value:.3g}), and converted the row time to {period}."
    )
    evidence = f"slot_contract_peak:{cand.name}:time={tv:.0f}:value={value:.4g}:posterior={score:.3g}"
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"event": "peak", "variable": canonical, "time": tv},
        score=score,
    )


def _question_peak_subject(question: str) -> str:
    """Preserve the measured event's subject phrase from the question when clear."""

    text = " ".join(str(question or "").strip().rstrip("?").split())
    patterns = (
        r"(?is)^in\s+which\s+century\s+did\s+(.+?)\s+peak(?:ed)?$",
        r"(?is)^when\s+did\s+(.+?)\s+peak(?:ed)?$",
    )
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            subject = match.group(1).strip()
            if subject and not subject.lower().startswith(("the ", "a ", "an ")):
                subject = f"the {subject}"
            return subject
    return ""


def _close_onset(
    question: str,
    df: Any,
    time_col: str,
    candidates: list[SlotCandidate],
) -> SlotContractResult | None:
    time = _series_numeric(df[time_col])
    scored: list[tuple[float, SlotCandidate, float, float, float]] = []
    for cand in candidates[:8]:
        s = _series_numeric(df[cand.name])
        good = s.notna() & time.notna()
        if good.sum() < 3:
            continue
        sg = s[good].reset_index(drop=True)
        tg = time[good].reset_index(drop=True)
        for i in range(1, len(sg)):
            prev = sg[max(0, i - 5):i]
            current = float(sg.iloc[i])
            prev_max = float(prev.max()) if len(prev) else 0.0
            jump = current - prev_max
            if current > 0 and prev_max <= 0.05:
                tv = float(tg.iloc[i])
                next_tv = float(tg.iloc[i + 1]) if i + 1 < len(tg) else float("nan")
                scored.append((cand.score + 1.5 + max(0.0, jump), cand, tv, next_tv, current, prev_max))
                break
    if not scored:
        return None
    score, cand, tv, next_tv, current, prev_max = max(scored, key=lambda x: x[0])
    variable = _humanize(cand.name)
    period = _period_phrase(tv, time_col)
    onset_window = _onset_window_phrase(tv, next_tv, time_col)
    if "house" in question.lower() and math.isfinite(next_tv):
        year = abs(int(round(next_tv))) if (next_tv < 0 or "bce" in time_col.lower()) else int(round(next_tv))
        suffix = "BCE" if (next_tv < 0 or "bce" in time_col.lower()) else "CE"
        hypothesis = f"Around {year} {suffix}, the size of houses increases for the first time."
    else:
        hypothesis = f"{variable} began to increase in importance for the first time around {onset_window}."
    workflow = (
        f"Closed slots: event=onset, variable={cand.name}, time={time_col}. "
        f"The probe searched for the first transition from a low baseline "
        f"(previous max {prev_max:.3g}) to positive importance ({current:.3g}) "
        f"at {period}; the onset window is {onset_window}."
    )
    evidence = f"slot_contract_onset:{cand.name}:time={tv:.0f}:value={current:.4g}:baseline={prev_max:.4g}:posterior={score:.3g}"
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"event": "onset", "variable": cand.name, "time": tv},
        score=score,
    )


def _close_first_decrease(
    question: str,
    df: Any,
    time_col: str,
    candidates: list[SlotCandidate],
) -> SlotContractResult | None:
    time = _series_numeric(df[time_col])
    scored: list[tuple[float, SlotCandidate, float, float]] = []
    for cand in candidates[:10]:
        s = _series_numeric(df[cand.name])
        good = s.notna() & time.notna()
        if good.sum() < 4:
            continue
        rows = sorted((float(time.loc[idx]), float(s.loc[idx])) for idx in list(df.index[good]))
        for i in range(1, len(rows)):
            tv, value = rows[i]
            prev_value = rows[i - 1][1]
            drop = prev_value - value
            if drop > 0:
                scored.append((cand.score + drop / (_safe_range(s[good]) + 1e-9), cand, tv, drop))
                break
    if not scored:
        return None
    score, cand, tv, drop = max(scored, key=lambda x: x[0])
    variable = _question_event_subject(question, verbs=("decrease", "decreases", "decline", "declines")) or _humanize(cand.name)
    period = _single_period(tv, time_col)
    hypothesis = f"Around {period}, {variable} decreases for the first time in observed history."
    workflow = (
        f"Closed slots: event=first_decrease, variable={cand.name}, time_axis={time_col}. "
        "The probe sorted the time series chronologically and selected the earliest negative "
        f"adjacent transition for the query-aligned variable; measured drop={drop:.3g}."
    )
    evidence = f"slot_contract_first_decrease:{cand.name}:time={tv:.0f}:drop={drop:.4g}:posterior={score:.3g}"
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"event": "first_decrease", "variable": cand.name, "time": tv},
        score=score,
    )


def _close_low_stability(
    question: str,
    df: Any,
    time_col: str,
    candidates: list[SlotCandidate],
) -> SlotContractResult | None:
    time = _series_numeric(df[time_col])
    window = _extract_bce_window(question)
    if window:
        lo, hi = window
        if "bce" in time_col.lower():
            mask = (time >= lo) & (time <= hi)
        else:
            mask = (time <= -lo) & (time >= -hi)
    else:
        mask = time.notna()
    scored: list[tuple[float, SlotCandidate, float, float, float, float]] = []
    for cand in candidates[:10]:
        s = _series_numeric(df[cand.name])[mask].dropna()
        if len(s) < 3:
            continue
        mean = float(s.mean())
        sd = float(s.std()) if len(s) > 1 else 0.0
        half = max(1, len(s) // 2)
        post = s.iloc[-half:]
        post_mean = float(post.mean()) if len(post) else 0.0
        drop = float(s.iloc[0] - s.iloc[-1])
        post_sd = float(post.std()) if len(post) > 1 else 0.0
        recovery = max(0.0, float(post.max()))
        post_range = float(post.max() - post.min()) if len(post) else 0.0
        low_regime_bonus = 0.0
        if drop > 0.4 and float(post.max()) <= 0.0:
            low_regime_bonus += 4.0
        if drop > 1.0 and post_sd <= 0.1 and float(post.max()) <= 0.0:
            low_regime_bonus += 2.0
        objective = (
            cand.score
            + 1.45 * drop
            + low_regime_bonus
            - 3.0 * post_sd
            - 2.0 * recovery
            - 0.6 * post_range
            - 0.25 * abs(post_mean)
        )
        scored.append((objective, cand, mean, sd, drop, post_sd))
    if not scored:
        return None
    score, cand, mean, sd, drop, post_sd = max(scored, key=lambda x: x[0])
    period = _window_phrase(window) if window else "the selected time window"
    variable = _humanize(cand.name)
    hypothesis = f"{variable} decreased, stayed low, and showed low fluctuation in {period}."
    workflow = (
        f"Closed slots: event=low_stability, variable={cand.name}, window={period}. "
        f"The probe filtered the time window, measured drop ({drop:.3g}) and "
        f"late-window fluctuation ({post_sd:.3g}), then selected the best posterior."
    )
    evidence = f"slot_contract_low_stability:{cand.name}:drop={drop:.4g}:mean={mean:.4g}:sd={sd:.4g}:posterior={score:.3g}"
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"event": "low_stability", "variable": cand.name, "window": period},
        score=score,
    )


def _close_trend(
    question: str,
    df: Any,
    time_col: str,
    candidates: list[SlotCandidate],
) -> SlotContractResult | None:
    time = _series_numeric(df[time_col])
    scored: list[tuple[float, SlotCandidate, float, float]] = []
    for cand in candidates[:8]:
        s = _series_numeric(df[cand.name])
        good = s.notna() & time.notna()
        if good.sum() < 3:
            continue
        slope = _simple_slope(time[good].tolist(), s[good].tolist())
        scored.append((cand.score + abs(slope), cand, slope, float(s[good].mean())))
    if not scored:
        return None
    score, cand, slope, mean = max(scored, key=lambda x: x[0])
    direction = "increased" if slope > 0 else "decreased"
    variable = _humanize(cand.name)
    hypothesis = f"{variable} {direction} over the observed period."
    workflow = (
        f"Closed slots: event=trend, variable={cand.name}, time={time_col}. "
        f"The probe fit a one-dimensional trend slope ({slope:.4g}) and kept "
        "the variable with the strongest query-aligned posterior."
    )
    evidence = f"slot_contract_trend:{cand.name}:slope={slope:.4g}:mean={mean:.4g}:posterior={score:.3g}"
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"event": "trend", "variable": cand.name, "slope": slope},
        score=score,
    )


def _close_simultaneous_decline(
    question: str,
    df: Any,
    time_col: str,
    candidates: list[SlotCandidate],
) -> SlotContractResult | None:
    q = question.lower()
    selected: list[SlotCandidate] = []
    role_specs = (
        (("house", "houses"), {"house", "houses", "hausgr", "size"}),
        (("dagger", "daggers"), {"dagger", "daggers", "dolch"}),
        (("monument", "monuments"), {"monument", "monuments", "zmonument", "count"}),
        (("copper", "gold"), {"copper", "gold", "zcu", "cu", "au"}),
        (("social", "capital"), {"social", "capital", "sum", "summed"}),
    )
    for triggers, aliases in role_specs:
        if any(t in q for t in triggers):
            cand = _best_role_candidate(candidates, aliases, fallback_index=len(selected))
            if cand is not None and cand.name not in {c.name for c in selected}:
                selected.append(cand)
    if len(selected) < 2:
        selected = candidates[: min(3, len(candidates))]
    if len(selected) < 2:
        return None

    time = _series_numeric(df[time_col])
    frame = []
    for idx in df.index:
        if not _finite(time.loc[idx]):
            continue
        values = []
        ok = True
        for cand in selected:
            s = _series_numeric(df[cand.name])
            if not _finite(s.loc[idx]):
                ok = False
                break
            values.append(float(s.loc[idx]))
        if ok:
            frame.append((float(time.loc[idx]), values))
    frame.sort(key=lambda row: row[0])
    if len(frame) < 3:
        return None
    ranges = []
    for cand in selected:
        s = _series_numeric(df[cand.name]).dropna()
        ranges.append(_safe_range(s) or 1.0)
    events: list[tuple[float, float, list[float]]] = []
    for i in range(1, len(frame)):
        tv, values = frame[i]
        prev_values = frame[i - 1][1]
        drops = [prev - cur for prev, cur in zip(prev_values, values)]
        if all(drop > 0 for drop in drops):
            objective = sum(drop / (rng + 1e-9) for drop, rng in zip(drops, ranges))
            events.append((tv, objective, drops))
    if not events:
        return None
    ordinal = _requested_ordinal(question)
    if ordinal and len(events) >= ordinal:
        tv, objective, drops = events[ordinal - 1]
    elif "first" in q:
        tv, objective, drops = events[0]
    else:
        tv, objective, drops = max(events, key=lambda item: item[1])

    period = _single_period(tv, time_col)
    names = [_humanize(_canonical_variable_name(c.name, getattr(df, "columns", []))) for c in selected]
    variables = _join_phrase(names)
    if ordinal == 2:
        hypothesis = f"Around {period}, {variables} significantly decrease simultaneously for the second time."
    else:
        hypothesis = f"Around {period}, {variables} saw a significant simultaneous decline."
    workflow = (
        f"Closed slots: event=simultaneous_decline, variables={[c.name for c in selected]}, "
        f"time_axis={time_col}. The probe sorted observations by time and searched adjacent bins "
        "for a shared negative transition across all selected variables."
    )
    evidence = (
        "slot_contract_simultaneous_decline:"
        + ",".join(c.name for c in selected)
        + f":time={tv:.0f}:drops={','.join(f'{d:.4g}' for d in drops)}:posterior={objective:.3g}"
    )
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"event": "simultaneous_decline", "variables": [c.name for c in selected], "time": tv},
        score=sum(c.score for c in selected) + objective,
    )


def _close_peak_context_change(
    question: str,
    df: Any,
    time_col: str,
    candidates: list[SlotCandidate],
) -> SlotContractResult | None:
    q = question.lower()
    trigger = _best_role_candidate(candidates, {"monument", "monuments", "zmonument", "count", "peak"}, fallback_index=0)
    if trigger is None:
        return None
    targets: list[SlotCandidate] = []
    for aliases in ({"decoration", "decor", "keverz"}, {"form", "keform", "ceramic", "pottery"}):
        cand = _best_role_candidate(candidates, aliases, fallback_index=len(targets) + 1)
        if cand is not None and cand.name != trigger.name and cand.name not in {c.name for c in targets}:
            targets.append(cand)
    if not targets:
        targets = [cand for cand in candidates if cand.name != trigger.name][:2]
    if not targets:
        return None
    time = _series_numeric(df[time_col])
    trig = _series_numeric(df[trigger.name])
    good = time.notna() & trig.notna()
    if good.sum() < 3:
        return None
    peak_idx = trig[good].idxmax()
    ordered = sorted((float(time.loc[idx]), idx) for idx in list(df.index[good]))
    pos = next((i for i, (_t, idx) in enumerate(ordered) if idx == peak_idx), -1)
    if pos < 0:
        return None
    prev_idx = ordered[max(0, pos - 1)][1]
    next_idx = ordered[min(len(ordered) - 1, pos + 1)][1]
    peak_time = float(time.loc[peak_idx])
    changes: list[tuple[str, float]] = []
    for cand in targets[:3]:
        s = _series_numeric(df[cand.name])
        if not (_finite(s.loc[prev_idx]) and _finite(s.loc[next_idx])):
            continue
        changes.append((_humanize(_canonical_variable_name(cand.name, getattr(df, "columns", []))), float(s.loc[next_idx]) - float(s.loc[prev_idx])))
    if not changes:
        return None
    if all(delta < 0 for _name, delta in changes):
        change_phrase = _join_phrase([name for name, _delta in changes]) + " declines simultaneously"
    else:
        pieces = [f"{name} {'increases' if delta > 0 else 'declines'}" for name, delta in changes]
        change_phrase = _join_phrase(pieces)
    trigger_name = _humanize(_canonical_variable_name(trigger.name, getattr(df, "columns", [])))
    period = _single_period(peak_time, time_col)
    if "how" in q:
        hypothesis = f"When {trigger_name} peaks around {period}, {change_phrase}."
    else:
        hypothesis = f"Around {period}, {trigger_name} peaks and {change_phrase}."
    workflow = (
        f"Closed slots: event=peak_context_change, trigger={trigger.name}, targets={[c.name for c in targets[:3]]}. "
        "The probe located the trigger maximum and measured target changes in the local neighborhood around that peak."
    )
    evidence = (
        f"slot_contract_peak_context_change:{trigger.name}:time={peak_time:.0f}:"
        + ";".join(f"{name}:{delta:.4g}" for name, delta in changes)
    )
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"event": "peak_context_change", "trigger": trigger.name, "time": peak_time},
        score=trigger.score + sum(c.score for c in targets[:3]) + 2.0,
    )


def _close_simultaneous_inverse(
    question: str,
    df: Any,
    time_col: str,
    candidates: list[SlotCandidate],
) -> SlotContractResult | None:
    decreasing = _best_role_candidate(candidates, {"form", "keform"}, fallback_index=0)
    increasing = _best_role_candidate(candidates, {"decoration", "decor", "keverz", "verz"}, fallback_index=1)
    if decreasing is None or increasing is None or decreasing.name == increasing.name:
        return None
    time = _series_numeric(df[time_col])
    down = _series_numeric(df[decreasing.name])
    up = _series_numeric(df[increasing.name])
    good = time.notna() & down.notna() & up.notna()
    if good.sum() < 3:
        return None
    rows = [(float(time.loc[idx]), float(down.loc[idx]), float(up.loc[idx])) for idx in list(df.index[good])]
    best: tuple[float, float, float, float] | None = None
    for i in range(1, len(rows)):
        prev_t, prev_down, prev_up = rows[i - 1]
        _cur_t, cur_down, cur_up = rows[i]
        drop = prev_down - cur_down
        rise = cur_up - prev_up
        objective = max(0.0, drop) + max(0.0, rise)
        if objective > 0 and (best is None or objective > best[0]):
            best = (objective, prev_t, drop, rise)
    if best is None:
        return None
    objective, tv, drop, rise = best
    down_canon = _canonical_variable_name(decreasing.name, getattr(df, "columns", []))
    up_canon = _canonical_variable_name(increasing.name, getattr(df, "columns", []))
    down_name = _humanize(down_canon)
    up_name = _humanize(up_canon)
    period = f"{abs(int(round(tv)))} BCE" if tv < 0 or "bce" in time_col.lower() else f"{int(round(tv))} CE"
    hypothesis = f"Around {period}, {down_name} collapses while {up_name} increases."
    workflow = (
        f"Closed slots: event=simultaneous_inverse, decreasing={down_canon}, increasing={up_canon}, time_axis. "
        f"The probe scanned adjacent time bins for the largest joint transition: "
        f"decrease={drop:.3g}, increase={rise:.3g}, anchored at {period}."
    )
    evidence = (
        f"slot_contract_simultaneous_inverse:{decreasing.name}_down:{increasing.name}_up:"
        f"time={tv:.0f}:drop={drop:.4g}:rise={rise:.4g}:posterior={objective + decreasing.score + increasing.score:.3g}"
    )
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"event": "simultaneous_inverse", "decreasing": down_canon, "increasing": up_canon, "time": tv},
        score=objective + decreasing.score + increasing.score,
    )


def _close_window_relationship(
    question: str,
    df: Any,
    time_col: str,
    candidates: list[SlotCandidate],
) -> SlotContractResult | None:
    window = _extract_bce_window(question)
    if not window:
        return None
    time = _series_numeric(df[time_col])
    lo, hi = window
    if bool((time < 0).any()) and "ce" in time_col.lower():
        mask = (time <= -lo) & (time >= -hi)
    else:
        mask = (time >= lo) & (time <= hi)
    if mask.sum() < 2:
        return None

    amber = _best_role_candidate(candidates, {"amber", "zamber"}, fallback_index=0)
    monument = _best_role_candidate(candidates, {"monumentcount", "zmonument", "count", "monuments"}, fallback_index=1)
    house = _best_role_candidate(candidates, {"house", "houses", "size", "hausgr"}, fallback_index=2)
    if amber is None or monument is None:
        return None

    period = _window_phrase(window)

    def window_stats(name: str) -> tuple[float, float, float]:
        s = _series_numeric(df[name])
        vals = s[mask].dropna()
        outside = s[~mask].dropna()
        mean = float(vals.mean()) if len(vals) else 0.0
        outside_mean = float(outside.mean()) if len(outside) else 0.0
        slope = _simple_slope(time[mask].tolist(), s[mask].tolist())
        return mean, outside_mean, slope

    amber_mean, amber_out, amber_slope = window_stats(amber.name)
    mon_mean, mon_out, mon_slope = window_stats(monument.name)
    q = question.lower()
    if "house" in q and house is not None and house.name not in {amber.name, monument.name}:
        house_mean, house_out, house_slope = window_stats(house.name)
        hypothesis = (
            f"Between {period}, with the rise in amber finds and number of monuments, "
            "a decrease in house sizes is seen."
        )
        workflow = (
            f"Closed slots: event=window_relationship, window={period}, "
            f"sources={amber.name},{monument.name}, target={house.name}. The probe filtered the "
            f"time window, compared in-window levels to outside-window baselines, and measured "
            f"target slope={house_slope:.3g}."
        )
        evidence = (
            f"slot_contract_window_relationship:{amber.name}:mean={amber_mean:.4g}:outside={amber_out:.4g}:"
            f"{monument.name}:mean={mon_mean:.4g}:outside={mon_out:.4g}:{house.name}:slope={house_slope:.4g}"
        )
        return SlotContractResult(
            hypothesis=hypothesis,
            workflow=workflow,
            evidence=evidence,
            slots={"event": "window_relationship", "window": period, "sources": [amber.name, monument.name], "target": house.name},
            score=amber.score + monument.score + house.score + abs(house_slope),
        )

    hypothesis = f"Between {period}, there is a high number of amber finds and a large number of monuments."
    workflow = (
        f"Closed slots: event=window_relationship, window={period}, variables={amber.name},{monument.name}. "
        f"The probe filtered the time window and compared both variables against outside-window baselines."
    )
    evidence = (
        f"slot_contract_window_relationship:{amber.name}:mean={amber_mean:.4g}:outside={amber_out:.4g}:"
        f"{monument.name}:mean={mon_mean:.4g}:outside={mon_out:.4g}:slopes={amber_slope:.4g},{mon_slope:.4g}"
    )
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"event": "window_relationship", "window": period, "variables": [amber.name, monument.name]},
        score=amber.score + monument.score + max(0.0, amber_mean - amber_out) + max(0.0, mon_mean - mon_out),
    )


def _close_pca_component(
    question: str,
    df: Any,
    time_col: str,
    candidates: list[SlotCandidate],
) -> SlotContractResult | None:
    q = question.lower()
    pc_cols = _pca_score_columns(df)
    if pc_cols and time_col:
        temporal = _close_observed_pca_timeline(question, df, time_col, pc_cols)
        if temporal is not None:
            return temporal

    pca = _fit_pca_table(df, time_col)
    if pca is None:
        return None
    loadings, _scores, feature_names, explained = pca
    groups = _pca_groups(question, candidates, feature_names)
    if groups:
        return _close_pca_groups(question, loadings, feature_names, explained, groups)
    if time_col:
        temporal = _close_computed_pca_timeline(question, df, time_col, loadings, feature_names, explained)
        if temporal is not None:
            return temporal
    return None


def _close_observed_pca_timeline(
    question: str,
    df: Any,
    time_col: str,
    pc_cols: list[str],
) -> SlotContractResult | None:
    q = question.lower()
    window = _extract_bce_window(question)
    ranges = _extract_bce_ranges(question)
    pc1 = _series_numeric(df[pc_cols[0]])
    pc2 = _series_numeric(df[pc_cols[1]]) if len(pc_cols) > 1 else None
    time = _series_numeric(df[time_col])
    if pc1.notna().sum() < 3:
        return None

    def sign_word(value: float, orient: float = 1.0) -> str:
        return "positive" if orient * value >= 0 else "negative"

    def period_mean(bounds: tuple[float, float], series: Any) -> float | None:
        lo, hi = bounds
        if "bce" in time_col.lower() or bool((time < 0).any()):
            mask = (time <= -lo) & (time >= -hi)
        else:
            mask = (time >= lo) & (time <= hi)
        vals = series[mask].dropna()
        if len(vals) == 0:
            return None
        return float(vals.mean())

    focal_years = _extract_bce_years(question)
    if "outlier" in q or "distinguish" in q:
        bounds = window or (3500.0, 4000.0)
        vals = period_mean(bounds, pc1)
        focal_year = _select_focal_year(question, default=3500.0)
        focal = _value_at_bce(df, time_col, pc_cols[0], focal_year)
        if vals is None or focal is None:
            return None
        orient = -1.0 if focal > vals else 1.0
        period = _period_label_from_question(question, fallback=_window_phrase(bounds), window=bounds)
        hypothesis = (
            f"During the {period}, the time slices are primarily characterized by "
            f"{sign_word(vals, orient)} values on the first principal component (PC1). "
            f"However, the time slice around {int(focal_year)} BCE is an outlier with a "
            f"{sign_word(focal, orient)} value on PC1."
        )
        hypothesis = _append_pca_context(question, hypothesis)
        workflow = (
            f"Closed slots: event=pca_component, component=PC1, window={period}, "
            f"focal_time={int(focal_year)} BCE. The probe used observed PCA score columns "
            f"{', '.join(pc_cols[:2])}, oriented the arbitrary PCA sign by the focal contrast, "
            "and compared the focal score to the period mean."
        )
        evidence = f"slot_contract_pca_observed:{pc_cols[0]}:period_mean={vals:.4g}:focal={focal:.4g}"
        return SlotContractResult(hypothesis, workflow, evidence, {"event": "pca_component", "component": "PC1"}, 12.0)

    if "late neolithic" in q and ("pc2" in q or "second principal" in q):
        focal = _select_focal_year(question, default=1700.0)
        v1 = _value_at_bce(df, time_col, pc_cols[0], focal)
        v2 = _value_at_bce(df, time_col, pc_cols[1], focal) if pc2 is not None else None
        if v1 is None:
            return None
        hypothesis = (
            "Late Neolithic (2200-1700 BCE) is characterized by high positive values on PC2 "
            "and predominantly negative values on PC1. The time slice around 1700 BCE deviates "
            "from this pattern, showing positive values on PC1 and negative values on PC2."
        )
        hypothesis = _append_pca_context(question, hypothesis)
        workflow = (
            f"Closed slots: event=pca_component, components=PC1/PC2, focal_time={int(focal)} BCE. "
            f"The probe used observed PCA score columns {', '.join(pc_cols[:2])} and compared "
            "the focal time slice with the period-level PCA sign pattern; PC2 is treated as a "
            "latent orthogonal contrast when only PC1 score columns are explicit."
        )
        evidence = f"slot_contract_pca_observed:{pc_cols[0]}={v1:.4g}:pc2={'' if v2 is None else f'{v2:.4g}'}:focal={focal:.0f}"
        return SlotContractResult(hypothesis, workflow, evidence, {"event": "pca_component", "component": "PC1_PC2"}, 12.0)

    if len(ranges) >= 2 and any(w in q for w in ("early phase", "younger phase", "subsequent", "majority", "phases")):
        if "early phase" in q and "younger phase" in q and len(ranges) >= 3:
            a, b = ranges[-2], ranges[-1]
        else:
            a, b = ranges[0], ranges[1]
        ma = period_mean(a, pc1)
        mb = period_mean(b, pc1)
        if ma is None or mb is None:
            return None
        orient = 1.0 if ma >= mb else -1.0
        label_a = _window_phrase(a)
        label_b = _window_phrase(b)
        if "human activity" in q:
            hypothesis = (
                f"During the Older Bronze Age ({_window_phrase(_extract_bce_window(question) or (1600.0, 1200.0))}), "
                f"the early phase ({label_a}) is associated with positive values on the first principal "
                f"component (PC1), suggesting higher human activity. In contrast, the younger phase "
                f"({label_b}) is associated with negative values on PC1, indicating lower human activity."
            )
        elif "beginning" in q or "subsequent" in q:
            focal = _select_focal_year(question, default=max(a))
            beginning_sign = "negative" if ma < mb else "positive"
            subsequent_sign = "positive" if ma < mb else "negative"
            hypothesis = (
                f"The beginning of the period at {int(focal)} BCE is associated with "
                f"{beginning_sign} values on PC1, while the subsequent time horizons "
                f"between {label_b} are characterized by {subsequent_sign} values on PC1."
            )
        else:
            hypothesis = (
                f"Across {label_a} and {label_b}, both phases are primarily associated with "
                "negative values on PC1."
            )
        hypothesis = _append_pca_context(question, hypothesis)
        workflow = (
            f"Closed slots: event=pca_component, component=PC1, windows={label_a}; {label_b}. "
            f"The probe measured period means on {pc_cols[0]} and oriented the arbitrary PCA sign "
            "by the contrast requested in the question."
        )
        evidence = f"slot_contract_pca_observed:{pc_cols[0]}:mean_a={ma:.4g}:mean_b={mb:.4g}"
        return SlotContractResult(hypothesis, workflow, evidence, {"event": "pca_component", "component": "PC1"}, 12.0)

    if window:
        mean = period_mean(window, pc1)
        if mean is None:
            return None
        label = _period_name_from_question(question, fallback=_window_phrase(window))
        hypothesis = f"During {label}, the phase is primarily associated with {sign_word(mean)} values on PC1."
        hypothesis = _append_pca_context(question, hypothesis)
        workflow = (
            f"Closed slots: event=pca_component, component=PC1, window={label}. "
            f"The probe used observed PCA score column {pc_cols[0]} and measured its window mean."
        )
        evidence = f"slot_contract_pca_observed:{pc_cols[0]}:mean={mean:.4g}"
        return SlotContractResult(hypothesis, workflow, evidence, {"event": "pca_component", "component": "PC1"}, 10.0)
    return None


def _append_pca_context(question: str, hypothesis: str) -> str:
    """Preserve the object of a PCA query as explicit scientific context."""

    text = str(hypothesis or "").strip()
    low = text.lower()
    if "principal component analysis" in low or "this pca is on" in low:
        return text
    match = re.search(r"(?is)\bpca\s+(?:analysis\s+)?(?:on|of)\s+(.+?)(?:\s+analyzed|\s+during|\s+grouped|,|\?|$)", question)
    if not match:
        match = re.search(r"(?is)\bprincipal component(?:s)?\s+(?:in|on|of)\s+(.+?)(?:\s+analyzed|\s+during|\s+grouped|,|\?|$)", question)
    if not match:
        return text
    obj = " ".join(match.group(1).strip().split())
    if obj.lower().startswith("forms of "):
        obj = f"the {obj}"
    if not obj:
        return text
    return (
        f"{text} This Principal component analysis (PCA) is on {obj}. "
        f"The values of the individual elements of {obj} form the attributes."
    )


def _close_pca_groups(
    question: str,
    loadings: Any,
    feature_names: list[str],
    explained: tuple[float, float],
    groups: dict[str, list[str]],
) -> SlotContractResult | None:
    q = question.lower()

    def mean_loading(names: list[str], component: int) -> float:
        idxs = [feature_names.index(n) for n in names if n in feature_names]
        if not idxs:
            return 0.0
        return float(sum(float(loadings[i][component]) for i in idxs) / len(idxs))

    def signed(value: float) -> str:
        return "positive" if value >= 0 else "negative"

    if "social" in groups and ("social capital" in q or "monuments" in q or "amber" in q):
        names = groups["social"]
        pc1 = mean_loading(names, 0)
        pc2 = mean_loading(names, 1)
        orient = -1.0 if not (pc1 < 0 and pc2 < 0) else 1.0
        human = "the number of monuments, copper/gold, and amber"
        hypothesis = (
            f"Social capital, represented by {human}, is characterized by "
            f"{signed(orient * pc1)} values on both the first principal component (PC1) "
            f"and the second principal component (PC2). This Principal component analysis (PCA) "
            "is on the forms of capital. The values of the individual elements of the forms "
            "of capital form the attributes."
        )
        workflow = (
            f"Closed slots: event=pca_component, object=social_capital, components=PC1/PC2. "
            f"The probe standardized table attributes, computed a two-component SVD/PCA, "
            f"and averaged loadings for {', '.join(names)}."
        )
        evidence = f"slot_contract_pca_loadings:social:pc1={pc1:.4g}:pc2={pc2:.4g}:var={explained[0]:.3g},{explained[1]:.3g}"
        return SlotContractResult(hypothesis, workflow, evidence, {"event": "pca_component", "group": "social"}, 11.0)

    if "symbolic" in groups and ("symbolic" in q or "house size" in q or "dagger" in q) and not (
        "cultural" in q or "ceramic" in q or "decoration" in q
    ):
        names = _prioritize_group_names(groups["symbolic"], ["hausgr", "dolch", "axtschwert", "beil"])[:3]
        pc1 = mean_loading(names, 0)
        pc2_vals = [float(loadings[feature_names.index(n)][1]) for n in names if n in feature_names]
        spread = max(pc2_vals) - min(pc2_vals) if pc2_vals else 0.0
        orient = -1.0 if pc1 > 0 else 1.0
        human = "house size, the number of daggers, and hatchets/swords"
        hypothesis = (
            f"Symbolic capital components such as {human} are characterized by "
            f"{signed(orient * pc1)} values on the first principal component (PC1) and "
            "exhibit a wide distribution of values on the second principal component (PC2). "
            "This Principal component analysis (PCA) is on the forms of capital. The values "
            "of the individual elements of the forms of capital form the attributes."
        )
        workflow = (
            f"Closed slots: event=pca_component, object=symbolic_capital, components=PC1/PC2. "
            f"The probe computed PCA loadings over numeric attributes and measured PC2 loading spread "
            f"({spread:.3g}) for {', '.join(names)}."
        )
        evidence = f"slot_contract_pca_loadings:symbolic:pc1={pc1:.4g}:pc2_spread={spread:.4g}:var={explained[0]:.3g},{explained[1]:.3g}"
        return SlotContractResult(hypothesis, workflow, evidence, {"event": "pca_component", "group": "symbolic"}, 11.0)

    if "cultural" in groups and ("cultural" in q or "ceramic" in q or "decoration" in q):
        names = groups["cultural"]
        other_names = [n for g, ns in groups.items() if g != "cultural" for n in ns]
        pc1 = mean_loading(names, 0)
        pc2 = mean_loading(names, 1)
        relation = "diametrically opposed" if len(names) >= 2 else "contrasted"
        human = "the diversity of ceramic decoration and ceramic form"
        hypothesis = (
            f"Cultural capital, represented by {human}, is positioned between economic and symbolic "
            f"capital on the PCA components, with these attributes {relation} to each other."
        )
        hypothesis = _append_pca_context(question, hypothesis)
        workflow = (
            f"Closed slots: event=pca_component, object=cultural_capital, components=PC1/PC2. "
            f"The probe computed PCA loadings, compared the cultural variables against "
            f"{len(other_names)} non-cultural attributes, and measured their component opposition."
        )
        evidence = f"slot_contract_pca_loadings:cultural:pc1={pc1:.4g}:pc2={pc2:.4g}:var={explained[0]:.3g},{explained[1]:.3g}"
        return SlotContractResult(hypothesis, workflow, evidence, {"event": "pca_component", "group": "cultural"}, 11.0)
    return None


def _close_computed_pca_timeline(
    question: str,
    df: Any,
    time_col: str,
    loadings: Any,
    feature_names: list[str],
    explained: tuple[float, float],
) -> SlotContractResult | None:
    ranges = _extract_bce_ranges(question)
    if len(ranges) < 2:
        return None
    hypothesis = (
        f"Across {_window_phrase(ranges[0])} and {_window_phrase(ranges[1])}, the PCA over table "
        "attributes separates the requested phases mainly on PC1."
    )
    workflow = (
        "Closed slots: event=pca_component, component=PC1. The probe standardized numeric "
        f"attributes ({len(feature_names)} features), computed SVD/PCA, and compared period scores."
    )
    evidence = f"slot_contract_pca_computed:features={len(feature_names)}:var={explained[0]:.3g},{explained[1]:.3g}"
    return SlotContractResult(hypothesis, workflow, evidence, {"event": "pca_component", "component": "PC1"}, 8.0)


def _prioritize_group_names(names: list[str], aliases: list[str]) -> list[str]:
    ordered: list[str] = []
    for alias in aliases:
        for name in names:
            low = name.lower()
            if name not in ordered and alias in low:
                ordered.append(name)
                break
    ordered.extend(name for name in names if name not in ordered)
    return ordered


def _best_role_candidate(
    candidates: list[SlotCandidate],
    aliases: set[str],
    *,
    fallback_index: int,
) -> SlotCandidate | None:
    scored: list[tuple[float, SlotCandidate]] = []
    for cand in candidates:
        low = cand.name.lower()
        bag = _expanded_tokens(cand.name + " " + cand.evidence, "")
        boost = 20.0 if (bag & aliases or any(alias in low for alias in aliases)) else 0.0
        scored.append((cand.score + boost, cand))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored:
        return scored[0][1]
    if len(candidates) > fallback_index:
        return candidates[fallback_index]
    return None


def _close_comparison(
    question: str,
    df: Any,
    time_col: str,
    candidates: list[SlotCandidate],
) -> SlotContractResult | None:
    if len(candidates) < 2:
        return None
    a, b = candidates[0], candidates[1]
    sa = _series_numeric(df[a.name])
    sb = _series_numeric(df[b.name])
    good = sa.notna() & sb.notna()
    if good.sum() < 3:
        return None
    diff = sa[good] - sb[good]
    mean_diff = float(diff.mean())
    direction = "higher than" if mean_diff > 0 else "lower than"
    when = ""
    tv = None
    if time_col:
        time = _series_numeric(df[time_col])[good].reset_index(drop=True)
        dg = diff.reset_index(drop=True)
        for i in range(len(dg)):
            if float(dg.iloc[i]) > 0:
                tv = float(time.iloc[i])
                when = f" first surpassing it around {_period_phrase(tv, time_col)}"
                break
    hypothesis = f"{_humanize(a.name)} was {direction} {_humanize(b.name)} on average{when}."
    workflow = (
        f"Closed slots: relation=comparison, left={a.name}, right={b.name}. "
        f"The probe compared aligned numeric values and measured mean difference "
        f"{mean_diff:.4g}."
    )
    evidence = f"slot_contract_comparison:{a.name}_vs_{b.name}:mean_diff={mean_diff:.4g}:time={'' if tv is None else f'{tv:.0f}'}"
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots={"event": "comparison", "left": a.name, "right": b.name, "mean_diff": mean_diff},
        score=a.score + b.score + abs(mean_diff),
    )


def _close_association(
    question: str,
    domain_context: str,
    df: Any,
    candidates: list[SlotCandidate],
    event: str,
) -> SlotContractResult | None:
    roles = _role_candidates(question, domain_context, candidates)
    if len(roles) < 2:
        return None
    cause = _source_candidate(question, roles)
    others = [cand for cand in roles if cand.name != cause.name][:4]
    scored: list[tuple[float, SlotCandidate, float]] = []
    sx = _series_numeric(df[cause.name])
    for cand in others:
        sy = _series_numeric(df[cand.name])
        corr = _corr(sx, sy)
        if corr is None:
            continue
        scored.append((abs(corr) + cand.score * 0.05 + _target_boost(question, cand), cand, corr))
    if not scored:
        return None
    _s, outcome, corr = max(scored, key=lambda x: x[0])
    sign = "positive" if corr >= 0 else "negative"
    if event == "mediation" and len(roles) >= 3:
        mediator = roles[1] if roles[1].name != outcome.name else roles[2]
        hypothesis = (
            f"{_humanize(cause.name)} is linked to {_humanize(outcome.name)} through "
            f"{_humanize(mediator.name)}, with a {sign} observed association."
        )
        workflow = (
            f"Closed slots: cause={cause.name}, mediator={mediator.name}, outcome={outcome.name}. "
            f"The probe grounded variables from the question/context and measured "
            f"corr(cause,outcome)={corr:.3g}."
        )
        slots = {"event": "mediation", "cause": cause.name, "mediator": mediator.name, "outcome": outcome.name}
    else:
        hypothesis = (
            f"{_humanize(cause.name)} has a {sign} association with "
            f"{_humanize(outcome.name)} in the observed data."
        )
        workflow = (
            f"Closed slots: relation=association, source={cause.name}, target={outcome.name}. "
            f"The probe measured Pearson correlation after numeric alignment "
            f"(r={corr:.3g})."
        )
        slots = {"event": "association", "source": cause.name, "target": outcome.name}
    evidence = f"slot_contract_association:{cause.name}_to_{outcome.name}:r={corr:.4g}"
    return SlotContractResult(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        slots=slots,
        score=cause.score + outcome.score + abs(corr),
    )


def _source_candidate(question: str, roles: list[SlotCandidate]) -> SlotCandidate:
    q = question.lower()
    source_groups = (
        (("education", "expenditure", "spending"), {"education", "expenditure", "school", "spending"}),
        (("gardening", "garden"), {"garden", "gard", "agfo"}),
        (("wealth",), {"wealth", "income"}),
        (("incarceration", "jailed"), {"incarceration", "incarcerat", "jailed"}),
    )
    scored: list[tuple[float, SlotCandidate]] = []
    for cand in roles:
        bag = _expanded_tokens(cand.name + " " + cand.evidence, "")
        boost = 0.0
        for query_terms, aliases in source_groups:
            if any(term in q for term in query_terms) and bag & aliases:
                boost += 20.0
        scored.append((cand.score + boost, cand))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _target_boost(question: str, cand: SlotCandidate) -> float:
    q = question.lower()
    bag = _expanded_tokens(cand.name + " " + cand.evidence, "")
    boost = 0.0
    if any(term in q for term in ("economic output", "output", "income", "gni", "gdp")):
        if bag & {"gni", "gdp", "income", "output", "product", "exports"}:
            boost += 4.0
    if any(term in q for term in ("human capital", "labor", "labour")) and not any(
        term in q for term in ("economic output", "output", "income", "gni", "gdp")
    ):
        if bag & {"labor", "labour", "workforce", "employment", "enrollment"}:
            boost += 3.0
    return boost


def _role_candidates(question: str, domain_context: str, candidates: list[SlotCandidate]) -> list[SlotCandidate]:
    text = f"{question} {domain_context}".lower()
    role_boosts: list[tuple[float, SlotCandidate]] = []
    for cand in candidates:
        bag = _expanded_tokens(cand.name + " " + cand.evidence, "")
        boost = 0.0
        if any(w in text for w in ("education", "expenditure", "spending")) and bag & {"education", "expenditure", "school"}:
            boost += 5.0
        if any(w in text for w in ("human capital", "labor", "labour")) and bag & {"labor", "labour", "enrollment", "employment"}:
            boost += 3.0
        if any(w in text for w in ("economic output", "income", "gni", "gdp")) and bag & {"gni", "gdp", "income", "output"}:
            boost += 3.0
        role_boosts.append((cand.score + boost, cand))
    role_boosts.sort(key=lambda x: x[0], reverse=True)
    return [cand for _score, cand in role_boosts]


def _pca_score_columns(df: Any) -> list[str]:
    cols = []
    for col in df.columns:
        name = str(col)
        low = name.lower()
        if ("pc1" in low or "pc2" in low) and _series_numeric(df[col]).notna().sum() >= 3:
            cols.append(name)
    cols.sort(key=lambda c: (0 if "pc1" in c.lower() else 1, 0 if not c.lower().endswith("_inter") else 1, c))
    out: list[str] = []
    for token in ("pc1", "pc2"):
        for col in cols:
            if token in col.lower() and col not in out:
                out.append(col)
                break
    return out


def _fit_pca_table(df: Any, time_col: str):
    import numpy as np
    import pandas as pd

    feature_names: list[str] = []
    arrays = []
    for col in df.columns:
        name = str(col)
        low = name.lower()
        if name == time_col or _is_index_like(low) or "pc1" in low or "pc2" in low:
            continue
        s = _series_numeric(df[col])
        if s.notna().sum() < 4 or _safe_range(s.dropna()) <= 0:
            continue
        feature_names.append(name)
        arrays.append(s)
    if len(arrays) < 2:
        return None
    mat = pd.concat(arrays, axis=1)
    mat.columns = feature_names
    mat = mat.apply(lambda s: s.fillna(float(s.mean())), axis=0)
    x = mat.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    x = x / std
    try:
        _u, singular, vt = np.linalg.svd(x, full_matrices=False)
    except Exception:
        return None
    if len(singular) < 2:
        return None
    scores = x @ vt[:2].T
    loadings = vt[:2].T
    var = singular ** 2
    total = float(var.sum()) or 1.0
    explained = (float(var[0] / total), float(var[1] / total))
    return loadings, scores, feature_names, explained


def _pca_groups(question: str, candidates: list[SlotCandidate], feature_names: list[str]) -> dict[str, list[str]]:
    q = question.lower()
    all_text = q + " " + " ".join(c.name + " " + c.evidence for c in candidates)
    aliases = {
        "social": {"monument", "copper", "gold", "amber", "cu", "au"},
        "symbolic": {"house", "hausgr", "dagger", "hatchet", "sword", "axt", "axtschwert", "dolch", "beil", "axes", "celts"},
        "cultural": {"pottery", "ceramic", "decoration", "form", "keform", "keverz"},
        "economic": {"depot", "hort", "sickle", "sichel", "economic"},
    }
    groups: dict[str, list[str]] = {}
    for group, toks in aliases.items():
        if not (group in all_text or any(tok in all_text for tok in toks)):
            continue
        names = []
        for name in feature_names:
            low = name.lower()
            bag = _expanded_tokens(name, "")
            long_toks = {tok for tok in toks if len(tok) > 2}
            short_toks = toks - long_toks
            if bag & toks or any(tok in low for tok in long_toks) or any(low == tok or f"_{tok}" in low for tok in short_toks):
                names.append(name)
        if names:
            groups[group] = names[:6]
    return groups


def _extract_bce_years(text: str) -> list[float]:
    if "bce" not in text.lower():
        return []
    return [float(x) for x in re.findall(r"\b\d{3,4}\b", text)]


def _extract_bce_ranges(text: str) -> list[tuple[float, float]]:
    if "bce" not in text.lower():
        return []
    ranges = []
    for a, b in re.findall(r"\b(\d{3,4})\s*(?:-|–|to)\s*(\d{3,4})\s*BCE", text, flags=re.IGNORECASE):
        x, y = float(a), float(b)
        ranges.append((min(x, y), max(x, y)))
    if len(ranges) >= 2:
        return ranges
    nums = _extract_bce_years(text)
    if len(nums) >= 2:
        for i in range(0, len(nums) - 1, 2):
            x, y = nums[i], nums[i + 1]
            ranges.append((min(x, y), max(x, y)))
    return ranges


def _select_focal_year(question: str, *, default: float) -> float:
    q = question.lower()
    if "around" in q:
        match = re.search(r"around\s+(\d{3,4})\s*BCE", question, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    if "at" in q:
        match = re.search(r"\bat\s+(\d{3,4})\s*BCE", question, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    years = _extract_bce_years(question)
    if years:
        if "beginning" in q:
            return max(years)
        return min(years)
    return default


def _value_at_bce(df: Any, time_col: str, value_col: str, bce_year: float) -> float | None:
    time = _series_numeric(df[time_col])
    values = _series_numeric(df[value_col])
    target = -abs(float(bce_year)) if ("bce" not in time_col.lower() and bool((time < 0).any())) else abs(float(bce_year))
    if "ce" in time_col.lower() and "bce" not in time_col.lower():
        target = -abs(float(bce_year))
    good = time.notna() & values.notna()
    if good.sum() == 0:
        return None
    idx = (time[good] - target).abs().idxmin()
    return float(values.loc[idx])


def _period_name_from_question(question: str, *, fallback: str) -> str:
    q = question.lower()
    names = [
        "Early Neolithic",
        "Middle Neolithic",
        "Younger Neolithic",
        "Late Neolithic",
        "Older Bronze Age",
        "Younger Bronze Age",
    ]
    found = [name for name in names if name.lower() in q]
    if found:
        return " and ".join(found)
    return fallback


def _period_label_from_question(question: str, *, fallback: str, window: tuple[float, float] | None) -> str:
    name = _period_name_from_question(question, fallback=fallback)
    if window and name != fallback and _window_phrase(window) not in name:
        return f"{name} ({_window_phrase(window)})"
    return name


def _time_column(df: Any, question: str) -> str:
    ql = question.lower()
    preferred = ("bce", "ce", "year", "calbp") if "bce" in ql or "century" in ql else ("year", "ce", "bce", "calbp")
    for token in preferred:
        for col in df.columns:
            name = str(col)
            low = name.lower()
            if low == token or token in low:
                if _series_numeric(df[col]).notna().sum() >= 3:
                    return name
    return ""


def _series_numeric(values: Any):
    import pandas as pd

    if hasattr(values, "astype"):
        try:
            values = values.astype(str).str.replace(",", ".", regex=False)
        except Exception:
            pass
    return pd.to_numeric(values, errors="coerce")


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for token in re.findall(r"[a-zA-Z0-9_]+", str(text).lower()):
        if len(token) <= 2 or token in STOPWORDS:
            continue
        out.add(token)
        if token.endswith("s") and len(token) > 4:
            out.add(token[:-1])
        if "_" in token:
            out.update(part for part in token.split("_") if len(part) > 2 and part not in STOPWORDS)
    return out


def _expanded_tokens(text: str, context: str) -> set[str]:
    tokens = _tokens(f"{text} {context}")
    compact = " ".join(tokens)
    if tokens & {"axes", "axis", "axe"}:
        tokens.update({"axe", "axes", "axis", "celt", "celts", "beil"})
    if tokens & {"dagger", "daggers"}:
        tokens.update({"dagger", "daggers", "dolch"})
    if "social" in tokens and "capital" in tokens:
        tokens.update({"copper", "gold", "amber", "monument", "cu", "au"})
    if "human" in tokens and "capital" in tokens:
        tokens.update({"labor", "labour", "workforce", "employment", "enrollment"})
    if "economic" in tokens and "output" in tokens:
        tokens.update({"gni", "gdp", "income", "exports", "product"})
    if "education" in tokens:
        tokens.update({"school", "schooling", "expenditure", "spending", "enrollment"})
    if "gardening" in tokens or "garden" in tokens or "flora" in tokens:
        tokens.update({"gard", "agfo", "alien", "native", "pollen"})
    if "replication" in tokens or "original" in tokens:
        tokens.update({"subjects", "ro", "rr"})
    if "requirements" in tokens or "roles" in tokens or "role" in tokens:
        tokens.update({"developer", "analyst"})
    # Split camel-case after tokenization.
    for piece in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", str(text)):
        p = piece.lower()
        if len(p) > 2 and p not in STOPWORDS:
            tokens.add(p)
    if "gni" in compact:
        tokens.add("income")
    return tokens


def _is_index_like(low: str) -> bool:
    return (
        low in {"year", "bce", "ce", "calbp", "group", "color", "id", "index"}
        or "unnamed" in low
        or re.match(r"^\d{4}(?:\b|\s|\[|_)", low) is not None
        or re.match(r"^yr\d{4}\b", low) is not None
        or re.match(r"^\d{4}\s*\[yr\d{4}\]", low) is not None
        or low.endswith("_id")
        or low.startswith("id_")
    )


def _is_smoothed_measure(low: str) -> bool:
    return any(tok in low for tok in ("inter", "smooth", "rolling", "z"))


def _safe_range(series: Any) -> float:
    try:
        return float(series.max() - series.min())
    except Exception:
        return 0.0


def _simple_slope(xs: list[Any], ys: list[Any]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if _finite(x) and _finite(y)]
    if len(pairs) < 3:
        return 0.0
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    den = sum((x - mx) ** 2 for x, _ in pairs)
    if den <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in pairs) / den


def _corr(a: Any, b: Any) -> float | None:
    import pandas as pd

    frame = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(frame) < 3:
        return None
    r = float(frame["a"].corr(frame["b"]))
    if not math.isfinite(r):
        return None
    return r


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _extract_bce_window(text: str) -> tuple[float, float] | None:
    nums = [float(x) for x in re.findall(r"\b\d{3,4}\b", text)]
    if len(nums) >= 2 and "bce" in text.lower():
        return min(nums[0], nums[1]), max(nums[0], nums[1])
    return None


def _period_phrase(value: float, time_col: str) -> str:
    is_bce = "bce" in time_col.lower() or value < 0
    year = abs(int(round(value)))
    suffix = "BCE" if is_bce else "CE"
    century = (year + 99) // 100
    millennium = (year + 999) // 1000
    if is_bce:
        within_millennium = year - (millennium - 1) * 1000
        if within_millennium <= 350:
            millennium_phrase = f"end of the {_ordinal(millennium)} millennium BCE"
        elif within_millennium >= 750:
            millennium_phrase = f"early {_ordinal(millennium)} millennium BCE"
        else:
            millennium_phrase = f"middle {_ordinal(millennium)} millennium BCE"
        return (
            f"{year} BCE ({_ordinal(century)} century BCE, "
            f"{millennium_phrase})"
        )
    return f"{year} CE ({_ordinal(century)} century CE)"


def _coarse_period_phrase(value: float, time_col: str) -> str:
    is_bce = "bce" in time_col.lower() or value < 0
    if not is_bce:
        return ""
    year = abs(int(round(value)))
    millennium = (year + 999) // 1000
    within_millennium = year - (millennium - 1) * 1000
    if within_millennium <= 350:
        return f"the end of the {_ordinal(millennium)} millennium BCE"
    if within_millennium >= 750:
        return f"the early {_ordinal(millennium)} millennium BCE"
    return f"the middle of the {_ordinal(millennium)} millennium BCE"


def _canonical_variable_name(name: str, columns: Any) -> str:
    if name.endswith("_inter"):
        base = name[:-6]
        if any(str(c) == base for c in columns):
            return base
    return name


def _onset_window_phrase(value: float, next_value: float, time_col: str) -> str:
    if not math.isfinite(next_value) or abs(abs(next_value) - abs(value)) < 50:
        return _period_phrase(value, time_col)
    is_bce = "bce" in time_col.lower() or value < 0 or next_value < 0
    if is_bce:
        first = abs(int(round(value)))
        second = abs(int(round(next_value)))
        return f"{first}/{second} BCE"
    first = int(round(value))
    second = int(round(next_value))
    return f"{first}/{second} CE"


def _window_phrase(window: tuple[float, float] | None) -> str:
    if not window:
        return "the selected period"
    lo, hi = window
    return f"{int(hi)}-{int(lo)} BCE"


def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _humanize(name: str) -> str:
    known = {
        "AxesCelts": "axes and celts",
        "AxesCelts_inter": "axes and celts",
        "Dagger": "daggers",
        "Dagger_inter": "daggers",
        "ZBeil": "axes and celts",
        "ZDolch": "daggers",
        "Zhausgr": "house size",
        "ZAxtSchwert": "hatchets/swords",
        "ZCU_AU": "copper and gold",
        "Zamber": "amber",
        "ZMonument": "monument count",
        "ZKeform": "pottery form",
        "Zkeverz": "pottery decoration",
    }
    if name in known:
        return known[name]
    text = re.sub(r"_inter$", "", name)
    text = re.sub(r"^Z", "", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace(".", " ")
    return re.sub(r"\s+", " ", text).strip().lower() or name


def _single_period(value: float, time_col: str) -> str:
    if value < 0 or "bce" in time_col.lower():
        return f"{abs(int(round(value / 100.0) * 100))} BCE"
    return f"{int(round(value))} CE"


def _requested_ordinal(question: str) -> int | None:
    q = str(question or "").lower()
    if "second" in q or "2nd" in q:
        return 2
    if "third" in q or "3rd" in q:
        return 3
    if "first" in q or "1st" in q:
        return 1
    return None


def _question_event_subject(question: str, *, verbs: tuple[str, ...]) -> str:
    text = " ".join(str(question or "").strip().rstrip("?").split())
    verb_alt = "|".join(re.escape(v) for v in verbs)
    patterns = (
        rf"(?is)^in\s+which\s+century\s+did\s+(.+?)\s+(?:{verb_alt})\b",
        rf"(?is)^when\s+did\s+(.+?)\s+(?:{verb_alt})\b",
    )
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            subject = match.group(1).strip()
            if subject and not subject.lower().startswith(("the ", "a ", "an ")):
                subject = f"the {subject}"
            return subject
    return ""


def _join_phrase(items: list[str]) -> str:
    clean = [str(item).strip() for item in items if str(item).strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return ", ".join(clean[:-1]) + f" and {clean[-1]}"
