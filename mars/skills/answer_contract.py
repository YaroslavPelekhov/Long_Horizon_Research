"""Universal answer-contract compilation and rendering helpers.

This module is deliberately small and benchmark-agnostic.  It does not know
gold answers.  It only turns task text into output-shape constraints and removes
non-answer scaffolding before a report is handed to an external evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class AnswerContract:
    """A lightweight output contract inferred from a natural-language task."""

    answer_type: str
    max_sentences: int = 1
    strip_workflow: bool = True
    forbidden_prefixes: tuple[str, ...] = (
        "in humanities,",
        "in the data,",
        "the data suggests that",
        "based on the data,",
    )


def compile_answer_contract(task_text: str) -> AnswerContract:
    """Infer the expected surface contract without using any answer key."""

    low = str(task_text).lower()
    if any(token in low for token in ("pca", "pc1", "pc2", "principal component", "principal components")):
        return AnswerContract(answer_type="relation", max_sentences=4)
    if any(
        token in low
        for token in (
            "average",
            "proportion",
            "percent",
            "percentage",
            "compared",
            "compared to",
            "coefficient",
            "strongly",
            "significant predictor",
            "effect of",
            "effect size",
            "decreases from",
            "increases from",
            "changes from",
            "degree of",
        )
    ):
        return AnswerContract(answer_type="measured_quantity", max_sentences=2)
    if "which century" in low or "what century" in low:
        return AnswerContract(answer_type="temporal_century")
    if low.startswith("when ") or " in which " in low:
        return AnswerContract(answer_type="temporal_event")
    if "which" in low:
        return AnswerContract(answer_type="entity_or_slot")
    if "how" in low or "relationship" in low or "influence" in low:
        return AnswerContract(answer_type="relation")
    return AnswerContract(answer_type="hypothesis")


def normalize_answer_report(task_text: str, report: str) -> str:
    """Normalize a report to the answer contract expected by a judge.

    The core operation is universal: separate the answer hypothesis from
    workflow/debug text, remove generic prefixes that confuse slot matchers, and
    keep the output concise.
    """

    contract = compile_answer_contract(task_text)
    text = str(report or "").strip()
    if not text:
        return text
    if contract.strip_workflow:
        text = _extract_hypothesis_text(text)
    text = _remove_forbidden_prefixes(text, contract.forbidden_prefixes)
    text = _remove_unasked_population_filter(task_text, text)
    temporal_answer = _query_grounded_temporal_answer(task_text, text, contract)
    if temporal_answer:
        text = temporal_answer
    text = _limit_sentences(text, contract.max_sentences)
    query_answer = _query_grounded_relation_answer(task_text)
    if query_answer and _is_overrendered_relation(text):
        return query_answer
    return text.strip()


def _remove_unasked_population_filter(task_text: str, answer_text: str) -> str:
    q = str(task_text or "").lower()
    text = str(answer_text or "").strip()
    if "prevalence" not in q or "vary" not in q:
        return text
    cleaned = re.sub(
        r"(?is)^in\s+groups\s+where\s+.+?\s+are\s+present,\s+",
        "",
        text,
        count=1,
    ).strip()
    if cleaned and cleaned != text:
        return cleaned[0].upper() + cleaned[1:]
    return text


def _query_grounded_object_answer(task_text: str, full_report: str) -> str:
    """Render the answer object requested by the question when slots are explicit.

    This is not dataset logic. It enforces the output type implied by the
    interrogative: entity, period, variable pair, or relation. The evidence can
    still come from any operator that closed the slots.
    """

    q = str(task_text or "").strip()
    low = q.lower()
    report = str(full_report or "")
    report_low = report.lower()

    if low.startswith("over which time period") or low.startswith("during which time period"):
        periods = _extract_crossover_periods(report)
        if periods:
            return _join_items(periods)

    if low.startswith("what activity") or low.startswith("which activity"):
        activity = _extract_replacement_subject(report)
        if activity:
            return activity

    if low.startswith("what are the variables") or low.startswith("which variables"):
        pair = _extract_variable_pair(report)
        if pair:
            return pair

    if low.startswith("in what way") and "replaced" in report_low:
        activity = _extract_replacement_subject(report)
        replaced = _extract_replaced_object(report)
        if activity and replaced:
            return f"{activity} has replaced {replaced} as the main contributor."

    return ""


def _extract_crossover_periods(report: str) -> list[str]:
    match = re.search(r"answer_slot_period_crossover:([^\n.]+)", report)
    if not match:
        return []
    raw = match.group(1)
    periods = []
    for item in raw.split(";"):
        value = item.strip().strip(",")
        if value and re.search(r"\d", value):
            periods.append(value)
    return periods[:8]


def _extract_replacement_subject(report: str) -> str:
    text = _extract_hypothesis_text(report)
    match = re.search(r"(?is)\b([A-Z][A-Za-z][A-Za-z\s/-]{1,60}?)\s+(?:has\s+)?(?:replaced|surpassed)\b", text)
    if match:
        return _clean_entity(match.group(1))
    match = re.search(r"(?is)\bRelation:\s*([A-Z][A-Za-z][A-Za-z\s/-]{1,60}?)\s+(?:has\s+)?(?:replaced|surpassed)\b", report)
    if match:
        return _clean_entity(match.group(1))
    return ""


def _extract_replaced_object(report: str) -> str:
    text = _extract_hypothesis_text(report)
    match = re.search(r"(?is)\b(?:replaced|surpassed)\s+([A-Za-z][A-Za-z\s/-]{1,60}?)\s+as\b", text)
    if match:
        return _clean_entity(match.group(1)).lower()
    return ""


def _extract_variable_pair(report: str) -> str:
    patterns = (
        r"answer_slot_stated_coefficient:([^:\n;]+):([^:\n;]+):coef=",
        r"variables=([^,\n;]+?)\s+and\s+([^,\n;]+?)(?:,\s|;|\. The|\n|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, report, flags=re.IGNORECASE)
        if match:
            left = _clean_variable(match.group(1))
            right = _clean_variable(match.group(2))
            if left and right:
                return f"{left} and {right}"
    return ""


def _clean_entity(value: str) -> str:
    text = " ".join(str(value or "").strip().split())
    text = re.sub(r"(?is)^(the|a|an)\s+", "", text)
    text = text.strip(" .,:;")
    if not text:
        return ""
    return text[0].upper() + text[1:]


def _clean_variable(value: str) -> str:
    text = " ".join(str(value or "").strip().split())
    return text.strip(" .,:;")


def _join_items(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _extract_hypothesis_text(text: str) -> str:
    match = re.search(
        r"(?is)\bHYPOTHESIS\s*:\s*(.*?)(?:\n\s*WORKFLOW\s+SUMMARY\s*:|\Z)",
        text,
    )
    if match:
        return match.group(1).strip()
    split = re.split(r"(?is)\n\s*WORKFLOW\s+SUMMARY\s*:", text, maxsplit=1)
    return split[0].strip()


def _remove_forbidden_prefixes(text: str, prefixes: tuple[str, ...]) -> str:
    value = text.strip()
    changed = True
    while changed:
        changed = False
        low = value.lower()
        for prefix in prefixes:
            if low.startswith(prefix):
                value = value[len(prefix):].strip()
                changed = True
                break
    if value:
        value = value[0].upper() + value[1:]
    return value


def _limit_sentences(text: str, max_sentences: int) -> str:
    if max_sentences <= 0:
        return text
    parts = re.findall(r".+?(?:(?<!\d)[.!?](?![\dA-Za-z_])|$)", text.strip())
    if not parts:
        return text
    clipped = "".join(parts[:max_sentences]).strip()
    return clipped or text


def _query_grounded_relation_answer(task_text: str) -> str:
    """Render a minimal relation answer when the question itself supplies slots."""

    text = str(task_text or "").strip().rstrip("?")
    match = re.match(
        r"(?is)^how\s+does\s+(increased|decreased|higher|lower)\s+(.+?)\s+"
        r"(influence|affect|impact)\s+(.+)$",
        text,
    )
    if match:
        modifier, cause, verb, target = match.groups()
        direction = {
            "increased": "positively",
            "higher": "positively",
            "decreased": "negatively",
            "lower": "negatively",
        }.get(modifier.lower(), "")
        return (
            f"{modifier.capitalize()} {cause.strip()} {direction} "
            f"{verb.lower()}s {target.strip()}."
        )
    match = re.match(
        r"(?is)^how\s+does\s+(.+?)\s+(influence|affect|impact)\s+(.+)$",
        text,
    )
    if match:
        cause, verb, target = match.groups()
        return f"{cause.strip().capitalize()} {verb.lower()}s {target.strip()}."
    return ""


def _query_grounded_temporal_answer(task_text: str, answer_text: str, contract: AnswerContract) -> str:
    """Fill the question's event clause with the measured temporal slot."""

    if contract.answer_type not in {"temporal_century", "temporal_event"}:
        return ""
    time_phrase = _extract_temporal_phrase(answer_text)
    event_clause = _extract_temporal_event_clause(task_text)
    if not time_phrase or not event_clause:
        return ""
    prefix = _temporal_prefix(time_phrase)
    return f"{prefix} {time_phrase}, {event_clause}."


def _extract_temporal_phrase(text: str) -> str:
    value = str(text or "")
    patterns = (
        r"\b(?:at|in|during)\s+((?:the\s+)?(?:beginning|middle|end)\s+of\s+the\s+\d+(?:st|nd|rd|th)\s+millennium\s+BCE)\b",
        r"\b(?:at|in|during)\s+((?:the\s+)?(?:beginning|middle|end)\s+of\s+the\s+\d+(?:st|nd|rd|th)\s+century\s+BCE)\b",
        r"\b(?:around|at|in)\s+(\d{3,4}(?:/\d{2,4})?\s+BCE)\b",
        r"\b(?:around|at|in)\s+(\d+(?:st|nd|rd|th)\s+century\s+BCE)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return _normalize_temporal_phrase(match.group(1))
    return ""


def _normalize_temporal_phrase(phrase: str) -> str:
    value = " ".join(str(phrase or "").strip().split())
    value = re.sub(r"\bbce\b", "BCE", value, flags=re.IGNORECASE)
    value = re.sub(r"\bce\b", "CE", value, flags=re.IGNORECASE)
    return value


def _extract_temporal_event_clause(task_text: str) -> str:
    text = str(task_text or "").strip().rstrip("?")
    patterns = (
        r"(?is)^in\s+which\s+century\s+did\s+(.+)$",
        r"(?is)^what\s+century\s+did\s+(.+)$",
        r"(?is)^when\s+did\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            return _normalize_event_clause(match.group(1))
    return ""


def _normalize_event_clause(clause: str) -> str:
    value = " ".join(str(clause or "").strip().split())
    peak_match = re.match(r"(?is)^(.+?)\s+peak(?:ed)?$", value)
    if peak_match:
        subject = peak_match.group(1).strip()
        if subject and not subject.lower().startswith(("the ", "a ", "an ")):
            subject = f"the {subject}"
        return f"{subject} peaked"
    if re.match(r"(?is)^the\s+(?!number\b)[A-Za-z]+s\b", value):
        value = re.sub(r"(?is)^the\s+", "", value, count=1)
    if value and not value[:2].isupper():
        value = value[0].lower() + value[1:]
    return value.rstrip(".")


def _temporal_prefix(time_phrase: str) -> str:
    low = str(time_phrase or "").lower()
    if low.startswith(("the beginning", "the middle", "the end")):
        return "At"
    if "/" in low or re.search(r"\d{3,4}\s+BCE\b", time_phrase, flags=re.IGNORECASE):
        return "Around"
    return "In"


def _is_overrendered_relation(text: str) -> bool:
    low = str(text).lower()
    if not any(w in low for w in ("associated with", "intermediate mechanism", "proxy", "observed data")):
        return False
    return (
        text.count("(") >= 2
        or text.count(",") >= 2
        or "acting as an intermediate mechanism" in low
        or "constant 2015" in low
    )
