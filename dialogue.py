"""Questions, answers and stated gaps in a KT conversation (P0-11).

Real KT sessions are dialogues: the receiver asks, the giver answers, and
often the honest answer is "we don't have that". Before this module the
receiver's questions were mapped as facts ("Staging -> Is there a staging
environment?" in the environments table, "What is the RTO?" listed as a
high-risk period), answers were classified without their question ("We
never agreed one with the business." landed in Ownership), and "there is no
DR plan" counted as disaster-recovery coverage.
"""
import re
from typing import Callable, List, TypeVar

T = TypeVar("T")

_QUESTION_START_RE = re.compile(
    r"^(?:what|who|whom|whose|which|when|where|why|how|is|are|was|were|do|does|did|can|could|"
    r"should|would|will|have|has|had|any)\b",
    re.IGNORECASE,
)

# Statements that something is absent, undecided or unknown. Deliberately
# subject + verb based, so imperatives such as "Never delete the table" or
# "We never deploy on Fridays" (real operational rules) do not match, and
# "has no owner yet" (an open responsibility, captured as such) does not
# either.
_GAP_RES = [
    re.compile(r"\bthere(?:\s+is|\s+are|'s)\s+(?:no|not\s+(?:a|an|any))\s+(?:[a-z]+\s+){0,2}?"
               r"(?:plan|process|runbook|documentation|docs?|backups?|monitoring|alert(?:ing|s)?|tests?|testing|"
               r"dr|disaster\s+recovery|rto|rpo|sla|staging|environment|policy|procedure|on-?call|"
               r"rotation|dashboards?|automation|rollback|failover|owner|escalation|contact|diagram)\b", re.IGNORECASE),
    re.compile(r"\b(?:we|i|they|nobody|no\s*one|the\s+team)\s+(?:have\s+|has\s+|had\s+)?(?:never|not)\s+"
               r"(?:yet\s+)?(?:agreed|defined|documented|tested|checked|set\s+up|written|configured|decided|"
               r"measured|reviewed|practi[cs]ed|tried|done)\b", re.IGNORECASE),
    re.compile(r"\bnobody\s+(?:has\s+)?(?:tested|checked|knows|documented|reviewed)\b", re.IGNORECASE),
    re.compile(r"\b(?:i'?m|i\s+am|we'?re|we\s+are)\s+not\s+(?:really\s+)?sure\b", re.IGNORECASE),
    re.compile(r"\b(?:i|we)\s+(?:don'?t|do\s+not)\s+(?:really\s+)?know\b", re.IGNORECASE),
    re.compile(r"\b(?:i|we)\s+have\s+never\s+(?:checked|tested|seen|verified)\b", re.IGNORECASE),
    re.compile(r"\b(?:not|never)\s+(?:been\s+)?(?:documented|tested|defined|agreed)\b", re.IGNORECASE),
]
# An answer that opens with a plain "no" to the question before it.
_NEGATIVE_ANSWER_RE = re.compile(r"\?\s*(?:no|nope|not\s+really|not\s+yet|none|nothing)\b", re.IGNORECASE)


def is_question(text: str) -> bool:
    text = (text or "").strip()
    return text.endswith("?") or bool(_QUESTION_START_RE.match(text) and len(text.split()) <= 14 and "?" in text)


def is_gap_statement(text: str) -> bool:
    """True when the sentence (or question + answer unit) says the thing is
    not in place, not decided or not known, which is a gap, not coverage."""
    text = text or ""
    return bool(_NEGATIVE_ANSWER_RE.search(text) or any(rx.search(text) for rx in _GAP_RES))


def is_field_candidate(text: str) -> bool:
    """A sentence that may fill a document field: not a bare question and
    not a statement that the thing is missing."""
    return not (is_question(text) or is_gap_statement(text))


def merge_question_answers(sentences: List[T], text_of: Callable[[T], str],
                           merge: Callable[[T, T], T]) -> List[T]:
    """Join each question with the sentence that answers it, so the pair is
    classified and rendered as one statement. A question followed by
    another question, or by nothing, stays on its own (an unanswered
    question)."""
    out: List[T] = []
    i = 0
    while i < len(sentences):
        current = sentences[i]
        if (i + 1 < len(sentences) and text_of(current).strip().endswith("?")
                and not text_of(sentences[i + 1]).strip().endswith("?")):
            out.append(merge(current, sentences[i + 1]))
            i += 2
            continue
        out.append(current)
        i += 1
    return out
