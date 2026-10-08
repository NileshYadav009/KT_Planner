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
    # A topic put off to a later session is not covered: "We will cover
    # deployment and the rest next week."
    re.compile(r"\b(?:we(?:'ll|\s+will)|let'?s|i(?:'ll|\s+will))\s+(?:cover|go\s+(?:over|through)|discuss|"
               r"talk\s+about|walk\s+through|do)\b[^.]*\b(?:next\s+(?:week|time|session|call)|later|"
               r"another\s+(?:session|time|call)|tomorrow|separately)\b", re.IGNORECASE),
]
# An answer that opens with a plain "no" to the question before it.
_NEGATIVE_ANSWER_RE = re.compile(r"\?\s*(?:no|nope|not\s+really|not\s+yet|none|nothing)\b", re.IGNORECASE)


def is_question(text: str) -> bool:
    """A bare question. A question merged with its answer ("Is there a test
    environment? Yes, one UAT server with last month's data.") is not one:
    read as a question, Environments was reported Missing."""
    text = (text or "").strip()
    if text.endswith("?"):
        return True
    if not (_QUESTION_START_RE.match(text) and len(text.split()) <= 14 and "?" in text):
        return False
    answer = text.rsplit("?", 1)[1]
    return len(re.findall(r"[A-Za-z0-9]+", answer)) < 3


def is_gap_statement(text: str) -> bool:
    """True when the sentence (or question + answer unit) says the thing is
    not in place, not decided or not known, which is a gap, not coverage."""
    text = text or ""
    return bool(_NEGATIVE_ANSWER_RE.search(text) or any(rx.search(text) for rx in _GAP_RES))


# "Cosmos has continuous backup turned on, but we have never tried a restore"
# states a control that is in place, then a gap in it. Read as a gap alone,
# Disaster Recovery was reported Missing although the backup was stated.
_CONTRAST_RE = re.compile(r"(?:[,;]\s*|\s+)(?:but|however|though|although)\b", re.IGNORECASE)
# "Is it backed up? I think so, but I have never checked" states nothing.
_HEDGE_RE = re.compile(r"\b(?:i\s+think|i\s+believe|i\s+guess|probably|maybe|perhaps|supposedly|should\s+be)\b",
                       re.IGNORECASE)


def states_something_before_gap(text: str) -> bool:
    """True when the gap is only in a contrasting clause ("..., but we have
    never tested it") and the clause before it states something on its own:
    for a question and its answer, the answer's own words, unhedged."""
    text = text or ""
    match = _CONTRAST_RE.search(text)
    if not match:
        return False
    head = text[:match.start()].split("?")[-1]
    return len(head.split()) >= 4 and not is_gap_statement(head) and not _HEDGE_RE.search(head)


# Session framing, not knowledge: a greeting in front of a sentence, and a
# sentence that only announces what is being handed over. The document's
# title already names the system, yet "Hi everyone, today I will be handing
# over the AWS e-commerce platform." was printed as System Overview content.
_GREETING_PREFIX_RE = re.compile(
    r"^\s*(?:(?:hi|hello|hey|hiya|good\s+(?:morning|afternoon|evening))"
    r"(?:\s+(?:everyone|everybody|all|team|folks|guys|there|all\s+of\s+you))?"
    r"|(?:thanks|thank\s+you)(?:\s+(?:everyone|all|team|folks))?\s+for\s+(?:joining|coming|being\s+here|"
    r"jumping\s+on|making\s+(?:the\s+)?time)(?:\s+today)?"
    r"|welcome(?:\s+(?:everyone|all|team|back))?)\s*[,.!:;—-]+\s*",
    re.IGNORECASE,
)
# "Today I'm handing over X", "This KT is for X", "I'll walk you through X".
_ANNOUNCEMENT_RE = re.compile(
    r"^\W*(?:(?:so|okay|ok|alright|right|well|now|and|um|uh)\W+)*"
    r"(?:(?:today|now|in\s+this\s+(?:session|call|kt)|this\s+(?:morning|afternoon))\s*,?\s+)?"
    r"(?:(?:i|we)(?:'m|'re|\s+am|\s+are|'ll|\s+will|'d\s+like\s+to|\s+would\s+like\s+to|\s+want\s+to|"
    r"(?:'m|'re|\s+am|\s+are)\s+going\s+to)?\s+(?:be\s+)?"
    r"(?:hand(?:ing)?\s+over|walk(?:ing)?\s+(?:you\s+)?through|tak(?:e|ing)\s+you\s+through|present(?:ing)?|"
    r"cover(?:ing)?|talk(?:ing)?\s+(?:you\s+)?(?:about|through))"
    r"|let\s+me\s+(?:hand\s+over|walk\s+you\s+through|take\s+you\s+through|talk\s+about)"
    r"|(?:this|today'?s|the)\s+(?:kt|session|handover|hand-?over|knowledge\s+transfer|call|walkthrough)\s+"
    r"(?:is\s+(?:for|about|on)|covers|will\s+cover)"
    r"|this\s+is\s+the\s+(?:kt|handover|hand-?over|knowledge\s+transfer|walkthrough)\s+(?:for|of|on))\s+"
    r"(?P<what>[^,;:()—]+?)(?:\s+(?:today|now|here|with\s+you))?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_WELCOME_RE = re.compile(
    r"^\W*welcome\s+to\s+(?:the\s+|our\s+)?(?P<what>[^,;:()—]+?)\s+"
    r"(?:kt|handover|hand-?over|knowledge\s+transfer|session|walkthrough)(?:\s+session)?\s*[.!]?\s*$",
    re.IGNORECASE,
)
# What follows the announcement is only a name when it is short and says
# nothing more: "TripWise, the booking backend behind our app" (a comma) and
# "the payments platform that runs on EKS" (a clause) describe the system.
_CLAUSE_RE = re.compile(r"\b(?:which|that|who|whose|where|when|because|so|but|and\s+(?:it|we|they|this))\b",
                        re.IGNORECASE)


def strip_greeting(text: str) -> str:
    """The sentence without a greeting in front ("Hi everyone, today ..." ->
    "Today ..."). A sentence that is only a greeting is returned unchanged:
    the classifier already drops those."""
    out = text or ""
    for _ in range(3):              # "Hey, thanks for jumping on, ..."
        match = _GREETING_PREFIX_RE.match(out)
        rest = out[match.end():] if match else ""
        if not match or not rest.strip():
            break
        out = rest[:1].upper() + rest[1:]
    return out


def is_handover_announcement(text: str) -> bool:
    """True when the sentence only announces what is being handed over (its
    name, which the document's title already gives) and states nothing else."""
    sentence = strip_greeting(text or "").strip()
    match = _ANNOUNCEMENT_RE.match(sentence) or _WELCOME_RE.match(sentence)
    if not match:
        return False
    what = match.group("what")
    return 0 < len(re.findall(r"[\w./&+-]+", what)) <= 8 and not _CLAUSE_RE.search(what)


def is_field_candidate(text: str) -> bool:
    """A sentence that may fill a document field: not a bare question and
    not a statement that the thing is missing."""
    return not (is_question(text) or (is_gap_statement(text) and not states_something_before_gap(text)))


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
