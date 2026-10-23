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
    # "have not been formally documented": an adverb between them.
    re.compile(r"\b(?:not|never)\s+(?:yet\s+)?(?:been\s+)?(?:(?:formally|properly|fully|really|officially|actually|"
               r"clearly|ever|yet)\s+)?(?:documented|tested|defined|agreed)\b", re.IGNORECASE),
    # "The failover procedure and the recovery runbook are still missing." Only
    # a document, procedure or control, so "the alert fires when a heartbeat
    # is missing" is not a gap.
    re.compile(r"\b(?:runbooks?|procedures?|documentation|docs|plans?|process(?:es)?|diagrams?|owners?|"
               r"playbooks?|polic(?:y|ies)|tests?|alerting|monitoring|dashboards?|backups?|rto|rpo|slas?)\b"
               r"[^.!?]{0,80}?\b(?:is|are)\s+(?:still\s+|currently\s+|also\s+)?missing\b", re.IGNORECASE),
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


def states_something(text: str) -> bool:
    """True when some sentence of `text` states something on its own: not a
    question, not a gap and not hedged ("The backup retention is seven days.
    I think we changed it to thirty days, but I'm not sure."), or the gap is
    only in a contrasting clause (states_something_before_gap)."""
    for part in re.split(r"(?<=[.!])\s+", text or ""):
        # Judged with its question: "Is there a staging environment? Not
        # really, ..." is a gap only read whole.
        answer = part.split("?")[-1]
        if len(answer.split()) >= 4 and not is_gap_statement(part) and not _HEDGE_RE.search(answer):
            return True
    return states_something_before_gap(text)


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
_ANNOUNCEMENT_LEAD = (
    r"^\W*(?:(?:so|okay|ok|alright|right|well|now|and|um|uh)\W+)*"
    r"(?:(?:today|now|in\s+this\s+(?:session|call|kt)|this\s+(?:morning|afternoon))\s*,?\s+)?"
    r"(?:(?:i|we)(?:'m|'re|\s+am|\s+are|'ll|\s+will|'d\s+like\s+to|\s+would\s+like\s+to|\s+want\s+to|"
    r"(?:'m|'re|\s+am|\s+are)\s+going\s+to)?\s+(?:be\s+)?"
    r"(?:hand(?:ing)?\s+over|walk(?:ing)?\s+(?:you\s+)?through|tak(?:e|ing)\s+you\s+through|present(?:ing)?|"
    r"cover(?:ing)?|talk(?:ing)?\s+(?:you\s+)?(?:about|through))"
    r"|let\s+me\s+(?:hand\s+over|walk\s+you\s+through|take\s+you\s+through|talk\s+about)"
    r"|(?:this|today'?s|the)\s+(?:kt|session|handover|hand-?over|knowledge\s+transfer|call|walkthrough)\s+"
    r"(?:is\s+(?:for|about|on)|covers|will\s+cover)"
    r"|this\s+is\s+(?:the|a|an|our)\s+(?:(?:quick|short|brief|final|first|last|second|small)\s+)?"
    r"(?:kt|handover|hand-?over|knowledge\s+transfer|walkthrough)\s+(?:for|of|on|about)"
    # The polish pass's own framing: "This document serves as the handover for Paygate, ..."
    r"|this\s+document\s+(?:serves\s+as|is)\s+(?:the|a|an)\s+(?:kt|handover|hand-?over|knowledge\s+transfer)"
    r"(?:\s+document)?\s+(?:for|of|on))\s+"
)
# "This is Streamline, our GCP data platform for clickstream analytics."
# introduces the system by name: "Streamline is our GCP data platform ...".
_THIS_IS_NAME_RE = re.compile(
    r"^(?i:this\s+is)\s+(?P<name>[A-Z][\w.&+-]*(?:\s+[A-Z][\w.&+-]*){0,3})\s*,\s*(?P<desc>(?i:our|the|a|an)\s+.+)$",
    re.DOTALL)
_ANNOUNCEMENT_RE = re.compile(
    _ANNOUNCEMENT_LEAD + r"(?P<what>[^,;:()—]+?)(?:\s+(?:today|now|here|with\s+you))?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_ANNOUNCEMENT_HEAD_RE = re.compile(_ANNOUNCEMENT_LEAD, re.IGNORECASE)
# What an announcement goes on to say about the system: "..., which processes
# 40,000 orders a day" or "..., the booking backend behind our app".
_RELATIVE_TAIL_RE = re.compile(
    r"(?P<np>[^,;:()—]+?)(?:\s+(?:today|now|here|with\s+you))?\s*,\s*(?:which|that|and\s+it)\s+(?P<rest>.+)$"
    r"|(?P<np2>[^,;:()—]+?\b(?:platform|service|system|app|application|backend|api|pipeline|portal|engine|tool|"
    r"product|site|website|database|cluster))\s+(?:which|that)\s+(?P<rest2>.+)$",
    re.IGNORECASE | re.DOTALL)
_APPOSITIVE_TAIL_RE = re.compile(
    r"(?P<np>[^,;:()—]+?)(?:\s+(?:today|now|here|with\s+you))?\s*,\s*(?P<desc>(?:the|a|an|our)\s+.+)$",
    re.IGNORECASE | re.DOTALL)
# Spoken lead-ins at the start of a sentence; the transcript cleaner removes
# most, but a value can also come from a pasted transcript or a merge.
_LEADING_FILLER_RE = re.compile(
    r"^\s*(?:(?:so|okay|ok|alright|right(?!\s+(?:now|away|after|before|there|here|then))|well|um+|uh+|erm|er|"
    r"basically|anyway|you\s+know)(?![\w'-])\s*[,.!]*\s*)+",
    re.IGNORECASE)
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
        from devops_transcription import capitalise_sentence_start
        out = capitalise_sentence_start(rest)
    return out


def without_framing(sentence: str) -> str:
    """The sentence as a statement about the system: no greeting, no spoken
    lead-in, and no "I'm handing over" when the sentence goes on to say
    something ("Hi team, today I'm handing over our AWS e-commerce platform,
    which processes around 40,000 orders daily." -> "Our AWS e-commerce
    platform processes around 40,000 orders daily."). A sentence that only
    announces the handover is returned unchanged; is_handover_announcement
    tells the caller to drop it. Nothing is added except "is" joining a name
    to its description."""
    text = strip_greeting(sentence or "")
    stripped = _LEADING_FILLER_RE.sub("", text)
    if stripped.strip():
        text = stripped
    head = _ANNOUNCEMENT_HEAD_RE.match(text)
    if head:
        tail = text[head.end():]
        relative = _RELATIVE_TAIL_RE.match(tail)
        appositive = None if relative else _APPOSITIVE_TAIL_RE.match(tail)
        if relative:
            np, rest = relative.group("np") or relative.group("np2"), relative.group("rest") or relative.group("rest2")
            text = f"{np.strip()} {rest.strip()}"
        elif appositive:
            text = f"{appositive.group('np').strip()} is {appositive.group('desc').strip()}"
    else:
        named = _THIS_IS_NAME_RE.match(text.strip())
        if named:
            text = f"{named.group('name')} is {named.group('desc').strip()}"
    text = text.strip()
    if text == (sentence or "").strip():
        # Nothing removed: the value stays exactly as it was ("thirty minutes
        # (replaces four hours ...)" was printed as "Thirty minutes ...").
        return text
    from devops_transcription import capitalise_sentence_start
    return capitalise_sentence_start(text)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def without_session_framing(text: str) -> str:
    """A paragraph or field value without the KT session's own framing: each
    sentence through without_framing, and sentences that only announce the
    handover (the document's title already names the system) left out."""
    kept = [without_framing(s) for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]
    return " ".join(s for s in kept if not is_handover_announcement(s)).strip()


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
    return not (is_question(text) or (is_gap_statement(text) and not states_something(text)))


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


# A pasted transcript carries its speaker on every turn ("Owen: Merge to main
# ..."), and a question-and-answer KT merges each question with its answer
# (merge_question_answers). The document printed both as facts: "Nadia: How
# do I deploy?" was a deployment step (KT-taker audit, dev d05; matrix T01).
# A speaker label is a name of one to three capitalised words that opens at
# least two turns; a heading said once ("Backups: nightly to S3") is not one.
_LABEL_RE = re.compile(r"(?:^|(?<=[.!?]\s))([A-Z][\w'.-]*(?:\s[A-Z][\w'.-]*){0,2}):\s")
_HEADING_WORDS = frozenset("""note notes step steps warning warnings example examples update updates important tip
tips error errors status summary action actions todo fix fixes escalation backups backup security cost costs
monitoring deployment rollback risk risks question answer topic agenda next unfinished overview owner owners
""".split())


def speaker_labels(text: str) -> set:
    """The speaker labels of a transcript: names that open two or more turns."""
    from collections import Counter

    counts = Counter(_LABEL_RE.findall(text or ""))
    names = {label: n for label, n in counts.items() if label.split()[0].lower() not in _HEADING_WORDS}
    # In a conversation (someone opens two or more turns), a name that opens
    # one turn is a speaker too ("Ana Ruiz: What happens at month end?").
    if not any(n >= 2 for n in names.values()):
        return set()
    return set(names)


def without_speaker_labels(text: str, labels) -> str:
    """`text` without the speaker labels that open its turns."""
    if not labels or not text:
        return text
    names = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    return re.sub(r"(?:^|(?<=[.!?]\s))(?:" + names + r"):\s+", "", text)


# The answer stands alone when it is a statement or an instruction, does not
# lean on the question ("We never agreed one", "It's hybrid", "Yes, ..."),
# and names what the question asked about ("How do I deploy?" / "... deploys
# to a staging slot"). Otherwise the question stays: "Costs? The App Service
# plan is oversized." says what the plan's size is about.
_LEANS_ON_QUESTION_RE = re.compile(
    r"\b(?:it|its|it's|that|this|these|those|they|them|their|one|ones|there|here|same|yes|no|yeah|nope)\b"
    r"|\bnot\s+really\b", re.IGNORECASE)
_QUESTION_LIGHT = frozenset("""what which who whom how where when why should would could can do does did is are was
were am i we you my our your the a an to of for in on at with me us about anything else other more and so okay ok
also any there have has had know will if something wrong exactly that this it be been tell me thing things happen
happens happening break breaks most often usually typically normally""".split())


# An answer that opens like a fragment needs its question: "Around eight
# thousand a day" (of what?), "Every night", "Dev, staging and production".
_FRAGMENT_ANSWER_RE = re.compile(
    r"^\W*(?:around|about|roughly|approximately|nearly|almost|over|under|every|each|in|on|at|for|from|by|with"
    r"|through|via|since|until|only|just|maybe|probably|mostly|usually|dev|staging|prod|production"
    r"|\d|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|hundred|thousand)\b",
    re.IGNORECASE)


def _question_stems(text: str) -> set:
    return {w.rstrip("s")[:5] for w in re.findall(r"[a-z][a-z'-]+", (text or "").lower())
            if w not in _QUESTION_LIGHT and len(w) > 2}


def without_redundant_question(text: str) -> str:
    """"How do I deploy? Merge to main and GitHub Actions deploys to a staging
    slot." -> the answer alone, when it says everything the question asked."""
    m = re.match(r"^\s*(?P<q>[^?]*\?)\s+(?P<a>[^?]+)$", text or "")
    if not m:
        return text
    question, answer = m.group("q"), m.group("a").strip()
    if len(answer.split()) < 4 or _LEANS_ON_QUESTION_RE.search(answer) or _FRAGMENT_ANSWER_RE.match(answer):
        return text
    if _question_stems(question) - _question_stems(answer):
        return text
    return answer


def display_text(text: str, labels) -> str:
    """A fact as the document shows it: no speaker labels, and no question its
    answer makes redundant."""
    return without_redundant_question(without_speaker_labels(text, labels))

