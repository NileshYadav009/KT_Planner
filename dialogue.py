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

# Uncertainty the speaker states: a gap, but in what is known, not in what is
# in place (is_unconfirmed).
_UNCONFIRMED_RES = [
    re.compile(r"\b(?:i'?m|i\s+am|we'?re|we\s+are)\s+not\s+(?:really\s+|completely\s+|entirely\s+|totally\s+|"
               r"fully\s+|quite\s+|(?:a\s+hundred|100)\s*(?:percent|%)\s+)?sure\b", re.IGNORECASE),
    # Left open by the speaker: "..., so that needs confirmation", "still to be
    # confirmed". Not a rule that needs an approval ("changes need
    # confirmation from the CAB").
    # "That value needs confirmation", "this needs verification", "the
    # readiness probe timeout also needs verification" (the user's KTs, 24
    # Oct). Not "Every schema change needs verification by the DBA team".
    re.compile(r"\b(?:that|this|it)\s+(?:(?:value|number|figure|limit|setting|target|estimate|date|detail|part|one)\s+)?"
               r"(?:still\s+|also\s+)?needs?\s+(?:to\s+be\s+)?(?:confirm(?:ed|ation)|verif(?:ied|ication)|validat(?:ed|ion))\b|"
               r"\b(?:still|also|yet)\s+needs?\s+(?:to\s+be\s+)?(?:confirm(?:ed|ation)|verif(?:ied|ication))\b|"
               r"\b(?:yet|still)\s+to\s+be\s+(?:confirmed|verified)\b|\btbc\b", re.IGNORECASE),
    re.compile(r"\b(?:i'?m|i\s+am|we'?re|we\s+are)\s+(?:still\s+)?(?:unsure|uncertain|not\s+certain)\b", re.IGNORECASE),
    # "..., but need to verify this", "I need to confirm whether this target
    # is still valid". Not an instruction ("you need to confirm that the pods
    # are healthy"): only "this", "it" or a closing "that", or the speaker's
    # own "I need to ... whether".
    re.compile(r"\bneeds?\s+to\s+(?:verify|confirm|double-check)\s+(?:this|it\b|that\s*(?:[.,;!?]|$))|"
               r"\bi\s+(?:still\s+)?need\s+to\s+(?:verify|confirm|check|find\s+out)\s+(?:whether|if)\b", re.IGNORECASE),
]
# A value given on an old document's word is a claim with doubt on it: "Our
# recovery point objective is four hours, or at least that is what the
# previous document says." It went into the RPO field as settled.
_DOUBTED_SOURCE_RE = re.compile(
    r"\bor\s+at\s+least\s+(?:that'?s|that\s+is)\s+what\s+(?:the|our|my)\s+(?:[a-z]+\s+){0,2}?"
    r"(?:document(?:ation)?|docs?|runbook|wiki|page|notes|spreadsheet|handover)\s+(?:says|said)\b|"
    r"\baccording\s+to\s+(?:the|our)\s+(?:old|previous|outdated|last|earlier)\s+"
    r"(?:document(?:ation)?|docs?|runbook|wiki|notes)\b", re.IGNORECASE)
_UNCONFIRMED_RES.append(_DOUBTED_SOURCE_RE)
# A caveat on a value, said as an instruction about the document: "Do not
# describe the one-hour recovery estimate as a verified guarantee." Printed as
# a Danger Zones "never" rule; it says the estimate is unconfirmed.
_DOCUMENT_CAVEAT_RE = re.compile(
    r"\b(?:do\s+not|don'?t|never)\s+(?:describe|present|report|document|state|treat|label|record|write\s+up)\s+"
    r"(?:the|this|that|these|those|our|any|it)\b[^.!?]{0,80}?\s+as\s+(?:an?\s+)?"
    r"(?:verified|confirmed|guaranteed|guarantee|final|fact|certain|settled)\b", re.IGNORECASE)
_UNCONFIRMED_RES.append(_DOCUMENT_CAVEAT_RE)
# An instruction about the document that says nothing about the system:
# "Please ensure these uncertainties are clearly identified instead of
# assuming values." (the user's KTs, 24 Oct; printed as a deployment step).
_DOCUMENT_INSTRUCTION_RE = re.compile(
    r"^\W*(?:please\s+)?(?:make\s+sure|ensure)\s+(?:that\s+)?(?:these|the|all|any|this|those)\s+(?:[a-z]+\s+)?"
    r"(?:uncertaint(?:y|ies)|gaps?|unknowns?|assumptions?|open\s+(?:points|questions|items)|values?|estimates?)\s+"
    r"(?:are|is)\s+(?:clearly\s+|explicitly\s+)?(?:identified|marked|flagged|called\s+out|noted|highlighted|listed)\b",
    re.IGNORECASE)


def is_document_instruction(text: str) -> bool:
    """An instruction to whoever writes the document, with nothing in it
    about the system ("Please ensure these uncertainties are clearly
    identified instead of assuming values.")."""
    return bool(_DOCUMENT_INSTRUCTION_RE.search(text or ""))

# Statements that something is absent, undecided or unknown. Deliberately
# subject + verb based, so imperatives such as "Never delete the table" or
# "We never deploy on Fridays" (real operational rules) do not match, and
# "has no owner yet" (an open responsibility, captured as such) does not
# either.
# A topic put off to a later session is not covered: "We will cover
# deployment and the rest next week."
_PUT_OFF_RE = re.compile(r"\b(?:we(?:'ll|\s+will)|let'?s|i(?:'ll|\s+will))\s+(?:cover|go\s+(?:over|through)|discuss|"
                         r"talk\s+about|walk\s+through|do)\b[^.]*\b(?:next\s+(?:week|time|session|call)|later|"
                         r"another\s+(?:session|time|call)|tomorrow|separately)\b", re.IGNORECASE)
_GAP_RES = [
    re.compile(r"\bthere(?:\s+is|\s+are|'s)\s+(?:no|not\s+(?:a|an|any))\s+(?:[a-z]+\s+){0,2}?"
               r"(?:plan|process|runbook|documentation|docs?|backups?|monitoring|alert(?:ing|s)?|tests?|testing|"
               r"dr|disaster\s+recovery|rto|rpo|sla|staging|environment|policy|procedure|on-?call|"
               r"rotation|dashboards?|automation|rollback|failover|owner|escalation|contact|diagram)\b", re.IGNORECASE),
    # Said with a contraction too: "We haven't documented the exact rollback
    # revision yet" was not a stated gap (a user's KT, 23 Oct).
    re.compile(r"\b(?:we|i|they|nobody|no\s*one|the\s+team)\s+(?:(?:have\s+|has\s+|had\s+)?(?:never|not)\s+|"
               r"(?:haven|hasn|hadn)'?t\s+(?:ever\s+)?)"
               r"(?:yet\s+)?(?:agreed|defined|documented|tested|checked|set\s+up|written|configured|decided|"
               r"measured|reviewed|practi[cs]ed|tried|done)\b", re.IGNORECASE),
    re.compile(r"\bnobody\s+(?:has\s+)?(?:tested|checked|knows|documented|reviewed)\b", re.IGNORECASE),
    *_UNCONFIRMED_RES,
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
    # Missing from a document, whatever it is: "The exact rollback revision
    # and weekend escalation contact are still missing from our runbook."
    re.compile(r"\b(?:is|are)\s+(?:still\s+|currently\s+|also\s+)?missing\s+from\s+(?:the|our|this)\s+(?:[a-z-]+\s+){0,2}?"
               r"(?:runbooks?|documentation|docs|handover|wiki|playbooks?|notes|document|page|confluence)\b", re.IGNORECASE),
    # Still to be written down: "Repository links ... still need to be added to
    # the handover document", "the team still needs to define response targets".
    re.compile(r"\b(?:still|yet)\s+needs?\s+to\s+(?:be\s+)?(?:add(?:ed)?|document(?:ed)?|defin(?:e|ed)|writ(?:e|ten)(?:\s+down)?|"
               r"creat(?:e|ed)|agree(?:d)?|decid(?:e|ed)|record(?:ed)?|captur(?:e|ed)|set\s+up)\b", re.IGNORECASE),
    # Not done yet: "has not been completed this year", "have not been
    # included in the handover documentation", "are not finalized".
    re.compile(r"\b(?:(?:has|have|had)\s+not|(?:hasn|haven|hadn)'?t)\s+(?:yet\s+)?been\s+(?:[a-z]+ly\s+)?"
               r"(?:completed|included|added|recorded|captured|finali[sz]ed|shared|written\s+down|performed)\b|"
               r"\b(?:is|are)\s+not\s+(?:yet\s+)?(?:finali[sz]ed|decided|known)\b", re.IGNORECASE),
    # Information said to be unavailable: "The latest recovery test date is
    # unavailable." Not "the service is unavailable", which is an outage.
    re.compile(r"\b(?:dates?|values?|numbers?|contacts?|owners?|urls?|links?|details?|revisions?|versions?|targets?|"
               r"figures?|schedules?|times?|names?)\s+(?:is|are)\s+(?:still\s+|currently\s+)?"
               r"(?:unavailable|unknown|not\s+(?:available|known))\b", re.IGNORECASE),
    # "The video-processing recovery procedure and formal escalation contacts
    # are still undocumented." (a user's KT, 25 Oct).
    re.compile(r"\b(?:is|are|was|were|remains?|stays?)\s+(?:still\s+|currently\s+|also\s+|largely\s+|so\s+far\s+)?"
               r"(?:undocumented|undefined|untested|unassigned|unowned|unwritten)\b", re.IGNORECASE),
    # "We do not have a confirmed rollback revision in the runbook."
    re.compile(r"\b(?:we|i|they|the\s+team)\s+(?:do\s+not|don'?t|does\s+not|doesn'?t)\s+(?:yet\s+|currently\s+)?have\s+"
               r"(?:a|an|any)\s+(?:confirmed|documented|agreed|defined|tested|written|current|up-to-date|verified)\b",
               re.IGNORECASE),
    _PUT_OFF_RE,
]
# An answer that opens with a plain "no" to the question before it.
# "Is there a DR plan? No." is a gap. "So it stores invoices in MySQL? No,
# it's PostgreSQL on Amazon RDS." corrects the guess with the value: not a
# gap (a three-person test, 25 Oct: Architecture read "stated as not in place").
_NEGATIVE_ANSWER_RE = re.compile(
    r"\?\s*(?:(?:no|nope)\b(?!,?\s+(?:it'?s|it\s+is|that'?s|that\s+is|they'?re|they\s+are|we\s+use|we'?re\s+(?:on|using)"
    r"|we\s+(?:moved|switched)|it\s+(?:runs|uses|lives|goes|was\s+moved)|it'?s\s+now|the\s+[\w-]+\s+(?:is|are|runs|uses))\b)"
    r"|not\s+really\b|not\s+yet\b|none\b|nothing\b)", re.IGNORECASE)


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


def is_unconfirmed(text: str) -> bool:
    """A stated gap that is a doubted claim, not an absence: "I think Redis is
    using 4 GB of memory, but I'm not completely sure" (a hedge on a claim,
    or a claim before "but"). Reported as unconfirmed, not "not in place".
    "Who do I escalate to? I am not sure anymore" claims nothing: absent."""
    text = text or ""
    if not any(rx.search(text) for rx in _UNCONFIRMED_RES):
        return False
    return bool(_HEDGE_RE.search(text) or _DOUBTED_SOURCE_RE.search(text) or _DOCUMENT_CAVEAT_RE.search(text)
                or states_something_before_gap(text))


# A back-reference in a sentence that only doubts what came before it.
_REFERS_BACK_RE = re.compile(r"\b(?:this|that|these|those)\b|^\W*it\b", re.IGNORECASE)


def doubts_statement_before(previous: str, sentence: str) -> bool:
    """`sentence` only doubts what `previous` said: "We also use Redis for
    caching, but I am unsure whether its memory limit is 4 GB or 8 GB." "That
    value needs confirmation before updating the operational documentation."
    Apart, the doubt was filed as a deployment step and the value read as
    settled, so the two are kept as one statement, as a correction is."""
    if not previous or not sentence or len(sentence.split()) > 16 or not _REFERS_BACK_RE.search(sentence):
        return False
    if not any(rx.search(sentence) for rx in _UNCONFIRMED_RES) or _DOUBTED_SOURCE_RE.search(sentence):
        return False
    # "I think Redis has a 4 GB memory limit, but this needs verification"
    # doubts its own claim, not the sentence before it, and "We haven't
    # documented the CDN cache rules yet, so that needs confirmation" states
    # a gap of its own (matrix C05).
    if _HEDGE_RE.search(sentence) or is_unconfirmed(sentence):
        return False
    if any(rx.search(sentence) for rx in _GAP_RES if rx not in _UNCONFIRMED_RES):
        return False
    return states_something(previous) or is_unconfirmed(previous)


# "Cosmos has continuous backup turned on, but we have never tried a restore"
# states a control that is in place, then a gap in it. Read as a gap alone,
# Disaster Recovery was reported Missing although the backup was stated.
# Not one that ends the sentence ("we have never done a cross-region restore
# though."): the clause it closes is the contrast.
_CONTRAST_RE = re.compile(r"(?:[,;]\s*|\s+)(?:but|however|though|although)\b(?!\W*$)", re.IGNORECASE)
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


# What a speaker says before admitting a gap, not a statement of its own:
# "..., and honestly we | haven't tested restoring", "I have to say we | ...",
# "... and that's | never been tested".
_GAP_LEAD_IN_RE = re.compile(
    r"(?:\b(?:and|so)\s+)?(?:\b(?:honestly|frankly|to\s+be\s+honest|i\s+(?:have\s+to|must)\s+(?:say|admit)|i\s+mean)\s*,?\s*)?"
    r"(?:\b(?:we|i|they|nobody|no\s*one|the\s+team|that'?s|that\s+(?:is|has)|it'?s|it\s+(?:is|has)|this\s+is"
    r"|which\s+(?:is|has))\s*)?$", re.IGNORECASE)
_RELATIVE_ANTECEDENT_RE = re.compile(
    r"\b(?:something|anything|nothing|everything|thing|things|one|part|area|bit|what|which|that|where)\s*$", re.IGNORECASE)


def states_something_before_gap(text: str) -> bool:
    """True when the gap is only in a contrasting clause ("..., but we have
    never tested it") and the clause before it states something on its own:
    for a question and its answer, the answer's own words, unhedged. With no
    contrast word ("We take nightly backups to S3 and we have never tested a
    restore", or speech to text that ran the two sentences together), the
    words before the gap must be a statement, of six words or more."""
    text = text or ""
    match = _CONTRAST_RE.search(text)
    if match:
        head = text[:match.start()].split("?")[-1]
        return len(head.split()) >= 4 and not is_gap_statement(head) and not _HEDGE_RE.search(head)
    # Only after an absence ("never tested"): a doubt is about the words
    # before it ("RPO four hours, or at least that is what the previous
    # document says"), which then state nothing settled; before a topic put
    # off, they close the session ("That is all we had time for today").
    if _PUT_OFF_RE.search(text) or any(rx.search(text) for rx in _UNCONFIRMED_RES):
        return False
    starts = [m.start() for m in (rx.search(text) for rx in _GAP_RES) if m]
    if not starts:
        return False
    head = _GAP_LEAD_IN_RE.sub("", text[:min(starts)].split("?")[-1]).strip()
    # The gap is a relative clause of the head: "the whole recovery thing is
    # something | we have never documented". Or the head is a short question
    # that lost its mark ("do we have a disaster recovery plan no honestly |
    # there is no dr plan"); a long one holds its answer ("what about backups
    # aurora has automated backups with point in time recovery ...").
    if _RELATIVE_ANTECEDENT_RE.search(head) or (_QUESTION_START_RE.match(head) and len(head.split()) <= 12):
        return False
    # A verb anywhere: a run of speech to text puts it far from the start
    # ("sql server full backup every night and log backups every fifteen
    # minutes copied to azure blob storage ...").
    from fact_units import _has_finite_verb
    return (len(head.split()) >= 6 and not is_gap_statement(head) and not _HEDGE_RE.search(head)
            and _has_finite_verb(head))


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
    r"cover(?:ing)?|talk(?:ing)?\s+(?:you\s+)?(?:about|through)|"
    # "Hi team, today I'll explain our order service." was printed under
    # System Overview (a user's KT, 23 Oct).
    r"explain(?:ing)?|introduc(?:e|ing)|describ(?:e|ing)|go(?:ing)?\s+(?:over|through)|"
    r"run(?:ning)?\s+(?:you\s+)?through|"
    r"giv(?:e|ing)\s+(?:you\s+)?(?:the|a|an)\s+(?:kt|handover|hand-?over|knowledge\s+transfer|overview|walk-?through|"
    r"tour)\s+(?:for|of|on|about))"
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
    return " ".join(s for s in kept if not is_handover_announcement(s) and not is_greeting_only(s)).strip()


# Words that carry no knowledge of their own: function words, filler verbs,
# reactions and placeholders ("thing", "way", "point").
_NOTHING_SPECIFIC_WORDS = frozenset("""
a an the and or but so to of in on at by for with from as is are was were be been being am it its it's this that
these those there here we you they i he she them us me my our your their his her him who what which when where why
how if then than too very really just also even still already only quite pretty much many more most some any all
each every no not yes yeah yep ok okay right well sure fine do does did done doing have has had having can could will
would should may might must shall let let's get gets got getting go goes going went gone make makes made take takes
took see sees saw seen look looks looked say says said tell told think thinks thought know knows knew want wants need
needs like likes mean means guess hope feel feels come comes came put give gives gave try tries tried keep keeps kept
show shows showed use uses used great good nice awesome cool useful easy simple interesting exciting annoying important
happy glad fantastic perfect amazing excellent wonderful thing things stuff way ways kind sort bit lot lots part side
point idea question questions thoughts sense one ones something anything everything nothing someone anyone everyone
time times today now later again back first last next actually basically probably maybe definitely obviously i'm i've
i'd you're we're we've they're that's there's here's don't doesn't didn't can't won't isn't aren't wasn't i'll we'll
you'll it'll thank thanks exactly absolutely correct true cheers bye hello hi hmm uh um
""".split())
# Where a system runs is knowledge even in two words: "It's all on premises."
_PLACE_WORDS = frozenset({"premises", "prem", "on-prem", "on-premises", "cloud", "region", "regions",
                          "datacenter", "datacentre", "datacenters", "datacentres"})
_LABEL_PREFIX_RE = re.compile(r"^[A-Z][\w'.-]*(?:\s[A-Z][\w'.-]*){0,2}:\s*")


def without_label(text: str) -> str:
    """The sentence without a leading speaker label ("Dana: All right, ...")."""
    return _LABEL_PREFIX_RE.sub("", (text or "").strip())


def says_nothing_specific(text: str) -> bool:
    """A remark with at most one word of its own and no tool, figure, name or
    known term: "Thoughts from your side?", "It does the job.", "This tool is
    very.", "And for sure, we will do that." Recorded talks filed dozens of
    these under System Overview (a 21-minute KT ran to 26 pages). Of the 1,180
    facts in the visible answer keys, none is a sentence like this."""
    from devops_transcription import get_known_terms

    body = without_label(text)
    # A question kept with the sentence after it is judged sentence by
    # sentence: "Thoughts from your side? It does the job." (matrix L05).
    parts = [p for p in re.split(r"(?<=[.!?])\s+", body) if p.strip()]
    if len(parts) > 1:
        return all(says_nothing_specific(p) for p in parts)
    # A figure, or a negation: "We don't have that." states a gap.
    if not body or re.search(r"\d|\b(?:no|not|never|none|nobody|nothing|nope)\b|n't\b", body, re.IGNORECASE):
        return False
    words = re.findall(r"[A-Za-z][\w'-]*", body)
    if any(w[:1].isupper() and w != "I" for w in words[1:]):
        return False
    # A line that names a topic leads what follows: in a question-and-answer
    # KT without punctuation "and my first week." and "who owns what." are the
    # receiver's questions; dropped, First 30-Day Ownership was reported
    # Missing (dev d05, noisy copy).
    from section_rules import _segment_topic_of, topic_anchor

    anchor = topic_anchor(body)
    if (anchor and anchor[0] == "heading") or any(_segment_topic_of(w.lower()) for w in words):
        return False
    own = [w for w in (x.lower() for x in words) if w not in _NOTHING_SPECIFIC_WORDS and len(w) > 1]
    if len(own) > 1:
        return False
    # A short instruction: "Swap back.", "Restart it."
    if own and words[0].lower() == own[0] and len(words) <= 4:
        return False
    known = get_known_terms()[1]
    return not any(w in known or w.rstrip("s") in known or w in _PLACE_WORDS for w in own)


def is_greeting_only(text: str) -> bool:
    """A sentence that is only a greeting: "Hello everyone." The copy edit can
    split "Hi everyone, today I will be handing over X" in two, and the
    greeting was printed under System Overview (the user's recordings, 24 Oct)."""
    match = _GREETING_PREFIX_RE.match(text or "")
    return bool(match) and not (text or "")[match.end():].strip()


def is_heading_only(text: str) -> bool:
    """A numbered heading read aloud: "Section two, architecture.", "Part
    four: release process." It names the topic of what follows
    (section_rules.segment_topics) and says nothing itself, yet it was
    printed as a bullet of its section (X82). Only a numbered one: in speech
    to text without punctuation a receiver's question reads like a heading
    ("Costs?" as "costs."), and dropping those left two sections of a
    question-and-answer KT reported Missing (dev d05, noisy copy). A phrase
    with a verb, a figure or a name is a statement."""
    from fact_units import _is_finite_verb, _tokens
    from section_rules import _NUMBERED_HEADING_RE, _SPEAKER_PREFIX_RE, _segment_topic_of, heading_phrase

    phrase = heading_phrase(text)
    numbered = _NUMBERED_HEADING_RE.match(_SPEAKER_PREFIX_RE.sub("", text or "", count=1).strip())
    if phrase is None or not numbered or re.search(r"\d", phrase):
        return False
    # A verb with something after it ("Terraform manages infrastructure"); a
    # last word that could be one is the heading's noun ("known issues").
    tokens = _tokens(phrase)
    if any(_is_finite_verb(tokens, i) for i in range(1, len(tokens) - 1)):
        return False
    # A capital after the first word is a name, unless the word is itself a
    # topic ("Backups and DR.").
    return not any(w[:1].isupper() and not _segment_topic_of(w) for w in phrase.split()[1:])


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

