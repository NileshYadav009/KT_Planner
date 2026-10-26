"""Who is handing over, and what of a conversation counts as the KT.

A KT session often has more than two people: the giver, the receiver, and
a colleague or a manager. The receiver asks, guesses and sometimes corrects
the giver; people make small talk. Before this module everything anyone said
was a fact: "So it stores invoices in MySQL, right?" printed as "It stores
invoices in MySQL." after the giver had said "No, it's PostgreSQL"; the
receiver's "I heard backups are kept for 30 days" stood beside the giver's
"Actually, 14 days"; the offsite small talk described the Production
environment (25 Oct, a three-person test).

Roles (detect_roles): the giver is the speaker who states the most, unless
the user chose another (the job's "giver"); a receiver is a speaker whose
turns are mostly questions; anyone else (a colleague, a manager) is a source
like the giver. What counts (apply_roles):

- small talk, by anyone, is left out;
- a receiver's question goes with the answer a trusted speaker gives it; a
  question nobody answers is left out, and when it challenges what the giver
  just said ("Isn't it PagerDuty now?" after "Alerts go to Slack."), the
  giver's statement is flagged to confirm;
- a receiver's statement counts when the giver confirms it ("Oh right, yes,
  it's five nodes"), and then what the giver said before it on that point is
  withdrawn; corrected by the giver, it is left out; never answered, it is
  listed to confirm, not shown as a fact.

Nothing is dropped silently: each left-out statement is recorded with its
reason (the facts ledger), and what needs confirming becomes a follow-up.
"""
import re
from typing import Any, Dict, List, Optional

from dialogue import speaker_labels

# A turn label at the start of a line or after a sentence: "Meera: ...".
_TURN_RE = re.compile(r"(?:^|(?<=[.!?])\s+|\n\s*)([A-Z][\w'.-]*(?:\s[A-Z][\w'.-]*){0,2}):\s")
_LABEL_LEAD_RE = re.compile(r"^\s*[A-Z][\w'.-]*(?:\s[A-Z][\w'.-]*){0,2}:\s+")

# A receiver asks more than they state: at least this share of their
# sentences are questions.
RECEIVER_QUESTION_SHARE = 0.4

# Talk about anything but the system: "How was the offsite?", "Good, the
# food was great.", "Ha, too much coffee though." Recognised by its social
# phrasing, not its nouns: "our food delivery platform", "demand during
# weekends and dinner hours" and "more on bank holidays" are about the system
# (a food-delivery KT and a toll KT lost them to a word list). Only with no
# sign of the system in it (no product, figure or operations word).
_LEISURE = r"(?:weekend|offsite|off-site|holiday|holidays|vacation|trip|break|lunch|breakfast|dinner|flight|party|" \
           r"game|match|time\s+off|day\s+off|conference)"
_SMALL_TALK_RE = re.compile(
    r"\bhow\s+(?:was|were|is)\s+(?:the|your)\s+" + _LEISURE + r"\b"
    r"|\bhow\s+are\s+(?:you|things)\b|\bhow'?s\s+(?:it\s+going|everyone|your\s+(?:day|week|weekend))\b"
    r"|\bdid\s+you\s+(?:have\s+a\s+(?:good|nice|great)|enjoy)\b"
    r"|\b(?:the\s+)?(?:food|coffee|weather|lunch|dinner|trip|offsite|weekend|party|venue|hotel)\s+(?:was|is|were)\s+"
    r"(?:\w+\s+)?(?:great|good|nice|lovely|amazing|awesome|terrible|awful|fine|ok(?:ay)?|bad|cold|hot)\b"
    r"|\btoo\s+much\s+(?:coffee|food|cake|tea)\b|\b(?:good|great|nice|lovely)\s+" + _LEISURE + r"\b"
    r"|\bnice\s+to\s+(?:meet|see)\s+you\b|\bgood\s+to\s+see\s+you\b|\blong\s+time\s+no\s+see\b"
    r"|\bhope\s+you\s+(?:had|are|'re)\b", re.IGNORECASE)
_SYSTEM_SIGNAL_RE = re.compile(
    r"\d|\b(?:prod\w*|staging|dev|deploy\w*|release\w*|database|db|cluster|server|service|api|pipeline|alert\w*|"
    r"backup\w*|incident\w*|outage|on-?call|queue|cache|node\w*|pod\w*|job\w*|config\w*|secret\w*|access|"
    r"monitor\w*|dashboard\w*|log\w*|customer\w*|users?|traffic\s+(?:spike|peak|grows))\b", re.IGNORECASE)

# The giver correcting what was just said: "No, it's PostgreSQL", "Actually,
# backups are kept for 14 days, not 30."
_CORRECTS_RE = re.compile(
    r"^\W*(?:(?:oh|well|um|uh)\W+)?(?:no\b|nope\b|not\s+quite\b|not\s+really\b|actually\b|that'?s\s+(?:not|wrong|"
    r"outdated|old)\b|it'?s\s+not\b|that\s+changed\b)|\bnot\s+(?:\d[\d.,]*|one|two|three|four|five|six|seven|"
    r"eight|nine|ten|twelve|thirty|sixty)\b", re.IGNORECASE)
# The giver agreeing: "Oh right, yes, it's five nodes since the scaling.",
# "Yes, that's right.", "Good catch."
_CONFIRMS_RE = re.compile(
    r"^\W*(?:(?:oh|ah|um|uh)\W+)*(?:yes|yeah|yep|yup|right|correct|exactly|true|indeed|"
    r"that'?s\s+(?:right|correct|true)|good\s+catch|you'?re\s+right|fair\s+point)\b", re.IGNORECASE)
# Moving the conversation on rather than answering: "Let's not go down that
# path now, we're short on time.", "Let's park that."
_DEFLECTS_RE = re.compile(
    r"\blet'?s\s+(?:not\s+go\s+(?:down|into)|park|move\s+on|skip|come\s+back\s+to|take\s+(?:that|it)\s+offline)\b|"
    r"\b(?:we'?re|we\s+are)\s+(?:short\s+on|out\s+of|running\s+out\s+of)\s+time\b", re.IGNORECASE)

# A question that doubts what was just said and offers another value:
# "Isn't it PagerDuty now?", "Wasn't that moved to Datadog?", "It's five
# nodes now, right?". "Do we use Datadog?" only asks.
_CHALLENGE_RE = re.compile(
    r"^\W*(?:(?:but|so|and|wait|hmm|hold\s+on)\W+)*(?:isn'?t|wasn'?t|aren'?t|weren'?t|shouldn'?t|didn'?t|don'?t|"
    r"doesn'?t|is\s+(?:it|that)\s+(?:still|not)|are\s+(?:we|you)\s+sure)\b|\b(?:now|instead|anymore|these\s+days)\W*\?\s*$"
    r"|,\s*(?:right|correct)\W*\?\s*$", re.IGNORECASE)
# Asking to hear it again: "Sorry, my connection dropped, can you repeat
# that?" was joined to the answer that followed as if it asked it.
_REPEAT_RE = re.compile(
    r"\b(?:can|could|would)\s+you\s+(?:please\s+)?(?:repeat|say\s+(?:that|it)\s+again|go\s+over\s+(?:that|it)\s+again)"
    r"\b|^\W*(?:sorry|pardon|come\s+again)\W*$", re.IGNORECASE)
_NUMBER_WORD_RE = re.compile(r"\b(?:\d[\d.,]*|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                             r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
                             r"fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)\b", re.IGNORECASE)

_PRODUCTS = None


def _products(text: str) -> set:
    """The products and tools a sentence names (the component catalog and
    the field populator's tool names: PagerDuty and Slack are only in the
    second)."""
    global _PRODUCTS
    if _PRODUCTS is None:
        from component_catalog import CATALOG_REGEX, lookup
        from field_populator import PATTERN_EXTRACTORS
        _PRODUCTS = (CATALOG_REGEX, lookup, PATTERN_EXTRACTORS["tools"])
    regex, lookup, tools = _PRODUCTS
    out = set()
    for m in regex.finditer(text or ""):
        product = lookup(m.group(1))
        out.add((product.name if product else m.group(1)).lower())
    out |= {name.lower() for _, _, name in tools.spans(text or "")}
    return out


def _values(text: str) -> set:
    """The numbers and products a sentence states."""
    return {m.group(0).lower() for m in _NUMBER_WORD_RE.finditer(text or "")} | _products(text)


def _body(text: str) -> str:
    return _LABEL_LEAD_RE.sub("", text or "", count=1).strip()


def turn_segments(transcript: str) -> Optional[List[Dict[str, Any]]]:
    """A pasted transcript with speaker labels ("Meera: ...") as one untimed
    segment per turn, each with its speaker; None without labels. The label
    stays in the text, as before: only who said it is added."""
    labels = speaker_labels(transcript or "")
    if not labels:
        return None
    starts = [(m.start(1), m.group(1)) for m in _TURN_RE.finditer(transcript) if m.group(1) in labels]
    if not starts:
        return None
    segments: List[Dict[str, Any]] = []
    head = transcript[:starts[0][0]].strip()
    if head:
        segments.append({"text": head, "speaker": None})
    for i, (pos, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(transcript)
        text = transcript[pos:end].strip()
        if text:
            segments.append({"text": text, "speaker": name})
    return [{"id": i, "seek": 0, "start": 0.0, "end": 0.0, "text": seg["text"], "avg_logprob": -0.3,
             "compression_ratio": None, "no_speech_prob": None, "speaker": seg["speaker"], "untimed": True}
            for i, seg in enumerate(segments)]


def rename_turn_labels(transcript: str, names: Dict[str, str]) -> str:
    """A pasted transcript with its speakers renamed ({"Speaker 1": "Priya
    Sharma"}): only the turn labels change, never what was said."""
    def rename(match):
        old = match.group(1)
        if old not in names:
            return match.group(0)
        whole, start = match.group(0), match.start(0)
        return whole[:match.start(1) - start] + names[old] + whole[match.end(1) - start:]

    return _TURN_RE.sub(rename, transcript or "")


# A greeting or the start of the session on its own: "Morning both.",
# "Okay, let's begin." (left in System Overview).
_SESSION_START_RE = re.compile(
    r"^\W*(?:(?:ok(?:ay)?|so|alright|right|well)\W+)*(?:(?:good\s+)?(?:morning|afternoon|evening)(?:\s+(?:both|all|"
    r"everyone|everybody|team|folks|guys))?|let'?s\s+(?:begin|start|get\s+started|kick\s+(?:it\s+)?off|dive\s+in|"
    r"get\s+going))\W*$", re.IGNORECASE)


def is_small_talk(text: str) -> bool:
    body = _body(text)
    if _SESSION_START_RE.match(body):
        return True
    return bool(_SMALL_TALK_RE.search(body)) and not _SYSTEM_SIGNAL_RE.search(body) and not _products(body)


def detect_roles(units: List[Any], giver: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """{"giver", "receivers", "others", "detected", "speakers"} for a
    conversation of two or more speakers (each unit has .text and .speaker);
    None otherwise. `giver` is the user's choice, when it names a speaker."""
    stats: Dict[str, Dict[str, int]] = {}
    for unit in units:
        name = getattr(unit, "speaker", None)
        if not name:
            continue
        s = stats.setdefault(name, {"words": 0, "questions": 0, "sentences": 0})
        body = _body(unit.text)
        s["sentences"] += 1
        if body.rstrip().endswith("?"):
            s["questions"] += 1
        elif not is_small_talk(body):
            s["words"] += len(body.split())
    if len(stats) < 2:
        return None
    order = list(stats)
    chosen = giver if giver in stats else max(order, key=lambda n: (stats[n]["words"], -order.index(n)))
    receivers = [n for n in order if n != chosen and stats[n]["questions"]
                 and stats[n]["questions"] >= RECEIVER_QUESTION_SHARE * stats[n]["sentences"]]
    return {"giver": chosen, "receivers": receivers,
            "others": [n for n in order if n not in receivers and n != chosen],
            "detected": giver not in stats,
            "speakers": [dict(stats[n], name=n) for n in order]}


def _shares_topic(a: str, b: str) -> bool:
    """Two sentences about one thing: a product, or a noun of three or more
    letters, in common (numbers aside)."""
    if _products(a) & _products(b):
        return True
    words = lambda t: {w[:5] for w in re.findall(r"[a-z]{4,}", t.lower())} - _COMMON
    return len(words(a) & words(b)) >= 1


_COMMON = {w[:5] for w in """that this there they them their what which with have from were will would could should
about think since right yeah just really actually going because still also okay sorry thanks maybe""".split()}


def apply_roles(units: List[Any], roles: Optional[Dict[str, Any]], merge, excluded: List[Dict[str, str]],
                findings: List[Dict[str, Any]]) -> List[Any]:
    """The units that count as the KT, in order, with each receiver question
    joined to the answer a trusted speaker gave it (`merge(question,
    answer)`). Left-out units go to `excluded` with the reason; what needs
    confirming goes to `findings` ({"kind", "text", "by", "about"})."""
    def leave_out(unit, reason):
        excluded.append({"text": re.sub(r"\s+", " ", (unit.text or "").strip()), "reason": reason})

    if roles is None:
        kept = []
        for unit in units:
            if is_small_talk(unit.text):
                leave_out(unit, "small talk, not about the system")
            elif _REPEAT_RE.search(_body(unit.text)):
                leave_out(unit, "asking to hear it again, not about the system")
            else:
                kept.append(unit)
        return kept

    giver = roles["giver"]
    receivers = set(roles["receivers"])

    def trusted(unit) -> bool:
        return bool(unit.speaker) and unit.speaker not in receivers

    out: List[Any] = []
    withdrawn = set()          # ids of units the giver took back
    skip = set()
    for i, unit in enumerate(units):
        if i in skip:
            continue
        body = _body(unit.text)
        if is_small_talk(unit.text):
            leave_out(unit, "small talk, not about the system")
            continue
        if _REPEAT_RE.search(body):
            leave_out(unit, "asking to hear it again, not about the system")
            continue
        if trusted(unit) and _DEFLECTS_RE.search(body) and not _products(body):
            leave_out(unit, "moving the conversation on, not about the system")
            continue
        if unit.speaker not in receivers:
            out.append(unit)
            continue

        # A receiver's turn. What the next trusted speaker says to it, within
        # the next two turns (another receiver may speak in between).
        reply_at = next((j for j in range(i + 1, min(len(units), i + 4)) if trusted(units[j])
                         and not is_small_talk(units[j].text) and not _REPEAT_RE.search(_body(units[j].text))), None)
        reply = units[reply_at] if reply_at is not None else None
        reply_body = _body(reply.text) if reply is not None else ""
        deflected = reply is not None and bool(_DEFLECTS_RE.search(reply_body))
        before = next((u for u in reversed(out) if u.speaker == giver), None)
        if body.endswith("?"):
            if reply is not None and not deflected and reply_at == i + 1:
                out.append(merge(unit, reply))
                skip.add(reply_at)
                continue
            leave_out(unit, f"a question from {unit.speaker} that was not answered")
            if before is not None and _CHALLENGE_RE.search(body) and _values(body) and (
                    _values(body) - _values(before.text) or _shares_topic(body, before.text)):
                findings.append({"kind": "challenged", "text": body, "by": unit.speaker,
                                 "about": _body(before.text), "giver": giver})
            else:
                findings.append({"kind": "unanswered", "text": body, "by": unit.speaker, "giver": giver})
            continue

        if reply is not None and not deflected and _CORRECTS_RE.search(reply_body):
            leave_out(unit, f"said by {unit.speaker}; corrected by {reply.speaker}")
            continue
        if reply is not None and not deflected and _CONFIRMS_RE.search(reply_body):
            # Confirmed: the point the receiver corrected is taken back, when
            # the two give it different values ("three nodes" / "it's five
            # now" / "Oh right, yes, it's five nodes").
            said = f"{body} {reply_body}"
            if before is not None and id(before) not in withdrawn and _values(before.text) \
                    and _values(said) - _values(before.text) and _shares_topic(said, before.text):
                withdrawn.add(id(before))
                out.remove(before)
                leave_out(before, f"corrected by {unit.speaker}, and {reply.speaker} agreed")
            # The receiver's words are kept when the agreement does not say
            # the point again ("Yes, that's right.").
            if len(_body(re.sub(_CONFIRMS_RE, "", reply_body)).split()) >= 3 and _shares_topic(body, reply_body):
                leave_out(unit, f"said by {unit.speaker}; {reply.speaker} agreed and said it again")
            else:
                out.append(merge(unit, reply))
                skip.add(reply_at)
            continue
        leave_out(unit, f"said by {unit.speaker}, not confirmed by {giver}")
        findings.append({"kind": "unconfirmed", "text": body, "by": unit.speaker, "giver": giver})
    return out


def roles_notice(roles: Optional[Dict[str, Any]]) -> Optional[str]:
    """One line for the document: who is handing over, and what counts."""
    if not roles:
        return None
    others = [n for n in roles["others"]]
    sources = " and ".join([roles["giver"]] + others) if others else roles["giver"]
    how = "detected: states the most" if roles["detected"] else "chosen on the KT page"
    text = f"Handed over by {roles['giver']} ({how})."
    if roles["receivers"]:
        text += (f" Facts come from {sources}; what {' and '.join(roles['receivers'])} said counts only where "
                 f"{roles['giver']} confirmed it, and the rest is listed to confirm.")
    return text
