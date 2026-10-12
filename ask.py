"""Ask the KT (P3-1): answer a question with the document's own statements.

"Who do I call when the import fails?" is answered by finding the statements
of the KT document that say so and showing them, each with the section it is
in and the transcript sentence it comes from. Nothing is written by a model,
so an answer can be wrong only by picking the wrong statement, never by
inventing one, and the reader can check every line against what was said.

Statements are ranked by the words they share with the question (BM25, with
light stemming) plus the topic the question is about: "who / call /
escalate" points at Ownership & Escalation, "back up / restore / RTO" at
Disaster Recovery. When the question's topic is one the KT never covered,
the answer says so and names the knowledge gap, instead of offering the
nearest unrelated sentence.

Runs on the API process, which loads no models.
"""
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from knowledge.evidence import block_units, source_lookup, unit_sources
from renderers.blocks.common import SECTION_STATUS_MESSAGES

MAX_QUESTION = 300
DERIVED_SECTIONS = frozenset({"quick_reference", "kt_coverage", "signoff"})
DIGEST_SECTION = "tribal_knowledge"
# A statement off the question's topic must share more than a common word.
MIN_OFF_TOPIC_SCORE = 1.0
# Which section a question is about, from its wording. A question may be
# about several ("what breaks during the month-end peak").
TOPICS = [
    ("ownership_escalation", r"\bwho\b|\bwhom\b|\bcall\b|\bcontact|\bescalat|\bowner|\bowns\b|\bresponsible|"
                             r"\bon[\s-]?call\b|\bpage[sd]?\b|\bpaging\b|\bteam\b|\bmanager\b"),
    ("deployment_and_rollback", r"\bdeploy|\brelease|\bship\b|\brollback|\broll\s+back|\bpipeline|\bci\b|\bcd\b|"
                                r"\bpromot|\bmerge|\bbuild"),
    ("disaster_recovery", r"\bbackup|\bback\s+up|\brestor|\bdisaster|\b(?:rto|rpo)\b|\bfailover|\breplica|"
                          r"\bregion\s+(?:goes|is)\s+down|\brecover"),
    ("monitoring_observability", r"\bmonitor|\balert|\bdashboard|\bmetric|\blogs?\b|\blogging|\bobservab|\bslo\b|"
                                 r"\bwatch|\btrac(?:e|ing)\b"),
    ("common_failures", r"\bfail|\bbreak|\bbroke|\berror|\bincident|\boutage|\bdown\b|\bfix|\bissue|\bproblem|"
                        r"\bgoes\s+wrong|\bstuck|\bslow"),
    ("danger_zones", r"\bnever\b|\bavoid|\bdanger|\bcareful|\brisk|\bshould\s*n[o']t|\bmust\s*n[o']t|\bdon'?t\b|"
                     r"\bdo\s+not\b|\btouch"),
    ("known_bad_days", r"\bwhen\b.*\b(?:busy|peak|freeze|quiet)|\bbusy|\bpeak|\bfreeze|\bholiday|\bmonth[\s-]end|"
                       r"\bquarter|\bcalendar|\bbad\s+days?"),
    ("environments", r"\benvironment|\bstaging|\bprod\b|\bproduction\b|\bdev\b|\btest\s+env|\buat\b|\bsandbox"),
    ("security_controls", r"\bsecur|\bsecret|\bpassword|\bcredential|\baccess\b|\bauth|\bencrypt|\bkey\s*vault|"
                          r"\bpermission|\biam\b|\bfirewall|\bvpn\b"),
    ("cost_optimization", r"\bcost|\bspend|\bbill|\bbudget|\bcheap|\bexpensive|\bsav(?:e|ing)"),
    ("day1_survival_checklist", r"\bfirst\s+day|\bday\s*(?:one|1)\b|\bstart(?:ing)?\b|\bget\s+access|\bonboard|"
                                r"\bnew\s+(?:joiner|engineer|starter)"),
    ("first_30_day_ownership", r"\bfirst\s+(?:month|30|thirty|weeks?)|\b30[\s-]day|\bweek\s+\w+"),
    ("open_responsibilities", r"\bunfinished|\bopen\s+(?:items?|work|tasks?)|\bleft\s+to\s+do|\bpending|\btodo|"
                              r"\bto[\s-]do|\bhalf[\s-]done|\bnot\s+(?:yet\s+)?done"),
    ("architecture_reference", r"\barchitect|\bbuilt\s+(?:on|with)|\bruns?\s+on\b|\bstack|\bdatabase|\bqueue|"
                               r"\bcache|\bcomponent|\bhosted|\binfrastructure"),
    ("system_overview", r"\bwhat\s+does\s+(?:it|the\s+\w+|\w+)\s+do\b|\bwhat\s+is\s+(?:it|this\s+system)\b|"
                        r"\bpurpose|\boverview|\busers?\b|\bcustomers?\b|\bvolume|\bhow\s+many"),
]
_TOPIC_RES = [(sid, re.compile(rx, re.IGNORECASE)) for sid, rx in TOPICS]
TOPIC_BOOST = 1.5
_STOPWORDS = frozenset("""a an the and or but if then so of to in on at by for with from into onto about as is are
was were be been being do does did done have has had having it its this that these those there here what which who
whom whose when where why how i me my we our you your they them their he she his her can could should would will
shall may might must not no any some all each every than too very just also only up out over under again once
please tell know""".split())


def _stem(word: str) -> str:
    for suffix in ("ations", "ation", "ities", "ity", "ness", "ings", "ing", "ers", "er", "ies", "ied", "es", "ed",
                   "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)] + ("y" if suffix in ("ies", "ied") else "")
    return word


# Two-word terms read as one: "roll back" a release is not the "back end".
_COMPOUNDS = re.compile(r"\b(roll|back|front|on|fail|set|log)[\s-](back|up|end|call|over|in)\b")


def _terms(text: str) -> List[str]:
    text = _COMPOUNDS.sub(r"\1\2", (text or "").lower())
    return [_stem(w) for w in re.findall(r"[a-z0-9][a-z0-9.+/#-]*", text) if w not in _STOPWORDS]


def question_topics(question: str) -> List[str]:
    topics = [sid for sid, rx in _TOPIC_RES if rx.search(question or "")]
    # "What is the RTO?" is about recovery, not the overview: the overview
    # counts only when nothing more specific does.
    return [t for t in topics if t != "system_overview"] or topics


def _labelled_units(block: Dict[str, Any]) -> List[str]:
    """block_units, with a table row's label in front of its value: on its
    own, "High" (business criticality) answers nothing."""
    units = block_units(block) or []
    rows = block.get("rows") or []
    if block.get("type") == "TechnologyGrid" and len(rows) == len(units):
        return [f"{r.get('label')}: {u}" if r.get("label") and u else u for r, u in zip(rows, units)]
    return units


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def statements(knowledge_object: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every fact-bearing unit of the document, with its section and sources.
    Additional Notes lines that repeat a statement of another section (its
    digest of the safety-critical ones) are left out."""
    out, digest = [], []
    for section in knowledge_object.get("rendered_sections") or []:
        sid = section.get("section_id")
        if not sid or sid in DERIVED_SECTIONS:
            continue
        for block in section.get("blocks") or []:
            for i, text in enumerate(_labelled_units(block)):
                text = str(text).strip()
                if text and text not in SECTION_STATUS_MESSAGES:
                    unit = {"text": text, "section_id": sid, "section": section_title(knowledge_object, sid),
                            "source_ids": unit_sources(block, i)}
                    (digest if sid == DIGEST_SECTION else out).append(unit)
    stated = [_key(u["text"]) for u in out]
    for unit in digest:
        head = _key(unit["text"].split("…")[0])[:60]
        if not any(head and head in k for k in stated):
            out.append(unit)
    return out


def section_title(knowledge_object: Dict[str, Any], section_id: str) -> str:
    """The section's title as the schema writes it (documents stored before
    the Title Case schema have upper-case titles)."""
    from kt_schema_loader import SCHEMA

    title = next((s.get("title") for s in SCHEMA if s.get("id") == section_id), None)
    if title:
        return title
    rendered = next((s.get("section_title") for s in knowledge_object.get("rendered_sections") or []
                     if s.get("section_id") == section_id), None)
    return rendered or section_id.replace("_", " ").capitalize()


def _gaps(knowledge_object: Dict[str, Any]) -> List[str]:
    for section in knowledge_object.get("sections") or []:
        if section.get("id") == "kt_coverage":
            return [str(g) for g in section.get("_knowledge_gaps") or [] if str(g).strip()]
    return []


def answer(knowledge_object: Dict[str, Any], question: str, limit: int = 5) -> Dict[str, Any]:
    question = re.sub(r"\s+", " ", question or "").strip()[:MAX_QUESTION]
    topics = question_topics(question)
    units = statements(knowledge_object)
    q_terms = set(_terms(question))
    docs = [Counter(_terms(u["text"])) for u in units]
    n = len(docs) or 1
    avg_len = (sum(sum(d.values()) for d in docs) / n) or 1.0
    df = Counter(t for d in docs for t in d)

    scored = []
    for unit, d in zip(units, docs):
        length = sum(d.values()) or 1
        words = sum(math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * d[t] * 2.2 / (d[t] + 1.2 * (0.25 + 0.75 * length / avg_len))
                    for t in q_terms if d.get(t))
        on_topic = unit["section_id"] in topics
        # On the question's topic, a statement answers even with no word in
        # common ("Who do I call?" / "Escalate to Marta Kowalski"). Off it,
        # it must share more than a common word.
        if not on_topic and words < MIN_OFF_TOPIC_SCORE:
            continue
        # The first topic is what is asked ("who" in "Who do I call when it
        # is down?"); later ones are the situation it is asked about.
        boost = TOPIC_BOOST * (1.0 if topics and unit["section_id"] == topics[0] else 0.5) if on_topic else 0.0
        scored.append((words + boost, on_topic, unit))
    scored.sort(key=lambda s: -s[0])
    # Once something on the topic answers, an off-topic line must beat it.
    best_on_topic = max((s for s, on, _ in scored if on), default=None)
    if best_on_topic is not None:
        scored = [s for s in scored if s[1] or s[0] > best_on_topic]

    lookup = source_lookup(knowledge_object)
    answers = []
    for score, on_topic, unit in scored[:limit]:
        answers.append({"text": unit["text"], "section": unit["section"], "section_id": unit["section_id"],
                        "score": round(score, 2), "on_topic": on_topic,
                        "sources": [{k: lookup[i].get(k) for k in ("id", "quote", "start", "speaker", "session")}
                                    for i in unit["source_ids"] if i in lookup]})

    covered = {u["section_id"] for u in units}
    uncovered = [sid for sid in topics if sid not in covered]
    gaps = []
    if uncovered and not any(a["on_topic"] for a in answers):
        titles = {sid: section_title(knowledge_object, sid) for sid in uncovered}
        words = {w for title in titles.values() for w in _terms(title) if len(w) > 3}
        gaps = [g for g in _gaps(knowledge_object) if words & set(_terms(g))] or \
               [f"{titles[sid]} was not covered in the KT session." for sid in uncovered]
    # Specific terms of the question no statement contains ("What is the
    # RTO?" when the KT gives none): the nearest statements are shown, and
    # the reader is told the KT never says.
    specific = re.findall(r"\b[A-Z][A-Z0-9]{1,6}\b|\b\w*\d\w*\b", question)
    never = [t for t in dict.fromkeys(specific) if df.get(_stem(t.lower()), 0) == 0]
    return {"question": question, "topics": topics, "answers": answers, "not_covered": gaps, "never_mentions": never}
