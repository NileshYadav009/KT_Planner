"""Atomic facts: the statements one spoken sentence holds.

People speak in sentences; a KT document is made of facts. "Developers own
business services, Platform Engineering owns EKS and deployment tooling, and
the Data Platform team manages MSK" is three ownership facts, and "Recovery
testing is performed twice a year, but the RTO and RPO have not been
documented" is a recovery fact and a knowledge gap. Classified as one unit,
such a sentence went to one section whole, and every field read it as one
value. The mapper classifies facts instead: split_into_facts() runs before
classification, and each fact keeps the sentence it came from.

A sentence is cut at a clause boundary (", and", ", but", ", while",
", whereas", ";", or the comma of a list of clauses) only when the pieces on
both sides are statements of their own: an explicit subject and a finite
verb. A piece that leans on the one before it stays attached:

- it starts with a pronoun or a connective ("and it is owned by...",
  "so restart...", "which caches..."), or refers back to the clause before
  ("..., and a second function rolls them up into ..."): the next step of
  the same flow;
- it has no subject ("..., pushes them to ECR");
- it explains the clause before it ("..., and the fix is to..."), or is
  its result ("..., Dataflow jobs process them, and the results land in
  BigQuery");
- the sentence opens with a condition ("If the queue backs up, ..."), which
  governs every clause after it;
- the clause before it is an instruction ("Never deploy on Fridays, we had
  an outage"): a rule keeps its reason and its next step;
- it continues a list of nouns ("We keep two environments, staging and
  prod, ...") or a lead-in phrase ("...; for anything with invoices, ...").

The pieces are verbatim slices of the sentence. Linguistic rules only: no
section, tool or transcript is named here.
"""
import re
from typing import List, Tuple

# A clause boundary and the connective it consumes. ", or" is left out: the
# pieces of "restart the pod, or fail over" are alternatives, not facts.
_BOUNDARY_RE = re.compile(r",\s+(?:and|but|while|whereas)\s+|;\s+(?:and\s+|but\s+)?|,\s+", re.IGNORECASE)

# A sentence that opens with a subordinate clause: the condition applies to
# every clause after it ("When traffic spikes, the autoscaler adds nodes, and
# the cluster rebalances").
_SUBORDINATE_OPENER_RE = re.compile(
    r"^\W*(?:if|when|whenever|once|after|before|unless|until|while|although|though|because|since|"
    r"as\s+soon\s+as|in\s+case|provided|whether|even\s+if)\b", re.IGNORECASE)

# Words a piece cannot start with and still stand alone.
_LEANING_OPENERS = {
    # pronouns and demonstratives pointing back
    "it", "its", "it's", "they", "them", "their", "theirs", "this", "that", "these", "those", "he", "she",
    "him", "her", "his", "which", "who", "whom", "whose", "where", "such", "same", "both", "either",
    "neither", "one", "ones", "other", "others", "that's", "they're", "they've", "it'll", "that'll", "he's",
    "she's", "who's", "they'll",
    # connectives that make the piece a consequence or a sequel
    "so", "then", "thus", "hence", "therefore", "otherwise", "because", "since", "as", "also", "too",
    "plus", "or", "nor", "and", "but", "if", "when", "unless", "until", "after", "before", "once",
    "though", "although", "whereas", "while", "except", "including", "especially", "mainly", "mostly",
    "usually", "typically", "sometimes", "later", "afterwards", "instead", "only",
}

# Pronouns that point back to something the sentence named before.
_BACK_REFERENCES = {"it", "its", "it's", "them", "they", "their", "theirs", "they're", "they've"}

# A piece opening with a preposition or a time word is a phrase that sets the
# scene for the clause after it ("During dinner peaks, the most common issue
# is ..."), not a statement ("dinner peaks" is not a subject and a verb).
_ADVERBIAL_OPENERS = {
    "during", "in", "on", "at", "for", "from", "with", "without", "by", "under", "over", "across", "within",
    "between", "through", "throughout", "per", "via", "around", "about", "outside", "inside", "like",
    "unlike", "besides", "despite", "following", "after", "before", "today", "currently", "now", "here",
    "right", "okay", "ok", "yes", "no", "well", "basically", "actually", "again", "first", "second",
    "third", "finally", "lastly", "next", "meanwhile", "overall", "generally", "normally",
}

# "..., and the fix is to ...", "...; the reason is ...": the piece explains
# the one before it. "..., Dataflow jobs process them, and the results land in
# BigQuery": the results (or output) of the step before are its next step, not
# a fact of their own. Cut off, a one-call LLM check filed "the results land
# in BigQuery" beside the system's purpose, and a flow was drawn from the
# system itself.
_EXPLAINS_RE = re.compile(
    r"^(?:the\s+)?(?:\w+\s+)?(?:fix|fixes|cause|causes|root\s+cause|reason|symptom|symptoms|remediation|workaround|"
    r"mitigation|resolution|solution|answer|results?|outcomes?|outputs?|impacts?|effects?|consequences?|"
    r"exceptions?|catch)\b",
    re.IGNORECASE)

_AUXILIARIES = {
    "is", "are", "was", "were", "am", "has", "have", "had", "does", "do", "did", "can", "cannot", "could",
    "will", "would", "should", "shall", "must", "may", "might", "need", "needs", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't", "can't", "couldn't",
    "won't", "wouldn't", "shouldn't", "mustn't", "gets", "got", "get", "keeps", "kept", "stays", "stay",
    "remains", "remain", "becomes", "become", "became", "seems", "seem", "goes", "go", "went", "comes",
    "come", "came",
}

# Verbs that state what a system, team or tool does. Finite forms are
# derived below; irregular pasts are listed.
_VERBS = """
accept access add aggregate alert allow apply approve archive arrive assign authenticate authorise
authorize automate back balance belong block book bootstrap break build cache call capture carry cause
change check clean collect combine compile configure connect consume contain control coordinate copy
cost cover crash create decide decrease define delete deliver depend deploy describe detect drive drop
email emit enable encrypt enforce enter escalate exceed expire expose fail feed fetch fill filter fire
fix flag flow forward freeze fund gate generate grant grow guard handle hit hold host include increase
ingest inspect install integrate invoke involve isolate issue keep kick lag land launch lead limit list
live load lock log look maintain manage map mean merge migrate mirror monitor mount move need notify own
page pass patch pay peak perform persist pick ping place poll power process produce promote protect
provide provision publish pull push query queue reach read rebuild receive record recover reduce refresh
register reject release rely remove render replace replicate report request require reserve reset
resolve respond restart restore retain retry return review roll rotate route run save scale scan
schedule score send serve set ship show shut sign sit slow sort spike spin split start stop store stream
submit support sync synchronise synchronize take talk target terminate test throttle time track trigger
tune update upload use validate verify wait warm watch work wrap write
""".split()
_IRREGULAR = {
    "built", "kept", "sent", "held", "took", "taken", "ran", "led", "paid", "made", "makes", "make", "wrote",
    "written", "read", "set", "sat", "hit", "put", "puts", "broke", "broken", "froze", "frozen", "fed", "lit",
    "spun", "shut", "split", "won", "lost", "loses", "lose", "found", "finds", "find", "gave", "given",
    "gives", "give", "told", "tells", "tell", "knew", "knows", "know", "thought", "thinks", "think",
    "brought", "brings", "bring", "bought", "sold", "sells", "chose", "chosen", "drew", "drawn", "grew",
    "grows", "fell", "falls", "fall", "rose", "rises", "rise", "began", "begins", "begin", "saw", "sees",
    "see",
}


def _finite_forms(base: str) -> set:
    forms = {base}
    if base.endswith("y") and base[-2:-1] not in "aeiou":
        forms |= {base[:-1] + "ies", base[:-1] + "ied"}
    elif base.endswith(("s", "sh", "ch", "x", "z")):
        forms |= {base + "es", base + "ed"}
    elif base.endswith("e"):
        forms |= {base + "s", base + "d"}
    else:
        forms |= {base + "s", base + "ed"}
        if re.fullmatch(r"[^aeiou]*[aeiou][bdgmnprt]", base):
            forms.add(base + base[-1] + "ed")   # stopped, mapped, logged
    return forms


_BASE_FORMS = set(_VERBS) | {
    "make", "lose", "find", "give", "tell", "know", "think", "bring", "fall", "rise", "begin", "see", "get",
    "go", "come", "stay", "remain", "become", "seem", "need", "do", "have",
}
_INFLECTED_FORMS = (set().union(*(_finite_forms(v) for v in _VERBS)) | _IRREGULAR) - _BASE_FORMS
_VERB_FORMS = _BASE_FORMS | _INFLECTED_FORMS | _AUXILIARIES

# Before one of these a verb-like word is a noun: "the cache for sessions",
# "our release process".
_DETERMINERS = {
    "the", "a", "an", "our", "your", "their", "its", "his", "her", "my", "this", "that", "these", "those",
    "each", "every", "any", "no", "some", "another", "which", "whose", "of", "for", "to", "in", "on", "at",
    "by", "with", "from", "per",
}
_SUBJECT_PRONOUNS = {"we", "you", "i", "they"}

# A finite verb further in than this is not the verb of a short subject.
_MAX_SUBJECT_WORDS = 8
_MIN_WORDS = 3
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*|\d[\d,.:%/-]*")


def _tokens(text: str) -> List[str]:
    return [w.replace("’", "'") for w in _WORD_RE.findall(text)]


def _words(text: str) -> List[str]:
    return [w.lower() for w in _tokens(text)]


# Words that introduce a subordinate or relative clause: a verb after one of
# them is that clause's verb, not the piece's ("the first week of January when
# the waiting lists are reopened").
_SUBORDINATORS = {
    "when", "where", "that", "which", "who", "whom", "whose", "if", "because", "while", "until", "before",
    "after", "as", "since", "unless", "whenever", "once", "so",
}
# Words ending in "s" that are not plural nouns.
_NOT_PLURAL = {
    "always", "sometimes", "perhaps", "thus", "us", "its", "his", "this", "is", "was", "has", "does", "plus",
    "across", "unless", "whereas", "as", "less", "yes", "besides", "towards", "afterwards", "ours", "yours",
}
# A reply, not a fact: "I think so", "I guess not".
_REPLY_RE = re.compile(r"^\W*(?:\w+\W+){0,3}(?:so|not)\W*$", re.IGNORECASE)
# An instruction: a rule and its reason or its next step stay together.
_IMPERATIVE_RE = re.compile(
    r"^\W*(?:(?:and|so|then|also|just|please)\W+)*(?:never|don'?t|do\s+not|always|avoid|please|make\s+sure|"
    r"ensure|remember|note)\b", re.IGNORECASE)


def _is_imperative(text: str) -> bool:
    if _IMPERATIVE_RE.match(text):
        return True
    tokens = _tokens(text)
    while tokens and tokens[0].lower() in ("and", "so", "then", "also", "just", "please"):
        tokens = tokens[1:]
    # "Check consumer lag ...", "Restart the pods ...": a base-form verb first.
    return bool(tokens) and tokens[0].lower() in _BASE_FORMS and (
        len(tokens) < 2 or tokens[1].lower() in _DETERMINERS | {"it", "them", "all", "every"})


def _is_finite_verb(tokens: List[str], i: int) -> bool:
    """Whether tokens[i] (i >= 1) reads as the finite verb of a clause."""
    word, prev = tokens[i].lower(), tokens[i - 1].lower()
    nxt = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
    # Capitalised inside a clause, it is part of a name ("Azure Cache for
    # Redis", "Service Bus"); after a determiner it is a noun ("the cache");
    # after "and"/"or" it is a second verb of a predicate that started earlier.
    if prev in _DETERMINERS or prev in ("and", "or") or tokens[i][:1].isupper():
        return False
    # A noun in a list: "inventory updates and shipment orchestration".
    if nxt in ("and", "or") and word not in _AUXILIARIES:
        return False
    # A participle: "three nodes managed by Patroni".
    if word.endswith("ed") and nxt == "by":
        return False
    if word in _AUXILIARIES or word in _INFLECTED_FORMS:
        return True
    # A base form needs a plural or named subject right before it ("Developers
    # own", "Prometheus and Grafana monitor", "we deploy") and something after
    # it ("the finance systems lead" is a noun).
    plural = prev.endswith("s") and not prev.endswith("ss") and prev not in _NOT_PLURAL
    return word in _BASE_FORMS and bool(nxt) and (prev in _SUBJECT_PRONOUNS or tokens[i - 1][:1].isupper() or plural)


def _has_finite_verb(text: str) -> bool:
    tokens = _tokens(text)
    return any(_is_finite_verb(tokens, i) for i in range(1, len(tokens)))


def is_statement(text: str, opens_sentence: bool = False, inflected_only: bool = False) -> bool:
    """True when `text` can stand alone as a statement: an explicit subject
    followed by a finite verb, and not leaning on what came before it.
    `opens_sentence`: the piece starts the spoken sentence, so a pronoun
    subject ("It processes 9,000 orders") is its own. `inflected_only`: a
    base-form verb is not enough (after a bare comma, "the invoicing platform
    our customers use" is an apposition)."""
    tokens = _tokens(text)
    if len(tokens) < _MIN_WORDS or _REPLY_RE.match(text):
        return False
    first = tokens[0].lower()
    if first in _ADVERBIAL_OPENERS:
        return False
    if not opens_sentence and (first in _LEANING_OPENERS or _EXPLAINS_RE.match(text.strip())):
        return False
    # A clause that points back to the one before ("..., and a second function
    # rolls them up into Azure Data Explorer") is the next step of the same
    # flow, not a fact of its own: cut off, "for the dashboards the fleet
    # managers use" filed a data pipeline step under Monitoring.
    if not opens_sentence and any(t.lower() in _BACK_REFERENCES for t in tokens):
        return False
    verb_at = next((i for i in range(1, min(len(tokens), _MAX_SUBJECT_WORDS + 1)) if _is_finite_verb(tokens, i)), None)
    if verb_at is None:
        return False
    verb = tokens[verb_at].lower()
    if inflected_only and verb not in _AUXILIARIES and verb not in _INFLECTED_FORMS:
        return False
    # A verb first is a bare predicate ("pushes them to ECR", "is shared by
    # all services") unless it is a noun subject with its own verb ("Alerts
    # go to Slack"); a participle first is a phrase ("Coming back to
    # architecture").
    if first in _AUXILIARIES:
        return False
    if first in _INFLECTED_FORMS | _BASE_FORMS and tokens[1].lower() in ("and", "or"):
        return False
    if first.endswith("ing") and verb not in _AUXILIARIES and verb not in _INFLECTED_FORMS:
        return False
    subject = [t.lower() for t in tokens[:verb_at]]
    if any(t in _SUBORDINATORS for t in subject[1:]):
        return False
    # A noun phrase with a prepositional tail ending in a verb-like plural is
    # a list item: "SQL connection exhaustion during sales peaks".
    if verb_at == len(tokens) - 1 and verb not in _AUXILIARIES and any(t in _ADVERBIAL_OPENERS for t in subject[1:]):
        return False
    return True


# "..., with demand increasing during weekends and dinner hours": a second
# fact in a participle phrase. The fact starts after "with". Only a verb's
# -ing form ("with three node groups" is a detail of the clause, not a fact).
_ABSOLUTE_WITH_RE = re.compile(r"^with\s+((?:[A-Za-z][\w-]*\s+){1,4}?)([a-z]+ing)\b")


def _ing_base(word: str) -> str:
    stem = word[:-3]
    for base in (stem, stem + "e", stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem):
        if base in _BASE_FORMS:
            return base
    return ""


def _absolute_phrase_start(text: str) -> int:
    """Offset in `text` where the fact of a "with <noun> <verb>-ing" phrase
    starts, or -1."""
    match = _ABSOLUTE_WITH_RE.match(text)
    if not match or not _ing_base(match.group(2).lower()):
        return -1
    if any(w.lower() in _AUXILIARIES for w in match.group(1).split()):
        return -1
    return match.start(1)


def split_into_facts(text: str) -> List[Tuple[int, int]]:
    """The facts `text` states, as (start, end) character spans of `text`.
    One span covering the sentence when it states one fact or its clauses
    lean on each other."""
    stripped = text.rstrip()
    whole = [(0, len(stripped))]
    if _SUBORDINATE_OPENER_RE.match(stripped):
        return whole
    boundaries = list(_BOUNDARY_RE.finditer(stripped))
    if not boundaries:
        return whole
    # (start, end, separator before it) for the text between boundaries.
    atoms = [(0, boundaries[0].start(), "")]
    for b, nxt in zip(boundaries, boundaries[1:] + [None]):
        atoms.append((b.end(), nxt.start() if nxt else len(stripped), b.group(0)))

    spans: List[Tuple[int, int]] = []
    cur_start, cur_end = atoms[0][0], atoms[0][1]
    for atom_start, atom_end, separator in atoms[1:]:
        left = stripped[cur_start:cur_end]
        right = stripped[atom_start:atom_end]
        left_states = is_statement(left, opens_sentence=(cur_start == 0))
        bare_comma = separator.strip() == ","
        if left_states and not _is_imperative(left) and is_statement(right, inflected_only=bare_comma):
            spans.append((cur_start, cur_end))
            cur_start, cur_end = atom_start, atom_end
            continue
        absolute = _absolute_phrase_start(right) if bare_comma and left_states and not _is_imperative(left) else -1
        if absolute >= 0:
            spans.append((cur_start, cur_end))
            cur_start, cur_end = atom_start + absolute, atom_end
            continue
        cur_end = atom_end
        # A statement followed by a phrase with no verb of its own ("We keep
        # two environments, staging and prod", "Questions go to Omar; for
        # anything with invoices, ...") is a list or a lead-in: what follows
        # belongs with it, so nothing after it is cut.
        right_tokens = _tokens(right)
        if left_states and right_tokens and right_tokens[0].lower() not in _VERB_FORMS \
                and not _has_finite_verb(right):
            cur_end = len(stripped)
            break
    spans.append((cur_start, cur_end))
    if len(spans) == 1:
        return whole
    # A cut that leaves a piece too short to be a fact is not a cut.
    if any(len(_words(stripped[a:b])) < _MIN_WORDS for a, b in spans):
        return whole
    return spans


def fact_texts(text: str) -> List[str]:
    """split_into_facts() as text slices."""
    return [text[a:b] for a, b in split_into_facts(text)]


