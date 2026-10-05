import re
from typing import Dict, Any, List, Tuple
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.ownership import build_block as build_ownership_table
from renderers.blocks.common import no_coverage_block
from conflicts import CORRECTION_RE


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


# "<the X team | developers | engineers> owns/own/is responsible for <what>."
_OWNS_RE = re.compile(
    r"(?:^|(?<=[.;]\s))(?P<who>(?:the\s+)?(?:[A-Za-z][A-Za-z/&\- ]{0,50}?\s)?"
    r"(?:team|teams|developers?|engineers?|group|squad|owners?))\s+"
    r"(?:owns?|is\s+responsible\s+for|are\s+responsible\s+for)\s+(?P<what>[^.;]{3,200})",
    re.IGNORECASE,
)


# One sentence often states two owners ("The mobile team owns the app, and
# our squad owns the backend"); matched whole, the second clause became part
# of the first owner's "what". Each clause is matched on its own.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(r";\s*|,\s*(?:and|while|but|whereas)\s+|\s+(?:while|whereas)\s+", re.IGNORECASE)
_LEAD_IN_RE = re.compile(r"^(?:actually|so|and|but|now|also|currently|basically|then|today)\b[\s,]*", re.IGNORECASE)
_WE_OWN_RE = re.compile(r"^(?:we|our team)\s+(?:own|are\s+responsible\s+for)\s+(?P<what>[^.;?]{3,200})", re.IGNORECASE)
_PRONOUN_OBJECT_RE = re.compile(r"^(?:it|this|that|them|everything)\b[\s,]*", re.IGNORECASE)


def _clean_what(what: str) -> str:
    what = what.strip().rstrip(".")
    # "owns it since the reorg": the pronoun stands for the system itself.
    rest = _PRONOUN_OBJECT_RE.sub("", what)
    if rest != what:
        what = "The service" + (f" ({rest})" if rest else "")
    return what[:1].upper() + what[1:]


def _explicit_ownership_statements(section: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for text in _coverage_paragraphs(section):
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            if sentence.strip().endswith("?"):
                continue  # a question names no owner
            for clause in _CLAUSE_SPLIT_RE.split(sentence):
                clause = _LEAD_IN_RE.sub("", clause.strip())
                we = _WE_OWN_RE.match(clause)
                if we:
                    out.append(("Our team (outgoing owner)", _clean_what(we.group("what"))))
                    continue
                match = _OWNS_RE.match(clause)
                if not match:
                    continue
                who = re.sub(r"^the\s+", "", match.group("who").strip(), flags=re.IGNORECASE)
                what = match.group("what")
                if who and what:
                    out.append((who[:1].upper() + who[1:], _clean_what(what)))
    return out


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Ownership & Escalation")
    fields = section.get("fields", {})

    rows: List[Dict[str, str]] = []
    paragraphs: List[str] = []

    if fields.get("oncall_tool", {}).get("value"):
        rows.append({"role": "On-call tool", "team": fields["oncall_tool"]["value"]})
    if fields.get("escalation_chain", {}).get("value"):
        paragraphs.append(f"Escalation chain: {fields['escalation_chain']['value']}")
    if fields.get("application_ownership", {}).get("value"):
        rows.append({"role": "Application ownership", "team": fields["application_ownership"]["value"]})
    if fields.get("infrastructure_ownership", {}).get("value"):
        rows.append({"role": "Infrastructure ownership", "team": fields["infrastructure_ownership"]["value"]})

    # Explicit "<who> owns <what>" statements, read straight from the
    # section's own sentences. Without an LLM the two fields above are never
    # filled, and even with one they only hold two coarse buckets -- a
    # transcript naming five owning teams rendered a table with a single
    # "On-call tool" row and the ownership matrix itself nowhere.
    known_teams = {str(r.get("team", "")).strip().lower() for r in rows}
    by_what: Dict[str, Dict[str, str]] = {}
    for who, what in _explicit_ownership_statements(section):
        if who.lower() in known_teams:
            continue
        known_teams.add(who.lower())
        # Two owners for the same thing ("The payments team owns the
        # service. Actually the platform team owns it since the reorg."): a
        # stated correction wins and the earlier owner is noted; otherwise
        # both are shown as a conflict to confirm (conflicts.py).
        key = re.sub(r"\s*\(.*\)\s*$", "", what).strip().lower()
        earlier = by_what.get(key)
        if earlier is not None:
            base = re.sub(r"\s*\(.*\)\s*$", "", what).strip()
            if CORRECTION_RE.search(what):
                earlier.update(role=base, team=f"{who} (previously {earlier['team']})")
            else:
                earlier.update(role=base, team=f"Conflict: {earlier['team']} or {who} — confirm with the outgoing owner")
            continue
        row = {"role": what, "team": who}
        by_what[key] = row
        rows.append(row)

    # Kept as its own paragraph, deliberately never folded into the
    # ownership table above: a general "contact X when unsure" instruction
    # is operational guidance about what to do, not a statement of who
    # formally owns the system (or the on-call tool's name) — conflating
    # them was a real bug (a real transcript's "If you are unsure about a
    # change, involve the appropriate platform or application owner."
    # ended up mislabeled as the On-call tool's NAME in a live PDF).
    guidance_paragraphs: List[str] = []
    if fields.get("operational_escalation_guidance", {}).get("value"):
        guidance_paragraphs.append(fields["operational_escalation_guidance"]["value"])

    blocks = []
    if paragraphs:
        blocks.append(build_narrative_block(title, paragraphs))
    if rows:
        blocks.append(build_ownership_table(title, rows))
    if guidance_paragraphs:
        blocks.append(build_narrative_block("Operational escalation guidance", guidance_paragraphs))
    if not blocks:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block(title, fallback))
        else:
            blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
