import re
from typing import Dict, Any, List, Tuple
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.ownership import build_block as build_ownership_table
from renderers.blocks.common import no_coverage_block


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


def _explicit_ownership_statements(section: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for text in _coverage_paragraphs(section):
        for match in _OWNS_RE.finditer(text):
            who = re.sub(r"^the\s+", "", match.group("who").strip(), flags=re.IGNORECASE)
            what = match.group("what").strip().rstrip(".")
            if who and what:
                out.append((who[:1].upper() + who[1:], what[:1].upper() + what[1:]))
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
    for who, what in _explicit_ownership_statements(section):
        if who.lower() in known_teams:
            continue
        known_teams.add(who.lower())
        rows.append({"role": what, "team": who})

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
