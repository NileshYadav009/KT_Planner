import re
from typing import Dict, Any, List
from renderers.blocks.table import build_block as build_decision_table
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.common import no_coverage_block

# Matches kt_schema_new.json's declared columns for the common_failures table
# section. The legacy pipe-parsed fallback below only ever produces 3 of these
# (Symptom/Cause/Fix) since that format never carried frequency/ticket data.
# "When it happens" carries the TRIGGERING CONDITION ("during high traffic",
# "after a deployment", "during market-open bursts"). Speakers state it in the
# same breath as the failure itself, and without a column for it the condition
# was simply dropped: "Azure SQL connection exhaustion during high traffic"
# rendered as "Azure SQL connection exhaustion" alone, losing the single most
# useful thing for recognising the failure before it escalates. Keeping
# issue -> condition -> cause -> first checks -> fix intact is the point of
# this table.
STRUCTURED_COLUMNS = [
    "Issue/Symptom", "When it happens", "Likely Cause", "First Checks",
    "How to Fix", "Frequency", "KEDB / Ticket Link",
]

HISTORICAL_COLUMNS = ["Incident", "Cause", "When", "Impact", "Resolution", "Preventive action"]
# A detail the KT did not give is left empty: every view marks an empty cell
# itself (the PDF with a dash and one note under the table).
NOT_COVERED = ""


def _structured_rows(section: Dict[str, Any]) -> List[Dict[str, str]]:
    failures = (section.get("_structured") or {}).get("failures")
    if not isinstance(failures, list):
        return []
    rows = []
    for item in failures:
        if not isinstance(item, dict):
            continue
        symptom = str(item.get("symptom") or "").strip()
        if not symptom:
            continue
        first_checks = item.get("first_checks")
        if isinstance(first_checks, list):
            first_checks = "; ".join(str(c).strip() for c in first_checks if str(c).strip())
        rows.append({
            "Issue/Symptom": symptom,
            "When it happens": str(item.get("condition") or "").strip(),
            "Likely Cause": str(item.get("cause") or "").strip(),
            # Diagnostic steps are a distinct fact from a remediation — a
            # transcript very often states what to check without stating a
            # fix, and that used to be dropped entirely for want of a column
            # to put it in (the schema-gap class of bug).
            "First Checks": str(first_checks or "").strip(),
            "How to Fix": str(item.get("fix") or "").strip(),
            "Frequency": str(item.get("frequency") or "").strip(),
            "KEDB / Ticket Link": str(item.get("ticket") or "").strip(),
        })
    return rows


def _historical_rows(section: Dict[str, Any]) -> List[Dict[str, str]]:
    """Entries carrying a "when" (a specific past occurrence, e.g. "last
    year") describe a one-off historical incident rather than a recurring
    issue — surface them as their own record instead of leaving that detail
    stranded in the shared Frequency column. Never infer a missing
    resolution/preventive action — an empty cell beats a fabricated fix."""
    failures = (section.get("_structured") or {}).get("failures")
    if not isinstance(failures, list):
        return []
    rows = []
    for item in failures:
        if not isinstance(item, dict):
            continue
        when = str(item.get("when") or "").strip()
        symptom = str(item.get("symptom") or "").strip()
        if not when or not symptom:
            continue
        rows.append({
            "Incident": symptom,
            "Cause": str(item.get("cause") or "").strip() or NOT_COVERED,
            "When": when,
            "Impact": str(item.get("impact") or "").strip() or NOT_COVERED,
            "Resolution": str(item.get("resolution") or "").strip() or NOT_COVERED,
            "Preventive action": str(item.get("preventive_action") or "").strip() or NOT_COVERED,
        })
    return rows


def _coverage_rows(section: Dict[str, Any]) -> List[Dict[str, str]]:
    failures = section.get("coverage_content") or []
    if isinstance(failures, str):
        failures = [failures]

    # Only genuinely structured "symptom | cause | fix" lines become table
    # rows. A plain sentence used to become a "Symptom" row with blank Cause
    # and Fix cells, which presented narrative ("There was a previous
    # incident where ...", "The corrective action was to ...") as if it were
    # a structured failure record with an unknown cause and no fix -- the
    # fix was often stated in the very next sentence. Plain sentences render
    # as narrative instead (see render()).
    rows = []
    for item in failures:
        text = str(item).strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split("|") if part.strip()]
        if len(parts) >= 3:
            rows.append({"Symptom": parts[0], "Cause": parts[1], "Fix": parts[2]})
        elif len(parts) == 2:
            rows.append({"Symptom": parts[0], "Cause": parts[1], "Fix": ""})
    return rows


# One record per failure (the client's review, 13 Nov): what it is, when it
# happens, the first checks as said, the fix or "Not stated in the KT", and
# where the incident contact is. The Fix column is kept when no failure has a
# fix: an empty column was dropped, and the table no longer showed that none
# was stated.
RECORD_COLUMNS = STRUCTURED_COLUMNS
RECORDS_TITLE = "Failure symptoms and remediation"
FIX_COLUMN = "How to Fix"
# A record that says who to call ("if it doesn't come back, call the network team").
_ESCALATES_RE = re.compile(r"\b(?:escalat\w*|call|page|contact|ring|notify|raise\s+(?:it\s+)?with)\b", re.IGNORECASE)
TRIAGE_TITLE = "First response to any alert"


def _record_blocks(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    from knowledge.knowledge_builder import failure_records

    found = failure_records(section)
    blocks: List[Dict[str, Any]] = []
    if found["triage"]:
        blocks.append(build_narrative_block(TRIAGE_TITLE, found["triage"]))
    records = found["records"]
    if not records:
        return blocks
    rows = [{"Issue/Symptom": r["failure"], "When it happens": r["when"], "Likely Cause": r["cause"],
             "First Checks": r["first_checks"], FIX_COLUMN: r["fix"], "Frequency": r["frequency"],
             "KEDB / Ticket Link": r["ticket"]} for r in records]
    table = build_decision_table(RECORDS_TITLE, RECORD_COLUMNS, rows)
    if FIX_COLUMN not in table["columns"]:
        table["columns"].insert(min(3, len(table["columns"])), FIX_COLUMN)
    blocks.append(table)
    contact = section.get("_incident_contact_in")
    own_escalation = any(_ESCALATES_RE.search(f"{r['first_checks']} {r['fix']}") for r in records)
    if "_incident_contact_in" in section and not own_escalation:
        note = build_narrative_block(section.get("title", "Common Failures"), [      # no heading of its own
            f"No escalation was stated for a particular failure; who to contact in an incident is under {contact}."
            if contact else "No escalation was stated in the KT, for these failures or any incident."])
        note["document_note"] = True
        blocks.append(note)
    return blocks


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Common Failures")
    blocks = _record_blocks(section)
    if any(b.get("title") == RECORDS_TITLE for b in blocks):
        historical_rows = _historical_rows(section)
        if historical_rows:
            blocks.append(build_decision_table("Historical incident record", HISTORICAL_COLUMNS, historical_rows))
        return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
    blocks = []
    structured_rows = _structured_rows(section)

    if structured_rows:
        blocks.append(build_decision_table("Failure symptoms and remediation", STRUCTURED_COLUMNS, structured_rows))
        historical_rows = _historical_rows(section)
        if historical_rows:
            blocks.append(build_decision_table("Historical incident record", HISTORICAL_COLUMNS, historical_rows))
        return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}

    rows = _coverage_rows(section)

    if rows:
        blocks.append(build_decision_table("Failure symptoms and remediation", ["Symptom", "Cause", "Fix"], rows))
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    paragraphs = [
        str(item).strip() for item in content
        if str(item).strip() and "|" not in str(item)
    ]
    if paragraphs:
        blocks.append(build_narrative_block("Failure notes from the KT session" if rows else title, paragraphs))

    if not blocks:
        blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
