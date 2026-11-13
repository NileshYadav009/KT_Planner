import re
from typing import Any, Dict
from renderers.blocks.table import build_block as build_decision_table
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.checklist import build_block as build_checklist_block
from renderers.blocks.common import no_coverage_block

COLUMNS = ["Domain", "Coverage", "Assessment"]

COMPLETENESS_INVARIANT = (
    "Every substantive fact must end in exactly one state: mapped, "
    "intentionally deduplicated, or explicitly surfaced as unmapped with a "
    "reason. Silent loss is unacceptable."
)

KNOWLEDGE_GAPS_TITLE = (
    "Follow up with the outgoing owner: topics not discussed, stated gaps "
    "and possible conflicts"
)
MATRIX_NOTE = "What is missing from each section that is partly or not covered is listed in the follow-up list below."

# The matrix shows each section's status; what is missing is in the follow-up
# list below it, which repeated the matrix's own wording line for line (the PDF
# clean-up, 12 Nov). The rows keep the full assessment (_coverage_rows: the web
# page, the scorers); the page shows it up to where these details start.
_DETAILS_RE = re.compile(r"; (?:not discussed|stated as not documented|stated as not done|no fix stated for)\b"
                         r"| (?:Stated gap|To confirm): \u201c|: \u201c")
_NO_FIX_RE = re.compile(r"; no fix stated for\b")


def short_assessment(row: Dict[str, Any]) -> str:
    """A coverage row's status in a few words: "4 of 5 topics covered.",
    "All 2 topics covered; not every failure has a fix.", "Optional." """
    text = str(row.get("Assessment") or "").strip()
    if text.startswith("Optional;"):
        return "Optional."
    match = _DETAILS_RE.search(text)
    short = (text[:match.start()] if match else text).rstrip(" .")
    if _NO_FIX_RE.search(text):
        short += "; not every failure has a fix"
    return f"{short}." if short else text


def knowledge_coverage_line(summary: Dict[str, Any]) -> str:
    """Where the transcript's facts are, in one line: they used to take a
    paragraph of counts (checked, in a section, under Additional Notes,
    gaps, recovered, left out)."""
    facts = int(summary.get("facts_identified", 0) or 0)
    mapped, unmapped = int(summary.get("mapped", 0) or 0), int(summary.get("unmapped", 0) or 0)
    shown = mapped + unmapped
    text = (f"All {facts} facts from the transcript are in this document" if shown >= facts
            else f"{shown} of {facts} facts from the transcript are in this document")
    if unmapped:
        text += f", {unmapped} of them under Additional Notes"
        if summary.get("recovered"):
            text += f" ({summary['recovered']} placed there by the final completeness check)"
    if summary.get("stated_gaps"):
        n = int(summary["stated_gaps"])
        text += f"; {n} {'says' if n == 1 else 'say'} something is missing or undecided (in the follow-up list)"
    if summary.get("excluded"):
        n = int(summary["excluded"])
        text += f"; {n} {'remark' if n == 1 else 'remarks'} with no content {'was' if n == 1 else 'were'} left out"
    return text + "."


COVERAGE_TITLE = "Coverage by section"
GROUPED_COLUMNS = ["Coverage", "Sections"]
STATUS_ORDER = ("Strong", "Partial", "Missing", "Not discussed")


def _status_words(short: str) -> str:
    """"4 of 5 topics covered (1 under Architecture Reference)." -> "4 of 5
    topics covered" for the grouped table; where is in the follow-up list."""
    return re.sub(r"\s*\([^)]*\)", "", short.strip()).strip().rstrip(".")


def merge_by_section(gaps, titles):
    """One follow-up line per section: "Disaster Recovery: Failover /
    standby" and "Disaster Recovery: “... the current RTO and RPO need
    confirmation.”" read as one line, the section named once. Lines that
    name no section (conflicts, unanswered questions) stay as they are."""
    merged, where = [], {}
    for line in gaps:
        title = next((t for t in titles if t and str(line).startswith(t + ": ")), None)
        if title is None:
            merged.append(line)
            continue
        rest = str(line)[len(title) + 2:]
        if title in where:
            merged[where[title]] += "; " + rest
        else:
            where[title] = len(merged)
            merged.append(line)
    return merged


MEASURES_TITLE = "How complete this handover is"
MEASURE_COLUMNS = ["Measure", "Result"]
READINESS_TITLE = "Operational readiness"
READINESS_COLUMNS = ["To be able to", "Readiness", "What is missing"]


def placement_line(summary: Dict[str, Any]) -> str:
    """Where the facts are shown and how their sections were decided. Not an
    accuracy figure: no answer key is checked here."""
    facts = int(summary.get("facts_identified", 0) or 0)
    own = int(summary.get("shown_in_own_section", 0) or 0)
    rule, llm = int(summary.get("placed_by_rule", 0) or 0), int(summary.get("placed_by_llm", 0) or 0)
    text = (f"{own} of {facts} facts are in the section they were filed under: {rule} filed by a routing rule, "
            f"{max(facts - rule - llm, 0)} by the classifier's closest match")
    if llm:
        text += f", {llm} after the LLM's check of a close call"
    return text + " (each one's basis is in the facts ledger)."


def template_line(rows) -> str:
    shown = [r for r in rows if r.get("Coverage") in ("Strong", "Partial", "Missing")]
    counts = {c: sum(1 for r in shown if r.get("Coverage") == c) for c in ("Strong", "Partial", "Missing")}
    return (f"{counts['Strong']} of {len(shown)} sections covered well, {counts['Partial']} partly, "
            f"{counts['Missing']} not covered (coverage matrix below).")


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "KT Coverage & Knowledge Gaps")
    rows = section.get("_coverage_rows") or []
    if not rows:
        return {"section_id": section.get("id"), "section_title": title, "blocks": [no_coverage_block(title)]}

    blocks = []

    # Four measures, apart (the client's review, 13 Nov): every fact kept is
    # not every section covered, and neither is a system the new owner can
    # run. Facts kept, where they are, the template's sections, and the
    # operational tasks the document describes.
    summary = section.get("_knowledge_coverage_summary") or {}
    readiness = section.get("_readiness") or {}
    if summary.get("verified_against_document"):
        # Counts measured against the rendered document itself (see
        # knowledge_builder.verify_document_coverage), not the pre-render
        # knowledge object.
        # Counted per fact: a sentence that states several facts was cut
        # into them before mapping (fact_units.py).
        measures = [{"Measure": "Fact retention", "Result": knowledge_coverage_line(summary)},
                    {"Measure": "Placement", "Result": placement_line(summary)}]
        if readiness.get("summary"):
            measures.append({"Measure": "Operational readiness", "Result": readiness["summary"] + (
                " Every fact kept is not a system the new owner can run: see below."
                if readiness.get("verdict") != "Ready" else "")})
        block = build_decision_table(MEASURES_TITLE, MEASURE_COLUMNS, measures)
        block["document_note"] = True
        blocks.append(block)
        if readiness.get("checks"):
            table = build_decision_table(READINESS_TITLE, READINESS_COLUMNS, [
                {"To be able to": c["capability"], "Readiness": c["status"], "What is missing": c["detail"]}
                for c in readiness["checks"]])
            table["document_note"] = True
            blocks.append(table)
    elif summary:
        blocks.append(build_narrative_block("Knowledge coverage", [
            f"{summary.get('facts_identified', 0)} fact-bearing sentence(s) identified in this KT session: "
            f"{summary.get('mapped', 0)} mapped to a section, "
            f"{summary.get('deduplicated', 0)} deduplicated (already captured elsewhere), "
            f"{summary.get('unmapped', 0)} surfaced as unmapped findings, "
            f"{summary.get('lost', 0)} lost."
        ]))

    # Sign-off is recorded at review, never a gap of the session: its row
    # said so on every KT.
    shown = [{"Domain": r.get("Domain"), "Coverage": r.get("Coverage"), "Assessment": short_assessment(r)}
             for r in rows if r.get("Coverage") != "After review"]
    # One row per status, each listing its sections (the client's review, 13
    # Nov: a row per section ran over a page beside the readiness table and
    # the follow-up list). The rows themselves are unchanged (_coverage_rows:
    # the web page and the scorers read them).
    grouped = []
    for status in STATUS_ORDER:
        members = [r for r in shown if r["Coverage"] == status]
        if members:
            # One section a line: run together, one section's status read
            # into the next one's name.
            grouped.append({"Coverage": status, "Sections": "\n".join(
                f"{r['Domain']} ({_status_words(r['Assessment'])})"
                if status == "Partial" or (status == "Missing" and not r["Assessment"].startswith("Not covered"))
                else r["Domain"] for r in members)})
    block = build_decision_table(COVERAGE_TITLE, GROUPED_COLUMNS, grouped)
    block["document_note"] = True
    blocks.append(block)
    # The completeness invariant is an engineering rule (see
    # knowledge_builder.verify_document_coverage), not reader content.

    # Kept visually/structurally separate from open_responsibilities' Open
    # Tasks table: a knowledge gap is "the KT session never covered this,"
    # not "someone agreed to do this" — conflating the two would misrepresent
    # an absence of information as an assigned responsibility.
    gaps = merge_by_section(section.get("_knowledge_gaps") or [], [str(r.get("Domain") or "") for r in rows])
    if gaps:
        blocks.append(build_checklist_block(KNOWLEDGE_GAPS_TITLE, gaps))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
