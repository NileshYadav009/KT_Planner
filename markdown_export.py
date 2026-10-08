"""The KT document outside the PDF (P2-2).

Teams keep runbooks in Confluence or Notion and track gaps in Jira. This
module gives them:

    job_markdown()  the document as Markdown, with the same sections, sign-off
                    and numbered sources as the PDF; it pastes into
                    Confluence, Notion, GitHub or GitLab
    gaps_csv()      the KT's knowledge gaps as a CSV that Jira (and most
                    trackers) import as one ticket per row

Both read the same document the PDF is built from, so a correction or a
sign-off shows up in every format.
"""
from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from knowledge.evidence import format_time
from renderers.blocks.common import NOT_COVERED_MESSAGE

_NOT_COVERED_CELL = "Not covered"


def _cell(value: Any) -> str:
    """A table cell: one line, pipes escaped."""
    text = re.sub(r"\s+", " ", str(value if value is not None else "")).strip()
    return text.replace("|", "\\|")


def _line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()


def _refs(block: Dict[str, Any], i: int) -> str:
    refs = block.get("sources") or []
    ids = refs[i] if i < len(refs) and isinstance(refs[i], list) else []
    return (" " + "".join(f"[{int(n)}]" for n in ids)) if ids else ""


def _table(headers: List[str], rows: Iterable[List[str]]) -> List[str]:
    out = ["| " + " | ".join(_cell(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(rows_cell for rows_cell in row) + " |" for row in rows]
    return out


def _block(block: Dict[str, Any], section_title: str) -> List[str]:
    kind = block.get("type")
    out: List[str] = []
    title = _line(block.get("title"))
    if title and title.lower() != section_title.lower() and kind not in ("WarningBlock",):
        out.append(f"### {title}")
        out.append("")
    if kind == "NarrativeBlock":
        for i, p in enumerate(block.get("paragraphs") or []):
            if _line(p):
                out += [_line(p) + _refs(block, i), ""]
    elif kind == "ChecklistBlock":
        out += [f"- {_line(item)}{_refs(block, i)}" for i, item in enumerate(block.get("items") or []) if _line(item)]
        out.append("")
    elif kind == "TroubleshootingBlock":
        steps = [(i, s) for i, s in enumerate(block.get("steps") or []) if _line(s)]
        out += [f"{n}. {_line(s)}{_refs(block, i)}" for n, (i, s) in enumerate(steps, start=1)]
        out.append("")
    elif kind == "WarningBlock":
        out += [f"> **Warning:** {_line(w)}{_refs(block, i)}" for i, w in enumerate(block.get("warnings") or []) if _line(w)]
        out.append("")
    elif kind == "TechnologyGrid":
        out += _table(["Item", "Value"], ([_cell(r.get("label")), _cell(r.get("value")) + _refs(block, i)]
                                          for i, r in enumerate(block.get("rows") or [])))
        out.append("")
    elif kind == "OwnershipTable":
        out += _table(["Responsibility", "Owner"], ([_cell(r.get("role")), _cell(r.get("team")) + _refs(block, i)]
                                                   for i, r in enumerate(block.get("rows") or [])))
        out.append("")
    elif kind == "DeploymentTimeline":
        out += [f"{n}. **{_line(e.get('label'))}**: {_line(e.get('description'))}{_refs(block, i)}"
                for n, (i, e) in enumerate(enumerate(block.get("entries") or []), start=1)]
        out.append("")
    elif kind == "DecisionTable":
        columns = list(block.get("columns") or [])
        rows = []
        for i, row in enumerate(block.get("rows") or []):
            cells = [_cell(row.get(c)) or _NOT_COVERED_CELL for c in columns]
            if cells:
                cells[-1] += _refs(block, i)
            rows.append(cells)
        if columns:
            out += _table(columns, rows)
            out.append("")
    elif kind == "CodeBlock":
        out += ["```", str(block.get("code") or "").rstrip(), "```", ""]
    elif kind == "DiagramBlock":
        caption = _line(block.get("caption"))
        out += [f"*Architecture diagram: see the PDF export.{(' ' + caption) if caption else ''}*", ""]
    elif kind == "ImageBlock":
        out += [f"*Screenshot: {_line(block.get('caption')) or 'see the PDF export'}.*", ""]
    elif _line(block.get("description")):
        out += [_line(block.get("description")), ""]
    return out


def job_markdown(job_id: str, job: Dict[str, Any], created: Optional[float] = None) -> str:
    """The KT document of a finished job as Markdown (sign-off applied)."""
    import signoff

    ko = job.get("knowledge_object") or {}
    title = _line(ko.get("system_name") or job.get("title") or "KT Document")
    date_str = (datetime.fromtimestamp(created) if created else datetime.now()).strftime("%d %B %Y")
    sections = signoff.apply_to_document(ko.get("rendered_sections") or [], job, int(job.get("document_version") or 1))

    out = [f"# {title}", "", f"Knowledge Transfer & Handover · {date_str} · Job {job_id[:8].upper()}", ""]
    for warning in job.get("warnings") or []:
        out.append(f"> **Review before relying on this document:** {_line(warning)}")
    if job.get("warnings"):
        out.append("")
    if job.get("notices"):
        out += [" ".join(_line(n) for n in job["notices"]), ""]
    for number, section in enumerate(sections, start=1):
        section_title = _line(section.get("section_title") or section.get("section_id") or f"Section {number}")
        out += [f"## {number}. {section_title}", ""]
        for block in section.get("blocks") or []:
            out += _block(block, section_title)
    sources = ko.get("sources") or []
    if sources:
        out += ["## Sources", "", "Each fact above is followed by the number of the transcript sentence it comes "
                "from. Values marked inferred were not stated in the session and have no source.", ""]
        for src in sources:
            when = format_time(src.get("start"))
            if src.get("session"):
                when = f"S{int(src['session'])}" + (f" · {when}" if when else "")
            who = _line(src.get("speaker"))
            lead = " ".join(part for part in (when, who + ":" if who else "") if part)
            out.append(f"{int(src['id'])}. {lead + ' ' if lead else ''}“{_line(src.get('quote'))}”")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def knowledge_gaps(job: Dict[str, Any]) -> List[str]:
    ko = job.get("knowledge_object") or {}
    for section in ko.get("sections") or []:
        if section.get("id") == "kt_coverage":
            return [_line(g) for g in section.get("_knowledge_gaps") or [] if _line(g) and _line(g) != NOT_COVERED_MESSAGE]
    return []


def gaps_csv(job_id: str, job: Dict[str, Any]) -> str:
    """One row per knowledge gap, in the columns Jira's CSV import maps by
    default (Summary, Description, Issue Type, Labels)."""
    ko = job.get("knowledge_object") or {}
    system = _line(ko.get("system_name") or job.get("title") or "KT")
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["Summary", "Description", "Issue Type", "Labels"])
    for gap in knowledge_gaps(job):
        summary = f"KT gap ({system}): {gap}"
        if len(summary) > 250:
            summary = summary[:249].rsplit(" ", 1)[0] + "…"
        description = (f"The knowledge transfer for {system} (job {job_id[:8].upper()}) did not cover this: {gap}\n\n"
                       f"Close it with the outgoing owner or in a follow-up KT session.")
        writer.writerow([summary, description, "Task", "kt-gap"])
    return buf.getvalue()
