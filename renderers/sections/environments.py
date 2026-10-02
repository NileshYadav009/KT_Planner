from typing import Any, Dict, List
from renderers.blocks.table import build_block as build_decision_table
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.common import (
    no_coverage_block,
    coverage_paragraphs as shared_coverage_paragraphs,
)

COLUMNS = ["Environment", "Known characteristics"]

import re

# (label, evidence pattern) for the environment details a reader might be
# tempted to infer. The note used to claim ALL of these were "not covered"
# unconditionally -- including on a KT that listed every environment by name.
# Now it names only the details the section's own content never mentions.
_DETAIL_EVIDENCE = [
    ("exact environment names", re.compile(r"\b(?:dev(?:elopment)?|qa|test|integration|uat|staging|pre-?prod(?:uction)?|production|sandbox)\b", re.I)),
    ("URLs", re.compile(r"https?://|\bwww\.|\.(?:com|net|io|internal)\b", re.I)),
    ("cloud accounts", re.compile(r"\baccount(?:s|\s+ids?)?\b|\bsubscription\b|\bproject\s+id\b", re.I)),
    ("regions", re.compile(r"\bregions?\b|\b(?:us|eu|ap|sa|ca|me|af)-[a-z]+-\d\b", re.I)),
    ("access procedures", re.compile(r"\baccess\b", re.I)),
    ("namespaces", re.compile(r"\bnamespaces?\b", re.I)),
]


def _not_provided_note(text: str):
    missing = []
    for label, pattern in _DETAIL_EVIDENCE:
        found = {m.group(0).lower() for m in pattern.finditer(text or "")}
        # One passing mention of "production" is not a list of environment
        # names; two or more distinct names are.
        needed = 2 if label == "exact environment names" else 1
        if len(found) < needed:
            missing.append(label)
    if not missing:
        return None
    if len(missing) == 1:
        listed = missing[0]
    else:
        listed = ", ".join(missing[:-1]) + " and " + missing[-1]
    return f"{listed[:1].upper() + listed[1:]} were not covered — do not infer them."


def _field_value(fields: Dict[str, Any], field_id: str):
    entry = fields.get(field_id)
    if isinstance(entry, dict):
        return entry.get("value")
    return None


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    # Shared implementation: splits polish-pass bullet blobs and drops
    # repeats. See renderers/blocks/common.coverage_paragraphs().
    return shared_coverage_paragraphs(section)


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Environments")
    fields = section.get("fields", {})

    rows = []
    for label, field_id in (
        ("Production", "production_notes"),
        ("Staging", "staging_notes"),
        ("Non-production", "non_production_notes"),
    ):
        value = _field_value(fields, field_id)
        if isinstance(value, str) and value.strip():
            rows.append({"Environment": label, "Known characteristics": value.strip()})

    blocks = []
    if rows:
        blocks.append(build_decision_table("Environment knowledge", COLUMNS, rows))
        known_differences = _field_value(fields, "known_differences")
        if isinstance(known_differences, str) and known_differences.strip():
            blocks.append(build_narrative_block("Known differences / limitations", [known_differences.strip()]))

        # The table has one row per schema-declared environment
        # (Production/Staging/Non-production), so a transcript naming any
        # OTHER environment has nowhere to put it — "The platform has
        # development, QA, staging, and production environments." was
        # dropped outright on a live run, losing the existence of the dev
        # and QA environments entirely. Anything captured for this section
        # that isn't already shown in a row still surfaces here.
        shown = {row["Known characteristics"] for row in rows}
        leftover = [p for p in _coverage_paragraphs(section) if p not in shown]
        if leftover:
            blocks.append(build_narrative_block("Additional environment notes", leftover))

        note = _not_provided_note(" ".join(_coverage_paragraphs(section)))
        if note:
            blocks.append(build_narrative_block("Do not over-infer", [note]))
        return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}

    fallback = _coverage_paragraphs(section)
    if fallback:
        blocks.append(build_narrative_block(title, fallback))
        note = _not_provided_note(" ".join(fallback))
        if note:
            blocks.append(build_narrative_block("Do not over-infer", [note]))
    else:
        blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
