import re
from typing import Any, Dict, List, Optional

from .entities import build_entities
from .evidence import build_evidence
from .facts import build_facts
from .relationships import build_relationships
from section_rules import is_tribal_knowledge

UNMAPPED_FINDINGS_SECTION_ID = "unmapped_findings"
UNMAPPED_FINDINGS_TITLE = "Additional Notes (Unmapped Findings)"
# A sentence below this many words is almost always a filler/transition
# ("Thank you.", "Okay so.") rather than a standalone fact worth surfacing.
# Generic length threshold — not keyed to any transcript's content.
_MIN_UNMAPPED_SENTENCE_WORDS = 4


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _collect_evidence(
    section_id: str,
    section_content: Dict[str, Any],
    populated_field: Dict[str, Any],
) -> List[Dict[str, Any]]:
    sentence_entries = section_content.get("sentences", []) if isinstance(section_content, dict) else []
    if not sentence_entries and isinstance(section_content, dict):
        # Same fallback field_populator.py's populate_fields() uses when
        # computing source_chunk_index — the two independent mechanisms that
        # populate section_content[id]['sentences'] vs. ['blocks'] can
        # disagree, leaving 'sentences' empty for a section that has real
        # coverage. Must use the identical list here so an index computed
        # there still refers to the right entry here.
        sentence_entries = [
            s for block in (section_content.get("blocks") or [])
            for s in (block.get("sentences") or [])
        ]

    if populated_field and populated_field.get("source_chunk_index") is not None:
        index = populated_field["source_chunk_index"]
        if 0 <= index < len(sentence_entries):
            sentence = sentence_entries[index]
            return [
                {
                    "sentence_index": index,
                    "text": _normalize_text(sentence.get("text", "")),
                    "start": sentence.get("start"),
                    "end": sentence.get("end"),
                    "speaker": sentence.get("speaker"),
                    "audio_confidence": sentence.get("audio_confidence"),
                }
            ]

    return [
        {
            "sentence_index": idx,
            "text": _normalize_text(sentence.get("text", "")),
            "start": sentence.get("start"),
            "end": sentence.get("end"),
            "speaker": sentence.get("speaker"),
            "audio_confidence": sentence.get("audio_confidence"),
        }
        for idx, sentence in enumerate(sentence_entries[:3])
    ]


def _infer_system_name(
    coverage: Dict[str, Any],
    populated_fields: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    if populated_fields:
        sys_field = (populated_fields.get("system_overview", {}) or {}).get("system_name", {})
        val = sys_field.get("value")
        if isinstance(val, str) and val.strip():
            normalized = val.strip().title()
            if len(normalized.split()) <= 6:
                return normalized

    overview = coverage.get("system_overview", {})
    content_list = overview.get("content", [])
    if isinstance(content_list, str):
        content_list = [content_list]
    combined = " ".join(str(c) for c in content_list)
    match = re.search(
        r"\b([A-Za-z0-9][A-Za-z0-9\s\-]{2,40}?)\s+(?:platform|system|application|service)\b",
        combined,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().title()
    return "KT Document"


INFERRED_MARKER = " *(inferred — not explicitly stated in the transcript)*"


def _apply_evidence_marker(value: Any, source: str) -> Any:
    """Flag genuinely-inferred values so the PDF/UI can distinguish them from
    transcript-grounded ones, without touching every renderer: only
    source == "llm" (field_populator.py's free-form gap-fill prompt) invents
    a value with no direct grounding sentence — pattern/semantic/
    llm_structured are all anchored to real transcript text and are left
    alone. Only applies to non-empty strings; other field types (bool,
    list, etc.) are returned unchanged.
    """
    if source == "llm" and isinstance(value, str) and value.strip():
        return value + INFERRED_MARKER
    return value


def build_knowledge_object(
    job_id: str,
    coverage: Dict[str, Any],
    dynamic_schema: List[Dict[str, Any]],
    populated_fields: Dict[str, Dict[str, Any]],
    section_content: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a consolidated Knowledge Object for the KT pipeline."""
    section_content = section_content or {}

    knowledge_sections = []
    for section in dynamic_schema:
        section_id = section.get("id")
        section_title = section.get("title") or section_id
        section_cov = coverage.get(section_id, {})
        section_evidence = section_content.get(section_id, {})

        raw_fields = populated_fields.get(section_id, {})
        field_objects = {fid: {
            "id": fid,
            "label": field.get("label", fid),
            "type": field.get("type", "text"),
            "value": _apply_evidence_marker(field.get("value"), field.get("source", "unfilled")),
            "confidence": float(field.get("confidence", 0.0) or 0.0),
            "source": field.get("source", "unfilled"),
            "evidence": _collect_evidence(section_id, section_evidence, field),
        } for fid, field in raw_fields.items()}

        section_facts = build_facts(section_id, field_objects)
        section_entities = build_entities(field_objects)
        section_evidence_list = build_evidence(section_evidence.get("sentences", []))
        section_relations = build_relationships(field_objects)

        # Extract content from coverage data: support both "content" (legacy) and "fragments" (current)
        content_data = section_cov.get("content", []) or section_cov.get("fragments", [])
        
        section_dict = {
            "id": section_id,
            "title": section_title,
            "section_type": section.get("type", "section"),
            "description": section.get("description", ""),
            "tier": section.get("tier", "core"),
            "status": section_cov.get("status", "missing"),
            "confidence": float(section_cov.get("confidence", 0.0) or 0.0),
            "sentence_count": int(section_cov.get("sentence_count", 0) or 0),
            "risk": float(section_cov.get("risk", 0.0) or 0.0),
            "coverage_content": content_data,
            "facts": section_facts,
            "entities": section_entities,
            "evidence": section_evidence_list,
            "relationships": section_relations,
            "fields": field_objects,
        }
        
        # Include structured data if available (e.g., for monitoring section)
        if "_structured" in section_cov:
            section_dict["_structured"] = section_cov["_structured"]
        
        knowledge_sections.append(section_dict)

    return {
        "job_id": job_id,
        "system_name": _infer_system_name(coverage, populated_fields),
        "sections": knowledge_sections,
        "summary": {
            "section_count": len(knowledge_sections),
            "covered_sections": sum(1 for sec in knowledge_sections if sec.get("status") in {"covered", "weak"}),
        },
    }


def append_unmapped_findings_section(
    knowledge_object: Dict[str, Any],
    unassigned_sentences: Optional[List[Any]],
) -> Dict[str, Any]:
    """Surface sentences that never confidently classified into any real
    schema section (context_mapper.py's StructuredKT.unassigned_sentences)
    instead of letting them vanish silently between stages.

    Generic by construction: this only depends on there being *some*
    leftover sentences the classifier couldn't place — it doesn't know or
    care what schema, transcript, or domain produced them, so it needs no
    changes for a different KT topic. Mutates and returns `knowledge_object`
    for convenient chaining; a no-op (returns it unchanged) when nothing
    survives the length filter, so no empty appendix/TOC entry is ever added.
    """
    sentences = unassigned_sentences or []
    surviving = [
        s for s in sentences
        if isinstance(getattr(s, "text", None), str) and len(s.text.split()) >= _MIN_UNMAPPED_SENTENCE_WORDS
    ]
    if not surviving:
        return knowledge_object

    coverage_content = [s.text.strip() for s in surviving]
    evidence = [
        {
            "sentence_index": idx,
            "text": _normalize_text(s.text),
            "start": getattr(s, "start", None),
            "end": getattr(s, "end", None),
            "speaker": getattr(s, "speaker", None),
            "audio_confidence": getattr(s, "audio_confidence", None),
        }
        for idx, s in enumerate(surviving)
    ]

    section_dict = {
        "id": UNMAPPED_FINDINGS_SECTION_ID,
        "title": UNMAPPED_FINDINGS_TITLE,
        "section_type": "appendix",
        "description": (
            "Sentences from the KT session that did not confidently match any "
            "other section. Reviewed by the outgoing owner for placement or "
            "removal."
        ),
        "status": "covered",
        "confidence": 0.5,
        "sentence_count": len(surviving),
        "risk": 0.0,
        "coverage_content": coverage_content,
        "facts": [],
        "entities": [],
        "evidence": evidence,
        "relationships": [],
        "fields": {},
    }

    sections = knowledge_object.setdefault("sections", [])
    sections.append(section_dict)

    summary = knowledge_object.setdefault("summary", {})
    summary["section_count"] = summary.get("section_count", len(sections) - 1) + 1
    summary["covered_sections"] = summary.get("covered_sections", 0) + 1

    return knowledge_object


def _find_section(knowledge_object: Dict[str, Any], section_id: str) -> Optional[Dict[str, Any]]:
    for section in knowledge_object.get("sections") or []:
        if section.get("id") == section_id:
            return section
    return None


def _field_value(section: Optional[Dict[str, Any]], *path: str) -> Optional[Any]:
    """Read a (possibly nested/group) field's value out of a knowledge-object
    section dict's `fields` map, e.g. _field_value(sec, "impact_if_down",
    "what_breaks"). Returns None if any part of the path is missing/unfilled."""
    if not section:
        return None
    node: Any = section.get("fields") or {}
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    if isinstance(node, dict):
        value = node.get("value")
        return value if value else None
    return None


def enrich_operational_calendar(knowledge_object: Dict[str, Any]) -> Dict[str, Any]:
    """Fold cost_optimization's already-populated levers into
    known_bad_days's section dict as `_cost_patterns`, so its renderer can
    show peak-period restrictions and cost-related operating patterns
    together as one "Operational Calendar" — reads a sibling section's
    already-built data rather than reclassifying anything (renderers are
    single-section scoped; this is the enrichment point that gets around
    that without changing the renderer contract everywhere).
    """
    calendar_section = _find_section(knowledge_object, "known_bad_days")
    cost_section = _find_section(knowledge_object, "cost_optimization")
    if not calendar_section or not cost_section:
        return knowledge_object

    levers = (cost_section.get("_structured") or {}).get("levers")
    rows: List[Dict[str, str]] = []
    if isinstance(levers, list):
        for item in levers:
            if not isinstance(item, dict):
                continue
            label = str(item.get("lever") or "").strip()
            detail = str(item.get("detail") or "").strip()
            if label:
                rows.append({"label": label, "value": detail})

    if not rows:
        content = cost_section.get("coverage_content") or []
        if isinstance(content, str):
            content = [content]
        for item in content:
            text = str(item).strip()
            parts = [p.strip() for p in text.split("|") if p.strip()]
            if len(parts) >= 2:
                rows.append({"label": parts[0], "value": parts[1]})

    if rows:
        calendar_section["_cost_patterns"] = rows

    return knowledge_object


# Generic, section-id-keyed heuristics for the Tribal Knowledge digest —
# not tied to any one transcript's wording. Extend by section id if a new
# section should carry tribal-knowledge weight, not by adding transcript-
# specific phrases here.
_TRIBAL_SOURCE_LABELS = {
    "danger_zones": ("Safety-critical", "Prevents a high-risk operational mistake."),
    "monitoring_observability": ("Operational shortcut", "Encodes experienced first-response behavior."),
    "common_failures": ("Troubleshooting heuristic", "Provides diagnostic ordering."),
    "architecture_reference": ("Tribal / operational", "Avoids misreading expected behavior as a failure."),
}
_TRIBAL_DEFAULT_LABEL = ("Operational", "Non-obvious operational detail worth remembering.")

# Sections whose entire purpose already IS non-obvious, safety-critical
# operational knowledge — every sentence there is tribal-knowledge-eligible,
# not just ones matching a phrase marker (keyed by section id, so this
# generalizes to any transcript, not this one's specific wording).
_TRIBAL_ALWAYS_ELIGIBLE_SECTIONS = {"danger_zones"}

TRIBAL_KNOWLEDGE_SECTION_ID = "tribal_knowledge"
TRIBAL_KNOWLEDGE_TITLE = "Tribal Knowledge"


def append_tribal_knowledge_section(
    knowledge_object: Dict[str, Any],
    section_content: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Digest of non-obvious, experience-based knowledge, pulled from
    wherever it already landed (section_content = kt.section_content) by
    tagging sentences that match generic tribal-knowledge phrasing
    (section_rules.is_tribal_knowledge) — additive tagging across sections,
    not a reclassification, so it works for any schema/transcript.
    """
    section_content = section_content or {}
    seen_texts = set()
    rows: List[Dict[str, str]] = []

    for section_id, entry in section_content.items():
        sentences = (entry or {}).get("sentences") or []
        label, value_text = _TRIBAL_SOURCE_LABELS.get(section_id, _TRIBAL_DEFAULT_LABEL)
        always_eligible = section_id in _TRIBAL_ALWAYS_ELIGIBLE_SECTIONS
        for sentence in sentences:
            text = (sentence or {}).get("text", "") if isinstance(sentence, dict) else ""
            if not text or not (always_eligible or is_tribal_knowledge(text)):
                continue
            normalized = _normalize_text(text)
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            rows.append({"Knowledge": text.strip(), "Value": value_text, "Classification": label})

    if not rows:
        return knowledge_object

    section_dict = {
        "id": TRIBAL_KNOWLEDGE_SECTION_ID,
        "title": TRIBAL_KNOWLEDGE_TITLE,
        "section_type": "digest",
        "description": "Non-obvious operational knowledge surfaced from across the KT session.",
        "status": "covered",
        "confidence": 0.5,
        "sentence_count": len(rows),
        "risk": 0.0,
        "coverage_content": [],
        "facts": [],
        "entities": [],
        "evidence": [],
        "relationships": [],
        "fields": {},
        "_tribal_rows": rows,
    }
    sections = knowledge_object.setdefault("sections", [])
    sections.append(section_dict)
    summary = knowledge_object.setdefault("summary", {})
    summary["section_count"] = summary.get("section_count", len(sections) - 1) + 1
    summary["covered_sections"] = summary.get("covered_sections", 0) + 1
    return knowledge_object


KT_COVERAGE_SECTION_ID = "kt_coverage"
KT_COVERAGE_TITLE = "KT Coverage & Knowledge Gaps"


def append_coverage_matrix_section(
    knowledge_object: Dict[str, Any],
    coverage: Dict[str, Any],
    dynamic_schema: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Reshape the pipeline's existing per-section coverage status/confidence
    (already computed before this runs — no new extraction) into a
    Domain/Coverage/Assessment matrix, matching the golden-reference "KT
    Coverage & Knowledge Gaps" page.
    """
    rows: List[Dict[str, str]] = []
    gaps: List[str] = []
    for section in dynamic_schema:
        section_id = section.get("id")
        title = section.get("title") or section_id
        cov = coverage.get(section_id) or {}
        status = cov.get("status", "missing")
        sentence_count = int(cov.get("sentence_count", 0) or 0)

        # Bucket directly from the pipeline's own status — it's already
        # required-vs-optional- and density-aware (semantic_coverage_score()
        # in context_mapper.py). A separate confidence threshold on top of
        # that was uncalibrated in practice: real runs showed status
        # correctly reaching "covered" while the mean block-confidence
        # signal stayed under an arbitrary 0.6, so "Strong" never fired.
        if status == "covered":
            bucket = "Strong"
            assessment = f"{sentence_count} supporting sentence(s) captured with good confidence."
        elif status == "weak":
            bucket = "Partial"
            assessment = f"{sentence_count} supporting sentence(s) captured; some detail may be missing."
        else:
            bucket = "Missing"
            assessment = "Not covered in the KT session."
            gaps.append(title)

        rows.append({"Domain": title, "Coverage": bucket, "Assessment": assessment})

    if not rows:
        return knowledge_object

    section_dict = {
        "id": KT_COVERAGE_SECTION_ID,
        "title": KT_COVERAGE_TITLE,
        "section_type": "digest",
        "description": "Coverage assessment across every section of this KT.",
        "status": "covered",
        "confidence": 0.5,
        "sentence_count": len(rows),
        "risk": 0.0,
        "coverage_content": [],
        "facts": [],
        "entities": [],
        "evidence": [],
        "relationships": [],
        "fields": {},
        "_coverage_rows": rows,
        # Distinct from open_responsibilities' actual assigned tasks — these
        # are areas the KT session never covered at all, not work items
        # someone agreed to do. Keeping the two concepts visually separate
        # (a knowledge gap isn't an open task) per the golden-reference
        # standard's own distinction.
        "_knowledge_gaps": gaps,
    }
    sections = knowledge_object.setdefault("sections", [])
    sections.append(section_dict)
    summary = knowledge_object.setdefault("summary", {})
    summary["section_count"] = summary.get("section_count", len(sections) - 1) + 1
    summary["covered_sections"] = summary.get("covered_sections", 0) + 1
    return knowledge_object


QUICK_REFERENCE_SECTION_ID = "quick_reference"
QUICK_REFERENCE_TITLE = "Quick Reference"


def _join_if_list(value: Any) -> Optional[str]:
    if isinstance(value, list):
        joined = "; ".join(str(v).strip() for v in value if str(v).strip())
        return joined or None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _first_matching_paragraph(section: Optional[Dict[str, Any]], *keywords: str) -> Optional[str]:
    """First coverage_content paragraph containing any of the given
    case-insensitive keywords — used for sections with no `fields` structure
    (e.g. danger_zones, open_responsibilities), where the fact lives only as
    raw/polished sentence text."""
    if not section:
        return None
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    for item in content:
        text = str(item).strip()
        if text and any(kw.lower() in text.lower() for kw in keywords):
            return text
    return None


def _first_failure_fix(section: Optional[Dict[str, Any]], *symptom_keywords: str) -> Optional[str]:
    """The "How to Fix" value from common_failures's _structured.failures
    list for the first entry whose symptom matches — common_failures has no
    `fields` structure either; its data lives under `_structured`."""
    failures = (section.get("_structured") or {}).get("failures") if section else None
    if not isinstance(failures, list):
        return None
    for item in failures:
        if not isinstance(item, dict):
            continue
        symptom = str(item.get("symptom") or "").lower()
        if any(kw.lower() in symptom for kw in symptom_keywords):
            fix = str(item.get("fix") or "").strip()
            if fix:
                return fix
    return None


def append_quick_reference_section(knowledge_object: Dict[str, Any]) -> Dict[str, Any]:
    """A cheat-sheet of the most operationally urgent facts, assembled
    purely by looking up data other sections already produced — each lookup
    matches that section's actual shape (some have a `fields` map, some only
    `_structured`, some only raw `coverage_content`). Any row whose source
    is empty is skipped, never fabricated, so this adapts to whatever the
    transcript actually covered.
    """
    rows: List[Dict[str, str]] = []

    monitoring = _find_section(knowledge_object, "monitoring_observability")
    value = _join_if_list(_field_value(monitoring, "first_response_steps"))
    if value:
        rows.append({"Situation": "Alert / PagerDuty trigger", "Immediate reference": value})

    failures = _find_section(knowledge_object, "common_failures")
    value = _first_failure_fix(failures, "pod", "deploy")
    if value:
        rows.append({"Situation": "Pod fails immediately after deployment", "Immediate reference": value})

    deployment = _find_section(knowledge_object, "deployment_and_rollback")
    value = _field_value(deployment, "rollback_procedure", "rollback_time")
    if value:
        rows.append({"Situation": "Rollback required", "Immediate reference": f"Target completion: {value}"})

    danger = _find_section(knowledge_object, "danger_zones")
    value = _first_matching_paragraph(danger, "terraform")
    if value:
        rows.append({"Situation": "Terraform state", "Immediate reference": value})

    open_resp = _find_section(knowledge_object, "open_responsibilities")
    value = _first_matching_paragraph(open_resp, "contact platform engineering", "unaware", "unsure")
    if value:
        rows.append({"Situation": "Production activity unclear", "Immediate reference": value})

    overview = _find_section(knowledge_object, "system_overview")
    value = _field_value(overview, "impact_if_down", "what_breaks")
    if value:
        rows.append({"Situation": "System unavailable", "Immediate reference": value})

    ownership = _find_section(knowledge_object, "ownership_escalation")
    value = _field_value(ownership, "escalation_chain")
    if value:
        rows.append({"Situation": "Escalation", "Immediate reference": value})

    if not rows:
        return knowledge_object

    section_dict = {
        "id": QUICK_REFERENCE_SECTION_ID,
        "title": QUICK_REFERENCE_TITLE,
        "section_type": "digest",
        "description": "Incident and production cheat sheet.",
        "status": "covered",
        "confidence": 0.5,
        "sentence_count": len(rows),
        "risk": 0.0,
        "coverage_content": [],
        "facts": [],
        "entities": [],
        "evidence": [],
        "relationships": [],
        "fields": {},
        "_quick_reference_rows": rows,
    }
    sections = knowledge_object.setdefault("sections", [])
    sections.append(section_dict)
    summary = knowledge_object.setdefault("summary", {})
    summary["section_count"] = summary.get("section_count", len(sections) - 1) + 1
    summary["covered_sections"] = summary.get("covered_sections", 0) + 1
    return knowledge_object
