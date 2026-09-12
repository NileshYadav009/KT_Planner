import re
from typing import Any, Dict, List, Optional

from .entities import build_entities
from .evidence import build_evidence
from .facts import build_facts
from .relationships import build_relationships
from section_rules import is_tribal_knowledge
from field_populator import PATTERN_EXTRACTORS

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
    transcript-grounded ones, without touching every renderer.

    field_populator.py's free-form gap-fill prompt now self-classifies each
    extraction as EXPLICIT (the transcript directly states this, even if
    paraphrased) or INFERRED (the model had to reason/guess beyond what's
    literally said) and tags the result "llm_explicit" or "llm"
    accordingly — only the latter is marked here. pattern/semantic/
    llm_structured/llm_explicit are all anchored to real transcript text and
    are left alone. Only applies to non-empty strings; other field types
    (bool, list, etc.) are returned unchanged.
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


def _iter_field_strings(fields: Any):
    """Yield every string value reachable from a section's `fields` map,
    recursing into group fields (which nest sub-fields directly rather than
    under a "value" key). Generic — doesn't know field ids or a schema."""
    if not isinstance(fields, dict):
        return
    for field in fields.values():
        if not isinstance(field, dict):
            continue
        value = field.get("value")
        if isinstance(value, str) and value.strip():
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    yield item
        elif value is None:
            # No "value" key at all likely means this is a nested group
            # (field_populator._populate_fields_recursive stores group
            # sub-fields directly, not under a "value" wrapper) — recurse.
            yield from _iter_field_strings(field)


_DEDUP_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")


def _dedup_normalize(text: str) -> str:
    """Lowercased, punctuation-stripped, whitespace-collapsed text — looser
    than _normalize_text() on purpose. Two mechanisms in this pipeline can
    each independently touch the same original sentence (an LLM-polish pass
    that fixes comma placement/wording for one section vs. a raw discourse-
    marker-trimming step for another), so an exact/whitespace-only compare
    misses real duplicates that a human would obviously recognize as the
    same fact. Stripping punctuation is enough to reconcile the polish-pass
    case; the substring check below (checked in both directions) handles
    the trimmed-discourse-marker case."""
    text = _DEDUP_PUNCT_RE.sub(" ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _mapped_text_chunks(knowledge_object: Dict[str, Any]) -> List[str]:
    """Normalized text of everything already classified into a real section
    (coverage_content + every populated field value), kept as separate
    chunks (not one joined blob) so a short, already-mapped fact can be
    matched as a substring of a longer unassigned sentence and vice versa.
    Generic: reads only the already-built knowledge_object, no section-id-
    specific logic."""
    chunks: List[str] = []
    for section in knowledge_object.get("sections") or []:
        content = section.get("coverage_content") or []
        if isinstance(content, str):
            content = [content]
        chunks.extend(str(c) for c in content if isinstance(c, str) and c.strip())
        chunks.extend(_iter_field_strings(section.get("fields")))
    return [_dedup_normalize(c) for c in chunks if str(c).strip()]


def _is_duplicate_of_mapped_content(sentence_text: str, mapped_chunks: List[str]) -> bool:
    """True if `sentence_text` is essentially the same fact as something
    already mapped into a real section — checked both directions since
    either side can be the "trimmed" one (an LLM-polished paragraph is
    often a superset of the raw sentence; a raw sentence with a leading
    discourse marker like "Coming back to architecture, ..." is a superset
    of the trimmed fact that made it into the real section)."""
    normalized = _dedup_normalize(sentence_text)
    if not normalized:
        return False
    for chunk in mapped_chunks:
        if not chunk:
            continue
        if normalized in chunk:
            return True
        # Only treat the mapped side as sufficient evidence on its own when
        # it's not a trivially short/generic fragment — avoids a short
        # mapped field value ("Yes", "Trivy") spuriously matching an
        # unrelated long unassigned sentence that happens to contain it.
        if len(chunk.split()) >= 4 and chunk in normalized:
            return True
    return False


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

    context_mapper.py's classifier can independently decide the same
    sentence both belongs in a real section (feeding that section's
    coverage_content/fields) AND is "unassigned" (its own separate
    mechanism) — without a cross-check, that sentence would render twice:
    once correctly, once again here as a fabricated "gap". Anything whose
    normalized text is already a substring of the mapped content is
    filtered out before the word-count filter runs.
    """
    sentences = unassigned_sentences or []
    mapped_chunks = _mapped_text_chunks(knowledge_object)
    surviving = [
        s for s in sentences
        if isinstance(getattr(s, "text", None), str)
        and len(s.text.split()) >= _MIN_UNMAPPED_SENTENCE_WORDS
        and not _is_duplicate_of_mapped_content(s.text, mapped_chunks)
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


def enrich_technology_summary(knowledge_object: Dict[str, Any]) -> Dict[str, Any]:
    """Fold tool mentions that correctly classified into their own dedicated
    sections (monitoring_observability's `tools`, security_controls'
    `security_scan_config`) into system_overview's `key_technologies` value,
    so the Technology Summary table is a genuine one-stop inventory instead
    of only reflecting whatever tools happened to be mentioned in the same
    sentences as the rest of system_overview's narrative. Those tools aren't
    reclassified or duplicated as their own facts — Monitoring/Security keep
    owning the real content; this only extends the flat tool-name list
    system_overview.py's renderer already categorizes into rows.
    Generic: driven entirely by section ids and field shapes that exist for
    any transcript, not this one's specific tool choices.
    """
    overview_section = _find_section(knowledge_object, "system_overview")
    if not overview_section:
        return knowledge_object

    existing_value = (overview_section.get("fields", {}).get("key_technologies") or {}).get("value")
    found: List[str] = [t.strip() for t in str(existing_value or "").split(",") if t.strip()]

    def _raw_text(section: Optional[Dict[str, Any]]) -> str:
        content = (section or {}).get("coverage_content") or []
        if isinstance(content, str):
            content = [content]
        return " ".join(str(c) for c in content if isinstance(c, str))

    monitoring_section = _find_section(knowledge_object, "monitoring_observability")
    if monitoring_section:
        tools = (monitoring_section.get("fields", {}).get("tools") or {}).get("value")
        if isinstance(tools, list):
            found.extend(str(t).strip() for t in tools if str(t).strip())
        elif isinstance(tools, str) and tools.strip():
            found.extend(m for m in PATTERN_EXTRACTORS["tools"].findall(tools))
        else:
            # LLM structured extraction can fail/be unavailable and leave
            # `fields` empty even though the raw sentences (still present in
            # coverage_content) clearly name real tools — fall back to the
            # same tools regex directly against the raw text rather than
            # losing this data whenever the LLM path doesn't fire.
            found.extend(m for m in PATTERN_EXTRACTORS["tools"].findall(_raw_text(monitoring_section)))

    security_section = _find_section(knowledge_object, "security_controls")
    if security_section:
        scan_config = (security_section.get("fields", {}).get("security_scan_config") or {}).get("value")
        if isinstance(scan_config, str) and scan_config.strip():
            found.extend(m for m in PATTERN_EXTRACTORS["tools"].findall(scan_config))
        else:
            found.extend(m for m in PATTERN_EXTRACTORS["tools"].findall(_raw_text(security_section)))

    if not found:
        return knowledge_object

    # Dedupe case-insensitively while preserving first-seen casing/order —
    # system_overview.py's own categorization joins same-category tools with
    # "; ", so an uncaught duplicate here would render as "Trivy; Trivy".
    seen = set()
    deduped = []
    for tool in found:
        key = tool.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(tool)

    overview_section.setdefault("fields", {}).setdefault(
        "key_technologies", {"id": "key_technologies", "label": "Key Technologies", "type": "text", "confidence": 0.6, "source": "cross_section", "evidence": []}
    )["value"] = ", ".join(deduped)

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


def _leaf_field_specs(fields_schema: Optional[List[Dict[str, Any]]]) -> List[tuple]:
    """(field_id, label) for schema fields that land as their own,
    independently-checkable field_objects entry — i.e. everything except
    "list" fields (static reference bullets, not per-transcript extracted
    facts) and "group" containers (field_populator nests their sub-fields
    directly rather than surfacing the group itself as a value, so checking
    the group id alone would always read as unfilled). Generic: driven
    entirely by each section's own schema field list, not by section id."""
    specs = []
    for f in fields_schema or []:
        ftype = f.get("type", "text")
        if ftype in ("list", "group"):
            continue
        specs.append((f.get("id"), f.get("label", f.get("id"))))
    return specs


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
        elif status == "weak":
            bucket = "Partial"
        else:
            bucket = "Missing"

        if bucket == "Missing":
            assessment = "Not covered in the KT session."
            gaps.append(title)
        else:
            # Prefer a meaningful "how many of this section's known fields
            # actually got captured" readout over a raw sentence count —
            # five sentences can back one field or ten, so the count alone
            # says nothing about completeness. Falls back to the sentence
            # count only for schema-less/digest sections (no leaf fields to
            # check against, e.g. danger_zones' free-text list).
            leaf_specs = _leaf_field_specs(section.get("fields"))
            section_obj = _find_section(knowledge_object, section_id)
            field_objects = (section_obj or {}).get("fields") or {}
            if leaf_specs:
                missing_labels = []
                filled_count = 0
                for fid, label in leaf_specs:
                    entry = field_objects.get(fid) or {}
                    has_value = (
                        entry.get("source", "unfilled") != "unfilled"
                        and entry.get("value") not in (None, "", [], {})
                    )
                    if has_value:
                        filled_count += 1
                    else:
                        missing_labels.append(label)
                assessment = f"{filled_count} of {len(leaf_specs)} known field(s) captured"
                if missing_labels:
                    shown = ", ".join(missing_labels[:4])
                    extra = len(missing_labels) - 4
                    if extra > 0:
                        shown += f" (+{extra} more)"
                    assessment += f"; missing: {shown}."
                else:
                    assessment += "."
            else:
                assessment = (
                    f"{sentence_count} supporting sentence(s) captured with good confidence."
                    if bucket == "Strong"
                    else f"{sentence_count} supporting sentence(s) captured; some detail may be missing."
                )

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
