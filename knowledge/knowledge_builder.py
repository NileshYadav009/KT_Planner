import re
from typing import Any, Dict, List, Optional

from .entities import build_entities
from .evidence import build_evidence
from .facts import build_facts
from .relationships import build_relationships
from section_rules import is_tribal_knowledge
from field_populator import PATTERN_EXTRACTORS, SYSTEM_NAME_STOPWORDS, _trim_name_capture
from architecture_diagram import build_architecture_flow_diagram
from renderers.blocks.common import split_bullet_blob

UNMAPPED_FINDINGS_SECTION_ID = "unmapped_findings"
UNMAPPED_FINDINGS_TITLE = "Additional Notes (Unmapped Findings)"
# A sentence below this many words is almost always a filler/transition
# ("Thank you.", "Okay so.") rather than a standalone fact worth surfacing.
# Generic length threshold — not keyed to any transcript's content.
_MIN_UNMAPPED_SENTENCE_WORDS = 4


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _section_sentences(section_content: Any) -> List[Dict[str, Any]]:
    """Every sentence dict for one section_content entry.

    The two independent mechanisms that populate section_content[id]
    ['sentences'] vs. ['blocks'] can disagree, leaving 'sentences' empty for
    a section that has real coverage — so anything reading sentences for a
    section must consult both or it silently sees nothing. Shared by
    _collect_evidence() (where an index computed elsewhere must refer to the
    same list) and append_tribal_knowledge_section() (which lost real tagged
    sentences to exactly this gap on a live run).
    """
    if not isinstance(section_content, dict):
        return []
    sentences = section_content.get("sentences") or []
    if sentences:
        return sentences
    return [
        s for block in (section_content.get("blocks") or [])
        for s in (block.get("sentences") or [])
    ]


def _collect_evidence(
    section_id: str,
    section_content: Dict[str, Any],
    populated_field: Dict[str, Any],
) -> List[Dict[str, Any]]:
    sentence_entries = _section_sentences(section_content)

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
    # re.search (not finditer) used to take the leftmost match unconditionally
    # — since "the system"/"the platform" is an extremely common phrase, the
    # non-greedy capture frequently landed on a bare stopword ("The") instead
    # of a real name. finditer + a stopword-rejection guard keeps trying
    # later candidates in the same text until a substantive one is found.
    for match in re.finditer(
        r"\b([A-Za-z0-9][A-Za-z0-9\s\-']{2,40}?)\s+(?:platform|system|application|service)\b",
        combined,
        re.IGNORECASE,
    ):
        name = _trim_name_capture(match.group(1).strip())
        if name and name.lower() not in SYSTEM_NAME_STOPWORDS:
            return name.title()
    return "KT Document"


def _flatten_schema_fields(fields_schema: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Flatten a schema section's field list into {field_id: field_def},
    recursing into "group" fields (whose sub-fields are what populate_fields()
    actually stores flat entries for). Used to recover a populated field's
    real label/type, since the populated entry itself never carries them."""
    flat: Dict[str, Dict[str, Any]] = {}
    for f in fields_schema or []:
        fid = f.get("id")
        if fid:
            flat[fid] = f
        if f.get("type") == "group":
            flat.update(_flatten_schema_fields(f.get("fields")))
    return flat


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
        # field_populator.py's populated entries are {"value", "confidence",
        # "source", ...} only — they never carry the schema's own "label"/
        # "type" (those live on the SCHEMA field definition, a different
        # dict). Reading field.get("label", fid) straight off the populated
        # entry always fell through to the bare field id, and field.get(
        # "type", "text") always returned "text" regardless of the real
        # declared type (url/date/boolean/single_select/...) — silently
        # wrong for every field in every section, just rarely visible
        # because most renderers hardcode their own display labels instead
        # of trusting knowledge_object["sections"][x]["fields"][id]["label"].
        schema_fields_by_id = _flatten_schema_fields(section.get("fields"))
        field_objects = {fid: {
            "id": fid,
            "label": (schema_fields_by_id.get(fid) or {}).get("label", fid),
            "type": (schema_fields_by_id.get(fid) or {}).get("type", "text"),
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


# Content words below this length carry no identifying signal for the
# overlap comparison below (articles, prepositions, auxiliaries).
_DEDUP_STOPWORDS = frozenset(
    "a an and are as at be been but by can for from had has have if in into is it its "
    "of on or that the their then there these this to was were when which while will "
    "with you your".split()
)
# Share of a candidate's content words that must already appear in one
# mapped chunk before it counts as the same fact. High enough that two
# genuinely different sentences about the same technology stay distinct
# ("Redis provides caching" vs "Redis memory saturation caused an
# incident" overlap on only ~1 content word out of 4-5).
_DEDUP_OVERLAP_RATIO = 0.8
_DEDUP_MIN_CONTENT_WORDS = 5


def _content_words(text: str) -> List[str]:
    return [w for w in _dedup_normalize(text).split() if w not in _DEDUP_STOPWORDS]


def _is_near_duplicate(candidate: str, chunk: str) -> bool:
    """True when nearly all of `candidate`'s content words already appear in
    `chunk` — catching the same fact after a rewording that defeats a plain
    substring test.

    Real case this exists for: the polish pass turned "Tribal knowledge,
    service bus backlog can temporarily increase during large deployments.
    Check whether the deployment has completed..." into "Tribal knowledge
    indicates that the service bus backlog can temporarily increase during
    large deployments. Verify whether the deployment has completed...".
    Neither string contains the other, so the same fact was published twice
    — once in its real section and again as an "unmapped finding".
    """
    cand_words = _content_words(candidate)
    if len(cand_words) < _DEDUP_MIN_CONTENT_WORDS:
        return False
    chunk_words = set(_content_words(chunk))
    if not chunk_words:
        return False
    shared = sum(1 for w in cand_words if w in chunk_words)
    return (shared / len(cand_words)) >= _DEDUP_OVERLAP_RATIO


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
        if _is_near_duplicate(sentence_text, chunk):
            return True
    return False


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Share of a transcript sentence's content words that must appear somewhere
# in the mapped document for it to count as retained. Lower than the
# deduplication ratio on purpose: this answers the much weaker question
# "did this survive anywhere at all?", so it must not cry loss over a
# sentence the polish pass legitimately condensed.
_RETENTION_RATIO = 0.75


def _unretained_transcript_sentences(
    transcript: str,
    mapped_chunks: List[str],
    already_listed: List[str],
) -> List[str]:
    """Transcript sentences whose content reached no part of the document.

    The classifier can drop a sentence without ever reporting it as
    unassigned, and the polish pass can drop one while rewriting a section —
    in both cases the sentence is simply gone, and the knowledge-coverage
    summary cannot see it, because that summary is computed from what the
    pipeline *knows* about (mapped + deduplicated + unassigned) rather than
    from the transcript itself. Confirmed live on a real KT: "Production
    Kubernetes configuration must not be changed manually." — a danger zone —
    vanished from the document while the summary still reported zero loss.

    This compares the actual transcript against the actual document, so a
    silently dropped fact becomes visible instead of disappearing.
    """
    if not transcript:
        return []
    mapped_words = set()
    for chunk in mapped_chunks:
        mapped_words.update(chunk.split())
    for text in already_listed:
        mapped_words.update(_dedup_normalize(text).split())

    missing: List[str] = []
    seen: set = set()
    for raw in _SENTENCE_SPLIT_RE.split(transcript):
        sentence = raw.strip()
        if len(sentence.split()) < _MIN_UNMAPPED_SENTENCE_WORDS:
            continue
        words = [w for w in _dedup_normalize(sentence).split() if w not in _DEDUP_STOPWORDS]
        if len(words) < _DEDUP_MIN_CONTENT_WORDS:
            continue
        retained = sum(1 for w in words if w in mapped_words) / len(words)
        if retained >= _RETENTION_RATIO:
            continue
        key = _dedup_normalize(sentence)
        if key in seen:
            continue
        seen.add(key)
        missing.append(sentence)
    return missing


def append_unmapped_findings_section(
    knowledge_object: Dict[str, Any],
    unassigned_sentences: Optional[List[Any]],
    transcript: Optional[str] = None,
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
    # Split into two named passes (rather than one combined filter) so the
    # counts at each stage can be reported separately — sub-word-count
    # fragments are filler, never "facts" to begin with (§2's "substantive
    # fact" carve-out), so they're excluded before either count is taken.
    substantive = [
        s for s in sentences
        if isinstance(getattr(s, "text", None), str)
        and len(s.text.split()) >= _MIN_UNMAPPED_SENTENCE_WORDS
    ]
    surviving = [
        s for s in substantive
        if not _is_duplicate_of_mapped_content(s.text, mapped_chunks)
    ]
    coverage_content = [s.text.strip() for s in surviving]

    # Safety net: anything in the transcript that reached no part of the
    # document, whether or not the classifier ever reported it as
    # unassigned. See _unretained_transcript_sentences().
    recovered = _unretained_transcript_sentences(transcript or "", mapped_chunks, coverage_content)
    coverage_content.extend(recovered)

    # Stashed on the knowledge_object (not just this section's dict) so
    # append_coverage_matrix_section can build its knowledge-coverage
    # summary from these exact counts even when nothing survives to
    # render below — popped once consumed, see that function.
    knowledge_object["_dedup_stats"] = {
        "substantive_unassigned": len(substantive) + len(recovered),
        "deduplicated": len(substantive) - len(surviving),
        "unmapped": len(coverage_content),
        "recovered": len(recovered),
    }
    if not coverage_content:
        return knowledge_object
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
    # Recovered sentences come from the transcript text itself, so they have
    # no sentence object to read timings/speaker from — the text is still
    # real, verbatim evidence and must be recorded as such.
    evidence.extend(
        {
            "sentence_index": len(surviving) + idx,
            "text": _normalize_text(text),
            "start": None,
            "end": None,
            "speaker": None,
            "audio_confidence": None,
        }
        for idx, text in enumerate(recovered)
    )

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
        "sentence_count": len(coverage_content),
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

    # The architecture component inventory (built just before this, see
    # pipeline.py's ordering note) is the most complete list of what the
    # system actually runs on — folding it in is what makes this a genuine
    # one-stop technology summary instead of a list of whichever tools
    # happened to be named in system_overview's own sentences.
    arch_section = _find_section(knowledge_object, "architecture_reference")
    if arch_section:
        found.extend(str(c).strip() for c in (arch_section.get("_architecture_components") or []) if str(c).strip())

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


# A transcript very commonly says a service's full/branded name once
# ("Amazon RDS", "Azure Key Vault") and its bare acronym on every later
# mention ("RDS", "Key Vault") — both forms are separately valid
# PATTERN_EXTRACTORS["tools"] matches, which without this would show up as
# two different-looking entries in the same flat component list ("Amazon
# RDS, ..., RDS") even though they name the exact same thing. Collapses
# every captured form to one canonical display name before dedup.
_CANONICAL_TERM_ALIASES = {
    "rds": "Amazon RDS",
    "ecr": "Amazon ECR",
    "sqs": "Amazon SQS",
    "acr": "Azure Container Registry",
    "aks": "Azure Kubernetes Service",
    "key vault": "Azure Key Vault",
    "service bus": "Azure Service Bus",
    "blob storage": "Azure Blob Storage",
    "gke": "Google Kubernetes Engine",
    "argo cd": "ArgoCD",
    # See the ".NET microservices" note in field_populator.PATTERN_EXTRACTORS
    # — matched without its leading dot, restored to the real product name
    # here so the document never shows a bare "NET microservices".
    "net microservices": ".NET",
    "net microservice": ".NET",
    "dotnet": ".NET",
    "asp.net": "ASP.NET",
}


def _canonicalize_component_term(term: str) -> str:
    return _CANONICAL_TERM_ALIASES.get(term.strip().lower(), term)


# Sections whose sentences genuinely describe the architecture itself
# (what the system is built from and how requests flow through it), as
# opposed to sections that merely mention tools while describing a
# procedure. Only these contribute Architecture Details lines — see
# _scan_text() inside enrich_architecture_knowledge().
_ARCHITECTURE_SENTENCE_SECTIONS = frozenset({"architecture_reference", "system_overview"})


def enrich_architecture_knowledge(
    knowledge_object: Dict[str, Any],
    section_content: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Architecture Reference's own field schema holds only 3 administrative
    facts (doc link, last-updated, verified-by) — the real architecture
    knowledge (which services/frameworks the system is built on) gets
    classified across whatever sections the sentences naturally land in
    (system_overview, architecture_reference, environments, ...), so a
    transcript with a rich architecture discussion but no stated Confluence
    link used to render the section as "0 of 3 fields" as if nothing had
    been captured at all. This scans every section's raw transcript content
    for known infrastructure/framework names (the same PATTERN_EXTRACTORS
    tools regex enrich_technology_summary already uses) and attaches:
    - `_architecture_components`: the deduplicated flat component list —
      real knowledge, kept independent of which section's field schema
      happened to hold the sentence that mentioned it.
    - `_architecture_sentences`: the real transcript sentences that named
      those components verbatim — a bare component name ("Redis") says
      nothing about its ROLE; the sentence that mentioned it usually does
      ("Redis is used for caching and short-lived session data"). Sourced
      from BOTH section_content's raw per-sentence data (kt.section_content,
      clean atomic sentences) AND coverage_content (coarser, possibly
      LLM-polished multi-sentence chunks) — not an either/or fallback, since
      a section's raw ['sentences'] and its coverage_content are populated
      by two independent mechanisms that can disagree (see below). Never
      paraphrased or generated — verbatim transcript text only.
    - `_architecture_diagram`: a top-down "mental model" diagram built from
      the component list's coarse layer classification (see
      architecture_diagram.py) — None when nothing resembling a
      request-flow position was named.
    """
    arch_section = _find_section(knowledge_object, "architecture_reference")
    if not arch_section:
        return knowledge_object

    tools_pattern = PATTERN_EXTRACTORS["tools"]
    found: List[str] = []
    descriptive_sentences: List[str] = []

    def _scan_text(text: str, *, describes_architecture: bool = True) -> None:
        """Collect components from `text`; additionally keep the sentence
        itself as an Architecture Details line only when it came from a
        section that actually describes the architecture.

        The component list deliberately scans EVERYTHING (a component named
        anywhere is still a real component). The sentence list must not:
        in a DevOps KT nearly every sentence names some tool, so collecting
        all of them turned Architecture Details into a near-copy of the
        whole transcript — on a live run it restated the deployment,
        monitoring, DR, rollback and danger-zone sections verbatim, each of
        which already renders that same text in its own section. "Do not
        treat every technology mention as an architecture relationship."
        """
        text = (text or "").strip()
        if not text:
            return
        matches = tools_pattern.findall(text)
        if matches:
            found.extend(matches)
            if describes_architecture:
                descriptive_sentences.append(text)

    if section_content:
        for sc_id, sc in section_content.items():
            architectural = sc_id in _ARCHITECTURE_SENTENCE_SECTIONS
            for s in _section_sentences(sc):
                _scan_text(
                    (s or {}).get("text", "") if isinstance(s, dict) else "",
                    describes_architecture=architectural,
                )

    # Always ALSO scan coverage_content — never gated on "found nothing at
    # all yet". context_mapper.py populates a section's raw ['sentences']
    # and its ['content']/coverage_content via two independent mechanisms
    # that can disagree (documented in field_populator.py's
    # populate_fields()): a section can have real coverage_content while
    # its own raw ['sentences'] stays empty or incomplete. A single
    # "nothing found anywhere yet" gate here would silently drop a REAL
    # component the instant any OTHER section's raw sentences matched
    # something first — confirmed live: a transcript's "Amazon RDS
    # PostgreSQL is the primary database... Amazon SQS handles async order
    # processing" and "The back end consists of Python FastAPI
    # microservices" sentences were visibly rendered elsewhere in the KT
    # (system_overview's own content) yet never reached the component list
    # or diagram, because some other section's sentences had already
    # produced a match first. Duplicate matches across both sources are
    # harmless — dedup below collapses them.
    for section in knowledge_object.get("sections") or []:
        content = section.get("coverage_content") or []
        if isinstance(content, str):
            content = [content]
        architectural = section.get("id") in _ARCHITECTURE_SENTENCE_SECTIONS
        for item in content:
            _scan_text(str(item), describes_architecture=architectural)

    if not found:
        return knowledge_object

    seen: Dict[str, int] = {}
    deduped: List[str] = []
    for term in found:
        canon = _canonicalize_component_term(term)
        key = canon.lower()
        if key not in seen:
            seen[key] = len(deduped)
            deduped.append(canon)
        else:
            # The regex captures verbatim casing from wherever it first
            # matched — a tool named mid-sentence in lowercase ("...modify
            # terraform state manually...") can otherwise permanently win
            # the display slot over a later, properly-capitalized mention
            # ("Terraform is used for infrastructure provisioning...")
            # purely because of scan order. Prefer whichever form is
            # actually capitalized.
            existing = deduped[seen[key]]
            if existing[:1].islower() and canon[:1].isupper():
                deduped[seen[key]] = canon

    seen_sentences = set()
    deduped_sentences: List[str] = []
    for text in descriptive_sentences:
        key = text.lower()
        if key not in seen_sentences:
            seen_sentences.add(key)
            deduped_sentences.append(text)

    arch_section["_architecture_components"] = deduped
    if deduped_sentences:
        arch_section["_architecture_sentences"] = deduped_sentences

    diagram = build_architecture_flow_diagram(deduped)
    if diagram:
        arch_section["_architecture_diagram"] = diagram

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

    def _consider(text: str, label: str, value_text: str, always_eligible: bool) -> None:
        text = (text or "").strip()
        if not text or not (always_eligible or is_tribal_knowledge(text)):
            return
        normalized = _normalize_text(text)
        if normalized in seen_texts:
            return
        # A polished restatement of a sentence already captured from the raw
        # text is the same knowledge, not a second entry.
        if any(_is_near_duplicate(text, _dedup_normalize(r["Knowledge"])) for r in rows):
            return
        seen_texts.add(normalized)
        rows.append({"Knowledge": text, "Value": value_text, "Classification": label})

    for section_id, entry in section_content.items():
        label, value_text = _TRIBAL_SOURCE_LABELS.get(section_id, _TRIBAL_DEFAULT_LABEL)
        always_eligible = section_id in _TRIBAL_ALWAYS_ELIGIBLE_SECTIONS
        for sentence in _section_sentences(entry):
            text = (sentence or {}).get("text", "") if isinstance(sentence, dict) else ""
            _consider(text, label, value_text, always_eligible)

    # Also scan each section's (possibly LLM-polished) coverage_content. A
    # section's raw ['sentences'] and its coverage_content are populated by
    # two independent mechanisms that can disagree, so a sentence explicitly
    # framed as tribal knowledge can exist ONLY in the polished text —
    # confirmed live, where "Tribal knowledge, service bus backlog can
    # temporarily increase during large deployments." reached the document
    # but never the Tribal Knowledge digest that exists to collect it.
    # Bullet blobs are split first so one card holds one piece of knowledge.
    for section in knowledge_object.get("sections") or []:
        section_id = section.get("id")
        if section_id == TRIBAL_KNOWLEDGE_SECTION_ID:
            continue
        label, value_text = _TRIBAL_SOURCE_LABELS.get(section_id, _TRIBAL_DEFAULT_LABEL)
        content = section.get("coverage_content") or []
        if isinstance(content, str):
            content = [content]
        for item in split_bullet_blob([str(c) for c in content if str(c).strip()]):
            # Not `always_eligible` here: a whole section's polished prose
            # must earn its place in the digest by actually reading as
            # tribal knowledge, or every danger zone would be duplicated.
            _consider(item, label, value_text, always_eligible=False)

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
    mapped_sentence_total = 0
    for section in dynamic_schema:
        section_id = section.get("id")
        title = section.get("title") or section_id
        cov = coverage.get(section_id) or {}
        status = cov.get("status", "missing")
        sentence_count = int(cov.get("sentence_count", 0) or 0)
        mapped_sentence_total += sentence_count

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
            knowledge_items = (section_obj or {}).get("_architecture_components") or []
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

                # A section whose leaf fields are purely administrative
                # metadata (e.g. architecture_reference's doc link / last
                # updated / verified-by) can have real knowledge captured
                # elsewhere in the section (its own _architecture_components
                # digest) even when none of those admin fields were ever
                # discussed. Reporting only "0 of 3 fields" there reads as
                # "nothing was captured" when the opposite is true — split
                # the assessment into knowledge captured vs. metadata
                # discussed instead of collapsing both into one field count.
                if section.get("fields_role") == "metadata" and knowledge_items:
                    shown_items = ", ".join(knowledge_items[:8])
                    extra = len(knowledge_items) - 8
                    if extra > 0:
                        shown_items += f" (+{extra} more)"
                    assessment = (
                        f"{len(knowledge_items)} component(s) identified ({shown_items}); "
                        f"{filled_count} of {len(leaf_specs)} metadata field(s) discussed"
                    )
                    if missing_labels:
                        shown = ", ".join(missing_labels[:4])
                        assessment += f" (missing: {shown})."
                    else:
                        assessment += "."
                else:
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

    # Knowledge coverage (how many transcript facts were captured) is a
    # genuinely separate metric from the Domain/Coverage/Assessment matrix
    # above (template-field population) — §12's "must never be confused".
    # `mapped` is the number of sentences the classifier actually placed
    # into a real section; `deduplicated`/`unmapped` come from the same
    # substantive-unassigned-sentence split append_unmapped_findings_section
    # already computed (see its docstring) — read from the stats it stashed
    # on knowledge_object rather than recomputed here, so the two can never
    # drift apart. `lost` is an explicit self-check, not an aspiration: it's
    # only ever nonzero if facts_identified was computed independently of
    # mapped+deduplicated+unmapped and the two disagree, which would mean a
    # real accounting bug rather than something to paper over.
    dedup_stats = knowledge_object.pop("_dedup_stats", {}) or {}
    deduplicated = int(dedup_stats.get("deduplicated", 0) or 0)
    unmapped = int(dedup_stats.get("unmapped", 0) or 0)
    facts_identified = mapped_sentence_total + deduplicated + unmapped
    knowledge_coverage_summary = {
        "facts_identified": facts_identified,
        "mapped": mapped_sentence_total,
        "deduplicated": deduplicated,
        "unmapped": unmapped,
        "lost": facts_identified - (mapped_sentence_total + deduplicated + unmapped),
    }

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
        "_knowledge_coverage_summary": knowledge_coverage_summary,
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
