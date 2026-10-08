import re
from typing import Any, Dict, List, Optional

from .entities import build_entities
from .evidence import build_evidence
from .facts import build_facts
from .relationships import build_relationships
from section_rules import is_tribal_knowledge, _GREETING_ONLY_RE
from field_populator import PATTERN_EXTRACTORS, SYSTEM_NAME_STOPWORDS, _trim_name_capture, smart_title
from architecture_diagram import build_architecture_graph, describe_connections, render_architecture_svg
from component_catalog import display_name
from dialogue import is_gap_statement, is_field_candidate, states_something_before_gap
from coverage_topics import AFTER_REVIEW_SECTIONS, assess_topics


def _short_quote(text: str, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"
from renderers.blocks.common import split_bullet_blob

UNMAPPED_FINDINGS_SECTION_ID = "unmapped_findings"
UNMAPPED_FINDINGS_TITLE = "Additional Notes (Unmapped Findings)"
# A sentence below this many words is almost always a filler/transition
# ("Thank you.", "Okay so.") rather than a standalone fact worth surfacing.
# Generic length threshold — not keyed to any transcript's content.
_MIN_UNMAPPED_SENTENCE_WORDS = 4

# Session pleasantries carry no operational knowledge, but they are long
# enough to clear the word-count filter and distinct enough to survive
# deduplication, so they land in "Additional Notes (Unmapped Findings)" and
# get presented to the reader as a finding. A real generated PDF's entire
# Additional Notes section read: "Hi everyone, today I will be handing over
# the Azure Order Processing Platform." -- which tells an incoming engineer
# nothing and makes a genuine loss-reporting mechanism look like noise.
#
# Matched as whole phrases anchored at the start of a sentence so that real
# content is never dropped: "Thanks to the platform team for the runbook" is
# a fact about who owns what and must survive, whereas "Thanks everyone,
# that concludes the session" must not.
_PLEASANTRY_PATTERNS = (
    r"^(hi|hello|hey|good\s+(morning|afternoon|evening))\b",
    r"^(thanks|thank\s+you)\s+(everyone|all|folks|team)\b",
    r"^(welcome|glad)\s+(to|you|everyone)\b",
    r"^(let'?s|let\s+us)\s+(get\s+started|begin|start)\b",
)
_PLEASANTRY_RE = re.compile("|".join(_PLEASANTRY_PATTERNS), re.IGNORECASE)

# Closing remarks need a SECOND condition, not just the verb. "That
# concludes..." is only a pleasantry when what it concludes is the session:
# "That concludes the rollback if the canary thresholds are breached." is a
# real operational fact that the verb alone threw away. Found by probing the
# filter against phrasings it had not been written for, not by a test that
# merely re-stated the patterns.
_CLOSING_VERB_RE = re.compile(
    r"^(that|this)\s+(concludes|wraps\s+up|is\s+the\s+end\s+of)\b", re.IGNORECASE
)
_SESSION_NOUN_RE = re.compile(
    r"\b(handover|hand\s*over|session|kt|knowledge\s+transfer|walkthrough|"
    r"walk\s*through|meeting|call|presentation|briefing|demo)\b",
    re.IGNORECASE,
)


def _is_closing_remark(text: str) -> bool:
    """A closing verb applied to the session itself, not to a procedure."""
    return bool(_CLOSING_VERB_RE.search(text) and _SESSION_NOUN_RE.search(text))

# A pleasantry that also states real facts must be KEPT -- the filter exists
# to remove noise, and it is not worth losing a fact to tidy up a greeting.
#
# Length is the wrong test for that, and measuring it that way was a real
# bug: "Hi everyone, the platform processes 120,000 orders per day and runs
# on AKS." was short enough to be dropped, and "That concludes the rollback
# if the canary thresholds are breached." -- a genuine operational fact --
# matched the closing-remark pattern outright. Both were caught by probing
# the filter against phrasings it had not been written for.
#
# So the test is SUBSTANCE, not length: a sentence is only filler when it
# carries no number and names no technology. That reuses field_populator's
# curated tool vocabulary rather than a second list that would drift from it.
_PLEASANTRY_DIGIT_RE = re.compile(r"\d")


def _has_content_signal(text: str) -> bool:
    """True when a sentence carries substance a KT reader would want kept:
    any figure, or any named technology."""
    if _PLEASANTRY_DIGIT_RE.search(text or ""):
        return True
    try:
        # Imported lazily so this module never participates in an import
        # cycle with field_populator.
        from field_populator import PATTERN_EXTRACTORS
    except Exception:
        return False
    tools = PATTERN_EXTRACTORS.get("tools")
    return bool(tools and tools.search(text or ""))


def _is_session_pleasantry(text: str) -> bool:
    """True for opening/closing remarks that carry no operational knowledge."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    # The classifier's own opener/closer test: "That concludes the GCP data
    # platform." was dropped from the sections, then put back under
    # Additional Notes by the completeness check.
    if _GREETING_ONLY_RE.match(stripped):
        return True
    if not (_PLEASANTRY_RE.search(stripped) or _is_closing_remark(stripped)):
        return False
    return not _has_content_signal(stripped)


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
    overview = coverage.get("system_overview", {})
    content_list = overview.get("content", [])
    if isinstance(content_list, str):
        content_list = [content_list]
    combined = " ".join(str(c) for c in content_list)

    if populated_fields:
        sys_field = (populated_fields.get("system_overview", {}) or {}).get("system_name", {})
        val = sys_field.get("value")
        if isinstance(val, str) and val.strip():
            # A name the speaker cased themselves ("MedRelay", "CorePay")
            # keeps that casing, and so do acronyms ("GCP Data and Machine
            # Learning"); .title() made "Medrelay" and "Gcp Data And ...".
            stripped = val.strip()
            has_own_casing = any(ch.isupper() for word in stripped.split() for ch in word[1:])
            normalized = stripped if has_own_casing else smart_title(stripped, combined)
            if len(normalized.split()) <= 6:
                return normalized
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
            return smart_title(name, combined)
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
            # The section's transcript sentences, verbatim. coverage_content
            # may be an LLM rewrite; the rendering safety net checks these.
            "_source_sentences": [
                str(s.get("text") or "").strip()
                for s in (section_cov.get("sentences") or [])
                if isinstance(s, dict) and str(s.get("text") or "").strip()
            ],
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
        if _is_session_pleasantry(sentence):
            # Same carve-out as the classifier-reported pass above: an
            # opening or closing remark is not a lost fact, and surfacing it
            # here would re-add exactly what that filter just removed.
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
        and not _is_session_pleasantry(s.text)
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


def _replace_superseded(value: Any, superseded: Dict[str, Dict[str, Optional[str]]]) -> Any:
    """A field value naming a tool the session says was replaced or dropped
    gets the current tool (or is marked as no longer used). Only values that
    ARE tool names are touched; a quoted sentence stays the speaker's words."""
    def _swap(item: str) -> Optional[str]:
        info = superseded.get(item.strip().lower())
        if not info:
            return item
        return info["new"] or None

    if isinstance(value, list):
        out: List[Any] = []
        for item in value:
            new = _swap(item) if isinstance(item, str) else item
            if new is not None and new not in out:
                out.append(new)
        return out
    if isinstance(value, str):
        if value.strip().lower() in superseded:
            return _swap(value) or f"{value.strip()} (no longer used, per the KT session)"
        if "," in value:  # key_technologies is a comma-separated tool list
            parts = [p.strip() for p in value.split(",")]
            if any(p.lower() in superseded for p in parts):
                kept: List[str] = []
                for p in parts:
                    new = _swap(p)
                    if new and new not in kept:
                        kept.append(new)
                return ", ".join(kept)
    return value


def apply_conflicts(knowledge_object: Dict[str, Any], coverage: Dict[str, Any]) -> Dict[str, Any]:
    """Apply what the session said about change and contradiction (P0-5,
    conflicts.py) to the knowledge object before it is rendered:

    - tools the speaker says were replaced ("we moved off PagerDuty") are
      replaced by the current one in tool-valued fields and the technology
      summary;
    - pairs of statements that permit and forbid the same thing are stored
      on knowledge_object["_conflicts"] for the section and the coverage
      page to show.
    """
    from conflicts import find_contradictions, find_superseded_tools

    sentence_section: Dict[str, str] = {}
    texts: List[str] = []
    for sid, cov in (coverage or {}).items():
        for s in (cov or {}).get("sentences") or []:
            text = (s or {}).get("text", "") if isinstance(s, dict) else ""
            # Classification units can hold several sentences (semantic
            # chunking joined "Deploying on Fridays is fine for us. Never
            # deploy on Fridays..."); contradictions are found per sentence.
            for part in re.split(r"(?<=[.!?])\s+", text.strip()):
                if part.strip():
                    texts.append(part)
                    sentence_section.setdefault(part, sid)

    superseded = find_superseded_tools(texts)
    if superseded:
        def _walk(fields: Any) -> None:
            if not isinstance(fields, dict):
                return
            for entry in fields.values():
                if isinstance(entry, dict) and "value" in entry:
                    new_value = _replace_superseded(entry.get("value"), superseded)
                    if new_value != entry.get("value"):
                        entry["value"] = new_value
                        entry["superseded"] = sorted(info["old"] for info in superseded.values())
                elif isinstance(entry, dict):
                    _walk(entry)

        for section in knowledge_object.get("sections") or []:
            _walk(section.get("fields"))
        knowledge_object["_superseded_tools"] = list(superseded.values())

    knowledge_object["_conflicts"] = [
        {"section_id": sentence_section.get(a), "a": a, "b": b} for a, b in find_contradictions(texts)
    ]
    return knowledge_object


def attach_conflict_warnings(rendered_sections: List[Dict[str, Any]], knowledge_object: Dict[str, Any]) -> None:
    """Show each contradiction found by apply_conflicts() as a warning in the
    section where it was said."""
    by_section: Dict[str, List[str]] = {}
    for c in knowledge_object.get("_conflicts") or []:
        by_section.setdefault(c.get("section_id"), []).append(f"“{c['a']}” but also “{c['b']}”")
    for sec in rendered_sections or []:
        warnings = by_section.get(sec.get("section_id"))
        if warnings:
            sec.setdefault("blocks", []).insert(0, {
                "type": "WarningBlock",
                "title": "Possible conflict — confirm with the outgoing owner",
                "warnings": warnings,
            })


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
        tool = _canonicalize_component_term(tool)
        key = tool.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(tool)
    deduped = _drop_subsumed_components(deduped)

    entry = overview_section.setdefault("fields", {}).setdefault(
        "key_technologies", {"id": "key_technologies", "label": "Key Technologies", "type": "text", "confidence": 0.6, "source": "cross_section", "evidence": []}
    )
    entry["value"] = ", ".join(deduped)
    if entry.get("source", "unfilled") == "unfilled":
        # An existing unfilled entry kept source "unfilled" while gaining a
        # value, so the coverage matrix listed Key Technologies as missing
        # right after the Technology summary had rendered it.
        entry["source"] = "cross_section"
        entry["confidence"] = 0.6

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
    # AWS acronyms that name exactly one product. (WAF/KMS/OpenSearch are
    # left alone: the bare word is not AWS-specific, and the vendor-
    # qualified form is collapsed by _drop_subsumed_components instead.)
    "alb": "Application Load Balancer",
    "eks": "Amazon EKS",
    "msk": "Amazon MSK",
    "route53": "Route 53",
    "route 53": "Route 53",
    "node.js": "Node.js",
    "nodejs": "Node.js",
}

# A bare name is dropped when the same list also holds it with a vendor
# prefix or an engine qualifier ("Aurora" + "Aurora PostgreSQL", "WAF" +
# "AWS WAF"), or when it is the generic noun for a more specific entry
# ("Load Balancer" + "Application Load Balancer"). Listing both presented
# one component as two. Deliberately narrow: "Vault" is NOT subsumed by
# "Azure Key Vault" (different products).
_VENDOR_PREFIXES = ("amazon", "aws", "azure", "google", "gcp", "hashicorp")
_ENGINE_SUFFIXES = ("postgresql", "mysql", "postgres")
_GENERIC_COMPONENT_NOUNS = {"load balancer"}


def _drop_subsumed_components(components: List[str]) -> List[str]:
    lowered = [c.lower() for c in components]
    kept: List[str] = []
    for comp, low in zip(components, lowered):
        subsumed = False
        for other in lowered:
            if other == low:
                continue
            if any(other == f"{p} {low}" for p in _VENDOR_PREFIXES):
                subsumed = True
            elif any(other == f"{low} {s}" for s in _ENGINE_SUFFIXES):
                subsumed = True
            elif low in _GENERIC_COMPONENT_NOUNS and other.endswith(" " + low):
                subsumed = True
            if subsumed:
                break
        if not subsumed:
            kept.append(comp)
    # Bare "Kubernetes" (or "K8s") next to a managed Kubernetes product is
    # that product, not a second platform ("Google Kubernetes Engine,
    # Kubernetes" in the component list).
    if any(c.lower() in _MANAGED_KUBERNETES for c in kept):
        kept = [c for c in kept if c.lower() not in ("kubernetes", "k8s")]
    return kept


_MANAGED_KUBERNETES = frozenset({"amazon eks", "eks", "azure kubernetes service", "aks", "google kubernetes engine",
                                 "gke", "openshift", "rancher"})


def _canonicalize_component_term(term: str) -> str:
    alias = _CANONICAL_TERM_ALIASES.get(term.strip().lower())
    if alias:
        return alias
    # Products from component_catalog.py display under their catalog name
    # ("ec2" -> "Amazon EC2", "opentofu" -> "OpenTofu").
    return display_name(term) or term


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
    - `_architecture_graph`, `_architecture_svg`, `_architecture_connections`:
      the High-Level Architecture picture (architecture_diagram.py):
      components placed by what they are, joined only where a sentence
      states the connection, and those connections with their sentences.
      Absent when fewer than two components were named.
    """
    arch_section = _find_section(knowledge_object, "architecture_reference")
    if not arch_section:
        return knowledge_object

    tools_pattern = PATTERN_EXTRACTORS["tools"]
    found: List[str] = []
    descriptive_sentences: List[str] = []
    # For the diagram's stated connections: every sentence that names a
    # component (from any section), and the spellings each was named by.
    connection_sentences: List[str] = []
    spellings: Dict[str, set] = {}

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
            for match in matches:
                spellings.setdefault(_canonicalize_component_term(match).lower(), set()).add(match.lower())
            for part in split_bullet_blob([text]) or [text]:
                connection_sentences.extend(p.strip() for p in _SENTENCE_SPLIT_RE.split(part) if p.strip())

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
    deduped = _drop_subsumed_components(deduped)

    # Architecture Details renders these verbatim, so duplication here is
    # duplication in the PDF. Exact lowercase matching was not enough: the
    # list mixes RAW sentences with the polish pass's BULLET BLOBS, which
    # restate the same facts inside one string. A real generated document
    # showed "The frontend uses Angular." and "Infrastructure is provisioned
    # with Bicep." twice each, plus "...Azure Kubernetes Service, AKS." next
    # to "...Azure Kubernetes Service (AKS)." -- different by punctuation
    # alone, so exact matching kept both.
    #
    # Split blobs into their constituent sentences first (so a blob's items
    # are comparable to raw sentences at all), then compare on content words
    # via the same near-duplicate rule used for Additional Notes.
    expanded: List[str] = []
    for text in descriptive_sentences:
        parts = split_bullet_blob([text]) or [text]
        for part in parts:
            part = part.strip()
            if part:
                expanded.append(part)

    seen_sentences = set()
    deduped_sentences: List[str] = []
    for text in expanded:
        key = _dedup_normalize(text)
        if key in seen_sentences:
            continue
        if any(_is_near_duplicate(text, kept) for kept in deduped_sentences):
            continue
        seen_sentences.add(key)
        deduped_sentences.append(text)

    arch_section["_architecture_components"] = deduped
    if deduped_sentences:
        arch_section["_architecture_sentences"] = deduped_sentences

    # The High-Level Architecture picture: components placed by what they
    # are, joined only where a sentence states the connection (P1-9).
    graph = build_architecture_graph(
        deduped,
        list(dict.fromkeys(connection_sentences)),
        {name: sorted(names) for name, names in spellings.items()},
        knowledge_object.get("system_name"),
    )
    if graph:
        arch_section["_architecture_graph"] = graph
        arch_section["_architecture_svg"] = render_architecture_svg(graph)
        arch_section["_architecture_connections"] = describe_connections(graph)

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
        if f.get("dynamic"):
            # Dynamically-added fields (schema_generator.py's
            # TECH_STACK_FIELD_ADDITIONS) exist BECAUSE the transcript
            # mentioned the technology -- "Cache Layer" is created only
            # because Redis was named. Counting such a field as a gap is
            # circular: the transcript conjures the field and is then
            # marked down for not elaborating on it. A real generated PDF
            # reported "missing: Cache Layer" on a KT whose own Technology
            # Summary, on the same page, read "Cache | Redis".
            #
            # Template fields are different and stay in scope: they
            # represent what a KT *should* cover regardless of what was
            # said, so their absence is genuine news.
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
    titles = {s.get("id"): (s.get("title") or s.get("id")) for s in dynamic_schema}
    all_texts = [(sid, s.get("text", "")) for sid, c in (coverage or {}).items()
                 for s in ((c or {}).get("sentences") or []) if isinstance(s, dict)
                 and is_field_candidate(s.get("text", ""))]

    def other_texts_for(section_id):
        return [(sid, t) for sid, t in all_texts if sid != section_id]

    mentioned_elsewhere: Dict[str, List[Dict[str, str]]] = {}

    def elsewhere_quote(section_id, topic_label):
        """(section, sentence) for the most direct statement of the topic
        elsewhere: the shortest matching sentence ("Secret Manager handles
        secrets" over a long day-one list that also names Secret Manager)."""
        from coverage_topics import _COMPILED as topic_patterns
        rx = next((r for label, _, r in topic_patterns.get(section_id, []) if label == topic_label), None)
        if rx is None:
            return None
        hits = [(len(t), sid, t) for sid, t in all_texts if sid != section_id and rx.search(t)]
        return min(hits)[1:] if hits else None

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

        # "Do we have a DR plan? No, there is no DR plan" is a gap stated in
        # the session, not coverage (dialogue.py). A section discussed only
        # to say so is Missing, with the speaker's words as the reason.
        texts = [s.get("text", "") for s in (cov.get("sentences") or []) if isinstance(s, dict)]
        stated_gaps = [t for t in texts if is_gap_statement(t)]
        only_stated_gaps = bool(texts) and len(stated_gaps) == len(texts) and \
            not any(states_something_before_gap(t) for t in stated_gaps)
        if only_stated_gaps:
            bucket = "Missing"

        section_obj = _find_section(knowledge_object, section_id)
        field_objects = (section_obj or {}).get("fields") or {}
        knowledge_items = (section_obj or {}).get("_architecture_components") or []
        structured = cov.get("_structured") if isinstance(cov.get("_structured"), dict) else {}
        own_texts = [t for t in texts if is_field_candidate(t)]
        topic_states = assess_topics(section_id, field_objects, own_texts, other_texts_for(section_id), structured)
        if topic_states and not own_texts:
            # A section the session never discussed as such, though one of its
            # topics came up elsewhere ("Secret Manager handles secrets" under
            # Deployment): keep the sentence so the empty section can point to it.
            for label, state, where in topic_states:
                if state == "elsewhere" and where:
                    found = elsewhere_quote(section_id, label)
                    if found:
                        found_sid, quote = found
                        mentioned_elsewhere.setdefault(section_id, []).append(
                            {"topic": label, "section_id": found_sid,
                             "section_title": titles.get(found_sid, found_sid), "quote": quote})

        if section_id in AFTER_REVIEW_SECTIONS:
            # Sign-off happens when the document is reviewed, not in the
            # recorded session; listing it as a session gap was misleading.
            bucket = "After review"
            assessment = "Recorded when the document is reviewed and signed, not during the KT session."
        elif only_stated_gaps:
            bucket = "Missing"
            assessment = f"Discussed, but stated as not in place: “{_short_quote(stated_gaps[0])}”"
            gaps.append(f"{title}: stated as not in place (“{_short_quote(stated_gaps[0])}”)")
        elif topic_states is not None:
            # Coverage per expected topic, from what the session actually
            # said (coverage_topics.py), not from which template fields
            # happened to be filled.
            if section_id == "architecture_reference" and knowledge_items:
                topic_states = [("Components and data flow", "captured", None)] + topic_states
            total = len(topic_states)
            covered = [t for t in topic_states if t[1] != "missing"]
            elsewhere = [t for t in topic_states if t[1] == "elsewhere"]
            missing = [t[0] for t in topic_states if t[1] == "missing"]
            if not covered and not (texts or sentence_count):
                bucket = "Missing"
                assessment = "Not covered in the KT session."
                gaps.append(title)
            else:
                # A section with its own statements is at least Partial, even
                # when none of its expected topics matched: Danger Zones that
                # listed four danger zones was reported Missing.
                bucket = "Strong" if not missing else ("Partial" if covered or own_texts else "Missing")
                if not missing:
                    assessment = "Covered" if total == 1 else f"All {total} topics covered"
                elif not covered and own_texts:
                    assessment = f"{len(own_texts)} statement(s) captured, but not the expected topics"
                else:
                    assessment = f"{len(covered)} of {total} topics covered"
                if elsewhere:
                    where = sorted({titles.get(w, w) for _, _, w in elsewhere})
                    assessment += f" ({len(elsewhere)} under {', '.join(where)})"
                assessment += f"; not discussed: {', '.join(missing)}." if missing else "."
                if missing:
                    gaps.append(f"{title}: {', '.join(missing)}")
        else:
            if not (texts or sentence_count):
                bucket = "Missing"
                assessment = "Not covered in the KT session."
                gaps.append(title)
            else:
                bucket = "Partial"
                assessment = f"{len(texts) or sentence_count} statement(s) captured."

        if stated_gaps and not only_stated_gaps:
            assessment = f"{assessment} Stated gap: “{_short_quote(stated_gaps[0])}”"
            gaps.append(f"{title}: “{_short_quote(stated_gaps[0])}”")

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
    for conflict in knowledge_object.get("_conflicts") or []:
        gaps.append(
            f"Possible conflict ({titles.get(conflict.get('section_id'), 'KT session')}): "
            f"“{_short_quote(conflict['a'])}” vs “{_short_quote(conflict['b'])}” — confirm with the outgoing owner"
        )

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
    knowledge_object["_mentioned_elsewhere"] = mentioned_elsewhere
    return knowledge_object


MENTIONED_ELSEWHERE_TITLE = "Mentioned elsewhere in the KT"


def attach_elsewhere_mentions(rendered_sections: List[Dict[str, Any]], knowledge_object: Dict[str, Any]) -> None:
    """A section that would only say "not covered" points to what the session
    did say about its topics under another section ("Secrets management,
    under Deployment: 'Secret Manager handles secrets.'"), so the reader
    sees the coverage matrix's "covered elsewhere" in the section itself.
    Run after duplicate removal: these are quotes, not a second home."""
    from renderers.blocks.common import MENTIONED_ELSEWHERE_MESSAGE, NOT_COVERED_MESSAGE

    mentions = knowledge_object.get("_mentioned_elsewhere") or {}
    for section in rendered_sections or []:
        items = mentions.get(section.get("section_id"))
        blocks = section.get("blocks") or []
        only_not_covered = blocks and all(
            b.get("type") == "NarrativeBlock" and b.get("paragraphs") == [NOT_COVERED_MESSAGE] for b in blocks)
        if not items or not only_not_covered:
            continue
        lines = [f"{m['topic']}, under {m['section_title']}: “{_short_quote(m['quote'])}”" for m in items]
        blocks[0]["paragraphs"] = [MENTIONED_ELSEWHERE_MESSAGE]
        blocks.append({"type": "ChecklistBlock", "title": MENTIONED_ELSEWHERE_TITLE, "items": lines})


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
        # A question or "I'm not sure who" is not a reference to act on.
        if text and is_gap_statement(text) or text.rstrip().endswith("?"):
            continue
        if text and any(kw.lower() in text.lower() for kw in keywords):
            return text
    return None


QUICK_REFERENCE_COLUMNS = ["Situation", "What to do", "Source"]
_QR_MAX_FAILURES = 6
_QR_MAX_CAUTIONS = 6
# Digests and the coverage report restate other sections; the card reads
# real sections only.
_QR_SKIP_SECTIONS = frozenset({"quick_reference", "tribal_knowledge", "kt_coverage"})

_QR_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(A-Z0-9])")
_QR_LEAD_RE = re.compile(r"^(?:and|so|but|also|plus|then|by the way|one more thing|oh and)\b[,\s]*", re.IGNORECASE)
_ALERT_CUE_RE = re.compile(
    r"\b(alerts?|alerting|alarms?|pages? us|paged|pager|on-?call|threshold|drops? below|goes? (?:above|over)|exceeds?|"
    r"dashboard to (?:watch|check|open)|first (?:thing|check|step)s?)\b", re.IGNORECASE)
_ROLLBACK_CUE_RE = re.compile(
    r"\b(roll(?:ed|ing)?[ -]?back|revert(?:ed|ing)?|redeploy(?:ing)? the previous|previous (?:version|release|"
    r"task definition|image|build|deployment|revision))\b", re.IGNORECASE)
# An instruction not to do something, at the start of a sentence or clause
# ("Never delete…", "…, so don't run…"); "We never agreed…" is not one.
_PROHIBITION_RE = re.compile(
    r"(?:^|[,;:]\s*|\b(?:and|so|but|also|please)\s+)(never|don't|do not|avoid|must not|mustn't|should not|"
    r"shouldn't|under no circumstances)\b", re.IGNORECASE)
# The same, but only as the sentence's opening instruction: used to find
# never-do rules that were filed under another section.
_OPENING_PROHIBITION_RE = re.compile(
    r"^(?:(?:and|but|also|so|please)\s+)?(?:never|do not|don't|under no circumstances)\b"
    r"(?!\s+(?:worry|hesitate|mind|forget))", re.IGNORECASE)
_CAUTION_RE = re.compile(
    r"\b(careful|dangerous|danger|risky|risk|fragile|break|breaks|corrupt\w*|irreversibl\w*|data loss|lose|outage|"
    r"only copy|critical|impact\w*|without (?:telling|asking|approval))\b", re.IGNORECASE)
_FREEZE_CUE_RE = re.compile(
    r"\b(freeze|blackout|no (?:deploy\w*|changes|releases)|deploy\w* on fridays?|avoid deploying|do not deploy|"
    r"don't deploy|only (?:run|deploy|release)\w* on)\b", re.IGNORECASE)
_ESCALATION_CUE_RE = re.compile(r"\b(escalat\w*|page\s+(?!us\b)\w+|contact\s+\w+|reach out to|call\s+[A-Z]\w+|"
                                r"ping\s+[A-Z]\w+)", re.IGNORECASE)
# "One recurring problem is X", "The most common production issue is X",
# "Another issue is X, which …; fix …".
_FAILURE_INTRO_RE = re.compile(
    r"^(?:(?:one|another|a|an|the|our)\s+)?(?:[\w-]+\s+){0,4}?"
    r"(?:problem|issue|failure(?:\s+mode)?|error|incident|gotcha|outage)s?\s+"
    r"(?:is|was|we\s+(?:see|hit|get)(?:\s+is)?|that\s+(?:comes\s+up|happens)\s+is)\s+"
    r"(?P<what>.+?)\s*(?:[;,]\s*(?P<rest>.+))?$", re.IGNORECASE)
# The subject-first form: "Airflow DAG failure is another common issue",
# "Schema change is another important failure scenario because ...".
_FAILURE_NAMED_RE = re.compile(
    r"^(?P<what>.+?)\s+(?:is|was|are)\s+(?:another|a|an|one|the|our)\s+"
    r"(?:(?:most|other|common|known|recurring|frequent|important|big|main|typical|classic|nasty)\s+)*"
    r"(?:production\s+)?(?:problem|issue|failure(?:\s+(?:mode|scenario|case))?|gotcha|incident)s?\b"
    r"(?:[,;]?\s*(?:because|since|as|and)\s+(?P<rest>.+))?$", re.IGNORECASE)
# A "what" that is only the failure words themselves ("another common issue").
_FAILURE_WORD_ONLY_RE = re.compile(
    r"^(?:another|a|an|one|the)?\s*(?:(?:most|other|common|known|recurring|important)\s+)*"
    r"(?:problem|issue|failure|scenario)s?$", re.IGNORECASE)
_NOT_A_FAILURE_RE = re.compile(r"\b(?:non-?issue|not\s+(?:an?\s+)?(?:issue|problem)|no\s+problem)\b", re.IGNORECASE)
_FAILURE_WORD_RE = re.compile(r"\b(?:problem|issue|failure|outage|incident|backlog|lag|throttl\w*|crash\w*)\b",
                              re.IGNORECASE)
# "There was a major outage last year caused by …"
_PAST_INCIDENT_RE = re.compile(
    r"^there\s+(?:was|were|has\s+been|have\s+been)\s+(?:(?:a|an|one)\s+)?"
    r"(?P<what>.*\b(?:outage|incident|failure|problem|issue)s?\b.*)$", re.IGNORECASE)
# A runbook step: "When invoices get stuck, check whether …".
_RUNBOOK_RE = re.compile(
    r"^(?:when|if|whenever)\s+(?P<symptom>[^,;]{4,120}?),\s*(?:then\s+|first\s+)?"
    r"(?P<action>(?:check|restart|run|scale|rotate|renew|redeploy|roll\s+back|switch|verify|look\s+at|open|clear|"
    r"flush|drain|increase|bump|fail\s+over|re-?run|recreate|delete\s+the\s+pod|kill)\b.+)$", re.IGNORECASE)
_FOLLOW_UP_ACTION_RE = re.compile(
    r"^(?:then\s+)?(?:check|restart|run|scale|rotate|renew|redeploy|roll\s+back|switch|verify|clear|flush|drain|"
    r"increase|bump|fail\s+over|re-?run|recreate)\b", re.IGNORECASE)
_FIX_CUE_RE = re.compile(
    r"\b(fix|fixed|resolve\w*|workaround|restart\w*|rotate\w*|switch\w*|scale\w*|clear\w*|flush\w*|re-?run\w*|"
    r"redeploy\w*|roll\w* back|increase\w*|bump\w*|purge\w*|recreate\w*|failover|fail over|archive\w*|verify|check)\b",
    re.IGNORECASE)
_NO_FIX_STATED = "No fix was stated in the KT. Ask the outgoing owner."


def _qr_sentences(section: Optional[Dict[str, Any]], skip=None) -> List[str]:
    """The section's sentences that can be acted on: no questions, no
    statements that something is missing or unknown, and none that `skip`
    rejects (superseded or contradicted, see append_quick_reference_section)."""
    if not section:
        return []
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    out: List[str] = []
    # A polished bullet list ("- Danger zones include ...\n- The danger
    # zones are ...") is split into its items, without the bullet marks.
    for item in split_bullet_blob([str(c) for c in content if str(c).strip()]) or []:
        item = _QR_BULLET_RE.sub("", str(item).strip())
        for sentence in _QR_SENTENCE_RE.split(item):
            sentence = sentence.strip()
            if sentence and not sentence.endswith("?") and not is_gap_statement(sentence) \
                    and not (skip and skip(sentence)):
                out.append(sentence)
    return out


_QR_BULLET_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s+")


def _qr_key(text: str) -> str:
    return re.sub(r"\W+", " ", _tidy(text).lower()).strip()


def _tidy(text: str) -> str:
    """Drop a spoken lead-in ("And …", "So …") and start with a capital."""
    text = _QR_LEAD_RE.sub("", text.strip())
    return text[:1].upper() + text[1:] if text else text


def _clause(text: str) -> str:
    """The rest of a failure sentence after its subject ("which silently
    stop …", "so we archive …"), as a sentence of its own."""
    return _tidy(re.sub(r"^(?:which|that|so)\s+", "", text.strip(), flags=re.IGNORECASE))


def _known_failures(section: Optional[Dict[str, Any]], skip=None) -> List[tuple]:
    """(symptom, what to do) pairs from Common Failures: the structured
    list when the LLM produced one, otherwise the speaker's own sentences,
    grouped into one item per failure they introduced."""
    structured = (section.get("_structured") or {}).get("failures") if section else None
    if isinstance(structured, list) and structured:
        pairs = []
        for item in structured:
            if isinstance(item, dict) and str(item.get("symptom") or "").strip():
                fix = str(item.get("fix") or "").strip()
                check = str(item.get("first_checks") or "").strip()
                if not fix and check:
                    # What the KT did say: where to look first.
                    fix = f"First check: {check.rstrip('.')}. No fix was stated in the KT; ask the outgoing owner."
                pairs.append((str(item["symptom"]).strip(), fix or _NO_FIX_STATED))
        return pairs[:_QR_MAX_FAILURES]

    items: List[Dict[str, Any]] = []
    for sentence in _qr_sentences(section, skip):
        bare = _QR_LEAD_RE.sub("", sentence).rstrip(".")
        if _NOT_A_FAILURE_RE.search(bare):
            continue                      # "Another non-issue is ...": said not to be a problem
        named = _FAILURE_NAMED_RE.match(bare)
        intro = None if named else _FAILURE_INTRO_RE.match(bare)
        if intro and _FAILURE_WORD_ONLY_RE.match(intro.group("what")):
            intro = None                  # "X is another common issue" read backwards
        past = _PAST_INCIDENT_RE.match(bare)
        runbook = _RUNBOOK_RE.match(bare)
        if named:
            # Its "because ..." is why it matters, not what to do about it.
            items.append({"what": named.group("what"), "parts": []})
        elif intro:
            items.append({"what": intro.group("what"), "parts": [intro.group("rest")] if intro.group("rest") else []})
        elif past:
            items.append({"what": past.group("what"), "parts": []})
        elif runbook and not (items and not items[-1]["parts"]):
            items.append({"what": runbook.group("symptom"), "parts": [runbook.group("action")]})
        elif items and not (_FAILURE_WORD_RE.search(bare) and not _FIX_CUE_RE.search(bare)):
            # A sentence that brings up another problem is not this one's fix.
            items[-1]["parts"].append(sentence)
        elif _FIX_CUE_RE.search(sentence):
            items.append({"what": None, "parts": [sentence]})
    pairs = []
    for item in items:
        action = " ".join(_clause(p).rstrip(".") + "." for p in item["parts"] if p and p.strip())
        if item["what"]:
            situation = _tidy(item["what"].rstrip("."))
        elif re.match(r"^(?:check|start\s+with|look\s+at|first|for\s+[^,]{2,60},\s*(?:first\s+)?check)\b", action,
                      re.IGNORECASE):
            situation = "Where to look first"     # triage steps, not a named failure
        else:
            situation = "Known failure"
        pairs.append((situation, action or _NO_FIX_STATED))
    return pairs[:_QR_MAX_FAILURES]


def _runbook_steps(section: Optional[Dict[str, Any]], skip=None) -> List[tuple]:
    """"When X, check/restart …" steps (and the instruction right after
    one) wherever they were filed."""
    steps: List[tuple] = []
    sentences = _qr_sentences(section, skip)
    for i, sentence in enumerate(sentences):
        match = _RUNBOOK_RE.match(_QR_LEAD_RE.sub("", sentence).rstrip("."))
        if not match:
            continue
        action = _tidy(match.group("action")).rstrip(".") + "."
        if i + 1 < len(sentences) and _FOLLOW_UP_ACTION_RE.match(_QR_LEAD_RE.sub("", sentences[i + 1])):
            action += " " + _tidy(sentences[i + 1])
        steps.append((_tidy(match.group("symptom")), action))
    return steps


def append_quick_reference_section(knowledge_object: Dict[str, Any]) -> Dict[str, Any]:
    """The incident card: what to check when an alert fires, how to fix the
    known failures, how to roll back, who to page, and what never to do.

    Every line is a field value or the speaker's own sentence (a spoken
    lead-in such as "And" dropped), and names the section it came from in
    the Source column; nothing is written that the KT did not say, except
    that a known failure without a stated fix says so. Questions and "we
    don't know" statements are never quoted as guidance. Runbook steps and
    never-do rules are also picked up from other sections, because a
    misfiled step is still the step someone needs during an incident.
    """
    rows: List[Dict[str, str]] = []
    used: set = set()
    used_words: List[set] = []
    real_sections = [s for s in knowledge_object.get("sections") or [] if s.get("id") not in _QR_SKIP_SECTIONS]

    # What the session corrected or contradicted (apply_conflicts, P0-5) is
    # never quoted as an instruction: a sentence naming a tool the speaker
    # said was replaced is dropped (the correction itself is kept), and each
    # contradiction becomes one "confirm first" row instead of two orders.
    superseded = [t for t in knowledge_object.get("_superseded_tools") or [] if isinstance(t, dict) and t.get("old")]
    conflicts = [c for c in knowledge_object.get("_conflicts") or [] if c.get("a") and c.get("b")]
    conflicted = {_qr_key(c["a"]) for c in conflicts} | {_qr_key(c["b"]) for c in conflicts}

    def stale(sentence: str) -> bool:
        if _qr_key(sentence) in conflicted:
            return True
        return any(re.search(r"\b" + re.escape(t["old"]) + r"\b", sentence, re.IGNORECASE)
                   and _qr_key(sentence) != _qr_key(t.get("quote") or "") for t in superseded)

    def title_of(section: Optional[Dict[str, Any]], fallback: str) -> str:
        return str((section or {}).get("title") or fallback)

    def add(situation: str, action: Any, section: Optional[Dict[str, Any]], fallback_title: str) -> None:
        text = _join_if_list(action)
        if not text:
            return
        key = re.sub(r"\W+", " ", _tidy(text).lower()).strip()
        words = set(_content_words(text))
        if _NO_FIX_STATED.split(".")[0].lower() in key:
            # Each unfixed failure keeps its own row: the text is the same.
            key = f"{situation.lower()} | {key}"
            words = set(_content_words(f"{situation} {text}"))
        # The same statement said twice in different words ("Danger zones
        # include X, Y, Z outside ArgoCD" / "The danger zones are X, Y, Z")
        # is one row.
        if key in used or (len(words) >= 4 and any(words <= other for other in used_words)):
            return
        used.add(key)
        used_words.append(words)
        rows.append({"Situation": situation, "What to do": _tidy(text), "Source": title_of(section, fallback_title)})

    def first_matching(section, pattern, limit=2) -> Optional[str]:
        hits = [_tidy(s) for s in _qr_sentences(section, stale) if pattern.search(s)]
        return " ".join(hits[:limit]) or None

    monitoring = _find_section(knowledge_object, "monitoring_observability")
    add("An alert fires", _field_value(monitoring, "first_response_steps") or first_matching(monitoring, _ALERT_CUE_RE),
        monitoring, "Monitoring")

    failures = _find_section(knowledge_object, "common_failures")
    known = _known_failures(failures, stale)
    for symptom, fix in known:
        add(symptom, fix, failures, "Common Failures")
    for section in real_sections:
        if section.get("id") in ("common_failures", "deployment_and_rollback") or len(known) >= _QR_MAX_FAILURES:
            continue
        for symptom, action in _runbook_steps(section, stale):
            if len(known) < _QR_MAX_FAILURES:
                add(symptom, action, section, "")
                known.append((symptom, action))

    # Rollback: the speaker's own "how", plus trigger, approval and target
    # time when those fields were filled.
    deployment = _find_section(knowledge_object, "deployment_and_rollback")
    rollback_parts = [first_matching(deployment, _ROLLBACK_CUE_RE)]
    for label, field_id in (("Trigger", "rollback_trigger"), ("Approval", "rollback_approval"), ("Target time", "rollback_time")):
        value = _join_if_list(_field_value(deployment, "rollback_procedure", field_id))
        if value and not (rollback_parts[0] and value.lower() in rollback_parts[0].lower()):
            rollback_parts.append(f"{label}: {value.rstrip('.')}.")
    add("A release misbehaves (roll back)", " ".join(p for p in rollback_parts if p) or None, deployment, "Deployment")

    ownership = _find_section(knowledge_object, "ownership_escalation")
    escalation = _field_value(ownership, "escalation_chain")
    if not escalation:
        for sid in ("ownership_escalation", "monitoring_observability", "open_responsibilities", "day1_survival_checklist"):
            section = _find_section(knowledge_object, sid)
            escalation = first_matching(section, _ESCALATION_CUE_RE)
            if escalation:
                ownership = section
                break
    add("Who to page or escalate to", escalation, ownership, "Ownership & Escalation")

    danger = _find_section(knowledge_object, "danger_zones")
    cautions = []
    for sentence in _qr_sentences(danger, stale):
        if _PROHIBITION_RE.search(sentence):
            cautions.append(("Never", sentence, danger))
        elif _CAUTION_RE.search(sentence):
            cautions.append(("Be careful", sentence, danger))
    calendar = _find_section(knowledge_object, "known_bad_days")
    for section in real_sections:
        if section.get("id") == "danger_zones":
            continue
        for sentence in _qr_sentences(section, stale):
            if _OPENING_PROHIBITION_RE.match(sentence):
                cautions.append(("Never", sentence, section))
    for situation, sentence, section in cautions[:_QR_MAX_CAUTIONS]:
        add(situation, sentence, section, "Danger Zones")

    freeze = first_matching(calendar, _FREEZE_CUE_RE) or _field_value(calendar, "deployment_blackout_times")
    add("Do not deploy", freeze, calendar, "Operational Calendar")

    for c in conflicts:
        section = _find_section(knowledge_object, c.get("section_id") or "")
        add("Conflicting guidance — confirm first",
            f"“{_tidy(c['a'])}” but also “{_tidy(c['b'])}” Ask the outgoing owner before acting on either.",
            section, "KT session")

    # "If you are unsure ... contact X" guidance can be classified into any
    # of several sections (open responsibilities, ownership, danger zones);
    # look in all of them rather than only the one a past transcript used.
    for sid in ("open_responsibilities", "ownership_escalation", "danger_zones", "day1_survival_checklist"):
        section = _find_section(knowledge_object, sid)
        value = _first_matching_paragraph(section, "unaware", "unsure", "not sure", "in doubt")
        if value:
            add("Production activity unclear", value, section, "Ownership & Escalation")
            break

    overview = _find_section(knowledge_object, "system_overview")
    add("System unavailable (impact)", _field_value(overview, "impact_if_down", "what_breaks"), overview, "System Overview")

    dr = _find_section(knowledge_object, "disaster_recovery")
    rto, rpo = _field_value(dr, "rto_metric"), _field_value(dr, "rpo_metric")
    if rto or rpo:
        parts = [f"RTO {rto}" if rto else "", f"RPO {rpo}" if rpo else ""]
        add("Disaster recovery", "Recovery objectives: " + ", ".join(p for p in parts if p) + ".", dr, "Disaster Recovery")

    add("Planning a production change", _field_value(deployment, "deployment_window"), deployment, "Deployment")

    if not rows:
        return knowledge_object

    section_dict = {
        "id": QUICK_REFERENCE_SECTION_ID,
        "title": QUICK_REFERENCE_TITLE,
        "section_type": "digest",
        "description": "Incident card: alert, known failures, rollback, escalation and never-do list.",
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


# Sections that restate content from real sections (or are the coverage
# report itself). A sentence that appears ONLY in one of these has not
# reached a real section.
_DIGEST_SECTION_IDS = frozenset({TRIBAL_KNOWLEDGE_SECTION_ID, KT_COVERAGE_SECTION_ID, QUICK_REFERENCE_SECTION_ID})
RECOVERED_NOTE_TITLE = "Recovered by the final completeness check"
_PARAPHRASE_RETENTION_RATIO = 0.6


def verify_document_coverage(
    knowledge_object: Dict[str, Any],
    transcript_sentences: List[str],
    sentence_sections: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Check every fact-bearing transcript sentence against the RENDERED
    document and replace the knowledge-coverage summary with those counts.

    The summary built by append_coverage_matrix_section() is computed from
    the knowledge object before rendering, and its `lost` value was
    facts_identified - (mapped + deduplicated + unmapped) with
    facts_identified defined as that same sum -- zero by construction. On a
    real KT it reported "0 lost" while 30 of 101 sentences appeared nowhere
    in the PDF. This measures the document itself. A sentence found in no
    section is not reported and left missing: it is added to Additional
    Notes, and the summary says how many were recovered that way.
    """
    from pdf_rendering import _block_strings, _norm_for_match
    from pdf_rendering import is_text_represented as _is_represented
    from renderers import get_renderer

    rendered = knowledge_object.get("rendered_sections") or []
    real_norms: List[str] = []
    unmapped_norms: List[str] = []
    section_words: Dict[str, set] = {}
    for sec in rendered:
        sid = sec.get("section_id")
        if sid in _DIGEST_SECTION_IDS:
            continue
        norms = [_norm_for_match(s) for s in _block_strings(sec.get("blocks") or [])]
        (unmapped_norms if sid == UNMAPPED_FINDINGS_SECTION_ID else real_norms).extend(norms)
        section_words[sid] = {w for n in norms for w in n.split()}

    def _paraphrased_in_own_section(text: str) -> bool:
        # With an LLM, a section's facts are often restated (week-by-week
        # rows, checklist items) rather than quoted. Such a sentence is not
        # missing: most of its content words appear in the section it was
        # classified into. Scoped to that one section, so a sentence cannot
        # pass merely because its words are scattered across the document.
        sid = (sentence_sections or {}).get(_dedup_normalize(text))
        words = [w for w in _content_words(text)]
        if not sid or sid not in section_words or len(words) < _DEDUP_MIN_CONTENT_WORDS:
            return False
        own = section_words[sid]
        return sum(1 for w in words if w in own) / len(words) >= _PARAPHRASE_RETENTION_RATIO

    facts: List[str] = []
    seen = set()
    for text in transcript_sentences or []:
        text = (text or "").strip()
        if len(text.split()) < _MIN_UNMAPPED_SENTENCE_WORDS or _is_session_pleasantry(text):
            continue
        key = _dedup_normalize(text)
        if key in seen:
            continue
        seen.add(key)
        facts.append(text)

    mapped = unmapped = 0
    lost: List[str] = []
    for text in facts:
        if _is_represented(text, real_norms) or _paraphrased_in_own_section(text):
            mapped += 1
        elif _is_represented(text, unmapped_norms):
            unmapped += 1
        else:
            lost.append(text)

    if lost:
        unmapped_sec = next((s for s in rendered if s.get("section_id") == UNMAPPED_FINDINGS_SECTION_ID), None)
        block = {"type": "NarrativeBlock", "title": RECOVERED_NOTE_TITLE, "paragraphs": list(lost)}
        if unmapped_sec is None:
            unmapped_sec = {"section_id": UNMAPPED_FINDINGS_SECTION_ID, "section_title": UNMAPPED_FINDINGS_TITLE, "blocks": []}
            insert_at = next(
                (i for i, s in enumerate(rendered) if s.get("section_id") in _DIGEST_SECTION_IDS),
                len(rendered),
            )
            rendered.insert(insert_at, unmapped_sec)
        unmapped_sec.setdefault("blocks", []).append(block)

    summary = {
        "facts_identified": len(facts),
        "mapped": mapped,
        "unmapped": unmapped,
        "recovered": len(lost),
        "lost": 0,
        "verified_against_document": True,
    }
    coverage_section = _find_section(knowledge_object, KT_COVERAGE_SECTION_ID)
    if coverage_section is not None:
        coverage_section["_knowledge_coverage_summary"] = summary
        renderer = get_renderer(KT_COVERAGE_SECTION_ID)
        for i, sec in enumerate(rendered):
            if sec.get("section_id") == KT_COVERAGE_SECTION_ID:
                rendered[i] = renderer(coverage_section)
                break
    knowledge_object["rendered_sections"] = rendered
    knowledge_object["_knowledge_coverage_summary"] = summary
    return knowledge_object


# The template asks for the same fact in two places. When the transcript
# answers it once, the other copy must not say "Not discussed" beside it.
# (section_id, field_id) pairs; values are copied only into an unfilled side.
_LINKED_FIELDS = [
    (("architecture_reference", "verified_by_incoming_owner"), ("handover_completion", "architecture_verified")),
]


def reconcile_linked_fields(knowledge_object: Dict[str, Any]) -> Dict[str, Any]:
    def _entry(section_id: str, field_id: str):
        section = _find_section(knowledge_object, section_id)
        if not section:
            return None
        return (section.get("fields") or {}).get(field_id)

    def _filled(entry) -> bool:
        return bool(entry) and entry.get("value") not in (None, "", [], {}) and entry.get("source", "unfilled") != "unfilled"

    for left, right in _LINKED_FIELDS:
        a, b = _entry(*left), _entry(*right)
        for src, dst, dst_key in ((a, b, right), (b, a, left)):
            if _filled(src) and dst is not None and not _filled(dst):
                dst.update({
                    "value": src.get("value"),
                    "confidence": src.get("confidence", 0.0),
                    "source": "cross_section",
                    "evidence": list(src.get("evidence") or []),
                })
    return knowledge_object
