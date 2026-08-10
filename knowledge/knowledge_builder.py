import re
from typing import Any, Dict, List, Optional

from .entities import build_entities
from .evidence import build_evidence
from .facts import build_facts
from .relationships import build_relationships


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _collect_evidence(
    section_id: str,
    section_content: Dict[str, Any],
    populated_field: Dict[str, Any],
) -> List[Dict[str, Any]]:
    sentence_entries = section_content.get("sentences", []) if isinstance(section_content, dict) else []

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


_BAD_NAME_WORDS = {"provisioning", "infrastructure", "state", "deployment", "process", "workflow"}


def _is_plausible_system_name(candidate: str) -> bool:
    words = [w for w in candidate.strip().split() if w]
    if not (1 <= len(words) <= 5):
        return False
    return words[-1].lower() not in _BAD_NAME_WORDS


def _infer_system_name(
    coverage: Dict[str, Any],
    populated_fields: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    if populated_fields:
        sys_field = (populated_fields.get("system_overview", {}) or {}).get("system_name", {})
        val = sys_field.get("value")
        if isinstance(val, str) and val.strip():
            candidate = val.strip()
            if _is_plausible_system_name(candidate):
                return candidate
    # TODO: surface inferred system_name in the UI as an editable field before export.
    return "KT Document"


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
        section_structured = section_cov.get("_structured") if isinstance(section_cov, dict) else None

        raw_fields = populated_fields.get(section_id, {})
        field_objects = {fid: {
            "id": fid,
            "label": field.get("label", fid),
            "type": field.get("type", "text"),
            "value": field.get("value"),
            "confidence": float(field.get("confidence", 0.0) or 0.0),
            "source": field.get("source", "unfilled"),
            "evidence": _collect_evidence(section_id, section_evidence, field),
        } for fid, field in raw_fields.items()}

        section_facts = build_facts(section_id, field_objects)
        section_entities = build_entities(field_objects)
        section_evidence_list = build_evidence(section_evidence.get("sentences", []))
        section_relations = build_relationships(field_objects)

        knowledge_sections.append({
            "id": section_id,
            "title": section_title,
            "section_type": section.get("type", "section"),
            "description": section.get("description", ""),
            "status": section_cov.get("status", "missing"),
            "confidence": float(section_cov.get("confidence", 0.0) or 0.0),
            "sentence_count": int(section_cov.get("sentence_count", 0) or 0),
            "risk": float(section_cov.get("risk", 0.0) or 0.0),
            "coverage_content": section_cov.get("content", []),
            "_structured": section_structured,
            "facts": section_facts,
            "entities": section_entities,
            "evidence": section_evidence_list,
            "relationships": section_relations,
            "fields": field_objects,
        })
        def _section_gap_reason(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            fields = section.get("fields", {}) or {}
            if not fields:
                # No schema fields to assess — rely on status
                if section.get("status") == "missing":
                    return {"reason": "No content was detected for this section.", "severity": "missing"}
                return None

            filled = [
                f for f in fields.values()
                if f.get("value") not in (None, "", []) and float(f.get("confidence", 0.0)) >= 0.5
            ]
            fill_rate = len(filled) / len(fields) if fields else 0.0
            if section.get("status") == "missing" and fill_rate == 0:
                return {"reason": "No content was detected for this section.", "severity": "missing"}
            if fill_rate < 0.4:
                return {"reason": f"Only {len(filled)}/{len(fields)} fields have confident values.", "severity": "weak"}
            return None

        gaps = []
        for section in knowledge_sections:
            reason_obj = _section_gap_reason(section)
            if reason_obj is None:
                continue
            gaps.append({
                "section_id": section["id"],
                "section_title": section["title"],
                "status": section.get("status"),
                "confidence": section.get("confidence", 0.0),
                "sentence_count": section.get("sentence_count", 0),
                "reason": reason_obj.get("reason"),
                "severity": reason_obj.get("severity"),
                "suggestion": "Review this section and fill any missing details or evidence.",
            })

    return {
        "job_id": job_id,
        "system_name": _infer_system_name(coverage, populated_fields),
        "sections": knowledge_sections,
        "gaps": gaps,
        "summary": {
            "section_count": len(knowledge_sections),
            "covered_sections": sum(1 for sec in knowledge_sections if sec.get("status") in {"covered", "weak"}),
        },
    }
