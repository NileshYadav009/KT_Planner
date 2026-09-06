"""
validation.py
==============
A lightweight, non-fatal validation layer for the KT pipeline's core
artifacts (dynamic schema, populated fields, knowledge object). Every
function here returns a list of human-readable warning strings — it never
raises and never mutates its input, so it's safe to call unconditionally at
the end of a pipeline run without risking that run's success.

This exists because this session repeatedly found the same class of bug by
hand: a renderer or extraction path referencing a field id that doesn't
actually exist in the schema for that section, silently producing nothing
(see REPOSITORY_AUDIT.md §9l/9n for several concrete examples). These checks
catch that class of mismatch automatically, on every run, instead of relying
on someone noticing a blank PDF section.

Renderer/schema-id consistency for the section level (not field level) is
already checked separately at startup — see
renderers/sections/__init__.py:validate_renderer_registry(). This module is
the per-run, per-field/knowledge-object counterpart.
"""

from typing import Any, Dict, List, Optional, Set

VALID_STATUSES = {"missing", "weak", "covered"}


def _flatten_schema_field_ids(fields: List[Dict[str, Any]]) -> Set[str]:
    """All field ids declared for a section, including nested group fields
    (populate_fields() stores a group's own sub-fields at output[group_id][sub_id],
    so both levels need to be known valid ids)."""
    ids: Set[str] = set()
    for field in fields or []:
        field_id = field.get("id")
        if field_id:
            ids.add(field_id)
        if field.get("type") == "group":
            ids |= _flatten_schema_field_ids(field.get("fields") or [])
    return ids


def validate_populated_fields(
    populated_fields: Dict[str, Dict[str, Any]],
    dynamic_schema: List[Dict[str, Any]],
) -> List[str]:
    """Cross-check populated_fields against the schema it was supposedly
    populated from: every section id must be a real schema section, and every
    field id within it must be a field actually declared for that section.
    Does NOT check whether fields are filled — an "unfilled" field is a
    coverage gap, not a validation error.

    Sections with no `fields` array in the schema at all (monitoring_observability,
    security_controls, disaster_recovery, ownership_escalation, cost_optimization,
    common_failures) are populated by a different, deliberate mechanism —
    ai.wrap_structured_as_fields()'s LLM structured-JSON extraction
    (llm/prompts.py's SECTION_STRUCTURED_PROMPTS), not field_populator.py —
    and define their own field shape per prompt rather than a schema `fields`
    array. Field-id validation is skipped for those sections entirely; the
    section-existence check above still applies unconditionally.

    One of those sections (ownership_escalation) can still gain a single
    *dynamic* field (`oncall_tool`, tagged `"dynamic": True`) via
    schema_generator.py's TECH_STACK_FIELD_ADDITIONS when the transcript
    mentions a paging tool — so "has a non-empty fields array" alone isn't
    a reliable signal; a section with only dynamically-added fields and no
    real base fields is still a structured-extraction section underneath.
    Skip id validation when every declared field is dynamic, not just when
    the array is empty.
    """
    warnings: List[str] = []
    schema_by_id = {s.get("id"): s for s in dynamic_schema if s.get("id")}

    for section_id, fields in (populated_fields or {}).items():
        section = schema_by_id.get(section_id)
        if section is None:
            warnings.append(
                f"populated_fields has section '{section_id}' which does not "
                f"exist in the dynamic schema"
            )
            continue

        declared_fields = section.get("fields") or []
        non_dynamic_fields = [f for f in declared_fields if not f.get("dynamic")]
        if not non_dynamic_fields:
            # No real (non-dynamic) schema fields -> this section is
            # fundamentally populated via structured LLM extraction, not
            # field_populator.py, regardless of any dynamic bonus field it
            # may have also picked up. Its field ids are legitimately not
            # schema-declared; skip id validation.
            continue

        valid_ids = _flatten_schema_field_ids(declared_fields)
        for field_id, value in (fields or {}).items():
            if field_id not in valid_ids:
                warnings.append(
                    f"populated_fields['{section_id}']['{field_id}'] is not a "
                    f"field declared in that section's schema (declared: "
                    f"{sorted(valid_ids) or 'none'})"
                )
                continue
            if isinstance(value, dict) and "value" in value:
                # A leaf field entry — check it has the shape
                # field_populator.py's _emit() actually produces.
                for required_key in ("value", "confidence", "source"):
                    if required_key not in value:
                        warnings.append(
                            f"populated_fields['{section_id}']['{field_id}'] "
                            f"is missing expected key '{required_key}'"
                        )
                confidence = value.get("confidence")
                if isinstance(confidence, (int, float)) and not (0.0 <= confidence <= 1.0):
                    warnings.append(
                        f"populated_fields['{section_id}']['{field_id}'].confidence "
                        f"({confidence}) is outside the expected 0.0-1.0 range"
                    )

    return warnings


def validate_knowledge_object(ko: Dict[str, Any]) -> List[str]:
    """Structural checks on the assembled knowledge object — the thing
    actually handed to renderers and returned via /schema/{job_id}. Checks
    shape and value ranges, not content quality (a section legitimately
    having zero facts because the transcript never covered it is not a
    validation error).
    """
    warnings: List[str] = []
    if not isinstance(ko, dict):
        return [f"knowledge object is not a dict (got {type(ko).__name__})"]

    for required_key in ("job_id", "system_name", "sections", "summary"):
        if required_key not in ko:
            warnings.append(f"knowledge object is missing top-level key '{required_key}'")

    sections = ko.get("sections")
    if not isinstance(sections, list):
        warnings.append(f"knowledge object 'sections' is not a list (got {type(sections).__name__})")
        return warnings

    seen_ids: Set[str] = set()
    for idx, section in enumerate(sections):
        if not isinstance(section, dict):
            warnings.append(f"sections[{idx}] is not a dict")
            continue

        section_id = section.get("id")
        label = section_id or f"sections[{idx}]"

        if not section_id:
            warnings.append(f"sections[{idx}] has no 'id'")
        elif section_id in seen_ids:
            warnings.append(f"section id '{section_id}' appears more than once in sections")
        else:
            seen_ids.add(section_id)

        for required_key in ("title", "status", "facts", "entities", "evidence", "relationships", "fields"):
            if required_key not in section:
                warnings.append(f"section '{label}' is missing expected key '{required_key}'")

        status = section.get("status")
        if status is not None and status not in VALID_STATUSES:
            warnings.append(f"section '{label}' has status '{status}', expected one of {sorted(VALID_STATUSES)}")

        for numeric_key in ("confidence", "risk"):
            val = section.get(numeric_key)
            if isinstance(val, (int, float)) and not (0.0 <= val <= 1.0):
                warnings.append(f"section '{label}'.{numeric_key} ({val}) is outside the expected 0.0-1.0 range")

        for list_key in ("facts", "entities", "evidence", "relationships"):
            val = section.get(list_key)
            if val is not None and not isinstance(val, list):
                warnings.append(f"section '{label}'.{list_key} is not a list (got {type(val).__name__})")

    return warnings


def validate_pipeline_run(
    knowledge_object: Dict[str, Any],
    populated_fields: Optional[Dict[str, Dict[str, Any]]] = None,
    dynamic_schema: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Convenience entry point — runs every check this module has and
    returns one combined, deduplicated list. This is what pipeline.py calls;
    the individual validate_* functions above are exposed separately for
    targeted use (e.g. in tests).
    """
    warnings = list(validate_knowledge_object(knowledge_object))
    if populated_fields is not None and dynamic_schema is not None:
        warnings.extend(validate_populated_fields(populated_fields, dynamic_schema))
    # Stable de-dup, preserving order.
    return list(dict.fromkeys(warnings))
