from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.technology_grid import build_block as build_technology_grid
from renderers.blocks.common import no_coverage_block

# (field_id, display label) for the 3 template-defined administrative facts
# this section's schema actually has. These are metadata about the
# architecture doc, never the architecture knowledge itself.
_METADATA_FIELD_SPECS = [
    ("architecture_link", "Documentation link"),
    ("last_updated", "Last updated"),
    ("verified_by_incoming_owner", "Verified by incoming owner"),
]


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    return [str(item).strip() for item in content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Architecture Reference")
    fields = section.get("fields", {})

    blocks = []

    # Architecture Knowledge: the real components/services this system is
    # built on (knowledge_builder.enrich_architecture_knowledge), detected
    # across the whole transcript independent of which section's sentences
    # happened to mention them. This used to be entirely invisible here —
    # the section's only fields are the 3 admin facts below, so a rich
    # architecture discussion with no stated Confluence link rendered as
    # "0 of 3 fields" as if nothing had been captured at all.
    components = section.get("_architecture_components") or []
    if components:
        blocks.append(build_narrative_block("Architecture Knowledge", [", ".join(components)]))
    else:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block("Architecture Knowledge", fallback))

    # Architecture Metadata: always render all 3 admin fields explicitly,
    # one row each, with "Not discussed" for anything genuinely never
    # mentioned — rather than silently omitting the row or letting an
    # empty administrative field make the section look incomplete overall.
    metadata_rows = []
    for field_id, label in _METADATA_FIELD_SPECS:
        value = fields.get(field_id, {}).get("value")
        display_value = str(value) if value not in (None, "", []) else "Not discussed"
        metadata_rows.append({"label": label, "value": display_value})
    blocks.append(build_technology_grid("Architecture Metadata", metadata_rows))

    if not blocks:
        blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
