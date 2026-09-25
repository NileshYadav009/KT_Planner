from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.technology_grid import build_block as build_technology_grid
from renderers.blocks.code import build_block as build_code_block
from renderers.blocks.common import (
    no_coverage_block,
    coverage_paragraphs as shared_coverage_paragraphs,
)

# (field_id, display label) for the 3 template-defined administrative facts
# this section's schema actually has. These are metadata about the
# architecture doc, never the architecture knowledge itself.
_METADATA_FIELD_SPECS = [
    ("architecture_link", "Documentation link"),
    ("last_updated", "Last updated"),
    ("verified_by_incoming_owner", "Verified by incoming owner"),
]


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    # Shared implementation: splits polish-pass bullet blobs and drops
    # repeats. See renderers/blocks/common.coverage_paragraphs().
    return shared_coverage_paragraphs(section)


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
        # In-depth detail beyond the bare names above: the real transcript
        # sentences that named these components, which is usually where the
        # component's actual ROLE is stated ("Redis is used for caching and
        # short-lived session data") — a flat name list alone can't carry
        # that. Verbatim transcript text, never paraphrased/generated.
        sentences = section.get("_architecture_sentences") or []
        if sentences:
            blocks.append(build_narrative_block("Architecture Details", sentences))
    else:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block("Architecture Knowledge", fallback))

    # A top-down "mental model" diagram built from the component list's
    # coarse layer classification (architecture_diagram.py) — only present
    # when the transcript actually named something resembling a request-flow
    # position (frontend/cdn/load-balancer/compute/data-layer); never
    # fabricated to fill in a "typical" architecture the transcript didn't
    # describe.
    diagram = section.get("_architecture_diagram")
    if diagram:
        blocks.append(build_code_block("High-Level Architecture", diagram))

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
