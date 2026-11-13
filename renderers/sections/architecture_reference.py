from typing import Dict, Any, List
from architecture_diagram import CONNECTION_COLUMNS, describe_unconnected
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.technology_grid import build_block as build_technology_grid
from renderers.blocks.diagram import build_block as build_diagram_block
from renderers.blocks.table import build_block as build_table_block
from renderers.blocks.common import (
    no_coverage_block,
    coverage_paragraphs as shared_coverage_paragraphs,
)

# The PDF prints this table's last column as sentence numbers (pdf_rendering).
CONNECTIONS_TABLE_TITLE = "Connections in the diagram"

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
    details = None

    # Architecture Knowledge: the real components/services this system is
    # built on (knowledge_builder.enrich_architecture_knowledge), detected
    # across the whole transcript independent of which section's sentences
    # happened to mention them. This used to be entirely invisible here —
    # the section's only fields are the 3 admin facts below, so a rich
    # architecture discussion with no stated Confluence link rendered as
    # "0 of 3 fields" as if nothing had been captured at all.
    components = section.get("_architecture_components") or []
    if components:
        # The bare names, only when there is no diagram: the diagram and the
        # Technology summary already name every component (X62; dropping the
        # list moved no fact on any accuracy set, audit §9abg).
        if not section.get("_architecture_svg"):
            blocks.append(build_narrative_block("Architecture Knowledge", [", ".join(components)]))
        # In-depth detail beyond the bare names above: the real transcript
        # sentences that named these components, which is usually where the
        # component's actual ROLE is stated ("Redis is used for caching and
        # short-lived session data") — a flat name list alone can't carry
        # that. Verbatim transcript text, never paraphrased/generated.
        sentences = section.get("_architecture_sentences") or []
        if sentences:
            details = build_narrative_block("Architecture Details", sentences)
    else:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block("Architecture Knowledge", fallback))

    # The architecture picture (architecture_diagram.py): components placed
    # by what they are, joined only where the KT states the connection, with
    # each stated connection listed next to the sentence that says it.
    svg = section.get("_architecture_svg")
    if svg:
        # What each line rests on (the client's review, 13 Nov: confirmed,
        # inferred and unresolved apart). The connections table lists every
        # line with its basis and sentence (architecture_diagram.check_connections).
        graph = section.get("_architecture_graph") or {}
        # The drawing's own legend explains dashed lines when there are any.
        inferred = any(line.get("basis") == "inferred"
                       for line in list(graph.get("entry_links") or []) + list(graph.get("edges") or []))
        caption = ("" if inferred or not graph else "Every line is a connection a sentence of the KT states. ") + (
            "Managed services sit outside the compute platform; third-party services sit outside the cloud account.")
        unconnected = describe_unconnected(graph) if graph else []
        if unconnected:
            caption += (" Named without a stated connection, so drawn without a line: " + "; ".join(unconnected)
                        + ". Ask the outgoing owner how they connect.")
        blocks.append(build_diagram_block("High-Level Architecture", svg, caption))
        connections = section.get("_architecture_connections") or []
        if connections:
            columns = CONNECTION_COLUMNS if any("Basis" in row for row in connections) else \
                ["From", "Connection", "To", "Said in the KT"]
            blocks.append(build_table_block(CONNECTIONS_TABLE_TITLE, columns, connections))

    # The sentences after the picture: the diagram cannot break across pages,
    # and placed after the sentences it left a third of a page blank above
    # it whenever it did not fit below them (the client's e-commerce KT).
    if details is not None:
        blocks.append(details)

    # Architecture Metadata: only what the KT stated. Rows reading "Not
    # discussed" for all three were filler a reader had to skip (the
    # due-diligence review, P0-2); an undiscussed one is still reported, in
    # the coverage matrix and the follow-up list, never silently dropped.
    metadata_rows = []
    for field_id, label in _METADATA_FIELD_SPECS:
        value = fields.get(field_id, {}).get("value")
        if isinstance(value, bool):
            metadata_rows.append({"label": label, "value": "Confirmed" if value else "Not confirmed"})
        elif value not in (None, "", []):
            metadata_rows.append({"label": label, "value": str(value)})
    if metadata_rows:
        blocks.append(build_technology_grid("Architecture Metadata", metadata_rows))

    if not blocks:
        blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
