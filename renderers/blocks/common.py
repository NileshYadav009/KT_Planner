from typing import Any, Dict

from renderers.blocks.narrative import build_block as _build_narrative_block

NOT_COVERED_MESSAGE = (
    "This section was not covered in the KT session. Flag it for follow-up "
    "with the outgoing owner."
)


def no_coverage_block(title: str) -> Dict[str, Any]:
    """A single, consistently-worded fallback block for a section with zero
    coverage — used in place of the many ad hoc "being extracted/assembled/
    synthesized/derived" placeholder strings previously scattered across
    renderers/sections/*.py, which read like debug output rather than an
    intentional, visible gap in a real KT deliverable.
    """
    return _build_narrative_block(title, [NOT_COVERED_MESSAGE])
