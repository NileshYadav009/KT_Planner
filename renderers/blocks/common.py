import re
from typing import Any, Dict, List

from renderers.blocks.narrative import build_block as _build_narrative_block

# A bullet marker that begins a line, or one that follows a completed
# sentence. Deliberately NOT a bare " - ": that would split hyphenated
# ranges and clause dashes mid-sentence. Requiring either a line start or a
# preceding sentence terminator keeps "Bicep-managed", "read-only" and
# "Trigger: X - Window: Y" intact.
_BULLET_SPLIT_RE = re.compile(r"(?:(?<=[.!?])\s+|\n\s*)[-*•]\s+")
_LEADING_BULLET_RE = re.compile(r"^\s*[-*•]\s+")


def split_bullet_blob(items: List[str]) -> List[str]:
    """Split coverage strings that arrived as ONE markdown bullet list blob
    back into individual items.

    The LLM polish pass (ai.polish_coverage_sections) commonly returns a
    section as "- first item. - second item.", which then rendered as a
    single warning card containing both facts plus literal dashes -- seen on
    live Danger Zones and Operational Calendar output. Splitting here keeps
    one fact per rendered card without changing any renderer's contract.
    Text with no bullet markers is returned unchanged.
    """
    out: List[str] = []
    for item in items or []:
        text = str(item).strip()
        if not text:
            continue
        for part in _BULLET_SPLIT_RE.split(text):
            part = _LEADING_BULLET_RE.sub("", part).strip()
            if part:
                out.append(part)
    return out


def coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    """A section's coverage_content as clean, de-duplicated paragraphs.

    Every section renderer needs the same three things and each used to do
    only the first: read the content, split any bullet blob the polish pass
    returned as one string (otherwise literal "- " dashes render mid-
    paragraph), and drop repeats — the raw and polished forms of the same
    sentence frequently both arrive, which published the same fact twice in
    one block.
    """
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    items = [str(item).strip() for item in content if isinstance(item, str) and str(item).strip()]

    out: List[str] = []
    seen = set()
    for text in split_bullet_blob(items):
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


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
