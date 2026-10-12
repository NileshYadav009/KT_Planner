"""What changed between two versions of a KT document, or between two KTs of
the same system (P3-2).

A reviewer who opens version 4 needs to know what versions 2-4 changed, and
an engineer taking over a system that had a KT last year needs to know what
is different now. Both compare two document snapshots
(document_model.snapshot) unit by unit: a paragraph, list item, warning,
table row or timeline entry, per section. A unit only in the newer document
is added, one only in the older is removed, and a removed and an added unit
in the same section that read alike are one changed unit (before and after).

The Quick Reference and Coverage sections are left out: they are built from
the other sections, so their changes are already listed there.
"""
from difflib import SequenceMatcher
import re
from typing import Any, Dict, List, Optional

from knowledge.evidence import block_units
from renderers.blocks.common import SECTION_STATUS_MESSAGES

DERIVED_SECTIONS = frozenset({"quick_reference", "kt_coverage", "signoff"})
# How alike a removed and an added unit must read to be one edited unit.
CHANGED_SIMILARITY = 0.6


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def _section_units(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """section id -> {"title", "units": [text, ...]} in document order."""
    out: Dict[str, Dict[str, Any]] = {}
    for section in (snapshot or {}).get("rendered_sections") or []:
        sid = section.get("section_id")
        if not sid or sid in DERIVED_SECTIONS:
            continue
        entry = out.setdefault(sid, {"title": section.get("section_title") or section.get("title") or sid, "units": []})
        for block in section.get("blocks") or []:
            for text in block_units(block) or []:
                text = str(text).strip()
                if text and text not in SECTION_STATUS_MESSAGES and _key(text):
                    entry["units"].append(text)
    return out


def _pair_changes(removed: List[str], added: List[str]):
    """Pair each removed unit with the added unit that reads most like it."""
    changed, still_added = [], list(added)
    still_removed = []
    for old in removed:
        best, best_score = None, CHANGED_SIMILARITY
        for new in still_added:
            score = SequenceMatcher(None, _key(old), _key(new)).ratio()
            if score >= best_score:
                best, best_score = new, score
        if best is None:
            still_removed.append(old)
        else:
            changed.append({"before": old, "after": best})
            still_added.remove(best)
    return still_removed, still_added, changed


def diff_snapshots(old: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The changes from `old` to `new`, per section in the newer document's
    order (sections only in the older one last)."""
    before, after = _section_units(old), _section_units(new)
    order = list(after) + [sid for sid in before if sid not in after]
    sections, totals = [], {"added": 0, "removed": 0, "changed": 0, "unchanged": 0}
    for sid in order:
        old_units = (before.get(sid) or {}).get("units", [])
        new_units = (after.get(sid) or {}).get("units", [])
        old_keys, new_keys = {_key(u) for u in old_units}, {_key(u) for u in new_units}
        removed = [u for u in old_units if _key(u) not in new_keys]
        added = [u for u in new_units if _key(u) not in old_keys]
        removed, added, changed = _pair_changes(removed, added)
        unchanged = len({_key(u) for u in new_units} & old_keys)
        totals["added"] += len(added)
        totals["removed"] += len(removed)
        totals["changed"] += len(changed)
        totals["unchanged"] += unchanged
        if added or removed or changed:
            title = (after.get(sid) or before.get(sid))["title"]
            sections.append({"section_id": sid, "title": title, "added": added, "removed": removed,
                             "changed": changed, "unchanged": unchanged,
                             "now_covered": bool(new_units) and not old_units,
                             "no_longer_covered": bool(old_units) and not new_units})
    return {"sections": sections, "totals": totals}


def summary_line(diff: Dict[str, Any]) -> str:
    t = diff["totals"]
    parts = [f"{t[k]} {k}" for k in ("added", "changed", "removed") if t[k]]
    return ", ".join(parts) if parts else "No differences"
