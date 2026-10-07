"""Reviewer edits with version history (P1-1).

The document a reviewer sees, edits and exports is the server's copy: the
rendered sections of the job's knowledge object. Edits are operations on
its units (a paragraph, list item, warning, table row or timeline entry),
addressed by section id, block index and unit index in the version the
reviewer was looking at:

    approve   {"op": "approve", "section", "block", "unit"}
    replace   {"op": "replace", "section", "block", "unit", "text"}      text units
              {"op": "replace", "section", "block", "unit", "cells": {column: value}}   table rows
    delete    {"op": "delete", "section", "block", "unit"}
    add       {"op": "add", "section", "text"[, "block"]}
    move      {"op": "move", "section", "block", "unit", "to_section"}    text units

Each save is a new version; version 1 is what the pipeline generated, and
every earlier version can be read back and exported. A save names the
version it was made against: if someone saved in between, it is refused
(409) instead of overwriting their change. A signed document is locked
(signoff.py).
"""
import copy
import time
from typing import Any, Dict, List, Optional, Tuple

from renderers.blocks.common import SECTION_STATUS_MESSAGES

MAX_TEXT = 4000
MAX_EDITS = 200
GENERATED_AUTHOR = "Continuum"
_TEXT_UNITS = {"NarrativeBlock": "paragraphs", "ChecklistBlock": "items", "WarningBlock": "warnings",
               "TroubleshootingBlock": "steps"}
_ROW_UNITS = {"TechnologyGrid": "rows", "OwnershipTable": "rows", "DecisionTable": "rows",
              "DeploymentTimeline": "entries"}
ADDED_BLOCK_TITLE = "Added in review"


class EditError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def current_version(job: Dict[str, Any]) -> int:
    return int(job.get("document_version") or 1)


def snapshot(knowledge_object: Dict[str, Any]) -> Dict[str, Any]:
    """What a version keeps: the document as rendered, and its sources."""
    return {"system_name": knowledge_object.get("system_name"),
            "rendered_sections": copy.deepcopy(knowledge_object.get("rendered_sections") or []),
            "sources": copy.deepcopy(knowledge_object.get("sources") or [])}


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

def _units(block: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    """(the block's unit list key, whether units are rows)."""
    kind = block.get("type")
    if kind in _TEXT_UNITS:
        return _TEXT_UNITS[kind], False
    if kind in _ROW_UNITS:
        return _ROW_UNITS[kind], True
    return None, False


def _aligned(block: Dict[str, Any], key: str, size: int) -> List[Any]:
    values = list(block.get(key) or [])
    return (values + [None if key == "review" else []] * size)[:size]


def _find(sections: List[Dict[str, Any]], edit: Dict[str, Any], need_unit: bool = True):
    section = next((s for s in sections if s.get("section_id") == edit.get("section")), None)
    if section is None:
        raise EditError(f"Unknown section {edit.get('section')!r}")
    if not need_unit:
        return section, None, None, None
    try:
        block = section["blocks"][int(edit.get("block"))]
        unit = int(edit.get("unit"))
    except (KeyError, IndexError, TypeError, ValueError):
        raise EditError("Edit does not point at a block and unit of this version")
    key, _ = _units(block)
    if key is None or not (0 <= unit < len(block.get(key) or [])):
        raise EditError("Edit does not point at a unit of this version")
    return section, block, key, unit


def _clean_text(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        raise EditError("Text must be a non-empty string")
    if len(text) > MAX_TEXT:
        raise EditError(f"Text is longer than {MAX_TEXT} characters")
    return text.strip()


def _mark(actor: str, state: str) -> Dict[str, Any]:
    return {"state": state, "by": actor, "at": time.time()}


def _remove_unit(block: Dict[str, Any], key: str, unit: int) -> Tuple[Any, List[int], Any]:
    size = len(block[key])
    sources, review = _aligned(block, "sources", size), _aligned(block, "review", size)
    value = block[key].pop(unit)
    src, rev = sources.pop(unit), review.pop(unit)
    block["sources"], block["review"] = sources, review
    return value, src, rev


def _append_text(section: Dict[str, Any], text: str, sources: List[int], review: Dict[str, Any],
                 block_index: Optional[int] = None) -> None:
    blocks = section.setdefault("blocks", [])
    # A section that only said "not covered" now has content.
    for block in blocks:
        key, _ = _units(block)
        if key and not block.get("rows") and all(str(u).strip() in SECTION_STATUS_MESSAGES for u in block.get(key) or []):
            block[key], block["sources"], block["review"] = [], [], []
    target = None
    if block_index is not None:
        try:
            target = blocks[int(block_index)]
        except (IndexError, TypeError, ValueError):
            raise EditError("Edit does not point at a block of this version")
        if _units(target)[1] or _units(target)[0] is None:
            raise EditError("Text can only be added to a paragraph, list or warning block")
    else:
        target = next((b for b in blocks if b.get("type") in ("NarrativeBlock", "ChecklistBlock")), None)
    if target is None:
        target = {"type": "NarrativeBlock", "title": ADDED_BLOCK_TITLE, "paragraphs": []}
        blocks.append(target)
    key, _ = _units(target)
    size = len(target.get(key) or [])
    target["sources"] = _aligned(target, "sources", size) + [list(sources)]
    target["review"] = _aligned(target, "review", size) + [review]
    target.setdefault(key, []).append(text)


def _tidy(section: Dict[str, Any]) -> None:
    """Drop blocks left without units; a section left empty says so."""
    from renderers.blocks.common import no_coverage_block

    kept = []
    for block in section.get("blocks") or []:
        key, _ = _units(block)
        if key is None or block.get(key):
            kept.append(block)
    if not kept:
        kept = [no_coverage_block(section.get("section_title") or "")]
    section["blocks"] = kept


# --------------------------------------------------------------------------
# Applying edits
# --------------------------------------------------------------------------

def apply_edits(knowledge_object: Dict[str, Any], edits: List[Dict[str, Any]], actor: str) -> Dict[str, int]:
    """Apply `edits` (all addressed against the same version) to the
    knowledge object's rendered sections in place. Raises EditError and
    leaves nothing half-applied only if the caller passes a copy."""
    if not isinstance(edits, list) or not edits:
        raise EditError("No edits")
    if len(edits) > MAX_EDITS:
        raise EditError(f"At most {MAX_EDITS} edits per save")
    sections = knowledge_object.get("rendered_sections") or []
    counts = {"approve": 0, "replace": 0, "delete": 0, "add": 0, "move": 0}

    # Unit indexes refer to the version the reviewer saw: first the edits
    # that keep units in place, then removals from the last unit up, then
    # additions.
    in_place = [e for e in edits if e.get("op") in ("approve", "replace")]
    removals = [e for e in edits if e.get("op") in ("delete", "move")]
    additions = [e for e in edits if e.get("op") == "add"]
    if len(in_place) + len(removals) + len(additions) != len(edits):
        bad = next(e.get("op") for e in edits if e.get("op") not in counts)
        raise EditError(f"Unknown edit {bad!r}")

    for edit in in_place:
        _, block, key, unit = _find(sections, edit)
        size = len(block[key])
        review = _aligned(block, "review", size)
        if edit["op"] == "approve":
            review[unit] = _mark(actor, "approved")
        elif _units(block)[1]:
            cells = edit.get("cells")
            row = block[key][unit]
            allowed = set(row) | set(block.get("columns") or [])
            if not isinstance(cells, dict) or not cells or not set(cells) <= allowed:
                raise EditError("A table row is edited by its existing columns: {column: text}")
            for column, value in cells.items():
                row[column] = "" if value in (None, "") else _clean_text(value)
            review[unit] = _mark(actor, "edited")
        else:
            block[key][unit] = _clean_text(edit.get("text"))
            review[unit] = _mark(actor, "edited")
        block["sources"], block["review"] = _aligned(block, "sources", size), review
        counts[edit["op"]] += 1

    moved = []
    seen = set()
    for edit in sorted(removals, key=lambda e: (str(e.get("section")), int(e.get("block", 0) or 0),
                                                int(e.get("unit", 0) or 0)), reverse=True):
        address = (edit.get("section"), edit.get("block"), edit.get("unit"))
        if address in seen:
            raise EditError("The same unit is removed twice")
        seen.add(address)
        section, block, key, unit = _find(sections, edit)
        if edit["op"] == "move":
            target = next((s for s in sections if s.get("section_id") == edit.get("to_section")), None)
            if target is None or target is section:
                raise EditError("Move needs another section of this document as to_section")
            if _units(block)[1]:
                raise EditError("Table rows cannot be moved; delete the row and add the text instead")
        value, sources, _ = _remove_unit(block, key, unit)
        if edit["op"] == "move":
            moved.append((target, str(value), sources))
        counts[edit["op"]] += 1
        _tidy(section)

    touched = []
    for target, text, sources in reversed(moved):
        _append_text(target, text, sources, _mark(actor, "moved"))
        touched.append(target)
    for edit in additions:
        section, *_ = _find(sections, edit, need_unit=False)
        _append_text(section, _clean_text(edit.get("text")), [], _mark(actor, "added"), edit.get("block"))
        touched.append(section)
        counts["add"] += 1
    for section in touched:       # a "not covered" line that gave way leaves an empty block
        _tidy(section)
    return counts


_DIGEST_SECTIONS = {"kt_coverage", "quick_reference", "tribal_knowledge"}


def _norm(text: Any) -> str:
    import re

    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())).strip()


def move_edits(knowledge_object: Dict[str, Any], sentence: str, to_section: str) -> List[Dict[str, Any]]:
    """Edits that move every text unit stating `sentence` (it, or a unit it
    makes up most of) into `to_section` (a section correction, /feedback)."""
    sections = knowledge_object.get("rendered_sections") or []
    key = _norm(sentence)
    if not key or not any(s.get("section_id") == to_section for s in sections):
        return []
    edits = []
    for section in sections:
        if section.get("section_id") in _DIGEST_SECTIONS or section.get("section_id") == to_section:
            continue
        for b, block in enumerate(section.get("blocks") or []):
            unit_key, rows = _units(block)
            if unit_key is None or rows:
                continue
            for u, text in enumerate(block.get(unit_key) or []):
                norm = _norm(text)
                if norm == key or (key in norm and len(key) >= 0.6 * len(norm)):
                    edits.append({"op": "move", "section": section["section_id"], "block": b, "unit": u,
                                  "to_section": to_section})
    return edits


def summary_of(counts: Dict[str, int]) -> str:
    words = {"approve": "approved", "replace": "edited", "delete": "deleted", "add": "added", "move": "moved"}
    return ", ".join(f"{n} {words[op]}" for op, n in counts.items() if n) or "no changes"


def save_edits(pipeline, job_id: str, edits: List[Dict[str, Any]], base_version: int, actor: str) -> Dict[str, Any]:
    """Apply a reviewer's edits as a new version of the job's document.
    Returns {"version", "summary", "knowledge_object"}."""
    import signoff

    with pipeline.JOB_LOCK:
        job = pipeline.JOB_QUEUE.get(job_id)
        if job is None or job.get("status") not in pipeline.COMPLETED_STATUSES:
            raise EditError("Only a finished KT can be edited", 409)
        if signoff.is_locked(job):
            raise EditError("This KT is signed; signed versions cannot be changed", 409)
        version = current_version(job)
        if int(base_version) != version:
            raise EditError(f"The document changed since you opened it (now version {version}). "
                            "Reload it and make your edits again.", 409)
        ensure_first_version(pipeline, job_id, job)
        ko = copy.deepcopy(job.get("knowledge_object") or {})
        counts = apply_edits(ko, edits, actor)
        new_version = version + 1
        summary = summary_of(counts)
        if not pipeline.JOB_STORE.save_version(job_id, new_version, actor, summary, snapshot(ko)):
            raise EditError("Someone else saved at the same moment. Reload and try again.", 409)
        job["knowledge_object"] = ko
        job["document_version"] = new_version
        signoff.on_new_version(job, new_version, actor)
        pipeline.JOB_QUEUE[job_id] = job        # the stored PDF is now stale and re-renders on download
    return {"version": new_version, "summary": summary, "knowledge_object": ko}


def ensure_first_version(pipeline, job_id: str, job: Dict[str, Any]) -> None:
    """Keep the generated document as version 1 before the first change."""
    if not pipeline.JOB_STORE.versions(job_id):
        pipeline.JOB_STORE.save_version(job_id, 1, GENERATED_AUTHOR, "Generated from the KT session",
                                        snapshot(job.get("knowledge_object") or {}))


def versions(pipeline, job_id: str, job: Dict[str, Any]) -> Dict[str, Any]:
    stored = pipeline.JOB_STORE.versions(job_id)
    if not stored:
        stored = [{"version": 1, "created_at": pipeline.JOB_STORE.created_at(job_id), "author": GENERATED_AUTHOR,
                   "summary": "Generated from the KT session"}]
    return {"current": current_version(job), "versions": stored}


def version_snapshot(pipeline, job_id: str, job: Dict[str, Any], version: int) -> Optional[Dict[str, Any]]:
    if version == current_version(job):
        return snapshot(job.get("knowledge_object") or {})
    return pipeline.JOB_STORE.version(job_id, version)
