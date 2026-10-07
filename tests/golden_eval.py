"""Golden KT suite: annotated transcripts scored on the rendered document (P1-11).

Each file in tests/goldens/ is a transcript plus its annotation:

    facts     each fact: an id, a verbatim quote, the section(s) it belongs in
              ("in"), optionally sections it must never appear in ("not_in"),
              optional "alt" strings for facts the document may show as a
              field value rather than the sentence ("four hours"), and
              optional "terms" that must all appear in one table row (an
              ownership row: "clearing team", "ledger writer").
    covered   sections the KT discussed: must not be reported Missing.
    missing   sections the KT did not discuss: must be reported Missing.
    conflicts substrings that must appear in a flagged contradiction.

Scores, per golden:
    accuracy        facts shown in an allowed section / all facts
    wrong           facts shown, but only outside their allowed sections
    lost            facts shown nowhere (the completeness check should make this 0)
    violations      facts shown in a section they must never appear in
    false_missing   discussed sections reported Missing
    false_covered   undiscussed sections reported covered
    repeats         fact sentences printed more than once (P1-10)
    unsourced       facts shown with no source sentence that states them (P1-2)

Run without an LLM, so results are deterministic and use no quota.
`python scripts/golden_report.py` prints the table; tests/test_golden_suite.py
fails a change that lowers any score below tests/goldens/baseline.json.
"""
import json
import os
import re
from typing import Any, Dict, List

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
BASELINE_PATH = os.path.join(GOLDEN_DIR, "baseline.json")
TARGET_ACCURACY = 0.90

_DIGESTS = {"quick_reference", "tribal_knowledge", "kt_coverage"}
_QUOTING_TITLES = {"Connections stated in the KT", "Mentioned elsewhere in the KT"}
# Section status lines ("not covered", "mentioned elsewhere") are the same in
# every section they apply to; they are not facts, so they are not repeats.
_PLACEHOLDER = re.compile(r"not covered (?:in|during) the kt|flag it for follow|"
                          r"not discussed as its own topic|confirm the rest with", re.IGNORECASE)


def load_goldens() -> List[Dict[str, Any]]:
    goldens = []
    for name in sorted(os.listdir(GOLDEN_DIR)):
        if name.endswith(".json") and name != "baseline.json":
            with open(os.path.join(GOLDEN_DIR, name), encoding="utf-8") as fh:
                goldens.append(json.load(fh))
    return goldens


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def _quotes_on_purpose(block: Dict[str, Any]) -> bool:
    title = str(block.get("title") or "")
    return (block.get("type") in ("ImageBlock", "DiagramBlock") or title in _QUOTING_TITLES
            or (block.get("type") == "WarningBlock" and title.startswith("Possible conflict")))


def _block_strings(block: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("paragraphs", "warnings", "items", "steps"):
        out.extend(str(v) for v in block.get(key) or [])
    for entry in block.get("entries") or []:
        out.extend(str(v) for v in (entry.values() if isinstance(entry, dict) else [entry]))
    columns = block.get("columns")
    for row in block.get("rows") or []:
        if isinstance(row, dict):
            out.append(" ".join(str(row.get(c, "")) for c in (columns or list(row))))
    return out


def section_texts(knowledge_object: Dict[str, Any]) -> Dict[str, List[str]]:
    """What a reader sees in each real section of the rendered document."""
    texts: Dict[str, List[str]] = {}
    for sec in knowledge_object.get("rendered_sections") or []:
        sid = sec.get("section_id")
        if sid in _DIGESTS:
            continue
        for block in sec.get("blocks") or []:
            if not _quotes_on_purpose(block):
                texts.setdefault(sid, []).extend(_block_strings(block))
    return texts


def coverage_buckets(knowledge_object: Dict[str, Any]) -> Dict[str, str]:
    from kt_schema_loader import SCHEMA

    title_to_id = {s["title"]: s["id"] for s in SCHEMA}
    for sec in knowledge_object.get("sections") or []:
        if sec.get("id") == "kt_coverage":
            return {title_to_id.get(r.get("Domain"), r.get("Domain")): r.get("Coverage") for r in sec.get("_coverage_rows") or []}
    return {}


def repeated_sentences(texts: Dict[str, List[str]]) -> List[str]:
    seen, repeats = set(), []
    for strings in texts.values():
        for text in strings:
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                key = norm(sentence)
                if len(key.split()) < 5 or _PLACEHOLDER.search(sentence):
                    continue
                if key in seen:
                    repeats.append(sentence)
                seen.add(key)
    return repeats


def sourced_units(knowledge_object: Dict[str, Any]) -> List[tuple]:
    """(unit text, its source sentences) for every fact-bearing unit a reader
    sees in a real section (P1-2), normalised."""
    from knowledge.evidence import block_units, source_lookup, unit_sources

    lookup = source_lookup(knowledge_object)
    units = []
    for sec in knowledge_object.get("rendered_sections") or []:
        if sec.get("section_id") in _DIGESTS:
            continue
        for block in sec.get("blocks") or []:
            if _quotes_on_purpose(block):
                continue
            for i, text in enumerate(block_units(block) or []):
                quotes = [norm(lookup[n]["quote"]) for n in unit_sources(block, i) if n in lookup]
                units.append((norm(text), quotes))
    return units


def unsourced_facts(golden: Dict[str, Any], knowledge_object: Dict[str, Any], facts: List[Dict[str, Any]]) -> List[str]:
    """Facts the document shows where no unit showing them points to a
    transcript sentence that states them (P1-2)."""
    units = sourced_units(knowledge_object)
    by_id = {f["id"]: f for f in golden["facts"]}
    missing = []
    for fact in facts:
        if fact["status"] == "lost":
            continue
        spec = by_id[fact["id"]]
        needles = [n for n in [norm(spec["quote"])] + [norm(a) for a in spec.get("alt", [])] if n]
        terms = [norm(t) for t in spec.get("terms", [])]

        def shows(text):
            return any(n in text for n in needles) or (terms and all(t in text for t in terms))

        showing = [quotes for text, quotes in units if shows(text)]
        if showing and not any(shows(" ".join(quotes)) for quotes in showing):
            missing.append(fact["id"])
    return missing


def score(golden: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    ko = result.get("knowledge_object") or {}
    texts = section_texts(ko)
    normed = {sid: norm(" ".join(strings)) for sid, strings in texts.items()}
    facts = []
    for fact in golden["facts"]:
        needles = [norm(fact["quote"])] + [norm(a) for a in fact.get("alt", [])]
        found = {sid for sid, text in normed.items() for n in needles if n and n in text}
        # A fact the document shows as a table row ("The clearing service |
        # Clearing team"): all its terms in one rendered string.
        terms = [norm(t) for t in fact.get("terms", [])]
        if terms:
            found |= {sid for sid, strings in texts.items() if any(all(t in norm(s) for t in terms) for s in strings)}
        found = sorted(found)
        allowed = set(fact["in"])
        status = "correct" if allowed & set(found) else ("wrong" if found else "lost")
        facts.append({"id": fact["id"], "quote": fact["quote"], "status": status, "found_in": found,
                      "expected": fact["in"], "violations": sorted(set(found) & set(fact.get("not_in", [])))})
    buckets = coverage_buckets(ko)
    conflicts = " ".join(f"{c.get('a', '')} {c.get('b', '')}" for c in ko.get("_conflicts") or [])
    total = len(facts) or 1
    return {
        "id": golden["id"],
        "holdout": bool(golden.get("holdout")),
        "status": result.get("status"),
        "facts": facts,
        "accuracy": round(sum(f["status"] == "correct" for f in facts) / total, 3),
        "wrong": [f["id"] for f in facts if f["status"] == "wrong"],
        "lost": [f["id"] for f in facts if f["status"] == "lost"],
        "violations": [f["id"] for f in facts if f["violations"]],
        "false_missing": [s for s in golden.get("covered", []) if buckets.get(s) == "Missing"],
        "false_covered": [s for s in golden.get("missing", []) if buckets.get(s) not in ("Missing", None)],
        "missing_conflicts": [c for c in golden.get("conflicts", []) if c.lower() not in conflicts.lower()],
        "repeats": repeated_sentences(texts),
        "unsourced": unsourced_facts(golden, ko, facts),
        "evidence": ko.get("_evidence_stats") or {},
        "buckets": buckets,
    }


def run_golden(golden: Dict[str, Any]) -> Dict[str, Any]:
    """Run one golden through the real pipeline with no LLM (deterministic,
    no quota). The caller is responsible for patching providers off."""
    import pipeline
    from devops_transcription import clean_transcript

    job_id = f"golden-{golden['id']}"
    try:
        result = pipeline.run_kt_pipeline(job_id, clean_transcript(golden["transcript"]))
    finally:
        pipeline.JOB_STORE.delete(job_id)
    return score(golden, result)


def run_all(goldens=None) -> List[Dict[str, Any]]:
    import ai
    import pipeline
    from context_mapper import ContextMappingPipeline
    from kt_schema_loader import SCHEMA

    saved = (pipeline.get_llm_provider, ai.get_llm_provider, pipeline.record_candidates, pipeline.MAPPER_PIPELINE)
    pipeline.get_llm_provider = ai.get_llm_provider = lambda: None
    pipeline.record_candidates = lambda *a, **k: None
    pipeline.MAPPER_PIPELINE = ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)
    try:
        return [run_golden(g) for g in (goldens or load_goldens())]
    finally:
        pipeline.get_llm_provider, ai.get_llm_provider, pipeline.record_candidates, pipeline.MAPPER_PIPELINE = saved


def baseline_from(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The floor each golden must not fall below."""
    return {s["id"]: {"accuracy": s["accuracy"], "wrong": len(s["wrong"]), "violations": len(s["violations"]),
                      "false_missing": len(s["false_missing"]), "false_covered": len(s["false_covered"])}
            for s in scores}


def load_baseline() -> Dict[str, Any]:
    if not os.path.exists(BASELINE_PATH):
        return {}
    with open(BASELINE_PATH, encoding="utf-8") as fh:
        return json.load(fh)
