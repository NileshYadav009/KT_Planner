#!/usr/bin/env python3
"""
Demo: Coverage text polishing via Gemini (the production path).

This script reproduces EXACTLY what `main.py` does when building the coverage
payload sent to the frontend: it extracts the raw transcript fragments per
section, then runs them through `polish_coverage_text()` (which calls Gemini
when GEMINI_API_KEY is set, and falls back to local cleanup otherwise).

Use this to SEE the polish working end-to-end without running the web server:

    python demo_coverage_polish.py

NOTE on test_ecommerce_kt.py:
    That test prints coverage straight from context_mapper and never calls the
    polish step, so it will ALWAYS show raw fragments regardless of this fix.
    Its purpose is classification accuracy, not display polish.
"""
import json
import os
import sys

from context_mapper import ContextMappingPipeline

# Import the polish function the same way main.py does, so we exercise the
# production code path rather than a copy.
from ai import polish_coverage_text, GEMINI_ENABLED

# Reuse the canonical e-commerce transcript so the demo is directly comparable
# to the output you have been looking at.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))
from test_ecommerce_kt import TRANSCRIPT, build_segments  # noqa: E402


def build_coverage_with_polish(transcript, segments, schema):
    """Mirror of main.py's coverage assembly, with the polish pass included."""
    pipeline = ContextMappingPipeline(schema)
    kt = pipeline.process("demo-polish", transcript, segments)

    coverage = {}
    for sec_id, cov in kt.coverage.items():
        coverage_sentences = []
        blocks = getattr(cov, "blocks", []) or []
        for block in blocks:
            for s in block.sentences:
                coverage_sentences.append({"text": getattr(s, "text", "")})

        raw_content = [s.get("text", "") for s in coverage_sentences]

        # <-- THE FIX: professional polish pass on the way out
        try:
            polished = polish_coverage_text(sec_id, cov.section_title, raw_content)
            display_content = polished if polished else raw_content
        except Exception as e:
            print(f"  [warn] polish failed for {sec_id}: {e}; using raw")
            display_content = raw_content

        coverage[sec_id] = {
            "title": cov.section_title,
            "status": cov.status,
            "raw_content": raw_content,
            "content": display_content,
        }
    return coverage


def main():
    schema_path = os.path.join(os.path.dirname(__file__), "kt_schema_new.json")
    with open(schema_path) as f:
        schema = json.load(f)["sections"]

    cleaned, segments = build_segments(TRANSCRIPT)

    print("=" * 80)
    print("COVERAGE POLISH DEMO  (production path via main.py logic)")
    print("=" * 80)
    print(f"Gemini enabled : {GEMINI_ENABLED}")
    if not GEMINI_ENABLED:
        print("*** GEMINI_API_KEY is NOT set -> polish will use LOCAL cleanup only. ***")
        print("    Create a .env file with GEMINI_API_KEY=... and re-run to see LLM output.")
    print("=" * 80)

    coverage = build_coverage_with_polish(cleaned, segments, schema)

    for sec_id, info in coverage.items():
        if not info["raw_content"]:
            continue
        print(f"\n### {info['title'].upper()} ({info['status']})")
        print("--- RAW (as transcribed) ---")
        for t in info["raw_content"]:
            print(f"  - {t}")
        print("--- POLISHED (what the UI now shows) ---")
        for t in info["content"]:
            print(f"  > {t}")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
