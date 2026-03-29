#!/usr/bin/env python3
"""
Ultra-detailed debug: intercept the pipeline to see what happens to each sentence
"""

import json
import logging
from context_mapper import (
    ContextMappingPipeline, 
    ContextMappingOrchestrator,
    segment_sentences,
    AudioSegment
)

# Suppress verbose logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

# Load schema
with open("kt_schema_new.json") as f:
    SCHEMA = json.load(f)["sections"]

print("=" * 100)
print("ULTRA-DETAILED PIPELINE TRACE")
print("=" * 100)

# Test transcript
test_transcript = """
This is the platform overview. We have a cloud native order processing system.
The architecture includes microservices deployed on Kubernetes.
We have staging, QA, and production environments.
"""

# Create segments
segments = []
for i, line in enumerate(test_transcript.split(".")):
    line = line.strip()
    if line:
        segments.append({
            "text": line,
            "start": float(i),
            "end": float(i+1),
            "avg_logprob": -0.5,
            "speaker": None
        })

print(f"\n1. INPUT: {len(segments)} segments")

# Run orchestrator (not just pipeline) to see full processing
orchestrator = ContextMappingOrchestrator(SCHEMA, similarity_threshold=0.30)

print(f"\n2. RUNNING ORCHESTRATOR.EXECUTE()...")
try:
    result = orchestrator.execute(test_transcript, segments)
    kt_obj = result["kt_obj"]
    
    print(f"   ✓ Execution completed")
    print(f"   Coverage percent: {kt_obj.overall_coverage_percent:.1f}%")
    print(f"   Section content keys: {len(kt_obj.section_content)}")
    
    # Check what's in section_content
    for sec_id, content in kt_obj.section_content.items():
        sentences = content.get("sentences", [])
        if sentences:
            print(f"   {sec_id}: {len(sentences)} sentences")
    
    # Show full breakdown
    print(f"\n3. DETAILED BREAKDOWN:")
    total = 0
    for sec_id in sorted(kt_obj.section_content.keys()):
        sc = kt_obj.section_content[sec_id]
        count = len(sc.get("sentences", []))
        confidence = sc.get("confidence", 0.0)
        total += count
        if count > 0:
            print(f"   {sec_id:30s}: {count:2d} sentences (conf={confidence:.2f})")
            for s in sc.get("sentences", [])[:2]:
                text = s.get("text", "")[:60]
                print(f"       - {text}")
    
    print(f"\n   TOTAL: {total} sentences in section_content")
    
    if total == 0:
        print(f"\n   ⚠️ STILL EMPTY! This is a critical bug.")
        print(f"   The orchestrator is not populating section_content.")
        
except Exception as e:
    import traceback
    print(f"   ✗ FAILED: {e}")
    traceback.print_exc()

print("=" * 100)
