#!/usr/bin/env python3
"""
Comprehensive debug test that mimics the exact flow in main.py
"""

import json
import sys
import logging
from context_mapper import ContextMappingPipeline, AudioSegment

# Suppress verbose logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

# Load schema
with open("kt_schema_new.json") as f:
    SCHEMA = json.load(f)["sections"]

print("=" * 100)
print("COMPREHENSIVE PIPELINE DEBUG TEST")
print("=" * 100)

# Create a realistic test transcript (like the user's DevOps KT)
test_transcript = """
This is the platform overview. We have a cloud native order processing system.
The architecture includes microservices deployed on Kubernetes.
We have staging, QA, and production environments.
Jenkins handles CI/CD for all deployments.
The main dashboard shows real-time monitoring.
Pod crashes can happen due to memory leaks.
We use Docker for containerization.
Kubernetes rollback procedure takes about 5 minutes.
The infrastructure is managed by Terraform.
We have cross-region failover capabilities.
Database replication is automatic.
Alerts are sent to PagerDuty.
We maintain uptime SLAs of 99.9%.
"""

print(f"\n1. INPUT TRANSCRIPT:")
print(f"   Length: {len(test_transcript)} chars")
print(f"   Preview: {test_transcript[:100]}...")

# Create mock Whisper segments (same format main.py receives)
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

print(f"\n2. MOCK WHISPER SEGMENTS: {len(segments)} segments")
for i, seg in enumerate(segments[:3]):
    print(f"   [{i}] {seg['text'][:60]}")

# Initialize pipeline (exact same way as main.py)
print(f"\n3. INITIALIZING PIPELINE...")
pipeline = ContextMappingPipeline(SCHEMA, similarity_threshold=0.30)
print(f"   Sections indexed: {len(pipeline.classifier.section_metadata)}")
print(f"   Similarity threshold: {pipeline.classifier.similarity_threshold}")

# Run processing (this is what main.py does at line 257)
print(f"\n4. RUNNING PIPELINE.PROCESS()...")
try:
    kt = pipeline.process("debug_job_id", test_transcript, segments)
    print(f"   ✓ Pipeline completed successfully")
    
    print(f"\n5. RESULTS:")
    print(f"   Overall coverage: {kt.overall_coverage_percent:.1f}%")
    print(f"   Overall risk: {kt.overall_risk_score:.2f}")
    print(f"   Total section_content keys: {len(kt.section_content)}")
    print(f"   Total sections in coverage: {len(kt.coverage)}")
    
    print(f"\n6. COVERAGE BREAKDOWN:")
    total_sentences = 0
    for sec_id in sorted(kt.section_content.keys()):
        sentences = kt.section_content[sec_id].get("sentences", [])
        sent_count = len(sentences)
        total_sentences += sent_count
        status = kt.coverage.get(sec_id, {}).status if sec_id in kt.coverage else "unknown"
        if sent_count > 0 or sec_id in kt.coverage:
            print(f"   {sec_id:35s}: {sent_count:2d} sentences [status: {status}]")
            if sentences:
                for sent in sentences[:1]:
                    print(f"       └─ {sent.get('text', '')[:70]}")
    
    print(f"\n7. SUMMARY:")
    print(f"   Total sentences classified: {total_sentences}")
    print(f"   Non-empty sections: {sum(1 for v in kt.section_content.values() if v.get('sentences'))}")
    print(f"   Empty sections: {sum(1 for v in kt.section_content.values() if not v.get('sentences'))}")
    
    if total_sentences == 0:
        print(f"\n   ⚠️ PROBLEM IDENTIFIED: No sentences were classified!")
        print(f"   This is the root cause of empty coverage.")
    else:
        print(f"\n   ✓ CLASSIFICATION WORKING: {total_sentences} sentences assigned to sections")
    
except Exception as e:
    import traceback
    print(f"   ✗ Pipeline FAILED with exception:")
    print(f"   {type(e).__name__}: {str(e)}")
    print(f"\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

print("=" * 100)
