#!/usr/bin/env python3
"""
Test if assemble_kt can populate section_content from classified sentences
"""

import json
import logging
from context_mapper import (
    ContextClassifier,
    ContextRepair,
    segment_sentences,
    assemble_kt,
    AudioSegment,
    SectionCoverage
)

# Suppress logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

# Load schema
with open("kt_schema_new.json") as f:
    SCHEMA = json.load(f)["sections"]

print("=" * 100)
print("TEST: Can assemble_kt populate section_content?")
print("=" * 100)

# Test transcript
test_transcript = """
This is the platform overview. We have a cloud native order processing system.
The architecture includes microservices deployed on Kubernetes.
We have staging, QA, and production environments.
Jenkins handles CI/CD for all deployments.
"""

# Create AudioSegment objects
segments = []
for i, line in enumerate(test_transcript.split(".")):
    line = line.strip()
    if line:
        segments.append(AudioSegment(
            text=line,
            start=float(i),
            end=float(i+1),
            avg_logprob=-0.5,
            speaker=None
        ))

print(f"\n1. Created {len(segments)} audio segments")

# Create classifier and classify
classifier = ContextClassifier(similarity_threshold=0.30)
classifier.index_schema(SCHEMA)
print(f"2. Classifier ready")

# Segment into sentences
from context_mapper import segment_sentences
sentences = segment_sentences(segments)
print(f"3. Segmented into {len(sentences)} sentences")

# Classify each sentence
print(f"4. Classifying sentences...")
texts = [s.text for s in sentences]
embeddings = classifier.model.encode(texts, convert_to_tensor=True)

classified_sentences = []
for i, s in enumerate(sentences):
    cs = classifier.classify_sentence(s, sent_embedding=embeddings[i])
    classified_sentences.append(cs)
    if i < 2:
        print(f"   [{i}] {s.text[:60]}")
        print(f"       is_unassigned={cs.is_unassigned}, primary={cs.primary_classification.section_title if cs.primary_classification else 'None'}")

print(f"   Classified {len(classified_sentences)} total sentences")

# Check how many have primary_classification
assigned = sum(1 for cs in classified_sentences if cs.primary_classification)
print(f"   Assigned: {assigned}/{len(classified_sentences)}")

# Now test assemble_kt
print(f"\n5. Testing assemble_kt()...")

# Create dummy coverage and repair data
coverage = {}
for section in SCHEMA:
    coverage[section['id']] = SectionCoverage(
        section_id=section['id'],
        section_title=section['title'],
        status="missing",
        sentence_count=0,
        sentences=[],
        confidence_score=0.0,
        risk_score=1.0,
        required=section.get('required', False)
    )

repair = ContextRepair()
repaired_map = {}  # Empty: no repairs

assets = []

# Call assemble_kt
try:
    kt = assemble_kt(
        job_id="test",
        transcript=test_transcript,
        classified_sentences=classified_sentences,
        coverage=coverage,
        assets=assets,
        repaired_map=repaired_map
    )
    
    print(f"   ✓ assemble_kt completed")
    print(f"   section_content keys: {len(kt.section_content)}")
    print(f"   Total sentences in section_content: {sum(len(v.get('sentences', [])) for v in kt.section_content.values())}")
    
    # Show breakdown
    print(f"\n6. SECTION CONTENT BREAKDOWN:")
    for sec_id in sorted(kt.section_content.keys()):
        content = kt.section_content[sec_id]
        sents = content.get("sentences", [])
        if sents:
            print(f"   {sec_id:30s}: {len(sents):2d} sentences")
            for s in sents[:1]:
                print(f"       - {s.get('text', '')[:60]}")
    
    if sum(len(v.get('sentences', [])) for v in kt.section_content.values()) == 0:
        print(f"\n   ⚠️ PROBLEM: assemble_kt() also returns empty section_content!")
        print(f"   This means the bug is in the assemble_kt logic.")
    else:
        print(f"\n   ✓ SUCCESS: assemble_kt() populated section_content correctly!")
        
except Exception as e:
    import traceback
    print(f"   ✗ FAILED: {e}")
    traceback.print_exc()

print("=" * 100)
