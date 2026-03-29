#!/usr/bin/env python3
"""
Detailed tracer: Inspect what happens in stages 4-6
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(__file__))

from context_mapper import (
    ContextMappingPipeline, segment_sentences, AudioSegment,
    ContextClassifier, assemble_kt
)

# Load schema
with open('kt_schema_new.json') as f:
    schema = json.load(f)

# Test data  
whisper_output = [
    {"text": "This is the platform overview", "start": 0.0, "end": 2.5, "avg_logprob": -0.2},
    {"text": "We have a cloud native order processing system", "start": 2.5, "end": 5.0, "avg_logprob": -0.2},
    {"text": "The architecture includes microservices deployed on Kubernetes", "start": 5.0, "end": 8.0, "avg_logprob": -0.1},
    {"text": "We have staging, QA, and production environments", "start": 8.0, "end": 11.0, "avg_logprob": -0.15},
    {"text": "Jenkins handles CI/CD for all deployments", "start": 11.0, "end": 14.0, "avg_logprob": -0.2},
]

# Create pipeline
pipeline = ContextMappingPipeline(schema["sections"])

# Manually trace stages
print("=" * 80)
print("DETAILED STAGE-BY-STAGE TRACE")
print("=" * 80)

# Stage 1 - Pass the dict format that pipeline.process() expects
audio_segments = [
    AudioSegment(
        text=d["text"],
        start=d["start"],
        end=d["end"],
        avg_logprob=d["avg_logprob"]
    )
    for d in whisper_output
]

# Stage 2
sentences = segment_sentences(audio_segments)
print(f"\n✓ STAGE 2: Got {len(sentences)} sentences")

# Stage 3 - manually classify  
classifier = ContextClassifier(schema["sections"], similarity_threshold=0.30)
classified_sentences = []
for sent in sentences:
    classified = classifier.classify_sentence(sent)
    classified_sentences.append(classified)

print(f"✓ STAGE 3: Classified {len(classified_sentences)} sentences")
assigned = [s for s in classified_sentences if hasattr(s, 'primary_classification') and s.primary_classification]
print(f"  Assigned: {len(assigned)}")
if assigned:
    first = assigned[0]
    print(f"  First sentence: '{first.text[:40]}...'")
    print(f"    - primary_classification: {first.primary_classification}")

# Now check what the pipeline.process() returns
print(f"\n{'='*80}")
print("RUNNING FULL pipeline.process()")
print('='*80)

# Create a simple job ID
job_id = "test_job"
result_kt = pipeline.process(job_id, "", audio_segments)

print(f"\nResult from pipeline.process():")
print(f"  - Result type: {type(result_kt)}")

if isinstance(result_kt, dict):
    section_content = result_kt.get('section_content', {})
    print(f"  - section_content keys: {list(section_content.keys())}")
    print(f"  - Total sentences in section_content: {sum(len(v) for v in section_content.values())}")
    
    # Show which sections have content
    for sec_id, content_list in section_content.items():
        if content_list:
            print(f"    ✓ {sec_id}: {len(content_list)} sentences")
else:
    print(f"  - Result: {result_kt}")

print(f"\n{'='*80}")
if isinstance(result_kt, dict):
    if sum(len(v) for v in section_content.values()) == 0:
        print("❌ BUG CONFIRMED: section_content is empty!")
        print("\nPossible causes:")
        print("  1. Pipeline.process() calling completely different code path")
        print("  2. Stages 4-6 filtering out sentences aggressively")
        print("  3. assemble_kt() not being called or receiving wrong input")
    else:
        print(f"✅ section_content populated correctly!")

