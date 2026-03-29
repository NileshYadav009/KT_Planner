#!/usr/bin/env python3
"""
Trace each stage of the ContextMapping Pipeline to find where data is lost
"""

import json
import logging
import sys
from context_mapper import (
    ContextMappingPipeline,
    segment_sentences,
    AudioSegment
)

# Suppress logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

# Load schema
with open("kt_schema_new.json") as f:
    SCHEMA = json.load(f)["sections"]

print("=" * 100)
print("STAGE-BY-STAGE PIPELINE TRACE")
print("=" * 100)

# Test transcript
test_transcript = """
This is the platform overview. We have a cloud native order processing system.
The architecture includes microservices deployed on Kubernetes.
We have staging, QA, and production environments.
Jenkins handles CI/CD for all deployments.
"""

# Create segments (as main.py gets from Whisper)
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

print(f"\nINPUT:")
print(f"  Transcript length: {len(test_transcript)} chars")
print(f"  Whisper segments: {len(segments)}")

# Manually run each stage of the pipeline
pipeline = ContextMappingPipeline(SCHEMA, similarity_threshold=0.30)

print(f"\n" + "=" * 100)
print("STAGE 1: Convert Whisper segments to AudioSegment objects")
print("=" * 100)

audio_segments = [
    AudioSegment(
        text=seg.get("text", ""),
        start=seg.get("start", 0.0),
        end=seg.get("end", 0.0),
        avg_logprob=seg.get("avg_logprob", -1.0),
        speaker=seg.get("speaker")
    )
    for seg in segments
]
print(f"✓ Created {len(audio_segments)} AudioSegment objects")

print(f"\n" + "=" * 100)
print("STAGE 2: Segment audio segments into sentences")
print("=" * 100)

sentences = segment_sentences(audio_segments)
print(f"✓ Segmented into {len(sentences)} Sentence objects")
for s in sentences[:2]:
    print(f"  - {s.text[:50]}")

print(f"\n" + "=" * 100)
print("STAGE 3: Classify sentences")
print("=" * 100)

texts = [s.text for s in sentences]
embeddings = pipeline.classifier.model.encode(texts, convert_to_tensor=True)

classified_sentences = []
for i, s in enumerate(sentences):
    window = 3
    start = max(0, i - window)
    end = min(len(sentences), i + window + 1)
    neighbor_embs = [embeddings[j] for j in range(start, end) if j != i]
    cs = pipeline.classifier.classify_sentence(s, sent_embedding=embeddings[i], neighbor_embeddings=neighbor_embs)
    classified_sentences.append(cs)

assigned = sum(1 for cs in classified_sentences if cs.primary_classification)
unassigned = sum(1 for cs in classified_sentences if cs.is_unassigned)
print(f"✓ Classified {len(classified_sentences)} sentences")
print(f"  - Assigned: {assigned}")
print(f"  - Unassigned: {unassigned}")

for i, cs in enumerate(classified_sentences[:3]):
    if cs.primary_classification:
        print(f"  [{i}] {cs.sentence.text[:40]} → {cs.primary_classification.section_title}")
    else:
        print(f"  [{i}] {cs.sentence.text[:40]} → UNASSIGNED")

print(f"\n" + "=" * 100)
print("STAGE 4: Boost classifications (check policy)")
print("=" * 100)

# The pipeline.process calls all stages, the policy boost happens here
# For now just show classified_sentences is intact
print(f"✓ Classified sentences still: {len(classified_sentences)}")
print(f"  - With primary_classification: {sum(1 for cs in classified_sentences if cs.primary_classification)}")

print(f"\n" + "=" * 100)
print("NOW RUN FULL PIPELINE AND COMPARE")
print("=" * 100)

try:
    kt = pipeline.process("test_job", test_transcript, segments)
    print(f"✓ Full pipeline completed")
    print(f"  - section_content keys: {len(kt.section_content)}")
    print(f"  - Total sentences in SC: {sum(len(v.get('sentences', [])) for v in kt.section_content.values())}")
    
    if len(kt.section_content) == 0:
        print(f"\n⚠️ CRITICAL BUG FOUND:")
        print(f"  - After manual stage 3: {assigned} sentences assigned")
        print(f"  - After full pipeline: 0 sentences in section_content")
        print(f"  - Something between stage 3 and assembly is clearing the data!")
    
except Exception as e:
    import traceback
    print(f"✗ Pipeline failed:")
    traceback.print_exc()

print(f"\n" + "=" * 100)
