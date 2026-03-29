#!/usr/bin/env python3
"""
Debug script to trace sentence classification step-by-step.
Shows actual scores, threshold behavior, and why sentences aren't assigned.
"""

import json
import numpy as np
from context_mapper import (
    ContextMappingPipeline, 
    ContextClassifier,
    segment_sentences, 
    AudioSegment
)

# Load schema
with open("kt_schema_new.json") as f:
    SCHEMA = json.load(f)["sections"]

# Sample transcript
SAMPLE_TRANSCRIPT = """
Cloud Native Order Processing Platform is our core system. 
prod staging QA Dev environments are available.
Jenkins pipeline handles CI/CD deployments.
pod crashing looping errors occur in production.
Kubernetes rollback procedure takes 5 minutes.
Terraform modules manage infrastructure.
"""

print("=" * 100)
print("DEBUG: SENTENCE CLASSIFICATION ANALYSIS")
print("=" * 100)

# Create segments properly
segments = []
for text in SAMPLE_TRANSCRIPT.split("."):
    if text.strip():
        segments.append(AudioSegment(
            text=text.strip(),
            start=0.0,
            end=1.0,
            avg_logprob=-0.5,
            speaker=None
        ))

print(f"\n1. INPUT: {len(segments)} sentences from transcript")
for i, seg in enumerate(segments):
    print(f"   [{i}] {seg.text[:60]}")

# Create classifier
print(f"\n2. CLASSIFIER SETUP:")
classifier = ContextClassifier(similarity_threshold=0.30)
classifier.index_schema(SCHEMA)
print(f"   - Similarity threshold: {classifier.similarity_threshold}")
print(f"   - Filter threshold: {classifier.similarity_threshold * 0.6}")
print(f"   - Sections indexed: {len(classifier.section_metadata)}")
print(f"   - Section hints loaded: {len(classifier._section_hints)}")

# Now classify each sentence
print(f"\n3. CLASSIFICATION RESULTS:")
print()

total_classified = 0
total_unassigned = 0

for i, seg in enumerate(segments):
    print(f"   Sentence {i}: {seg.text[:60]}")
    
    # Classify without neighbors for simplicity
    classified = classifier.classify_sentence(seg)
    
    if classified.primary_classification:
        total_classified += 1
        print(f"      ✓ ASSIGNED to: {classified.primary_classification.section_title}")
        print(f"        Confidence: {classified.primary_classification.confidence:.4f}")
        print(f"        {classified.primary_classification.reason}")
    else:
        total_unassigned += 1
        print(f"      ✗ UNASSIGNED (all scores below threshold)")
        # Show top 3 candidates
        scores = []
        for sec_id in classifier.section_metadata.keys():
            emb = classifier.model.encode(seg.text, convert_to_tensor=True)
            sec_emb = classifier.section_embeddings[sec_id]
            sim = float((emb @ sec_emb.T).item()) if hasattr(emb, 'item') else 0.0
            scores.append((sec_id, classifier.section_metadata[sec_id]["title"], sim))
        
        scores.sort(key=lambda x: x[2], reverse=True)
        print(f"        Top 3 candidates:")
        for sec_id, title, sim in scores[:3]:
            print(f"          - {title:40s}: {sim:.4f} {'(would accept if boosted)' if sim >= -0.2 else ''}")
    print()

print("=" * 100)
print(f"SUMMARY: {total_classified} classified | {total_unassigned} unassigned")
print(f"PROBLEM: If unassigned > 0, sentences won't appear in coverage sections")
print("=" * 100)
