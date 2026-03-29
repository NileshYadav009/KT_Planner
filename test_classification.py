#!/usr/bin/env python3
"""
Test script to debug sentence classification in the context mapping pipeline.
"""

import json
import sys
from context_mapper import ContextMappingPipeline, segment_sentences, Sentence

# Load schema
with open("kt_schema_new.json") as f:
    SCHEMA = json.load(f)["sections"]

# Sample transcript from user
SAMPLE_TRANSCRIPT = """
Cloud Native Order Processing Platform is our core system. 
prod staging QA Dev environments are available.
Jenkins pipeline handles CI/CD deployments.
pod crashing looping errors occur in production.
Kubernetes rollback procedure takes 5 minutes.
Terraform modules manage infrastructure.
"""

# Create pipeline
pipeline = ContextMappingPipeline(SCHEMA)

# Create mock audio segments
mock_segments = [{"text": s, "start": 0.0, "end": 1.0, "avg_logprob": -0.5} 
                  for s in SAMPLE_TRANSCRIPT.split(".") if s.strip()]

print(f"Input transcript: {len(mock_segments)} segments")
print("=" * 80)

# Run pipeline
try:
    kt = pipeline.process("test_job", SAMPLE_TRANSCRIPT, mock_segments)
    
    print(f"\nProcessing complete. Coverage results:")
    print(f"Overall coverage: {kt.overall_coverage_percent:.1f}%")
    print(f"Overall risk: {kt.overall_risk_score:.2f}")
    print()
    
    # Show coverage for each section
    for sec_id in sorted(kt.coverage.keys()):
        cov = kt.coverage[sec_id]
        content = kt.section_content.get(sec_id, {})
        sentences = content.get("sentences", [])
        print(f"{sec_id:40s}: {cov.status:10s} ({len(sentences):2d} sentences, {cov.confidence_score:.2f} confidence)")
        if sentences:
            for sent in sentences[:2]:  # Show first 2
                print(f"  - {sent.get('text', '')[:70]}")
    
    print("\n" + "=" * 80)
    print("Classification details for first few sentences:")
    
    # Get detailed classification info
    from context_mapper import segment_sentences, Sentence
    sentences = segment_sentences([{"text": s, "start": 0.0, "end": 1.0, "avg_logprob": -0.5} 
                                   for s in SAMPLE_TRANSCRIPT.split(".") if s.strip()])
    
    for i, sent in enumerate(sentences[:5]):
        print(f"\nSentence {i}: {sent.text[:70]}")
        classified = pipeline.classifier.classify_sentence(sent)
        if classified.primary_classification:
            print(f"  Primary: {classified.primary_classification.section_title} (conf: {classified.primary_classification.confidence:.3f})")
            print(f"  Reason: {classified.primary_classification.reason}")
        else:
            print(f"  Primary: UNASSIGNED")
            
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
