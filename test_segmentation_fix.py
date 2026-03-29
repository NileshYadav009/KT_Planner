#!/usr/bin/env python3
"""
Test: Verify the new segment_sentences() logic with real stages from pipeline
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from context_mapper import segment_sentences, AudioSegment

# Simulate Whisper segments WITHOUT punctuation (like real Whisper output)
test_segments = [
    AudioSegment(text="This is the platform overview", start=0.0, end=2.5, avg_logprob=-0.2),
    AudioSegment(text="We have a cloud native order processing system", start=2.5, end=5.0, avg_logprob=-0.2),
    AudioSegment(text="The architecture includes microservices deployed on Kubernetes", start=5.0, end=8.0, avg_logprob=-0.1),
    AudioSegment(text="We have staging, QA, and production environments", start=8.0, end=11.0, avg_logprob=-0.15),
    AudioSegment(text="Jenkins handles CI/CD for all deployments", start=11.0, end=14.0, avg_logprob=-0.2),
]

print("=" * 80)
print("TESTING UPDATED segment_sentences()")
print("=" * 80)

print("\nINPUT: 5 Whisper segments WITHOUT punctuation")
for i, seg in enumerate(test_segments):
    print(f"  {i+1}. {seg.text}")

result = segment_sentences(test_segments)

print(f"\n✓ OUTPUT: {len(result)} sentences (was 1, now {len(result)})")
for i, sent in enumerate(result):
    print(f"  {i+1}. {sent.text}")
    print(f"     └─ Time: {sent.start:.1f}s-{sent.end:.1f}s | Confidence: {sent.audio_confidence:.2f}")

print("\n" + "=" * 80)
if len(result) > 1:
    print("✓ SUCCESS: Sentences now split correctly without punctuation!")
else:
    print("✗ FAILED: Still only 1 sentence")
