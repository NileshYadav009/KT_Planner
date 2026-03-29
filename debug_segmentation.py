#!/usr/bin/env python3
"""
Debug: Check what segment_sentences is producing from Whisper segments
"""
import sys
import re
from dataclasses import dataclass
from typing import List

@dataclass
class AudioSegment:
    id: str
    text: str
    start: float
    end: float
    confidence: float
    
    def confidence_score(self):
        return self.confidence

# Simulate the actual Whisper segments from the user's uploaded content
# Based on debug_stages.py output which showed 5 input audio_segments
test_segments = [
    AudioSegment("seg1", "This is the platform overview", 0.0, 2.5, 0.9),
    AudioSegment("seg2", "We have a cloud native order processing system", 2.5, 5.0, 0.9),
    AudioSegment("seg3", "The architecture includes microservices deployed on Kubernetes", 5.0, 8.0, 0.95),
    AudioSegment("seg4", "We have staging, QA, and production environments", 8.0, 11.0, 0.92),
    AudioSegment("seg5", "Jenkins handles CI/CD for all deployments", 11.0, 14.0, 0.9),
]

# Test the segment_sentences logic
full_text = " ".join(seg.text for seg in test_segments)
print("=" * 80)
print("INPUT SEGMENTS:")
print("=" * 80)
for i, seg in enumerate(test_segments):
    print(f"  Segment {i+1}: {seg.text}")

print("\n" + "=" * 80)
print("CONCATENATED TEXT:")
print("=" * 80)
print(f"  {full_text}")

print("\n" + "=" * 80)
print("SEARCHING FOR SENTENCE BOUNDARIES [.!?]:")
print("=" * 80)
sentence_endings = re.findall(r"[.!?]", full_text)
print(f"  Found {len(sentence_endings)} sentence-ending punctuation marks")
if len(sentence_endings) == 0:
    print("  ❌ NO PUNCTUATION FOUND!")
    print("  → segment_sentences() will treat entire text as 1 MEGA-SENTENCE")

print("\n" + "=" * 80)
print("SPLIT RESULT:")
print("=" * 80)
# This is the exact regex from segment_sentences
sentence_texts = re.split(r"(?<=[.!?])\s+", full_text.strip())
print(f"  Number of sentences after split: {len(sentence_texts)}")
for i, sent in enumerate(sentence_texts):
    print(f"  Sentence {i+1}: {sent}")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
if len(sentence_texts) == 1:
    print("  ⚠️  CRITICAL BUG: Entire text is treated as 1 sentence!")
    print("  📌 SOLUTION: Add punctuation to transcript or modify segment_sentences()")
    print("  📌 OPTIONS:")
    print("     1. Add sentence-ending punctuation to Whisper output")
    print("     2. Split on clause boundaries (commas, 'and', 'but') if no punctuation")
    print("     3. Use spaCy/NLTK sentence tokenizer that doesn't require perfect punctuation")
else:
    print(f"  ✓ Text split into {len(sentence_texts)} sentences correctly")
