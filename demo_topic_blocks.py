#!/usr/bin/env python3
"""
Demonstration: Topic Blocks Preserve Paragraph Continuity

Instead of storing sentences scattered across coverage, the system now
groups consecutive related sentences into TOPIC BLOCKS.

This preserves:
- Sequential ordering
- Paragraph boundaries  
- Semantic continuity
- Temporal relationships
"""

from context_mapper import ContextMappingPipeline
import json


# Sample transcript with clearly separated topics
DEMO_TRANSCRIPT = """
Our system has three main architectural components. 
The API layer handles all incoming requests. 
We route them to the appropriate microservices.

For deployment, we use Kubernetes with continuous integration.
The rollout process takes about twenty minutes.
We have automated health checks that verify the deployment.

Monitoring is crucial for production systems.
We use Grafana dashboards for real-time visibility.
All metrics are collected and stored in a time-series database.

If you see high latency, check the database query logs.
Look for slow queries in the application layer.
The troubleshooting guide has detailed procedures.
"""

# Audio segments aligned with transcript
DEMO_SEGMENTS = [
    {"text": "Our system has three main architectural components.", "start": 0.0, "end": 5.0, "avg_logprob": -25.0, "speaker": "Engineer1"},
    {"text": "The API layer handles all incoming requests.", "start": 5.0, "end": 10.0, "avg_logprob": -24.5, "speaker": "Engineer1"},
    {"text": "We route them to the appropriate microservices.", "start": 10.0, "end": 15.0, "avg_logprob": -25.2, "speaker": "Engineer1"},
    
    {"text": "For deployment, we use Kubernetes with continuous integration.", "start": 15.0, "end": 20.0, "avg_logprob": -26.0, "speaker": "Engineer2"},
    {"text": "The rollout process takes about twenty minutes.", "start": 20.0, "end": 25.0, "avg_logprob": -25.5, "speaker": "Engineer2"},
    {"text": "We have automated health checks that verify the deployment.", "start": 25.0, "end": 30.0, "avg_logprob": -25.1, "speaker": "Engineer2"},
    
    {"text": "Monitoring is crucial for production systems.", "start": 30.0, "end": 35.0, "avg_logprob": -24.8, "speaker": "Engineer1"},
    {"text": "We use Grafana dashboards for real-time visibility.", "start": 35.0, "end": 40.0, "avg_logprob": -25.3, "speaker": "Engineer1"},
    {"text": "All metrics are collected and stored in a time-series database.", "start": 40.0, "end": 45.0, "avg_logprob": -25.4, "speaker": "Engineer1"},
    
    {"text": "If you see high latency, check the database query logs.", "start": 45.0, "end": 50.0, "avg_logprob": -26.2, "speaker": "Engineer2"},
    {"text": "Look for slow queries in the application layer.", "start": 50.0, "end": 55.0, "avg_logprob": -25.9, "speaker": "Engineer2"},
    {"text": "The troubleshooting guide has detailed procedures.", "start": 55.0, "end": 60.0, "avg_logprob": -26.1, "speaker": "Engineer2"},
]

# KT schema
SCHEMA = [
    {"id": "architecture", "title": "System Architecture", "required": True},
    {"id": "deployment", "title": "Deployment Process", "required": True},
    {"id": "monitoring", "title": "Monitoring & Observability", "required": False},
    {"id": "troubleshooting", "title": "Troubleshooting Guide", "required": False},
]


def main():
    print("=" * 80)
    print("TOPIC BLOCKS: Preserving Paragraph Continuity")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    pipeline = ContextMappingPipeline(schema_sections=SCHEMA)
    
    # Process the demo transcript
    kt = pipeline.process(
        job_id="demo-001",
        transcript=DEMO_TRANSCRIPT,
        audio_segments=DEMO_SEGMENTS
    )
    
    print("BEFORE (Old Approach - Lost Paragraph Continuity):")
    print("-" * 80)
    print("coverage['deployment'].sentences = [sentence_3, sentence_5, sentence_4]")
    print("  → Sentences scattered, no ordering guarantee")
    print("  → Paragraph boundaries lost")
    print("  → Sequential relationships destroyed")
    print()
    
    print("AFTER (New Approach - Topic Blocks Preserve Continuity):")
    print("-" * 80)
    
    # Display the new topic block structure
    for section_id, cov in kt.coverage.items():
        if cov.blocks:
            print(f"\n📦 Section: {cov.section_title} ({section_id})")
            print(f"   Status: {cov.status} | Blocks: {cov.block_count} | Confidence: {cov.confidence_score:.2f}")
            print()
            
            for block_idx, block in enumerate(cov.blocks, 1):
                print(f"   Block {block_idx}:")
                print(f"   ├─ Timeline: {block.start_time:.1f}s - {block.end_time:.1f}s")
                print(f"   ├─ Duration: {block.duration} consecutive sentences")
                print(f"   ├─ Speaker: {block.speaker}")
                print(f"   ├─ Confidence: {block.confidence_score:.2f}")
                print(f"   └─ Sentences:")
                
                for sent_idx, sent in enumerate(block.sentences, 1):
                    text_preview = (sent.text[:60] + "...") if len(sent.text) > 60 else sent.text
                    print(f"       {sent_idx}. [{sent.start:.1f}s] {text_preview}")
    
    print()
    print("=" * 80)
    print("KEY ADVANTAGES:")
    print("=" * 80)
    print("✓ Consecutive sentences grouped into blocks")
    print("✓ Paragraph boundaries preserved")
    print("✓ Temporal ordering maintained")
    print("✓ Semantic continuity respected")
    print("✓ Easy to reconstruct coherent paragraphs")
    print()
    
    # Show serialization with blocks
    print("JSON Structure (with blocks):")
    print("-" * 80)
    section_content = kt.section_content
    if section_content:
        first_section = list(section_content.values())[0]
        if first_section.get("blocks"):
            print(json.dumps({"blocks": first_section["blocks"][:1]}, indent=2)[:500])
            print("    ... (truncated)")
    
    print()
    print("=" * 80)
    print(f"Overall Coverage: {kt.overall_coverage_percent:.1f}%")
    print(f"Overall Risk Score: {kt.overall_risk_score:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
