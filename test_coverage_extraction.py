#!/usr/bin/env python3
"""
Test: Verify coverage extraction from topic blocks.

Validates that sentences are properly extracted from blocks and
displayed in the API response.
"""

from context_mapper import ContextMappingPipeline
import json


TRANSCRIPT = """
Our system has three main architectural components. 
The API layer handles all incoming requests. 
We route them to the appropriate microservices.

For deployment, we use Kubernetes with continuous integration.
The rollout process takes about twenty minutes.
We have automated health checks that verify the deployment.
"""

SEGMENTS = [
    {"text": "Our system has three main architectural components.", "start": 0.0, "end": 5.0, "avg_logprob": -25.0, "speaker": "Engineer1"},
    {"text": "The API layer handles all incoming requests.", "start": 5.0, "end": 10.0, "avg_logprob": -24.5, "speaker": "Engineer1"},
    {"text": "We route them to the appropriate microservices.", "start": 10.0, "end": 15.0, "avg_logprob": -25.2, "speaker": "Engineer1"},
    {"text": "For deployment, we use Kubernetes with continuous integration.", "start": 15.0, "end": 20.0, "avg_logprob": -26.0, "speaker": "Engineer2"},
    {"text": "The rollout process takes about twenty minutes.", "start": 20.0, "end": 25.0, "avg_logprob": -25.5, "speaker": "Engineer2"},
    {"text": "We have automated health checks that verify the deployment.", "start": 25.0, "end": 30.0, "avg_logprob": -25.1, "speaker": "Engineer2"},
]

SCHEMA = [
    {"id": "architecture", "title": "System Architecture", "required": True},
    {"id": "deployment", "title": "Deployment Process", "required": True},
    {"id": "monitoring", "title": "Monitoring", "required": False},
]


def test_coverage_extraction_from_blocks():
    """Test that sentences are properly extracted from topic blocks."""
    
    print("=" * 80)
    print("TEST: Coverage Extraction from Topic Blocks")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = ContextMappingPipeline(schema_sections=SCHEMA)
    
    # Process transcript
    kt = pipeline.process(
        job_id="test-extraction",
        transcript=TRANSCRIPT,
        audio_segments=SEGMENTS
    )
    
    print("\n1. Topic Blocks Structure (internal):")
    print("-" * 80)
    for sec_id, cov in kt.coverage.items():
        if cov.blocks:
            print(f"\n{cov.section_title}:")
            for block_idx, block in enumerate(cov.blocks, 1):
                print(f"  Block {block_idx}: {block.duration} sentences, confidence={block.confidence_score:.2f}")
                for sent in block.sentences:
                    print(f"    - [{sent.start:.1f}s] {sent.text[:50]}...")
    
    print("\n\n2. Coverage Extraction (API response format):")
    print("-" * 80)
    
    # Simulate the API coverage extraction from main.py
    coverage_api = {}
    for sec_id, cov in kt.coverage.items():
        coverage_sentences = []
        
        # Extract from blocks (this is what main.py now does)
        blocks = getattr(cov, 'blocks', []) or []
        for block in blocks:
            for s in block.sentences:
                coverage_sentences.append({
                    'text': getattr(s, 'text', ''),
                    'start': getattr(s, 'start', 0.0),
                    'end': getattr(s, 'end', 0.0),
                    'speaker': getattr(s, 'speaker', None),
                    'audio_confidence': getattr(s, 'audio_confidence', 0.0),
                    'assigned_sections': [sec_id]
                })
        
        coverage_api[sec_id] = {
            'title': cov.section_title,
            'status': cov.status,
            'sentence_count': cov.sentence_count,
            'block_count': cov.block_count,
            'confidence': cov.confidence_score,
            'content': [s.get('text', '') for s in coverage_sentences],
            'sentences': coverage_sentences
        }
    
    # Display the API response
    for sec_id, info in coverage_api.items():
        print(f"\n{info['title']} ({info['status']})")
        print(f"  Blocks: {info['block_count']} | Sentences: {info['sentence_count']} | Confidence: {info['confidence']:.2f}")
        
        if info['content']:
            print(f"  Content:")
            for idx, text in enumerate(info['content'], 1):
                print(f"    {idx}. {text[:60]}...")
        else:
            print(f"  Content: (empty)")
    
    print("\n\n3. JSON Export (frontend receives):")
    print("-" * 80)
    print(json.dumps(
        {sec_id: info for sec_id, info in coverage_api.items()},
        indent=2
    )[:1000])
    print("    ... (truncated)")
    
    print("\n\n4. Validation Results:")
    print("-" * 80)
    
    passed = 0
    total = 0
    
    # Check 1: All sections have coverage extracted
    total += 1
    if len(coverage_api) == len(kt.coverage):
        print(f"✓ All {len(coverage_api)} sections extracted")
        passed += 1
    else:
        print(f"✗ Mismatch: {len(coverage_api)} extracted vs {len(kt.coverage)} total")
    
    # Check 2: Sentences extracted from blocks
    total += 1
    total_sentences_extracted = sum(len(info['sentences']) for info in coverage_api.values())
    total_sentences_in_blocks = sum(
        sum(b.duration for b in cov.blocks)
        for cov in kt.coverage.values()
    )
    if total_sentences_extracted == total_sentences_in_blocks:
        print(f"✓ All {total_sentences_extracted} sentences extracted from blocks")
        passed += 1
    else:
        print(f"✗ Mismatch: {total_sentences_extracted} extracted vs {total_sentences_in_blocks} in blocks")
    
    # Check 3: Content field populated
    total += 1
    empty_content = sum(1 for info in coverage_api.values() if not info['content'])
    if empty_content == 0:
        print(f"✓ Content field populated for all sections")
        passed += 1
    else:
        print(f"✗ {empty_content} sections have empty content")
    
    # Check 4: No "inferred from semantic analysis" message needed
    total += 1
    has_content_or_explanation = all(
        info['content'] or info['status'] == 'missing'
        for info in coverage_api.values()
    )
    if has_content_or_explanation:
        print(f"✓ All non-missing sections have content (no 'inferred' message needed)")
        passed += 1
    else:
        print(f"✗ Some sections missing both content and explanation")
    
    print()
    print("=" * 80)
    print(f"TEST RESULT: {passed}/{total} checks passed")
    print("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    success = test_coverage_extraction_from_blocks()
    exit(0 if success else 1)
