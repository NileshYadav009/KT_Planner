"""
Integration test for 7-stage context mapping pipeline.

Usage:
    python test_pipeline.py
"""

import json
from context_mapper import (
    ContextMappingPipeline,
    AudioSegment,
    segment_sentences,
    serialize_kt
)

# Mock schema for testing
TEST_SCHEMA = [
    {
        "id": "architecture",
        "title": "System Architecture",
        "description": "High-level system design and components",
        "keywords": ["architecture", "design", "components", "system"],
        "required": True
    },
    {
        "id": "deployment",
        "title": "Deployment Process",
        "description": "How to deploy and release the system",
        "keywords": ["deploy", "release", "rollout", "kubernetes", "ci/cd"],
        "required": True
    },
    {
        "id": "troubleshooting",
        "title": "Troubleshooting Guide",
        "description": "Common issues and how to fix them",
        "keywords": ["troubleshoot", "debug", "error", "fix", "issue"],
        "required": False
    },
    {
        "id": "monitoring",
        "title": "Monitoring & Observability",
        "description": "How to monitor and observe system health",
        "keywords": ["monitor", "observe", "grafana", "logs", "metrics"],
        "required": False
    }
]

# Mock Whisper segments for testing
TEST_SEGMENTS = [
    {
        "text": "Our system architecture consists of three main components: API layer, service layer, and data layer.",
        "start": 0.0,
        "end": 5.0,
        "avg_logprob": -0.5,
        "speaker": "Engineer1"
    },
    {
        "text": "The API layer handles incoming requests and routes them to the appropriate service.",
        "start": 5.0,
        "end": 10.0,
        "avg_logprob": -0.4,
        "speaker": "Engineer1"
    },
    {
        "text": "For deployment we use Kubernetes with a continuous integration pipeline.",
        "start": 10.0,
        "end": 15.0,
        "avg_logprob": -0.6,
        "speaker": "Engineer2"
    },
    {
        "text": "The rollout process takes about 20 minutes with automated health checks.",
        "start": 15.0,
        "end": 20.0,
        "avg_logprob": -0.7,
        "speaker": "Engineer2"
    },
    {
        "text": "We monitor everything through Grafana dashboards with real-time metrics.",
        "start": 20.0,
        "end": 25.0,
        "avg_logprob": -0.5,
        "speaker": "Engineer1"
    },
    {
        "text": "If you see high latency check the database query logs.",
        "start": 25.0,
        "end": 30.0,
        "avg_logprob": -0.8,
        "speaker": "Engineer2"
    }
]

def test_segmentation():
    """Test sentence segmentation."""
    print("\n=== Testing Sentence Segmentation ===")
    segments = [AudioSegment(**seg) for seg in TEST_SEGMENTS]
    sentences = segment_sentences(segments)
    
    print(f"Segmented into {len(sentences)} sentences:")
    for i, sent in enumerate(sentences):
        print(f"  {i+1}. [{sent.start:.1f}s-{sent.end:.1f}s] {sent.text[:60]}...")
        print(f"     Audio confidence: {sent.audio_confidence:.2f}, Speaker: {sent.speaker}")
    
    assert len(sentences) > 0, "Should produce sentences"
    print("[PASS] Segmentation test passed")


def test_classification():
    """Test semantic classification."""
    print("\n=== Testing Classification ===")
    pipeline = ContextMappingPipeline(TEST_SCHEMA)
    segments = [AudioSegment(**seg) for seg in TEST_SEGMENTS]
    sentences = segment_sentences(segments)
    
    classified_count = 0
    for sent in sentences:
        cs = pipeline.classifier.classify_sentence(sent)
        if cs.primary_classification:
            print(f"[OK] '{sent.text[:50]}' -> {cs.primary_classification.section_id} ({cs.primary_classification.confidence:.2f})")
            classified_count += 1
        else:
            print(f"[??] '{sent.text[:50]}' -> UNASSIGNED")
    
    print(f"\nClassified {classified_count}/{len(sentences)} sentences")
    assert classified_count > 0, "Should classify at least some sentences"
    print("[PASS] Classification test passed")


def test_full_pipeline():
    """Test complete 7-stage pipeline."""
    print("\n=== Testing Full Pipeline ===")
    
    pipeline = ContextMappingPipeline(TEST_SCHEMA)
    
    full_transcript = " ".join(seg["text"] for seg in TEST_SEGMENTS)
    job_id = "test-job-001"
    
    kt = pipeline.process(job_id, full_transcript, TEST_SEGMENTS)
    
    print(f"\nPipeline Results:")
    print(f"  Job ID: {kt.job_id}")
    print(f"  Timestamp: {kt.timestamp}")
    print(f"  Overall Coverage: {kt.overall_coverage_percent:.1f}%")
    print(f"  Overall Risk Score: {kt.overall_risk_score:.2f}")
    print(f"  Sections Processed: {len(kt.coverage)}")
    print(f"  Sentences Analyzed: {len(kt.sentences)}")
    print(f"  Unassigned Sentences: {len(kt.unassigned_sentences)}")
    print(f"  Assets Extracted: {len(kt.assets)}")
    print(f"  Missing Required Sections: {kt.missing_required_sections}")
    
    print("\n  Coverage by Section:")
    for sec_id, cov in kt.coverage.items():
        print(f"    {sec_id}: {cov.status} (confidence={cov.confidence_score:.2f}, risk={cov.risk_score:.2f})")
    
    print("\n  Assets Detected:")
    for asset in kt.assets:
        print(f"    {asset.asset_type}: {asset.content} ({asset.detected_component})")
    
    # Test serialization
    print("\n  Serialization test...")
    serialized = serialize_kt(kt)
    assert isinstance(serialized, dict), "Should serialize to dict"
    assert "job_id" in serialized, "Should have job_id"
    assert "coverage" in serialized, "Should have coverage"
    print(f"    [OK] Serialized successfully ({len(json.dumps(serialized))} bytes)")
    
    # Validate coverage metrics
    assert kt.overall_coverage_percent >= 0 and kt.overall_coverage_percent <= 100, "Coverage should be 0-100%"
    assert kt.overall_risk_score >= 0 and kt.overall_risk_score <= 1, "Risk should be 0-1"
    
    print("[PASS] Full pipeline test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("  7-Stage Context Mapping Pipeline - Integration Tests")
    print("=" * 60)
    
    try:
        test_segmentation()
        test_classification()
        test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("  All tests passed! [OK]")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
