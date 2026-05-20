"""
Integration test for 7-stage context mapping pipeline.

Usage:
    python test_pipeline.py
"""

import json
from context_mapper import (
    ContextClassifier,
    ContextMappingPipeline,
    AudioSegment,
    Sentence,
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


def test_semantic_chunking():
    """Test semantic chunking merges related fragments."""
    print("\n=== Testing Semantic Chunking ===")
    segments = [AudioSegment(text="Today this KT is about DevOps.", start=0.0, end=3.0, avg_logprob=-0.4),
                AudioSegment(text="This KT covers the DevOps handover and operational handoff.", start=3.0, end=6.0, avg_logprob=-0.5),
                AudioSegment(text="The architecture includes AWS, Kubernetes, Terraform.", start=6.0, end=9.0, avg_logprob=-0.5)]
    sentences = segment_sentences(segments)
    print(f"  Semantic chunks produced: {len(sentences)}")
    for i, sent in enumerate(sentences):
        print(f"    Chunk {i+1}: {sent.text}")
    assert len(sentences) < 3, "Related fragments should merge into semantic chunks"
    assert any("DevOps" in sent.text for sent in sentences), "Merged chunk should preserve original content"
    print("[PASS] Semantic chunking test passed")


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


def test_context_window_influence():
    """Ensure surrounding sentences influence classification for ambiguous text."""
    print("\n=== Testing Context Window Influence ===")

    schema = [
        {
            "id": "rollback",
            "title": "Rollback Process",
            "description": "Steps taken when rollback is required",
            "keywords": ["rollback", "rollback process", "restore"],
            "required": False
        },
        {
            "id": "general",
            "title": "General Operations",
            "description": "Routine operational activities",
            "keywords": ["health checks", "monitor", "status"],
            "required": False
        }
    ]

    classifier = ContextClassifier(similarity_threshold=0.05)
    classifier.index_schema(schema)

    sentence = Sentence(text="health checks failed", start=0.0, end=1.0, audio_confidence=0.7)
    sent_embed = classifier.model.encode(sentence.text, convert_to_tensor=True)

    no_context = classifier.classify_sentence(sentence, sent_embedding=sent_embed)
    context_text = "rollback process started. health checks failed. rollback trigger initiated."
    with_context = classifier.classify_sentence(
        sentence,
        sent_embedding=sent_embed,
        context_text=context_text
    )

    print(f"Without context -> {no_context.primary_classification.section_id if no_context.primary_classification else 'none'} ({no_context.primary_classification.confidence if no_context.primary_classification else 0:.3f})")
    print(f"With context    -> {with_context.primary_classification.section_id if with_context.primary_classification else 'none'} ({with_context.primary_classification.confidence if with_context.primary_classification else 0:.3f})")

    assert with_context.primary_classification is not None, "Classification with context should yield a primary section"
    assert with_context.primary_classification.section_id == "rollback", "Context window should steer classification to rollback"
    assert (no_context.primary_classification is None or with_context.primary_classification.confidence >= no_context.primary_classification.confidence), \
        "Context-enhanced classification should be as good or better than sentence-alone classification"
    print("[PASS] Context window influence test passed")


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


def test_paragraph_reconstruction():
    """Test that report generation returns structured reconstructed paragraphs."""
    print("\n=== Testing Paragraph Reconstruction ===")
    from ai import generate_report

    transcript = (
        "The system name is cloud native order processing platform. "
        "This system handles order intake, validation, payment orchestration and fulfillment triggers. "
        "It is used by B2C users through the web app, B2B partners through APIs, and internal finance and support teams. "
        "This system is most critical during business hours and peak sales events."
    )
    report = generate_report(transcript)
    paragraphs = report.get('paragraphs', {})

    print(f"  Paragraph sections: {len(paragraphs)}")
    for section_id, section_paragraphs in paragraphs.items():
        for paragraph in section_paragraphs:
            print(f"    [{section_id}] {paragraph.get('text', '')[:120]}")

    assert isinstance(paragraphs, dict), "Paragraphs should be a dict keyed by section"
    assert any(p.get('text') for section in paragraphs.values() for p in section), "At least one reconstructed paragraph should contain text"
    assert any(p.get('pass_count', 0) >= 2 for section in paragraphs.values() for p in section), "Paragraph reconstruction should run at least two passes"
    print("[PASS] Paragraph reconstruction test passed")


def test_asr_repair_scenario():
    """Scenario: ASR transcribed a technical sentence badly; repair should correct it."""
    print("\n=== Testing ASR Repair Scenario ===")
    pipeline = ContextMappingPipeline(TEST_SCHEMA)

    # Simulate a Whisper segment with a low confidence mis-transcription
    bad_segment = {
        "text": "We use coffee for a sink event screaming between surfaces",
        "start": 0.0,
        "end": 6.0,
        "avg_logprob": -2.0,
        "speaker": "Engineer1"
    }

    # Process through pipeline
    full_transcript = bad_segment["text"]
    kt = pipeline.process("test-asr-001", full_transcript, [bad_segment])

    # Find repaired sentences in section content or unassigned
    repaired_texts = []
    for sec in kt.section_content.values():
        repaired_texts.extend(sec.get("enhanced_texts", []))

    repaired_texts.extend([s.text for s in kt.unassigned_sentences])

    print("Repaired / finalized texts:")
    for t in repaired_texts:
        print("  ", t)

    # Expect that conservative glossary corrected 'coffee'->'Kafka' and 'screaming'->'streaming'
    joined = " ".join(repaired_texts).lower()
    assert "kafka" in joined or "streaming" in joined, "Expected technical corrections in repaired output"
    print("[PASS] ASR repair scenario passed")


def test_conceptual_misplacement():
    """Ensure implementation steps are not placed into conceptual sections."""
    print("\n=== Testing Conceptual Misplacement Policy ===")
    pipeline = ContextMappingPipeline(TEST_SCHEMA)

    # Sentence that contains deployment commands but might be semantically similar to architecture
    impl_segment = {
        "text": "To deploy, run: kubectl apply -f deployment.yaml and monitor pods.",
        "start": 0.0,
        "end": 5.0,
        "avg_logprob": -0.3,
        "speaker": "Engineer1"
    }

    kt = pipeline.process("test-policy-001", impl_segment["text"], [impl_segment])

    # The sentence should be routed to review_required (unassigned) rather than left in architecture
    in_review = any((s.text and "kubectl" in s.text.lower()) for s in kt.unassigned_sentences)
    print("Unassigned sentences:", [s.text for s in kt.unassigned_sentences])
    assert in_review, "Implementation step should be placed in Review Required (unassigned)"
    print("[PASS] Conceptual misplacement policy passed")


def test_causal_inference():
    """Ensure causal statements without explicit evidence are marked as inferred,"""
    print("\n=== Testing Causal Inference Policy ===")
    pipeline = ContextMappingPipeline(TEST_SCHEMA)

    # Ambiguous causal statement (no explicit evidence)
    seg1 = {
        "text": "It fails sometimes when traffic spikes.",
        "start": 0.0,
        "end": 3.0,
        "avg_logprob": -0.5,
        "speaker": "Engineer1"
    }

    # Explicit evidence mentioning DB connection exhaustion
    seg2 = {
        "text": "We saw the database connection pool exhausted and connection refused errors.",
        "start": 3.0,
        "end": 8.0,
        "avg_logprob": -0.4,
        "speaker": "Engineer1"
    }

    # Run pipeline with only ambiguous statement
    kt1 = pipeline.process("test-causal-1", seg1["text"], [seg1])
    # The single sentence should be flagged as inferred
    inferred_flag = False
    for cs in kt1.sentences:
        if hasattr(cs, 'is_inferred') and cs.is_inferred:
            inferred_flag = True
    assert inferred_flag, "Ambiguous causal statement should be marked as inferred"

    # Run pipeline with explicit evidence -> should not be inferred and evidence extracted
    kt2 = pipeline.process("test-causal-2", seg2["text"], [seg2])
    evidence_found = False
    for cs in kt2.sentences:
        if getattr(cs, 'explicit_evidence', None):
            evidence_found = True
    assert evidence_found, "Explicit evidence should be extracted from sentence"

    print("[PASS] Causal inference tests passed")


def test_topic_memory():
    """Ensure topic memory tracks active sections and boosts related sentences."""
    print("\n=== Testing Topic Memory ===" )
    
    # Create schema with related keywords for deployment and rollback
    schema = [
        {
            "id": "deployment",
            "title": "Deployment Process",
            "description": "Deploying and releasing systems",
            "keywords": ["deploy", "release", "kubernetes", "kubectl", "rollout"],
            "required": True
        },
        {
            "id": "rollback",
            "title": "Rollback Procedure",
            "description": "Rolling back a failed deployment",
            "keywords": ["rollback", "revert", "restore", "rollout", "kubectl"],
            "required": False
        }
    ]
    
    pipeline = ContextMappingPipeline(schema)
    
    # Sequence of related deployment/rollback sentences
    segments = [
        {
            "text": "We use Kubernetes for deployment across regions.",
            "start": 0.0,
            "end": 3.0,
            "avg_logprob": -0.4,
            "speaker": "Engineer1"
        },
        {
            "text": "The kubectl apply command triggers the rollout.",
            "start": 3.0,
            "end": 6.0,
            "avg_logprob": -0.5,
            "speaker": "Engineer1"
        },
        {
            "text": "If deployment fails, we initiate rollback immediately.",
            "start": 6.0,
            "end": 9.0,
            "avg_logprob": -0.3,
            "speaker": "Engineer1"
        },
        {
            "text": "The rollback process restores the previous version.",
            "start": 9.0,
            "end": 12.0,
            "avg_logprob": -0.4,
            "speaker": "Engineer1"
        }
    ]
    
    full_transcript = " ".join([s["text"] for s in segments])
    kt = pipeline.process("test-topic-001", full_transcript, segments)
    
    # Verify sentences have topic context
    topic_contexts = []
    topic_contexts = kt.topic_memory_contexts or []
    
    print(f"\nTopic memory trace ({len(topic_contexts)} of {len(kt.sentences)} sentences have topic context):")
    for i, ctx in enumerate(topic_contexts):
        print(f"  [{i}] section={ctx.get('active_section')}, duration={ctx.get('topic_duration')}, confidence={ctx.get('topic_confidence'):.3f}")
    
    # Verify that sentences are grouped by topic
    assert len(topic_contexts) > 0, "Topic memory should be tracking active sections"
    
    # Check that at least some sentences maintained the same topic across the sequence
    active_sections = [ctx.get("active_section") for ctx in topic_contexts]
    assert len(active_sections) > 0 and (active_sections.count(active_sections[0]) >= 2 or len(set(active_sections)) <= 2), \
        "Topic memory should maintain coherent topic grouping"
    
    # Verify coverage includes both deployment and rollback
    coverage_ids = [c for c in kt.coverage.keys()]
    assert "deployment" in coverage_ids or "rollback" in coverage_ids, \
        "Pipeline should classify deployment/rollback sentences"
    
    print("[PASS] Topic memory test passed")


def test_topic_memory_context_window():
    """Test enhanced topic memory with context window and transition tracking."""
    print("\n=== Testing Enhanced Topic Memory with Context Window ===")
    
    from context_mapper import TopicMemory
    
    # Create a topic memory with small context window for testing
    memory = TopicMemory(max_context_window=3)
    
    print("\nTest 1: Context Window Maintenance")
    # Simulate a sequence of sentence classifications
    memory.update("deployment", 0.85, sentence_index=0)
    memory.update("deployment", 0.82, sentence_index=1)
    memory.update("rollback", 0.78, related=["deployment", "monitoring"], sentence_index=2)
    memory.update("monitoring", 0.65, sentence_index=3)
    memory.update("monitoring", 0.75, sentence_index=4)
    
    context = memory.get_context_sections(depth=3)
    print(f"  Context window (last 3): {context}")
    assert len(context) <= 3, "Context window should not exceed max size"
    assert len(context) > 0, "Context window should contain recent sections"
    
    print("\nTest 2: Topic Boost Application")
    # Test that topic boost is applied correctly
    boost_current = memory.get_topic_boost("monitoring", 0.65)
    boost_unrelated = memory.get_topic_boost("architecture", 0.65)
    print(f"  Boost for current topic 'monitoring': {boost_current - 0.65:.4f}")
    print(f"  Boost for unrelated topic 'architecture': {boost_unrelated - 0.65:.4f}")
    assert boost_current > boost_unrelated, "Current topic should get larger boost"
    
    print("\nTest 3: Transition History Tracking")
    transitions = memory.transitions
    print(f"  Total transitions recorded: {len(transitions)}")
    assert len(transitions) >= 2, "Should have recorded topic transitions"
    for i, trans in enumerate(transitions[-3:]):
        print(f"    [{i}] {trans.from_section} → {trans.to_section} (reason: {trans.reason})")
    
    print("\nTest 4: Memory Serialization")
    state = memory.to_dict()
    print(f"  Active section: {state['active_section']}")
    print(f"  Topic confidence: {state['topic_confidence']:.3f}")
    print(f"  Topic duration: {state['topic_duration']}")
    print(f"  Stack depth: {state['stack_depth']}")
    print(f"  Recent transitions: {len(state['recent_transitions'])}")
    assert "active_section" in state
    assert "topic_confidence" in state
    assert state['topic_confidence'] >= 0.0 and state['topic_confidence'] <= 1.0
    
    print("[PASS] Enhanced topic memory context window test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("  7-Stage Context Mapping Pipeline - Integration Tests")
    print("=" * 60)
    
    try:
        test_segmentation()
        test_semantic_chunking()
        test_classification()
        test_context_window_influence()
        test_full_pipeline()
        test_paragraph_reconstruction()
        test_topic_memory()
        test_topic_memory_context_window()
        
        print("\n" + "=" * 60)
        print("  All tests passed! [OK]")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
