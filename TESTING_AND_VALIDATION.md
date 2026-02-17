# 🧪 Continuum Enterprise v2.0 - Testing & Validation Guide

## Overview

This guide helps you validate that the enterprise semantic mapper is working correctly and delivers on its guarantees.

---

## Phase 1: System Health Check (5 minutes)

### Test 1.1: Server is Running
```bash
curl http://localhost:8000/enterprise-status
```

**Expected Output:**
```json
{
  "system": "Continuum Enterprise Semantic Mapper",
  "version": "2.0",
  "capabilities": [
    "Sentence-level processing",
    "Semantic scoring",
    "Mixed sentence handling",
    ...
  ]
}
```

**Pass Criteria:** ✅ Response received with version "2.0"

---

## Phase 2: Core Feature Validation (20 minutes)

### Test 2.1: Semantic Classification (Not Keywords)

**Setup:** This test verifies that the system uses semantic understanding, not keyword matching.

```bash
curl -X POST http://localhost:8000/semantic-placement \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Our system utilizes container orchestration tools for managing microservices at scale. We leverage Kubernetes as the primary orchestration platform."
  }'
```

**Expected Result:**
- All sentences classified to "Architecture" (assuming you have this section)
- No keyword matching would require the word "architecture" to be present
- Semantic understanding captures "Kubernetes", "microservices", "orchestration" → Architecture

**Pass Criteria:** ✅ Both sentences assigned to same section with >0.80 confidence

---

### Test 2.2: Zero Duplication Guarantee

**Setup:** This test verifies the absolute guarantee of zero duplication.

```bash
curl -X POST http://localhost:8000/semantic-placement \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "We use Kubernetes. Docker containers are essential. We deploy using Kubernetes and Docker together in production."
  }'
```

**Expected Behavior:**
Option A: Smart splitting
- Sentence 1 → Architecture
- Sentence 2 → Architecture  
- Sentence 3 → Splits into clauses (Kubernetes→Architecture, Docker→Architecture OR Deployment)

Option B: No duplication
- Same sentence never assigned to multiple sections
- All content preserved (no data loss)

**Pass Criteria:** ✅ `duplicate_rate: 0.0` in response (absolute guarantee)

**Failure Criteria:** ❌ `duplicate_rate > 0.0` (bug - report immediately)

---

### Test 2.3: Mixed Sentence Splitting

**Setup:** Test intelligent clause splitting for complex sentences.

```bash
curl -X POST http://localhost:8000/semantic-placement \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "For high availability we use multiple zones but the infrastructure uses standard networking and all services are containerized with Docker for consistency."
  }'
```

**Expected Analysis:**
1. System detects semantic diversity (talks about multiple topics)
2. Splits on conjunctions/boundaries:
   - Clause 1: "high availability" + "multiple zones" → Architecture
   - Clause 2: "infrastructure" + "standard networking" → Architecture
   - Clause 3: "services are containerized with Docker" → Deployment

3. Each clause classified independently
4. All content preserved

**Pass Criteria:** ✅ All clauses classified, duplicate_rate = 0.0, all content present

---

### Test 2.4: Context Window Analysis

**Setup:** Test inference from surrounding context for unclear sentences.

```bash
curl -X POST http://localhost:8000/semantic-placement \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Our microservices architecture uses REST APIs. It scales well. We manage request routing with a load balancer."
  }'
```

**Expected Result:**
- Sentence 1: "microservices architecture" → Clear, high confidence (0.90+)
- Sentence 2: "It scales well" → Ambiguous alone, but context says it's about microservices → Architecture
- Sentence 3: "load balancer" → Clear, high confidence (0.90+)

**Pass Criteria:** ✅ Sentence 2 classified to Architecture using context (confidence 0.70-0.85)

---

### Test 2.5: Quality Metrics Reporting

**Setup:** Verify comprehensive quality metrics are generated.

```bash
curl -X POST http://localhost:8000/semantic-placement \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Typical session content here. Multiple sentences. About various topics."
  }'
```

**Expected Response Structure:**
```json
{
  "metrics": {
    "total_sentences": 3,
    "assigned_sentences": 2,
    "unclassified_sentences": 1,
    "duplicate_rate": 0.0,
    "avg_confidence": 0.85,
    "clauses_split": 0
  }
}
```

**Validation:** ✅ All metrics present and make mathematical sense

---

## Phase 3: Expert Training Mode (15 minutes)

### Test 3.1: Expert Correction Recording

**Setup:** Test that expert corrections are recorded and retrievable.

```bash
# Step 1: Process initial transcript
curl -X POST http://localhost:8000/semantic-placement \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "We deploy our application to AWS.",
    "job_id": "test-job-001"
  }'
```

**Note:** This returns a sentence assignment, e.g., classified to "Architecture"

```bash
# Step 2: Provide expert correction
curl -X POST http://localhost:8000/expert-correction \
  -H "Content-Type: application/json" \
  -d '{
    "sentence_id": "s1",
    "original_section": "architecture",
    "corrected_section": "deployment",
    "expert_notes": "Deployment is about provisioning and runtime"
  }'
```

**Expected Response:**
```json
{
  "status": "success",
  "message": "Expert correction recorded: s1 -> deployment"
}
```

**Pass Criteria:** ✅ Correction accepted, no errors

---

### Test 3.2: Training Statistics

```bash
curl http://localhost:8000/training-stats
```

**Expected Output:**
```json
{
  "training_statistics": {
    "total_corrections": 1,
    "sections_with_corrections": ["deployment"],
    "last_correction": "2026-02-16T10:30:00Z"
  }
}
```

**Pass Criteria:** ✅ Correction count increased, last_correction timestamp updated

---

### Test 3.3: Learning Impact (Optional)

**Setup:** Verify that corrections improve future classifications.

```bash
# After recording corrections:

# Step 1: Process similar new transcript
curl -X POST http://localhost:8000/semantic-placement \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Our deployment strategy uses AWS infrastructure."
  }'
```

**Expected Behavior:**
- Initially system might classify to Architecture
- After feedback corrections, similar sentences should show improved confidence toward Deployment
- *Note:* This requires multiple iterations to observe clearly

**Pass Criteria:** ✅ Confidence scores improve or remain consistent (not degrade)

---

## Phase 4: Edge Cases & Robustness (25 minutes)

### Test 4.1: Empty Input

```bash
curl -X POST http://localhost:8000/semantic-placement \
  -H "Content-Type: application/json" \
  -d '{"transcript": ""}'
```

**Expected:** Either graceful handling or clear error message
**Pass Criteria:** ✅ No crash, meaningful response

---

### Test 4.2: Very Long Sentences

```bash
curl -X POST http://localhost:8000/semantic-placement \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "While we initially considered a monolithic approach and evaluated various architectural patterns including monoliths, distributed systems, and hybrid approaches, we ultimately decided that microservices would be the best fit for our requirements because they provide better scalability and allow independent team ownership."
  }'
```

**Expected:** 
- Long sentence either kept intact or split intelligently
- High confidence if clearly about one topic
- Lower confidence if genuinely mixed
- Never duplicated

**Pass Criteria:** ✅ duplicate_rate = 0.0, all content preserved

---

### Test 4.3: Multiple Sections Equally Valid

```bash
curl -X POST http://localhost:8000/semantic-placement \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Testing is important for system reliability."
  }'
```

**Expected:** 
- Could be "QA/Testing" section
- Could be "Architecture" (reliability)
- System picks best match with confidence score
- No crash even if ambiguous

**Pass Criteria:** ✅ Clear assignment with confidence score (even if low)

---

### Test 4.4: Special Characters & Punctuation

```bash
curl -X POST http://localhost:8000/semantic-placement \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "We use Kubernetes (K8s) and Docker; it's our standard... isn't it? Deployment: critical!"
  }'
```

**Expected:** 
- Punctuation handled gracefully
- Parenthetical info understood
- Contractions handled
- Emphasis (capitals, ellipsis) not causing errors

**Pass Criteria:** ✅ No parsing errors, proper classification

---

## Phase 5: Performance Testing (10 minutes)

### Test 5.1: Processing Speed

**Setup:** Test performance with typical-size transcript.

```bash
# Create test file with ~500 words (1500 tokens, ~10 sentences)
curl -X POST http://localhost:8000/semantic-placement \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Sentence 1 about architecture. Sentence 2 about deployment. Sentence 3 about testing. ... [10 more sentences]"
  }'
```

**Expected:** Response within 2-5 seconds

**Pass Criteria:** ✅ Response time < 10 seconds

---

### Test 5.2: Batch Processing

```bash
# Submit 5 different transcripts in succession
for i in {1..5}; do
  curl -X POST http://localhost:8000/semantic-placement \
    -H "Content-Type: application/json" \
    -d "{\"transcript\": \"Test transcript $i ...\"}"
done
```

**Expected:** All complete successfully, no memory leaks
**Pass Criteria:** ✅ All 5 requests succeed

---

## Scoring Checklist

### Green Light ✅ (Ready for Production)
- [ ] All Phase 1 tests pass (Health Check)
- [ ] All Phase 2 tests pass (Core Features)
- [ ] Zero duplicates guaranteed (duplicate_rate = 0.0)
- [ ] Confidence scores reasonable (0.70-0.95)
- [ ] Phase 3 tests pass (Learning Mode)
- [ ] Phase 4 tests pass (Edge Cases)
- [ ] Performance acceptable (< 10s per request)

### Yellow Light ⚠️ (Ready with Caveats)
- [ ] Most Phase 2 tests pass
- [ ] Occasional unclassified sentences (< 10%)
- [ ] Duplicate rate always 0.0
- [ ] Learning mode working but improvements slow

### Red Light 🔴 (Not Ready)
- [ ] Duplicate rate > 0.0 (critical bug)
- [ ] Crashes on edge cases
- [ ] Performance unacceptable (> 30s)
- [ ] Major Phase 2 feature failures

---

## Validation Report Template

```
===== ENTERPRISE SEMANTIC MAPPER v2.0 VALIDATION =====

Date: [DATE]
Tester: [NAME]

PHASE 1 - System Health:       [PASS/FAIL]
PHASE 2 - Core Features:       [PASS/FAIL]
PHASE 3 - Expert Training:     [PASS/FAIL]
PHASE 4 - Edge Cases:          [PASS/FAIL]
PHASE 5 - Performance:         [PASS/FAIL]

Duplicate Rate Guarantee:       ✅ [Always 0.0%]
Average Confidence:            [X]% (Target: > 85%)
Coherence Score:               [X]% (Target: > 80%)
Processing Time:               [X]s (Target: < 10s)

Issues Found:
1. [Issue 1]
2. [Issue 2]

Recommendations:
1. [Recommendation 1]
2. [Recommendation 2]

Overall Assessment: [READY/CONDITIONAL/NOT READY]

Signature: _________________
```

---

## Common Test Result Patterns

### Pattern A: Excellent Results ✅✅✅
```
duplicate_rate: 0.0%
avg_confidence: 0.89
coherence_score: 0.87
unclassified: 1 out of 15 (6%)
→ PRODUCTION READY
```

### Pattern B: Good Results ✅✅
```
duplicate_rate: 0.0%
avg_confidence: 0.76
coherence_score: 0.81
unclassified: 2 out of 15 (13%)
→ PRODUCTION READY (with expert feedback)
```

### Pattern C: Warning Signs ⚠️
```
duplicate_rate: 0.0% ✓
avg_confidence: 0.55
coherence_score: 0.62
unclassified: 5 out of 15 (33%)
→ SCHEMA ISSUE or LOW-QUALITY TRANSCRIPTS
```

### Pattern D: Critical Issue 🔴
```
duplicate_rate: 0.5% ← BUG!
→ STOP IMMEDIATELY, DEBUG SENTENCE REGISTRY
```

---

## Support

If tests fail:
1. **Check main.py:** Is SEMANTIC_MAPPER initialized?
2. **Check enterprise_semantic_mapper.py:** Are all classes imported?
3. **Check server logs:** What error do you see?
4. **Restart server:** Sometimes helps
5. **Report:** Include test output and error logs

---

## Next Steps After Validation

- [ ] Integrate with UI (show quality metrics)
- [ ] Set up monitoring (track duplicate_rate)
- [ ] Create automated tests
- [ ] Document learned patterns
- [ ] Train other team members

✨ **Once validated, Continuum Enterprise is ready for real-world use!**
