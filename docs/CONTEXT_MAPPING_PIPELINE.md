# 7-Stage Context Mapping Pipeline - Documentation

## Overview

The Sentence-Level Context Mapping Engine is a deterministic, AI-assisted pipeline that transforms raw transcripts into structured Knowledge Transfer (KT) documents. It implements all 7 required stages with zero content loss guarantee and confidence-based gap detection.

## Architecture Diagram

```
Audio/Video Input
      ↓
[STAGE 1] Audio Confidence Validation
      ↓
[STAGE 2] Sentence Segmentation
      ↓
[STAGE 3] Context Classification (Semantic)
      ↓
[STAGE 4] Contextual Repair & Enhancement
      ↓
[STAGE 5] Gap Detection & Coverage Analysis
      ↓
[STAGE 6] Screenshot & URL Extraction
      ↓
[STAGE 7] Structured KT Assembly
      ↓
Structured KT Output (JSON)
```

## Stage-by-Stage Details

### STAGE 1: Audio Confidence Validation

**Responsibility:** Extract and validate audio confidence from Whisper transcription.

**Input:**
- Whisper segments with `avg_logprob` metadata

**Processing:**
- Convert logprob (-2.0 to 0.0) to confidence score (0.0 to 1.0)
- Formula: `confidence = max(0, min(1, (logprob + 2.0) / 2.0))`
- Mark low-confidence segments for repair in Stage 4

**Output:**
- `AudioSegment` objects with:
  - `text`: Transcribed text
  - `start`, `end`: Timestamps
  - `avg_logprob`: Raw confidence metric
  - `speaker`: Speaker identification (if multi-speaker)

**Performance:** Instant (metadata parsing only)

---

### STAGE 2: Sentence Segmentation

**Responsibility:** Split transcript into clean, timestamp-aware sentences.

**Input:**
- Audio segments from Stage 1

**Processing:**
1. Concatenate segments preserving timestamps
2. Split by sentence boundaries (`.!?`)
3. Map each sentence to originating segments
4. Preserve both raw and cleaned versions
5. Compute average audio confidence per sentence
6. Track speaker continuity

**Output:**
- `Sentence` objects with:
  - `text`: Cleaned sentence
  - `raw_text`: Original text
  - `start`, `end`: Timestamps
  - `speaker`: Speaker ID
  - `audio_confidence`: Average confidence (0.0-1.0)
  - `segment_ids`: Contributing segment indices

**Rules:**
- Never split mid-technical term
- Preserve punctuation
- Normalize whitespace
- Maintain segment-to-sentence mapping for evidence linking

**Performance:** < 100ms for 20-minute transcript

---

### STAGE 3: Context Classification Engine

**Responsibility:** Semantically map sentences to KT schema sections.

**Input:**
- Sentences from Stage 2
- KT schema (indexed)

**Processing:**
1. Encode each sentence using `sentence-transformers` (all-MiniLM-L6-v2)
2. Encode each schema section with title + description + keywords
3. Compute cosine similarity between sentence and all sections
4. Rank sections by similarity score
5. Assign primary and secondary classifications
6. Mark unassigned if max similarity < threshold (default: 0.30)

**Output:**
- `ClassifiedSentence` objects with:
  - `primary_classification`: Best-matching section
  - `secondary_classifications`: Ranked alternatives
  - `is_unassigned`: Flag if no match above threshold
  - Classification metadata:
    - `section_id`, `section_title`
    - `confidence`: Similarity score (0.0-1.0)
    - `similarity_score`: Raw cosine similarity
    - `reason`: Explanation (e.g., "Semantic match (similarity=0.75)")

**Rules:**
- **Never discard sentences** — unassigned sentences tracked for manual review
- **Multi-section mapping:** Sentences can map to multiple sections
- **Threshold-driven:** Configurable similarity threshold (default 0.30)
- **Explainability:** Each classification includes reason

**Performance:**
- Embedding: ~50ms per 100 sentences
- Batch similarity: O(n*m) where n=sentences, m=sections

---

### STAGE 4: Contextual Repair & Enhancement

**Responsibility:** Fix low-confidence sentences without hallucinating content.

**Input:**
- classified sentences from Stage 3
- Context from surrounding sentences

**Repair Triggers:**
1. Audio confidence < 0.4
2. Semantic confidence < 0.35
3. Grammar issues detected

**Processing:**
1. Evaluate each classified sentence against triggers
2. If repair needed:
   - Normalize whitespace
   - Add missing punctuation
   - Preserve original meaning
   - Never add new technical content
3. Compare surrounding context for meaning inference
4. Record: original_text, improved_text, repair_reason

**Output:**
- Enhanced sentence text
- `RepairAction` record (if modified):
  - `reason`: Why repair was applied
  - `original`: Original text
  - `improved`: Fixed text
  - `confidence`: Repair confidence (usually 0.9)

**Rules:**
- **Never hallucinate:** Only fix grammar, spelling, clarity
- **Preserve meaning:** Keep technical accuracy
- **Dual recording:** Store both versions for audit
- **Confident repairs:** Only apply if confidence > threshold

**Performance:** < 50ms per 100 sentences

---

### STAGE 5: Gap Detection & Coverage Analysis

**Responsibility:** Identify missing/weak/covered sections and compute risk.

**Input:**
- Classified sentences from Stage 3
- Schema metadata (required flags)

**Processing:**
1. Group sentences by section
2. Count sentences per section
3. Assign coverage status:
   - `missing`: 0 sentences
   - `weak`: 1 sentence (configurable threshold)
   - `covered`: 2+ sentences
4. Compute confidence score per section
5. Calculate risk score:
   - `missing & required`: 1.0 (critical)
   - `weak & required`: 0.6 (high)
   - `covered & low_confidence`: 0.1 (low)
   - `missing & optional`: 0.5 (medium)
6. Aggregate overall coverage %

**Output:**
- `SectionCoverage` objects with:
  - `status`: missing|weak|covered
  - `sentence_count`: # of mapped sentences
  - `confidence_score`: Avg confidence (0.0-1.0)
  - `risk_score`: Risk metric (0.0-1.0)
  - `sentences`: List of mapped Sentence objects
- Overall metrics:
  - `overall_coverage_percent`: 100 * covered / total
  - `overall_risk_score`: Avg risk across sections
  - `missing_required_sections`: List of critical gaps

**Rules:**
- **No negative penalties:** Only flag gaps
- **Explainable:** Each gap has sentence evidence
- **Risk scoring:** Higher for required sections

**Performance:** O(n+m) where n=sentences, m=sections

---

### STAGE 6: Screenshot & URL Extraction

**Responsibility:** Detect and capture visual evidence of dashboards, consoles, and URLs.

**Input:**
- Sentences from Stage 2
- Video/audio file path

**Processing:**
1. **URL Extraction:**
   - Regex match: `https?://[^\s]+`
   - Detect component type: jenkins, grafana, kubernetes, github, aws, datadog
   - Record URL and timestamp

2. **Dashboard Detection:**
   - Pattern match in sentence text against known keywords
   - Component types: grafana, jenkins, kubernetes, github, aws, datadog
   - Mark for screenshot capture if detected

3. **Screenshot Capture (async):**
   - For each detected dashboard mention
   - Extract single frame at relevant timestamp
   - Save to `/static/screenshots/{job_id}_{component}_{index}.jpg`
   - Link screenshot to associated sentences

**Output:**
- `ExtractedAsset` objects with:
  - `asset_type`: "screenshot" | "url" | "screenshot_candidate"
  - `content`: URL or path
  - `detected_component`: Component name (grafana, jenkins, etc.)
  - `timestamp`: Time of detection
  - `sentence_ids`: Associated sentence indices

**Known Dashboards:**
- **Grafana:** `grafana|dashboard`
- **Jenkins:** `jenkins|build|pipeline`
- **Kubernetes:** `kubernetes|k8s|kubectl|dashboard`
- **GitHub:** `github|repository|repo`
- **AWS:** `aws|console|ec2|s3|cloudwatch`
- **Datadog:** `datadog|monitoring`

**Future Enhancements:**
- OCR for burned-in URLs
- Browser extension for real-time capture
- Headless Chromium for full-page capture
- DOM inspection for SPA dashboards

**Performance:** < 1s per screenshot (I/O bound)

---

### STAGE 7: Structured KT Assembly

**Responsibility:** Assemble all data into final JSON-serializable KT document.

**Input:**
- All outputs from Stages 2-6
- Repair records from Stage 4

**Processing:**
1. **Section-wise aggregation:**
   - Group sentences by section
   - Include both original and enhanced text
   - List all repair actions applied
   - Attach screenshots to sections

2. **Confidence aggregation:**
   - Per-section: average of sentence confidences
   - Overall: weighted by section importance

3. **Risk aggregation:**
   - Per-section: from Stage 5 analysis
   - Overall: average risk score

4. **Unassigned tracking:**
   - Collect all unassigned sentences
   - Flag for manual review

5. **Metadata:**
   - Timestamp (ISO 8601)
   - Job ID
   - Transcript (full)
   - Processing statistics

**Output:**
- `StructuredKT` object with:
  ```python
  StructuredKT(
      job_id: str
      transcript: str  # Full transcript
      sentences: List[Sentence]  # All sentences with metadata
      coverage: Dict[str, SectionCoverage]  # Per-section analysis
      section_content: Dict[str, {
          section_id, section_title, sentences,
          enhanced_texts, repair_actions, screenshots,
          confidence, sentence_count
      }]
      missing_required_sections: List[str]
      unassigned_sentences: List[Sentence]
      assets: List[ExtractedAsset]
      overall_coverage_percent: float
      overall_risk_score: float
      timestamp: str
  )
  ```

**Serialization:**
- JSON-friendly via `serialize_kt(kt)` function
- Produces compact but complete JSON output
- Includes summaries and previews for large sections

---

## Quality Control Rules

1. **Zero Content Loss:** No sentences discarded; unassigned sentences tracked
2. **Never Hallucinate:** Grammar fixes only; no invented technical content
3. **Grammar Improvement:** Allowed (capitalization, punctuation, whitespace)
4. **Context Inference:** Only within transcript bounds
5. **Missing Section Flags:** Explicit and evidence-linked
6. **Human Override Support:** All automated decisions reversible
7. **Multi-section Mapping:** Sentences can belong to multiple sections
8. **Non-linear Speech:** No ordering assumptions; handles out-of-order explanations
9. **Speaker Continuity:** Tracked and preserved
10. **Incremental KT:** Supports multi-session concatenation

---

## Performance Requirements

| Stage | Input | Processing | Output | Time (20min) |
|-------|-------|-----------|--------|------------|
| 1 | Segments | Logprob parsing | Confidence | <10ms |
| 2 | Segments | Segmentation | Sentences | <100ms |
| 3 | Sentences | Embeddings + similarity | Classifications | ~500ms |
| 4 | Classified | Grammar repair | Enhanced | <50ms |
| 5 | Classified | Aggregation | Coverage | <50ms |
| 6 | Sentences | Regex + capture | Assets | ~5s |
| 7 | All | Assembly | StructuredKT | <100ms |
| **Total** | - | - | - | **~6s** |

**Target:** <60 seconds for 20-minute KT (5500+ segments)

---

## Integration with Continuum

### API Endpoints

```
POST /upload
├─ Returns: job_id
├─ Kicks off background processing
└─ Triggers all 7 stages via ContextMappingPipeline.process()

GET /status/{job_id}
├─ Returns: {status, progress, transcript, coverage, ...}
└─ Polls during processing

GET /kt/{job_id}
├─ Returns: Full StructuredKT (serialized)
├─ Includes: sections, sentences, coverage, risk, assets
└─ Available after status=completed

GET /coverage/{job_id}
├─ Returns: Coverage analysis with risk scoring
├─ Includes: missing_sections, risk_heatmap
└─ Quick query endpoint
```

### Schema Integration

KT schema (`kt_schema_new.json`) must include:
```json
{
  "sections": [
    {
      "id": "section_id",
      "title": "Section Title",
      "description": "Detailed description",
      "keywords": ["keyword1", "keyword2"],
      "required": true/false
    }
  ]
}
```

---

## Configuration

```python
# In main.py initialization
MAPPER_PIPELINE = ContextMappingPipeline(
    schema_sections=SCHEMA,
    similarity_threshold=0.30,  # For Stage 3 classification
    audio_confidence_threshold=0.4  # For Stage 4 repair
)
```

---

## Testing

Run integration tests:
```bash
python test_pipeline.py
```

Tests validate:
- Sentence segmentation
- Semantic classification
- Full pipeline end-to-end
- JSON serialization
- Coverage metrics

---

## Future Enhancements

1. **LLM Fallback:** Claude/GPT-4 for complex classification edge cases
2. **Explainability Logs:** Detailed reasoning for each classification decision
3. **Multi-tenant Support:** Per-organization schema and templates
4. **Incremental KT:** Resume from partial transcripts
5. **Custom Embeddings:** Fine-tuned model per organization
6. **Real-time Streaming:** Process as micro-segments arrive
7. **Multilingual Support:** Beyond English transcription
8. **Video Analysis:** Frame-by-frame classification
9. **Custom Repair Rules:** Organization-specific grammar/terminology

---

## Troubleshooting

### Low Coverage %?
- Check schema `keywords` field completeness
- Lower `similarity_threshold` slightly (0.25 instead of 0.30)
- Verify transcript quality and audio confidence

### Missing Required Sections?
- Review Stage 5 output for which sentences mapped where
- Check if relevant speech was inaudible (Stage 1)
- Use `GET /kt/{job_id}` to inspect `unassigned_sentences`

### High Risk Score?
- Focus on sections with status="weak" or "missing"
- Request follow-up recording for those topics
- Use visual evidence (screenshots) to validate understanding

### Slow Processing?
- Reduce batch size in Stage 6 (screenshot capture)
- Use smaller embedding model (distilbert-base-uncased)
- Enable GPU acceleration if available

---

## References

- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [FFmpeg Audio Processing](https://ffmpeg.org/)
- [Pydantic Data Validation](https://docs.pydantic.dev/)
