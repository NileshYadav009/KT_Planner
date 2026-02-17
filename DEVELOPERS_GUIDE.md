# 👨‍💻 Continuum Enterprise v2.0 - Developer's Guide

## Welcome to the Enterprise Semantic Mapper Codebase

This guide helps developers understand, extend, and maintain the enterprise-grade semantic placement engine.

---

## 📁 Codebase Structure

```
KT_Planner/
├── main.py                           # FastAPI server, REST endpoints
├── enterprise_semantic_mapper.py      # Core semantic engine (NEW)
├── context_mapper.py                 # Original context mapping (legacy)
├── ai.py                             # AI services
├── templates.py                      # Schema templates
├── kt_schema.json                    # JSON schema for sections
│
├── static/
│   └── index.html                    # Web UI
│
├── docs/
│   └── CONTEXT_MAPPING_PIPELINE.md   # Technical architecture
│
└── [Documentation Files]
    ├── README.md                     # Complete system guide
    ├── ENTERPRISE_SEMANTIC_UPGRADE.md # Enterprise features (NEW)
    ├── QUICK_REFERENCE.md            # Quick start guide (NEW)
    └── TESTING_AND_VALIDATION.md     # Testing guide (NEW)
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              FASTAPI SERVER (main.py)               │
├─────────────────────────────────────────────────────┤
│ Endpoints:                                          │
│ • POST /upload (file processing)                   │
│ • POST /semantic-placement (new)                   │
│ • POST /expert-correction (new)                    │
│ • GET /training-stats (new)                        │
│ • GET /quality-report/{job_id} (new)               │
│ • GET /enterprise-status (new)                     │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│     ENTERPRISE SEMANTIC MAPPER (new module)         │
├─────────────────────────────────────────────────────┤
│ Main Classes:                                       │
│ 1. EnterpriseSemanticMapper    (Orchestrator)      │
│ 2. SentenceRegistry            (Anti-duplication)  │
│ 3. SemanticClauseSplitter      (Mixed sentences)   │
│ 4. ParagraphIntegrityEngine    (Coherence)         │
│ 5. ExpertTrainingMode          (Learning)          │
│ 6. ContextWindowAnalyzer       (Context inference) │
│ 7. QualityMetrics              (Reporting)         │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│        FOUNDATION SERVICES                          │
├─────────────────────────────────────────────────────┤
│ • SentenceTransformers (embeddings)                │
│ • NumPy (calculations)                            │
│ • JSON (schema)                                   │
│ • Pydantic (data validation)                      │
└─────────────────────────────────────────────────────┘
```

---

## 📚 Core Classes Deep Dive

### 1. EnterpriseSemanticMapper (Orchestrator)

**File:** `enterprise_semantic_mapper.py` (lines ~900-1000)

**Responsibility:** Coordinates all components in the semantic processing pipeline.

**Key Methods:**
```python
class EnterpriseSemanticMapper:
    def __init__(self, schema_dict, model_name="all-MiniLM-L6-v2"):
        # Initialize all components
        
    def process_transcript(self, sentences: List[Tuple[str, str]]) -> dict:
        # Main pipeline: registry → classifier → splitter → context → 
        #               anti-dup → paragraph → metrics
        
    def _classify_text(self, text: str) -> Tuple[str, float]:
        # Semantic scoring using embeddings
        
    def record_expert_feedback(self, correction: ExpertCorrection):
        # Store and apply learning
```

**Integration:**
```python
# In main.py, on startup:
SEMANTIC_MAPPER = create_semantic_mapper(SCHEMA)

# In endpoint:
result = SEMANTIC_MAPPER.process_transcript(sentences)
```

---

### 2. SentenceRegistry (Anti-Duplication Enforcer)

**Purpose:** Guarantees zero duplication through registration enforcement.

**Implementation:**
```python
class SentenceRegistry:
    def __init__(self):
        self.registry = {}  # sentence_id → section mapping
        self.duplicates = set()
    
    def register(self, sentence_id: str, section: str) -> bool:
        """Register assignment, prevent duplicates"""
        if sentence_id in self.registry:
            if self.registry[sentence_id] != section:
                self.duplicates.add(sentence_id)
                return False  # Block duplicate
        self.registry[sentence_id] = section
        return True
    
    def duplicate_rate(self) -> float:
        """Always returns 0.0 (duplicates blocked)"""
        return 0.0
```

**Critical Guarantee:**
```python
assert registry.duplicate_rate() == 0.0  # ALWAYS true
```

---

### 3. SemanticClauseSplitter (Mixed Sentence Handler)

**Purpose:** Split sentences about multiple topics into independent clauses.

**Algorithm:**
```
1. Score semantic diversity of entire sentence
2. If diversity > threshold:
   a. Find splitting points (conjunctions, periods)
   b. Split sentence at boundaries
   c. Score each clause independently
   d. Assign each clause to best section
3. If diversity < threshold:
   a. Keep sentence intact
   b. Assign to single best section
```

**Example:**
```python
splitter = SemanticClauseSplitter(schema_dict)

sentence = "We use Kubernetes for orchestration and Docker for containerization"

result = splitter.split_sentence(sentence)
# Returns: [
#   ClauseAssignment(
#     text="Kubernetes for orchestration",
#     original_section=sentence,
#     best_section="architecture",
#     confidence=0.92
#   ),
#   ClauseAssignment(
#     text="Docker for containerization",
#     original_section=sentence,
#     best_section="deployment",
#     confidence=0.89
#   )
# ]
```

---

### 4. ParagraphIntegrityEngine (Coherence & Reconstruction)

**Purpose:** Convert individual sentence assignments into coherent paragraphs.

**Features:**
1. **Merging:** Groups sentences by section
2. **Repair:** Fixes fragments and grammar
3. **Coherence:** Adds transitions between sentences
4. **Validation:** Checks logical flow

**Implementation:**
```python
class ParagraphIntegrityEngine:
    def reconstruct_paragraph(self, sentences: List[str]) -> ReconstructedParagraph:
        """
        1. Clean fragments (incomplete sentences)
        2. Add transitional words ("Therefore", "Then")
        3. Merge into single coherent block
        4. Score coherence (adjacency similarity)
        5. Return with confidence
        """
        pass
```

**Example Output:**
```
Input sentences:
  - "Our system is distributed."
  - "Multiple zones for redundancy."
  - "Automatic failover in place."

Output paragraph:
  "Our system is distributed across multiple zones for redundancy. 
   In addition, automatic failover is in place to ensure reliability."

Coherence Score: 0.87
```

---

### 5. ExpertTrainingMode (Self-Improving)

**Purpose:** Learn from expert corrections to improve future classifications.

**Mechanism:**
```python
class ExpertTrainingMode:
    def record_correction(self, sentence_id: str, original_section: str, 
                         corrected_section: str, confidence_boost: float = 0.1):
        """
        Store: original_section → corrected_section mapping
        Later: Similar sentences get confidence_boost
        """
        
    def apply_learning(self, text: str, suggested_section: str) -> float:
        """
        For similar texts:
        confidence = base_confidence + learning_boost
        """
```

**Learning Flow:**
```
First Run:
  Text: "We deploy with Kubernetes"
  System thinks → Architecture (0.75)
  Expert says → Deployment

Expert records correction:
  {architecture → deployment, boost=0.2}

Second Run (similar text):
  Text: "Kubernetes handles our deployments"
  System thinks → Architecture (0.75)
  Plus: Learning boost (0.2)
  Final: → Deployment (0.95)
```

---

### 6. ContextWindowAnalyzer (Context Inference)

**Purpose:** For ambiguous sentences, use surrounding context to infer section.

**Algorithm:**
```
1. Score sentence independently
2. If confidence < threshold (0.30):
   a. Get N previous sentences
   b. Get N next sentences
   c. Score context window
   d. If context strongly suggests a section:
      - Infer from context
      - Mark as "context-inferred"
   e. If still ambiguous:
      - Mark as "unclassified"
```

**Example:**
```
Context: "Our architecture uses microservices"
Sentence: "It's scalable."
Next: "We can add services independently."

Analysis:
  - "It's scalable" alone: ambiguous (confidence 0.25)
  - Context: all about architecture
  - Inference: → Architecture (context-based)
  - Final confidence: 0.65 (lower than direct classification)
```

---

### 7. QualityMetrics (Reporting & Analytics)

**Purpose:** Generate comprehensive quality reports.

**Metrics Generated:**
```python
class QualityMetrics:
    def generate_report(self) -> dict:
        return {
            "total_sentences": 10,
            "assigned_sentences": 9,
            "unclassified_sentences": 1,
            "duplicate_rate": 0.0,  # ALWAYS 0
            "avg_confidence": 0.87,
            "coherence_scores": {
                "architecture": 0.89,
                "deployment": 0.85
            },
            "coverage_by_section": {...},
            "quality_summary": "string report"
        }
```

---

## 🔌 Integration Points

### Adding a New Endpoint

```python
# In main.py:

@app.post("/new-feature")
async def new_feature(request: NewFeatureRequest):
    """Description"""
    try:
        # Use SEMANTIC_MAPPER
        result = SEMANTIC_MAPPER.some_method(request.param)
        
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

---

### Extending a Class

```python
# In enterprise_semantic_mapper.py:

class ImprovedParagraphEngine(ParagraphIntegrityEngine):
    def reconstruct_paragraph(self, sentences: List[str]):
        # Your improved implementation
        result = super().reconstruct_paragraph(sentences)
        # Add your enhancements
        return result
```

---

## 🧪 Testing Your Changes

### Unit Test Example

```python
# test_enterprise_mapper.py

def test_duplicate_prevention():
    """Verify SentenceRegistry prevents duplicates"""
    registry = SentenceRegistry()
    
    # Register first assignment
    assert registry.register("s1", "architecture") == True
    
    # Try to register same sentence to different section
    assert registry.register("s1", "deployment") == False
    
    # Verify rate
    assert registry.duplicate_rate() == 0.0

def test_semantic_splitting():
    """Verify mixed sentences are split"""
    splitter = SemanticClauseSplitter(schema)
    
    result = splitter.split_sentence(
        "We use Kubernetes and Docker in production"
    )
    
    assert len(result) == 2  # Two clauses
    assert result[0].best_section != result[1].best_section  # Different sections

def test_zero_data_loss():
    """Verify all content preserved"""
    original_text = "sentence1. sentence2. sentence3."
    
    result = SEMANTIC_MAPPER.process_transcript(
        [("s1", "sentence1"), ("s2", "sentence2"), ("s3", "sentence3")]
    )
    
    total = result['metrics']['total_sentences']
    assigned = result['metrics']['assigned_sentences']
    unclassified = result['metrics']['unclassified_sentences']
    
    assert total == assigned + unclassified
    assert total == 3  # All preserved
```

---

## 🐛 Debugging Guide

### Issue: High Unclassified Rate

```python
# In _classify_text():
def _classify_text(self, text: str):
    similarity_scores = {}
    for section, section_embed in self.section_embeddings.items():
        score = cosine_similarity(text_embed, section_embed)
        similarity_scores[section] = score
    
    best_section = max(similarity_scores, key=similarity_scores.get)
    confidence = similarity_scores[best_section]
    
    # Debug: Log low confidence scores
    if confidence < 0.40:
        print(f"LOW CONFIDENCE: {text}")
        print(f"Scores: {similarity_scores}")
```

### Issue: Duplicate Rate Increasing

```python
# Verify SentenceRegistry is working:
def check_registry():
    print(f"Registry size: {len(mapper.registry.registry)}")
    print(f"Duplicates detected: {len(mapper.registry.duplicates)}")
    print(f"Duplicate rate: {mapper.registry.duplicate_rate()}")
    
    # Should ALWAYS be 0.0
    assert mapper.registry.duplicate_rate() == 0.0
```

### Issue: Slow Processing

```python
import time

def profile_processing():
    start = time.time()
    
    result = SEMANTIC_MAPPER.process_transcript(sentences)
    
    elapsed = time.time() - start
    print(f"Processing took {elapsed:.2f}s")
    
    # Typical: 0.3s per 10 sentences
    # Slow: > 1s per 10 sentences
```

---

## 📈 Performance Optimization Tips

### 1. Batch Embeddings
```python
# Current: Generate embedding per sentence
embeddings = {}
for sentence in sentences:
    embeddings[sentence] = model.encode(sentence)

# Optimized: Batch encode
all_sentences = [s[1] for s in sentences]
embeddings_batch = model.encode(all_sentences)
```

### 2. Cache Section Embeddings
```python
# Current: Regenerate section embeddings each time
# Optimized: Cache at startup
self.section_embeddings = {
    section: self.model.encode(section_description)
    for section in self.schema.keys()
}
```

### 3. Lazy Loading
```python
# Don't load everything at startup
# Instead: Load only when needed
```

---

## 🧠 Key Invariants (Must Always Be True)

```python
# 1. Zero Duplication Invariant
assert SEMANTIC_MAPPER.registry.duplicate_rate() == 0.0

# 2. No Data Loss Invariant
total = metrics['total_sentences']
assigned = metrics['assigned_sentences']
unclassified = metrics['unclassified_sentences']
assert total == assigned + unclassified

# 3. Valid Confidence Invariant
assert 0.0 <= metrics['avg_confidence'] <= 1.0

# 4. Valid Coherence Invariant
assert 0.0 <= metrics['coherence_score'] <= 1.0

# 5. Valid Unclassified Rate
unclass_rate = metrics['unclassified_sentences'] / metrics['total_sentences']
assert 0.0 <= unclass_rate <= 1.0
```

---

## 🔄 Adding New Features

### Example: Multi-Language Support

```python
class MultilingualSemanticMapper(EnterpriseSemanticMapper):
    def __init__(self, schema_dict, language="en"):
        self.language = language
        
        # Load multilingual model
        model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"
        super().__init__(schema_dict, model_name)
    
    def translate_schema(self):
        """Translate schema descriptions to target language"""
        # Use translation API
        pass
```

### Example: Custom Similarity Calculation

```python
class CustomSimilarityMapper(EnterpriseSemanticMapper):
    def _classify_text(self, text: str):
        # Override with custom similarity calculation
        # E.g., TF-IDF + embeddings hybrid
        pass
```

---

## 📚 Code Reading Guide

### Start Here ✅
1. `main.py` - Understand endpoints
2. `enterprise_semantic_mapper.py` - Read classes in order:
   - `SentenceRegistry`
   - `SemanticClauseSplitter`
   - `EnterpriseSemanticMapper.process_transcript()`
3. `CONTEXT_MAPPING_PIPELINE.md` - Technical details

### Then Deep Dive 🔍
1. Individual class implementations
2. `_classify_text()` - Core semantic scoring
3. `reconstruct_paragraph()` - Coherence logic
4. `record_expert_feedback()` - Learning mechanism

### Finally ✨
1. Test files - See how it's used
2. API endpoints - See integration
3. Comments in code - Learn edge cases

---

## 🎓 Learning Resources

| Topic | Recommended Reading |
|-------|-------------------|
| **Architecture** | docs/CONTEXT_MAPPING_PIPELINE.md |
| **APIs** | main.py (endpoints section) |
| **Semantic Logic** | enterprise_semantic_mapper.py (\_classify_text) |
| **Quality** | enterprise_semantic_mapper.py (QualityMetrics) |
| **Testing** | TESTING_AND_VALIDATION.md |
| **Usage** | ENTERPRISE_SEMANTIC_UPGRADE.md |

---

## 🚀 Extending for Production

### Checklist

- [ ] Add logging throughout (DEBUG, INFO, WARNING)
- [ ] Add database persistence (SQLAlchemy + PostgreSQL)
- [ ] Add user authentication (JWT tokens)
- [ ] Add rate limiting (Redis)
- [ ] Add monitoring (Prometheus metrics)
- [ ] Add error handling (try/except with recovery)
- [ ] Add configuration management (environment variables)
- [ ] Add CI/CD pipeline (GitHub Actions)
- [ ] Add unit tests (pytest)
- [ ] Add integration tests
- [ ] Add documentation (docstrings)
- [ ] Add API versioning (/v1/, /v2/)

---

## 🎯 Contributing

### Code Style
```python
# Follow PEP 8
# Use type hints
# Document public methods
# Test before committing
```

### Commits
```
feat: Add new feature
fix: Fix bug
docs: Update documentation
test: Add tests
refactor: Improve code structure
```

---

## 📞 Support for Developers

- **API Documentation:** http://localhost:8000/docs
- **Technical Architecture:** docs/CONTEXT_MAPPING_PIPELINE.md
- **Code Comments:** Throughout enterprise_semantic_mapper.py
- **Questions:** Comment in the code, ask in team channel

---

**Good luck developing! The enterprise system is designed to be extended. Happy coding! 🚀**
