# ENTERPRISE KT PLANNER - IMPLEMENTATION ROADMAP

Complete upgrade from MVP to enterprise-grade system with step-by-step implementation guide.

---

## PHASE OVERVIEW

```
Phase 1: Backend Infrastructure (Weeks 1-2)
├── Database schema setup
├── ORM models (SQLAlchemy)
├── API endpoints core implementation
└── WebSocket integration

Phase 2: AI Engine Improvements (Weeks 2-3)
├── Sentence-level embeddings
├── Semantic mapping refinement
├── Confusion detection engine
├── Pattern learning system
└── Missing content suggestions

Phase 3: Frontend Development (Weeks 3-4)
├── Zustand state management
├── React components
├── Drag-drop with dnd-kit
├── Real-time WebSocket updates
└── UI/UX polish

Phase 4: Testing & Deployment (Week 5)
├── Integration testing
├── Load testing (100+ sentences)
├── User acceptance testing
├── Production deployment
└── Rollback procedures
```

---

## IMPLEMENTATION CHECKLIST

### Phase 1: Backend Infrastructure

- [ ] **Database Setup**
  - [ ] Install PostgreSQL 14+
  - [ ] Enable pgvector extension for embeddings
  - [ ] Run migration: [DATABASE_SCHEMA.sql](DATABASE_SCHEMA.sql)
  - [ ] Create indexes for performance
  - [ ] Set up automated backups

- [ ] **Python Environment & Dependencies**
  ```bash
  # Install all enterprise dependencies
  pip install -r requirements.txt
  
  # Additional for enterprise features
  pip install sqlalchemy psycopg2-binary alembic
  ```

- [ ] **SQLAlchemy ORM Models Setup**
  - [ ] Create `models/sentence.py` (Sentence, SentenceEmbedding)
  - [ ] Create `models/mapping.py` (Mapping, UserCorrection)
  - [ ] Create `models/review.py` (Review, CodeReference)
  - [ ] Create `models/version.py` (Version, AuditLog)
  - [ ] Set up relationships and cascading deletes

- [ ] **Database Connection & Configuration**
  - [ ] Create `.env` file with DATABASE_URL
  - [ ] Set up connection pooling
  - [ ] Create database utilities in `db.py`

- [ ] **API v1 Structure**
  - [ ] Create `app/api/v1/routes/` directory structure
  - [ ] Implement [API_DESIGN.md](API_DESIGN.md) endpoints
  - [ ] Add request/response validation with Pydantic
  - [ ] Implement error handling middleware

- [ ] **Authentication & Authorization**
  - [ ] JWT token generation and validation
  - [ ] User roles (Admin, Editor, Reviewer, Viewer)
  - [ ] RBAC middleware for endpoints
  - [ ] Add audit logging

- [ ] **WebSocket Implementation**
  - [ ] Set up WebSocket endpoint: `/ws/projects/{project_id}`
  - [ ] Implement connection manager
  - [ ] Event broadcasting system
  - [ ] Message handling and routing

### Phase 2: AI Engine Improvements

- [ ] **Sentence Segmentation Service**
  - [ ] Implement `SentenceSegmentationService.segment_transcript()`
  - [ ] Test with various audio transcriptions
  - [ ] Benchmark performance: target < 100ms for 100 sentences

- [ ] **Embedding Generation**
  - [ ] Load SentenceTransformer model ('all-MiniLM-L6-v2')
  - [ ] Implement batch embedding generation
  - [ ] Store embeddings in PostgreSQL pgvector
  - [ ] Cache embeddings for repeated access

- [ ] **Semantic Mapping Engine**
  - [ ] Build `SemanticMappingEngine.build_section_embeddings()`
  - [ ] Implement classification with confidence scores
  - [ ] Generate alternative suggestions
  - [ ] Test accuracy on labeled data: target > 80%

- [ ] **Confusion Detection**
  - [ ] Implement multi-heuristic detection
  - [ ] Flag low confidence sentences
  - [ ] Detect ambiguous alternatives
  - [ ] Flag unusual patterns

- [ ] **Pattern Learning**
  - [ ] Track user corrections in database
  - [ ] Extract correction patterns (min 3 occurrences)
  - [ ] Apply patterns in remap-all operations
  - [ ] Measure improvement metrics

- [ ] **Missing Content Suggestions**
  - [ ] Create `MissingContentSuggestionEngine`
  - [ ] Generate templated questions
  - [ ] Create interactive checklist
  - [ ] Allow user to mark as resolved

### Phase 3: Frontend Development

- [ ] **Development Setup**
  ```bash
  # Create React app with Vite (faster than CRA)
  npm create vite@latest kt-planner-ui -- --template react-ts
  
  # Install dependencies
  npm install zustand @dnd-kit/core @dnd-kit/utilities
  npm install axios react-hot-toast react-icons
  npm install recharts  # For charts
  ```

- [ ] **State Management (Zustand)**
  - [ ] Create `store/projectStore.ts` from [component design](UI_COMPONENT_DESIGN.md)
  - [ ] Create `store/uiStore.ts` for UI state
  - [ ] Add devtools for debugging
  - [ ] Test state updates with React DevTools

- [ ] **Core Components**
  - [ ] `SentenceCard` - draggable, shows confidence, flags
  - [ ] `SectionColumn` - drop target for sentences
  - [ ] `DragDropContainer` - integrates dnd-kit
  - [ ] `SentenceDetailDrawer` - edit panel
  - [ ] `CodeReferencePanel` - code linking UI

- [ ] **Panels & Views**
  - [ ] `TranscriptPanel` - left sidebar with sentences
  - [ ] `SectionsPanel` - right sidebar with KT sections
  - [ ] `ReviewPanel` - confusing sentences
  - [ ] `CoverageReport` - section coverage analysis
  - [ ] `VersionHistory` - timeline of changes

- [ ] **Real-time Updates**
  - [ ] Implement `useWebSocket` hook ([component code](enterprise_react_components.tsx))
  - [ ] Auto-update UI on server events
  - [ ] Handle connection loss gracefully
  - [ ] Reconnection logic with exponential backoff

- [ ] **Styling & UX**
  - [ ] Apply color scheme from [design doc](UI_COMPONENT_DESIGN.md)
  - [ ] Responsive layout (split panels)
  - [ ] Keyboard shortcuts (Ctrl+E, Ctrl+A, etc.)
  - [ ] Accessibility (ARIA labels, keyboard nav)

- [ ] **API Integration**
  - [ ] Create `api/client.ts` with axios instance
  - [ ] Implement API hooks for all endpoints
  - [ ] Error handling and user feedback
  - [ ] Loading states and spinners

### Phase 4: Testing & Deployment

- [ ] **Backend Testing**
  - [ ] Unit tests for AI engine components
  - [ ] Integration tests for API endpoints
  - [ ] Test with 500+ sentences for performance
  - [ ] Load test: 10 concurrent projects

- [ ] **Frontend Testing**
  - [ ] Component tests with React Testing Library
  - [ ] Test drag-drop functionality
  - [ ] Test WebSocket real-time updates
  - [ ] E2E tests with Cypress

- [ ] **Performance Optimization**
  - [ ] Backend: query optimization with indexes
  - [ ] Backend: embeddings caching
  - [ ] Frontend: code splitting and lazy loading
  - [ ] Frontend: virtualized lists for large sentence counts

- [ ] **Deployment Preparation**
  - [ ] Docker containerization
  ```dockerfile
  # Backend Dockerfile
  FROM python:3.13-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
  
  - [ ] Docker Compose for full stack (backend + DB + frontend)
  - [ ] Environment configuration
  - [ ] Database migration strategy
  - [ ] Backup/restore procedures

- [ ] **Production Checklist**
  - [ ] Enable HTTPS/TLS
  - [ ] Set up monitoring (logs, metrics, alerts)
  - [ ] Implement rate limiting
  - [ ] Set up automated backups
  - [ ] Create rollback procedure
  - [ ] Document API for team

---

## KEY FILES & LOCATIONS

| File | Purpose |
|------|---------|
| [DATABASE_SCHEMA.sql](DATABASE_SCHEMA.sql) | PostgreSQL schema with all tables |
| [api_schemas.py](api_schemas.py) | Pydantic models for API validation |
| [API_DESIGN.md](API_DESIGN.md) | Complete API specification |
| [enterprise_ai_engine.py](enterprise_ai_engine.py) | AI/mapping logic |
| [enterprise_api_endpoints.py](enterprise_api_endpoints.py) | FastAPI endpoints |
| [enterprise_react_components.tsx](enterprise_react_components.tsx) | React components |
| [UI_COMPONENT_DESIGN.md](UI_COMPONENT_DESIGN.md) | UI component specs |
| [ENTERPRISE_ARCHITECTURE.md](ENTERPRISE_ARCHITECTURE.md) | System architecture |

---

## MIGRATION PATH FROM CURRENT SYSTEM

Since you have a working MVP, here's how to upgrade incrementally:

### Step 1: Add Database Layer (Week 1)
- Keep existing file-based storage temporarily
- Add PostgreSQL alongside
- Sync data between systems
- Test new queries on DB

### Step 2: Migrate Sentence Data (Week 1-2)
```python
# Migration script
def migrate_transcript_to_db(raw_text, project_id):
    # Segment into sentences
    sentences = segment_transcript(raw_text)
    
    # Generate embeddings
    sentences = generate_embeddings(sentences)
    
    # Store in DB
    for sent in sentences:
        db_sentence = Sentence(
            id=uuid.uuid4(),
            transcript_id=project_id,
            text=sent['text'],
            ...
        )
        db.add(db_sentence)
    db.commit()
```

### Step 3: Gradual API Rollout (Week 2-3)
- Deploy new API alongside old FastAPI
- Route new features to new endpoints
- Keep old endpoints working
- Gradually migrate users

### Step 4: Frontend Migration (Week 3-4)
- Build new React UI
- Deploy to separate route (e.g., `/new/`)
- Allow users to opt-in
- Gradually migrate all users

### Step 5: Full Cutover (Week 5)
- Retire old system
- Archive file-based data
- Keep backups
- Monitor metrics

---

## PERFORMANCE TARGETS

| Operation | Target | Current |
|-----------|--------|---------|
| Upload & Transcription | < 5 min | N/A |
| Sentence Segmentation | < 100ms | N/A |
| Embedding Generation | < 2s (100 sent.) | N/A |
| Initial Mapping | < 3s (100 sent.) | N/A |
| Drag-Drop Update | < 200ms | N/A |
| Remap All | < 15s (500 sent.) | N/A |
| Page Load | < 1s | TBD |
| Search | < 500ms (full-text) | N/A |

---

## INTEGRATION CHECKLIST

### With Existing Code

- [ ] Import existing `ai.py` functions into new AI engine
- [ ] Use existing `kt_schema_new.json`
- [ ] Reuse existing Whisper transcription logic
- [ ] Keep existing FFmpeg audio processing
- [ ] Migrate existing SECTION_HINTS

### New Dependencies

Add to `requirements.txt`:
```
# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
alembic>=1.13.0

# New AI Features
pgvector>=0.2.0

# API
python-jose>=3.3.0
passlib>=1.7.4

# Optional: For better async
asyncpg>=0.29.0
```

### Testing Your Setup

```python
# test_setup.py - Verify all components work

import asyncio
from enterprise_ai_engine import SentenceSegmentationService
from sentence_transformers import SentenceTransformer

async def test_setup():
    # Test 1: Segmentation
    text = "This is sentence one. This is sentence two."
    sentences = SentenceSegmentationService.segment_transcript(text)
    assert len(sentences) == 2
    print("✓ Segmentation works")
    
    # Test 2: Embeddings
    sentences = SentenceSegmentationService.generate_embeddings(sentences)
    assert 'embedding' in sentences[0]
    print("✓ Embeddings work")
    
    # Test 3: Semantic mapping
    from enterprise_ai_engine import SemanticMappingEngine
    engine = SemanticMappingEngine(schema, section_hints)
    result = engine.classify_sentence(
        sentences[0]['text'],
        np.array(sentences[0]['embedding'])\n    )\n    assert 'predicted_section_id' in result\n    print(\"✓ Semantic mapping works\")\n    \n    print(\"\\n✅ All components working!\")\n\nasyncio.run(test_setup())\n```\n\n---\n\n## TROUBLESHOOTING GUIDE\n\n### Issue: Low confidence scores\n**Solution**: \n- Retrain embedding model on domain data\n- Adjust confidence threshold in UI\n- Review section hints in schema\n\n### Issue: Drag-drop not working\n**Solution**:\n- Check dnd-kit is properly installed\n- Verify droppable zones have proper IDs\n- Check console for JavaScript errors\n\n### Issue: WebSocket disconnects\n**Solution**:\n- Implement auto-reconnect with backoff\n- Set reasonable connection timeout (30s)\n- Check server logs for connection errors\n\n### Issue: Remap all is slow\n**Solution**:\n- Batch embeddings (32 at a time)\n- Use async/await for I/O\n- Add database connection pooling\n- Profile with `time` command\n\n---\n\n## NEXT STEPS AFTER IMPLEMENTATION\n\n1. **Analytics & Monitoring**\n   - Track user actions (edits, mappings, corrections)\n   - Monitor API response times\n   - Calculate mapping accuracy\n   - Identify common confusing sections\n\n2. **Machine Learning Improvements**\n   - Fine-tune embeddings on company domain data\n   - Build classifier for section prediction\n   - Automated pattern extraction\n\n3. **Advanced Features**\n   - Multi-language support\n   - Audio fingerprinting for duplicate detection\n   - Video screenshot extraction\n   - Automatic documentation generation\n   - Integration with Confluence/Notion\n\n4. **Team Collaboration**\n   - Comment threads on sentences\n   - @mentions and notifications\n   - Approval workflows\n   - Role-based permissions\n\n5. **Archive & Knowledge Base**\n   - Full-text search across all KT docs\n   - Topic clustering\n   - Semantic similarity recommendations\n   - Export to PDF/Markdown/Wiki\n\n---\n\n## SUCCESS METRICS\n\nMeasure success of the upgrade with:\n\n1. **User Adoption**\n   - % of team using new mapping UI\n   - Time to complete KT documentation\n   - User satisfaction score\n\n2. **Quality**\n   - Mapping accuracy (manual review sample)\n   - Average confidence scores\n   - % of sentences marked confusing\n   - Coverage completeness\n\n3. **Performance**\n   - API response times\n   - Page load times\n   - WebSocket latency\n   - Database query times\n\n4. **Business Impact**\n   - Reduction in onboarding time\n   - Reduction in knowledge gaps\n   - Fewer production incidents from unclear KT\n   - Team ramp-up time improvement\n\n---\n\n## SUPPORT & DOCUMENTATION\n\nGenerated files to share with team:\n- API documentation (auto from Pydantic)\n- Component storybook (optional)\n- User guide for mapping UI\n- Developer guide for extensions\n- Database schema documentation\n\n**Access Swagger UI**: `http://localhost:8000/docs`\n\n**OpenAPI Spec**: `http://localhost:8000/openapi.json`\n"
   
    return mapping