# ENTERPRISE KT PLANNER - QUICK REFERENCE

Production-grade Knowledge Transfer system with enterprise features.

---

## 📋 DOCUMENTS CREATED

This upgrade package includes 8 comprehensive documents:

1. **ENTERPRISE_ARCHITECTURE.md** - System design, data flows, color coding
2. **DATABASE_SCHEMA.sql** - PostgreSQL schema with 13 tables, indexes, audit logs
3. **api_schemas.py** - 30+ Pydantic models for API contracts
4. **API_DESIGN.md** - Complete REST API specification (50+ endpoints)
5. **UI_COMPONENT_DESIGN.md** - React component hierarchy, Zustand store design
6. **enterprise_ai_engine.py** - AI/ML engine (1000+ lines of production code)
7. **enterprise_api_endpoints.py** - FastAPI endpoints (750+ lines)
8. **enterprise_react_components.tsx** - React components (500+ lines)
9. **ENTERPRISE_IMPLEMENTATION_ROADMAP.md** - Step-by-step 5-week implementation plan

---

## 🎯 KEY FEATURES IMPLEMENTED

### ✅ Smart Sentence Segmentation
- Break transcripts into meaningful sentence blocks
- Semantic embeddings for each sentence
- Confidence scoring (0.0 - 1.0)

### ✅ Drag-and-Drop Mapping (VERY IMPORTANT)
- Drag sentences from left panel → drop into KT sections
- Visual feedback with color coding
- Real-time updates via WebSocket
- Auto-learn from corrections

### ✅ Confusion/Review Mode (CRITICAL)
- Flag low-confidence sentences
- Visual highlighting (red/yellow)
- "Clarification Required" section
- Suggested alternative mappings

### ✅ AI-Assisted Auto-Mapping
- Semantic understanding (not keyword matching)
- Alternative suggestions with scores
- Pattern learning from user corrections
- Bulk remap with one click

### ✅ Context-Aware Filling
- No duplicate content across sections
- Best-fit placement based on embeddings
- Deduplication engine

### ✅ Inline Actions Panel
- Edit sentence text
- Quick assign to section
- Mark as important
- Add comment/review
- Link code snippets

### ✅ Versioning System
- Track original + edited versions
- Full audit trail with user attribution
- Undo/Redo support
- Version comparison

### ✅ Coverage Improvement Engine
- Auto-generate missing content checklist
- Suggest questions for KT provider
- Interactive checklist UI

### ✅ Code Linking (NEW)
- Attach code snippets to sentences
- Link to any section
- Multiple refs per sentence
- Syntax highlighting by language

### ✅ Structured Output
- Export as JSON, Markdown, YAML
- PDF-ready structure
- SOP/Runbook formats

---

## 🏗️ ARCHITECTURE HIGHLIGHTS

### Backend Stack
- **Framework**: FastAPI (modern, async, auto-docs)
- **Database**: PostgreSQL + pgvector (for embeddings)
- **ORM**: SQLAlchemy 2.0
- **AI**: SentenceTransformer (all-MiniLM-L6-v2)
- **Real-time**: WebSockets
- **Auth**: JWT tokens + RBAC

### Frontend Stack
- **Framework**: React 18 + TypeScript
- **State**: Zustand (lightweight, performant)
- **Drag-Drop**: dnd-kit (modern, flexible)
- **UI**: React components with inline styles
- **Real-time**: WebSocket hooks

### Database
- **Sentence-level** granularity
- **Version control** built-in
- **Audit logging** for compliance
- **Performance indexes** for common queries
- **Full-text search** on content

---

## 📊 DATA MODEL

```
Projects
  ├─ Transcripts
  │   ├─ Sentences (CORE)
  │   │   ├─ Embeddings (semantic search)
  │   │   ├─ Mappings (to KT sections)
  │   │   ├─ Reviews (confusing, comments)
  │   │   ├─ CodeReferences (linked code)
  │   │   └─ History (version tracking)
  │   │
  │   └─ Versions (snapshots)
  │
  ├─ UserCorrections (pattern learning)
  ├─ MissingContentSuggestions (checklist)
  └─ AuditLog (compliance)
```

---

## 🔄 USER WORKFLOWS

### Workflow 1: Upload & Auto-Map
1. User uploads video/audio
2. System transcribes (Whisper)
3. Sentences are segmented
4. Embeddings generated automatically
5. AI maps to KT sections
6. Low-confidence sentences flagged yellow
7. User sees color-coded results

### Workflow 2: Manual Correction & Remap
1. User identifies misclassified sentence
2. Drags to correct section
3. System learns from correction
4. User clicks "Remap All Similar"
5. AI applies learned pattern
6. All matching sentences updated

### Workflow 3: Clarification & Resolution
1. System flags confusing sentence (red)
2. User clicks "?"
3. Opens clarification review panel
4. User selects suggested section or adds comment
5. Marks as "resolved" or "needs KT provider"
6. Added to meeting agenda

### Workflow 4: Code Reference
1. User finds relevant code snippet
2. Selects "Link Code"
3. Pastes code, selects language, adds description
4. Code appears in right panel
5. Expands on hover with syntax highlighting

---

## �ℹ️ API ENDPOINTS SUMMARY

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/sentences/{id}` | Map/assign sentence |
| POST | `/api/v1/sentences/bulk-map` | Map multiple at once |
| POST | `/api/v1/reviews` | Mark confusing/add comment |
| POST | `/api/v1/code-references` | Link code snippet |
| GET | `/api/v1/projects/{id}/coverage` | Get coverage report |
| POST | `/api/v1/projects/{id}/remap-all` | Remap using patterns |
| WS | `/ws/projects/{id}` | Real-time updates |

**Full API docs**: See [API_DESIGN.md](API_DESIGN.md)

---

## 🎨 UI Color Scheme

| Color | Usage | Meaning |
|-------|-------|---------|
| 🟢 Green (#10b981) | Covered | High confidence & mapped |
| 🟡 Amber (#f59e0b) | Uncertain | Medium confidence |
| 🔴 Red (#ef4444) | Confusing | Low confidence / needs review |
| 🔵 Blue (#3b82f6) | Code Linked | Has code reference |
| ⚪ Gray (#e2e8f0) | Unmapped | Not yet assigned |

---

## 💾 DATABASE SCHEMA OVERVIEW

### Core Tables
- `kt_projects` - Project metadata
- `transcripts` - Raw + processed transcripts
- **`sentences`** - Sentence-level data (CORE TABLE)
- `sentence_embeddings` - Semantic vectors
- `mappings` - Sentence → Section assignments
- `code_references` - Linked code snippets
- `reviews` - Annotations (confusing, important)
- `versions` - Version history snapshots
- `user_corrections` - For pattern learning
- `missing_content_suggestions` - Auto-checklist
- `users` - Team members
- `audit_log` - Full compliance trail

**Schema**: See [DATABASE_SCHEMA.sql](DATABASE_SCHEMA.sql)

---

## 🚀 IMPLEMENTATION PHASES

| Phase | Duration | Focus |
|-------|----------|-------|
| **1** | Week 1-2 | Database + Backend APIs |
| **2** | Week 2-3 | AI Engine Improvements |
| **3** | Week 3-4 | React Frontend + DnD |
| **4** | Week 5 | Testing + Deployment |

**Roadmap**: See [ENTERPRISE_IMPLEMENTATION_ROADMAP.md](ENTERPRISE_IMPLEMENTATION_ROADMAP.md)

---

## 📦 PRODUCTION DEPLOYMENT

### Docker
```bash
# Build backend
docker build -f Dockerfile -t kt-planner:latest .

# Run with docker-compose
docker-compose up -d

# Compose includes:
# - FastAPI backend
# - PostgreSQL database
# - React frontend (Nginx)
```

### Environment Variables
```env
DATABASE_URL=postgresql://user:pass@localhost/kt_planner
JWT_SECRET_KEY=your-secret-key-here
EMBEDDING_MODEL=all-MiniLM-L6-v2
CONFIDENCE_THRESHOLD=0.65
ALLOWED_ORIGINS=http://localhost:3000,https://kt.company.com
```

---

## ✅ TESTING CHECKLIST

- [ ] Segment 500 sentences: < 2s
- [ ] Generate embeddings: < 5s
- [ ] Map 100 sentences: < 3s
- [ ] Drag-drop update: < 200ms
- [ ] WebSocket latency: < 100ms
- [ ] Accuracy on blind test: > 80%
- [ ] Confusion detection: high precision
- [ ] Pattern learning: works on 3+ corrections
- [ ] Remap all: 500 sentences < 15s
- [ ] 10 concurrent projects: no slowdown

---

## 🔍 PERFORMANCE TARGETS

```
Sentence Segmentation:    < 100ms / 100 sents
Embedding Generation:     < 2s / 100 sents
Initial AI Mapping:       < 3s / 100 sents
Drag-Drop Update:         < 200ms
Bulk Map (100 sents):     < 2s
Semantic Search:          < 500ms
WebSocket Latency:        < 100ms
Page Load:                < 1s
API Response:             < 100ms (p95)
```

---

## 📚 FILE STRUCTURE

```
KT_Planner/
├── ENTERPRISE_ARCHITECTURE.md
├── DATABASE_SCHEMA.sql
├── api_schemas.py
├── API_DESIGN.md
├── UI_COMPONENT_DESIGN.md
├── enterprise_ai_engine.py
├── enterprise_api_endpoints.py
├── enterprise_react_components.tsx
├── ENTERPRISE_IMPLEMENTATION_ROADMAP.md
│
├── frontend/
│   ├── src/
│   │   ├── store/
│   │   │   ├── projectStore.ts
│   │   │   └── uiStore.ts
│   │   ├── components/
│   │   │   ├── TranscriptPanel/
│   │   │   ├── SectionsPanel/
│   │   │   └── DragDropContainer.tsx
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts
│   │   └── api/
│   │       └── client.ts
│   └── package.json
│
├── backend/
│   ├── models/
│   │   ├── sentence.py
│   │   ├── mapping.py
│   │   └── ...
│   ├── api/
│   │   └── v1/
│   │       └── routes/
│   ├── db.py
│   ├── main.py
│   └── requirements.txt
│
└── docker-compose.yml
```

---

## 🎓 LEARNING RESOURCES

1. **Semantic Embeddings**: https://www.sbert.net/
2. **dnd-kit**: https://docs.dnd-kit.com/
3. **Zustand**: https://github.com/pmndrs/zustand
4. **FastAPI**: https://fastapi.tiangolo.com/
5. **PostgreSQL Vectors**: https://github.com/pgvector/pgvector

---

## ⚡ QUICK START

### For Backend Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start server
uvicorn main:app --reload

# API Docs: http://localhost:8000/docs
```

### For Frontend Development
```bash
cd frontend/

# Install dependencies
npm install

# Start dev server
npm run dev

# App runs at: http://localhost:5173
```

---

## 🐛 COMMON ISSUES & SOLUTIONS

| Issue | Cause | Solution |
|-------|-------|----------|
| Low confidence scores | Wrong section hints | Review schema hints, retrain model |
| Drag-drop not working | dnd-kit not installed | `npm install @dnd-kit/core` |
| WebSocket drops | Network timeout | Implement auto-reconnect |
| Remap all slow | No indexes | Run database optimization |
| High API latency | N+1 queries | Use SQLAlchemy eager loading |

---

## 📞 SUPPORT

- **Questions**: Check documentation files
- **Bugs**: Add to issue tracker with reproduction steps
- **Feature Requests**: Create GitHub discussion
- **Performance**: Profile with provided tools, check performance targets

---

## 📋 NEXT STEPS

1. **Review** all 9 documents
2. **Set up** PostgreSQL + Python environment
3. **Run** DATABASE_SCHEMA.sql migrations
4. **Implement** Phase 1 (Backend Infrastructure)
5. **Test** with sample transcripts
6. **Proceed** to Phases 2-4

---

## 📝 DOCUMENT NAVIGATION

Start with: [ENTERPRISE_ARCHITECTURE.md](ENTERPRISE_ARCHITECTURE.md)
Then read: [DATABASE_SCHEMA.sql](DATABASE_SCHEMA.sql) → [AI_DESIGN.md](AI_DESIGN.md)

**Implementation**: [ENTERPRISE_IMPLEMENTATION_ROADMAP.md](ENTERPRISE_IMPLEMENTATION_ROADMAP.md)

**Code Reference**: 
- Python: [enterprise_ai_engine.py](enterprise_ai_engine.py), [enterprise_api_endpoints.py](enterprise_api_endpoints.py)
- React: [enterprise_react_components.tsx](enterprise_react_components.tsx)
- Schemas: [api_schemas.py](api_schemas.py)

---

**Status**: ✅ Production-Ready Design
**Last Updated**: March 29, 2026
**Version**: 1.0 Enterprise Edition
