"""
ENTERPRISE KT PLANNER - SYSTEM ARCHITECTURE DIAGRAM

Visual representation of the complete enterprise system.
"""

# =====================================================
# COMPLETE SYSTEM DIAGRAM
# =====================================================

SYSTEM_ARCHITECTURE = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    ENTERPRISE KT PLANNER SYSTEM                           ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND LAYER (React)                            │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │  Left Panel      │  │  Right Panel     │  │  Detail Panel    │         │
│  │  (Sentences)     │  │  (KT Sections)   │  │  (Editor)        │         │
│  │                  │  │                  │  │                  │         │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │  │ Sentence Edit    │         │
│  │ │ Unmapped (8) │ │  │ │ Deployment   │ │  │ Code Reference   │         │
│  │ │ ☐ Sentence 1 │ │  │ │ [12/12] ███  │ │  │ Reviews          │         │
│  │ │ ☐ Sentence 2 │ │  │ │ ☐ Sent 5     │ │  │ History          │         │
│  │ │ ☐ Sentence 3 │◄─┼──┤► Sent 6      │ │  │                  │         │
│  │ │ 🟨 Confusing │ │  │ │ [Drag Here] │ │  │ [Save] [Cancel]  │         │
│  │ │              │ │  │ │             │ │  │                  │         │
│  │ └──────────────┘ │  │ └─────────────┘ │  │                  │         │
│  │                  │  │                  │  │                  │         │
│  │ [Search] [Filter]│  │ Coverage: 75%   │  │                  │         │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘         │
│                                                                             │
│  State Management: Zustand Store                                          │
│  ├─ selectedSentenceIds (Set<string>)                                     │
│  ├─ currentProject, currentTranscript                                     │
│  ├─ filters (search, status, section, confidence)                         │
│  └─ uiState (panel widths, activeTab, isDragging)                         │
│                                                                             │
│  Key Technologies:                                                         │
│  • React 18 with TypeScript                                              │
│  • Zustand for state management (devtools enabled)                        │
│  • dnd-kit for drag-and-drop                                             │
│  • WebSocket hook for real-time updates                                  │
│  • Axios for API calls                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                               ↕ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BACKEND LAYER (FastAPI)                               │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │ REST API Endpoints (50+)                               │             │
│  │                                                         │             │
│  │ POST   /api/v1/sentences/{id}        → Update Mapping  │             │
│  │ POST   /api/v1/sentences/bulk-map    → Bulk Operations │             │
│  │ POST   /api/v1/reviews               → Add Review      │             │
│  │ POST   /api/v1/code-references       → Link Code       │             │
│  │ GET    /api/v1/projects/{id}/coverage → Report         │             │
│  │ POST   /api/v1/projects/{id}/remap-all → Pattern Learn │             │
│  │ WS     /ws/projects/{id}             → Real-time      │             │
│  └──────────────────────────────────────────────────────────┘             │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │ AI Engine (enterprise_ai_engine.py)                    │             │
│  │ ┌────────────────────────────────────────────────────┐ │             │
│  │ │ 1. SentenceSegmentationService                     │ │             │
│  │ │    └─ segment_transcript() → List[Sentence]      │ │             │
│  │ │    └─ generate_embeddings() → embeddings         │ │             │
│  │ └────────────────────────────────────────────────────┘ │             │
│  │                                                         │             │
│  │ ┌────────────────────────────────────────────────────┐ │             │
│  │ │ 2. SemanticMappingEngine                          │ │             │
│  │ │    └─ build_section_embeddings()                 │ │             │
│  │ │    └─ classify_sentence() → prediction, conf     │ │             │
│  │ │    └─ get_alternatives() → top 3 suggestions    │ │             │
│  │ └────────────────────────────────────────────────────┘ │             │
│  │                                                         │             │
│  │ ┌────────────────────────────────────────────────────┐ │             │
│  │ │ 3. ConfusionDetectionEngine                       │ │             │
│  │ │    └─ detect_confusing_sentences()               │ │             │
│  │ │    └─ multi_heuristic_analysis()                 │ │             │
│  │ │    └─ generate_clarification_suggestion()        │ │             │
│  │ └────────────────────────────────────────────────────┘ │             │
│  │                                                         │             │
│  │ ┌────────────────────────────────────────────────────┐ │             │
│  │ │ 4. PatternLearningEngine                          │ │             │
│  │ │    └─ record_correction() → pattern tracking     │ │             │
│  │ │    └─ get_correction_patterns() → list           │ │             │
│  │ │    └─ suggest_bulk_remaps()                      │ │             │
│  │ └────────────────────────────────────────────────────┘ │             │
│  │                                                         │             │
│  │ ┌────────────────────────────────────────────────────┐ │             │
│  │ │ 5. DeduplicationEngine                            │ │             │
│  │ │    └─ deduplicate_mappings()                      │ │             │
│  │ └────────────────────────────────────────────────────┘ │             │
│  │                                                         │             │
│  │ ┌────────────────────────────────────────────────────┐ │             │
│  │ │ 6. MissingContentSuggestionEngine                 │ │             │
│  │ │    └─ generate_suggestions() → checklist         │ │             │
│  │ │    └─ auto_create_meeting_agenda()               │ │             │
│  │ └────────────────────────────────────────────────────┘ │             │
│  └──────────────────────────────────────────────────────────┘             │
│                                                                             │
│  Key Technologies:                                                         │
│  • FastAPI (modern async Python framework)                               │
│  • SentenceTransformer (semantic embeddings)                             │
│  • SQLAlchemy ORM (database queries)                                     │
│  • Pydantic (request/response validation)                                │
│  • WebSocket (real-time updates)                                         │
│  • JWT authentication                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                               ↕ SQL
┌─────────────────────────────────────────────────────────────────────────────┐
│                 DATABASE LAYER (PostgreSQL 14+)                            │
│                                                                             │
│  ┌──────────────────────────────────────┐  ┌─────────────────────────┐   │
│  │ Core Tables                          │  │ Support Tables          │   │
│  ├──────────────────────────────────────┤  ├─────────────────────────┤   │
│  │ • kt_projects                        │  │ • versions              │   │
│  │ • transcripts                        │  │ • audit_log             │   │
│  │ • sentences (PRIMARY)                │  │ • users                 │   │
│  │ • sentence_embeddings (pgvector)    │  │ • user_corrections      │   │
│  │ • mappings                           │  │ • missing_content_...   │   │
│  │ • reviews                            │  │                         │   │
│  │ • code_references                    │  │                         │   │
│  └──────────────────────────────────────┘  └─────────────────────────┘   │
│                                                                             │
│  Features:                                                                 │
│  • Sentence-level granularity (not document-level)                         │
│  • Vector similarity search (pgvector extension)                           │
│  • Full-text search on content                                            │
│  • Audit trail for compliance                                             │
│  • Version control built-in                                               │
│  • Performance indexes on common queries                                  │
│  • JSONB for flexible nested data                                         │
│                                                                             │
│  Key Indexes:                                                              │
│  • idx_sentences_transcript_status                                        │
│  • idx_sentences_section_status                                           │
│  • idx_sentence_embeddings (for vector search)                            │
│  • Full-text search index on sentence text                                │
│  • Composite indexes for JOIN queries                                     │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════

DATA FLOW: USER UPLOADS VIDEO → COMPLETE KT DOCUMENTATION

   Upload
    │
    ▼
┌────────────────────────────┐
│ Video/Audio File           │
│ (user_presentation.mp4)    │
└────────────────────────────┘
    │
    ├─[FFmpeg]─► Convert to WAV
    │
    ▼
┌────────────────────────────┐
│ Audio (.wav)               │
│ Optimized for Whisper      │
└────────────────────────────┘
    │
    ├─[Whisper]─► Transcribe
    │
    ▼
┌────────────────────────────┐
│ Raw Transcript             │
│ \"The deployment process... │
│  First we SSH into...\"     │
└────────────────────────────┘
    │
    ├─[NLTK]─► Segment into sentences
    │
    ▼
┌────────────────────────────┐
│ Sentences                  │
│ 1. \"The deployment...\"    │
│ 2. \"First we SSH...\"     │
│ 3. \"Then we run...\"      │
│ ... (50 sentences)         │
└────────────────────────────┘
    │
    ├─[SentenceTransformer]─► Generate embeddings
    │
    ▼
┌────────────────────────────┐
│ Embeddings (1024-dim)      │
│ Stored in pgvector         │
└────────────────────────────┘
    │
    ├─[Mapping Engine]─► Classify to KT sections
    │
    ▼
┌────────────────────────────┐
│ Initial Mappings           │
│ Section: confidence_score  │
│ • Deployment: 0.87         │
│ • Operations: 0.62 (?)     │
│ • Troubleshooting: 0.45 (!) │
└────────────────────────────┘
    │
    ├─[Confusion Detection]─► Flag low confidence
    │
    ▼
┌────────────────────────────┐
│ UI Displays Results        │
│ ✓ Green: High confidence   │
│ 🟡 Yellow: Needs review    │
│ 🔴 Red: Confusing          │
└────────────────────────────┘
    │
    ├─ User [Drag → Drop] OR [Click Assign]
    │
    ▼
┌────────────────────────────┐
│ User Correction            │
│ Sentence #3 → Deployment   │
│ (was Troubleshooting)      │
└────────────────────────────┘
    │
    ├─[Pattern Learning]─► Learn from correction
    │
    ▼
┌────────────────────────────┐
│ Pattern Recorded           │
│ Troubleshooting → Deploy   │
│ (Frequency: 1)             │
└────────────────────────────┘
    │
    ├─ User clicks \"Remap All Similar\"
    │
    ▼
┌────────────────────────────┐
│ Bulk Remap Applied         │
│ 7 similar sentences        │
│ remapped to Deployment     │
└────────────────────────────┘
    │
    ├─[Coverage Analysis]─► Check completeness
    │
    ▼
┌────────────────────────────┐
│ Coverage Report            │
│ Deployment: 100% ✓         │
│ Operations: 75% ⚠          │
│ Security: 0% ✗             │
└────────────────────────────┘
    │
    ├─[Auto Suggestions]─► Generate missing checklist
    │
    ▼
┌────────────────────────────┐
│ Clarification Checklist    │
│ ☐ Ask about network setup  │
│ ☐ Ask about firewall rules │
│ ☐ Ask about DNS config     │
└────────────────────────────┘
    │
    ├── User [Link Code Reference]
    │   \"kubectl apply -f deploy.yaml\"
    │
    ├── User [Add Comment]
    │   \"Need more detail on rollback\"
    │
    ├── User [Mark Resolved Items]
    │
    ▼
┌────────────────────────────┐
│ Export Final KT Document   │
│ • JSON (for DB)            │
│ • Markdown (for Wiki)      │
│ • PDF (for printing)       │
└────────────────────────────┘
    │
    ▼
📦 Complete, searchable knowledge base
   - Auto-categorized sections
   - Linked code examples
   - Version history
   - Full audit trail

═══════════════════════════════════════════════════════════════════════════════

REAL-TIME UPDATE FLOW (WebSocket)

Front-end                    WebSocket                   Back-end
   │                            │                           │
   │ User drags sentence        │                           │
   ├─────────────────────────────────────────────────────────►│
   │                            │ POST /api/v1/sentences/{id}│
   │                            │                           │
   │                            │ ◄─ Update sentence {id}   │
   │                            │                           │
   │                            │ Check for patterns ◄─────┤
   │                            │ Update DB                 │
   │                            │                           │
   │ ◄────────────────────────────── WS: sentence_updated   │
   │                            │                           │
   │ Update UI (no refresh!)    │                           │
   │ - Move card to section     │                           │
   │ - Update confidence badge  │                           │
   │ - Show toast notification  │                           │
   │                            │                           │
   │ Performance: < 200ms       │                           │
   │ No page refresh required   │                           │
   │ Multiple users see updates │                           │
   │                            │                           │

Typical WebSocket Events:
SENTENCE_UPDATED
REVIEW_ADDED
MAPPING_CREATED
REMAP_ALL_COMPLETE
PROJECT_STATUS_CHANGED

═══════════════════════════════════════════════════════════════════════════════

KEY METRICS & PERFORMANCE TARGETS

Operation                      Target      Current Status
─────────────────────────────────────────────────────────
Upload & Transcription         < 5 min     TBD (depends on duration)
Sentence Segmentation          < 100ms     Optimized with NLTK
Embedding Generation           < 2s        Batch processing (32 at a time)
Initial Mapping                < 3s        Semantic similarity optimized
Drag-Drop Update               < 200ms     Real-time with WebSocket
Bulk Map (100 sentences)       < 2s        Batch API endpoint
Remap All (500 sentences)      < 15s       Pattern learning optimized
Semantic Search                < 500ms     Full-text + vector search
WebSocket Latency              < 100ms     Real-time updates
Page Load                      < 1s        Code splitting + React 18
API Response (p95)             < 100ms     FastAPI + connection pooling
Database Query (p95)           < 50ms      Indexed queries
Overall Coverage Complete      30-60 min   Depends on transcript length

═══════════════════════════════════════════════════════════════════════════════

AUTHENTICATION & AUTHORIZATION MODEL

JWT Token Contains:
├─ user_id
├─ email
├─ role (admin, editor, reviewer, viewer)
├─ project_ids (projects user can access)
└─ exp (expiration time)

Role Permissions:
┌────────┬──────────┬──────────┬────────────┬────────┐
│ Action │  Admin   │  Editor  │  Reviewer  │ Viewer │
├────────┼──────────┼──────────┼────────────┼────────┤
│ Upload │    ✓     │    ✓     │     ✗      │   ✗    │
│ Assign │    ✓     │    ✓     │     ✓      │   ✗    │
│ Edit   │    ✓     │    ✓     │     ✗      │   ✗    │
│ Delete │    ✓     │    ✗     │     ✗      │   ✗    │
│ Export │    ✓     │    ✓     │     ✓      │   ✓    │
│ Review │    ✓     │    ✓     │     ✓      │   ✓    │
└────────┴──────────┴──────────┴────────────┴────────┘

All actions logged to audit_log table for compliance.

═══════════════════════════════════════════════════════════════════════════════
"""

print(SYSTEM_ARCHITECTURE)
