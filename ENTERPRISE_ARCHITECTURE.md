# Enterprise KT Planner - System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND LAYER (React)                        │
│  ┌───────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │ Transcript Panel  │  │   KT Sections    │  │ Review Panel │ │
│  │ (Sentence Level)  │  │   (Editable)     │  │ (Confusing)  │ │
│  └───────────────────┘  └──────────────────┘  └──────────────┘ │
│  ┌───────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │  Code Panel       │  │ Inline Actions   │  │ Version Hist │ │
│  │  (Expandable)     │  │   (Per Sentence) │  │              │ │
│  └───────────────────┘  └──────────────────┘  └──────────────┘ │
│                                                                  │
│  State Manager: Zustand + Redux DevTools                        │
│  DnD Library: dnd-kit                                           │
└─────────────────────────────────────────────────────────────────┘
                            ↕ WebSocket/REST
┌─────────────────────────────────────────────────────────────────┐
│                  BACKEND LAYER (FastAPI)                        │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Upload API   │  │ Mapping API  │  │ Version API  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Editing API  │  │ Review API   │  │ Export API   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  AI Engine:                                                      │
│  • Sentence Embeddings (SentenceTransformer)                    │
│  • Semantic Mapping (Cosine Similarity)                         │
│  • Confidence Scoring                                           │
│  • Confusion Detection                                          │
│  • Pattern Learning (User Corrections)                          │
└─────────────────────────────────────────────────────────────────┘
                            ↕ ORM / SQL
┌─────────────────────────────────────────────────────────────────┐
│               DATABASE LAYER (PostgreSQL)                        │
│                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │ KT Projects          │  │ Transcripts          │            │
│  │ (metadata)           │  │ (raw + processed)    │            │
│  └──────────────────────┘  └──────────────────────┘            │
│                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │ Sentences            │  │ Mappings             │            │
│  │ (sentence-level)     │  │ (sentence→section)   │            │
│  └──────────────────────┘  └──────────────────────┘            │
│                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │ Code References      │  │ Versions             │            │
│  │ (code linking)       │  │ (history + undo)     │            │
│  └──────────────────────┘  └──────────────────────┘            │
│                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │ Reviews              │  │ User Corrections     │            │
│  │ (annotations)        │  │ (for pattern learn)  │            │
│  └──────────────────────┘  └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Data Flows

### 1. Upload & Transcription Flow
```
Video/Audio Upload
  ↓
FFmpeg Conversion
  ↓
Whisper Transcription
  ↓
Sentence Segmentation (NLTK)
  ↓
Store Sentences with embeddings
  ↓
Initial Auto-Mapping (Semantic)
  ↓
Confidence Scoring
  ↓
Detect Confusing Sentences
  ↓
Return to Frontend
```

### 2. Drag-Drop Mapping Flow
```
User Drags Sentence → Section
  ↓
Update Sentence.section_id
  ↓
Update Mapping History
  ↓
Learn Pattern from Correction
  ↓
Trigger "Re-map All" option
  ↓
Update UI in Real-time (WebSocket)
```

### 3. Confusion Detection Flow
```
Sentence Confidence < Threshold
  ↓
Semantic Similarity Check
  ↓
Mark as "needs_review"
  ↓
Add to Clarification Section
  ↓
Provide Inline Actions
  ↓
User Can Reassign Manually
```

---

## Color Coding Scheme

| Color   | Status       | Meaning                          |
|---------|--------------|----------------------------------|
| Green   | `covered`    | Mapped with high confidence     |
| Yellow  | `uncertain`  | Low confidence / needs review   |
| Red     | `confusing`  | AI couldn't map / requires help |
| Blue    | `code_ref`   | Has linked code/reference       |
| Gray    | `unmapped`   | Not yet assigned                |

---

## Authentication & Authorization

- JWT tokens for API security
- Role-based access (Admin, Editor, Reviewer)
- Audit logging for all changes
- Version tracking with user attribution

