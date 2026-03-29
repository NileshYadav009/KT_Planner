# Enterprise Features Integration Guide

## Overview

Three new modules are now available to integrate enterprise features into your working KT Planner application:

1. **`enterprise_features.py`** - Core business logic (sentence mapping, confusion detection, code linking)
2. **`enterprise_api_routes.py`** - FastAPI endpoints that expose features as REST API
3. **`enterprise_react_components.tsx`** - React components for frontend UI

## Step 1: Integrate into main.py

### 1.1 Add Import

At the top of `main.py`, add:

```python
from enterprise_api_routes import create_enterprise_router
```

### 1.2 Register Router

After your FastAPI app initialization (around line 16), add:

```python
# Register enterprise API routes
app.include_router(create_enterprise_router())
```

So the sequence should be:

```python
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register enterprise API routes
app.include_router(create_enterprise_router())

# Serve the frontend static files
app.mount("/static", StaticFiles(directory="static"), name="static")
```

### 1.3 Hook Transcription Pipeline

In the `process_upload_task()` function (around line 100-150), after transcription is complete and text is obtained, add:

```python
# NEW: Segment and analyze sentences
from enterprise_features import segment_and_analyze_transcript

# After you get transcript_text from Whisper, add this:
sentences = segment_and_analyze_transcript(transcript_text, job_id)
JOB_QUEUE[job_id]["sentences"] = sentences
```

## Step 2: Test New Endpoints

### 2.1 Start the Server

```bash
cd c:\Users\dell\Continumm\KT_Planner
python -m uvicorn main:app --reload --port 8000
```

### 2.2 Test Endpoints

```bash
# Get all sentences (should be empty initially)
curl http://localhost:8000/api/v1/sentences

# Get confusing sentences
curl http://localhost:8000/api/v1/confusing-sentences

# After uploading audio, sentences will be created automatically
```

### 2.3 Test via Python Script

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Assuming you've uploaded audio and got sentences
sentences_response = requests.get(f"{BASE_URL}/api/v1/sentences")
sentences = sentences_response.json()

if sentences:
    sentence_id = sentences[0]["id"]
    
    # Map sentence to section
    response = requests.post(
        f"{BASE_URL}/api/v1/sentences/map",
        json={
            "sentence_id": sentence_id,
            "section_id": "objective",
            "project_id": "default"
        }
    )
    print(response.json())
```

## Step 3: Available API Endpoints

All endpoints are prefixed with `/api/v1`:

### Sentences

```
GET     /sentences                           - List all sentences
GET     /sentences?status=mapped              - Filter by status
GET     /sentences/{sentence_id}              - Get specific sentence
POST    /sentences/{sentence_id}/edit         - Edit sentence text
POST    /sentences/{sentence_id}/importance   - Toggle importance flag
```

### Mapping (Drag & Drop)

```
POST    /sentences/map                     - Map single sentence to section
POST    /sentences/bulk-map                - Map multiple sentences at once
```

### Code References

```
POST    /sentences/{sentence_id}/code-references      - Add code reference
GET     /sentences/{sentence_id}/code-references      - Get code references
```

### Reviews & Confusion

```
POST    /sentences/{sentence_id}/reviews      - Mark confusing/add review
GET     /sentences/{sentence_id}/reviews      - Get reviews for sentence
GET     /confusing-sentences                  - List all confusing sentences
```

### Analysis

```
POST    /coverage-analysis                 - Analyze coverage across sections
```

### WebSocket (Real-time)

```
WS      /ws/{project_id}                    - Connect to real-time updates
```

## Step 4: Frontend Integration

### 4.1 Create React Setup (if needed)

If you don't have React set up yet, create a new React app in the `static` folder:

```bash
cd static
npx create-react-app .
npm install zustand @dnd-kit/core @dnd-kit/utilities
```

### 4.2 Use Components

In your React component:

```typescript
import { DragDropContainer, useWebSocketUpdates } from './enterprise_react_components'
import { useState, useEffect } from 'react'

export function KTPlannerPage() {
  const [sentences, setSentences] = useState([])
  const [sections, setSections] = useState([])
  const projectId = 'default'

  // Fetch initial data
  useEffect(() => {
    Promise.all([
      fetch('/api/v1/sentences').then(r => r.json()),
      fetch('/static/kt_schema_new.json').then(r => r.json())
    ]).then(([sent, schema]) => {
      setSentences(sent)
      setSections(schema.sections)
    })
  }, [])

  // Set up real-time updates
  useWebSocketUpdates(projectId, (update) => {
    if (update.event === 'sentence_mapped') {
      // Re-fetch or update local state
      setSentences(prev => prev.map(s =>
        s.id === update.sentence_id 
          ? { ...s, assigned_section: update.section_id }
          : s
      ))
    }
  })

  return (
    <DragDropContainer
      sentences={sentences}
      sections={sections}
      onSentenceMapped={async (sentenceId, sectionId) => {
        await fetch('/api/v1/sentences/map', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sentence_id: sentenceId,
            section_id: sectionId,
            project_id: projectId
          })
        })
      }}
      projectId={projectId}
    />
  )
}
```

## Step 5: Test the Full Flow

1. **Open** http://localhost:8000/ in your browser
2. **Upload** an audio file 
3. **Wait** for transcription (should see segmentation progress)
4. **Check** sentences appear in the UI at `/api/v1/sentences`
5. **Drag & drop** sentences to sections in the UI
6. **Watch** real-time updates via WebSocket

## Step 6: Troubleshooting

### "Module not found: enterprise_features"

**Solution:** Make sure `enterprise_features.py` is in the same directory as `main.py`:

```bash
ls -la c:\Users\dell\Continumm\KT_Planner\enterprise_features.py
```

### "Sentences are not being created"

**Solution:** Check that transcription completes and `segment_and_analyze_transcript()` is being called in `process_upload_task()`:

```python
# Add debug logging
print(f"Creating {len(sentences)} sentences")
```

### "WebSocket not connecting"

**Solution:** Ensure your frontend is connecting to the correct URL:

```typescript
// Should be lowercase 'ws', and match your server URL
const wsUrl = `ws://localhost:8000/api/v1/ws/default`
console.log('Connecting to:', wsUrl)
```

### "Got error: 'Sentence' model not found"

**Solution:** This is expected in-memory for now. The database models are optional - all data is stored in Python dictionaries. To persist to database, create SQLAlchemy models (optional upgrade step).

## Step 7: Optional Database Integration

To add persistent database storage:

1. Install dependencies:
```bash
pip install sqlalchemy psycopg2-binary alembic
```

2. Create `models.py` with SQLAlchemy ORM definitions matching the schema in the docs

3. Update `enterprise_features.py` to use database instead of in-memory stores

Currently, the in-memory implementation is perfect for development and testing.

## Feature Map

- ✅ **Sentence-level granularity** - Transcript split into sentences
- ✅ **Drag & drop mapping** - Drag sentences to sections via HTTP + WebSocket
- ✅ **Confusion detection** - AI flags confusing/unclear sentences
- ✅ **Code linking** - Attach code snippets to sentences
- ✅ **Real-time updates** - WebSocket broadcasts changes to all clients
- ✅ **Bulk operations** - Map multiple sentences at once
- ✅ **Coverage analysis** - See which sections are missing content
- ⏳ **Pattern learning** - Learn from corrections (ready, not wired yet)
- ⏳ **Versioning** - Track mapping history (schema ready, not implemented)
- ⏳ **Database** - Persist all data (optional enhancement)

## Common Tasks

### Upload and map a transcript

```python
import requests
import time

BASE = "http://localhost:8000"

# 1. Upload audio file
with open("interview.wav", "rb") as f:
    files = {"file": f}
    resp = requests.post(f"{BASE}/api/v1/upload-and-transcribe", files=files)
    job_id = resp.json()["job_id"]

# 2. Wait for processing
while True:
    status = requests.get(f"{BASE}/api/v1/status/{job_id}").json()
    if status["status"] == "completed":
        break
    time.sleep(1)

# 3. Get sentences
sentences = requests.get(f"{BASE}/api/v1/sentences").json()

# 4. Map sentences
for sent in sentences[:3]:  # Map first 3
    requests.post(f"{BASE}/api/v1/sentences/map", json={
        "sentence_id": sent["id"],
        "section_id": "objective",
        "project_id": "default"
    })

# 5. Check coverage
coverage = requests.post(f"{BASE}/api/v1/coverage-analysis", json=[
    {"id": f"section_{i}", "title": f"Section {i}"}
    for i in range(5)
]).json()

print(f"Coverage: {coverage['coverage_percentage']}%")
print(f"Confusing: {coverage['confusing_count']}")
```

### React component with live updates

```typescript
import React, { useState, useEffect } from 'react'
import { DragDropContainer, useWebSocketUpdates } from './enterprise_react_components'

export default function App() {
  const [sentences, setSentences] = useState([])
  const [sections, setSections] = useState([])

  useEffect(() => {
    // Fetch on mount
    fetch('/api/v1/sentences')
      .then(r => r.json())
      .then(setSentences)

    // Query schema from static files
    fetch('/kt_schema_new.json')
      .then(r => r.json())
      .then(d => setSections(d.sections))
  }, [])

  // Listen for live updates
  useWebSocketUpdates('default', (msg) => {
    if (msg.event === 'sentence_mapped') {
      setSentences(prev => prev.map(s =>
        s.id === msg.sentence_id
          ? { ...s, assigned_section: msg.section_id }
          : s
      ))
    }
  })

  return (
    <DragDropContainer
      sentences={sentences}
      sections={sections}
      onSentenceMapped={(sentenceId, sectionId) =>
        fetch('/api/v1/sentences/map', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sentence_id: sentenceId, section_id: sectionId, project_id: 'default' })
        })
      }
      projectId="default"
    />
  )
}
```

---

**You're all set!** Start with Step 1 (integrate into main.py) and work through the flow. The features will be immediately available as you complete each step.

For questions or issues, check the troubleshooting section or review the inline documentation in each module.
