# KT Planner v2.0 - Production-Ready

**Transform business conversations into Knowledge Transfer documentation in minutes.**

Continuum's KT Planner is a FastAPI-powered application that automatically transcribes audio/video, classifies content into Knowledge Transfer sections, and generates organized documentation.

---

## ⚡ Quick Start (5 minutes)

### 1. Setup
```bash
# Clone and setup
git clone <repo>
cd KT_Planner
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Server
```bash
uvicorn main:app --reload
# Server runs at http://localhost:8000
```

### 3. Upload Content
- Open browser to `http://localhost:8000`
- Click "Upload Audio/Video File"
- Wait for processing (~3 min for 10-min audio)
- Review extracted sentences and download documentation

---

## 🏗️ Architecture

### System Overview
```
User Upload File
    ↓
FFmpeg Normalize Audio
    ↓
Whisper Transcription (tiny model)
    ↓
NLTK Sentence Tokenization
    ↓
Classification into KT Sections
    ├─ Overview
    ├─ Architecture
    ├─ Deployment
    ├─ Troubleshooting
    └─ ... (user-defined in kt_schema_new.json)
    ↓
SentenceTransformers Semantic Mapping
    ↓
Enterprise Features
    ├─ Confusion Detection
    ├─ Code Reference Linking
    └─ Quality Metrics
    ↓
REST API + UI Display
```

### Core Modules

| Module | Purpose | Key Functions |
|--------|---------|---|
| **main.py** | FastAPI server, job orchestration | /upload endpoint, process_upload_task(), WebSocket |
| **ai.py** | Classification, embedding, analysis | classify_section(), generate_embeddings(), analyze_quality() |
| **templates.py** | KT schema management | load schema, validate structure |
| **enterprise_features.py** | Sentence processing, mapping, confusion detection | segment_and_analyze_transcript(), map_sentence_to_section() |
| **enterprise_api_routes.py** | REST endpoints for enterprise features | GET/POST /api/v1/sentences, /api/v1/mapping, etc. |

### Data Flow

```
1. User uploads file
   POST /upload → background task started (job_id: "abc123")

2. Background processing
   ├─ transcribe_audio() → text
   ├─ tokenize_to_sentences() → list of sentences
   ├─ classify_section(sentence) → "Architecture", "Troubleshooting", etc.
   ├─ generate_embeddings(sentence) → numeric vector
   └─ ent.segment_and_analyze_transcript() → populates SENTENCE_STORE

3. User polls status
   GET /status/{job_id} → { "status": "completed", "transcript_length": 5432 }

4. User views enterprise data
   GET /api/v1/sentences → [
     { "id": 1, "text": "...", "section": "Overview", "confidence": 0.92 },
     ...
   ]

5. User can add reviews, mark confusing, link code
   POST /api/v1/sentences/{id}/reviews → { "status": "success" }
```

---

## 🛠️ Installation & Configuration

### Requirements
- Python 3.9+
- FFmpeg (for audio/video processing)
- ~2GB RAM minimum
- ~1GB disk for ML models

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Download ML models (first run only)
python -c "import whisper; whisper.load_model('tiny')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Environment Configuration

Optional environment variables:
```bash
export KT_SCHEMA_FILE="kt_schema_new.json"      # KT section definitions
export WHISPER_MODEL="tiny"                      # tiny, base, small, medium
export CONFIDENCE_THRESHOLD="0.65"               # Min confidence for classification
export MAX_UPLOAD_SIZE_MB="500"                  # Max file upload size
```

---

## 📡 API Reference

### Core Endpoints

#### Upload & Status
```
POST /upload
  Body: { "file": <audio/video file> }
  Response: { "job_id": "abc123", "status": "processing" }

GET /status/{job_id}
  Response: { "status": "completed", "transcript_length": 5432, "sections_found": 12 }
```

#### Enterprise Features
```
GET /api/v1/sentences
  Response: [{ "id": 1, "text": "...", "section": "Overview", "confidence": 0.92 }, ...]

GET /api/v1/sentences/{id}
  Response: { "id": 1, "text": "...", "section": "Overview", "code_references": [...] }

POST /api/v1/sentences/{id}/reviews
  Body: { "review": "Confusing", "notes": "Need clarification" }
  Response: { "id": 1, "review_added": true }

GET /api/v1/mapping
  Response: [{ "sentence": "...", "section": "Architecture", "mapping_score": 0.95 }, ...]

GET /api/v1/code-references
  Response: [{ "code": "def process():", "section": "Deployment", "reference_count": 2 }, ...]
```

#### WebSocket
```
WS /api/v1/ws/{project_id}
  Subscribe to real-time updates during processing
  Messages: { "type": "sentence_added", "data": {...} }
```

For full API docs, start server and visit `http://localhost:8000/docs`

---

## 🎯 Features

### Core Capabilities
- ✅ **Multi-format upload**: MP3, WAV, MP4, MOV, OGG, etc.
- ✅ **Accurate transcription**: OpenAI Whisper (95%+ accuracy)
- ✅ **Intelligent classification**: ML-based section assignment
- ✅ **Semantic mapping**: SentenceTransformers for context understanding
- ✅ **Real-time processing**: WebSocket updates during transcription
- ✅ **Confusion detection**: Flag unclear statements for review
- ✅ **Code linking**: Associate code snippets with documentation
- ✅ **Export options**: Markdown, JSON, custom formats

### Enterprise Features
- 🏢 Sentence-level editing and review
- 🏢 Quality scoring per section
- 🏢 Batch processing for multiple files
- 🏢 Custom KT schemas per team
- 🏢 Collaborative editing (coming soon)

---

## 🔧 Troubleshooting

### Common Issues

**"Out of Memory" Error**
```
Solution: Use smaller Whisper model
export WHISPER_MODEL="tiny"  # instead of "base" or "small"
```

**"Slow Transcription"**
```
Solution 1: Pre-normalize audio
ffmpeg -i input.mp3 -q:a 5 -f mp3 output.mp3

Solution 2: Use GPU acceleration (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**"API Timeout"**
```
Solution: Increase timeout for large files
# main.py: modify REQUEST_TIMEOUT setting
```

**"Classification Accuracy Low"**
```
Solution: Check KT schema matches your domain
# Edit kt_schema_new.json to add domain-specific section types
```

### Debug Mode

```bash
# Enable verbose logging
export DEBUG=1
uvicorn main:app --reload --log-level debug
```

---

## 🧪 Testing

Run the built-in test pipeline:

```bash
python test_pipeline.py
# Tests: transcription, classification, API endpoints, WebSocket
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | You are here. Start with Quick Start ⬆️ |
| [DEVELOPERS_GUIDE.md](DEVELOPERS_GUIDE.md) | Architecture, modules, refactoring details, deployment |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | API quick reference, common tasks |

---

## 🚀 Deployment

### Local Development
```bash
uvicorn main:app --reload
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

### Docker (Optional)
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

For security best practices, see [DEVELOPERS_GUIDE.md - Security](DEVELOPERS_GUIDE.md#-security-considerations)

---

## 📊 Performance

### Typical Processing Times (10-min audio)
- Transcription: ~2-3 minutes
- Classification: ~30 seconds
- Semantic mapping: ~20 seconds
- **Total**: ~3 minutes

### Resource Requirements
- CPU: 2+ cores recommended
- RAM: 2GB minimum, 8GB recommended
- Storage: 1GB for models + ~100MB per project
- Network: 100Mbps minimum

---

## 🤝 Contributing

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 💬 Support

For issues and questions:
- 📋 [GitHub Issues](https://github.com/continuum/kt-planner/issues)
- 💡 [Discussions](https://github.com/continuum/kt-planner/discussions)
- 📧 Email: dev@continuum.ai

---

**Last Updated**: March 29, 2026  
**Version**: 2.0 (Production-Ready)  
**Maintained By**: Continuum Dev Team
