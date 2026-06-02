# 📚 Continuum KT Planner

## Overview

Continuum KT Planner is a Python-based knowledge transfer automation system. It ingests audio or video, transcribes the spoken content, maps each sentence into structured knowledge transfer sections, detects missing coverage, and produces a coherent KT output.

## What this repository contains

- `main.py` - primary application entry point and API server
- `context_mapper.py` - sentence-level semantic classification pipeline
- `templates.py` - schema and template manager for section definitions
- `kt_schema_new.json` - active knowledge transfer schema
- `requirements.txt` - Python dependency list
- `static/` - simple UI and captured screenshot assets
- `docs/` - technical documentation for the pipeline
- `tests/` - validation and regression tests
- `archive/` - legacy docs and moved artifacts

## Functionality

Continuum currently supports:

- Audio/video ingestion and transcription
- Sentence segmentation with timestamps
- Semantic section classification using sentence embeddings
- Context-aware neighbor scoring for better placement
- Confidence assessment and gap detection
- Evidence extraction from text and screenshots
- Template-based schema management for required sections
- Structured KT assembly for downstream review or export

## Core architecture

The active pipeline is centered around `main.py` and `context_mapper.py`.

- `main.py` receives upload requests, tracks jobs, and returns processed results.
- `context_mapper.py` performs the 7-stage mapping pipeline:
  1. Audio confidence validation
  2. Sentence segmentation
  3. Semantic classification
  4. Repair of unclear or low-confidence text
  5. Gap detection for missing sections
  6. Evidence extraction
  7. Structured knowledge assembly

Additional helpers such as `templates.py`, `policy.py`, `runtime_policy.py`, and `glossary.py` support schema management, runtime configuration, and glossary-based text corrections.

## Folder structure

```
KT_Planner/
├── archive/                    # Legacy documents and moved artifacts
├── app/                        # Placeholder for API/models/services modules
│   ├── api/
│   ├── models/
│   └── services/
├── docs/                       # Pipeline and technical documentation
│   └── CONTEXT_MAPPING_PIPELINE.md
├── scripts/                    # Small helper scripts for manual workflows
│   ├── check_kt.py
│   ├── generate_audio.py
│   ├── README.md
│   └── upload_and_get_kt.py
├── static/                     # Web UI files and evidence screenshots
│   ├── index.html
│   ├── review.html
│   └── screenshots/
├── templates/                  # Stored template files (currently empty)
├── tests/                      # Test code and validation helpers
│   └── test_pipeline.py
├── ai.py                       # Legacy AI classification helpers
├── check_kt.py                 # CLI/utility validator for KT output
├── context_mapper.py           # Main semantic mapping pipeline
├── devops_transcription.py     # Transcription-specific helper script
├── enterprise_semantic_mapper.py # Alternate semantic mapper implementation
├── generate_audio.py           # Audio generation or processing utility
├── glossary.json               # Glossary term definitions
├── glossary.py                 # Glossary support and corrections
├── DOCUMENTATION_INDEX.md      # Index of active documentation files
├── README.md                   # This file
├── QUICK_REFERENCE.md          # Short project overview and quick start
├── QUICK_START.md              # Setup and run instructions
├── DEVELOPERS_GUIDE.md         # Developer guide and architecture notes
├── TESTING_AND_VALIDATION.md   # Testing and validation instructions
├── CODE_ORGANIZATION.md        # Code layout and responsibilities
├── REQUIREMENTS_AUDIT.md       # Dependency and compliance checklist
├── requirements.txt            # Python dependencies
├── kt_schema_new.json          # Active schema definition
├── policy.json                 # Runtime policy configuration
├── runtime_policy.py           # Policy loader and runtime behavior
├── templates.py                # Template manager for schema versions
├── upload_and_get_kt.py        # Upload helper and KT retrieval utility
└── main.py                     # FastAPI application entry point
```

## File descriptions

### `main.py`
The main application entry point. It handles HTTP API routes, file uploads, job scheduling, and result delivery.

### `context_mapper.py`
The core processing engine. This file contains the sentence-level semantic mapper and the 7-stage pipeline for turning transcript text into structured KT output.

### `templates.py`
Manages schema templates, versioning, and audit logs. It stores template metadata and supports loading templates for different KT workflows.

### `kt_schema_new.json`
The active knowledge transfer schema. It defines section IDs, labels, hints, and required status for the current KT model.

### `policy.json` and `runtime_policy.py`
`policy.json` stores runtime thresholds and policy flags. `runtime_policy.py` loads those settings and makes them available to the pipeline.

### `glossary.json` and `glossary.py`
Defines glossary terms and applies conservative corrections to transcription text to improve semantic quality.

### `ai.py`
Legacy support for older classification logic. It is kept as a reference and backup; active sentence-level mapping is handled by `context_mapper.py`.

### `enterprise_semantic_mapper.py`
An alternate or experimental semantic mapper implementation. It is not currently part of the main processing flow but is useful for comparison or future enhancement.

### `pii_anonymizer.py`
Wraps PII detection and anonymization logic (based on Microsoft Presidio). It provides:
- `PIIAnonymizer`: class that detects and (optionally) redacts PII from strings
- `anonymize_transcript_before_classification()`: convenience function used by the transcription pipeline

Note: In the current workspace the automatic redaction call that replaced detected values with the literal "[REDACTED]" has been commented out to avoid blocking transcripts and test coverage output. See the "Recent Changes" section below for details and how to re-enable.

### `devops_transcription.py`
Transcription-specific cleanup and normalization helpers (fillers removal, repeated phrase collapse, DevOps-specific corrections). The pipeline previously invoked `pii_anonymizer` as the first normalization step; that call is now commented out to preserve original transcript content by default.

### `scripts/`
Contains helper scripts for manual workflows, such as generating audio, uploading test files, and performing KT checks.

### `static/`
Contains the browser-based UI and screenshot storage for evidence capture during processing.

### `docs/`
Contains deeper technical documentation about the system and pipeline.

### `tests/test_pipeline.py`
Contains the automated pipeline validation logic. Run this file to verify the core processing stages.

### `archive/`
Contains legacy or moved documents that are not required for the current setup. This folder preserves historical references without cluttering the active project root.

## Setup and run

1. Create and activate a Python virtual environment.
2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Start the server:

```powershell
uvicorn main:app --reload
```

4. Open the web UI from `static/index.html` or use the available API routes.

## Quick commands

- Run the pipeline tests:

```powershell
python test_pipeline.py
```

- Validate output with the helper script:

```powershell
python check_kt.py
```

## Notes
## Recent Changes

- The Presidio-based anonymization that replaced sensitive tokens with "[REDACTED]" has been temporarily disabled in the codebase to prevent those placeholders from blocking downstream transcript and coverage outputs. Affected files:
  - `pii_anonymizer.py` — the `anonymize_transcript`/`anonymize_transcript_before_classification` call is commented and now returns the original transcript with an empty report.
  - `devops_transcription.py` — the call that invoked `anonymize_transcript_before_classification` at the start of `clean_transcript()` is commented out.

These changes preserve original transcripts while keeping the detection code in place for future toggling.

## Python libraries and why they are used

Key runtime dependencies (see `requirements.txt`):

- **fastapi, uvicorn**: Web API server and ASGI runtime for job submission and result retrieval.
- **pydantic / pydantic-settings**: Structured configuration and input validation.
- **faster-whisper**: Fast local transcription engine (Whisper-compatible) for audio -> text.
- **transformers, sentence-transformers, torch**: Semantic models and embeddings used by `context_mapper.py` for accurate section classification.
- **ffmpeg-python, librosa, soundfile, pydub**: Audio processing and normalization utilities used by the transcription helpers.
- **numpy, scipy, pandas**: Numeric and data utilities for scoring, signal processing, and structured logs.
- **scikit-learn, nltk, textdistance**: Scoring, text cleanup, and fuzzy matching to improve mapping accuracy.
- **presidio-analyzer, presidio-anonymizer**: PII detection and anonymization (these are present in `requirements.txt`; anonymization calls are currently disabled but detection wrappers remain in `pii_anonymizer.py`).
- **python-dotenv, python-json-logger**: Environment and structured logging helpers for production deployments.
- **security libs (python-jose, passlib, cryptography)**: JWT and credential handling for API security and auth flows.

If you decide to re-enable Presidio anonymization, ensure the Presidio packages and their NLP model dependencies (e.g. spacy models) are installed and available in the runtime environment.

## Application flow (high level)

```mermaid
flowchart TD
  A[Upload audio/video] --> B[Transcription]
  B --> C[clean_transcript() - normalization]
  C --> D[context_mapper.py - sentence segmentation & classification]
  D --> E[Evidence extraction & gap detection]
  E --> F[Template population (`templates.py`) & assemble KT]
  F --> G[KT artifact storage & API response]
  subgraph optional
    C --> H[PII detection (`pii_anonymizer.py`) - detection only]
  end
```

## How to re-enable or configure PII anonymization

1. Install the Presidio dependencies (already listed in `requirements.txt`) and the required NLP models (e.g., spacy model if needed).
2. In `pii_anonymizer.py` restore the anonymization call in `anonymize_transcript()` (the commented block), or set `use_anonymizer=False` to run detection-only flows.
3. In `devops_transcription.py` re-enable the `anonymize_transcript_before_classification` invocation in `clean_transcript()` if you want PII replaced prior to mapping.

Notes:
- Replacing values with literal placeholders (e.g., "[REDACTED]") affects downstream entity extraction and coverage metrics; consider running detection-only reports (no replacement) if you need counts without modifying the transcript text.

## Tests & verification

- Unit & integration test targeting Presidio behavior: `test_presidio_integration.py` (useful when toggling anonymization).
- Pipeline tests: `tests/test_pipeline.py` and `test_pipeline.py` in root.

Run tests locally inside the activated environment:

```powershell
python -m pip install -r requirements.txt
python test_presidio_integration.py
python tests/test_pipeline.py
```

## Files changed in this update

- `pii_anonymizer.py` — anonymization return behavior adjusted to avoid replacing text by default.
- `devops_transcription.py` — anonymization invocation commented out from `clean_transcript()`.

## Extra notes & next steps

- If you need configurable toggles, I can add an environment-driven flag (via `runtime_policy.py` or `.env`) to enable detection-only vs. replacement modes without editing code.
- I can also add a short migration script to produce a redaction audit report linking original -> redacted values for audited stores if required for compliance.

---

If you'd like, I can:

- Add an environment-configurable toggle to switch anonymization modes.
- Re-run tests and report any failures after these README changes.

Tell me which of those you'd like next.


- The active schema configuration is `kt_schema_new.json`.
- `archive/` contains older documentation and artifacts moved out of the core setup.
- `app/` currently contains placeholder folders for future API, model, and service code.
- Use `QUICK_REFERENCE.md` for a short operational overview and `DEVELOPERS_GUIDE.md` for deeper implementation details.
