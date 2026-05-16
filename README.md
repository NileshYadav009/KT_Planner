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

- The active schema configuration is `kt_schema_new.json`.
- `archive/` contains older documentation and artifacts moved out of the core setup.
- `app/` currently contains placeholder folders for future API, model, and service code.
- Use `QUICK_REFERENCE.md` for a short operational overview and `DEVELOPERS_GUIDE.md` for deeper implementation details.
