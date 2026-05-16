# 🚀 Continuum KT Planner - Quick Reference

A short operational summary for the current active setup.

## What this repo does

Continuum KT Planner converts spoken content into structured knowledge transfer output by:

- accepting audio/video input
- transcribing speech to text
- breaking the transcript into sentences
- classifying each sentence into KT sections
- detecting missing coverage and low-confidence items
- producing structured KT output with quality metrics

## Active core files

- `main.py` - FastAPI server and job orchestration
- `context_mapper.py` - core sentence-level semantic pipeline
- `templates.py` - schema/template loading and versioning
- `kt_schema_new.json` - active KT schema definition
- `requirements.txt` - dependency list
- `static/` - UI and screenshot storage
- `docs/` - technical pipeline documentation
- `tests/` - validation tests
- `archive/` - legacy docs moved out of the active root

## Key behavior

- Uses sentence-level semantic classification rather than large chunking.
- Applies neighbor-aware embeddings to improve section placement.
- Marks low-confidence sentences for review.
- Ensures duplicate content is not assigned to multiple sections.
- Uses `context_mapper.py` as the primary processing engine.

## Quick commands

Start the server:

```powershell
cd "c:\Users\dell\Continumm\KT_Planner"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Run tests:

```powershell
python test_pipeline.py
```

Validate output manually:

```powershell
python check_kt.py
```

## Main API endpoints

- `POST /semantic-placement`
  - Input: transcript text or uploaded audio payload
  - Output: section assignments, confidence, coverage

- `POST /expert-correction`
  - Input: corrected sentence/section mapping
  - Output: confirmation and learning update

- `GET /training-stats`
  - Output: correction history and improvement data

- `GET /quality-report/{job_id}`
  - Output: detailed coverage and quality metrics

- `GET /enterprise-status`
  - Output: service health and available capabilities

## Useful metrics

- `duplicate_rate`: should be `0.0`
- `avg_confidence`: target `> 0.85`
- `coherence_score`: target `> 0.80`
- `unclassified_count`: target minimal

## Important notes

- The active schema is `kt_schema_new.json`.
- `ai.py` is legacy helper code; `context_mapper.py` is the primary mapper.
- `archive/` contains older docs and moved artifacts.
- `app/` folders are placeholders for future module expansion.

## Typical workflow

1. Start the server
2. Upload transcript or audio
3. Wait for processing to finish
4. Review structured KT output
5. Check quality metrics
6. Apply expert corrections if needed

## Current folder summary

```
KT_Planner/
├── archive/          # legacy docs and moved artifacts
├── app/              # placeholder modules
├── docs/             # technical docs
├── static/           # UI and screenshots
├── tests/            # validation tests
├── context_mapper.py # active semantic pipeline
├── main.py           # server entry point
├── templates.py      # template manager
├── kt_schema_new.json# active schema
└── requirements.txt  # dependencies
```

## Next step

After setup, run `python test_pipeline.py` to verify the pipeline and confirm the repo is working correctly.
