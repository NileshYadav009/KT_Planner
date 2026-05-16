# 🚀 Continuum KT Planner - Quick Start

This quick start covers the current active setup for the Continuum KT Planner repository.

## 1. What this repo does

Continuum KT Planner automatically converts spoken content into structured knowledge transfer output by:

- accepting audio/video input
- transcribing speech to text
- splitting text into sentences
- classifying sentences into KT sections
- detecting missing coverage
- packaging structured KT output

The core logic lives in `main.py` and `context_mapper.py`.

## 2. Current active files

- `main.py` - primary FastAPI entry point
- `context_mapper.py` - sentence-level semantic mapping pipeline
- `templates.py` - template/schema manager
- `kt_schema_new.json` - active KT schema definition
- `requirements.txt` - Python dependencies
- `static/` - browser UI and screenshot storage
- `docs/` - technical documentation
- `tests/` - validation tests
- `archive/` - legacy docs moved out of the active setup

## 3. Setup instructions

### Step 1: Create and activate a virtual environment

```powershell
cd "c:\Users\dell\Continumm\KT_Planner"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 3: Start the API server

```powershell
uvicorn main:app --reload --port 8000
```

### Step 4: Open the UI

Open `static/index.html` in a browser or connect through the API endpoints.

## 4. Key commands

- Run the app:
  ```powershell
  uvicorn main:app --reload --port 8000
  ```
- Run tests:
  ```powershell
  python test_pipeline.py
  ```
- Validate output manually:
  ```powershell
  python check_kt.py
  ```

## 5. What to expect

Once the server is running, the application can:

- receive files via HTTP upload
- process transcripts using sentence-level classification
- return structured KT coverage and confidence scores
- identify sections with missing content
- support template-based schema definitions

## 6. Important notes

- The active knowledge schema is `kt_schema_new.json`.
- Legacy docs and reports have been moved to `archive/`.
- `ai.py` is legacy helper code; `context_mapper.py` is the main processing engine.
- `app/` contains placeholder module folders and is not required for the current setup.

## 7. Useful files to read next

- `README.md` - full project overview and file guide
- `QUICK_REFERENCE.md` - short operational summary
- `DEVELOPERS_GUIDE.md` - developer-level architecture and implementation details
- `TESTING_AND_VALIDATION.md` - how to test and validate the pipeline

## 8. Folder structure

```
KT_Planner/
├── archive/                    # legacy docs and moved artifacts
├── app/                        # placeholder folders for future APIs/models/services
├── docs/                       # technical documentation
├── scripts/                    # helper scripts
├── static/                     # UI and screenshot storage
├── templates/                  # template storage (currently empty)
├── tests/                      # validation and regression tests
├── ai.py                       # legacy AI helper file
├── check_kt.py                 # validation utility script
├── context_mapper.py           # active semantic mapping pipeline
├── devops_transcription.py     # transcription helper
├── enterprise_semantic_mapper.py # alternate semantic mapper implementation
├── glossary.py                 # glossary corrections support
├── kt_schema_new.json          # active schema definition
├── main.py                     # FastAPI server entry point
├── requirements.txt            # package requirements
└── README.md                   # project documentation
```

## 9. Quick troubleshooting

- If the server fails to start: verify the virtual environment is active and dependencies are installed.
- If `uvicorn` is missing: install it with `python -m pip install uvicorn`.
- If the UI does not load: open `static/index.html` directly or use the API endpoints.

## 10. Next step

After setup, run `python test_pipeline.py` to verify the processing pipeline and confirm the repository is working correctly.
