Code organization notes (quick tidy suggestions):

- Core pipeline modules:
  - `context_mapper.py` — 7-stage pipeline logic
  - `main.py` — FastAPI server and job orchestration
  - `glossary.py`, `policy.py`, `runtime_policy.py` — supporting modules

- Helper scripts (move to `scripts/`):
  - `generate_audio.py`, `upload_and_get_kt.py`, `check_kt.py`

- Static assets:
  - `static/` for HTML and any saved screenshots (screenshot capture is currently disabled)

- Tests:
  - `test_pipeline.py` — integration/unit tests

- Recommended next steps:
  1. Move helper scripts into `scripts/` and remove duplicates.
  2. Consider creating a `src/kt_planner` package for core modules to improve imports and packaging.
  3. Add a `make` or `invoke` task to run common workflows (start server, run tests, build artifacts).

Screenshot extraction is commented out in `context_mapper.py` and `main.py` per your request; it can be re-enabled later.
