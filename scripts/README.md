# Scripts

Helper scripts for manual workflows. They are not part of the app, and `pytest` does not collect them (`pytest.ini` limits it to `tests/`).

| Script | What it does |
|---|---|
| `test_kt_from_transcript.py` | Sends a transcript to a running server (`/kt-from-transcript`) and waits for the KT |
| `upload_and_get_kt.py` | Uploads a recording to a running server and downloads the result |
| `check_kt.py` | Quick check of a running server's KT output |
| `generate_audio.py` | Turns a sample transcript into speech for test recordings (needs `pyttsx3`) |
| `review_vocabulary.py` | Approve or reject learned vocabulary candidates |
| `debug_glossary.py` | Try glossary corrections on a sentence |
| `test_pipeline.py`, `test_coverage_extraction.py` | Older walkthroughs of the mapping pipeline that print each stage |
| `demo_coverage_polish.py`, `demo_topic_blocks.py` | Demonstrations of the polish pass and topic blocks |

Run them from the project root, e.g. `python scripts/test_kt_from_transcript.py`. Scripts that import project modules add the project root to `sys.path` themselves.
