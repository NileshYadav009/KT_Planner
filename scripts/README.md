# Scripts

This folder is intended to contain helper scripts and one-off tooling used to interact with the KT Planner application (audio generation, upload-and-retrieve flows, quick checks).

Current guidance:
- Keep production code inside the project root modules (`main.py`, `context_mapper.py`, etc.).
- Place helper scripts here (e.g. `generate_audio.py`, `upload_and_get_kt.py`, `check_kt.py`).
- Do not rely on screenshot extraction; screenshot functionality has been disabled in core code by request.

To move existing scripts into this folder, copy the files and update any references you may have in tooling.
