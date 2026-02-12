# KT Planner

Minimal KT Planner web app — upload audio/video, transcribe using Whisper, map to KT template using Sentence-Transformers.

## Performance Optimizations

- **Tiny Whisper model**: ~4x faster than "small" while maintaining good accuracy
- **MP3 conversion**: Skips WAV (faster encoding, smaller files)
- **Batch embeddings**: Encodes all chunks at once instead of individually (~10x speedup)
- **Vectorized similarity**: Uses numpy dot products instead of sequential comparisons
- **Async background processing**: Returns job ID immediately; frontend polls for results (no timeout)

Typical speed: ~5-10 min for 120 MB video (down from 40+ min).

## Prereqs

- Python 3.10+
- ffmpeg binary on PATH (not the Python wrapper)

## Install

```bash
python -m pip install -r requirements.txt
# On Windows install ffmpeg (example):
winget install -e --id Gyan.FFmpeg
```

## Run

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Open

- **Frontend**: http://127.0.0.1:8000/
- **API docs**: http://127.0.0.1:8000/docs
- **Job status**: GET `/status/{job_id}` to poll processing results

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Upload audio/video; returns `job_id` for polling |
| GET | `/status/{job_id}` | Poll job status; returns transcript, coverage, missing sections |
| GET | `/schema` | Get KT template schema |

## Notes

- Models (Whisper & Sentence-Transformers) download on first run
- Frontend polls every 2 seconds for job completion
- For production, use Celery/RQ for robust background job management
