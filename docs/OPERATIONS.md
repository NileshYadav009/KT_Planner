# Operating Continuum

How to build, run and watch a Continuum deployment. For single sign-on, see
[SSO_SETUP.md](SSO_SETUP.md).

## Build

The image has every dependency pinned (`requirements.lock`) and every model
baked in at a pinned revision, so it runs with no network access.

```bash
docker build -t continuum .
```

After changing `requirements.txt`, regenerate the lock and run the tests
before committing it:

```bash
python scripts/lock_requirements.py
```

To move a model to a new revision, change its commit in
`scripts/fetch_models.py`, then run the golden suite
(`python scripts/golden_report.py`).

Speaker diarisation (`diarization.py`) uses two ONNX files from the
sherpa-onnx releases, pinned by SHA-256 in `diarization.MODELS` and fetched
by the same script: pyannote segmentation-3.0 (MIT, CNRS) and WeSpeaker
ResNet34 trained on VoxCeleb (CC BY 4.0, WeNet community; credit it where
you list third-party components). They run locally; no token or hosted
service is involved and the audio does not leave the machine.

The section classifier also compares each sentence with labelled examples
(`section_examples.json`). It is generated from the synthetic, non-holdout
goldens; after changing those, run `python scripts/build_section_examples.py`
(the tests fail while it is out of date).

CI (`.github/workflows/ci.yml`) runs the tests from the lock, checks that the
lock matches `requirements.txt`, builds the image, and checks that the
container becomes ready with networking switched off.

## Run

**One container** (API with one worker thread inside it):

```bash
docker run -p 8000:8000 -v continuum-data:/app/data -e CONTINUUM_SECRET_KEY=... continuum
```

**API and workers as separate processes** (recommended): the API loads no
models and only accepts uploads and answers requests; each worker runs one KT
at a time. A worker that crashes or runs out of memory has its KT picked up
again by another worker.

```bash
CONTINUUM_SECRET_KEY=... GROQ_API_KEY=... docker compose up --build
docker compose up --scale worker=3        # more KTs at once (about 4 GB each)
```

Without Docker: `uvicorn main:app` with `CONTINUUM_WORKERS=0`, plus one or
more `python worker.py` processes using the same `CONTINUUM_DB_PATH` and
`CONTINUUM_UPLOAD_DIR`.

The queue is SQLite, so the API and the workers must share one host and a
local disk (not NFS).

### Health checks

| Endpoint | Meaning |
|---|---|
| `GET /healthz` | Liveness: the process is serving. Restart the container only if this fails. |
| `GET /readyz` | Readiness: database writable, ffmpeg present and, if this process runs KTs, models loaded. 503 lists what is not ready. |
| `python job_queue.py --worker-alive` | A worker on this host checked in within the last minute (the compose health check for workers). |

### Settings

| Variable | Default | Purpose |
|---|---|---|
| `CONTINUUM_WORKERS` | `1` | Worker threads inside the API process; `0` when workers run separately. |
| `CONTINUUM_WORKER_CONCURRENCY` | `1` | KTs one worker process runs at once (each needs its own memory). |
| `CONTINUUM_TASK_LEASE_SECONDS` | `120` | How long a worker may go silent before its KT is retried. |
| `CONTINUUM_TASK_MAX_ATTEMPTS` | `2` | Tries before a KT whose worker keeps dying fails. |
| `CONTINUUM_UPLOAD_DIR` | system temp | Where uploads wait for a worker; shared with the workers. |
| `CONTINUUM_MAX_UPLOAD_MB` | `2048` | Largest upload accepted. |
| `CONTINUUM_LOG_FORMAT` | unset (`json` in the image) | `json` or `text` log lines with job, tenant and request ids. |
| `CONTINUUM_METRICS_TOKEN` | unset | Bearer token for `GET /metrics`; unset turns the endpoint off. |
| `CONTINUUM_ALERT_WEBHOOK_URL` | unset | Slack/Teams-style webhook for alerts. |
| `CONTINUUM_ALERT_QUEUE_AGE_SECONDS` | `900` | Alert when the oldest queued KT has waited this long. |
| `SENTRY_DSN` | unset | Send errors to Sentry (no request bodies, local variables or personal data). |
| `CONTINUUM_REVIEW_WEBHOOK_URL` | unset | Slack/Teams-style webhook told when a sign-off starts, someone acknowledges and the KT is signed. |
| `LLM_BATCH_CALLS` | `0` | `1` batches classification checks and field gap-fills (see Submitting KTs). |
| `WHISPER_BATCHED` | `1` | `0` transcribes segment by segment instead of in batches (about twice as slow on CPU). |
| `WHISPER_BATCH_SIZE` | `8` | Speech chunks transcribed together when batched. |
| `CONTINUUM_DIARIZATION` | `1` | `0` skips speaker diarisation (sources then show no speaker). |
| `CONTINUUM_DIARIZATION_DIR` | `$HF_HOME/continuum-diarization` | Where the diarisation models are. |
| `CONTINUUM_DIARIZATION_THRESHOLD` | `0.6` | How alike two voices must be to count as one speaker. |

## Submitting KTs

`POST /upload` and `POST /kt-from-transcript` queue the KT and return at once
with `status: "queued"`; `GET /status/{job_id}` shows `queue_position` while it
waits. Send an `Idempotency-Key` header (any string of letters, digits and
`. _ : -`, up to 128 characters): a retry with the same key within 24 hours
returns the first KT instead of creating a second one. The web app does this
for you.

When a KT finishes, the worker renders its PDF and stores it, so a download is
a read. A document changed after that (an edit, a sign-off step) is rendered
again on the next download.

**Follow-up sessions.** `POST /kt/{job_id}/sessions` (a pasted transcript) or
`POST /kt/{job_id}/sessions/upload` (a recording) adds a session to a
finished KT. A worker rebuilds the document from every session as the next
version; sources name their session (`S2 · 03:15`). `GET /kt/{job_id}/sessions`
lists the sessions and the agenda for the next one (the knowledge gaps left).
A signed KT takes no more sessions; a failed rebuild leaves the KT as it was
and shows the reason. Reviewer edits to the previous version are not carried
onto the rebuilt document; that version stays in History.

**Exports and search.** `GET /export/markdown/{job_id}` (any `?version=`) for
Confluence, Notion or a repository; `GET /export/gaps/{job_id}` is a CSV of the
knowledge gaps that Jira imports as one ticket per row. `GET /search?q=`
searches the finished KTs of the caller's workspace.

**Speakers and language.** Recordings are diarised when the models are
present (sources say who said each sentence) and checked for language:
parts that are clearly not English, and pasted text that does not read as
English, put a warning on the document.

**LLM calls.** `LLM_BATCH_CALLS=1` asks the classification checks
(`LLM_VERIFY_BATCH_SIZE`, default 15 per call) and each section's field
gap-fills (`LLM_GAP_FILL_BATCH_SIZE`, default 12) in batches: about 70% fewer
calls for the same document with a deterministic test model. It is off until
real LLM runs confirm the answers match one call each.

## Watching it

**Logs.** With `CONTINUUM_LOG_FORMAT=json` each line is one JSON object with
`job_id`, `tenant_id` and `request_id` where they apply. Every response carries
`X-Request-ID`.

**Metrics.** `GET /metrics` with `Authorization: Bearer $CONTINUUM_METRICS_TOKEN`
serves Prometheus metrics: KT outcomes by reason, duration, time per pipeline
stage and stage failures, LLM calls, tokens and fallbacks, the false-missing
estimate (sections reported missing whose topics came up under another
section), queue depth and age, and live workers. Import
`deploy/grafana-dashboard.json` for a dashboard and load
`deploy/prometheus-alerts.yml` for the alert rules.

Without Prometheus, `python scripts/ops_report.py [--hours N]` prints the same
figures from the database.

**Alerts.** With `CONTINUUM_ALERT_WEBHOOK_URL` set, the workers post an alert
when a KT fails for a system reason (not when the upload simply was not a KT)
and when the queue backs up.

**Load test.** `python scripts/load_test.py --url ... --file clip.wav -n 10`
submits 10 KTs at once and reports API latency until all of them finish.

## Reviewing and signing off a KT

The document a reviewer edits is the server's copy. Each save is a new
version; every version can be read (`GET /documents/{job_id}/versions/{n}`)
and exported (`GET /export/pdf/{job_id}?version=n`). A save names the version
it was made against and is refused (409) if someone saved in between.

Sign-off names a KT giver, a KT receiver and an approver from the workspace.
Every knowledge gap the document lists must be closed or accepted as a risk,
with a note, before anyone can acknowledge. The KT is signed when all three
have acknowledged the same version; that version is then locked. Each step
(start, gap resolution, acknowledgement, signature) is in the audit log.
