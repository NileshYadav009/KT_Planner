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
