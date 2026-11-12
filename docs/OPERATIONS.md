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

The script downloads only the files the app loads: 3.2 GB of the 11 GB
the six repos hold. It skips the TensorFlow, Flax, Rust and ONNX copies and
the PyTorch `.bin` where safetensors exist. For `mdeberta-v3-base` it takes
only the tokenizer, because GLiNER loads its own weights. It also fetches the
punctuation model (`punctuation.py`, 210 MB ONNX, §9abf), used only for a
long run of transcript text with almost no punctuation.

In the image (`--half`), bge-large and GLiNER are stored in fp16. That takes
the models from 3.2 GB to 2.0 GB, and they still run in fp32. On 13 Oct the
golden suite and the e2e KTs rendered byte-identical documents. `--verify`
fails the build if either model loads in anything but fp32. The packages'
own test folders and torch's C++ headers are deleted in the install layer.

The resulting image is a 2.6 GB download and 4.1 GB unpacked. Before these
changes it was 3.25 GB and 5.5 GB. One container with every model loaded
uses about 3.8 GB of RAM while idle.

Downloads use plain HTTPS (`HF_HUB_DISABLE_XET=1`). In a Docker Desktop
build on 13 Oct the Xet client stalled at about 40 KB/s. A first build took
about 30 minutes on a 4-CPU laptop: about 8 for packages, 15 for models and
the offline check, and the rest to export the image.

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

With no `GEMINI_API_KEY` or `GROQ_API_KEY`, KTs run rules-only. The log says
so once, and each job's LLM usage shows no provider and no calls. Before
13 Oct, a deployment with no key attempted about 40 LLM calls per KT. Each
call failed and logged a warning, and the job was reported as "gemini".

The queue is SQLite, so the API and the workers must share one host and a
local disk (not NFS).

### Health checks

| Endpoint | Meaning |
|---|---|
| `GET /healthz` | Liveness: the process is serving. Restart the container only if this fails. |
| `GET /readyz` | Readiness: database writable and, if this process runs KTs, ffmpeg present and models loaded (an API-only process runs no media tool, so it needs no ffmpeg). 503 lists what is not ready. |
| `python job_queue.py --worker-alive` | A worker on this host checked in within the last minute (the compose health check for workers). |

### Settings

| Variable | Default | Purpose |
|---|---|---|
| `CONTINUUM_WORKERS` | `1` | Worker threads inside the API process; `0` when workers run separately. |
| `CONTINUUM_WORKER_CONCURRENCY` | `1` | KTs one worker process runs at once (each needs its own memory). |
| `CONTINUUM_TORCH_THREADS` | `auto` | CPU threads for the embedding and entity models. `auto` shares the logical CPUs among the KTs a worker runs at once, never fewer than torch's own default (one per physical core); on a 2-core, 4-thread laptop that made embedding 20-30% faster with the same results. A number sets it; `0` keeps torch's default. On a large server, measure before relying on `auto`. |
| `CONTINUUM_TASK_LEASE_SECONDS` | `120` | How long a worker may go silent before its KT is retried. |
| `CONTINUUM_TASK_MAX_ATTEMPTS` | `2` | Tries before a KT whose worker keeps dying fails. |
| `CONTINUUM_RETRY_HOURS` | `24` | A KT that failed that way keeps its input (the pasted text, or the uploaded recording) this long, and its page offers Try again (`POST /jobs/{job_id}/retry`), with no new upload. Then the workers delete the input and the recording's files; deleting the KT deletes them at once. A KT that failed on what was in it, or in a pipeline stage, is not offered a retry: it would fail the same way. `0` keeps nothing. |
| `CONTINUUM_KTS_PER_DAY` | `0` (no limit) | KTs and follow-up sessions one workspace may submit in any 24 hours. One more is refused with 429, a message saying when the next one fits, and `Retry-After`; nothing is queued. A workspace's own limit replaces it (`python -m auth limits-set`; see Submitting KTs, "Workspace limits"). |
| `CONTINUUM_RECORDING_MINUTES_PER_DAY` | `0` (no limit) | Minutes of recording one workspace may submit in any 24 hours, by the length ffprobe reads when a worker checks the recording (the API runs no media tool). Over it, the KT or session fails with a message saying when it fits, its upload is deleted and it is not counted as a KT. A recording is counted once, however many times its worker tries. A file whose container does not state its length counts as a KT only, and is cut at the length limit. |
| `CONTINUUM_LLM_TOKENS_PER_DAY` | `0` (no limit) | LLM tokens (sent and received) one workspace's KTs may use in any 24 hours. Checked before each LLM call; once reached, the KT's remaining LLM steps keep their rules-only results and the document says so. Counted when a KT ends, so two KTs of one workspace running at once can go over it by what they use. |
| `CONTINUUM_UPLOAD_DIR` | system temp | Where uploads wait for a worker; shared with the workers. |
| `CONTINUUM_MAX_UPLOAD_MB` | `2048` | Largest upload accepted. |
| `CONTINUUM_LOG_FORMAT` | unset (`json` in the image) | `json` or `text` log lines with job, tenant and request ids. |
| `CONTINUUM_METRICS_TOKEN` | unset | Bearer token for `GET /metrics`; unset turns the endpoint off. |
| `CONTINUUM_ALERT_WEBHOOK_URL` | unset | Slack/Teams-style webhook for alerts. |
| `CONTINUUM_ALERT_QUEUE_AGE_SECONDS` | `900` | Alert when the oldest queued KT has waited this long. |
| `SENTRY_DSN` | unset | Send errors to Sentry (no request bodies, local variables or personal data). |
| `CONTINUUM_REVIEW_WEBHOOK_URL` | unset | Slack/Teams-style webhook told when a sign-off starts, someone acknowledges, the KT is signed or its review is due. Used only on a deployment with one workspace (or sign-in off): with several, give each its own channel, `python -m auth review-webhook-set <tenant_id> --url-env VAR` (VAR holds that workspace's webhook URL), so no workspace's KT names or people reach another's channel. |
| `LLM_BATCH_CALLS` | `1` | Batches classification checks and field gap-fills (see Submitting KTs); `0` sends one call each. |
| `LLM_MAX_CALLS_PER_KT` | `0` (no limit) | Live LLM requests one KT may send. Later requests are skipped (the KT keeps its rules-only results) and counted as skipped in the KT's LLM usage, so one long KT cannot empty a daily quota. Cache hits do not count. |
| `LLM_INPUT_TOKENS_PER_MINUTE` | `0` (learn) | Input tokens a minute the provider allows. `0` learns it from the first refusal that names it (Groq's free plan for `qwen/qwen3.8-27b`: 7,000 on this account). |
| `LLM_OUTPUT_TOKENS_PER_MINUTE` | `0` (learn) | Output tokens a minute (1,000 on this account). A call never asks for more than this less the headroom. |
| `LLM_RATE_HEADROOM` | `0.9` | Share of each per-minute limit the throttle uses. The nightly recording test uses 0.85. |
| `LLM_CACHE` | `readwrite` | `read`, `off`, or `replay`: answer only from stored responses and never send an unrecorded request (offline runs of recorded LLM output; `scripts/e2e_kts.py --llm replay`). |
| `CONTINUUM_KT_REVIEW_DAYS` | `180` | A finished KT nobody has changed for this many days shows "Review due", and the workers post one reminder per KT to its workspace's review webhook (above), naming the incoming owner. `0` turns it off. |
| `WHISPER_MODEL` | `small` | The Whisper model. A GPU is used automatically when one is present; there `medium` or `large-v3` transcribes more accurately at a usable speed. On CPU, `medium` also needs `WHISPER_ALLOW_MEDIUM_CPU=1`. |
| `WHISPER_COMPUTE_TYPE` | `auto` | Passed to faster-whisper (`int8` on CPU, `float16` on GPU when `auto`). |
| `WHISPER_BATCHED` | `1` | `0` transcribes segment by segment instead of in batches (about twice as slow on CPU). |
| `WHISPER_BATCH_SIZE` | `8` | Speech chunks transcribed together when batched. |
| `CONTINUUM_DIARIZATION` | `1` | `0` skips speaker diarisation (sources then show no speaker). |
| `CONTINUUM_DIARIZATION_DIR` | `$HF_HOME/continuum-diarization` | Where the diarisation models are. |
| `CONTINUUM_DIARIZATION_THRESHOLD` | `0.5` | How alike two voices must be to count as one speaker. Higher values merge more voices into one (at `0.6` two similar voices were often one speaker); lower ones split one speaker in two (`0.3`). |
| `CONTINUUM_DIARIZATION_WINDOW_SHIFT` | `0.5` | How far the 10-second listening window moves each step, as a share of the window. `0.1` (the library's default) reads each second ten times and takes about five times as long, with the same results on the recordings measured. |
| `CONTINUUM_COPYEDIT` | `1` | `0` turns off the copy edit, which rewrites sentences that read as speech into written English (`copyedit.py`). It uses the LLM provider when one is configured and the workspace allows it, else the local 4-bit model; sentences the LLM could not take (a dead key, a rate limit mid-KT, no network, the per-KT call limit) go to the local model. A rewrite is refused, and the speaker's sentence kept, when it drops a tool name, a number, a "never"/"must"/"still" or more than 15% of the content words, or adds words the speaker never said. |
| `CONTINUUM_COPYEDIT_MODEL` | Qwen2.5-1.5B-Instruct, 4-bit | The local model: a path to a `.gguf` file, or `off`. The default (1.1 GB, Apache-2.0) is downloaded by `scripts/fetch_models.py`, never during a KT. It is loaded only for the copy edit, freed after it, and needs about 1.3 GB of RAM while it runs. |
| `CONTINUUM_COPYEDIT_MAX_SENTENCES` | `200` | Sentences of one KT the local model rewrites: the first ones that read as speech, in order; later ones stay as said. A count, not a clock, so the same input gives the same document however busy the machine is (about 2.7 s a sentence on a 2-core laptop; the recordings measured needed 20 to 152). |
| `CONTINUUM_COPYEDIT_MAX_SECONDS` | `1800` | Safety stop for the local copy edit in one KT, so every KT finishes. A KT that reaches it says so in its document's notices ("shown as said"). |
| `CONTINUUM_COPYEDIT_THREADS` | half the CPU cores | CPU threads for the local model, so the machine stays usable while it runs. |
| `PUNCTUATION_MODEL` | on | `0` turns off the punctuation model (`punctuation.py`). It runs only on a run of 120 words or more with fewer than one punctuation mark per 30 words (some speech-to-text exports; Whisper punctuates), and takes only the sentence ends from it: the words stay as said. A missing package or model leaves the text as it is. Loading takes about 10 s, once per worker; a 400-word run takes under a second. Only workers load it: the API queues such a paste as it came, and the worker cleans and checks it (a paste it cannot use fails with the reason, as the API would have refused it). |
| `CONTINUUM_COPYEDIT_MIN_FREE_GB` | `1.5` | Checked once, when a worker starts (or at its first KT): with less memory free than this, that worker never uses the local model and its KTs keep their rules-only sentences. Never checked per KT, which made the same recording read differently on a busy machine (X131). Size the worker's memory limit to include the model's 1.5 GB. |
| `CONTINUUM_MAX_TRANSCRIPT_CHARS` | `150000` | The longest pasted transcript or session (about 25,000 words); a longer one is refused (413) and should be added as sessions. Cleaning runs off the request path, on a thread; 150,000 characters take about 13 s on a laptop. |
| `CONTINUUM_SESSION_TTL` | `43200` (12 h) | A browser session's longest life. |
| `CONTINUUM_SESSION_IDLE` | `1800` (30 min) | A session also ends after this long with no activity; `0` keeps only the lifetime. What the page asks for on its own (polling) does not count as activity; reading and scrolling do (the page reports them every 5 minutes at most). |
| `CONTINUUM_API_DOCS` | on | The API reference (`/docs`, `/openapi.json`) is served to signed-in users only; `0` turns it off, and the app's link with it. |
| `CONTINUUM_HSTS_SECONDS` | `31536000` | `Strict-Transport-Security` on HTTPS responses (directly, or behind a proxy that sends `X-Forwarded-Proto: https`); `0` leaves it out. The other browser security headers (a Content-Security-Policy naming the pages' inline scripts by hash, `X-Frame-Options`, `Referrer-Policy`, `X-Content-Type-Options`, `Permissions-Policy`) are always sent (`browser_security.py`). |
| `CONTINUUM_RETENTION_DAYS` | unset | Delete KTs unchanged for this many days, with their screenshots. Applied when the API starts and every hour by each worker. |

## Backups and restore

Everything a deployment knows is in the job database (`CONTINUUM_DB_PATH`:
KTs, document versions, the queue, people, API keys, sessions, the audit log,
vocabulary candidates), the screenshots folder (`KT_ASSETS_DIR`) and the
approved vocabulary (`glossary.json` beside the database, or
`CONTINUUM_GLOSSARY_PATH`). `scripts/backup.py` copies all three while the app
runs:

```bash
python scripts/backup.py backup --dest /backups --keep 14   # SQLite online backup + screenshots + manifest
python scripts/backup.py verify /backups/continuum-<time>     # checksums and SQLite's integrity check
python scripts/backup.py drill                              # back up, restore into a scratch folder, read every KT back
```

Run `backup` from cron or a scheduled job (in compose:
`docker compose exec api python scripts/backup.py backup --dest /app/data/backups`)
and copy each new folder off the host (an object store or another machine)
in the same job: a backup on the same disk does not survive the disk. Run
`drill` after changing the schedule and at least monthly; it fails when a
KT cannot be read back.

To restore, stop the API and the workers, then
`python scripts/backup.py restore /backups/continuum-<time> --force`. It checks
the backup first and keeps what it replaces beside it
(`.before-restore-<time>`). With a nightly backup the most that can be lost is
a day of KTs (RPO 24 hours; hourly backups make it an hour). A restore takes
seconds for the database plus the models' load time when the app starts (2 to
3 minutes on CPU), so the RTO is about 15 minutes including the copy back from
off-host storage.

## Database migrations

Each module creates its tables as they are now. A change to an existing table
is a numbered migration in `migrations.py`, applied once and in order when the
app starts, and recorded in `schema_migrations`. `python migrations.py` lists
what a database has and what is pending; `--apply` applies it without starting
the app. Migrations go forward only: take a backup before upgrading, and roll
back a release by restoring it. An older release started on a database a
newer one migrated logs an error naming the migration it does not know.

## Vocabulary candidates

Terms the vocabulary learning spots in transcripts are kept per workspace in
the job database, deleted with the workspace. `python scripts/review_vocabulary.py`
lists them with their workspace; `--approve`/`--reject PHRASE` (with
`--workspace TENANT_ID` when the phrase is pending in more than one) decides.
An approved term goes into `glossary.json` and corrects every workspace's
transcripts. The file sits beside the job database, so in Docker it is on the
data volume (`CONTINUUM_GLOSSARY_PATH` puts it elsewhere); a `glossary.json`
in the working directory, where it used to be written, is copied there once
and left in place. Each save writes a new file and moves it into place, so a
crash never leaves half a file. A file from before (`CONTINUUM_VOCAB_CANDIDATES_PATH`) is read
once into the database; a phrase rejected there stays rejected for every
workspace.

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

**Workspaces take turns.** The workers are shared by every workspace of a
deployment. Of the KTs waiting, a worker takes one of the workspace with the
fewest KTs running, then the one whose last KT started longest ago, and within
a workspace the oldest: ten KTs one workspace queued hold another's back by one
KT at most. With one workspace, KTs run in the order they came.
`queue_position` counts the KTs that will start first in that order.

**Recordings are checked by the workers.** The API saves an upload and looks
only at its leading bytes: a document, an image, a playlist or anything else
that is not an audio or video container is refused at once (422). It runs no
media tool on the file. A worker then probes it with ffprobe before anything
decodes it; a recording it refuses (no audio track, longer than
`CONTINUUM_MAX_MEDIA_MINUTES`, unreadable) fails its KT with the reason, and
the upload is deleted. With `CONTINUUM_WORKERS=0` and separate workers (the
compose file), the process that takes requests never parses media.

**Workspace limits.** Each workspace may also be held to daily limits: KTs and
sessions, recording minutes and LLM tokens, each over the last 24 hours. Set
them for the deployment (`CONTINUUM_KTS_PER_DAY`,
`CONTINUUM_RECORDING_MINUTES_PER_DAY`, `CONTINUUM_LLM_TOKENS_PER_DAY`) and,
where one workspace needs more or less, for that workspace:

```bash
python -m auth limits <tenant_id>                                   # limits and use in the last 24 hours
python -m auth limits-set <tenant_id> --kts 30 --tokens 80000       # 0: no limit; "default": the deployment's
```

`GET /usage` gives a signed-in user the same for their own workspace. Every
submission, recording minute and LLM token is recorded with or without a limit
(numbers and ids only, removed with the workspace). On Groq's free plan the
whole account has 200,000 tokens a day; three pilot workspaces could each have
60,000: 20 to 30 minutes of recorded KT a day each with the LLM (a recorded KT
uses 1,900-2,800 tokens a minute of audio), and rules only after that. Request rate limits (sign-in
attempts, API calls a second) belong to the reverse proxy.

**Follow-up sessions.** `POST /kt/{job_id}/sessions` (a pasted transcript) or
`POST /kt/{job_id}/sessions/upload` (a recording) adds a session to a
finished KT. A worker rebuilds the document from every session as the next
version; sources name their session (`S2 · 03:15`). `GET /kt/{job_id}/sessions`
lists the sessions and the agenda for the next one (the knowledge gaps left).
A signed KT takes no more sessions; a failed rebuild leaves the KT as it was
and shows the reason. A session counts against the workspace's daily limits
as a KT does. Reviewer edits to the previous version are not carried
onto the rebuilt document; that version stays in History.

**Exports and search.** `GET /export/markdown/{job_id}` (any `?version=`) for
Confluence, Notion or a repository; `GET /export/gaps/{job_id}` is a CSV of the
knowledge gaps that Jira imports as one ticket per row. `GET /search?q=`
searches the finished KTs of the caller's workspace.

**Publishing to Confluence and Jira.** Connect a workspace once; the token stays
in an environment variable of the API process and only its name is stored:

```bash
python -m auth confluence-set <tenant_id> --url https://acme.atlassian.net/wiki --space OPS \
    --email bot@acme.com --token-env CONFLUENCE_TOKEN [--parent-id 123456]
python -m auth jira-set <tenant_id> --url https://acme.atlassian.net --project OPS \
    --email bot@acme.com --token-env JIRA_TOKEN [--issue-type Task]
python -m auth integrations <tenant_id>          # what is connected
```

With `--email` the token is an Atlassian Cloud API token; without it, a Data
Center personal access token. Givers and reviewers then see **Confluence** and
**Jira issues** buttons: `POST /publish/confluence/{job_id}` makes the document
one page (publishing again gives that page a new version) and
`POST /publish/jira/{job_id}` makes one issue per knowledge gap (a gap already
sent is skipped). Not yet tried against a real Atlassian site; the tests use a
local stand-in for both APIs.

**Reading a KT.** `POST /kt/{job_id}/ask` answers a question with the
document's own statements and their sources (no model writes the answer) and
says when the topic was not covered. `GET /export/runbook/{job_id}` is one
runbook per known failure, built from the incident card. `GET
/documents/{job_id}/diff` lists what changed between two versions, or (with
`?against=<job_id>`) since another KT of the workspace. `GET /dependencies`
lists the components that several KTs of the workspace name.

**Speakers and language.** Recordings are diarised when the models are
present (sources say who said each sentence) and checked for language:
parts that are clearly not English, and pasted text that does not read as
English, put a warning on the document.

**LLM calls.** By default (`LLM_BATCH_CALLS=1`) the classification checks
(`LLM_VERIFY_BATCH_SIZE`, default 15 per call) and each section's field
gap-fills (`LLM_GAP_FILL_BATCH_SIZE`, default 12) go out in batches. On 16
Oct, on Groq `qwen/qwen3.8-27b`, the five end-to-end KTs took 66 requests and
35k tokens batched, against 238 requests and 87k tokens one call each, in
half the time, with every checklist equal or better (REPOSITORY_AUDIT.md
§9aai). `LLM_BATCH_CALLS=0` sends one call each.

Groq accounts also cap output tokens per minute (1,000 on the account
measured). A request asking for more is refused whatever is retried, so the
provider keeps the limit Groq names in the refusal and asks within it from
then on.

Groq's free plan for `qwen/qwen3.8-27b` (console docs, 28 Oct): 30 requests and
8,000 tokens a minute, 1,000 requests and 200,000 tokens a day. This account splits
the minute into 7,000 input and 1,000 output tokens. The provider keeps both under
the limit less `LLM_RATE_HEADROOM`, from the first refusal that names each or from
`LLM_INPUT_TOKENS_PER_MINUTE` and `LLM_OUTPUT_TOKENS_PER_MINUTE`. The OpenAI client
does not retry on its own (`max_retries=0`): it retried each 429 twice, unseen by the
throttle and the usage counters, and on 27 Oct Groq's console showed more 429s than
successes for three recorded KTs while the app logged 18. A recorded KT costs about
1,900-2,800 tokens per minute of audio, so the free plan's daily tokens cover about
70-100 minutes of recordings; a 67-minute KT alone uses most of a day and, paced, runs
past `LLM_KT_DEADLINE_SECONDS`. `qwen/qwen3.8-27b` is a preview model, which Groq says
may be discontinued at short notice.

On 12 Oct the four end-to-end KTs (tests/e2e_kts) ran through the API on Groq
`qwen/qwen3.8-27b` with `LLM_BATCH_CALLS=1`, `LLM_MAX_CALLS_PER_KT=40` and
`LLM_MAX_TOKENS_PER_MINUTE=7000` (the account allows 8,000 tokens a minute): 9 to 15
requests and 4,100-5,600 input tokens (370-480 per request) for a 110-160-word KT (46
requests in all), no failures or retries. A rerun after code changes reused every
unchanged answer from the cache (19 new requests); Danger Zones are no longer sent to
the polish pass. See docs/TESTING.md, "End-to-end KT suite".

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
