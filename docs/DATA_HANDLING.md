# How Continuum handles your data

For a customer's security and privacy review. It describes the application as
it is in this repository. Where the answer depends on how a deployment is set
up (the host, its disks, its backups), it says so.

## What comes in

- **Recordings** of KT sessions (audio or video). An upload is streamed to
  disk with a size cap and checked before anything decodes it. Only audio and
  video containers are accepted, and playlist formats that could make the media
  tools read other files or fetch URLs are refused. The server that receives
  uploads runs no media tool on them; the workers, which take no requests from
  outside, check and decode them.
- **Pasted transcripts**, including Teams and Zoom caption files.
- **Names and roles** of the people who sign in, and the names typed for a
  meeting's participants.

## Where it is processed

Everything below runs on the deployment's own servers:

- transcription (Whisper), telling speakers apart, reading the shared screen,
  restoring punctuation;
- placing facts in sections (embedding and entity models) and the rewriting of
  spoken sentences (a small local model);
- building the PDF.

The models are part of the server image, and the image runs with
`HF_HUB_OFFLINE=1`: no model is downloaded while a KT is processed, and no
transcript content goes to a model hub.

**The one external service is the optional LLM** (Groq, through an
OpenAI-compatible API):

- It is used only when the deployment has an API key. A workspace can turn it
  off entirely (`python -m auth llm-policy <workspace> none`), and its KTs are
  then built by the local models and rules alone.
- When it is used, it receives short pieces of the transcript text: sentences
  to rewrite or classify, a section's statements to format, and fields to
  fill. It never receives recordings, screenshots or documents.
- Credentials are removed before any of this. Keys, tokens, private keys,
  passwords in connection strings and variables named like secrets are
  replaced with `[REDACTED]` before the text reaches the LLM, the cache, the
  database or the PDF.
- Every prompt tells the model the transcript is data, not instructions, and
  every answer is checked against the transcript before it is used.
- What Groq keeps is governed by your agreement with Groq (or with the
  operator's account), not by Continuum.

## What is stored, and for how long

| What | Where | Kept until |
|---|---|---|
| The upload (recording) | The upload folder on the server | Deleted when its KT finishes. If the worker died twice, kept up to 24 hours (`CONTINUUM_RETRY_HOURS`) so the KT can be tried again, then deleted |
| Transcript, document, its versions, sources, sign-off record | The job database (SQLite) | The KT is deleted, by a user with the right, by the retention setting, or with the workspace |
| Screenshots from a shared screen | The assets folder, one folder per KT | Deleted with the KT |
| Rendered PDF | The job database | Deleted with the KT |
| LLM responses (no prompts) | The LLM cache, labelled with the workspace | 90 days after it was stored (`LLM_CACHE_TTL_DAYS`), or deleted with the workspace. **Deleting one KT does not remove its cached responses.** |
| Terms learned from transcripts (vocabulary candidates) | Per workspace | Deleted with the workspace |
| Sign-ins, role changes, deletions, sign-off steps | The audit log | Deleted with the workspace |
| Run numbers (timings, counts, tokens) and daily use | The job database | Numbers and ids only, no content. Deleted with the workspace |
| Application logs | The server's log output | Set by the host. Log lines are written to carry job, workspace and request ids and counts, not transcript text |
| Error reports (only if Sentry is configured) | Sentry | Sent without request bodies, local variables or personal data |

- **Retention:** `CONTINUUM_RETENTION_DAYS` deletes KTs unchanged for that many
  days. The workers apply it every hour. Unset, KTs are kept until someone
  deletes them.
- **Deleting a workspace** (`python -m auth delete-tenant`) removes its KTs,
  screenshots, cached LLM responses, vocabulary, people, keys, sessions,
  connections, limits and recorded use.

## Who can see it

- Each KT belongs to one workspace. Every request is checked against the
  caller's workspace, and another workspace's KT answers as if it did not
  exist.
- Within a workspace, roles decide who may create, correct, delete or only
  read. Receivers only read.
- Sign-in is through the customer's identity provider (OpenID Connect) or an
  API key. Keys and session tokens are stored only as hashes.
- Sign-off notices and review reminders go only to the workspace's own
  webhook. They carry the KT's name and the people's emails.
- Publishing to Confluence or Jira sends the document, or its gaps, to the
  workspace's own Atlassian site, and only when a user chooses to publish.

## Encryption

- **In transit:** TLS is terminated by the reverse proxy in front of the app.
  The app sends HSTS on HTTPS and marks its session cookie Secure there.
- **At rest:** the application does not encrypt its database or files itself.
  They rely on the host's disk encryption (for example encrypted cloud
  volumes), which the deployment must provide and state.
- **Backups** (`scripts/backup.py`) copy the database, screenshots and
  approved vocabulary to a folder, with checksums. Where that folder is kept,
  whether it is encrypted, and for how long, is set by the operator.

## What is not in the application yet

- Per-workspace retention: retention is one setting for the deployment.
- Encryption at rest inside the application: it relies on the host's disks.
- A deletion certificate: deletion is recorded in the audit log, but no
  certificate is issued.
