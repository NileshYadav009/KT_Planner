# Continuum application security audit (3 Nov 2026)

Scope: the application code as it runs on a development machine (`uvicorn
main:app`, workers, SQLite, local models, Groq). Hosting, network, WAF, cloud
IAM and managed secrets are out of scope; they are handled when the app moves
platform. Every finding below was checked against the code or by running it.
Nothing was tested against an external system.

## Summary

The application layer was already hardened in earlier rounds (audit §9aaz to
§9abg): sign-in with hashed keys and sessions, a CSRF header on cookie
writes, tenant isolation on every job read, a hash-based Content Security
Policy, upload and paste limits, a PDF renderer that loads nothing from
outside, secret redaction before any LLM call, and budgets on every LLM call.
This audit found no way to read another tenant's data and no injection path.

It found and fixed eight defects: one that corrupted a document, five that
let bad input reach a server error or the error text, one dependency
regression that would have silently broken PDFs, and 12 vulnerable packages
on the input path. Sixteen packages still carry advisories, most in the
model stack that was deliberately left alone.

| | Count |
|---|---|
| Fixed in this audit | 8 findings, 12 packages upgraded |
| Mitigated in code, recorded | 3 (python-jose, diskcache, gzip) |
| Open, low risk on a local machine | 5 |

## What is in place (verified)

| Area | Control | Where |
|---|---|---|
| Authentication | API keys of 256 random bits (`ckt_` + `token_urlsafe(32)`), stored as SHA-256 hashes; browser sessions as hashed tokens in an HttpOnly, SameSite=Lax cookie, Secure over HTTPS; 12 h lifetime, 30 min idle sign-out | `auth.py` |
| Authentication | Sign-in is on unless `CONTINUUM_AUTH=disabled`, which logs a warning on first use | `auth.auth_enabled`, `require_principal` |
| SSO | OpenID Connect with state cookie and nonce; ID tokens accepted only with an asymmetric algorithm from an allow-list, the provider's issuer and this client's audience (`azp` checked for several audiences) | `sso.py` |
| CSRF | A cookie-authenticated write needs the `X-Continuum-CSRF` header, which a cross-site form cannot send; CORS allows only configured origins | `auth.require_principal`, `main.py` |
| Authorization | Four roles (admin, reviewer, giver, receiver); create, correct and delete need the permission; a receiver reads only | `auth.PERMISSIONS`, `require_permission` |
| Tenant isolation | Every job read goes through `ensure_job_access`, which answers 404 for another tenant's job (the same as a missing one); lists and search are filtered by tenant in SQL | `auth.ensure_job_access`, `job_store.list/search` |
| Sign-off integrity | Only the named person can acknowledge a role, only the current version, only with every gap closed; a signed version is locked | `signoff.acknowledge`, `document_model.save_edits` |
| Browser | CSP: scripts only from this origin, one CDN and the hashes of the page's own inline scripts (recomputed when a page changes); `frame-ancestors 'none'`, `X-Frame-Options: DENY`, `nosniff`, `Referrer-Policy: same-origin`, HSTS over HTTPS; app shell `no-store` | `browser_security.py`, `api/routes.root` |
| XSS | Every value the page writes as HTML goes through `escapeHtml`; transcript text is set with `textContent`; links through `safeLink` | `static/index.html` |
| Browser storage | Only the theme and the last KT's id; an old API key entry is removed | `static/index.html` |
| Uploads | Streamed to disk with a size cap (2 GB), refused early from Content-Length; identified by ffprobe, only audio and video containers; HLS playlists and concat lists refused (they can make ffmpeg read server files or URLs); safe ffmpeg arguments; the file is deleted when its task ends | `media_guard.py`, `main.refuse_oversized_uploads` |
| Pastes | 150,000 characters, refused early from Content-Length; cleaned on a thread; an input gate refuses text with no operational content | `input_gate.py`, `api/routes._paste_task` |
| PDF rendering | WeasyPrint loads only `data:` URIs and the bundled font files by exact path; any other URL in a document is refused, so transcript text cannot make the renderer read files or call internal addresses | `pdf_rendering._document_fetcher` |
| Screenshots | Served only for the exact `screen_NN.jpg` name under a validated job id, with a short-lived HMAC signature in the URL | `api/routes.kt_asset`, `auth.sign_asset_urls` |
| SQL | Parameterised everywhere; the few f-strings insert fixed table and column names only | `job_store.py`, `auth.py`, `migrations.py` |
| Secrets | Credentials read out or pasted are redacted before the LLM, the cache, the database and the PDF; the Groq key lives only in `.env` (git-ignored) and is never logged | `redaction.py`, `.gitignore` |
| LLM: prompt injection | All seven prompts say the transcript is data and its instructions are ignored; every answer is checked: grounded against the transcript, limited to the offered choices, and refused if it adds specifics, flips a negation or drops a fragment | `llm/prompts.py`, `copyedit.py`, `context_mapper.py`, `field_populator.py`, `conflicts.py`, `grounding.py`, `ai._polish_one` |
| LLM: resource use | 60 s per request, 900 s and a call cap per KT, no SDK retries; token-rate pacing; a tenant can forbid external LLM calls | `llm/cached_provider.py`, `llm_provider.py` |
| Data lifecycle | Delete removes a KT's record, transcript, document and screenshots; tenant delete; retention by age; backups with a restore drill | `pipeline.delete_job_data`, `auth.delete_tenant`, `scripts/backup.py` |
| Operations endpoints | `/metrics` only with its own bearer token (404 without one configured); API docs only for signed-in users | `api/routes.metrics`, `openapi_schema` |

## Findings fixed in this audit

| ID | Sev | Finding | Fix | Test |
|---|---|---|---|---|
| S1 | High | **A section correction moved the wrong sentence.** The Transcript tab numbers sentences across the whole KT; `/feedback` looked the number up within each section, so it moved another section's sentence and saved that as a new document version. It also left the moved sentence in its old section's list, and answered "feedback applied" for a sentence that did not exist. | The number is read as the page counts it; the page also sends the sentence's text, and a mismatch is refused (409) instead of guessed; the sentence leaves its old section; an unknown sentence is a 404 and a KT still being built a 409 | `tests/test_review_signoff.py::test_a_move_picks_the_sentence_the_transcript_tab_numbered`; browser journey J09 |
| S2 | Med | **Wrong-typed input reached server errors (500)** on nine requests across four causes: a non-number `base_version` in edits, a version past SQLite's integer range in the version, diff and PDF routes, a non-text section id in `/feedback`, and an unchecked version in sign-off acknowledge. | Version numbers are validated once (`document_model.version_number`) and out-of-range versions never reach SQLite; types are checked before use | `tests/test_review_signoff.py::test_wrong_typed_version_numbers_are_refused_not_server_errors`, `::test_an_acknowledgement_of_a_wrong_typed_version_is_refused_not_a_server_error`; a probe over every write route now finds no 500 |
| S3 | Med | **Upload errors returned Python's exception text**, which can name server paths ("Permission denied: 'C:\\...\\upload-…'"). | Logged in full; the client gets a plain message with the request id to give the operator | `tests/test_upload_hardening.py::test_an_unexpected_upload_failure_does_not_echo_server_details` |
| S4 | Med | **No size limit on most request bodies.** Only uploads and pastes were capped; a megabyte or more of JSON to any other route was read whole. A follow-up session's recording also had no early size check. | 1 MB cap on every other body; the session upload gets the upload limit | `tests/test_upload_hardening.py::test_other_request_bodies_have_a_size_limit` |
| S5 | Low | **Ask the KT took a question of any length.** | 500 characters (the page allows 300) | `tests/test_ask.py` |
| S6 | High | **Vulnerable packages on the input path.** pip-audit: 28 of 164 packages, 249 advisories. | Upgraded the packages that parse untrusted input or make outbound calls: starlette 0.52 → 1.10 with fastapi 0.129 → 0.162, python-multipart 0.0.22 → 0.0.33, pillow 12.3 → 13.0, weasyprint 69 → 70, urllib3 2.6 → 2.8, requests 2.32 → 2.34, cryptography 46 → 50, idna, certifi, pyasn1, anyio, python-dotenv. `requirements.lock` updated; minimums raised in `requirements.txt` | After: 16 packages, 148 advisories (table below) |
| S7 | High | **WeasyPrint 70 broke the PDF's resource loading** (found by the tests after S6). It removed `default_url_fetcher`, which the safe fetcher imported, so every font and every embedded screenshot was refused: PDFs were set in a system font and would have dropped their screenshots. | The fetcher is a `URLFetcher` subclass with the same allow-list; it still works with older WeasyPrint | `tests/test_pdf_security.py` (font, refusals, embedded screenshots) |
| S8 | Low | `@app.on_event` startup and shutdown are deprecated (FastAPI) and removed in a later Starlette | A lifespan handler | app start checked |

## Mitigated in code, recorded

| Item | Why it does not apply here |
|---|---|
| python-jose 3.5.0, CVE-2026-85394 (no fixed release): HS256 tokens forged with the public key "when algorithms are not explicitly restricted" | `sso.verify_id_token` passes exactly one algorithm, taken from an allow-list of RS, PS and ES algorithms only. Recommended: move SSO to a maintained JWT library (PyJWT or joserfc) when SSO is next touched. |
| diskcache 5.6.3, CVE-2025-69872 (pickle in the cache) | Used only by `llama_cpp_python` for the optional local copy-edit model; exploiting it needs write access to the local cache folder. |
| Response compression (GZip, added for speed) and BREACH | Compressed responses carry no secret: API keys and session tokens are never in a body, and the CSRF defence is a header, not a token in the page. |

## Open, low risk on a local machine

| Item | Note |
|---|---|
| Model stack advisories: torch 2.10 (2), transformers 5.1 (6), sentence-transformers 5.2.3 (1), nltk 3.9.2 (73) | Kept by decision: an upgrade can move placement accuracy, so it needs the full accuracy sets re-run. The models load only from the local Hugging Face cache (`HF_HUB_OFFLINE=1` in the harnesses), not from untrusted input. |
| Tooling and test-only packages: pypdf 6.14 (34, used only by tests and evaluation on PDFs the app wrote), pip, setuptools, pygments, filelock, fsspec, msgpack, brotli, ecdsa, pydantic-settings | Upgrade with the next dependency refresh. |
| No sign-in throttling on `/login` | Keys carry 256 random bits, so guessing is not practical; rate limits belong at the proxy when hosting moves. |
| A request without Content-Length is not refused early | Browsers always send it; uploads are still capped while streaming (`media_guard.save_upload`). A proxy body limit covers the rest in production. |
| `CONTINUUM_AUTH=disabled` makes every request the local admin | Local development only; it logs a warning. Never set it on a shared machine. |
| WeasyPrint 70 warns that HarfBuzz-Subset will be required by a future version | Install the system library before the next WeasyPrint major upgrade. |

## How it was checked

- Code read: every route in `api/routes.py`, `main.py` middleware, `auth.py`, `sso.py` token checks, `signoff.py`, `media_guard.py`, `pdf_rendering.py` fetcher, every prompt, the page's HTML-writing code and storage use.
- A probe sent wrong-typed JSON (null, numbers, strings, lists, objects, booleans, huge numbers) to every write route and odd version numbers to every read route of a finished KT: 21 server errors before (9 real, 12 from the probe's own data), none after.
- pip-audit over the app's environment, before and after the upgrade.
- ruff (pyflakes and bugbear rules) over the code: no undefined names; two repeated dictionary keys with different values fixed (one dropped a test expectation without anyone knowing, one mislabelled Amazon ECR as Security in the technology summary).
- The browser journey (`evaluation/browser_journey.py`) checks sign-in, roles, sign-out and the back button, and that no script error or CSP violation occurs; its results are in `TEST_CASES.md`.
