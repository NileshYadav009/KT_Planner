import contextlib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import os
import re

import logging
logger = logging.getLogger(__name__)

from api import router
import browser_security
import input_gate
import media_guard
import observability
import pipeline
import worker

# Structured logs and error tracking (P1-6; see observability.py).
observability.configure_logging()
observability.init_error_tracking()

@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup and shutdown, load_models() and stop_worker() below (a lifespan
    handler: on_event is deprecated in FastAPI)."""
    load_models()
    try:
        yield
    finally:
        stop_worker()


# The API reference is served by api/routes.py, to signed-in users only.
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

# CORS configuration - restrict to allowed origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Continuum-CSRF", "X-Continuum-Background",
                   "Idempotency-Key"],
)



@app.middleware("http")
async def request_id(request: Request, call_next):
    """Every log line written while serving a request carries its id, which
    is returned as X-Request-ID (or taken from the proxy's, if it sent one)."""
    import re
    import uuid

    supplied = request.headers.get("X-Request-ID", "")
    rid = supplied if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", supplied) else uuid.uuid4().hex[:16]
    token = observability.REQUEST_ID.set(rid)
    try:
        response = await call_next(request)
    finally:
        observability.REQUEST_ID.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


_PASTE_PATH_RE = re.compile(r"/kt-from-transcript|/kt/[^/]+/sessions")
_UPLOAD_PATH_RE = re.compile(r"/upload|/kt/[^/]+/sessions/upload")
# Every other request body is a small JSON form (an edit, a question, a
# sign-off step); none comes near this. They had no limit at all.
MAX_JSON_BODY_BYTES = 1024 * 1024


@app.middleware("http")
async def refuse_oversized_uploads(request: Request, call_next):
    """Refuse an upload that declares a size over the limit before its body
    is read (media_guard.save_upload also stops one that does not declare
    it), and likewise a pasted transcript far over its limit (the route
    counts its characters). Set the same limits on the reverse proxy in
    production."""
    if request.method == "POST":
        length = request.headers.get("content-length", "")
        length = int(length) if length.isdigit() else 0
        path = request.url.path
        if _UPLOAD_PATH_RE.fullmatch(path):
            if length > media_guard.max_upload_bytes() + 64 * 1024:   # + multipart overhead
                detail = (f"The file is larger than {media_guard.upload_limit_label()}. "
                          "Upload a shorter recording or the audio track only.")
                return JSONResponse({"detail": detail}, status_code=413)
        # A character is at most 6 bytes of JSON ("é").
        elif _PASTE_PATH_RE.fullmatch(path):
            if length > input_gate.MAX_PASTE_CHARS * 6 + 64 * 1024:
                detail = (f"The transcript is longer than the limit of {input_gate.MAX_PASTE_CHARS:,} characters. "
                          "Split it into sessions and add them to the KT one by one.")
                return JSONResponse({"detail": detail}, status_code=413)
        elif length > MAX_JSON_BODY_BYTES:
            return JSONResponse({"detail": "The request is too large."}, status_code=413)
    return await call_next(request)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Browser security headers on every response, including the refusals
    above (added last, so it runs first; see browser_security.py)."""
    response = await call_next(request)
    https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").lower() == "https"
    for name, value in browser_security.headers_for(request.url.path, https).items():
        response.headers.setdefault(name, value)
    return response


# A finished KT's /status and /schema are 100-200 KB of JSON (the document
# object); compressed they are about a seventh of that. Added last, so it
# compresses the final response.
app.add_middleware(GZipMiddleware, minimum_size=2048, compresslevel=5)     # 9 (the default) is slower, barely smaller

# Serve the frontend static files
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)


def load_models():
    """Load heavy models on startup so endpoints can use them."""
    interrupted = pipeline.JOB_STORE.mark_interrupted()
    if interrupted:
        logger.warning("%d job(s) were interrupted by the last shutdown and marked failed", interrupted)
    expired = pipeline.apply_retention()
    if expired:
        logger.info("Retention: deleted %d job(s) older than CONTINUUM_RETENTION_DAYS", expired)
    # KTs run on workers (worker.py). With CONTINUUM_WORKERS=0 they run in
    # separate worker processes and this process needs no models at all.
    threads = worker.configured_threads()
    if threads:
        pipeline.load_models()
        app.state.worker = worker.Worker(threads).start()


def stop_worker():
    """Stop claiming new KTs. A KT still running is picked up again by the
    next worker once its lease runs out (job_queue.py)."""
    running = getattr(app.state, "worker", None)
    if running is not None:
        running.stop(timeout=5)
