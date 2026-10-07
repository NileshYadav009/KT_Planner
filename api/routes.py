"""HTTP endpoints for the KT Planner API.

Moved out of main.py as part of the Phase 3 architecture split (see
REPOSITORY_AUDIT.md) — logic unchanged, just relocated onto an APIRouter.
Job state is read via `pipeline.JOB_QUEUE`/`pipeline.JOB_LOCK` (module-attribute
access, not `from pipeline import ...`). Submitted KTs are queued
(`pipeline.TASKS`, job_queue.py) and run by workers (worker.py), in this
process or another; job state written by a worker is read from the database.
"""

import logging
import os
import re
import uuid
from datetime import datetime

from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, Depends, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool

from auth import (
    ROLE_LABELS, SESSION_COOKIE, SESSION_TTL_SECONDS, Principal, audit, can_delete_job, create_session, end_session,
    ensure_job_access, find_user, optional_principal, principal_for_key, require_permission, require_principal,
    sign_asset_urls, tenant_name, verify_asset_signature,
)
import document_model
import signoff
from devops_transcription import clean_transcript
from input_gate import assess_transcript
from job_queue import valid_idempotency_key
from kt_schema_loader import SCHEMA
import media_guard
import pdf_rendering
import pipeline
import sso

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/healthz")
async def healthz():
    """Liveness: the process is up and serving. Models may still be loading
    (see /readyz); an orchestrator restarts the container only when this fails."""
    ready = pipeline.MAPPER_PIPELINE is not None and pipeline.MODEL is not None
    return JSONResponse({"status": "ok", "ready": ready})


@router.get("/readyz")
async def readyz():
    """Readiness: send traffic only once the database is writable, ffmpeg is
    installed and, when this process runs KTs itself, the models are loaded.
    503 lists what is not ready."""
    import shutil

    import worker

    checks = {
        "database_writable": pipeline.JOB_STORE.writable(),
        "ffmpeg": shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None,
    }
    if worker.configured_threads():
        checks["models_loaded"] = pipeline.MAPPER_PIPELINE is not None and pipeline.MODEL is not None
    ready = all(checks.values())
    return JSONResponse({"ready": ready, "checks": checks}, status_code=200 if ready else 503)


@router.get("/metrics")
async def metrics(request: Request):
    """Prometheus metrics (P1-6): KT outcomes, stage latency and failures, LLM
    calls, tokens and fallbacks, the false-missing estimate, and the queue.
    Served only when CONTINUUM_METRICS_TOKEN is set, to that bearer token;
    deployment-wide figures, so no workspace's key or session opens it."""
    import hmac

    import observability

    token = os.getenv("CONTINUUM_METRICS_TOKEN")
    if not token:
        raise HTTPException(status_code=404, detail="Not found")
    supplied = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(status_code=401, detail="Metrics token required")
    runs = await run_in_threadpool(pipeline.JOB_STORE.runs)
    queue = await run_in_threadpool(pipeline.TASKS.stats)
    return Response(observability.prometheus_text(runs, queue), media_type="text/plain; version=0.0.4")


@router.get("/schema")
async def get_schema():
    return {"sections": SCHEMA}


@router.get("/schema/{job_id}")
async def get_job_schema(job_id: str, principal: Principal = Depends(require_principal)):
    with pipeline.JOB_LOCK:
        job = ensure_job_access(pipeline.JOB_QUEUE.get(job_id), principal)
    job = sign_asset_urls(job)
    dynamic = job.get("dynamic_schema", SCHEMA)
    populated = job.get("populated_fields", {})
    return {
        "sections": dynamic,
        "populated_fields": populated,
        "field_count": sum(len(v) for v in populated.values()),
        "auto_filled_count": sum(
            1 for sec in populated.values()
            for f in sec.values()
            if isinstance(f, dict) and f.get("source") not in ("unfilled", "")
        ),
        "knowledge_object": job.get("knowledge_object", {}),
        "validation_warnings": job.get("validation_warnings", []),
        "quality_score": job.get("quality_score", {}),
    }


@router.get("/export/pdf/{job_id}")
async def export_pdf(job_id: str, version: Optional[int] = None, principal: Principal = Depends(require_principal)):
    with pipeline.JOB_LOCK:
        job = ensure_job_access(pipeline.JOB_QUEUE.get(job_id), principal)
    if job.get("status") not in pipeline.COMPLETED_STATUSES:
        raise HTTPException(status_code=404, detail="Job not found or not completed")

    current = document_model.current_version(job)
    if version is not None and version != current:
        # An earlier version (P1-1), exactly as it was saved.
        snap = await run_in_threadpool(document_model.version_snapshot, pipeline, job_id, job, version)
        if snap is None:
            raise HTTPException(status_code=404, detail="No such version")
        old = {"knowledge_object": snap, "warnings": job.get("warnings"),
               "notices": [f"Version {version} of {current}: superseded by a later version."]}
        pdf_bytes = await run_in_threadpool(pdf_rendering.build_job_pdf, job_id, old,
                                            pipeline.JOB_STORE.created_at(job_id))
        return Response(content=pdf_bytes, media_type="application/pdf",
                        headers={"Content-Disposition": f"attachment; filename=\"kt_{job_id[:8]}_v{version}.pdf\""})

    # The worker renders and stores the PDF when the KT finishes (P1-4), so
    # a download is normally a read. A KT changed since (a correction) or
    # finished before documents were stored is rendered here once, off the
    # event loop, and stored for the next download.
    pdf_bytes = await run_in_threadpool(pipeline.JOB_STORE.current_document, job_id)
    if pdf_bytes is None:
        try:
            import weasyprint  # noqa: F401
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="WeasyPrint is not available. Install required native dependencies and Python packages.",
            ) from exc
        pdf_bytes = await run_in_threadpool(pipeline.render_and_store_document, job_id)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=\"kt_{job_id[:8]}.pdf\""},
    )


@router.get("/kt-assets/{job_id}/{name}")
def kt_asset(job_id: str, name: str, exp: Optional[int] = None, sig: Optional[str] = None):
    """A screenshot captured from the shared screen (screen_capture.py).
    Only the job id and the exact screen_NN.jpg name pattern are accepted, so
    no other file can be reached through this route. An <img> tag cannot send
    an API key, so the URL must carry a short-lived signature issued by
    /status or /schema to a caller with access to the job."""
    if not re.fullmatch(r"[0-9A-Za-z\-]{8,64}", job_id or "") or not re.fullmatch(r"screen_\d{2}\.jpg", name or ""):
        raise HTTPException(status_code=404, detail="Not found")
    if not verify_asset_signature(f"/kt-assets/{job_id}/{name}", exp, sig):
        raise HTTPException(status_code=404, detail="Not found")
    path = os.path.join(pipeline.KT_ASSETS_DIR, job_id, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="image/jpeg")


def _new_job(principal: Principal) -> dict:
    return {"status": "queued", "progress": 0, "tenant_id": principal.tenant_id,
            "created_by": principal.key_label, "created_by_id": principal.actor}


def _idempotency_key(request: Request) -> Optional[str]:
    """The client's Idempotency-Key header (P1-4): a retried or double-clicked
    submission with the same key returns the first job instead of paying
    for the work twice. Keys are per workspace and kept for 24 hours."""
    key = request.headers.get("Idempotency-Key")
    if key is None:
        return None
    key = key.strip()
    if not valid_idempotency_key(key):
        raise HTTPException(status_code=400, detail="Idempotency-Key must be 1 to 128 letters, digits or . _ : -")
    return key


def _queued_response(job_id: str, message: str, duplicate: bool = False) -> dict:
    job = pipeline.JOB_QUEUE.get(job_id) or {}
    response = {"job_id": job_id, "status": job.get("status", "queued"), "message": message}
    if duplicate:
        response["duplicate"] = True
    return response


async def _reserve(principal: Principal, key: Optional[str], job_id: str) -> Optional[str]:
    """None when this request may create `job_id`; the earlier job's id when
    the key was already used."""
    if not key:
        return None
    winner = await run_in_threadpool(pipeline.TASKS.reserve_key, principal.tenant_id, key, job_id)
    return None if winner == job_id else winner


@router.post("/upload")
async def upload(file: UploadFile, request: Request, principal: Principal = Depends(require_permission("kt:create"))):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    key = _idempotency_key(request)
    job_id = str(uuid.uuid4())
    earlier = await _reserve(principal, key, job_id)
    if earlier:
        return _queued_response(earlier, "Already submitted. Poll /status/{job_id} for results.", duplicate=True)

    input_path = None
    try:
        # Streamed to disk with a size cap, then identified by ffprobe with
        # only audio/video containers allowed, before anything decodes it
        # (media_guard.py). A refusal comes back now, not as a failed job.
        input_path = await media_guard.save_upload(file)
        media = await run_in_threadpool(media_guard.probe, input_path)

        with pipeline.JOB_LOCK:
            pipeline.JOB_QUEUE[job_id] = _new_job(principal)
        # A worker transcribes and builds it (worker.py); the file is deleted
        # when the task ends.
        pipeline.TASKS.enqueue(job_id, "upload", {"input_path": input_path, "media_format": media["format"]})
        return _queued_response(job_id, "File queued for processing. Poll /status/{job_id} for results.")
    except Exception as e:
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)
        if key:
            pipeline.TASKS.release_key(principal.tenant_id, key, job_id)
        if isinstance(e, media_guard.MediaRejected):
            raise HTTPException(status_code=e.status, detail=e.message)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/kt-from-transcript")
async def kt_from_transcript(payload: dict, request: Request,
                             principal: Principal = Depends(require_permission("kt:create"))):
    """Run the KT pipeline directly from a transcript, skipping audio/Whisper entirely.

    A test/dev entry point so classification, vocabulary corrections, and
    rendering can be iterated on without recording audio or using the UI each
    time (see scripts/test_kt_from_transcript.py). Poll the returned job_id via
    the existing GET /status/{job_id} and GET /export/pdf/{job_id} endpoints —
    same as a real upload.
    """
    transcript = payload.get("transcript", "")
    if not isinstance(transcript, str) or not transcript.strip():
        raise HTTPException(status_code=400, detail="transcript is required and must be non-empty.")
    key = _idempotency_key(request)

    # Same normalization real audio gets, so this path actually exercises the
    # vocabulary-correction pipeline being tested.
    cleaned_transcript = clean_transcript(transcript)

    # Refuse text that cannot produce a KT document (too little operational
    # content, or caption-style text with no punctuation). "force": true skips
    # the refusal for testing; the reasons still reach the document as warnings.
    gate = assess_transcript(cleaned_transcript, source="paste")
    if gate["verdict"] == "reject" and not payload.get("force"):
        raise HTTPException(status_code=422, detail=" ".join(gate["reasons"]))

    job_id = str(uuid.uuid4())
    earlier = await _reserve(principal, key, job_id)
    if earlier:
        return _queued_response(earlier, "Already submitted. Poll /status/{job_id} for results.", duplicate=True)
    with pipeline.JOB_LOCK:
        pipeline.JOB_QUEUE[job_id] = _new_job(principal)
    pipeline.TASKS.enqueue(job_id, "transcript", {
        "transcript": cleaned_transcript, "warnings": gate["reasons"] if gate["verdict"] != "ok" else []})
    return _queued_response(job_id, "Transcript queued for processing. Poll /status/{job_id} for results.")


@router.get("/jobs")
async def list_jobs(limit: int = 50, principal: Principal = Depends(require_principal)):
    """The caller's KT jobs, newest first (persisted across restarts)."""
    jobs = pipeline.JOB_STORE.list(principal.tenant_id, limit=max(1, min(limit, 200)))
    for job in jobs:
        job["can_delete"] = can_delete_job(job, principal)
        job.pop("created_by_id", None)
    return {"jobs": jobs}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, principal: Principal = Depends(require_principal)):
    """Delete a KT: its record, transcript, document and screenshots.
    Admins may delete any KT in the workspace; others only their own."""
    with pipeline.JOB_LOCK:
        job = ensure_job_access(pipeline.JOB_QUEUE.get(job_id), principal)
        if not can_delete_job(job, principal):
            raise HTTPException(status_code=403, detail="Only a workspace admin or the person who created this KT can delete it.")
        title = (job.get("knowledge_object") or {}).get("system_name") or job.get("title")
        pipeline.delete_job_data(job_id)
    audit(principal.tenant_id, principal.key_label, "kt_deleted", {"job_id": job_id, "title": title})
    return {"status": "deleted", "job_id": job_id}


# --------------------------------------------------------------------------
# Reviewer edits and versions (P1-1, document_model.py)
# --------------------------------------------------------------------------

@router.get("/documents/{job_id}/versions")
async def document_versions(job_id: str, principal: Principal = Depends(require_principal)):
    with pipeline.JOB_LOCK:
        job = ensure_job_access(pipeline.JOB_QUEUE.get(job_id), principal)
    return document_model.versions(pipeline, job_id, job)


@router.get("/documents/{job_id}/versions/{version}")
async def document_version(job_id: str, version: int, principal: Principal = Depends(require_principal)):
    with pipeline.JOB_LOCK:
        job = ensure_job_access(pipeline.JOB_QUEUE.get(job_id), principal)
    snap = document_model.version_snapshot(pipeline, job_id, job, version)
    if snap is None:
        raise HTTPException(status_code=404, detail="No such version")
    return dict(snap, version=version, current=document_model.current_version(job))


@router.post("/documents/{job_id}/edits")
async def edit_document(job_id: str, payload: dict, principal: Principal = Depends(require_permission("kt:correct"))):
    """Save a reviewer's edits as a new version. 409 if the document changed
    since `base_version`, or it is signed."""
    with pipeline.JOB_LOCK:
        job = ensure_job_access(pipeline.JOB_QUEUE.get(job_id), principal)
    if signoff.state(job).get("status") != "draft" and any(
            isinstance(e, dict) and e.get("section") == "signoff" for e in payload.get("edits") or []):
        raise HTTPException(status_code=409, detail="The Sign-off section is the recorded sign-off; it is not edited")
    try:
        saved = await run_in_threadpool(document_model.save_edits, pipeline, job_id, payload.get("edits"),
                                        payload.get("base_version", 0), principal.actor)
    except document_model.EditError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))
    audit(principal.tenant_id, principal.key_label, "kt_edited",
          {"job_id": job_id, "version": saved["version"], "summary": saved["summary"]})
    job = pipeline.JOB_QUEUE.get(job_id) or {}
    return _with_signoff_view({"version": saved["version"], "summary": saved["summary"],
                               "knowledge_object": saved["knowledge_object"], "signoff": job.get("signoff"),
                               "document_version": saved["version"]})


# --------------------------------------------------------------------------
# Sign-off (P1-3, signoff.py)
# --------------------------------------------------------------------------

def _signoff_response(job_id: str, job: dict, principal: Principal) -> dict:
    return dict(signoff.public_state(job, principal.user_id, document_model.current_version(job)), job_id=job_id)


def _signoff_change(job_id: str, principal: Principal, change):
    """Run `change(job)` on the stored job and save it; SignoffError -> HTTP."""
    with pipeline.JOB_LOCK:
        job = ensure_job_access(pipeline.JOB_QUEUE.get(job_id), principal)
        if job.get("status") not in pipeline.COMPLETED_STATUSES:
            raise HTTPException(status_code=409, detail="Only a finished KT can be signed off")
        try:
            result = change(job)
        except signoff.SignoffError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc))
        pipeline.JOB_QUEUE[job_id] = job
    return job, result


def _actor(principal: Principal) -> dict:
    return {"user_id": principal.user_id, "email": principal.key_label}


@router.get("/signoff/{job_id}")
async def get_signoff(job_id: str, principal: Principal = Depends(require_principal)):
    with pipeline.JOB_LOCK:
        job = ensure_job_access(pipeline.JOB_QUEUE.get(job_id), principal)
    return _signoff_response(job_id, job, principal)


@router.post("/signoff/{job_id}/start")
async def start_signoff(job_id: str, payload: dict, principal: Principal = Depends(require_permission("kt:correct"))):
    """Name the giver, receiver and approver (emails of people in this
    workspace). Their knowledge gaps come from the document as it is now."""
    people = {}
    for role in signoff.ROLES:
        email = payload.get(role)
        person = find_user(principal.tenant_id, email=email) if isinstance(email, str) and email.strip() else None
        if email and person is None:
            raise HTTPException(status_code=400, detail=f"{email} is not an active person in this workspace")
        people[role] = person
    job, state = _signoff_change(job_id, principal, lambda j: signoff.start(
        j, people, document_model.current_version(j)))
    audit(principal.tenant_id, principal.key_label, "kt_signoff_started",
          {"job_id": job_id, "participants": {r: p["email"] for r, p in state["participants"].items()},
           "gaps": len(state["gaps"])})
    return _signoff_response(job_id, job, principal)


@router.post("/signoff/{job_id}/gaps/{gap_id}")
async def resolve_signoff_gap(job_id: str, gap_id: str, payload: dict, principal: Principal = Depends(require_principal)):
    """Close a knowledge gap or accept it as a risk, with a note. Any named
    participant (or a workspace admin) may do it."""
    with pipeline.JOB_LOCK:
        job = ensure_job_access(pipeline.JOB_QUEUE.get(job_id), principal)
    if not signoff.roles_of(job, principal.user_id) and principal.role != "admin":
        raise HTTPException(status_code=403, detail="Only the people named for this sign-off can resolve its gaps")
    job, gap = _signoff_change(job_id, principal, lambda j: signoff.resolve_gap(
        j, gap_id, payload.get("resolution"), payload.get("note"), _actor(principal)))
    audit(principal.tenant_id, principal.key_label, "kt_gap_resolved",
          {"job_id": job_id, "gap_id": gap_id, "resolution": gap["resolution"], "note": gap["note"]})
    return _signoff_response(job_id, job, principal)


@router.post("/signoff/{job_id}/acknowledge")
async def acknowledge_signoff(job_id: str, payload: dict, principal: Principal = Depends(require_principal)):
    """The named giver, receiver or approver acknowledges a version. The KT is
    signed, and that version locked, when all three have acknowledged it
    with no knowledge gap left open."""
    role = payload.get("role")
    job, signed = _signoff_change(job_id, principal, lambda j: signoff.acknowledge(
        j, role, payload.get("version", 0), _actor(principal), document_model.current_version(j)))
    state = signoff.state(job)
    audit(principal.tenant_id, principal.key_label, "kt_signoff_acknowledged",
          {"job_id": job_id, "role": role, "version": payload.get("version")})
    if signed:
        audit(principal.tenant_id, principal.key_label, "kt_signed", {
            "job_id": job_id, "version": state["signed_version"],
            "participants": {r: p["email"] for r, p in state["participants"].items()},
            "acknowledged_at": {r: a["at"] for r, a in state["acknowledgements"].items()},
            "gaps": [{"gap": g["text"][:200], "resolution": g["resolution"], "by": g.get("by")} for g in state["gaps"]],
        })
    return _signoff_response(job_id, job, principal)


@router.get("/status/{job_id}")
async def get_status(job_id: str, principal: Principal = Depends(require_principal)):
    """Poll job status."""
    with pipeline.JOB_LOCK:
        job = ensure_job_access(pipeline.JOB_QUEUE.get(job_id), principal)
    job = sign_asset_urls(dict(job, job_id=job_id))
    if job.get("status") == "queued":
        # KTs ahead of this one, across the deployment (a count only).
        job["queue_position"] = pipeline.TASKS.position(job_id)
    return _with_signoff_view(job)


def _with_signoff_view(job: dict) -> dict:
    """The document as it is exported: its Sign-off section is the recorded
    sign-off once one has started (P1-3)."""
    ko = job.get("knowledge_object")
    if isinstance(ko, dict) and ko.get("rendered_sections"):
        version = document_model.current_version(job)
        job["knowledge_object"] = dict(ko, rendered_sections=signoff.apply_to_document(
            ko["rendered_sections"], job, version))
        job["document_version"] = version
    return job


_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
# The app shell and the sign-in page must never be served from a cache: a
# cached app shell would appear after sign-out.
_NO_STORE = {"Cache-Control": "no-store"}


@router.get("/")
async def root(request: Request):
    """The app for a signed-in user; the sign-in page for everyone else."""
    if optional_principal(request) is None:
        return FileResponse(os.path.join(_STATIC_DIR, "login.html"), headers=_NO_STORE)
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, headers=_NO_STORE)
    return HTMLResponse(content="KT Planner API is running.")


def _secure_cookies(request: Request) -> bool:
    return request.url.scheme == "https" or os.getenv("CONTINUUM_SECURE_COOKIES") == "1"


def _set_session_cookie(resp, request: Request, principal: Principal) -> None:
    resp.set_cookie(SESSION_COOKIE, create_session(principal), max_age=SESSION_TTL_SECONDS, httponly=True,
                    samesite="lax", secure=_secure_cookies(request), path="/")


@router.post("/login")
async def login(payload: dict, request: Request):
    """Exchange an API key for an HttpOnly session cookie (browser sign-in)."""
    principal = principal_for_key(str(payload.get("api_key") or "").strip())
    if principal is None:
        raise HTTPException(status_code=401, detail="That API key was not accepted.")
    if sso.sso_enforced(principal.tenant_id):
        audit(principal.tenant_id, principal.key_label, "sign_in_refused", {"method": "key", "reason": "sso_required"})
        raise HTTPException(status_code=403, detail="This workspace signs in with single sign-on. "
                                                    "Use Continue with SSO and your work email.")
    resp = JSONResponse({"status": "signed_in", "workspace": tenant_name(principal.tenant_id)})
    _set_session_cookie(resp, request, principal)
    audit(principal.tenant_id, principal.key_label, "sign_in", {"method": "key"})
    return resp


@router.post("/logout")
async def logout(request: Request):
    principal = end_session(request.cookies.get(SESSION_COOKIE))
    if principal is not None:
        audit(principal.tenant_id, principal.key_label, "sign_out", {"method": principal.method})
    resp = JSONResponse({"status": "signed_out"})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@router.get("/me")
async def me(principal: Principal = Depends(require_principal)):
    """Who the caller is signed in as (workspace, person and role in the UI header)."""
    return {"tenant_id": principal.tenant_id, "workspace": tenant_name(principal.tenant_id),
            "key_label": principal.key_label, "user": principal.key_label, "method": principal.method,
            "role": principal.role, "role_label": ROLE_LABELS.get(principal.role, principal.role),
            "permissions": sorted(principal.permissions),
            "auth": "disabled" if principal.method == "disabled" else "on"}


# --------------------------------------------------------------------------
# Single sign-on (OpenID Connect, see sso.py)
# --------------------------------------------------------------------------

def _sso_failed(code: str) -> RedirectResponse:
    """Back to the sign-in page, which turns the code into a message. Only
    known codes are passed, never free text."""
    resp = RedirectResponse(f"/?sso_error={code}", status_code=302, headers=_NO_STORE)
    resp.delete_cookie(sso.STATE_COOKIE, path="/sso")
    return resp


@router.get("/sso/start")
async def sso_start(request: Request, email: str = "", next: str = "/"):
    """Send the browser to the identity provider of the workspace that owns
    this email domain."""
    provider = sso.provider_for_email(email)
    if provider is None:
        return _sso_failed("sso_unknown_domain")
    try:
        url, state = await run_in_threadpool(sso.begin, provider, sso.redirect_uri_for(str(request.base_url)),
                                              next, email.strip().lower())
    except sso.SSOError as exc:
        logger.warning("SSO start failed for tenant %s: %s", provider["tenant_id"], exc.reason)
        return _sso_failed("sso_failed")
    resp = RedirectResponse(url, status_code=302, headers=_NO_STORE)
    resp.set_cookie(sso.STATE_COOKIE, state, max_age=sso.REQUEST_TTL_SECONDS, httponly=True, samesite="lax",
                    secure=_secure_cookies(request), path="/sso")
    return resp


@router.get("/sso/callback")
async def sso_callback(request: Request, code: Optional[str] = None, state: Optional[str] = None,
                       error: Optional[str] = None):
    """The identity provider sends the browser back here with a code."""
    tenant_id = sso.tenant_for_state(state)
    if error:
        audit(tenant_id, None, "sign_in_failed", {"method": "sso", "reason": "provider_error", "error": error[:80]})
        return _sso_failed("sso_denied")
    try:
        principal, next_path = await run_in_threadpool(sso.complete, code, state, request.cookies.get(sso.STATE_COOKIE))
    except sso.SSOError as exc:
        logger.warning("SSO sign-in failed (tenant %s): %s", tenant_id, exc.reason)
        audit(tenant_id, None, "sign_in_failed", {"method": "sso", "reason": exc.code, "detail": exc.reason[:200]})
        return _sso_failed(exc.code)
    resp = RedirectResponse(next_path, status_code=302, headers=_NO_STORE)
    resp.delete_cookie(sso.STATE_COOKIE, path="/sso")
    _set_session_cookie(resp, request, principal)
    audit(principal.tenant_id, principal.key_label, "sign_in", {"method": "sso", "role": principal.role})
    return resp


@router.post('/feedback')
async def receive_feedback(payload: dict, principal: Principal = Depends(require_permission("kt:correct"))):
    """Accept human feedback from the UI, apply correction to coverage, and return updated state.

    Expected payload keys: job_id, sentence_id, corrected_classification, user, feedback_notes
    """
    job_id = payload.get('job_id')
    if not job_id:
        raise HTTPException(status_code=400, detail='job_id is required')

    sentence_id = payload.get('sentence_id')
    corrected_section = payload.get('corrected_classification')
    if sentence_id is None or not corrected_section:
        raise HTTPException(status_code=400, detail='sentence_id and corrected_classification required')

    # Validate corrected_section against schema
    valid_section_ids = {s["id"] for s in SCHEMA}
    if corrected_section not in valid_section_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid section id '{corrected_section}'. Valid options: {sorted(valid_section_ids)}"
        )

    with pipeline.JOB_LOCK:
        job = ensure_job_access(pipeline.JOB_QUEUE.get(job_id), principal)

        # Record feedback
        fb = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'sentence_id': sentence_id,
            'corrected_classification': corrected_section,
            # Who made the change comes from the authenticated key, never
            # from the request body (which let any caller sign as anyone).
            'user': f"{principal.tenant_id}:{principal.key_label}",
            'notes': payload.get('feedback_notes', '')
        }
        job.setdefault('human_feedback', []).append(fb)

        # Apply correction to coverage data
        coverage = job.get('coverage', {})

        # Find the sentence in the flat list across all sections
        sentence_found = False
        moved_text = None
        for sec_id, sec_info in coverage.items():
            sentences = sec_info.get('sentences', [])
            for idx, sent in enumerate(sentences):
                # Match by sentence text or by index
                if idx == sentence_id or (isinstance(sentence_id, str) and sent.get('text') == sentence_id):
                    # Remove sentence from old sections and add to new section
                    old_sections = sent.get('assigned_sections', [])

                    # Update the sentence's assigned_sections
                    sent['assigned_sections'] = [corrected_section]

                    # Move sentence content to the correct section
                    sentence_text = sent.get('text', '')

                    # Remove from old sections' content list
                    for old_sec_id in old_sections:
                        if old_sec_id in coverage and old_sec_id != corrected_section:
                            old_content = coverage[old_sec_id].get('content', [])
                            if sentence_text in old_content:
                                old_content.remove(sentence_text)
                            coverage[old_sec_id]['content'] = old_content
                            coverage[old_sec_id]['sentence_count'] = len(coverage[old_sec_id].get('sentences', []))

                    # Add to new section's content list
                    if corrected_section not in coverage:
                        coverage[corrected_section] = {
                            'title': corrected_section,
                            'status': 'covered',
                            'sentence_count': 0,
                            'confidence': 0.85,
                            'risk': 0.0,
                            'content': [],
                            'sentences': []
                        }
                    if sentence_text not in coverage[corrected_section].get('content', []):
                        coverage[corrected_section]['content'].append(sentence_text)
                    if sent not in coverage[corrected_section].get('sentences', []):
                        coverage[corrected_section]['sentences'].append(sent)

                    # Recalculate section metrics
                    for sec_data in coverage.values():
                        sent_count = len(sec_data.get('sentences', []))
                        sec_data['sentence_count'] = sent_count
                        if sent_count >= 2:
                            sec_data['status'] = 'covered'
                        elif sent_count == 1:
                            sec_data['status'] = 'weak'
                        else:
                            sec_data['status'] = 'missing'

                    sentence_found = True
                    moved_text = sentence_text
                    break
            if sentence_found:
                break

        # Recalculate missing_required and progress
        missing_required = [sec_id for sec_id, info in coverage.items() if info.get('status') == 'missing' and info.get('required')]
        covered_sections = sum(1 for info in coverage.values() if info.get('status') in {'covered', 'weak'})
        progress = int(100 * covered_sections / len(coverage)) if coverage else 0

        job['coverage'] = coverage
        job['missing_required'] = missing_required
        job['progress'] = progress
        pipeline.JOB_QUEUE[job_id] = job  # persist the correction

    # The document itself changes too: the sentence moves to its corrected
    # section as a new version (it used to stay where it was, also appear in
    # the new section, and never reach the PDF).
    version = document_model.current_version(job)
    edits = (document_model.move_edits(job.get("knowledge_object") or {}, moved_text, corrected_section)
             if moved_text else [])
    if edits:
        try:
            saved = await run_in_threadpool(document_model.save_edits, pipeline, job_id, edits, version,
                                            principal.actor)
        except document_model.EditError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc))
        version = saved["version"]
        audit(principal.tenant_id, principal.key_label, "kt_edited",
              {"job_id": job_id, "version": version, "summary": saved["summary"]})

    return JSONResponse({
        'status': 'ok',
        'message': 'feedback applied',
        'feedback': fb,
        'coverage': coverage,
        'missing_required': missing_required,
        'progress': progress,
        'document_version': version,
    })
