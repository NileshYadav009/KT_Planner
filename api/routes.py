"""HTTP endpoints for the KT Planner API.

Moved out of main.py as part of the Phase 3 architecture split (see
REPOSITORY_AUDIT.md) — logic unchanged, just relocated onto an APIRouter.
Job state is read via `pipeline.JOB_QUEUE`/`pipeline.JOB_LOCK` (module-attribute
access, not `from pipeline import ...`) so writes made by
`pipeline.process_upload_task` in the background task are always visible here.
"""

import os
import re
import tempfile
import uuid
from datetime import datetime

from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, BackgroundTasks, Depends, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from ai import generate_report
from auth import (
    SESSION_COOKIE, SESSION_TTL_SECONDS, Principal, create_session, end_session, ensure_job_access,
    optional_principal, principal_for_key, require_principal, sign_asset_urls, tenant_name, verify_asset_signature,
)
from devops_transcription import clean_transcript
from input_gate import assess_transcript
from kt_schema_loader import SCHEMA
import pdf_rendering
import pipeline

router = APIRouter()


@router.get("/healthz")
async def healthz():
    """Liveness and readiness: ready once the models are loaded."""
    ready = pipeline.MAPPER_PIPELINE is not None and pipeline.MODEL is not None
    return JSONResponse({"status": "ok" if ready else "starting", "ready": ready}, status_code=200 if ready else 503)


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
async def export_pdf(job_id: str, principal: Principal = Depends(require_principal)):
    with pipeline.JOB_LOCK:
        job = ensure_job_access(pipeline.JOB_QUEUE.get(job_id), principal)
    if job.get("status") not in pipeline.COMPLETED_STATUSES:
        raise HTTPException(status_code=404, detail="Job not found or not completed")

    knowledge_object = job.get("knowledge_object", {}) or {}
    title = knowledge_object.get("system_name") or job.get("title") or "KT Document"
    # The date the KT was created, not the date this copy was downloaded:
    # an older KT reopened from the Past KTs list keeps its own date.
    created = pipeline.JOB_STORE.created_at(job_id)
    date_str = (datetime.fromtimestamp(created) if created else datetime.now()).strftime("%d %B %Y")
    rendered_sections = knowledge_object.get("rendered_sections")

    if not isinstance(rendered_sections, list) or not rendered_sections:
        coverage = job.get("coverage", {}) or {}
        rendered_sections = []
        for sec_id, sec_info in coverage.items():
            rendered_sections.append({
                "section_id": sec_id,
                "section_title": sec_info.get("title", sec_id),
                "blocks": [{
                    "type": "NarrativeBlock",
                    "title": sec_info.get("title", sec_id),
                    # Plain text: the renderer escapes and formats it.
                    "paragraphs": [item for item in sec_info.get("content", []) if isinstance(item, str)],
                }]
            })

    html_doc = pdf_rendering.render_pdf_html(
        title=title,
        job_id=job_id,
        rendered_sections=rendered_sections,
        coverage=job.get("coverage", {}),
        date_str=date_str,
        warnings=job.get("warnings"),
        notices=job.get("notices"),
    )
    try:
        import weasyprint  # noqa: F401
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="WeasyPrint is not available. Install required native dependencies and Python packages.",
        ) from exc

    # Restricted fetcher (no file:// or network access), and off the event
    # loop so one render does not stall every other request.
    pdf_bytes = await run_in_threadpool(pdf_rendering.html_to_pdf_bytes, html_doc)
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


@router.post("/upload")
async def upload(file: UploadFile, background_tasks: BackgroundTasks,
                 principal: Principal = Depends(require_principal)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    input_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            tmp.write(await file.read())
            input_path = tmp.name

        job_id = str(uuid.uuid4())
        audio_path = f"{input_path}.mp3"

        with pipeline.JOB_LOCK:
            pipeline.JOB_QUEUE[job_id] = {"status": "processing", "progress": 0, "tenant_id": principal.tenant_id}

        # Queue background task and return immediately
        background_tasks.add_task(pipeline.process_upload_task, job_id, input_path, audio_path)

        return {
            "job_id": job_id,
            "status": "processing",
            "message": "File queued for processing. Poll /status/{job_id} for results."
        }
    except Exception as e:
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/kt-from-transcript")
async def kt_from_transcript(payload: dict, background_tasks: BackgroundTasks,
                             principal: Principal = Depends(require_principal)):
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
    with pipeline.JOB_LOCK:
        pipeline.JOB_QUEUE[job_id] = {"status": "processing", "progress": 0, "tenant_id": principal.tenant_id}

    background_tasks.add_task(pipeline.run_kt_pipeline, job_id, cleaned_transcript,
                              warnings=gate["reasons"] if gate["verdict"] != "ok" else [])

    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Transcript queued for processing. Poll /status/{job_id} for results."
    }


@router.get("/jobs")
async def list_jobs(limit: int = 50, principal: Principal = Depends(require_principal)):
    """The caller's KT jobs, newest first (persisted across restarts)."""
    return {"jobs": pipeline.JOB_STORE.list(principal.tenant_id, limit=max(1, min(limit, 200)))}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, principal: Principal = Depends(require_principal)):
    """Delete a KT: its record, transcript, document and screenshots."""
    with pipeline.JOB_LOCK:
        ensure_job_access(pipeline.JOB_QUEUE.get(job_id), principal)
        pipeline.delete_job_data(job_id)
    return {"status": "deleted", "job_id": job_id}


@router.get("/status/{job_id}")
async def get_status(job_id: str, principal: Principal = Depends(require_principal)):
    """Poll job status."""
    with pipeline.JOB_LOCK:
        job = ensure_job_access(pipeline.JOB_QUEUE.get(job_id), principal)
        return sign_asset_urls(dict(job, job_id=job_id))


@router.post("/semantic-placement")
async def semantic_placement(payload: dict, principal: Principal = Depends(require_principal)):
    """Classify raw transcript text into KT sections and return quality metrics."""
    transcript = payload.get("transcript", "")
    if not isinstance(transcript, str) or not transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is required and must be non-empty.")

    similarity_threshold = payload.get("similarity_threshold", 0.20)
    try:
        similarity_threshold = float(similarity_threshold)
    except (TypeError, ValueError):
        similarity_threshold = 0.20

    try:
        report = generate_report(transcript, similarity_threshold=similarity_threshold)
        assignment_counts = {sid: len(report["analysis"][sid].get("chunks", [])) for sid in report["analysis"]}
        total_assigned = sum(assignment_counts.values())
        return {
            "status": "success",
            "transcript": transcript,
            "metrics": {
                "total_sentences": total_assigned,
                "assigned_sentences": total_assigned,
                "unclassified_sentences": 0,
                "duplicate_rate": 0.0,
                "avg_confidence": float(report["summary"]["confidence_score"]),
                "clauses_split": 0,
            },
            "assignments": {sid: report["analysis"][sid].get("chunks", []) for sid in report["analysis"]},
            "paragraphs": report["paragraphs"],
            "summary": report["summary"],
            "explainability": report["explainability"],
            "risk_warning": report["risk_warning"],
            "recommended_state": report["recommended_state"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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


@router.post("/login")
async def login(payload: dict, request: Request):
    """Exchange an API key for an HttpOnly session cookie (browser sign-in)."""
    principal = principal_for_key(str(payload.get("api_key") or "").strip())
    if principal is None:
        raise HTTPException(status_code=401, detail="That API key was not accepted.")
    token = create_session(principal)
    resp = JSONResponse({"status": "signed_in", "workspace": tenant_name(principal.tenant_id)})
    secure = request.url.scheme == "https" or os.getenv("CONTINUUM_SECURE_COOKIES") == "1"
    resp.set_cookie(SESSION_COOKIE, token, max_age=SESSION_TTL_SECONDS, httponly=True, samesite="lax",
                    secure=secure, path="/")
    return resp


@router.post("/logout")
async def logout(request: Request):
    end_session(request.cookies.get(SESSION_COOKIE))
    resp = JSONResponse({"status": "signed_out"})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@router.get("/me")
async def me(principal: Principal = Depends(require_principal)):
    """Who the caller is signed in as (workspace shown in the UI header)."""
    return {"tenant_id": principal.tenant_id, "workspace": tenant_name(principal.tenant_id),
            "key_label": principal.key_label, "role": principal.role,
            "auth": "disabled" if principal.tenant_id == "local" and principal.key_label == "auth-disabled" else "on"}


@router.post('/feedback')
async def receive_feedback(payload: dict, principal: Principal = Depends(require_principal)):
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

    return JSONResponse({
        'status': 'ok',
        'message': 'feedback applied',
        'feedback': fb,
        'coverage': coverage,
        'missing_required': missing_required,
        'progress': progress
    })
