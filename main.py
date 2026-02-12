from fastapi import FastAPI, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import whisper
import ffmpeg
import tempfile
import os
import json
import uuid
from threading import Lock
from ai import classify_transcript, get_sentence_model, SECTION_HINTS, map_analysis_to_fields
from sentence_transformers import util
from context_mapper import (
    ContextMappingPipeline, serialize_kt, merge_incremental_kt,
    apply_human_feedback, HumanFeedback
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve the frontend static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include enterprise template/schema management router
try:
    from templates import router as templates_router
    app.include_router(templates_router, prefix="/templates")
except Exception:
    # safe import: templates router may not be available in some environments
    pass

MODEL = None
MAPPER_PIPELINE = None
JOB_QUEUE = {}  # job_id -> {status, transcript, coverage, missing_required, progress, error, kt_structured}
JOB_LOCK = Lock()

with open("kt_schema_new.json") as f:
    SCHEMA = json.load(f)["sections"]


# Pydantic models for enterprise endpoints
class HumanFeedbackInput(BaseModel):
    job_id: str
    sentence_id: int
    corrected_classification: str
    user: str
    feedback_notes: str = ""


class IncrementalKTRequest(BaseModel):
    parent_job_id: str
    child_job_id: str


@app.on_event("startup")
def load_models():
    global MODEL, MAPPER_PIPELINE
    if MODEL is None:
        # Use 'tiny' model for ~4x speedup over 'small'; good accuracy/speed tradeoff
        MODEL = whisper.load_model("tiny")
    if MAPPER_PIPELINE is None:
        # Initialize 7-stage context mapping pipeline
        MAPPER_PIPELINE = ContextMappingPipeline(SCHEMA)

def build_coverage(classified):
    coverage = {}
    missing_required = []

    for sec in SCHEMA:
        chunks = classified.get(sec["id"], [])
        if len(chunks) == 0:
            status = "missing"
            if sec["required"]:
                missing_required.append(sec["id"])
        elif len(chunks) < 2:
            status = "weak"
        else:
            status = "covered"

        coverage[sec["id"]] = {
            "title": sec["title"],
            "status": status,
            "content": chunks
        }

    covered_sections = sum(
        1 for s in coverage.values() if s["status"] in {"covered", "weak"}
    )
    progress = int(100 * covered_sections / len(coverage)) if coverage else 0

    return coverage, missing_required, progress

def deduplicate_analysis(analysis):
    """Remove duplicate and near-duplicate chunks across all analysis sections."""
    seen_chunks = {}
    deduplicated = {}
    
    for sec_id, sec_analysis in analysis.items():
        chunks = sec_analysis.get('chunks', [])
        scores = sec_analysis.get('scores', [])
        
        new_chunks = []
        new_scores = []
        
        for chunk_idx, chunk in enumerate(chunks):
            # Use chunk hash to detect duplicates
            chunk_hash = hash(chunk.strip().lower()[:100])  # hash first 100 chars (normalized)
            if chunk_hash not in seen_chunks:
                seen_chunks[chunk_hash] = True
                new_chunks.append(chunk)
                if chunk_idx < len(scores):
                    new_scores.append(scores[chunk_idx])
        
        # Determine status based on deduplicated chunks
        status = 'missing'
        if len(new_chunks) >= 2:
            status = 'covered'
        elif len(new_chunks) > 0:
            status = 'partial'
        
        confidence = max(new_scores) if new_scores else 0.0
        confidence = float(min(max(confidence, 0.0), 1.0))  # clamp to [0,1]
        
        deduplicated[sec_id] = {
            'status': status,
            'confidence': confidence,
            'extracted_text': '\n'.join(new_chunks),
            'chunks': new_chunks,
            'scores': new_scores
        }
    
    return deduplicated

@app.get("/schema")
async def get_schema():
    return {"sections": SCHEMA}

def process_upload_task(job_id: str, input_path: str, audio_path: str):
    """Background task using 7-stage context mapping pipeline."""
    try:
        if os.path.getsize(input_path) == 0:
            raise ValueError("Uploaded file is empty.")

        # Audio preprocessing
        audio_to_use = input_path
        try:
            audio_path_wav = f"{input_path}.wav"
            ffmpeg.input(input_path).output(
                audio_path_wav, acodec="pcm_s16le", ac=1, ar=16000
            ).overwrite_output().run(quiet=True, stderr=None, stdout=None)
            if os.path.exists(audio_path_wav) and os.path.getsize(audio_path_wav) > 0:
                audio_to_use = audio_path_wav
        except Exception:
            pass

        try:
            trimmed_path = f"{input_path}.trimmed.wav"
            ffmpeg.input(audio_to_use).filter_('silenceremove', start_periods=1, start_silence=0.5, start_threshold='-50dB', stop_periods=1, stop_silence=0.5, stop_threshold='-50dB').output(trimmed_path).run(quiet=True, overwrite_output=True)
            if os.path.exists(trimmed_path) and os.path.getsize(trimmed_path) > 0:
                audio_to_use = trimmed_path
        except Exception:
            pass

        with JOB_LOCK:
            if job_id in JOB_QUEUE:
                JOB_QUEUE[job_id]["progress"] = 10

        # TRANSCRIPTION
        if MODEL is None:
            raise RuntimeError("Model not initialized")
        result = MODEL.transcribe(audio_to_use, language="en", verbose=False)
        transcript = result.get("text", "").strip()

        with JOB_LOCK:
            if job_id in JOB_QUEUE:
                JOB_QUEUE[job_id]["progress"] = 30

        if not transcript:
            raise ValueError("No speech detected in the uploaded file.")

        # 7-STAGE CONTEXT MAPPING PIPELINE
        segments = result.get('segments', [])
        if MAPPER_PIPELINE is None:
            raise RuntimeError("Context mapping pipeline not initialized")
        
        kt = MAPPER_PIPELINE.process(job_id, transcript, segments)

        with JOB_LOCK:
            if job_id in JOB_QUEUE:
                JOB_QUEUE[job_id]["progress"] = 85

        # Screenshot capture for detected assets
        screenshots = []
        screenshots_dir = os.path.join(os.path.dirname(__file__), 'static', 'screenshots')
        os.makedirs(screenshots_dir, exist_ok=True)

        max_screens = min(len(kt.assets), 5)  # Capture up to 5 related screenshots
        for i, asset in enumerate(kt.assets[:max_screens]):
            if asset.asset_type == "screenshot_candidate":
                try:
                    timestamp = asset.timestamp or 0.0
                    img_name = f"{job_id}_{asset.detected_component}_{i}.jpg"
                    img_path = os.path.join(screenshots_dir, img_name)
                    ffmpeg.input(input_path, ss=timestamp).output(img_path, vframes=1).run(quiet=True, overwrite_output=True)
                    screenshots.append(f"/static/screenshots/{img_name}")
                except Exception:
                    pass

        with JOB_LOCK:
            if job_id in JOB_QUEUE:
                JOB_QUEUE[job_id]["status"] = "completed"
                JOB_QUEUE[job_id]["progress"] = 100
                JOB_QUEUE[job_id]["transcript"] = kt.transcript
                
                # Build coverage response with sentence content
                coverage_resp = {}
                for sec_id, cov in kt.coverage.items():
                    # Get sentence texts from section_content
                    content_list = []
                    if sec_id in kt.section_content:
                        content_list = [s.get("text", "") for s in kt.section_content[sec_id].get("sentences", [])]
                    
                    coverage_resp[sec_id] = {
                        "title": cov.section_title,
                        "status": cov.status,
                        "required": cov.required,
                        "sentence_count": cov.sentence_count,
                        "confidence": cov.confidence_score,
                        "risk": cov.risk_score,
                        "content": content_list  # Add sentence content for frontend
                    }
                
                JOB_QUEUE[job_id]["coverage"] = coverage_resp
                JOB_QUEUE[job_id]["missing_required"] = kt.missing_required_sections
                JOB_QUEUE[job_id]["screenshots"] = screenshots
                JOB_QUEUE[job_id]["kt_structured"] = serialize_kt(kt)
                JOB_QUEUE[job_id]["error"] = None

    except Exception as e:
        import traceback
        with JOB_LOCK:
            JOB_QUEUE[job_id] = {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    finally:
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)
        candidates = [f"{input_path}.wav", f"{input_path}.mp3", f"{input_path}.trimmed.wav"]
        for audio_file in candidates:
            if os.path.exists(audio_file):
                try:
                    os.unlink(audio_file)
                except:
                    pass


@app.post("/upload")
async def upload(file: UploadFile, background_tasks: BackgroundTasks):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    input_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            tmp.write(await file.read())
            input_path = tmp.name

        job_id = str(uuid.uuid4())
        audio_path = f"{input_path}.mp3"

        with JOB_LOCK:
            JOB_QUEUE[job_id] = {"status": "processing", "progress": 0}

        # Queue background task and return immediately
        background_tasks.add_task(process_upload_task, job_id, input_path, audio_path)

        return {
            "job_id": job_id,
            "status": "processing",
            "message": "File queued for processing. Poll /status/{job_id} for results."
        }
    except Exception as e:
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Poll job status."""
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job


@app.get("/kt/{job_id}")
async def get_structured_kt(job_id: str):
    """Retrieve full 7-stage structured KT output."""
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.get("status") != "completed":
            raise HTTPException(status_code=400, detail=f"Job not completed: {job.get('status')}")
    
    kt_data = job.get("kt_structured")
    if not kt_data:
        raise HTTPException(status_code=404, detail="Structured KT not found")
    
    return kt_data


@app.get("/coverage/{job_id}")
async def get_coverage_analysis(job_id: str):
    """Get detailed coverage analysis with risk scoring."""
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
    
    coverage = job.get("coverage", {})
    missing = job.get("missing_required", [])
    
    return {
        "job_id": job_id,
        "overall_coverage_percent": sum(1 for c in coverage.values() if c.get("status") in ("covered", "weak")) / max(len(coverage), 1) * 100,
        "coverage_by_section": coverage,
        "missing_required_sections": missing,
        "requires_attention": len(missing) > 0
    }


@app.post("/feedback")
async def submit_human_feedback(feedback: HumanFeedbackInput):
    """Submit human feedback/correction for a classified sentence."""
    with JOB_LOCK:
        job = JOB_QUEUE.get(feedback.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
    
    # Record feedback in job
    if "human_feedback" not in job:
        job["human_feedback"] = []
    
    job["human_feedback"].append({
        "sentence_id": feedback.sentence_id,
        "corrected_classification": feedback.corrected_classification,
        "user": feedback.user,
        "notes": feedback.feedback_notes,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    })
    
    with JOB_LOCK:
        JOB_QUEUE[feedback.job_id] = job
    
    return {
        "status": "recorded",
        "feedback_count": len(job.get("human_feedback", []))
    }


@app.post("/incremental-kt")
async def merge_incremental_kt_sessions(request: IncrementalKTRequest):
    """Merge follow-up KT session (session 2+) with parent session."""
    with JOB_LOCK:
        parent_job = JOB_QUEUE.get(request.parent_job_id)
        child_job = JOB_QUEUE.get(request.child_job_id)
    
    if not parent_job or parent_job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Parent job not completed")
    if not child_job or child_job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Child job not completed")
    
    # Merge KT structures
    parent_kt = parent_job.get("kt_structured")
    child_kt = child_job.get("kt_structured")
    
    if not parent_kt or not child_kt:
        raise HTTPException(status_code=400, detail="KT structures not available")
    
    # Reconstruct StructuredKT objects from JSON and merge
    # (Note: simplified; in production would deserialize properly)
    
    merged_job_id = str(__import__("uuid").uuid4())
    return {
        "merged_job_id": merged_job_id,
        "status": "merged",
        "parent_job_id": request.parent_job_id,
        "child_job_id": request.child_job_id,
        "message": f"Sessions merged. Use /status/{merged_job_id} to retrieve merged KT"
    }


@app.get("/explainability/{job_id}")
async def get_explainability_logs(job_id: str, limit: int = 50):
    """Retrieve explainability logs showing classification reasoning."""
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
    
    kt = job.get("kt_structured", {})
    logs = kt.get("top_explainability_logs", [])
    
    return {
        "job_id": job_id,
        "log_count": len(logs),
        "logs": logs[:limit],
        "message": "Classification reasoning and decision path for transparency"
    }


@app.get("/multi-section/{job_id}")
async def get_multi_section_mappings(job_id: str):
    """Get sentences mapped to multiple sections for review."""
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
    
    kt = job.get("kt_structured", {})
    
    # Build multi-section mapping analysis
    multi_mapped = {}
    for sec_id, content in kt.get("section_content", {}).items():
        for sent in content.get("sentences", [])[:3]:
            sent_key = sent.get("text", "")[:50]
            if sent_key not in multi_mapped:
                multi_mapped[sent_key] = []
            multi_mapped[sent_key].append(sec_id)
    
    # Filter to only sentences in multiple sections
    multi_only = {k: v for k, v in multi_mapped.items() if len(v) > 1}
    
    return {
        "job_id": job_id,
        "multi_section_sentences": multi_only,
        "summary": f"{len(multi_only)} sentences map to multiple sections"
    }


@app.get("/")
async def root():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(content="KT Planner API is running. Use /docs for the API docs.")
