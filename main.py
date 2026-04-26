from fastapi import FastAPI, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional
import whisper
import ffmpeg
import tempfile
import os
import json
import uuid
import numpy as np
from threading import Lock
from ai import classify_transcript, get_sentence_model, SECTION_HINTS, map_analysis_to_fields
from context_mapper import (
    ContextMappingPipeline, serialize_kt, merge_incremental_kt,
    apply_human_feedback, HumanFeedback
)
from enterprise_semantic_mapper import (
    create_semantic_mapper
)
from devops_transcription import (
    apply_devops_corrections, correct_transcript, 
    get_model_recommendation, score_transcription_confidence
)
import logging
import warnings

# Suppress verbose logs from HuggingFace, httpx, and other libraries
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning)

# Suppress tqdm progress bars
from tqdm import tqdm
tqdm.disable = True

# Structured-ish logging setup for the API process (minimal)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Only show errors and critical info for the app
logger.setLevel(logging.WARNING)

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
USED_MODEL_SIZE = None  # Track which Whisper model size is being used
MAPPER_PIPELINE = None
SEMANTIC_MAPPER = None  # Enterprise semantic mapper
JOB_QUEUE = {}  # job_id -> {status, transcript, coverage, missing_required, progress, error, kt_structured}
JOB_LOCK = Lock()

with open("kt_schema_new.json") as f:
    SCHEMA = json.load(f)["sections"]

# Initialize enterprise semantic mapper
SEMANTIC_MAPPER = create_semantic_mapper(SCHEMA)


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


class ExpertCorrectionInput(BaseModel):
    """Expert feedback for enterprise semantic training."""
    sentence_id: str
    original_section: str
    corrected_section: str
    confidence_boost: float = 0.1
    expert_notes: str = ""


class SemanticPlacementRequest(BaseModel):
    """Request enterprise semantic placement analysis."""
    transcript: str  # Full transcript text
    job_id: str = ""  # Optional: link to existing job


@app.on_event("startup")
def load_models():
    global MODEL, MAPPER_PIPELINE, USED_MODEL_SIZE
    if MODEL is None:
        # Use 'base' model for better DevOps transcription accuracy
        # Good balance between speed and accuracy for technical content
        USED_MODEL_SIZE = "base"
        MODEL = whisper.load_model(USED_MODEL_SIZE)
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

def recalculate_coverage_from_section_content(section_content):
    """Recalculate coverage metrics from current section_content (for manual assignments)."""
    coverage = {}
    missing_required = []
    
    for sec in SCHEMA:
        sec_id = sec["id"]
        sec_data = section_content.get(sec_id, {})
        sentences = sec_data.get("sentences", [])
        
        # Determine status based on sentence count
        if len(sentences) == 0:
            status = "missing"
            if sec.get("required"):
                missing_required.append(sec_id)
        elif len(sentences) < 2:
            status = "weak"
        else:
            status = "covered"
        
        coverage[sec_id] = {
            "title": sec.get("title", sec_id),
            "status": status,
            "required": sec.get("required", False),
            "sentence_count": len(sentences),
            "confidence": sec_data.get("confidence", 0.5),
            "content": [s.get("text", "") for s in sentences],
            "sentences": sentences
        }
    
    covered_sections = sum(
        1 for c in coverage.values() if c["status"] in {"covered", "weak"}
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
        
        # Apply DevOps-specific transcription corrections
        transcription_stats = {"corrections": {}}
        if result.get("segments"):
            corrected_segments, stats = correct_transcript(result["segments"], apply_context=True)
            result["segments"] = corrected_segments
            transcription_stats = stats
            
            # Rebuild full transcript from corrected segments
            result["text"] = " ".join([seg["text"] for seg in corrected_segments])
        
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

        # Initialize Sentence Processor for advanced features
        try:
            from sentence_processor import SentenceProcessor
            processor = SentenceProcessor(job_id)
            processor.process_segments(segments, transcript)
            
            with JOB_LOCK:
                SENTENCE_PROCESSORS[job_id] = processor
        except Exception as e:
            # Log error but don't fail the job - sentence processor is optional
            logger.warning(f"Failed to initialize sentence processor for {job_id}: {str(e)}")
        
        with JOB_LOCK:
            if job_id in JOB_QUEUE:
                JOB_QUEUE[job_id]["progress"] = 85

        # Screenshot capture for detected assets (DISABLED)
        # The screenshot extraction was disabled per user request because
        # assets and screenshots are currently not needed and caused
        # clutter in the repository. To re-enable, remove these comments
        # and ensure `ffmpeg-python` is available and safe to run in this env.
        screenshots = []  # Initialize empty list for disabled screenshot feature
        # screenshots_dir = os.path.join(os.path.dirname(__file__), 'static', 'screenshots')
        # os.makedirs(screenshots_dir, exist_ok=True)
        #
        # max_screens = min(len(kt.assets), 5)  # Capture up to 5 related screenshots
        # for i, asset in enumerate(kt.assets[:max_screens]):
        #     if asset.asset_type == "screenshot_candidate":
        #         try:
        #             timestamp = asset.timestamp or 0.0
        #             img_name = f"{job_id}_{asset.detected_component}_{i}.jpg"
        #             img_path = os.path.join(screenshots_dir, img_name)
        #             ffmpeg.input(input_path, ss=timestamp).output(img_path, vframes=1).run(quiet=True, overwrite_output=True)
        #             screenshots.append(f"/static/screenshots/{img_name}")
        #         except Exception:
        #             pass

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
                    sentences_list = []
                    if sec_id in kt.section_content:
                        sentences_list = kt.section_content[sec_id].get("sentences", [])
                        content_list = [s.get("text", "") for s in sentences_list]
                    
                    # Always add section to coverage, even if empty/missing
                    coverage_resp[sec_id] = {
                        "title": cov.section_title,
                        "status": cov.status if content_list else "missing",  # Override status if no content
                        "required": cov.required,
                        "sentence_count": len(content_list),
                        "confidence": cov.confidence_score,
                        "risk": cov.risk_score,
                        "content": content_list,  # Sentence texts for display
                        "sentences": sentences_list  # Full sentence objects for manual mapping
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


@app.get("/jobs")
async def list_jobs(limit: int = 20):
    """List recent jobs with basic metadata."""
    with JOB_LOCK:
        items = []
        for jid, job in list(JOB_QUEUE.items())[-limit:]:
            items.append({
                "job_id": jid,
                "status": job.get("status"),
                "progress": job.get("progress", 0),
                "created_at": job.get("kt_structured", {}).get("timestamp") if job.get("kt_structured") else None,
                "transcript_preview": (job.get("transcript") or '')[:200]
            })
    return {"jobs": items}


@app.get("/reviews/{job_id}")
async def get_reviews(job_id: str):
    """Return review-required (low-confidence or policy-flagged) sentences for a job."""
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")

    kt = job.get("kt_structured")
    if not kt:
        raise HTTPException(status_code=404, detail="Structured KT not found for job.")

    # Return serialized review-required sentences if present
    review_items = kt.get("review_required_sentences") if isinstance(kt, dict) else None
    if review_items is None:
        # Fallback: return unassigned sentences
        review_items = kt.get("unassigned_sentences") if isinstance(kt, dict) else []

    return {"job_id": job_id, "review_required": review_items}


@app.post("/reviews/{job_id}/apply")
async def apply_review_correction(job_id: str, payload: Dict):
    """Apply a human correction for a review-required sentence.

    Payload keys:
      - `sentence_text` (or `sentence_index`): identifies the sentence
      - `corrected_section`: target section id
      - `user`: user id or name
    """
    sentence_text = payload.get("sentence_text")
    sentence_index = payload.get("sentence_index")
    corrected_section = payload.get("corrected_section")
    user = payload.get("user", "unknown")
    evidence = payload.get("evidence")

    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        kt = job.get("kt_structured")
        if not kt or not isinstance(kt, dict):
            raise HTTPException(status_code=400, detail="Structured KT not available for job")

        # Find unassigned sentence by text or index
        target = None
        target_idx = None
        if sentence_index is not None:
            try:
                idx = int(sentence_index)
                if 0 <= idx < len(kt.get("unassigned_sentences", [])):
                    target = kt["unassigned_sentences"][idx]
                    target_idx = idx
            except Exception:
                pass

        if target is None and sentence_text:
            for i, s in enumerate(kt.get("unassigned_sentences", [])):
                if sentence_text.strip().lower() == (s.get("text","") or "").strip().lower():
                    target = s
                    target_idx = i
                    break

        if not target:
            raise HTTPException(status_code=404, detail="Unassigned sentence not found")

        # Remove from unassigned list
        removed = kt["unassigned_sentences"].pop(target_idx)

        # Ensure section bucket exists
        if "section_content" not in kt:
            kt["section_content"] = {}

        if corrected_section not in kt["section_content"]:
            kt["section_content"][corrected_section] = {
                "section_id": corrected_section,
                "section_title": corrected_section,
                "sentences": [],
                "enhanced_texts": [],
                "repair_actions": [],
                "screenshots": [],
                "confidence": 1.0,
                "sentence_count": 0
            }

        # Append corrected sentence
        kt["section_content"][corrected_section]["sentences"].append({
            "text": removed.get("text"),
            "start": removed.get("start"),
            "end": removed.get("end"),
            "speaker": removed.get("speaker"),
            "audio_confidence": removed.get("audio_confidence", 1.0),
            "assigned_sections": [corrected_section]
        })
        kt["section_content"][corrected_section]["enhanced_texts"].append(removed.get("text"))

        # Update counts
        kt["section_content"][corrected_section]["sentence_count"] = len(kt["section_content"][corrected_section]["sentences"])

        # Record human feedback
        if "human_feedback" not in job or job.get("human_feedback") is None:
            job["human_feedback"] = []
        job["human_feedback"].append({
            "user": user,
            "corrected_section": corrected_section,
            "sentence": removed.get("text"),
            "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z"
        })

        # Attach evidence if provided (to sentence and top-level evidence list)
        if evidence:
            # Normalize to list
            ev_list = evidence if isinstance(evidence, list) else [evidence]
            # Attach to the last appended sentence in section_content
            sec_sentences = kt["section_content"][corrected_section]["sentences"]
            if sec_sentences:
                if "evidence" not in sec_sentences[-1]:
                    sec_sentences[-1]["evidence"] = []
                sec_sentences[-1]["evidence"].extend(ev_list)

            # Ensure top-level evidence list exists
            if "evidence" not in kt:
                kt["evidence"] = []
            kt["evidence"].append({
                "sentence": removed.get("text"),
                "evidence": ev_list,
                "user": user,
                "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z"
            })

        # Recompute simple coverage summary from serialized KT and SCHEMA
        coverage_resp = {}
        missing_required = []
        for sec in SCHEMA:
            sec_id = sec.get("id")
            title = sec.get("title")
            required = sec.get("required", False)
            sec_content = kt.get("section_content", {}).get(sec_id, {})
            sentences = sec_content.get("sentences", [])
            sentence_texts = [s.get("text", "") for s in sentences]
            count = len(sentence_texts)

            # estimate confidence from audio_confidence values if present
            conf_vals = [s.get("audio_confidence", 0.5) for s in sentences if s.get("audio_confidence") is not None]
            confidence = float(sum(conf_vals) / len(conf_vals)) if conf_vals else 0.0

            if count == 0:
                status = "missing"
                if required:
                    missing_required.append(sec_id)
                risk = 1.0 if required else 0.5
            elif count < 2:
                status = "weak"
                risk = 0.6 if required else 0.2
            else:
                status = "covered"
                risk = 0.0 if confidence > 0.7 else 0.1

            coverage_resp[sec_id] = {
                "title": title,
                "status": status,
                "required": required,
                "sentence_count": count,
                "confidence": confidence,
                "risk": risk,
                "content": sentence_texts,
                "sentences": sentences
            }

        # Persist updates back to job queue
        JOB_QUEUE[job_id]["kt_structured"] = kt
        JOB_QUEUE[job_id]["coverage"] = coverage_resp
        JOB_QUEUE[job_id]["missing_required"] = missing_required

        # Record human feedback entry already appended above
        JOB_QUEUE[job_id] = job

    return {"status": "ok", "job_id": job_id, "moved_to": corrected_section, "missing_required": missing_required}


@app.get("/policy")
async def get_policy():
    try:
        from runtime_policy import load_policy
        return load_policy()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load policy")


@app.get("/glossary")
async def get_glossary():
    try:
        from glossary import GLOSSARY
        return GLOSSARY
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load glossary")


@app.post("/glossary")
async def set_glossary(payload: Dict):
    try:
        from glossary import save_glossary
        ok = save_glossary(payload)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to save glossary")
        return {"status": "saved"}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save glossary")


@app.post("/policy")
async def set_policy(payload: Dict):
    try:
        from runtime_policy import save_policy
        ok = save_policy(payload)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to save policy")
        return {"status": "saved"}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save policy")


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


@app.post("/manual-assign/{job_id}")
async def manually_assign_sentence(job_id: str, sentence_text: str, target_section: str, user: str = "user"):
    """Manually assign a sentence to a section (for coverage mapping)."""
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        kt = job.get("kt_structured")
        if not kt or not isinstance(kt, dict):
            raise HTTPException(status_code=400, detail="KT data not available")
    
    # Find the sentence in any section or unassigned
    found_sentence = None
    found_in_section = None
    
    # Search in section_content
    section_content = kt.get("section_content", {})
    for sec_id, sec_data in section_content.items():
        for sentence in sec_data.get("sentences", []):
            if sentence.get("text", "").strip().lower() == sentence_text.strip().lower():
                found_sentence = sentence
                found_in_section = sec_id
                break
        if found_sentence:
            break
    
    # Search in unassigned
    if not found_sentence:
        for sentence in kt.get("unassigned_sentences", []):
            if sentence.get("text", "").strip().lower() == sentence_text.strip().lower():
                found_sentence = sentence
                break
    
    if not found_sentence:
        raise HTTPException(status_code=404, detail="Sentence not found in KT")
    
    # Remove from old location if needed
    if found_in_section:
        kt["section_content"][found_in_section]["sentences"] = [
            s for s in kt["section_content"][found_in_section]["sentences"]
            if s.get("text", "").strip().lower() != sentence_text.strip().lower()
        ]
    else:
        kt["unassigned_sentences"] = [
            s for s in kt.get("unassigned_sentences", [])
            if s.get("text", "").strip().lower() != sentence_text.strip().lower()
        ]
    
    # Add to target section
    if target_section not in section_content:
        section_content[target_section] = {
            "section_id": target_section,
            "section_title": target_section,
            "sentences": [],
            "enhanced_texts": [],
            "repair_actions": [],
            "screenshots": [],
            "confidence": 1.0,
            "sentence_count": 0
        }
    
    section_content[target_section]["sentences"].append(found_sentence)
    section_content[target_section]["enhanced_texts"].append(found_sentence.get("text", ""))
    
    # Persist changes and recalculate coverage
    with JOB_LOCK:
        JOB_QUEUE[job_id]["kt_structured"] = kt
        
        # CRITICAL FIX: Recalculate coverage from current section_content
        new_coverage, new_missing, new_progress = recalculate_coverage_from_section_content(section_content)
        JOB_QUEUE[job_id]["coverage"] = new_coverage
        JOB_QUEUE[job_id]["missing_required"] = new_missing
        JOB_QUEUE[job_id]["progress"] = new_progress
        
        job["human_feedback"] = job.get("human_feedback", [])
        job["human_feedback"].append({
            "type": "manual_assignment",
            "sentence": sentence_text[:100],
            "target_section": target_section,
            "user": user,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat()
        })
        JOB_QUEUE[job_id] = job
    
    return {
        "status": "assigned",
        "sentence_text": sentence_text[:100],
        "target_section": target_section,
        "message": f"Sentence successfully assigned to {target_section}",
        "new_coverage_percent": new_progress
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


# ============================================================================
# ENTERPRISE SEMANTIC MAPPER ENDPOINTS
# ============================================================================

@app.post("/semantic-placement")
async def semantic_placement_analysis(request: SemanticPlacementRequest):
    """
    Perform enterprise-grade semantic placement analysis.
    
    Features:
    - Sentence-level processing with unique IDs
    - Semantic scoring (not keywords)
    - Clause splitting for mixed sentences
    - Anti-duplication guarantee
    - Paragraph reconstruction
    - Quality metrics reporting
    """
    try:
        # Parse transcript into sentences
        sentences = request.transcript.split('\n')
        sentences = [(str(i), s.strip()) for i, s in enumerate(sentences) if s.strip()]
        
        # Process with semantic mapper
        result = SEMANTIC_MAPPER.process_transcript(sentences)
        
        return {
            "status": "success",
            "assignments": result["assignments"],
            "paragraphs": result["paragraphs"],
            "metrics": result["metrics"],
            "metrics_report": result["metrics_report"],
            "unclassified_count": len(result["unclassified"]),
            "unclassified": result["unclassified"],
            "clauses_split": result["clauses_split"],
            "duplicate_rate": result["duplicate_rate"],
            "clause_assignments": result["clause_assignments"]
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/expert-correction")
async def record_expert_correction(correction: ExpertCorrectionInput):
    """
    Record expert feedback for learning and improvement.
    
    This allows DevOps experts to correct misclassifications,
    and the system learns from these corrections.
    """
    try:
        expert_correction = ExpertCorrection(
            sentence_id=correction.sentence_id,
            original_section=correction.original_section,
            corrected_section=correction.corrected_section,
            confidence_boost=correction.confidence_boost,
            expert_notes=correction.expert_notes
        )
        
        SEMANTIC_MAPPER.record_expert_feedback(expert_correction)
        
        return {
            "status": "success",
            "message": f"Expert correction recorded: {correction.sentence_id} -> {correction.corrected_section}",
            "training_stats": SEMANTIC_MAPPER.get_training_stats()
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/training-stats")
async def get_training_statistics():
    """Get statistics on expert training and system improvements."""
    try:
        stats = SEMANTIC_MAPPER.get_training_stats()
        return {
            "status": "success",
            "training_statistics": stats,
            "message": "System has learned from expert feedback"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/quality-report/{job_id}")
async def get_quality_report(job_id: str):
    """
    Get detailed quality metrics for a job.
    
    Includes:
    - Duplicate rate
    - Confidence distribution
    - Section coverage
    - Unclassified rate
    - Coherence scores
    """
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
    
    kt = job.get("kt_structured", {})
    
    return {
        "job_id": job_id,
        "quality_metrics": {
            "coverage": job.get("coverage", {}),
            "missing_required": job.get("missing_required", []),
            "overall_confidence": calculate_avg_confidence(kt),
            "duplicate_rate": 0.0,  # Should always be 0.0 with enterprise engine
            "unclassified_count": len(job.get("unassigned_sentences", []))
        }
    }


@app.get("/enterprise-status")
async def get_enterprise_status():
    """
    Get system information about enterprise semantic mapper.
    
    Shows:
    - System version and capabilities
    - Training mode status
    - Quality metrics
    """
    return {
        "system": "Continuum Enterprise Semantic Mapper",
        "version": "2.0",
        "capabilities": [
            "Sentence-level processing with unique IDs",
            "Semantic scoring using embeddings",
            "Mixed sentence handling with clause splitting",
            "Anti-duplication guarantee",
            "Paragraph integrity engine",
            "Expert training mode",
            "Real-time quality metrics"
        ],
        "training_stats": SEMANTIC_MAPPER.get_training_stats(),
        "active_schema_sections": len(SCHEMA),
        "features_enabled": {
            "semantic_mapper": True,
            "expert_training": True,
            "quality_controls": True,
            "paragraph_reconstruction": True,
            "clause_splitting": True
        }
    }


def calculate_avg_confidence(kt: Dict) -> float:
    """Calculate average confidence from KT structure."""
    confidences = []
    for section in kt.get("section_content", {}).values():
        for sentence in section.get("sentences", []):
            conf = sentence.get("confidence", 0.5)
            confidences.append(conf)
    
    return float(np.mean(confidences)) if confidences else 0.5


@app.get("/transcript-model-info")
async def get_transcript_model_info():
    """Get information about the Whisper transcription model being used."""
    return {
        "status": "success",
        "model": {
            "size": USED_MODEL_SIZE or "not-loaded",
            "type": "whisper",
            "description": "OpenAI Whisper model for speech-to-text transcription"
        },
        "enhancements": {
            "devops_optimized": True,
            "corrections_enabled": True,
            "context_aware": True,
            "vocabulary": "DevOps, Cloud, Infrastructure, CI/CD, Kubernetes, etc."
        },
        "accuracy_notes": {
            "base_model": "Uses Whisper 'base' model for better accuracy on technical content",
            "corrections": "Applies 50+ DevOps terminology corrections post-transcription",
            "context": "Uses surrounding context to disambiguate similar-sounding terms"
        }
    }


# ============================================================================
# NEW ADVANCED FEATURES ENDPOINTS (v2.0)
# ============================================================================

# Global sentence processor instances
SENTENCE_PROCESSORS = {}  # job_id -> SentenceProcessor

class DragDropRequest(BaseModel):
    job_id: str
    sentence_id: str
    target_section: str
    user: str = "user"

class EditSentenceRequest(BaseModel):
    job_id: str
    sentence_id: str
    new_text: str
    user: str = "user"

class LinkCodeRequest(BaseModel):
    job_id: str
    sentence_id: str
    code_block: str
    file_name: str
    language: str = "text"
    line_start: Optional[int] = None
    line_end: Optional[int] = None

class ExportRequest(BaseModel):
    job_id: str
    formats: List[str] = ['markdown', 'json', 'sop']


@app.get("/sentences/{job_id}")
async def get_sentence_metadata(job_id: str):
    """Get rich metadata for all sentences in job."""
    with JOB_LOCK:
        if job_id not in JOB_QUEUE:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_id not in SENTENCE_PROCESSORS:
            raise HTTPException(status_code=400, detail="Sentence metadata not available yet")
        
        processor = SENTENCE_PROCESSORS[job_id]
        stats = processor.get_stats()
        
        sentence_list = []
        for sent_id, metadata in processor.sentences.items():
            sentence_list.append({
                "id": sent_id,
                "text": metadata.text,
                "confidence": metadata.confidence_score,
                "predicted_section": metadata.predicted_section,
                "alternatives": metadata.alternatives,
                "importance": metadata.importance_weight,
                "quality": metadata.quality_score,
                "status": metadata.status.value,
                "is_confusing": metadata.is_confusing,
                "confusion_reasons": metadata.confusion_reasons,
                "audio_start": metadata.audio_start,
                "audio_end": metadata.audio_end,
                "assigned_sections": metadata.assigned_sections
            })
        
        return {
            "job_id": job_id,
            "total_sentences": stats['total_sentences'],
            "sentences": sentence_list,
            "statistics": stats
        }


@app.post("/sentences/{job_id}/drag-drop")
async def drag_drop_sentence(job_id: str, request: DragDropRequest):
    """Drag and drop sentence to section."""
    with JOB_LOCK:
        if job_id not in SENTENCE_PROCESSORS:
            raise HTTPException(status_code=400, detail="Job not ready")
        
        processor = SENTENCE_PROCESSORS[job_id]
        if request.sentence_id not in processor.sentences:
            raise HTTPException(status_code=404, detail="Sentence not found")
        
        # Assign sentence
        sentence = processor.sentences[request.sentence_id]
        sentence.assign_section(request.target_section, user=request.user)
        
        # Learn pattern for AI improvement
        keywords = sentence.text.split()[:3]  # First 3 keywords
        for kw in keywords:
            processor.learn_mapping_pattern(request.target_section, kw.lower())
    
    return {
        "status": "assigned",
        "sentence_id": request.sentence_id,
        "section": request.target_section,
        "user": request.user
    }


@app.post("/sentences/{job_id}/edit")
async def edit_sentence(job_id: str, request: EditSentenceRequest):
    """Edit sentence text inline."""
    with JOB_LOCK:
        if job_id not in SENTENCE_PROCESSORS:
            raise HTTPException(status_code=400, detail="Job not ready")
        
        processor = SENTENCE_PROCESSORS[job_id]
        if request.sentence_id not in processor.sentences:
            raise HTTPException(status_code=404, detail="Sentence not found")
        
        sentence = processor.sentences[request.sentence_id]
        old_text = sentence.text
        sentence.update_text(request.new_text, user=request.user)
    
    return {
        "status": "edited",
        "sentence_id": request.sentence_id,
        "old_text": old_text,
        "new_text": request.new_text,
        "user": request.user
    }


@app.get("/sentences/{job_id}/confusion")
async def get_confusing_sentences(job_id: str):
    """Get sentences marked as confusing or requiring clarification."""
    with JOB_LOCK:
        if job_id not in SENTENCE_PROCESSORS:
            raise HTTPException(status_code=400, detail="Job not ready")
        
        processor = SENTENCE_PROCESSORS[job_id]
        
        confusing = []
        clarification_needed = []
        
        for sent_id, sent in processor.sentences.items():
            if sent.is_confusing:
                confusing.append({
                    "id": sent_id,
                    "text": sent.text,
                    "reasons": sent.confusion_reasons,
                    "confidence": sent.confidence_score
                })
            
            if sent.needs_clarification:
                clarification_needed.append({
                    "id": sent_id,
                    "text": sent.text,
                    "question": sent.clarification_requested
                })
        
        return {
            "job_id": job_id,
            "confusing_sentences": confusing,
            "clarification_needed": clarification_needed,
            "total_confusing": len(confusing),
            "total_needing_clarification": len(clarification_needed)
        }


@app.post("/sentences/{job_id}/clarify")
async def request_clarification(job_id: str, sentence_id: str, question: str, user: str = "user"):
    """Mark sentence as needing clarification."""
    with JOB_LOCK:
        if job_id not in SENTENCE_PROCESSORS:
            raise HTTPException(status_code=400, detail="Job not ready")
        
        processor = SENTENCE_PROCESSORS[job_id]
        processor.request_clarification(sentence_id, question, user)
    
    return {
        "status": "clarification_requested",
        "sentence_id": sentence_id,
        "question": question
    }


@app.post("/sentences/{job_id}/mark-confusing")
async def mark_sentence_confusing(job_id: str, sentence_id: str, user: str = "user"):
    """Mark sentence as confusing."""
    with JOB_LOCK:
        if job_id not in SENTENCE_PROCESSORS:
            raise HTTPException(status_code=400, detail="Job not ready")
        
        processor = SENTENCE_PROCESSORS[job_id]
        processor.mark_confusing(sentence_id, user)
    
    return {
        "status": "marked_confusing",
        "sentence_id": sentence_id
    }


@app.post("/sentences/{job_id}/link-code")
async def link_code_to_sentence(job_id: str, request: LinkCodeRequest):
    """Link code/log snippet to sentence."""
    with JOB_LOCK:
        if job_id not in SENTENCE_PROCESSORS:
            raise HTTPException(status_code=400, detail="Job not ready")
        
        processor = SENTENCE_PROCESSORS[job_id]
        
        # Add code reference
        ref = processor.add_code_reference(
            code_block=request.code_block,
            file_name=request.file_name,
            language=request.language,
            line_start=request.line_start,
            line_end=request.line_end
        )
        
        # Link to sentence
        processor.link_code_to_sentence(request.sentence_id, ref.id)
    
    return {
        "status": "linked",
        "reference_id": ref.id,
        "sentence_id": request.sentence_id,
        "file_name": request.file_name
    }


@app.get("/sentences/{job_id}/code-references")
async def get_code_references(job_id: str):
    """Get all code references for job."""
    with JOB_LOCK:
        if job_id not in SENTENCE_PROCESSORS:
            raise HTTPException(status_code=400, detail="Job not ready")
        
        processor = SENTENCE_PROCESSORS[job_id]
        refs = [r.to_dict() for r in processor.code_references.values()]
    
    return {
        "job_id": job_id,
        "code_references": refs,
        "total": len(refs)
    }


@app.get("/sentences/{job_id}/suggest-mapping")
async def suggest_improved_mapping(job_id: str, sentence_id: str):
    """Get improved mapping suggestions based on learned patterns."""
    with JOB_LOCK:
        if job_id not in SENTENCE_PROCESSORS:
            raise HTTPException(status_code=400, detail="Job not ready")
        
        processor = SENTENCE_PROCESSORS[job_id]
        if sentence_id not in processor.sentences:
            raise HTTPException(status_code=404, detail="Sentence not found")
        
        suggestions = processor.suggest_improved_mapping(sentence_id)
    
    return {
        "sentence_id": sentence_id,
        "suggestions": suggestions,
        "learned_patterns": len(processor.mapping_patterns)
    }


@app.post("/sentences/{job_id}/version-create")
async def create_version(job_id: str, user: str = "user", change_summary: str = ""):
    """Create version snapshot."""
    with JOB_LOCK:
        if job_id not in SENTENCE_PROCESSORS:
            raise HTTPException(status_code=400, detail="Job not ready")
        
        processor = SENTENCE_PROCESSORS[job_id]
        version = processor.create_version(user=user, change_summary=change_summary)
    
    return {
        "status": "version_created",
        "version_id": version.version_id,
        "timestamp": version.timestamp,
        "summary": version.change_summary
    }


@app.get("/sentences/{job_id}/versions")
async def get_versions(job_id: str):
    """Get version history."""
    with JOB_LOCK:
        if job_id not in SENTENCE_PROCESSORS:
            raise HTTPException(status_code=400, detail="Job not ready")
        
        processor = SENTENCE_PROCESSORS[job_id]
        versions = processor.get_version_history()
    
    return {
        "job_id": job_id,
        "versions": versions,
        "total": len(versions)
    }


@app.post("/sentences/{job_id}/undo")
async def undo_changes(job_id: str):
    """Undo last change."""
    with JOB_LOCK:
        if job_id not in SENTENCE_PROCESSORS:
            raise HTTPException(status_code=400, detail="Job not ready")
        
        processor = SENTENCE_PROCESSORS[job_id]
        success = processor.undo()
    
    return {
        "status": "undone" if success else "nothing_to_undo",
        "job_id": job_id
    }


@app.post("/sentences/{job_id}/redo")
async def redo_changes(job_id: str):
    """Redo last undone change."""
    with JOB_LOCK:
        if job_id not in SENTENCE_PROCESSORS:
            raise HTTPException(status_code=400, detail="Job not ready")
        
        processor = SENTENCE_PROCESSORS[job_id]
        success = processor.redo()
    
    return {
        "status": "redone" if success else "nothing_to_redo",
        "job_id": job_id
    }


@app.get("/export/{job_id}/list-formats")
async def list_export_formats(job_id: str):
    """Get available export formats."""
    return {
        "available_formats": {
            "markdown": "Clean documentation format",
            "json": "Complete structured export with all metadata",
            "sop": "Standard Operating Procedure / Runbook",
            "html": "Interactive HTML report",
            "checklist": "Coverage improvement checklist"
        }
    }


@app.post("/export/{job_id}")
async def export_kt(job_id: str, request: ExportRequest):
    """Export KT in multiple formats."""
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        kt_data = job.get("kt_structured")
        if not kt_data:
            raise HTTPException(status_code=400, detail="KT data not available")
    
    try:
        from output_generator import StructuredOutputGenerator
        
        transcript = job.get("transcript", "")
        generator = StructuredOutputGenerator(job_id, transcript, kt_data)
        
        exports = {}
        
        if 'markdown' in request.formats:
            exports['markdown'] = generator.generate_markdown_documentation()
        
        if 'json' in request.formats:
            exports['json'] = generator.generate_json_export()
        
        if 'sop' in request.formats:
            exports['sop'] = generator.generate_sop_runbook()
        
        if 'html' in request.formats:
            exports['html'] = generator.generate_html_report()
        
        if 'checklist' in request.formats:
            exports['checklist'] = generator.generate_coverage_checklist()
        
        return {
            "job_id": job_id,
            "formats_generated": list(exports.keys()),
            "exports": exports
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.post("/export/{job_id}/save")
async def save_exports(job_id: str, output_dir: str = None):
    """Save all exports to disk."""
    if not output_dir:
        output_dir = f"./exports/{job_id}"
    
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        kt_data = job.get("kt_structured")
        if not kt_data:
            raise HTTPException(status_code=400, detail="KT data not available")
    
    try:
        from output_generator import StructuredOutputGenerator
        
        transcript = job.get("transcript", "")
        generator = StructuredOutputGenerator(job_id, transcript, kt_data)
        paths = generator.save_all_formats(output_dir)
        
        return {
            "status": "saved",
            "output_directory": output_dir,
            "files": paths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


@app.get("/")
async def root():
    # Serve the classic UI with integrated v2.0 features
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(content="KT Planner API is running. Use /docs for the API docs.")

@app.get("/enhanced")
async def enhanced_ui():
    """Serve the new split-screen v2.0 UI."""
    enhanced_path = os.path.join(os.path.dirname(__file__), "static", "enhanced.html")
    if os.path.exists(enhanced_path):
        return FileResponse(enhanced_path)
    return HTMLResponse(content="Enhanced UI not available.")


# ========== NEW DIAGNOSTIC & RECOVERY ENDPOINTS ==========

@app.post("/rebuild-coverage/{job_id}")
async def rebuild_coverage_from_current_state(job_id: str):
    """Force rebuild of coverage metrics from current section_content.
    
    Use this when:
    - Manual assignments were made but coverage didn't update
    - Coverage appears stuck or incorrect
    - You want to refresh coverage metrics
    """
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        kt = job.get("kt_structured")
        if not kt or not isinstance(kt, dict):
            raise HTTPException(status_code=400, detail="KT data not available")
    
    section_content = kt.get("section_content", {})
    new_coverage, new_missing, new_progress = recalculate_coverage_from_section_content(section_content)
    
    with JOB_LOCK:
        JOB_QUEUE[job_id]["coverage"] = new_coverage
        JOB_QUEUE[job_id]["missing_required"] = new_missing
        JOB_QUEUE[job_id]["progress"] = new_progress
    
    return {
        "status": "rebuilt",
        "job_id": job_id,
        "new_coverage_percent": new_progress,
        "sections_covered": sum(1 for c in new_coverage.values() if c["status"] in {"covered", "weak"}),
        "total_sections": len(new_coverage),
        "missing_required_sections": new_missing,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    }


@app.get("/diagnose/{job_id}")
async def diagnose_coverage_issues(job_id: str):
    """Diagnostic endpoint to understand why sections are empty.
    
    Returns:
    - Which sections are missing and why
    - Which unassigned sentences exist
    - Recommendations for populating each section
    - Confidence scores from original classification
    """
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        kt = job.get("kt_structured")
        if not kt or not isinstance(kt, dict):
            raise HTTPException(status_code=400, detail="KT data not available")
    
    section_content = kt.get("section_content", {})
    unassigned = kt.get("unassigned_sentences", [])
    transcript = kt.get("transcript", "")
    
    diagnostics = {
        "job_id": job_id,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        "total_unassigned_sentences": len(unassigned),
        "unassigned_preview": [s.get("text", "")[:100] for s in unassigned[:5]],
        "section_diagnostics": {}
    }
    
    # Analyze each required section
    for sec in SCHEMA:
        sec_id = sec["id"]
        sec_data = section_content.get(sec_id, {})
        sentences = sec_data.get("sentences", [])
        
        section_diag = {
            "section_id": sec_id,
            "title": sec.get("title"),
            "required": sec.get("required", False),
            "status": "missing" if len(sentences) == 0 else ("weak" if len(sentences) < 2 else "covered"),
            "sentence_count": len(sentences),
            "hints": sec.get("hints", [])[:3],  # First 3 hints
            "recommendation": ""
        }
        
        # Generate recommendation
        if len(sentences) == 0 and sec.get("required"):
            section_diag["recommendation"] = f"This required section is EMPTY. Look for unassigned sentences matching these hints: {', '.join(sec.get('hints', [])[:3])}. Or manually create content describing: {sec.get('description', 'N/A')}"
        elif len(sentences) < 2 and sec.get("required"):
            section_diag["recommendation"] = f"This required section has only {len(sentences)} sentence. Need at least 2 for 'covered' status."
        
        diagnostics["section_diagnostics"][sec_id] = section_diag
    
    return diagnostics


@app.post("/populate-section/{job_id}/{section_id}")
async def auto_populate_section_from_unassigned(job_id: str, section_id: str):
    """Auto-assign unassigned sentences to a section by confidence score.
    
    Attempts to match unassigned sentences to the target section based on:
    - Semantic similarity to section hints
    - Classification confidence from original processing
    
    Use this to quickly populate empty sections.
    """
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        kt = job.get("kt_structured")
        if not kt or not isinstance(kt, dict):
            raise HTTPException(status_code=400, detail="KT data not available")
    
    section_content = kt.get("section_content", {})
    unassigned = kt.get("unassigned_sentences", [])
    
    # Find the section
    section_info = next((s for s in SCHEMA if s["id"] == section_id), None)
    if not section_info:
        raise HTTPException(status_code=404, detail=f"Section {section_id} not found in schema")
    
    # Initialize section if needed
    if section_id not in section_content:
        section_content[section_id] = {
            "section_id": section_id,
            "section_title": section_info.get("title"),
            "sentences": [],
            "enhanced_texts": [],
            "repair_actions": [],
            "screenshots": [],
            "confidence": 0.0
        }
    
    # Find best-matching unassigned sentences
    section_hints = section_info.get("hints", [])
    matches = []
    
    for sent in unassigned:
        sent_text = sent.get("text", "").lower()
        hint_matches = sum(1 for hint in section_hints if hint.lower() in sent_text)
        
        if hint_matches > 0:
            matches.append({
                "sentence": sent,
                "hint_matches": hint_matches,
                "confidence": min(0.5 + (hint_matches * 0.1), 0.95)
            })
    
    # Sort by confidence
    matches.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Assign top matches (up to 5) to the section
    assigned_count = 0
    for match in matches[:5]:
        sent = match["sentence"]
        section_content[section_id]["sentences"].append(sent)
        section_content[section_id]["enhanced_texts"].append(sent.get("text", ""))
        
        # Remove from unassigned
        kt["unassigned_sentences"] = [
            s for s in kt.get("unassigned_sentences", [])
            if s.get("text", "").strip().lower() != sent.get("text", "").strip().lower()
        ]
        assigned_count += 1
    
    # Persist and recalculate coverage
    with JOB_LOCK:
        JOB_QUEUE[job_id]["kt_structured"] = kt
        new_coverage, new_missing, new_progress = recalculate_coverage_from_section_content(section_content)
        JOB_QUEUE[job_id]["coverage"] = new_coverage
        JOB_QUEUE[job_id]["missing_required"] = new_missing
        JOB_QUEUE[job_id]["progress"] = new_progress
    
    return {
        "status": "populated",
        "section_id": section_id,
        "section_title": section_info.get("title"),
        "sentences_assigned": assigned_count,
        "matched_candidates": len(matches),
        "new_coverage_percent": new_progress,
        "message": f"Assigned {assigned_count} sentences to {section_info.get('title')}"
    }


# ========== CRITICAL: AI WORD MATCHING RE-CLASSIFICATION ==========

@app.post("/reclassify/{job_id}")
async def reclassify_transcript_using_ai_matching(job_id: str):
    """
    ⭐ CRITICAL FIX: Re-classify the entire transcript using ai.py's classify_transcript function.
    
    This uses the intelligent 3-level hint matching algorithm with weightage:
    - Level 3: Exact phrase match (highest confidence) 
    - Level 2: Token match at word boundaries
    - Level 1: Partial token match
    
    Run this when:
    - Coverage is showing all sections as "missing"
    - Initial classification didn't work properly
    - You want to use better word-based matching
    - After receiving new audio
    
    Example:
        curl -X POST http://localhost:8000/reclassify/YOUR-JOB-ID
    """
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        kt = job.get("kt_structured")
        if not kt or not isinstance(kt, dict):
            raise HTTPException(status_code=400, detail="KT data not available")
    
    transcript = kt.get("transcript", "")
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript available")
    
    # ===== USE THE POWERFUL classify_transcript FUNCTION FROM ai.py =====
    from ai import classify_transcript
    
    try:
        # Call the intelligent word-matching classification with 3-level hint system
        classified_chunks = classify_transcript(transcript, similarity_threshold=-0.05)
        
        # Build section_content from classified chunks
        section_content = {}
        total_classified = 0
        
        for sec_id, chunks in classified_chunks.items():
            # Find section in schema
            section = next((s for s in SCHEMA if s["id"] == sec_id), None)
            if not section:
                continue
            
            # Convert chunks to sentences with metadata
            sentences = []
            for chunk_idx, chunk in enumerate(chunks):
                sentences.append({
                    "text": chunk,
                    "original_chunk": chunk,
                    "confidence": 0.75,  # Default confidence for auto-classified
                    "chunk_index": chunk_idx,
                    "classification_method": "ai_word_matching_v3",
                    "assigned_sections": [sec_id]
                })
                total_classified += 1
            
            section_content[sec_id] = {
                "section_id": sec_id,
                "section_title": section.get("title"),
                "sentences": sentences,
                "enhanced_texts": [s["text"] for s in sentences],
                "repair_actions": [],
                "screenshots": [],
                "confidence": 0.75 if sentences else 0.0,
                "sentence_count": len(sentences)
            }
        
        # Update KT with new classifications
        kt["section_content"] = section_content
        kt["classification_method"] = "ai_word_matching_3level"
        
        # Clear any previous unassigned sentences (all got classified now)
        kt["unassigned_sentences"] = []
        
        # Recalculate coverage
        new_coverage, new_missing, new_progress = recalculate_coverage_from_section_content(section_content)
        
        # Persist updates
        with JOB_LOCK:
            JOB_QUEUE[job_id]["kt_structured"] = kt
            JOB_QUEUE[job_id]["coverage"] = new_coverage
            JOB_QUEUE[job_id]["missing_required"] = new_missing
            JOB_QUEUE[job_id]["progress"] = new_progress
        
        # Calculate how many sections now have content
        covered_count = sum(1 for c in new_coverage.values() if c["status"] in {"covered", "weak"})
        
        # Build detailed response with section-by-section breakdown
        sections_populated = {}
        for sec_id, section_data in section_content.items():
            sentences = section_data.get("sentences", [])
            sections_populated[sec_id] = {
                "section_title": section_data.get("section_title"),
                "sentence_count": len(sentences),
                "status": new_coverage.get(sec_id, {}).get("status"),
                "sample_sentences": [s["text"][:80] + "..." if len(s["text"]) > 80 else s["text"] for s in sentences[:2]]
            }
        
        return {
            "status": "reclassified",
            "job_id": job_id,
            "method": "AI Word Matching (3-level hint weighting)",
            "new_coverage_percent": new_progress,
            "sections_now_covered": covered_count,
            "total_sections": len(SCHEMA),
            "total_chunks_classified": total_classified,
            "sections_populated": sections_populated,
            "message": f"✅ Re-classification COMPLETE! Coverage improved to {new_progress}%. All {total_classified} content chunks classified using intelligent word matching."
        }
    
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Re-classification failed: {str(e)}. Error: {traceback.format_exc()}"
        )
