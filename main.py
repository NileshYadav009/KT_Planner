from fastapi import FastAPI, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import ffmpeg
import tempfile
import os
import json
import uuid
from threading import Lock
from datetime import datetime
import torch
from ai import classify_transcript, get_sentence_model, SECTION_HINTS, map_analysis_to_fields, build_section_paragraphs, polish_coverage_sections
from context_mapper import ContextMappingPipeline, serialize_kt
from devops_transcription import clean_transcript
from llm_provider import get_llm_provider
from sentence_transformers import util

# Optional environment config for Whisper model
DEFAULT_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
DEFAULT_WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
DEFAULT_WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "2"))

app = FastAPI()

# CORS configuration - restrict to allowed origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Serve the frontend static files
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL = None
MAPPER_PIPELINE = None
JOB_QUEUE = {}  # job_id -> {status, transcript, coverage, missing_required, progress, error, kt_structured}
JOB_LOCK = Lock()

with open("kt_schema_new.json") as f:
    SCHEMA = json.load(f)["sections"]


@app.on_event("startup")
def load_models():
    """Load heavy models on startup so endpoints can use them."""
    global MODEL, MAPPER_PIPELINE
    if MODEL is None:
        model_name = DEFAULT_WHISPER_MODEL
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = DEFAULT_WHISPER_COMPUTE_TYPE

        # Use faster CPU-friendly model unless the environment explicitly allows medium on CPU.
        if device == "cpu" and model_name == "medium" and os.getenv("WHISPER_ALLOW_MEDIUM_CPU", "0") != "1":
            model_name = "small"

        MODEL = WhisperModel(model_name, device=device, compute_type=compute_type)

    if MAPPER_PIPELINE is None:
        provider = get_llm_provider()
        MAPPER_PIPELINE = ContextMappingPipeline(
            SCHEMA,
            llm_fallback_fn=provider.generate if provider else None
        )

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
    """Background task for transcription and classification."""
    try:
        if os.path.getsize(input_path) == 0:
            raise ValueError("Uploaded file is empty.")

        # Try to extract/convert audio; fall back to original if conversion fails
        audio_to_use = input_path
        try:
            # Try to extract audio as WAV (more compatible than MP3)
            audio_path_wav = f"{input_path}.wav"
            if not input_path.lower().endswith(('.wav', '.mp3')):
                ffmpeg.input(input_path).output(
                    audio_path_wav, acodec="pcm_s16le", ac=1, ar=16000
                ).overwrite_output().run(quiet=True, stderr=None, stdout=None)
            else:
                audio_path_wav = input_path
            if os.path.exists(audio_path_wav) and os.path.getsize(audio_path_wav) > 0:
                audio_to_use = audio_path_wav
        except Exception as e:
            # If conversion fails, try the original file directly
            pass

        # Trim long silent sections to speed up transcription
        try:
            trimmed_path = f"{input_path}.trimmed.wav"
            # remove silence at start/end and long pauses (ffmpeg silenceremove)
            ffmpeg.input(audio_to_use).filter_('silenceremove', start_periods=1, start_silence=0.5, start_threshold='-50dB', stop_periods=1, stop_silence=0.5, stop_threshold='-50dB').output(trimmed_path).run(quiet=True, overwrite_output=True)
            if os.path.exists(trimmed_path) and os.path.getsize(trimmed_path) > 0:
                audio_to_use = trimmed_path
        except Exception:
            # If trimming fails, continue with original audio
            pass

        # Transcribe using faster-whisper with faster settings by default.
        # On CPU we prefer the small model, while GPU can use medium when configured.
        transcribe_kwargs = {
            "language": "en",
            "beam_size": DEFAULT_WHISPER_BEAM_SIZE,
            "task": "transcribe"
        }
        segments, info = MODEL.transcribe(audio_to_use, **transcribe_kwargs)
        segments = list(segments)

        # Clean each segment once and build the joined transcript from cleaned parts
        cleaned_segments = []
        raw_parts = []
        for s in segments:
            cleaned_text = clean_transcript(s.text)
            raw_parts.append(cleaned_text)
            cleaned_segments.append({
                "id": s.id,
                "seek": s.seek,
                "start": s.start,
                "end": s.end,
                "text": cleaned_text,
                "avg_logprob": getattr(s, "avg_logprob", None),
                "compression_ratio": getattr(s, "compression_ratio", None),
                "no_speech_prob": getattr(s, "no_speech_prob", None)
            })

        raw_text = " ".join(raw_parts).strip()

        result = {
            "text": raw_text,
            "segments": cleaned_segments,
            "language": info.language if info else "en"
        }
        transcript = result.get("text", "")

        # update progress after transcription
        with JOB_LOCK:
            if job_id in JOB_QUEUE:
                JOB_QUEUE[job_id]["progress"] = 30

        if not transcript:
            raise ValueError("No speech detected in the uploaded file.")

        # Process transcript with the full 7-stage context mapping pipeline
        if MAPPER_PIPELINE is None:
            raise RuntimeError("Context mapping pipeline not initialized")

        kt = MAPPER_PIPELINE.process(job_id, transcript, result.get('segments', []))

        paragraph_data = {}
        try:
            paragraph_data = build_section_paragraphs(transcript) or {}
        except Exception:
            paragraph_data = {}

        coverage = {}
        missing_required = kt.missing_required_sections or []
        for sec_id, cov in kt.coverage.items():
            coverage_sentences = []
            
            # NEW: Extract sentences from topic blocks (preserves order and grouping)
            blocks = getattr(cov, 'blocks', []) or []
            for block in blocks:
                for s in block.sentences:
                    coverage_sentences.append({
                        'text': getattr(s, 'text', ''),
                        'start': getattr(s, 'start', 0.0),
                        'end': getattr(s, 'end', 0.0),
                        'speaker': getattr(s, 'speaker', None),
                        'audio_confidence': getattr(s, 'audio_confidence', 0.0),
                        'assigned_sections': [sec_id]
                    })
            
            # Fallback: try old structure for backwards compatibility
            if not coverage_sentences:
                old_sentences = getattr(cov, 'sentences', []) or []
                for s in old_sentences:
                    coverage_sentences.append({
                        'text': getattr(s, 'text', ''),
                        'start': getattr(s, 'start', 0.0),
                        'end': getattr(s, 'end', 0.0),
                        'speaker': getattr(s, 'speaker', None),
                        'audio_confidence': getattr(s, 'audio_confidence', 0.0),
                        'assigned_sections': [sec_id]
                    })
            
            # Last resort: check section_content
            if not coverage_sentences:
                section_content = kt.section_content.get(sec_id, {})
                coverage_sentences = section_content.get('sentences', []) if isinstance(section_content, dict) else []

            # Raw transcribed fragments as extracted from the transcript.
            raw_content = [s.get('text', '') for s in coverage_sentences]

            coverage[sec_id] = {
                'title': cov.section_title,
                'fragments': raw_content,
                'status': cov.status,
                'required': cov.required,
                'sentence_count': cov.sentence_count,
                'confidence': cov.confidence_score,
                'risk': cov.risk_score,
                'sentences': coverage_sentences,
                'blocks': [b.to_dict() for b in blocks]
            }

        # Batch polish all coverage sections in one Gemini request per KT.
        try:
            polished_sections = polish_coverage_sections(
                {sid: {
                    'title': coverage[sid]['title'],
                    'fragments': coverage[sid]['fragments']
                } for sid in coverage},
                max_fragments_per_section=8,
            )
        except Exception as e:
            logger.warning("Batch coverage polish failed: %s", e)
            polished_sections = {}

        for sid, section_payload in coverage.items():
            display_content = polished_sections.get(sid)
            if not display_content:
                display_content = section_payload['fragments']
            coverage[sid]['content'] = display_content
            del coverage[sid]['fragments']

        progress = int(round(kt.overall_coverage_percent or 0))
        transcript = kt.transcript
        kt_structured = serialize_kt(kt)
        if paragraph_data:
            kt_structured["paragraphs"] = paragraph_data

        with JOB_LOCK:
            if job_id in JOB_QUEUE:
                JOB_QUEUE[job_id]["progress"] = 60

        # Post-process segments: disabled screenshot capture for now
        screenshots = []
        # try:
        #     segments = result.get('segments', [])
        # except Exception:
        #     segments = []
        #
        # # Ensure screenshots directory
        # screenshots_dir = os.path.join(os.path.dirname(__file__), 'static', 'screenshots')
        # os.makedirs(screenshots_dir, exist_ok=True)
        #
        # try:
        #     sentence_model = get_sentence_model()
        # except Exception:
        #     sentence_model = None
        #
        # max_screens = 2
        # for seg in segments:
        #     seg_text = seg.get('text', '').strip()
        #     start_t = seg.get('start', 0)
        #     avg_logprob = seg.get('avg_logprob', None)
        #
        #     low_conf = False
        #     if avg_logprob is not None:
        #         # threshold heuristic: very low average logprob indicates low confidence
        #         if avg_logprob < -1.0 or len(seg_text) < 10:
        #             low_conf = True
        #
        #     if low_conf:
        #         # build context from nearby segments
        #         context = seg_text
        #         # find best hint across all SECTION_HINTS using sentence-transformers
        #         hints = []
        #         for sec_id, hint_set in SECTION_HINTS.items():
        #             for h in hint_set:
        #                 hints.append((sec_id, h))
        #         if hints:
        #             texts = [h for (_, h) in hints]
        #             emb_ctx = sentence_model.encode(context or ' '.join(texts[:1]), convert_to_tensor=True)
        #             emb_hints = sentence_model.encode(texts, convert_to_tensor=True, batch_size=64)
        #             sims = util.cos_sim(emb_ctx, emb_hints)[0]
        #             best_idx = int(sims.argmax())
        #             guessed = texts[best_idx]
        #         else:
        #             guessed = '[inaudible]'
        #
        #         # append guessed note into transcript and coverage (simple approach)
        #         transcript += f"\n[inaudible - guessed: {guessed}]"
        #
        #     # Capture screenshot for segments that likely map to a section
        #     if len(screenshots) < max_screens and len(seg_text) > 10:
        #         sec_class = classify_transcript(seg_text)
        #         # if any section matched, capture a frame
        #         if any(len(v) > 0 for v in sec_class.values()):
        #             img_name = f"{job_id}_{int(start_t*1000)}.jpg"
        #             img_path = os.path.join(screenshots_dir, img_name)
        #             try:
        #                 # extract single frame at time 'start_t'
        #                 ffmpeg.input(input_path, ss=start_t).output(img_path, vframes=1).run(quiet=True, overwrite_output=True)
        #                 screenshots.append(f"/static/screenshots/{img_name}")
        #             except Exception:
        #                 pass

        with JOB_LOCK:
            if job_id in JOB_QUEUE:
                JOB_QUEUE[job_id]["progress"] = 85


        field_analysis = {
            sec_id: {
                "chunks": info.get("content", []),
                "scores": [float(info.get("confidence", 0.0))] * len(info.get("content", []))
            }
            for sec_id, info in coverage.items()
        }
        mapped_fields = map_analysis_to_fields(field_analysis, SCHEMA)

        with JOB_LOCK:
            JOB_QUEUE[job_id] = {
                "status": "completed",
                "transcript": transcript,
                "coverage": coverage,
                "mapped_fields": mapped_fields,
                "missing_required": missing_required,
                "progress": progress,
                "screenshots": screenshots,
                "kt_structured": kt_structured,
                "error": None
            }
    except Exception as e:
        with JOB_LOCK:
            JOB_QUEUE[job_id] = {
                "status": "failed",
                "error": str(e)
            }
    finally:
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)
        # Clean up any extracted audio files
        # remove any temporary audio files we created
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


@app.get("/")
async def root():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(content="KT Planner API is running. Use /docs for the API docs.")


@app.post('/feedback')
async def receive_feedback(payload: dict):
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

    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail='Job not found')

        # Record feedback
        fb = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'sentence_id': sentence_id,
            'corrected_classification': corrected_section,
            'user': payload.get('user', 'unknown'),
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

    return JSONResponse({
        'status': 'ok',
        'message': 'feedback applied',
        'feedback': fb,
        'coverage': coverage,
        'missing_required': missing_required,
        'progress': progress
    })
