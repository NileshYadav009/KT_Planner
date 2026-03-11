from fastapi import FastAPI, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import whisper
import ffmpeg
import tempfile
import os
import json
import uuid
from threading import Lock
from ai import classify_transcript, get_sentence_model, SECTION_HINTS, map_analysis_to_fields
from sentence_transformers import util

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve the frontend static files
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL = None
JOB_QUEUE = {}  # job_id -> {status, transcript, coverage, missing_required, progress, error}
JOB_LOCK = Lock()

with open("kt_schema_new.json") as f:
    SCHEMA = json.load(f)["sections"]


@app.on_event("startup")
def load_models():
    """Load heavy models on startup so endpoints can use them."""
    global MODEL
    if MODEL is None:
        # Use 'tiny' model for ~4x speedup over 'small'; good accuracy/speed tradeoff
        MODEL = whisper.load_model("tiny")

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
            ffmpeg.input(input_path).output(
                audio_path_wav, acodec="pcm_s16le", ac=1, ar=16000
            ).overwrite_output().run(quiet=True, stderr=None, stdout=None)
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

        # Transcribe (tiny model is faster)
        result = MODEL.transcribe(audio_to_use, language="en", verbose=False)
        transcript = result.get("text", "").strip()

        # update progress after transcription
        with JOB_LOCK:
            if job_id in JOB_QUEUE:
                JOB_QUEUE[job_id]["progress"] = 30

        if not transcript:
            raise ValueError("No speech detected in the uploaded file.")

        # Analyze transcript to produce structured KT coverage
        analysis = classify_transcript(transcript) if False else None
        try:
            from ai import analyze_transcript
            analysis = analyze_transcript(transcript, similarity_threshold=0.30)  # raised threshold for selectivity
            # Deduplicate to prevent same chunks appearing in multiple sections
            analysis = deduplicate_analysis(analysis)
        except Exception:
            # fallback to old classifier (mapping only)
            classified = classify_transcript(transcript)
            coverage, missing_required, progress = build_coverage(classified)
            analysis = {k: {"status": ("covered" if v else "missing"), "confidence": 0.0, "extracted_text": "\n".join(v), "chunks": v, "scores": []} for k, v in classified.items()}

        # build missing_required and progress from analysis and schema
        coverage = {}
        missing_required = []
        for sec in SCHEMA:
            sid = sec['id']
            a = analysis.get(sid, {"status": "missing", "confidence": 0.0, "extracted_text": "", "chunks": []})
            coverage[sid] = {
                'title': sec.get('title', sid),
                'status': a['status'],
                'confidence': a.get('confidence', 0.0),
                'content': a.get('chunks', []),
                'extracted_text': a.get('extracted_text', ''),
                'chunks': a.get('chunks', [])
            }
            if a['status'] == 'missing' and sec.get('required'):
                missing_required.append(sid)

        covered_sections = sum(1 for s in coverage.values() if s['status'] in {'covered', 'partial'})
        progress = int(100 * covered_sections / len(coverage)) if coverage else 0

        # Rebuild missing_required
        missing_required = []
        for sec in SCHEMA:
            a = coverage.get(sec['id'], {})
            if a.get('status') == 'missing' and sec.get('required'):
                missing_required.append(sec['id'])

        with JOB_LOCK:
            if job_id in JOB_QUEUE:
                JOB_QUEUE[job_id]["progress"] = 60

        # Post-process segments: autofill low-confidence segments and capture screenshots
        screenshots = []
        try:
            segments = result.get('segments', [])
        except Exception:
            segments = []

        # Ensure screenshots directory
        screenshots_dir = os.path.join(os.path.dirname(__file__), 'static', 'screenshots')
        os.makedirs(screenshots_dir, exist_ok=True)

        sentence_model = get_sentence_model()
        max_screens = 5
        for seg in segments:
            seg_text = seg.get('text', '').strip()
            start_t = seg.get('start', 0)
            avg_logprob = seg.get('avg_logprob', None)

            low_conf = False
            if avg_logprob is not None:
                # threshold heuristic: very low average logprob indicates low confidence
                if avg_logprob < -1.0 or len(seg_text) < 10:
                    low_conf = True

            if low_conf:
                # build context from nearby segments
                context = seg_text
                # find best hint across all SECTION_HINTS using sentence-transformers
                hints = []
                for sec_id, hint_set in SECTION_HINTS.items():
                    for h in hint_set:
                        hints.append((sec_id, h))
                if hints:
                    texts = [h for (_, h) in hints]
                    emb_ctx = sentence_model.encode(context or ' '.join(texts[:1]), convert_to_tensor=True)
                    emb_hints = sentence_model.encode(texts, convert_to_tensor=True, batch_size=64)
                    sims = util.cos_sim(emb_ctx, emb_hints)[0]
                    best_idx = int(sims.argmax())
                    guessed = texts[best_idx]
                else:
                    guessed = '[inaudible]'

                # append guessed note into transcript and coverage (simple approach)
                transcript += f"\n[inaudible - guessed: {guessed}]"

            # Capture screenshot for segments that likely map to a section
            if len(screenshots) < max_screens and len(seg_text) > 10:
                sec_class = classify_transcript(seg_text)
                # if any section matched, capture a frame
                if any(len(v) > 0 for v in sec_class.values()):
                    img_name = f"{job_id}_{int(start_t*1000)}.jpg"
                    img_path = os.path.join(screenshots_dir, img_name)
                    try:
                        # extract single frame at time 'start_t'
                        ffmpeg.input(input_path, ss=start_t).output(img_path, vframes=1).run(quiet=True, overwrite_output=True)
                        screenshots.append(f"/static/screenshots/{img_name}")
                    except Exception:
                        pass

        with JOB_LOCK:
            if job_id in JOB_QUEUE:
                JOB_QUEUE[job_id]["progress"] = 85


        # Attempt to map analyzed chunks into concrete schema fields
        try:
            mapped_fields = map_analysis_to_fields(analysis, SCHEMA, min_similarity=0.25)
        except Exception as e:
            mapped_fields = {}
            print('map_analysis_to_fields failed:', e)

        with JOB_LOCK:
            JOB_QUEUE[job_id] = {
                "status": "completed",
                "transcript": transcript,
                "coverage": coverage,
                "mapped_fields": mapped_fields,
                "missing_required": missing_required,
                "progress": progress,
                "screenshots": screenshots,
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
