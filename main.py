from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import whisper
import ffmpeg
import tempfile
import os
import json
from ai import classify_transcript

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = whisper.load_model("base")

with open("kt_schema.json") as f:
    SCHEMA = json.load(f)["sections"]

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

@app.get("/schema")
async def get_schema():
    return {"sections": SCHEMA}

@app.post("/upload")
async def upload(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    input_path = None
    audio_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            input_path = tmp.name

        if os.path.getsize(input_path) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        audio_path = f"{input_path}.wav"
        ffmpeg.input(input_path).output(audio_path).run(quiet=True, overwrite_output=True)

        result = model.transcribe(audio_path)
        transcript = result.get("text", "").strip()

        if not transcript:
            raise HTTPException(
                status_code=422,
                detail="No speech detected in the uploaded file."
            )

        classified = classify_transcript(transcript)
        coverage, missing_required, progress = build_coverage(classified)

        return {
            "transcript": transcript,
            "coverage": coverage,
            "missing_required": missing_required,
            "progress": progress
        }
    finally:
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
