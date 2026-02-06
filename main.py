from fastapi import FastAPI, UploadFile
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

@app.post("/upload")
async def upload(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    audio_path = input_path + ".wav"
    ffmpeg.input(input_path).output(audio_path).run(quiet=True, overwrite_output=True)

    result = model.transcribe(audio_path)
    transcript = result["text"]

    classified = classify_transcript(transcript)

    coverage = {}
    missing_required = []

    for sec in SCHEMA:
        chunks = classified[sec["id"]]
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

    os.unlink(input_path)
    os.unlink(audio_path)

    return {
        "transcript": transcript,
        "coverage": coverage,
        "missing_required": missing_required,
        "progress": int(
            100 * sum(1 for s in coverage.values() if s["status"] == "covered")
            / len(coverage)
        )
    }