from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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

MODEL = None

with open("kt_schema.json") as f:
    SCHEMA = json.load(f)["sections"]

def get_transcription_model():
    global MODEL
    if MODEL is None:
        MODEL = whisper.load_model("base")
    return MODEL

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

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Continuum — KT Coverage</title>
        <style>
          :root {
            color-scheme: light;
            font-family: "Inter", system-ui, -apple-system, sans-serif;
          }
          body {
            margin: 0;
            background: #f6f7fb;
            color: #1f2933;
          }
          header {
            background: #101828;
            color: #fff;
            padding: 32px 24px;
          }
          header h1 {
            margin: 0 0 8px;
            font-size: 28px;
          }
          header p {
            margin: 0;
            opacity: 0.8;
          }
          main {
            max-width: 1000px;
            margin: 0 auto;
            padding: 32px 24px 48px;
          }
          .grid {
            display: grid;
            gap: 24px;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          }
          .card {
            background: #fff;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 24px rgba(16, 24, 40, 0.08);
          }
          .card h2 {
            margin-top: 0;
            font-size: 18px;
          }
          .upload {
            display: flex;
            flex-direction: column;
            gap: 12px;
          }
          button {
            background: #3751ff;
            color: #fff;
            border: none;
            border-radius: 10px;
            padding: 12px 16px;
            font-weight: 600;
            cursor: pointer;
          }
          button:disabled {
            background: #c7c9d9;
            cursor: not-allowed;
          }
          .badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            background: #e0f2fe;
            color: #075985;
          }
          .status {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 8px 0;
            border-bottom: 1px solid #eef2f6;
          }
          .status span {
            font-size: 14px;
          }
          .status strong {
            text-transform: capitalize;
          }
          textarea {
            width: 100%;
            min-height: 180px;
            border-radius: 12px;
            border: 1px solid #d8dee9;
            padding: 12px;
            font-family: inherit;
          }
        </style>
      </head>
      <body>
        <header>
          <h1>Continuum</h1>
          <p>Structured KT coverage and handover readiness in one place.</p>
        </header>
        <main>
          <div class="grid">
            <div class="card">
              <h2>Upload KT Recording</h2>
              <form class="upload" id="upload-form">
                <input type="file" id="file-input" accept="audio/*,video/*" />
                <button type="submit">Analyze KT Coverage</button>
              </form>
              <p class="badge" id="progress">Progress: 0%</p>
              <div id="missing"></div>
            </div>
            <div class="card">
              <h2>Section Coverage</h2>
              <div id="coverage"></div>
            </div>
            <div class="card">
              <h2>Transcript</h2>
              <textarea id="transcript" readonly></textarea>
            </div>
          </div>
        </main>
        <script>
          const coverageEl = document.getElementById("coverage");
          const transcriptEl = document.getElementById("transcript");
          const progressEl = document.getElementById("progress");
          const missingEl = document.getElementById("missing");
          const form = document.getElementById("upload-form");
          const fileInput = document.getElementById("file-input");

          const renderCoverage = (coverage = {}) => {
            coverageEl.innerHTML = "";
            Object.entries(coverage).forEach(([_, value]) => {
              const row = document.createElement("div");
              row.className = "status";
              row.innerHTML = `<span>${value.title}</span><strong>${value.status}</strong>`;
              coverageEl.appendChild(row);
            });
          };

          form.addEventListener("submit", async (event) => {
            event.preventDefault();
            const file = fileInput.files[0];
            if (!file) {
              alert("Please select a file.");
              return;
            }
            const formData = new FormData();
            formData.append("file", file);
            form.querySelector("button").disabled = true;
            form.querySelector("button").textContent = "Analyzing...";
            try {
              const response = await fetch("/upload", { method: "POST", body: formData });
              if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || "Upload failed.");
              }
              const data = await response.json();
              transcriptEl.value = data.transcript || "";
              progressEl.textContent = `Progress: ${data.progress || 0}%`;
              renderCoverage(data.coverage || {});
              if (data.missing_required && data.missing_required.length) {
                missingEl.innerHTML = `<p><strong>Missing required sections:</strong> ${data.missing_required.join(", ")}</p>`;
              } else {
                missingEl.innerHTML = "<p><strong>All required sections covered.</strong></p>";
              }
            } catch (err) {
              alert(err.message);
            } finally {
              form.querySelector("button").disabled = false;
              form.querySelector("button").textContent = "Analyze KT Coverage";
            }
          });

          fetch("/schema")
            .then((res) => res.json())
            .then((data) => {
              if (data.sections) {
                const coverage = {};
                data.sections.forEach((section) => {
                  coverage[section.id] = { title: section.title, status: "pending" };
                });
                renderCoverage(coverage);
              }
            })
            .catch(() => {});
        </script>
      </body>
    </html>
    """

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

        model = get_transcription_model()
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
