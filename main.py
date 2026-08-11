from fastapi import FastAPI, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import ffmpeg
import tempfile
import os
import json
import uuid
from threading import Lock
from datetime import datetime
import re
import torch
from ai import analyze_transcript, classify_transcript, generate_report, get_sentence_model, SECTION_HINTS, map_analysis_to_fields, build_section_paragraphs, polish_coverage_sections
from context_mapper import ContextMappingPipeline, serialize_kt
from devops_transcription import clean_transcript
from llm_provider import get_llm_provider
from schema_generator import generate_dynamic_schema
from field_populator import populate_fields
from knowledge import build_knowledge_object
from renderers import get_renderer
from sentence_transformers import util

import logging
logger = logging.getLogger(__name__)

try:
    from markdown import markdown as markdown_to_html
except ImportError:
    markdown_to_html = None

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except ImportError:
    Environment = None

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

@app.get("/schema/{job_id}")
async def get_job_schema(job_id: str):
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
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
    }


def html_escape(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return (
        value.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;")
    )


def build_toc_sections(rendered_sections: list) -> list:
    toc = []
    for idx, section in enumerate(rendered_sections, start=1):
        section_title = section.get("section_title") or section.get("section_id") or f"Section {idx}"
        anchor = section.get("section_id") or f"section-{idx}"
        toc.append({"title": section_title, "anchor": anchor})
    return toc


def _render_paragraph_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    content = text.strip()
    if not content:
        return ""
    if re.search(r"<[^>]+>", content):
        return text
    if markdown_to_html:
        return markdown_to_html(text)
    return html_escape(text)


def render_section_blocks(rendered_sections: list) -> str:
    html = []
    for section in rendered_sections:
        section_title = html_escape(section.get("section_title") or section.get("section_id") or "Section")
        section_id = section.get("section_id") or section_title.lower().replace(" ", "-")
        html.append(f"<section class=\"section-block\" id=\"{html_escape(section_id)}\">")
        html.append(f"<h2 class=\"section-title\">{section_title}</h2>")
        for block in section.get("blocks", []):
            block_title = html_escape(block.get("title") or block.get("type", "Block"))
            block_type = block.get("type")
            html.append(f"<div class=\"block-card\">")
            html.append(f"<h3 class=\"block-title\">{block_title}</h3>")
            if block_type == "NarrativeBlock":
                for p in block.get("paragraphs", []):
                    html.append(f"<div class=\"narrative-para\">{_render_paragraph_text(p)}</div>")
            elif block_type == "ChecklistBlock":
                html.append("<ul>")
                for item in block.get("items", []):
                    html.append(f"<li>{html_escape(item)}</li>")
                html.append("</ul>")
            elif block_type == "WarningBlock":
                for warning in block.get("warnings", []):
                    html.append(f"<p><strong>{html_escape(warning)}</strong></p>")
            elif block_type == "TechnologyGrid":
                html.append("<div class=\"table-wrapper\"><table>")
                for row in block.get("rows", []):
                    html.append(
                        f"<tr><td>{html_escape(row.get('label',''))}</td>"
                        f"<td>{html_escape(row.get('value',''))}</td></tr>"
                    )
                html.append("</table></div>")
            elif block_type == "DeploymentTimeline":
                html.append("<ol>")
                for entry in block.get("entries", []):
                    html.append(
                        f"<li><strong>{html_escape(entry.get('label',''))}</strong>: {html_escape(entry.get('description',''))}</li>"
                    )
                html.append("</ol>")
            elif block_type == "OwnershipTable":
                html.append("<div class=\"table-wrapper\"><table>")
                for row in block.get("rows", []):
                    html.append(
                        f"<tr><td>{html_escape(row.get('role',''))}</td>"
                        f"<td>{html_escape(row.get('team',''))}</td></tr>"
                    )
                html.append("</table></div>")
            elif block_type == "DecisionTable":
                columns = block.get("columns", [])
                html.append("<div class=\"table-wrapper\"><table>")
                html.append("<thead><tr>")
                for col in columns:
                    html.append(f"<th>{html_escape(col)}</th>")
                html.append("</tr></thead><tbody>")
                for row in block.get("rows", []):
                    html.append("<tr>")
                    for col in columns:
                        html.append(f"<td>{html_escape(str(row.get(col, '')))}</td>")
                    html.append("</tr>")
                html.append("</tbody></table></div>")
            elif block_type == "TroubleshootingBlock":
                html.append("<ol>")
                for step in block.get("steps", []):
                    html.append(f"<li>{html_escape(step)}</li>")
                html.append("</ol>")
            elif block_type == "CodeBlock":
                html.append(
                    f"<pre><code>{html_escape(block.get('code', ''))}</code></pre>"
                )
            else:
                html.append(f"<p>{html_escape(block.get('description', ''))}</p>")
            html.append("</div>")
        html.append("</section>")
    return "\n".join(html)


def load_template_environment():
    if Environment is None:
        raise RuntimeError("Jinja2 is not installed. Install with: pip install jinja2")
    template_dir = os.path.join(os.path.dirname(__file__), "pdf", "templates")
    loader = FileSystemLoader(template_dir)
    return Environment(loader=loader, autoescape=select_autoescape(["html", "xml"]))


def render_pdf_html(title: str, job_id: str, rendered_sections: list, coverage: dict, date_str: str) -> str:
    env = load_template_environment()
    template = env.get_template("kt_document.html")
    toc_sections = build_toc_sections(rendered_sections)
    content_html = render_section_blocks(rendered_sections)
    css_path = os.path.join(os.path.dirname(__file__), "pdf", "templates", "kt_document.css")
    style_css = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as css_file:
            style_css = css_file.read()

    return template.render(
        title=title,
        job_id=job_id,
        date_str=date_str,
        toc_sections=toc_sections,
        content_html=content_html,
        style_css=style_css,
    )


def _build_fallback_paragraphs(section: dict) -> list:
    seen = set()
    paragraphs = []

    def _add(text: str):
        if not isinstance(text, str):
            return
        normalized = re.sub(r"\s+", " ", text.strip()).lower()
        if not normalized:
            return
        keys = {normalized[:120]}
        if ": " in normalized:
            suffix = normalized.split(": ", 1)[1]
            keys.add(suffix[:120])
        for key in keys:
            if key in seen:
                return
        seen.update(keys)
        paragraphs.append(text.strip())

    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    for item in coverage_content:
        _add(item)

    for fact in section.get("facts", []) or []:
        value = fact.get("value")
        if isinstance(value, str) and value.strip():
            label = fact.get("label") or fact.get("id") or "Fact"
            _add(f"{label}: {value.strip()}")

    for evidence in section.get("evidence", []) or []:
        _add(evidence.get("text", ""))

    if not paragraphs and section.get("description"):
        _add(str(section["description"]))

    return paragraphs or ["Rendered content not available."]


def _has_meaningful_rendered_blocks(rendered: dict) -> bool:
    blocks = rendered.get("blocks") or []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "NarrativeBlock":
            paragraphs = [p for p in block.get("paragraphs", []) if isinstance(p, str) and p.strip()]
            if paragraphs:
                lowered = [p.lower() for p in paragraphs]
                if not any("rendered content not available" in p or "being built" in p or "synthesized" in p or "assembled" in p for p in lowered):
                    return True
        elif block_type in {"ChecklistBlock", "TechnologyGrid", "DeploymentTimeline", "OwnershipTable", "DecisionTable", "TroubleshootingBlock", "WarningBlock"}:
            return True
    return False


def build_rendered_sections(knowledge_object: dict) -> list:
    rendered_sections = []
    for section in knowledge_object.get("sections", []):
        section_id = section.get("id")
        renderer = get_renderer(section_id)
        rendered = None
        if renderer:
            rendered = renderer(section)

        if rendered and _has_meaningful_rendered_blocks(rendered):
            rendered_sections.append(rendered)
            continue

        fallback_paragraphs = _build_fallback_paragraphs(section)
        rendered_sections.append({
            "section_id": section_id,
            "section_title": section.get("title") or section_id,
            "blocks": [{
                "type": "NarrativeBlock",
                "title": section.get("title") or section_id,
                "paragraphs": fallback_paragraphs,
            }],
        })
    return rendered_sections


@app.get("/export/pdf/{job_id}")
async def export_pdf(job_id: str):
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Job not found or not completed")

    knowledge_object = job.get("knowledge_object", {}) or {}
    title = knowledge_object.get("system_name") or job.get("title") or "KT Document"
    date_str = datetime.utcnow().strftime("%d %B %Y")
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
                    "paragraphs": [
                        markdown_to_html(item) if markdown_to_html and isinstance(item, str) else html_escape(item)
                        for item in sec_info.get("content", []) if isinstance(item, str)
                    ]
                }]
            })

    html_doc = render_pdf_html(
        title=title,
        job_id=job_id,
        rendered_sections=rendered_sections,
        coverage=job.get("coverage", {}),
        date_str=date_str,
    )
    try:
        from weasyprint import HTML
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="WeasyPrint is not available. Install required native dependencies and Python packages.",
        ) from exc

    pdf_bytes = HTML(string=html_doc, base_url=os.path.dirname(__file__)).write_pdf()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=\"kt_{job_id[:8]}.pdf\""},
    )


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

        try:
            dynamic_schema = generate_dynamic_schema(
                coverage=coverage,
                base_schema=SCHEMA,
                include_missing_required=True,
            )
        except Exception as exc:
            logger.warning("Dynamic schema generation failed: %s", exc)
            dynamic_schema = SCHEMA

        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            embedding_model = None

        try:
            llm_provider = get_llm_provider()
            populated_fields = populate_fields(
                dynamic_schema=dynamic_schema,
                coverage=coverage,
                llm_provider=llm_provider,
                embedding_model=embedding_model,
            )
        except Exception as exc:
            logger.warning("Field population failed: %s", exc)
            populated_fields = {}

        try:
            knowledge_object = build_knowledge_object(
                job_id=job_id,
                coverage=coverage,
                dynamic_schema=dynamic_schema,
                populated_fields=populated_fields,
                section_content=kt.section_content,
            )
        except Exception as exc:
            logger.warning("Knowledge object build failed: %s", exc)
            knowledge_object = {}

        try:
            knowledge_object["rendered_sections"] = build_rendered_sections(knowledge_object)
        except Exception as exc:
            logger.warning("Rendered sections build failed: %s", exc)
            knowledge_object["rendered_sections"] = []

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
                "knowledge_object": knowledge_object,
                "mapped_fields": mapped_fields,
                "missing_required": missing_required,
                "progress": progress,
                "screenshots": screenshots,
                "kt_structured": kt_structured,
                "dynamic_schema": dynamic_schema,
                "populated_fields": populated_fields,
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


@app.post("/semantic-placement")
async def semantic_placement(payload: dict):
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
