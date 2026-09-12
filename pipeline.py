"""Upload orchestration and job state.

Owns the in-memory job queue and the background task that runs transcription,
classification, knowledge extraction, and rendering for one uploaded file.
Moved out of main.py as part of the Phase 3 architecture split (see
REPOSITORY_AUDIT.md).

MODEL and MAPPER_PIPELINE are reassigned wholesale inside load_models() (not
mutated in place), so anything needing them must go through this module's
functions/attributes rather than `from pipeline import MODEL`, which would
capture a stale None at import time. Nothing outside this module needs direct
access to them — main.py's startup handler just calls load_models().
"""

import logging
import os
from threading import Lock
from typing import List, Optional

import ffmpeg
import torch
from faster_whisper import WhisperModel

from ai import map_analysis_to_fields, polish_coverage_sections, _extract_structured_section, wrap_structured_as_fields
from context_mapper import ContextMappingPipeline
from devops_transcription import clean_transcript
from field_populator import populate_fields
from knowledge import (
    build_knowledge_object,
    append_unmapped_findings_section,
    enrich_operational_calendar,
    enrich_technology_summary,
    append_tribal_knowledge_section,
    append_coverage_matrix_section,
    append_quick_reference_section,
)
from kt_schema_loader import SCHEMA
from llm_provider import get_llm_provider
from pdf_rendering import build_rendered_sections
from quality_score import compute_quality_score
from renderers.sections import validate_renderer_registry
from schema_generator import generate_dynamic_schema
from validation import validate_pipeline_run
from vocabulary_learning import detect_vocabulary_candidates, record_candidates

logger = logging.getLogger(__name__)

# Optional environment config for Whisper model
DEFAULT_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
DEFAULT_WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
DEFAULT_WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "2"))

MODEL = None
MAPPER_PIPELINE = None
JOB_QUEUE = {}  # job_id -> {status, transcript, coverage, missing_required, progress, error}
JOB_LOCK = Lock()


def load_models():
    """Load heavy models on startup so endpoints can use them."""
    global MODEL, MAPPER_PIPELINE

    # Validate renderer registry before anything else
    validate_renderer_registry(SCHEMA)

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


def run_kt_pipeline(job_id: str, transcript: str, segments: Optional[List[dict]] = None) -> dict:
    """Run classification through rendering for an already-transcribed, already-cleaned
    transcript, and store the result in JOB_QUEUE.

    Shared by process_upload_task (audio path) and the /kt-from-transcript endpoint
    (text-only test path) — this is everything that doesn't depend on audio.

    `segments` should be Whisper-shaped dicts (id/seek/start/end/text/avg_logprob/
    compression_ratio/no_speech_prob). If omitted, a single segment spanning the
    whole transcript is synthesized so MAPPER_PIPELINE.process still gets its
    expected shape.

    Always writes either a "completed" or "failed" result into JOB_QUEUE[job_id]
    and returns that same dict — safe to hand directly to BackgroundTasks.
    """
    try:
        if segments is None:
            segments = [{
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 0.0,
                "text": transcript,
                "avg_logprob": -0.3,
                "compression_ratio": None,
                "no_speech_prob": None,
            }]

        # Process transcript with the full 7-stage context mapping pipeline
        if MAPPER_PIPELINE is None:
            raise RuntimeError("Context mapping pipeline not initialized")

        kt = MAPPER_PIPELINE.process(job_id, transcript, segments)

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

        # Extract structured data for sections with structured prompts
        structured_data = {}
        for section_id, section_payload in coverage.items():
            if section_id in (
                "monitoring_observability", "security_controls",
                "disaster_recovery", "ownership_escalation",
                "cost_optimization", "common_failures",
                "open_responsibilities",
            ):  # Sections with SECTION_STRUCTURED_PROMPTS (llm/prompts.py)
                try:
                    structured = _extract_structured_section(
                        section_id,
                        section_payload.get('title', section_id),
                        section_payload.get('fragments', [])
                    )
                    if structured:
                        structured_data[section_id] = structured
                except Exception as e:
                    logger.warning("Structured extraction failed for %s: %s", section_id, e)

        for sid, section_payload in coverage.items():
            display_content = polished_sections.get(sid)
            if not display_content:
                display_content = section_payload['fragments']
            coverage[sid]['content'] = display_content
            # Store structured data if available
            if sid in structured_data:
                coverage[sid]['_structured'] = structured_data[sid]
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
            # Reuse the classification stage's own embedding model
            # (BAAI/bge-large-en-v1.5 — a materially stronger model than the
            # small all-MiniLM-L6-v2 this used to load separately) for
            # field_populator.py's semantic field-matching too, instead of
            # loading a second, weaker model. Same model, already resident
            # in memory for this job — no extra download/load cost, and a
            # stronger embedding space directly helps the exact class of
            # error observed live: near-synonymous fields (e.g.
            # system_overview's business_impact/worst_case/what_breaks) or
            # near-synonymous sections competing for the same sentence are
            # disambiguated by embedding similarity, so a better embedding
            # model is the highest-leverage single lever available here.
            embedding_model = getattr(getattr(MAPPER_PIPELINE, "classifier", None), "model", None)
            if embedding_model is None:
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
                section_content=kt.section_content,
            )
        except Exception as exc:
            logger.warning("Field population failed: %s", exc)
            populated_fields = {}

        # Merge structured-extraction data into populated_fields using the exact
        # field ids the section's renderer (and knowledge_builder's facts/entities/
        # relationships) already expect. These sections have no "fields" array in
        # the schema, so populate_fields() above never produces anything for them —
        # this is additive, not an overwrite of real field-populator output.
        #
        # Restricted to the flat-scalar-field sections only. cost_optimization/
        # common_failures return a *list* of records (levers/failures), which
        # doesn't fit wrap_structured_as_fields()'s {field_id: value} shape — they
        # stay on the `_structured` passthrough (set above) and are read directly
        # by their renderers instead (see renderers/sections/cost_optimization.py,
        # common_failures.py).
        FLAT_STRUCTURED_SECTIONS = {
            "monitoring_observability", "security_controls",
            "disaster_recovery", "ownership_escalation",
        }
        for section_id, structured in structured_data.items():
            if section_id not in FLAT_STRUCTURED_SECTIONS:
                continue
            raw_sentences = (kt.section_content.get(section_id, {}) or {}).get("sentences", [])
            raw_sentence_texts = [s.get("text", "") for s in raw_sentences if isinstance(s, dict)]
            fields_from_structured = wrap_structured_as_fields(structured, raw_sentence_texts)
            if fields_from_structured:
                populated_fields.setdefault(section_id, {}).update(fields_from_structured)

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

        # Fold cost_optimization's levers into known_bad_days as a combined
        # "Operational Calendar" (peak periods + cost-related patterns) —
        # reads a sibling section's already-built data, no reclassification.
        try:
            knowledge_object = enrich_operational_calendar(knowledge_object)
        except Exception as exc:
            logger.warning("Operational calendar enrichment failed: %s", exc)

        # Fold tool mentions correctly classified into their own dedicated
        # sections (Monitoring's tools, Security's scanner) into System
        # Overview's Technology Summary so it's a genuine one-stop
        # inventory, not just whatever happened to be mentioned in the same
        # sentences as the rest of the overview narrative.
        try:
            knowledge_object = enrich_technology_summary(knowledge_object)
        except Exception as exc:
            logger.warning("Technology summary enrichment failed: %s", exc)

        # Surface sentences the classifier never confidently placed in any
        # real section instead of letting them vanish silently (see
        # knowledge_builder.append_unmapped_findings_section docstring).
        try:
            knowledge_object = append_unmapped_findings_section(
                knowledge_object, kt.unassigned_sentences
            )
        except Exception as exc:
            logger.warning("Unmapped findings appendix failed: %s", exc)

        # Cross-section digests synthesized from data the pipeline has
        # already produced above — no new classification, just reshaping.
        try:
            knowledge_object = append_tribal_knowledge_section(knowledge_object, kt.section_content)
        except Exception as exc:
            logger.warning("Tribal knowledge digest failed: %s", exc)

        try:
            knowledge_object = append_coverage_matrix_section(knowledge_object, coverage, dynamic_schema)
        except Exception as exc:
            logger.warning("Coverage matrix digest failed: %s", exc)

        try:
            knowledge_object = append_quick_reference_section(knowledge_object)
        except Exception as exc:
            logger.warning("Quick reference digest failed: %s", exc)

        try:
            knowledge_object["rendered_sections"] = build_rendered_sections(knowledge_object)
        except Exception as exc:
            logger.warning("Rendered sections build failed: %s", exc)
            knowledge_object["rendered_sections"] = []

        # Non-fatal structural validation (schema/field-id consistency,
        # expected shapes/ranges) — see validation.py. Never blocks the
        # pipeline; just surfaces the kind of silent id-mismatch bug this
        # session repeatedly found by hand (REPOSITORY_AUDIT.md §9l/9n) so
        # future ones show up in logs/API output instead of a blank PDF section.
        try:
            validation_warnings = validate_pipeline_run(
                knowledge_object=knowledge_object,
                populated_fields=populated_fields,
                dynamic_schema=dynamic_schema,
            )
            for warning in validation_warnings:
                logger.warning("[validation] %s: %s", job_id, warning)
        except Exception as exc:
            logger.warning("Validation itself failed: %s", exc)
            validation_warnings = []

        # Document-level quality score — aggregates the per-section
        # coverage/confidence/risk that already exist plus the validation
        # warnings above into one number. Deliberately separate from
        # kt.overall_coverage_percent (used below as job "progress"), which
        # is a raw coverage percentage with no notion of required-vs-optional
        # weighting or structural validity. See quality_score.py.
        try:
            quality_score = compute_quality_score(
                coverage=coverage,
                dynamic_schema=dynamic_schema,
                validation_warnings=validation_warnings,
            )
        except Exception as exc:
            logger.warning("Quality score computation failed: %s", exc)
            quality_score = {}

        # Learn new DevOps vocabulary from this transcript. Detection only ever
        # records candidates for later human review (see vocabulary_learning.py) —
        # it never changes what clean_transcript() corrects on its own.
        try:
            candidates = detect_vocabulary_candidates(transcript)
            if candidates:
                record_candidates(candidates, job_id)
        except Exception as exc:
            logger.warning("Vocabulary candidate detection failed: %s", exc)

        progress = int(round(kt.overall_coverage_percent or 0))
        transcript = kt.transcript

        with JOB_LOCK:
            if job_id in JOB_QUEUE:
                JOB_QUEUE[job_id]["progress"] = 60

        # Screenshot capture is disabled (see REPOSITORY_AUDIT.md §4.3 for the removed
        # implementation and why); kept as an empty list since the frontend reads
        # job.screenshots unconditionally.
        screenshots = []

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

        result = {
            "status": "completed",
            "transcript": transcript,
            "coverage": coverage,
            "knowledge_object": knowledge_object,
            "mapped_fields": mapped_fields,
            "missing_required": missing_required,
            "progress": progress,
            "screenshots": screenshots,
            "dynamic_schema": dynamic_schema,
            "populated_fields": populated_fields,
            "validation_warnings": validation_warnings,
            "quality_score": quality_score,
            "error": None
        }
    except Exception as e:
        result = {
            "status": "failed",
            "error": str(e)
        }

    with JOB_LOCK:
        JOB_QUEUE[job_id] = result
    return result


def process_upload_task(job_id: str, input_path: str, audio_path: str):
    """Background task for transcription and classification from an uploaded audio file."""
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

        run_kt_pipeline(job_id, transcript, result.get('segments', []))
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
