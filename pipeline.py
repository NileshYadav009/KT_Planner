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
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Optional

import ffmpeg
import torch
from faster_whisper import WhisperModel

from ai import map_analysis_to_fields, polish_coverage_sections, _extract_structured_section, wrap_structured_as_fields
from context_mapper import ContextMappingPipeline
from devops_transcription import clean_transcript
from input_gate import assess_transcript
from field_populator import populate_fields, extract_rto_rpo, find_source_sentence_index
from knowledge import (
    build_knowledge_object,
    append_unmapped_findings_section,
    enrich_operational_calendar,
    enrich_technology_summary,
    enrich_architecture_knowledge,
    append_tribal_knowledge_section,
    append_coverage_matrix_section,
    append_quick_reference_section,
    reconcile_linked_fields,
    verify_document_coverage,
    apply_conflicts,
    attach_conflict_warnings,
    dedupe_rendered_sections,
    attach_elsewhere_mentions,
)
from knowledge.evidence import attach_evidence, transcript_sentences
from kt_schema_loader import SCHEMA
from llm_provider import get_llm_provider, LLM_PARALLEL_WORKERS
from llm.usage import LLMUsageTracker, start_tracking, stop_tracking
import contextvars
import media_guard
import screen_capture
from job_queue import TaskQueue
from observability import FAILURE_ERROR, FAILURE_INPUT, checkpoint
from job_store import JobStore, PersistentJobs
from auth import tenant_llm_policy
from llm.tenant_context import CURRENT_TENANT, LLM_POLICY
from redaction import redact_secrets

# Screenshots captured from shared screens, one folder per job (gitignored).
KT_ASSETS_DIR = os.getenv("KT_ASSETS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "kt_assets"))
from pdf_rendering import build_job_pdf, build_rendered_sections
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
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "true").strip().lower() not in ("0", "false", "no", "off")

MODEL = None
MAPPER_PIPELINE = None
# job_id -> {status, transcript, coverage, missing_required, progress, error};
# persisted to SQLite (job_store.py) so a restart loses nothing.
JOB_STORE = JobStore()
JOB_QUEUE = PersistentJobs(JOB_STORE)
# Submitted KTs waiting for or running on a worker (job_queue.py, worker.py).
TASKS = TaskQueue(JOB_STORE)
JOB_LOCK = Lock()
# A finished job whose document can be exported. completed_with_warnings means
# stages failed or LLM calls fell back; the warnings are shown to the reader.
COMPLETED_STATUSES = ("completed", "completed_with_warnings")


def delete_job_data(job_id: str) -> bool:
    """Remove a job's record and its captured screenshots."""
    import shutil
    if re.fullmatch(r"[0-9A-Za-z-]{8,64}", job_id or ""):
        shutil.rmtree(os.path.join(KT_ASSETS_DIR, job_id), ignore_errors=True)
    try:
        del JOB_QUEUE[job_id]
        return True
    except KeyError:
        return False


def render_and_store_document(job_id: str) -> bytes:
    """Render a finished KT's PDF and store it, so downloads are reads (P1-4).
    The stored copy is tied to the job version it was rendered from: the
    version is read first, so a change made during rendering makes it stale."""
    version = JOB_STORE.updated_at(job_id)
    job = JOB_QUEUE[job_id]
    pdf = build_job_pdf(job_id, job, JOB_STORE.created_at(job_id))
    JOB_STORE.save_document(job_id, pdf, version)
    return pdf


def apply_retention() -> int:
    """Delete jobs (and their screenshots) older than CONTINUUM_RETENTION_DAYS.
    Unset or 0 keeps everything."""
    try:
        days = float(os.getenv("CONTINUUM_RETENTION_DAYS", "0") or 0)
    except ValueError:
        days = 0
    if days <= 0:
        return 0
    import shutil
    removed = JOB_STORE.purge_older_than(days)
    for job_id in removed:
        if re.fullmatch(r"[0-9A-Za-z-]{8,64}", job_id or ""):
            shutil.rmtree(os.path.join(KT_ASSETS_DIR, job_id), ignore_errors=True)
    return len(removed)


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


class DocumentAssemblyError(RuntimeError):
    """A stage without which no usable document exists failed."""


class InputRejected(ValueError):
    """What was submitted cannot make a KT document (empty, no speech, not
    a KT). The job fails with this message; it is not a system error."""


def _stage_failed(stage_errors: list, stage: str, exc: Exception, notice: Optional[str] = None,
                  *, fatal: bool = False) -> None:
    """Record a failed pipeline stage instead of only logging it.

    `notice` is the sentence shown to the reader (UI banner and PDF) when the
    failure changes what the document contains; None for internal stages.
    A fatal stage fails the job: a document without it would look complete
    while missing its sections."""
    logger.warning("%s failed: %s", stage, exc, exc_info=True)
    stage_errors.append({"stage": stage, "error": f"{type(exc).__name__}: {exc}", "notice": notice})
    if fatal:
        raise DocumentAssemblyError(f"{stage} failed: {exc}") from exc


def _job_warnings(stage_errors: list, usage: dict, upstream: Optional[List[str]] = None) -> List[str]:
    """Reader-facing warnings for a completed job: stage failures that change
    the document, and LLM calls that failed and fell back to rules."""
    warnings = list(upstream or [])
    for err in stage_errors:
        if err.get("notice") and err["notice"] not in warnings:
            warnings.append(err["notice"])
    failed, calls = int(usage.get("failed") or 0), int(usage.get("llm_calls") or 0)
    if failed:
        warnings.append(
            f"{failed} of {calls} LLM calls failed (provider error or quota). The affected fields and "
            f"sections were filled by rules only or left empty, so some 'not covered' entries may be wrong."
        )
    return warnings


def run_kt_pipeline(job_id: str, transcript: str, segments: Optional[List[dict]] = None,
                    screen: Optional[dict] = None, warnings: Optional[List[str]] = None,
                    time_offset: float = 0.0) -> dict:
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
    # Every LLM call made for this KT (including those on the thread pools
    # below, which copy this context) is counted here. See llm/usage.py.
    usage_tracker = LLMUsageTracker(job_id)
    usage_token = start_tracking(usage_tracker)
    stage_errors: list = []

    # The tenant this job belongs to scopes the LLM cache and decides whether
    # transcript content may reach an external LLM at all (P0-9).
    tenant_id = ((JOB_QUEUE.get(job_id) or {}) if job_id in JOB_QUEUE else {}).get("tenant_id") or "local"
    llm_policy = tenant_llm_policy(tenant_id)
    tenant_token = CURRENT_TENANT.set(tenant_id)
    policy_token = LLM_POLICY.set(llm_policy)

    # Credentials never go further than this line: not to the LLM, the
    # cache, the job store or the PDF.
    transcript, redacted = redact_secrets(transcript)
    if segments is not None:
        clean_segments = []
        for seg in segments:
            text, n = redact_secrets(seg.get("text", "") if isinstance(seg, dict) else "")
            redacted += n
            clean_segments.append(dict(seg, text=text) if isinstance(seg, dict) else seg)
        segments = clean_segments
    # Whisper's segments carry the time each sentence was said; a pasted
    # transcript has none. `time_offset` is the leading silence trimmed off
    # before transcription, so source times match the original recording.
    timed_segments = segments
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
        checkpoint("Classification")

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
            else:
                # Justified secondary placements (ENABLE_MULTI_SECTION_MAPPING)
                # exist only in section_content -- topic blocks are built from
                # each sentence's primary section alone. Without this, a
                # sentence placed in a second section fed that section's
                # fields but never reached its rendered content.
                present = {s.get('text') for s in coverage_sentences}
                for s in (kt.section_content.get(sec_id, {}) or {}).get('sentences', []) or []:
                    if isinstance(s, dict) and not s.get('is_primary_section', True) and s.get('text') not in present:
                        coverage_sentences.append(dict(s, assigned_sections=s.get('assigned_sections') or [sec_id]))
                        present.add(s.get('text'))

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
            _stage_failed(stage_errors, "Text polishing", e, "Text polishing failed; sections show the speaker's original sentences.")
            polished_sections = {}
        checkpoint("Polishing")

        # Extract structured data for sections with structured prompts. Each
        # section's extraction is an independent LLM call (its own prompt,
        # its own result slot) — dispatched concurrently instead of one at a
        # time so the dominant cost (waiting out each network round trip
        # serially) doesn't stack across up to 7 sections. The provider's
        # rate throttle is shared/thread-safe, so this sends no more
        # requests per minute than running them one at a time did.
        structured_data = {}
        structured_section_ids = [
            section_id for section_id in coverage
            if section_id in (
                "monitoring_observability", "security_controls",
                "disaster_recovery", "ownership_escalation",
                "cost_optimization", "common_failures",
                "open_responsibilities",
            )  # Sections with SECTION_STRUCTURED_PROMPTS (llm/prompts.py)
        ]

        def _extract_one(section_id: str):
            section_payload = coverage[section_id]
            return _extract_structured_section(
                section_id,
                section_payload.get('title', section_id),
                section_payload.get('fragments', [])
            )

        # Only worth a thread pool when there's an LLM provider to actually
        # wait on — _extract_structured_section() no-ops immediately without
        # one, so spinning up worker threads for that would be pure
        # overhead with nothing to overlap.
        if structured_section_ids and get_llm_provider() is not None:
            with ThreadPoolExecutor(max_workers=min(LLM_PARALLEL_WORKERS, len(structured_section_ids))) as pool:
                # copy_context(): the worker threads count their calls
                # against this KT's usage tracker.
                futures = {
                    pool.submit(contextvars.copy_context().run, _extract_one, sid): sid
                    for sid in structured_section_ids
                }
                for future in as_completed(futures):
                    section_id = futures[future]
                    try:
                        structured = future.result()
                        if structured:
                            structured_data[section_id] = structured
                    except Exception as e:
                        _stage_failed(stage_errors, f"Structured extraction ({section_id})", e, f"Structured extraction failed for {coverage.get(section_id, {}).get('title', section_id)}; that section shows plain sentences instead of its table.")
        else:
            for section_id in structured_section_ids:
                try:
                    structured = _extract_one(section_id)
                    if structured:
                        structured_data[section_id] = structured
                except Exception as e:
                    _stage_failed(stage_errors, f"Structured extraction ({section_id})", e, f"Structured extraction failed for {coverage.get(section_id, {}).get('title', section_id)}; that section shows plain sentences instead of its table.")

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
            _stage_failed(stage_errors, "Dynamic schema", exc, None)
            dynamic_schema = SCHEMA

        checkpoint("Structured extraction")
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
            _stage_failed(stage_errors, "Field population", exc, "Field extraction failed; owners, RTO/RPO, environments and other fields were not filled, and the content appears as plain sentences.")
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

        # Deterministic RTO/RPO capture, independent of LLM availability (see
        # extract_rto_rpo's docstring). Writes to its OWN rto_metric/
        # rpo_metric field ids, not rto_steps/rpo_steps -- those already
        # belong to the LLM prompt's recovery-PROCEDURE narrative ("restore
        # from backups", "recreate infrastructure"), a different fact from
        # the duration metric. Confirmed live that writing the metric into
        # that same slot silently destroyed a real procedure the LLM had
        # captured. Runs after the structured-LLM merge above but never
        # collides with it, since the field ids don't overlap.
        # `coverage["disaster_recovery"]["sentences"]` (built above, lines
        # ~124-155) is the authoritative source, already carrying its own
        # blocks/sentences/section_content fallback chain -- kt.section_content
        # directly is a DIFFERENT, less reliable structure (see
        # knowledge_builder._collect_evidence()'s own docstring on the same
        # disagreement); confirmed live that section_content left this text
        # unreachable even via its own ['blocks'] fallback, silently starving
        # this extractor of any text.
        dr_sentences = coverage.get("disaster_recovery", {}).get("sentences", [])
        dr_sentence_texts = [s.get("text", "") for s in dr_sentences if isinstance(s, dict)]
        dr_text = " ".join(dr_sentence_texts)
        rto_rpo = extract_rto_rpo(dr_text)
        if rto_rpo:
            # knowledge_builder._collect_evidence() resolves source_chunk_index
            # against kt.section_content, NOT this coverage-based list -- the
            # two can disagree in both content AND length (confirmed live: an
            # index valid here landed out of range there and silently fell
            # back to evidence for a different sentence). Search the SAME
            # sentence list _collect_evidence() will actually index into, so
            # a set index is guaranteed to resolve to the right sentence.
            dr_section_content = kt.section_content.get("disaster_recovery", {}) or {}
            dr_evidence_sentences = dr_section_content.get("sentences") or [
                s for block in (dr_section_content.get("blocks") or [])
                for s in (block.get("sentences") or [])
            ]
            dr_evidence_texts = [s.get("text", "") for s in dr_evidence_sentences if isinstance(s, dict)]

            dr_fields = populated_fields.setdefault("disaster_recovery", {})
            for field_id, value in rto_rpo.items():
                entry = {"value": value, "confidence": 0.90, "source": "pattern"}
                idx = find_source_sentence_index(value, dr_evidence_texts)
                if idx is not None:
                    entry["source_chunk_index"] = idx
                dr_fields[field_id] = entry

        checkpoint("Field population")
        try:
            knowledge_object = build_knowledge_object(
                job_id=job_id,
                coverage=coverage,
                dynamic_schema=dynamic_schema,
                populated_fields=populated_fields,
                section_content=kt.section_content,
            )
        except Exception as exc:
            _stage_failed(stage_errors, "Document assembly", exc, fatal=True)

        try:
            knowledge_object = reconcile_linked_fields(knowledge_object)
        except Exception as exc:
            _stage_failed(stage_errors, "Linked fields", exc, None)

        # Fold cost_optimization's levers into known_bad_days as a combined
        # "Operational Calendar" (peak periods + cost-related patterns) —
        # reads a sibling section's already-built data, no reclassification.
        try:
            knowledge_object = enrich_operational_calendar(knowledge_object)
        except Exception as exc:
            _stage_failed(stage_errors, "Operational calendar", exc, "The Operational Calendar summary could not be built.")

        # Fold tool mentions correctly classified into their own dedicated
        # sections (Monitoring's tools, Security's scanner) into System
        # Overview's Technology Summary so it's a genuine one-stop
        # inventory, not just whatever happened to be mentioned in the same
        # sentences as the rest of the overview narrative.
        # Architecture Reference's own field schema is 3 admin facts (doc
        # link, last-updated, verified-by) — real architecture knowledge
        # (the actual components/services the system runs on) lives in
        # whichever section's sentences happened to mention it. Surface it
        # as its own digest so the section isn't judged solely by whether
        # those 3 admin fields were discussed.
        #
        # Runs BEFORE the technology summary so that summary can reuse the
        # component inventory this builds: without that ordering the summary
        # only saw whatever tools appeared in system_overview's own
        # sentences, and a real Azure KT rendered a "Technology summary"
        # missing its compute, database, cache, messaging and ingress tiers
        # even though all of them were in the component list right above it.
        try:
            knowledge_object = enrich_architecture_knowledge(knowledge_object, kt.section_content)
        except Exception as exc:
            _stage_failed(stage_errors, "Architecture summary", exc, "The architecture summary and diagram could not be built.")

        try:
            knowledge_object = enrich_technology_summary(knowledge_object)
        except Exception as exc:
            _stage_failed(stage_errors, "Technology summary", exc, "The technology summary may be incomplete.")

        # Corrections and contradictions the speaker made: replaced tools are
        # swapped for the current ones, and contradictory rules are kept for
        # the reader to confirm (conflicts.py).
        try:
            knowledge_object = apply_conflicts(knowledge_object, coverage)
        except Exception as exc:
            _stage_failed(stage_errors, "Conflict detection", exc,
                          "Corrections and contradictions in the session could not be checked; review single-valued facts (RTO, owners, tools).")

        # Surface sentences the classifier never confidently placed in any
        # real section instead of letting them vanish silently (see
        # knowledge_builder.append_unmapped_findings_section docstring).
        try:
            # No transcript: the pre-render word-bag safety net is superseded
            # by verify_document_coverage() below, which checks the rendered
            # document and recognises a section's paraphrase of its own
            # sentence. The word-bag net counted an LLM-polished danger zone
            # as lost and published the same warning twice.
            knowledge_object = append_unmapped_findings_section(
                knowledge_object, kt.unassigned_sentences
            )
        except Exception as exc:
            _stage_failed(stage_errors, "Additional notes", exc, "Sentences that fit no section could not be listed under Additional Notes.")

        # Cross-section digests synthesized from data the pipeline has
        # already produced above — no new classification, just reshaping.
        try:
            knowledge_object = append_tribal_knowledge_section(knowledge_object, kt.section_content)
        except Exception as exc:
            _stage_failed(stage_errors, "Tribal knowledge", exc, None)

        try:
            knowledge_object = append_coverage_matrix_section(knowledge_object, coverage, dynamic_schema)
        except Exception as exc:
            _stage_failed(stage_errors, "Coverage summary", exc, "The KT coverage summary could not be built.")

        try:
            knowledge_object = append_quick_reference_section(knowledge_object)
        except Exception as exc:
            _stage_failed(stage_errors, "Quick reference", exc, "The quick reference page could not be built.")

        checkpoint("Document assembly")
        try:
            knowledge_object["rendered_sections"] = build_rendered_sections(knowledge_object)
            attach_conflict_warnings(knowledge_object["rendered_sections"], knowledge_object)
        except Exception as exc:
            _stage_failed(stage_errors, "Document rendering", exc, fatal=True)

        checkpoint("Rendering")

        # One home per fact: a sentence already shown is not printed again,
        # and the Tribal Knowledge digest points to where it is (P1-10).
        try:
            knowledge_object["_dedup"] = dedupe_rendered_sections(knowledge_object["rendered_sections"])
        except Exception as exc:
            _stage_failed(stage_errors, "Duplicate removal", exc, None)
        try:
            attach_elsewhere_mentions(knowledge_object["rendered_sections"], knowledge_object)
        except Exception as exc:
            _stage_failed(stage_errors, "Cross-section pointers", exc, None)

        # Dashboards and links shown on the shared screen (screen_capture.py),
        # placed in the section that was being discussed at the time.
        screen_assets = list((screen or {}).get("assets") or [])
        if screen_assets:
            try:
                screen_capture.assign_sections(screen_assets, coverage, (screen or {}).get("transcript_offset", 0.0))
                screen_capture.attach_to_rendered_sections(knowledge_object["rendered_sections"], screen_assets, job_id)
            except Exception as exc:
                _stage_failed(stage_errors, "Screen captures", exc, "Captured dashboards and links could not be placed in the document.")

        # Final completeness check against the RENDERED document: every
        # fact-bearing transcript sentence must appear somewhere in it, or be
        # added to Additional Notes. See verify_document_coverage().
        try:
            sentence_sections = {
                re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", (sent.get("text") or "").lower())).strip(): sid
                for sid, cov in coverage.items()
                for sent in (cov.get("sentences") or [])
                if isinstance(sent, dict)
            }
            knowledge_object = verify_document_coverage(
                knowledge_object, [s.text for s in (kt.sentences or [])], sentence_sections
            )
        except Exception as exc:
            _stage_failed(stage_errors, "Completeness check", exc, "The completeness check did not run, so some transcript sentences may be missing from this document.")

        # Evidence per fact (P1-2): each fact in the document points to the
        # transcript sentences it came from, and when they were said.
        try:
            attach_evidence(knowledge_object, transcript_sentences(transcript, timed_segments, time_offset))
        except Exception as exc:
            _stage_failed(stage_errors, "Source links", exc,
                          "Facts could not be linked to the transcript sentences they came from.")

        checkpoint("Post-processing")

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
            _stage_failed(stage_errors, "Validation", exc, None)
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
            _stage_failed(stage_errors, "Quality score", exc, None)
            quality_score = {}

        # Learn new DevOps vocabulary from this transcript. Detection only ever
        # records candidates for later human review (see vocabulary_learning.py) —
        # it never changes what clean_transcript() corrects on its own.
        try:
            candidates = detect_vocabulary_candidates(transcript)
            if candidates:
                record_candidates(candidates, job_id)
        except Exception as exc:
            _stage_failed(stage_errors, "Vocabulary learning", exc, None)

        checkpoint("Validation and scoring")
        progress = int(round(kt.overall_coverage_percent or 0))
        transcript = kt.transcript

        JOB_QUEUE.set_progress(job_id, 60)

        # Dashboards and links captured from the shared screen (empty for a
        # pasted transcript or an audio-only upload).
        section_titles = {sid: (cov or {}).get("title", sid) for sid, cov in coverage.items()}
        screen_view = screen_capture.ui_payload(screen_assets, job_id, section_titles)
        screenshots = screen_view["screenshots"]

        JOB_QUEUE.set_progress(job_id, 85)

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
            "screen_links": screen_view["links"],
            "screen_capture": (screen or {}).get("stats"),
            "dynamic_schema": dynamic_schema,
            "populated_fields": populated_fields,
            "validation_warnings": validation_warnings,
            "quality_score": quality_score,
            "error": None
        }
    except Exception as e:
        logger.exception("KT pipeline failed for %s", job_id)
        result = {
            "status": "failed",
            "error": str(e)
        }
    finally:
        stop_tracking(usage_token)
        LLM_POLICY.reset(policy_token)
        CURRENT_TENANT.reset(tenant_token)
    result["llm_usage"] = usage_tracker.summary()
    result["stage_errors"] = stage_errors

    # A document that was built with failed stages or failed LLM calls must
    # say so: it can look complete while missing what those stages produce.
    if result["status"] == "completed":
        job_warnings = _job_warnings(stage_errors, result["llm_usage"], warnings)
        result["warnings"] = job_warnings
        result["notices"] = []
        if llm_policy == "none":
            result["notices"].append("External LLM disabled for this workspace: rules and local models only.")
        elif get_llm_provider() is None:
            result["notices"].append("Generated without an LLM: rules and local models only.")
        if redacted:
            result["notices"].append(f"{redacted} credential(s) found in the transcript were redacted before processing.")
        rejected = int(result["llm_usage"].get("rejected") or 0)
        if rejected:
            result["notices"].append(
                f"{rejected} LLM-generated value(s) were discarded because the transcript does not "
                f"support them; those fields are left empty rather than guessed."
            )
        if job_warnings:
            result["status"] = "completed_with_warnings"

    with JOB_LOCK:
        JOB_QUEUE[job_id] = result
    return result


def trim_leading_trailing_silence(input_path: str, output_path: str) -> bool:
    """Trim silence from the start and end of an audio file only.

    NOTE: a single forward `silenceremove` pass with `stop_periods` set does
    NOT trim trailing silence -- it stops the whole filter output at the
    FIRST silence gap found anywhere in the stream and discards everything
    after it. Any real recording with a normal pause between sentences would
    have most of its audio silently dropped before Whisper ever saw it.
    Trimming trailing silence safely requires reversing the stream, trimming
    leading silence again, then reversing back -- never `stop_periods`.

    Returns True and writes `output_path` on success, False otherwise.
    """
    (
        ffmpeg.input(input_path)
        .filter_('silenceremove', start_periods=1, start_silence=0.5, start_threshold='-50dB')
        .filter_('areverse')
        .filter_('silenceremove', start_periods=1, start_silence=0.5, start_threshold='-50dB')
        .filter_('areverse')
        .output(output_path)
        .run(quiet=True, overwrite_output=True)
    )
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def _extract_audio(input_path: str, media_format: Optional[str]) -> str:
    """The upload's audio as 16 kHz mono WAV. Only this file, which ffmpeg
    wrote, reaches the Whisper decoder; the upload itself never does."""
    return media_guard.extract_audio(input_path, media_format, f"{input_path}.wav")


def process_upload_task(job_id: str, input_path: str, audio_path: str, media_format: Optional[str] = None):
    """Background task for transcription and classification from an uploaded
    recording. The route has already checked it (media_guard.probe)."""
    try:
        if os.path.getsize(input_path) == 0:
            raise InputRejected("Uploaded file is empty.")

        try:
            audio_to_use = _extract_audio(input_path, media_format)
        except media_guard.MediaRejected as exc:
            raise InputRejected(exc.message) from exc

        # Trim only leading/trailing silence to speed up transcription.
        time_offset = 0.0
        try:
            trimmed_path = f"{input_path}.trimmed.wav"
            if trim_leading_trailing_silence(audio_to_use, trimmed_path):
                # Transcript times now start after the cut silence; sources
                # add it back so they point into the original recording.
                time_offset = screen_capture.leading_silence_seconds(audio_to_use)
                audio_to_use = trimmed_path
        except Exception:
            # If trimming fails, continue with original audio
            pass

        # Transcribe using faster-whisper with faster settings by default.
        # On CPU we prefer the small model, while GPU can use medium when configured.
        transcribe_kwargs = {
            "language": "en",
            "beam_size": DEFAULT_WHISPER_BEAM_SIZE,
            "task": "transcribe",
            # Skip non-speech (faster-whisper's built-in Silero VAD). On a
            # real KT recording with meeting pauses, Whisper without it
            # filled the silences with text nobody said (an invented
            # "deployment window is outside peak business hours" and a
            # "rollback time" repetition loop); with it, no invented text and
            # 27% faster. Continuous speech transcribes identically.
            "vad_filter": WHISPER_VAD_FILTER,
        }
        checkpoint("Audio extraction")
        segments, info = MODEL.transcribe(audio_to_use, **transcribe_kwargs)
        segments = list(segments)

        # Clean each segment once and build the joined transcript from cleaned parts
        cleaned_segments = []
        raw_parts = []
        for s in segments:
            cleaned_text, _ = redact_secrets(clean_transcript(s.text))
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
        JOB_QUEUE.set_progress(job_id, 30)

        checkpoint("Transcription")
        if not transcript:
            raise InputRejected("No speech detected in the uploaded file.")

        # Refuse recordings that cannot produce a KT document (an empty
        # meeting, a few pleasantries, a stand-up uploaded by mistake); carry
        # softer findings to the document as warnings.
        upstream_warnings: List[str] = []
        gate = assess_transcript(transcript, source="audio")
        if gate["verdict"] == "reject":
            raise InputRejected("The recording does not contain enough KT content to build a document. "
                             + " ".join(gate["reasons"]))
        if gate["verdict"] == "warn":
            upstream_warnings.extend(gate["reasons"])

        # A recorded meeting with a screen share: capture the dashboards and
        # links that were shown and discussed (screen_capture.py). Runs here
        # because the uploaded file is deleted when this task ends. Never
        # fails the KT: on any error the document is built without captures.
        screen = None
        if screen_capture.enabled():
            try:
                JOB_QUEUE.set_progress(job_id, 35)
                offset = screen_capture.leading_silence_seconds(input_path)
                screen = screen_capture.analyze_screen_shares(
                    input_path, result.get('segments', []), os.path.join(KT_ASSETS_DIR, job_id),
                    transcript_offset=offset,
                )
                screen["transcript_offset"] = offset
            except Exception as exc:
                logger.warning("Screen capture failed: %s", exc, exc_info=True)
                upstream_warnings.append("Screen capture failed; dashboards and links shown on screen were not captured.")

        if screen is not None:
            checkpoint("Screen capture")
        run_kt_pipeline(job_id, transcript, result.get('segments', []), screen=screen, warnings=upstream_warnings,
                        time_offset=time_offset)
    except Exception as e:
        if not isinstance(e, InputRejected):
            logger.exception("Upload processing failed for %s", job_id)
        with JOB_LOCK:
            JOB_QUEUE[job_id] = {
                "status": "failed",
                "error": str(e),
                # What was uploaded cannot make a KT: the user is told why,
                # and nobody is paged (observability.alert_on_run).
                "failure": FAILURE_INPUT if isinstance(e, InputRejected) else FAILURE_ERROR,
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
