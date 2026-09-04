import hashlib
import json
import re
import numpy as np
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    class _Util:
        @staticmethod
        def cos_sim(a, b):
            a = np.asarray(a)
            b = np.asarray(b)
            a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
            b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
            return np.dot(a_norm, b_norm.T)
    util = _Util()

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import time
import os

# Load environment variables from a local .env file if present, so Gemini/Groq
# config works without manually exporting env vars before each run. Must run
# BEFORE importing llm_provider (or anything else that reads os.getenv for
# config at import time) below — llm_provider.py now also self-loads .env as
# the real fix, this ordering here is defense in depth for this module too.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv is optional; if it's missing we simply rely on the real env.
    pass

from devops_transcription import clean_transcript
from context_mapper import AudioSegment, ContextClassifier, segment_sentences
from enterprise_semantic_mapper import create_semantic_mapper
from llm_provider import get_llm_provider
import requests

logger = logging.getLogger(__name__)

# ============================================================================
# GEMINI LLM SETUP
# ============================================================================

# Gemini config (Google Generative AI)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_CLIENT = None
GEMINI_ENABLED = False
GEMINI_CACHE: Dict[str, str] = {}
try:
    from google import genai
    if GEMINI_API_KEY:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_ENABLED = True
except Exception:
    GEMINI_CLIENT = None
    GEMINI_ENABLED = False


def _parse_duration_from_string(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        # Accept values like '19s', '19.7s', '20', '20 sec', '20 seconds'
        match = re.search(r"(\d+(?:\.\d+)?)(?:\s*)(s|sec|seconds)?", str(value), re.IGNORECASE)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def _extract_retry_delay_from_error(error: Exception) -> Optional[float]:
    # Prefer explicit Retry-After header from HTTP responses.
    response = getattr(error, 'response', None)
    if response is not None:
        try:
            if hasattr(response, 'headers'):
                retry_after = response.headers.get('Retry-After')
                delay = _parse_duration_from_string(retry_after)
                if delay is not None:
                    return delay
        except Exception:
            pass
        try:
            if hasattr(response, 'json'):
                body = response.json()
                body_text = json.dumps(body)
                delay = _parse_duration_from_string(body_text)
                if delay is not None:
                    return delay
        except Exception:
            pass
    # Fall back to scanning the exception text.
    return _parse_duration_from_string(getattr(error, 'message', None) or str(error))


def _extract_json_response(text: str):
    payload = text.strip()
    # Strip markdown fences if present
    payload = re.sub(r'^```(?:json)?\s*', '', payload, flags=re.IGNORECASE)
    payload = re.sub(r'```$', '', payload, flags=re.IGNORECASE)
    payload = payload.strip()

    # Try direct load first
    try:
        return json.loads(payload)
    except Exception:
        pass

    # Find the first JSON array/object in the text body
    for pattern in [r'(\[.*\])', r'(\{.*\})']:
        match = re.search(pattern, payload, flags=re.DOTALL)
        if match:
            candidate = match.group(1)
            try:
                return json.loads(candidate)
            except Exception:
                continue
    # Last resort: try loading any JSON-looking substring by braces count
    return None


def _local_cleanup(text: str) -> str:
    text = ' '.join(text.split())
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in '.!?':
        text += '.'
    return text.strip()


def gemini_refiner(prompt: str, metadata: Optional[dict] = None) -> str:
    """Refine text using Google Gemini via the genai SDK or direct HTTP with retry logic."""
    if not (GEMINI_API_KEY or GEMINI_CLIENT):
        raise RuntimeError("Gemini client not configured. Set GEMINI_API_KEY to enable.")

    cache_key = hashlib.sha256(
        (prompt + json.dumps(metadata or {}, sort_keys=True)).encode('utf-8')
    ).hexdigest()
    if cache_key in GEMINI_CACHE:
        logger.debug("Gemini cache hit for prompt key %s", cache_key)
        return GEMINI_CACHE[cache_key]

    max_retries = 3
    retry_delay = 1.0
    last_exception = None

    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()
            if GEMINI_CLIENT:
                response = GEMINI_CLIENT.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=1024,
                        stop_sequences=["\n\n"]
                    )
                )
                text = getattr(response, "text", None) or str(response)
            else:
                url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
                payload = {
                    "prompt": {"text": prompt},
                    "temperature": 0.2,
                    "maxOutputTokens": 1024,
                    "stop_sequences": ["\n\n"]
                }
                r = requests.post(url, json=payload, timeout=(10, 120))
                try:
                    r.raise_for_status()
                except Exception as err:
                    raise err
                data = r.json()
                text = ""
                if isinstance(data, dict):
                    if "candidates" in data and data["candidates"]:
                        parts = []
                        for c in data["candidates"]:
                            if isinstance(c, dict):
                                parts.append(c.get("content") or c.get("output") or c.get("text", ""))
                            else:
                                parts.append(str(c))
                        text = "\n".join([p for p in parts if p])
                    elif "output" in data:
                        text = data.get("output")
                    elif "response" in data and isinstance(data.get("response"), dict):
                        text = data.get("response", {}).get("output", "") or json.dumps(data.get("response", {}))
                    else:
                        text = data.get("text") or json.dumps(data)
                else:
                    text = str(data)

            elapsed = time.perf_counter() - start_time
            logger.debug("Gemini request duration: %.2fs", elapsed)

            text = (text or "").strip()
            text = re.sub(r'^(\n|\r|\s)+', '', text)
            if not text:
                raise ValueError("Gemini returned no completion text")

            GEMINI_CACHE[cache_key] = text
            return text
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                retry_delay_override = _extract_retry_delay_from_error(e)
                if retry_delay_override is not None:
                    retry_delay = max(0.5, retry_delay_override + 0.5)
                logger.warning(
                    "Gemini attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    e,
                    retry_delay,
                )
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60.0)
            else:
                logger.warning("Gemini all %d retries exhausted. Last error: %s", max_retries, e)

    raise last_exception


# Track if we've warmed up the Gemini client
_gemini_warmed_up = False


def warmup_models():
    """Warm up Gemini if enabled."""
    global _gemini_warmed_up

    if GEMINI_ENABLED and not _gemini_warmed_up:
        try:
            logger.info("Warming up Gemini model...")
            gemini_refiner("hello")
            _gemini_warmed_up = True
            logger.info("Gemini warmup complete")
        except Exception as e:
            logger.warning("Gemini warmup failed (non-critical): %s", e)
            _gemini_warmed_up = True



from kt_schema_loader import SCHEMA

SENT_MODEL: Optional[SentenceTransformer] = None
SECTION_EMBEDS: Optional[Dict[str, np.ndarray]] = None
CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence for auto-classification

def _build_section_hints():
    hints_map = {}
    def collect_field_text(field):
        texts = []
        if isinstance(field, dict):
            if field.get("label"):
                texts.append(field["label"])
            if field.get("id"):
                texts.append(field["id"].replace('_', ' '))
            if field.get("options"):
                texts.extend([str(o) for o in field.get("options")])
            if field.get("columns"):
                texts.extend(field.get("columns"))
            if field.get("fields"):
                for sub in field.get("fields"):
                    texts.extend(collect_field_text(sub))
        return texts

    for s in SCHEMA:
        items = set()
        for h in s.get("hints", []):
            items.add(h.lower())
        for field in s.get("fields", []) or []:
            for t in collect_field_text(field):
                items.add(t.lower())
        hints_map[s["id"]] = items
    return hints_map

SECTION_HINTS = _build_section_hints()

WORD_RE = re.compile(r"\b\w+\b")

def get_sentence_model(model_name: str = "all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:
    global SENT_MODEL
    if SENT_MODEL is None:
        try:
            SENT_MODEL = SentenceTransformer(model_name)
        except Exception:
            SENT_MODEL = None
    return SENT_MODEL

def get_section_embeds() -> Dict[str, np.ndarray]:
    """Compute and cache embeddings for each section (title + hints).

    Returns a dict mapping section id -> numpy embedding array.
    """
    global SECTION_EMBEDS
    if SECTION_EMBEDS is not None:
        return SECTION_EMBEDS

    model = get_sentence_model()
    embeds = {}
    for s in SCHEMA:
        parts = [s.get("title", "")]
        parts.extend(s.get("hints", []))

        # include field labels, option values and column names to improve hint coverage
        def collect_field_text(field):
            texts = []
            if isinstance(field, dict):
                if field.get("label"):
                    texts.append(field["label"])
                if field.get("id"):
                    texts.append(field["id"].replace('_', ' '))
                if field.get("options"):
                    texts.extend([str(o) for o in field.get("options")])
                if field.get("columns"):
                    texts.extend(field.get("columns"))
                if field.get("fields"):
                    for sub in field.get("fields"):
                        texts.extend(collect_field_text(sub))
            return texts

        for field in s.get("fields", []) or []:
            parts.extend(collect_field_text(field))

        text = "\n".join([p for p in parts if p])
        if model is not None:
            vec = model.encode(text, convert_to_tensor=False, normalize_embeddings=True)
            embeds[s["id"]] = np.asarray(vec, dtype=np.float32)
        else:
            v = np.random.rand(384).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            embeds[s["id"]] = v

    SECTION_EMBEDS = embeds
    return SECTION_EMBEDS

CONTEXT_CLASSIFIER: Optional[ContextClassifier] = None


def get_context_classifier(similarity_threshold: float = 0.20) -> ContextClassifier:
    global CONTEXT_CLASSIFIER
    if CONTEXT_CLASSIFIER is None:
        CONTEXT_CLASSIFIER = ContextClassifier(similarity_threshold=similarity_threshold)
        CONTEXT_CLASSIFIER.index_schema(SCHEMA)
    else:
        CONTEXT_CLASSIFIER.similarity_threshold = similarity_threshold
    return CONTEXT_CLASSIFIER


def build_section_paragraphs(transcript: str):
    """Build reconstructed paragraphs for each section from a transcript."""
    import traceback
    try:
        provider = get_llm_provider()
        sentences = _prepare_sentences(transcript)
        if not sentences:
            return {}

        sentence_tuples = [(f"sent_{idx}", sent.text) for idx, sent in enumerate(sentences)]
        llm_refiner = provider.generate if provider else None
        mapper = create_semantic_mapper(SCHEMA, llm_refiner=llm_refiner)
        result = mapper.process_transcript(sentence_tuples)
        paragraphs = result.get("paragraphs", {})
        if paragraphs:
            return paragraphs
        # If no paragraphs produced, log and return empty
        logger.debug("build_section_paragraphs: mapper returned empty paragraphs dict")
        return {}
    except Exception as e:
        logger.exception("build_section_paragraphs failed: %s", e)
        return {}


PRIORITY_COVERAGE_SECTION_IDS = [
    'system_overview',
    'deployment_and_rollback',
    'common_failures'
]

from llm.prompts import SECTION_STRUCTURED_PROMPTS, SECTION_POLISH_PROMPTS


def _build_polish_inputs(
    section_id: str,
    section_title: str,
    fragments: List[str],
    max_fragments_per_call: int
) -> Optional[dict]:
    cleaned = [f.strip() for f in (fragments or []) if f and f.strip()]
    if not cleaned:
        return None
    clipped = cleaned[:max_fragments_per_call]
    return {
        'section_id': section_id,
        'title': section_title,
        'fragments': clipped,
        'original_count': len(cleaned)
    }


def _extract_structured_section(
    section_id: str,
    section_title: str,
    fragments: List[str],
) -> Optional[dict]:
    """Extract structured JSON data from section fragments using LLM."""
    if section_id not in SECTION_STRUCTURED_PROMPTS:
        return None
    
    cleaned = [f.strip() for f in (fragments or []) if f and f.strip()]
    if not cleaned:
        return None
    
    provider = get_llm_provider()
    if provider is None:
        return None
    
    joined_fragments = "\n".join(f"- {fragment}" for fragment in cleaned[:8])
    prompt_template = SECTION_STRUCTURED_PROMPTS[section_id]
    prompt = prompt_template.format(
        title=section_title,
        fragments=joined_fragments,
    )
    
    try:
        response = provider.generate(
            prompt,
            temperature=0.2,
            max_output_tokens=512,
            stop_sequences=[],
            system_prompt=(
                "You are a JSON extractor for a Knowledge Transfer document. "
                "Return ONLY valid JSON. Do not add markdown or extra text."
            ),
        )
        response_text = response.strip() if isinstance(response, str) else ""
        parsed = _extract_json_response(response_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        logger.warning("Structured extraction failed for %s: %s", section_id, e)
    
    return None


def wrap_structured_as_fields(structured: dict) -> Dict[str, dict]:
    """Turn a flat {field_id: value} dict (as SECTION_STRUCTURED_PROMPTS's JSON
    schemas produce) into the same {field_id: {"value", "confidence", "source"}}
    shape field_populator.populate_fields() produces, so it can be merged
    straight into populated_fields — feeding renderers and
    knowledge_builder.build_facts/entities/relationships through the one
    existing mechanism, no separate plumbing per section.

    Deliberately generic — the prompts already return values in the exact
    string/list shape each renderer expects (see llm/prompts.py), so no
    per-field join/format logic belongs here.
    """
    return {
        field_id: {"value": value, "confidence": 0.75, "source": "llm_structured"}
        for field_id, value in (structured or {}).items()
        if value
    }


def polish_coverage_sections(
    sections: Dict[str, dict],
    *,
    max_fragments_per_section: int = 8,
) -> Dict[str, List[str]]:
    """Polish coverage sections with section-specific prompts and graceful fallback."""
    cleaned_sections = {}
    for section_id, section_data in sections.items():
        if section_id in SECTION_STRUCTURED_PROMPTS:
            continue
        section_input = _build_polish_inputs(
            section_id,
            section_data.get('title', section_id),
            section_data.get('fragments', []),
            max_fragments_per_section,
        )
        if section_input is not None:
            cleaned_sections[section_id] = section_input

    if not cleaned_sections:
        return {sid: [] for sid in sections}

    def _local_cleanup_list(inputs: List[str]) -> List[str]:
        return [_local_cleanup(text) for text in inputs]

    provider = get_llm_provider()
    if provider is None:
        return {
            sid: _local_cleanup_list(section['fragments'])
            for sid, section in cleaned_sections.items()
        }

    results = {}
    def _enforce_list_item_rule(prompt_text: str) -> str:
        rule = (
            "IMPORTANT: Each list item must be on its own line.\n"
            "Do NOT compress multiple items onto one line.\n"
            "Use actual newlines between items, not semicolons or dashes inline.\n\n"
        )
        if 'Source fragments:' in prompt_text:
            return prompt_text.replace('Source fragments:', rule + 'Source fragments:')
        return rule + prompt_text

    for sid, section in cleaned_sections.items():
        prompt_template = SECTION_POLISH_PROMPTS.get(sid, SECTION_POLISH_PROMPTS["default"])
        joined_fragments = "\n".join(f"- {fragment}" for fragment in section["fragments"])
        prompt = prompt_template.format(
            title=section["title"],
            fragments=joined_fragments,
        )
        prompt = _enforce_list_item_rule(prompt)
        try:
            response = provider.generate(
                prompt,
                temperature=0.2,
                max_output_tokens=1024,
                stop_sequences=[],
                system_prompt=(
                    "You are a senior technical writer for a Knowledge Transfer document. "
                    "Follow the formatting instructions exactly and do not invent facts."
                ),
            )
            cleaned_text = response.strip() if isinstance(response, str) else ""
            results[sid] = [cleaned_text] if cleaned_text else _local_cleanup_list(section["fragments"])
        except Exception as e:
            logger.warning("Polish failed for %s: %s", sid, e)
            results[sid] = _local_cleanup_list(section["fragments"])

    return results


def polish_coverage_text(
    section_id: str,
    section_title: str,
    fragments: List[str],
    *,
    max_fragments_per_call: int = 8,
) -> List[str]:
    batch = {
        section_id: {
            'title': section_title,
            'fragments': fragments,
        }
    }
    polished = polish_coverage_sections(batch, max_fragments_per_section=max_fragments_per_call)
    return polished.get(section_id, [])


def _prepare_sentences(transcript: str):
    """Segment transcript into sentence objects using context_mapper."""
    cleaned_text = clean_transcript(transcript)
    audio_seg = AudioSegment(text=cleaned_text, start=0.0, end=0.0, avg_logprob=-1.0)
    return segment_sentences([audio_seg])


def _classify_transcript_sentences(transcript: str, similarity_threshold: float = 0.20):
    """Classify transcript at sentence granularity with neighbor-aware embeddings."""
    sentences = _prepare_sentences(transcript)
    if not sentences:
        return []

    classifier = get_context_classifier(similarity_threshold)
    texts = [s.text for s in sentences]
    embeddings = classifier.model.encode(texts, convert_to_tensor=True)

    classified_sentences = []
    for idx, sentence in enumerate(sentences):
        window_size = 2
        start = max(0, idx - window_size)
        end = min(len(sentences), idx + window_size + 1)
        context_text = " ".join([sentences[j].text for j in range(start, end)])
        classified_sentence = classifier.classify_sentence(
            sentence,
            sent_embedding=embeddings[idx],
            context_text=context_text
        )
        classified_sentences.append(classified_sentence)
    return classified_sentences


def classify_transcript(transcript: str, *, similarity_threshold: float = 0.20):
    """Classify transcript at sentence level into KT sections."""
    coverage = {s["id"]: [] for s in SCHEMA}
    classified_sentences = _classify_transcript_sentences(transcript, similarity_threshold=similarity_threshold)

    for classified in classified_sentences:
        if classified.is_unassigned or classified.primary_classification is None:
            continue
        confidence = classified.primary_classification.confidence or 0.0
        if confidence < similarity_threshold:
            continue
        section_id = classified.primary_classification.section_id
        coverage[section_id].append(classified.sentence.text)

    return coverage


def analyze_transcript(transcript: str, *, similarity_threshold: float = 0.20, min_chunks_for_covered: int = 2):
    """Produce structured KT coverage per section using sentence-level classification."""
    section_ids = [s['id'] for s in SCHEMA]
    classified_sentences = _classify_transcript_sentences(transcript, similarity_threshold=similarity_threshold)

    if not classified_sentences:
        return {
            s['id']: {
                'status': 'missing',
                'confidence': 0.0,
                'extracted_text': '',
                'chunks': [],
                'scores': []
            }
            for s in SCHEMA
        }

    results = {sec_id: {'matched_chunks': [], 'matched_scores': []} for sec_id in section_ids}
    for classified in classified_sentences:
        if classified.is_unassigned or classified.primary_classification is None:
            continue
        confidence = classified.primary_classification.confidence or 0.0
        if confidence < similarity_threshold:
            continue
        sec_id = classified.primary_classification.section_id
        results[sec_id]['matched_chunks'].append(classified.sentence.text)
        results[sec_id]['matched_scores'].append(confidence)

    final_results = {}
    for sec_id in section_ids:
        matched_chunks = results[sec_id]['matched_chunks']
        matched_scores = results[sec_id]['matched_scores']

        if len(matched_chunks) == 0:
            status = 'missing'
        elif len(matched_chunks) < min_chunks_for_covered:
            status = 'partial'
        else:
            status = 'covered'

        confidence = max(matched_scores) if matched_scores else 0.0
        confidence = float(np.clip(confidence, 0.0, 1.0))
        final_results[sec_id] = {
            'status': status,
            'confidence': confidence,
            'extracted_text': '\n'.join(matched_chunks),
            'chunks': matched_chunks,
            'scores': matched_scores
        }

    return final_results


def map_analysis_to_fields(analysis: dict, schema: list, *, min_similarity: float = 0.25):
    """Map analyzed chunks to concrete schema fields.

    Returns a dict: { section_id: { field_id: {value, confidence, source_chunk_index} } }
    """
    model = get_sentence_model()
    mappings = {}

    # Precompute chunk texts per section
    for sec in schema:
        sid = sec['id']
        sec_info = analysis.get(sid, {})
        chunks = sec_info.get('chunks', [])
        mappings[sid] = {}

        if not chunks:
            # nothing to map
            continue

        # encode all chunks for this section
        if model is not None:
            chunk_embs = model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)
        else:
            chunk_embs = np.random.rand(len(chunks), 384)

        for field in sec.get('fields', []) or []:
            fid = field.get('id')
            if not fid:
                continue
            # build a representative text for the field: label + id + options + columns
            parts = []
            if field.get('label'):
                parts.append(field['label'])
            parts.append(field.get('id', ''))
            if field.get('options'):
                parts.extend([str(o) for o in field.get('options')])
            if field.get('columns'):
                parts.extend(field.get('columns'))
            field_text = " ".join(parts)
            if not field_text.strip():
                continue
            if model is not None:
                field_emb = model.encode(field_text, convert_to_tensor=True, normalize_embeddings=True)
            else:
                field_emb = np.random.rand(384)

            # compute similarities
            if model is not None:
                sims = util.cos_sim(field_emb, chunk_embs)[0]
            else:
                sims = np.dot(field_emb, chunk_embs.T)
                sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
            best_idx = int(sims.argmax().item())
            best_score = float(sims[best_idx].item())

            if best_score >= min_similarity:
                # pick the chunk as value (or refine for tables)
                value = chunks[best_idx]
                mappings[sid][fid] = {
                    'value': value,
                    'confidence': best_score,
                    'source_chunk_index': best_idx
                }

    return mappings

def summarize_coverage(analysis: dict, schema: list):
    required_ids = [s['id'] for s in schema if s.get('required')]
    covered = 0
    partial = 0
    confidences = []
    missing_ids = []
    for sid in required_ids:
        info = analysis.get(sid, {})
        status = info.get('status', 'missing')
        if status == 'covered':
            covered += 1
        elif status == 'partial':
            partial += 1
        else:
            missing_ids.append(sid)
        confidences.append(float(info.get('confidence', 0.0)))
    total = len(required_ids)
    coverage_pct = 0.0 if total == 0 else (covered + 0.5 * partial) / total
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    risk = 1.0 - coverage_pct
    if total > 0:
        risk += len(missing_ids) / total * 0.5
    risk = float(np.clip(risk, 0.0, 1.0))
    return {
        'coverage_percentage': coverage_pct,
        'confidence_score': avg_conf,
        'risk_score': risk,
        'missing_required_sections': missing_ids
    }

def explainability_logs(analysis: dict):
    logs = {}
    for sid, info in analysis.items():
        status = info.get('status', 'missing')
        if status == 'missing':
            expected = list(SECTION_HINTS.get(sid, []))[:10]
            logs[sid] = {
                'status': status,
                'reason': 'no matched chunks',
                'expected_hints': expected,
                'matched': []
            }
        else:
            logs[sid] = {
                'status': status,
                'matched_count': len(info.get('chunks', [])),
                'top_score': max(info.get('scores', [0.0])) if info.get('scores') else 0.0
            }
    return logs

def generate_report(transcript: str, similarity_threshold: float = 0.20, min_chunks_for_covered: int = 2, tenant_id: str = "", project_id: str = "", team_id: str = "", session_state: str = "In Progress"):
    analysis = analyze_transcript(transcript, similarity_threshold=similarity_threshold, min_chunks_for_covered=min_chunks_for_covered)
    summary = summarize_coverage(analysis, SCHEMA)
    logs = explainability_logs(analysis)
    heatmap = {}
    for s in SCHEMA:
        sid = s['id']
        info = analysis.get(sid, {})
        status = info.get('status', 'missing')
        conf = float(info.get('confidence', 0.0))
        if status == 'covered':
            heat = 1.0 * conf
        elif status == 'partial':
            heat = 0.5 * conf
        else:
            heat = 0.0
        heatmap[sid] = heat
    incomplete_mandatory = summary['missing_required_sections']
    recommended_state = "Pending Review"
    complete_with_risk = False
    if len(incomplete_mandatory) > 0:
        recommended_state = "Completed with Risk"
        complete_with_risk = True
    paragraph_data = build_section_paragraphs(transcript)
    audit = {
        'timestamp': datetime.utcnow().isoformat() + "Z",
        'tenant_id': tenant_id,
        'project_id': project_id,
        'team_id': team_id,
        'session_state': session_state,
        'coverage_summary': summary,
        'incomplete_mandatory_sections': incomplete_mandatory
    }
    return {
        'analysis': analysis,
        'coverage_map': {sid: analysis[sid]['status'] for sid in analysis},
        'heatmap': heatmap,
        'paragraphs': paragraph_data,
        'auto_highlight_incomplete_mandatory': incomplete_mandatory,
        'summary': summary,
        'risk_warning': complete_with_risk,
        'recommended_state': recommended_state,
        'audit_log': audit,
        'explainability': logs
    }
