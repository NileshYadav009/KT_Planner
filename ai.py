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

from field_populator import find_source_sentence_index
from grounding import ground_structured, added_specifics
from llm.usage import record_rejected as record_llm_rejected
from llm_provider import get_llm_provider, LLM_PARALLEL_WORKERS
import contextvars
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    # Optional: repairs slightly malformed LLM JSON. Without it, such output
    # is discarded exactly as before.
    from json_repair import repair_json
except ImportError:
    repair_json = None
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

    # The first complete JSON value, wherever it starts. raw_decode stops at
    # its end, so a lead-in or a trailing remark from the model no longer
    # breaks parsing. The old regexes tried "[...]" first and returned an
    # array nested inside the object ("escalation_chain"), discarding the
    # answer whenever the model added a sentence after its JSON.
    start = min((i for i in (payload.find('{'), payload.find('[')) if i >= 0), default=-1)
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(payload[start:])
        return value
    except Exception:
        pass

    # Last resort: repair common syntax slips (trailing commas, single
    # quotes, unquoted keys, None/True, missing commas, comments). Only for
    # output that ends like complete JSON: an answer cut off by the token
    # limit must not be completed with guessed closing brackets.
    if repair_json is not None:
        body = payload[start:].rstrip()
        if body.endswith(('}', ']')):
            try:
                repaired = repair_json(body, return_objects=True)
            except Exception:
                repaired = None
            if isinstance(repaired, (dict, list)) and repaired:
                return repaired
    return None


def _local_cleanup(text: str) -> str:
    text = ' '.join(text.split())
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in '.!?':
        # A clause cut from a longer sentence ends in a comma; appending a
        # period produced "...platform,." in the document.
        text = text.rstrip(' ,;:') + '.'
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


def get_sentence_model(model_name: str = "all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:
    global SENT_MODEL
    if SENT_MODEL is None:
        try:
            SENT_MODEL = SentenceTransformer(model_name)
        except Exception:
            SENT_MODEL = None
    return SENT_MODEL


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
        # Fragments beyond the per-call cap. The polished result REPLACES the
        # section's content, so these must be carried through (locally
        # cleaned) rather than dropped: they used to vanish from every
        # section with more than max_fragments_per_call sentences.
        'overflow': cleaned[max_fragments_per_call:],
        'original_count': len(cleaned)
    }


# How many of a section's sentences the structured-extraction prompt sees.
STRUCTURED_MAX_FRAGMENTS = int(os.getenv("STRUCTURED_MAX_FRAGMENTS", "20"))


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
    
    # Was cleaned[:8]: the failures table, ownership fields etc. never saw
    # anything said after a section's eighth sentence.
    joined_fragments = "\n".join(f"- {fragment}" for fragment in cleaned[:STRUCTURED_MAX_FRAGMENTS])
    prompt_template = SECTION_STRUCTURED_PROMPTS[section_id]
    prompt = prompt_template.format(
        title=section_title,
        fragments=joined_fragments,
    )
    
    try:
        response = provider.generate(
            prompt,
            temperature=0.2,
            max_output_tokens=768,
            stop_sequences=[],
            system_prompt=(
                "You are a JSON extractor for a Knowledge Transfer document. "
                "Return ONLY valid JSON. Do not add markdown or extra text."
            ),
        )
        response_text = response.strip() if isinstance(response, str) else ""
        parsed = _extract_json_response(response_text)
        if isinstance(parsed, dict):
            # Keep only values the fragments support: an invented contact,
            # region or tool would otherwise reach the document as fact.
            dropped: List[str] = []
            parsed = ground_structured(parsed, "\n".join(cleaned[:STRUCTURED_MAX_FRAGMENTS]), dropped)
            for item in dropped:
                logger.warning("Discarded unsupported structured value for %s: %s", section_id, item[:200])
                record_llm_rejected("structured_extraction", f"{section_id}: {item[:120]}")
            return parsed
    except Exception as e:
        logger.warning("Structured extraction failed for %s: %s", section_id, e)
    
    return None


def wrap_structured_as_fields(structured: dict, raw_sentence_texts: Optional[List[str]] = None) -> Dict[str, dict]:
    """Turn a flat {field_id: value} dict (as SECTION_STRUCTURED_PROMPTS's JSON
    schemas produce) into the same {field_id: {"value", "confidence", "source"}}
    shape field_populator.populate_fields() produces, so it can be merged
    straight into populated_fields — feeding renderers and
    knowledge_builder.build_facts/entities/relationships through the one
    existing mechanism, no separate plumbing per section.

    Deliberately generic — the prompts already return values in the exact
    string/list shape each renderer expects (see llm/prompts.py), so no
    per-field join/format logic belongs here.

    raw_sentence_texts (kt.section_content[section_id]'s sentences, if passed):
    used to attach a precise source_chunk_index per field via the same
    find_source_sentence_index() field_populator.populate_fields() uses —
    without it, knowledge_builder._collect_evidence() falls back to attaching
    the same generic "first 3 sentences of the section" to every fact in the
    section, regardless of which one actually supports it.
    """
    fields = {}
    for field_id, value in (structured or {}).items():
        if not value:
            continue
        entry = {"value": value, "confidence": 0.75, "source": "llm_structured"}
        if raw_sentence_texts:
            idx = find_source_sentence_index(value, raw_sentence_texts)
            if idx is not None:
                entry["source_chunk_index"] = idx
        fields[field_id] = entry
    return fields


_POLISH_STOPWORDS = frozenset(
    "a an an and are as at be been but by for from had has have if in into is it its of on "
    "or that the their then there these this to was were when which while with you your "
    # Modals/negations carry no identifying signal for a presence check, and
    # a legitimate rewrite routinely swaps them ("must not be changed" ->
    # "never change"), which would otherwise look like a dropped fragment.
    "must not do does should would will can never no also".split()
)
# Words are compared on a short prefix so ordinary inflection ("changed" vs
# "change", "integrations" vs "integration") doesn't read as a loss.
_POLISH_STEM_LEN = 5
# Share of a fragment's content words that must still be present in the
# polished text for that fragment to count as retained. Below 1.0 because a
# legitimate polish does rephrase ("bicep" -> "Bicep", dropping filler), but
# high enough that an omitted sentence is caught.
_POLISH_RETENTION_RATIO = 0.7
# Fragments shorter than this carry too few content words for the ratio to
# mean anything, so they are not checked.
_POLISH_MIN_CONTENT_WORDS = 4


def _fragments_missing_from(fragments: List[str], polished_text: str) -> List[str]:
    """Fragments whose content is not adequately represented in `polished_text`.

    Word-level, not substring: the polish pass is expected to rewrite
    wording, so the test is whether a fragment's distinctive content words
    survived, not whether the sentence is reproduced verbatim.
    """
    def _stems(text: str) -> List[str]:
        return [
            w[:_POLISH_STEM_LEN]
            for w in re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower()).split()
            if w not in _POLISH_STOPWORDS
        ]

    polished_stems = set(_stems(polished_text))
    missing = []
    for fragment in fragments or []:
        stems = _stems(str(fragment))
        if len(stems) < _POLISH_MIN_CONTENT_WORDS:
            continue
        retained = sum(1 for s in stems if s in polished_stems) / len(stems)
        if retained < _POLISH_RETENTION_RATIO:
            missing.append(fragment)
    return missing


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

    def _all_fragments(section: dict) -> List[str]:
        return list(section['fragments']) + list(section.get('overflow') or [])

    provider = get_llm_provider()
    if provider is None:
        return {
            sid: _local_cleanup_list(_all_fragments(section))
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

    def _polish_one(sid: str, section: dict) -> List[str]:
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
            if not cleaned_text:
                return _local_cleanup_list(_all_fragments(section))
            added = added_specifics(cleaned_text, list(section["fragments"]) + [prompt])
            if added:
                # The rewrite states specifics (names, numbers, products) that
                # none of its fragments contain: an invented fact. Same rule as
                # a dropped fact below: keep the speaker's own words.
                logger.warning("Polish for %s added unsupported specifics %s; keeping raw text instead", sid, added[:5])
                record_llm_rejected("section_polish", f"{sid}: added {', '.join(added[:5])}")
                return _local_cleanup_list(_all_fragments(section))
            dropped = _fragments_missing_from(section["fragments"], cleaned_text)
            if dropped:
                # The polish pass rewrites a section wholesale, and its
                # output then REPLACES the raw fragments as that section's
                # content — so anything the model silently omits is gone
                # from the document entirely. Observed live: a Danger Zones
                # polish returned only one of two prohibitions, and
                # "Production Kubernetes configuration must not be changed
                # manually." vanished from the KT with nothing reporting a
                # loss. Prose quality is never worth losing a stated fact,
                # so a lossy rewrite is discarded in favour of the raw text.
                logger.warning(
                    "Polish for %s dropped %d fragment(s); keeping raw text instead",
                    sid, len(dropped),
                )
                return _local_cleanup_list(_all_fragments(section))
            # Only the first chunk is sent for polishing (token budget); the
            # rest follows verbatim so no sentence is lost from the section.
            return [cleaned_text] + _local_cleanup_list(section.get("overflow") or [])
        except Exception as e:
            logger.warning("Polish failed for %s: %s", sid, e)
            return _local_cleanup_list(_all_fragments(section))

    # Each section's polish call is fully independent (its own prompt, its
    # own result slot) — dispatching them concurrently instead of one at a
    # time removes the dominant cost (waiting out each network round trip
    # serially) without changing a single prompt or outcome. The provider's
    # own rate throttle is shared/thread-safe, so this can't send more
    # requests per minute than the sequential version did.
    with ThreadPoolExecutor(max_workers=min(LLM_PARALLEL_WORKERS, len(cleaned_sections)) or 1) as pool:
        # copy_context(): worker threads count their calls against the
        # current KT's usage tracker (llm/usage.py).
        futures = {
            pool.submit(contextvars.copy_context().run, _polish_one, sid, section): sid
            for sid, section in cleaned_sections.items()
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

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
