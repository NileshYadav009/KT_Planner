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

# Enhanced modules for better accuracy
try:
    import librosa
    from librosa import feature as librosa_feature
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import time
from devops_transcription import clean_transcript
from context_mapper import AudioSegment, ContextClassifier, segment_sentences
from enterprise_semantic_mapper import create_semantic_mapper
import requests

logger = logging.getLogger(__name__)

# ============================================================================
# GEMINI LLM SETUP
# ============================================================================
import os

# Gemini config (Google Generative AI)
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_CLIENT = None
GEMINI_ENABLED = False
try:
    from google import genai
    if GEMINI_API_KEY:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_ENABLED = True
except Exception:
    GEMINI_CLIENT = None
    GEMINI_ENABLED = False


def gemini_refiner(prompt: str, metadata: Optional[dict] = None) -> str:
    """Refine text using Google Gemini via the genai SDK or direct HTTP with retry logic."""
    if not (GEMINI_API_KEY or GEMINI_CLIENT):
        raise RuntimeError("Gemini client not configured. Set GEMINI_API_KEY to enable.")

    max_retries = 2
    retry_delay = 1
    last_exception = None

    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()
            # Prefer SDK client when available
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
                # Use direct HTTP call as a fallback
                url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
                payload = {
                    "prompt": {"text": prompt},
                    "temperature": 0.2,
                    "maxOutputTokens": 1024,
                    "stop_sequences": ["\n\n"]
                }
                r = requests.post(url, json=payload, timeout=(10, 120))
                r.raise_for_status()
                data = r.json()
                # Try to extract text from common response shapes
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
            return text
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning("Gemini attempt %d/%d failed: %s. Retrying in %ds...", attempt + 1, max_retries, e, retry_delay)
                time.sleep(retry_delay)
                retry_delay *= 2
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



with open("kt_schema_new.json") as f:
    SCHEMA = json.load(f)["sections"]

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

def assess_audio_quality(audio_path: str) -> Dict[str, float]:
    """
    Analyze audio quality metrics before transcription.
    Helps predict transcription accuracy and suggest improvements.
    
    Returns metrics for quality assessment (NEW in upgraded modules).
    """
    if not HAS_LIBROSA:
        return {"status": "librosa_not_available", "score": 0.0}
    
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Compute energy and zero-crossing rate
        rms = librosa_feature.rms(y=y)[0]
        zcr = librosa_feature.zero_crossing_rate(y)[0]
        
        # Detect silence regions
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        silence_frames = np.sum(S_db < -40) / S_db.shape[1]
        
        avg_energy = float(np.mean(rms))
        avg_zcr = float(np.mean(zcr))
        
        # Quality score (0-100)
        quality_score = (
            min(avg_energy * 100, 50) +  # Energy contribution
            min(avg_zcr * 10, 30) +       # Voice activity contribution
            max(0, 20 - silence_frames * 100)  # Silence penalty
        )
        
        return {
            "score": float(quality_score),
            "energy": avg_energy,
            "voice_activity": avg_zcr,
            "silence_ratio": float(silence_frames),
            "recommendation": "good" if quality_score > 60 else "fair" if quality_score > 40 else "poor"
        }
    except Exception as e:
        return {"error": str(e), "score": 0.0}

def classify_with_confidence(sentence: str, section_embeddings: Dict[str, np.ndarray], 
                            section_ids: List[str]) -> Dict:
    """
    Enhanced classification with confidence scores.
    Returns section assignment with confidence level (IMPROVED).
    """
    model = get_sentence_model()
    if model is None:
        return {"error": "Model not loaded", "section": None, "confidence": 0.0}
    
    sent_embed = model.encode(sentence, normalize_embeddings=True)
    
    # Compute cosine similarities
    similarities = util.cos_sim([sent_embed], list(section_embeddings.values()))[0]
    similarities = similarities.cpu().numpy() if hasattr(similarities, 'cpu') else similarities
    
    # Get top matches
    top_indices = np.argsort(similarities)[-1:][::-1]
    
    return {
        "section": section_ids[int(top_indices[0])],
        "confidence": float(similarities[int(top_indices[0])]),
        "alternatives": [
            {"section": section_ids[int(idx)], "score": float(similarities[int(idx)])}
            for idx in top_indices[1:] if similarities[int(idx)] > CONFIDENCE_THRESHOLD * 0.8
        ],
        "requires_review": float(similarities[int(top_indices[0])]) < CONFIDENCE_THRESHOLD
    }

def validate_sentence_quality(sentence: str) -> Dict[str, any]:
    """
    Validate sentence quality using NLP metrics (NEW).
    Detects potential transcription errors.
    """
    if not HAS_NLTK:
        return {"status": "nltk_not_available", "quality_score": 0.0}
    
    try:
        words = sentence.split()
        word_count = len(words)
        
        # Basic metrics
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Check for common transcription errors
        quality_issues = []
        if word_count < 3:
            quality_issues.append("too_short")
        if word_count > 100:
            quality_issues.append("too_long")
        if avg_word_length > 20:
            quality_issues.append("unusual_word_length")
        
        # Scoring
        score = 100
        if quality_issues:
            score -= len(quality_issues) * 10
        
        return {
            "quality_score": max(0, score),
            "word_count": word_count,
            "avg_word_length": float(avg_word_length),
            "issues": quality_issues,
            "status": "acceptable" if score > 70 else "warning" if score > 40 else "review_needed"
        }
    except Exception as e:
        return {"error": str(e), "quality_score": 0.0}

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

def chunk_text(text: str, size: int = 200):
    """Legacy chunking helper.

    This remains available for backward compatibility but is no longer used
    for section classification. Sentence-level classification from
    context_mapper is preferred.
    """
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i + size])


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
        # Warm up models (Gemini preferred) before processing
        warmup_models()
        
        sentences = _prepare_sentences(transcript)
        if not sentences:
            return {}

        sentence_tuples = [(f"sent_{idx}", sent.text) for idx, sent in enumerate(sentences)]
        llm_refiner = gemini_refiner if GEMINI_ENABLED else None
        mapper = create_semantic_mapper(SCHEMA, llm_refiner=llm_refiner)
        result = mapper.process_transcript(sentence_tuples)
        paragraphs = result.get("paragraphs", {})
        if paragraphs:
            return paragraphs
        # If no paragraphs produced, log and return empty
        print(f"[DEBUG] build_section_paragraphs: mapper returned empty paragraphs dict")
        return {}
    except Exception as e:
        print(f"[ERROR] build_section_paragraphs failed: {e}")
        traceback.print_exc()
        return {}


def _prepare_sentences(transcript: str):
    """Segment transcript into sentence objects using context_mapper."""
    cleaned_text = clean_transcript(transcript)
    audio_seg = AudioSegment(text=cleaned_text, start=0.0, end=0.0, avg_logprob=-1.0)
    return segment_sentences([audio_seg])


def process_coverage_with_gemini(transcript: str, sentences_per_chunk: int = 20):
    """Chunk transcript sentences, send each chunk to Gemini, and return structured results.

    Returns a list of dicts: {"chunk_index": int, "text": str, "gemini_raw": str, "gemini_json": Optional[dict]}
    """
    sentences = _prepare_sentences(transcript)
    if not sentences:
        return []

    # Prepare section catalog to help Gemini map chunks to KT sections
    section_list = []
    for s in SCHEMA:
        title = s.get("title") or s.get("id")
        section_list.append(f"{s.get('id')}: {title}")
    section_hint = "\n".join(section_list)

    # Chunk sentences
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        part = " ".join([sentences[j].text for j in range(i, min(i + sentences_per_chunk, len(sentences)))])
        chunks.append(part)

    results = []
    for idx, chunk_text in enumerate(chunks):
        prompt = f"""
You are a KT mapping assistant. Given the list of KT sections below, read the transcript chunk and map it to the most relevant sections. Return JSON only as a list of objects with keys: section_id, confidence (0-1 float), summary.

Sections:
{section_hint}

Transcript chunk:
{chunk_text}

Return JSON only.
"""
        try:
            gem = gemini_refiner(prompt)
            gem_json = None
            try:
                gem_json = json.loads(gem)
            except Exception:
                # If Gemini didn't return machine JSON, keep raw text
                gem_json = None

            results.append({
                "chunk_index": idx,
                "text": chunk_text,
                "gemini_raw": gem,
                "gemini_json": gem_json,
            })
        except Exception as e:
            results.append({
                "chunk_index": idx,
                "text": chunk_text,
                "gemini_raw": "",
                "gemini_json": None,
                "error": str(e),
            })

    return results


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

class KTSessionAggregator:
    def __init__(self, schema: list, min_chunks_for_covered: int = 2):
        self.schema = schema
        self.min_chunks_for_covered = min_chunks_for_covered
        self.sessions = []
        self.aggregate = {s['id']: {'chunks': [], 'scores': []} for s in schema}
    def add_transcript(self, transcript: str, similarity_threshold: float = 0.20):
        res = analyze_transcript(transcript, similarity_threshold=similarity_threshold, min_chunks_for_covered=self.min_chunks_for_covered)
        self.sessions.append(res)
        for s in self.schema:
            sid = s['id']
            info = res.get(sid, {})
            self.aggregate[sid]['chunks'].extend(info.get('chunks', []))
            self.aggregate[sid]['scores'].extend(info.get('scores', []))
    def aggregated_analysis(self):
        out = {}
        for s in self.schema:
            sid = s['id']
            chunks = self.aggregate[sid]['chunks']
            scores = self.aggregate[sid]['scores']
            if len(chunks) == 0:
                status = 'missing'
            elif len(chunks) < self.min_chunks_for_covered:
                status = 'partial'
            else:
                status = 'covered'
            conf = max(scores) if scores else 0.0
            conf = float(np.clip(conf, 0.0, 1.0))
            out[sid] = {
                'status': status,
                'confidence': conf,
                'extracted_text': '\n'.join(chunks),
                'chunks': chunks,
                'scores': scores
            }
        return out

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


# ---------------------------------------------------------------------------
# Open Risks & Knowledge Gaps inference (KT methodology §OPEN RISKS)
# ---------------------------------------------------------------------------
# Non-hallucinated: only restates gaps the pipeline already detected and asks
# a follow-up question per gap derived from the section's own description.
# Never invents missing content; only surfaces what is absent or thin.

def _schema_lookup():
    """Return a dict of {section_id: section_meta} from the loaded SCHEMA."""
    return {s.get("id"): s for s in SCHEMA if isinstance(s, dict) and s.get("id")}


def _followup_question(section_meta: dict) -> str:
    """Derive a single follow-up question from a section's title/description."""
    title = (section_meta.get("title") or section_meta.get("id") or "this area").strip()
    desc = (section_meta.get("description") or "").strip()
    if desc:
        # Use the description as the gist of what to ask about.
        return f"What is the current state of {title.lower()}? ({desc})"
    return f"Can the outgoing owner walk through {title.lower()} in more detail?"


def infer_open_risks(kt) -> dict:
    """Produce a structured Open Risks & Knowledge Gaps report from a StructuredKT.

    Reads kt.coverage (status, required, risk_score, sentence_count),
    kt.missing_required_sections, and kt.unassigned_sentences. Returns:

        {
          "risks": [
            {
              "section": "deployment_and_rollback",
              "section_title": "...",
              "gap": "missing" | "weak" | "required_missing" | "unassigned_evidence",
              "severity": "high" | "medium" | "low",
              "detail": "<plain-language explanation>",
              "suggested_followup": "<question for outgoing owner>"
            }, ...
          ],
          "summary": {
            "total_risks": N,
            "high_severity": N,
            "missing_required_count": N,
            "unassigned_sentence_count": N
          }
        }
    """
    schema_by_id = _schema_lookup()
    risks: list = []
    seen_keys: set = set()

    def _add(section_id: str, gap: str, severity: str, detail: str):
        key = (section_id, gap)
        if key in seen_keys:
            return
        seen_keys.add(key)
        meta = schema_by_id.get(section_id, {})
        risks.append({
            "section": section_id,
            "section_title": meta.get("title", section_id),
            "gap": gap,
            "severity": severity,
            "detail": detail,
            "suggested_followup": _followup_question(meta) if meta else
                f"Clarify coverage for '{section_id}'.",
        })

    # 1. Explicitly missing required sections (highest severity)
    for sec_id in (getattr(kt, "missing_required_sections", None) or []):
        _add(
            sec_id,
            gap="required_missing",
            severity="high",
            detail="Required section has no classified content in the transcript.",
        )

    # 2. Walk coverage: missing or weak sections
    coverage = getattr(kt, "coverage", None) or {}
    for sec_id, cov in coverage.items():
        status = getattr(cov, "status", None) or ""
        required = bool(getattr(cov, "required", False))
        sentence_count = int(getattr(cov, "sentence_count", 0) or 0)
        risk = float(getattr(cov, "risk_score", 0.0) or 0.0)
        title = getattr(cov, "section_title", sec_id)

        if status == "missing":
            sev = "high" if required else "medium"
            _add(
                sec_id,
                gap="missing",
                severity=sev,
                detail=f"Section '{title}' has no content{' (required)' if required else ''}.",
            )
        elif status == "weak":
            sev = "medium" if required or risk >= 0.5 else "low"
            _add(
                sec_id,
                gap="weak",
                severity=sev,
                detail=(
                    f"Section '{title}' has thin coverage "
                    f"({sentence_count} sentence(s), risk={risk:.2f})."
                ),
            )

    # 3. Unassigned sentences = ambiguous evidence the reviewer should triage
    unassigned = getattr(kt, "unassigned_sentences", None) or []
    if unassigned:
        sample = []
        for s in unassigned[:3]:
            t = getattr(s, "text", None) or (s.get("text") if isinstance(s, dict) else None)
            if t:
                sample.append(t.strip())
        detail = (
            f"{len(unassigned)} sentence(s) could not be confidently mapped to any "
            f"section."
        )
        if sample:
            detail += " Examples: " + " | ".join(f'"{t[:80]}"' for t in sample)
        _add(
            "unassigned",
            gap="unassigned_evidence",
            severity="medium",
            detail=detail,
        )

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    risks.sort(key=lambda r: (severity_rank.get(r["severity"], 9), r["section"]))

    return {
        "risks": risks,
        "summary": {
            "total_risks": len(risks),
            "high_severity": sum(1 for r in risks if r["severity"] == "high"),
            "missing_required_count": len(getattr(kt, "missing_required_sections", None) or []),
            "unassigned_sentence_count": len(unassigned),
        },
    }
