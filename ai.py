import json
import re
import numpy as np
import requests
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
from devops_transcription import clean_transcript
from context_mapper import AudioSegment, ContextClassifier, segment_sentences
from enterprise_semantic_mapper import create_semantic_mapper

PHI3_MINI_ENDPOINT = "http://localhost:11434/v1/completions"
PHI3_MINI_MODEL = "phi3:mini"


def phi3_mini_refiner(prompt: str, metadata: Optional[dict] = None) -> Tuple[str, bool]:
    """Refine text using local phi3:mini via the local server endpoint."""
    payload = {
        "model": PHI3_MINI_MODEL,
        "prompt": prompt,
        "temperature": 0.2,
        "max_tokens": 1024,
        "stop": ["\n\n"]
    }

    def normalize_response(raw_text: str) -> str:
        text = raw_text.strip()
        prompt_text = prompt.strip()
        if prompt_text and text.startswith(prompt_text):
            text = text[len(prompt_text):].strip()
        # Remove any leading response markers
        return re.sub(r'^(\n|\r|\s)+', '', text)

    try:
        response = requests.post(PHI3_MINI_ENDPOINT, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()

        text = ""
        if isinstance(data, dict):
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                text = choice.get("text") or choice.get("message", {}).get("content", "")
            else:
                text = data.get("text", "")
        elif isinstance(data, str):
            text = data

        text = normalize_response(str(text))
        did_refine = bool(text and text != prompt.strip() and text != "")
        return (text or prompt.strip()), did_refine
    except Exception:
        return prompt.strip(), False

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
    """Build reconstructed paragraphs for each section from a transcript.

    If the enterprise mapper fails to produce paragraphs, apply a conservative
    professionalization fallback to return at least one polished paragraph.
    """
    try:
        sentences = _prepare_sentences(transcript)
        if not sentences:
            return {}

        sentence_tuples = [(f"sent_{idx}", sent.text) for idx, sent in enumerate(sentences)]
        mapper = create_semantic_mapper(SCHEMA, llm_refiner=phi3_mini_refiner)
        result = mapper.process_transcript(sentence_tuples)
        paragraphs = result.get("paragraphs", {})

        if paragraphs:
            return paragraphs

        # Fallback: produce a single professional paragraph for safety using a conservative local professionalizer
        try:
            def _local_professionalize(text: str) -> Tuple[str, bool, str]:
                import re
                if not text or not text.strip():
                    return text, False, 'empty'
                # Simple dedupe of adjacent duplicate sentences
                parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', text) if p.strip()]
                deduped = []
                for p in parts:
                    if not deduped or p.lower() != deduped[-1].lower():
                        deduped.append(p)
                merged_parts = []
                for p in deduped:
                    if merged_parts:
                        prev = merged_parts[-1]
                        prev_tokens = set(re.findall(r"\w+", prev.lower()))
                        cur_tokens = set(re.findall(r"\w+", p.lower()))
                        if prev_tokens and len(prev_tokens & cur_tokens) / max(1, len(cur_tokens)) > 0.7:
                            if len(p) < len(prev):
                                merged_parts[-1] = p
                            continue
                    merged_parts.append(p)
                refined = ' '.join(merged_parts)
                refined = re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), refined)
                refined = re.sub(r'\s+', ' ', refined).strip()
                if refined and not refined.endswith(('.', '!', '?')):
                    refined += '.'
                return refined, refined.strip() != text.strip(), 'local_fallback'

            merged = " ".join([t for _, t in sentence_tuples])
            prof_text, did, details = _local_professionalize(merged)
            return {"system_overview": [{
                'section_id': 'system_overview',
                'original_sentence_ids': [sid for sid, _ in sentence_tuples],
                'text': prof_text,
                'word_count': len(prof_text.split()),
                'coherence_score': 1.0,
                'is_repaired': False,
                'pass_count': 1 + int(did),
                'repair_details': details,
                'is_professionalized': bool(did),
                'professional_details': details
            }]}
        except Exception:
            return {}
    except Exception:
        return {}


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
