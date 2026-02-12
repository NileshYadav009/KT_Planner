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
from typing import Dict, List, Optional
from datetime import datetime

with open("kt_schema_new.json") as f:
    SCHEMA = json.load(f)["sections"]

SENT_MODEL: Optional[SentenceTransformer] = None
SECTION_EMBEDS: Optional[Dict[str, np.ndarray]] = None

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
        # Convert to numpy for vectorized operations
        if model is not None:
            embeds[s["id"]] = model.encode(text, convert_to_tensor=False)
        else:
            embeds[s["id"]] = np.random.rand(384)

    SECTION_EMBEDS = embeds
    return SECTION_EMBEDS

def chunk_text(text: str, size: int = 120):
    """Yield chunks of approximately `size` words (smaller chunks improve matching)."""
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i + size])

def classify_transcript(transcript: str, *, similarity_threshold: float = 0.20):
    """Classify transcript chunks into KT sections (deduplicating across sections).
    
    Each chunk is assigned to its best-matching section (prevents duplication).
    Optimized: batch encodes chunks, uses vectorized numpy operations, deduplicates.
    """
    coverage = {s["id"]: [] for s in SCHEMA}
    section_embeds = get_section_embeds()
    model = get_sentence_model()
    
    # Collect all chunks first
    chunks = list(chunk_text(transcript))
    if not chunks:
        return coverage
    
    # Batch encode all chunks at once (much faster than one-by-one)
    if model is not None:
        chunk_embeddings = model.encode(chunks, convert_to_tensor=False, batch_size=32)
    else:
        chunk_embeddings = np.random.rand(len(chunks), 384)
    
    # Vectorized cosine similarity computation
    section_ids = list(section_embeds.keys())
    section_emb_list = np.array([section_embeds[sid] for sid in section_ids])
    
    # Compute similarity matrix: (num_chunks, num_sections)
    similarities = np.dot(chunk_embeddings, section_emb_list.T)
    
    # Normalize by magnitudes for cosine similarity
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    section_norms = np.linalg.norm(section_emb_list, axis=1, keepdims=True).T
    similarities = similarities / (chunk_norms * section_norms + 1e-9)
    
    # Assign each chunk to best matching section (deduplication)
    lower_chunks = [c.lower() for c in chunks]
    for chunk_idx, chunk in enumerate(chunks):
        chunk_lower = lower_chunks[chunk_idx]
        chunk_tokens = {token.lower() for token in WORD_RE.findall(chunk)}
        
        best_sec_idx = -1
        best_score = similarity_threshold - 0.01
        best_has_hint = False
        
        # Find the BEST matching section for this chunk
        for sec_idx, sec_id in enumerate(section_ids):
            score = float(similarities[chunk_idx, sec_idx])
            hints = SECTION_HINTS.get(sec_id, set())
            has_token_hint = bool(chunk_tokens & hints)
            has_substr_hint = any(h in chunk_lower for h in hints)
            has_hint = has_token_hint or has_substr_hint
            
            # Prefer high scores; break ties using hint presence
            should_update = False
            if has_hint and not best_has_hint:
                should_update = True  # hint match beats no-hint match
            elif has_hint == best_has_hint and score > best_score:
                should_update = True  # better score when hint status is same
            
            if should_update:
                best_sec_idx = sec_idx
                best_score = score
                best_has_hint = has_hint
        
        # Assign chunk only to best section (no duplication)
        if best_sec_idx >= 0:
            coverage[section_ids[best_sec_idx]].append(chunk)
    
    return coverage


def analyze_transcript(transcript: str, *, similarity_threshold: float = 0.20, min_chunks_for_covered: int = 2):
    """Produce structured KT coverage per section (using deduplicated chunks).

    Returns a dict mapping section id -> {
        status: 'covered'|'partial'|'missing',
        confidence: float (0..1),
        extracted_text: str,
        chunks: [str],
        scores: [float]
    }
    """
    section_embeds = get_section_embeds()
    model = get_sentence_model()

    chunks = list(chunk_text(transcript))
    if not chunks:
        # return missing for all
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

    # encode chunks and compute similarities
    if model is not None:
        chunk_embeddings = model.encode(chunks, convert_to_tensor=False, batch_size=32)
    else:
        chunk_embeddings = np.random.rand(len(chunks), 384)
    section_ids = list(section_embeds.keys())
    section_emb_list = np.array([section_embeds[sid] for sid in section_ids])
    sims = np.dot(chunk_embeddings, section_emb_list.T)
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    section_norms = np.linalg.norm(section_emb_list, axis=1, keepdims=True).T
    sims = sims / (chunk_norms * section_norms + 1e-9)

    # Assign each chunk to BEST matching section (deduplicated)
    results = {sec_id: {'matched_chunks': [], 'matched_scores': []} for sec_id in section_ids}
    lower_chunks = [c.lower() for c in chunks]
    
    for c_idx, chunk in enumerate(chunks):
        chunk_lower = lower_chunks[c_idx]
        chunk_tokens = {token.lower() for token in WORD_RE.findall(chunk)}
        
        best_sec_idx = -1
        best_score = similarity_threshold - 0.01
        best_has_hint = False
        
        # Find BEST section for this chunk
        for sec_idx, sec_id in enumerate(section_ids):
            score = float(sims[c_idx, sec_idx])
            hints = SECTION_HINTS.get(sec_id, set())
            has_token_hint = bool(chunk_tokens & hints)
            has_substr_hint = any(h in chunk_lower for h in hints)
            has_hint = has_token_hint or has_substr_hint
            
            should_update = False
            if has_hint and not best_has_hint:
                should_update = True
            elif has_hint == best_has_hint and score > best_score:
                should_update = True
            
            if should_update:
                best_sec_idx = sec_idx
                best_score = score
                best_has_hint = has_hint
        
        # Assign to best section only
        if best_sec_idx >= 0:
            best_sec_id = section_ids[best_sec_idx]
            results[best_sec_id]['matched_chunks'].append(chunk)
            results[best_sec_id]['matched_scores'].append(best_score)
    
    # Build final results
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
            chunk_embs = model.encode(chunks, convert_to_tensor=True)
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
                field_emb = model.encode(field_text, convert_to_tensor=True)
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
        'auto_highlight_incomplete_mandatory': incomplete_mandatory,
        'summary': summary,
        'risk_warning': complete_with_risk,
        'recommended_state': recommended_state,
        'audit_log': audit,
        'explainability': logs
    }
