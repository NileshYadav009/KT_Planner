import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List

with open("kt_schema_new.json") as f:
    SCHEMA = json.load(f)["sections"]

SENT_MODEL: SentenceTransformer | None = None
SECTION_EMBEDS: Dict[str, np.ndarray] | None = None

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

def get_sentence_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global SENT_MODEL
    if SENT_MODEL is None:
        SENT_MODEL = SentenceTransformer(model_name)
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
        embeds[s["id"]] = model.encode(text, convert_to_tensor=False)

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
    chunk_embeddings = model.encode(chunks, convert_to_tensor=False, batch_size=32)
    
    # Vectorized cosine similarity computation
    section_ids = list(section_embeds.keys())
    section_emb_list = np.array([section_embeds[sid] for sid in section_ids])
    
    # Compute similarity matrix: (num_chunks, num_sections)
    similarities = np.dot(chunk_embeddings, section_emb_list.T)  # shape: (chunks, sections)
    
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
    chunk_embeddings = model.encode(chunks, convert_to_tensor=False, batch_size=32)
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
        chunk_embs = model.encode(chunks, convert_to_tensor=True)

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
            field_emb = model.encode(field_text, convert_to_tensor=True)

            # compute similarities
            sims = util.cos_sim(field_emb, chunk_embs)[0]
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
