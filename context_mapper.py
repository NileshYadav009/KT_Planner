"""
Sentence-Level Context Mapping Engine (7-Stage Pipeline)

Deterministic structure with AI-assisted semantic classification.
Confidence-based gap detection with screenshot/URL extraction support.

Pipeline stages:
1. Audio confidence validation
2. Sentence segmentation
3. Context classification (semantic)
4. Contextual repair & enhancement
5. Gap detection & coverage
6. Screenshot & URL extraction
7. Structured KT assembly
"""

import re
import json
import inspect
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict

import importlib.util
import os
from policy import (
    CONFIDENCE_ACCEPT_THRESHOLD,
    IMPLEMENTATION_INDICATORS,
    CONCEPTUAL_SECTIONS
)

from runtime_policy import load_policy
from devops_transcription import clean_transcript
from entity_extractor import EntityExtractor
from section_rules import match_section_rules, find_overview_reassignment, apply_rule_overrides

# Detect if sentence_transformers package is installed but avoid importing it at module import time.
# Use the installed package when available; env var can override real embeddings usage.
_SENT_TRANS_SPEC = importlib.util.find_spec("sentence_transformers")
_USE_REAL_EMBEDDINGS = bool(_SENT_TRANS_SPEC and os.getenv("USE_REAL_EMBEDDINGS", "").lower() in ("1", "true", "yes"))
_CROSS_ENCODER = None  # Lazy-loaded cross-encoder for reranking

# Cache the semantic chunking model and util to avoid reloading repeatedly.
_SEMANTIC_CHUNK_MODEL = None
_SEMANTIC_CHUNK_UTIL = None

# Fallback: simple token-set embedding + Jaccard similarity to avoid heavy deps at module import time
if not _USE_REAL_EMBEDDINGS:
    import statistics as np

    class SimpleEmbedding:
        def __init__(self, text: str):
            self.words = set(re.findall(r"\w+", text.lower()))

    class SentenceTransformer:
        def __init__(self, model_name=None):
            pass
        def encode(self, texts, convert_to_tensor=False, normalize_embeddings=False, batch_size=None):
            single = isinstance(texts, str)
            items = [texts] if single else texts
            embs = [SimpleEmbedding(t) for t in items]
            return embs[0] if single else embs

    class util:
        @staticmethod
        def cos_sim(a, b):
            if not hasattr(a, 'words') or not hasattr(b, 'words'):
                return [[0.0]]
            inter = a.words.intersection(b.words)
            union = a.words.union(b.words)
            score = float(len(inter) / len(union)) if union else 0.0
            return [[score]]
else:
    # Real modules will be imported lazily inside ContextClassifier to avoid heavy startup costs
    util = None

# Ensure `np.mean` is available for segmentation even when sentence-transformers
# is present; prefer numpy when installed, otherwise fall back to statistics
try:
    import numpy as np
except Exception:
    import statistics as np

# Sentence tokenization support for better transcript splitting.
try:
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False
    def sent_tokenize(text):
        return re.split(r"(?<=[.!?])\s+", text)

logger = logging.getLogger(__name__)

# Glossary-based conservative corrections for technical terms
try:
    from glossary import apply_glossary_corrections, detect_ambiguous_usage, GLOSSARY
except Exception:
    def apply_glossary_corrections(text, conf, min_confidence_for_correction=0.6):
        return text, False

# ============================================================================
# STAGE 1: Audio Confidence Validation
# ============================================================================

@dataclass
class AudioSegment:
    """Whisper segment with confidence metadata."""
    text: str
    start: float
    end: float
    avg_logprob: float
    speaker: Optional[str] = None
    
    def confidence_score(self, logprob_threshold: float = -1.0) -> float:
        """
        Compute audio confidence (0.0 to 1.0).
        Logprob ranges from ~-2.0 (low) to ~0.0 (high).
        """
        if self.avg_logprob is None:
            return 0.5  # neutral
        # Normalize: logprob -2.0 → 0%, 0.0 → 100%
        conf = max(0.0, min(1.0, (self.avg_logprob + 2.0) / 2.0))
        return float(conf)


# ============================================================================
# STAGE 2: Sentence Segmentation
# ============================================================================

@dataclass
class Sentence:
    """Single sentence with timestamp and speaker info."""
    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    raw_text: Optional[str] = None  # Pre-cleaned version
    audio_confidence: float = 0.5
    segment_ids: List[int] = None  # Which Whisper segments contributed
    explicit_evidence: List[str] = None
    is_inferred: bool = False
    
    def __post_init__(self):
        if self.segment_ids is None:
            self.segment_ids = []
        if self.raw_text is None:
            self.raw_text = self.text
        if self.explicit_evidence is None:
            self.explicit_evidence = []


def segment_sentences(
    audio_segments: List[AudioSegment],
    sentence_ending_pattern: str = r"[.!?]"
) -> List[Sentence]:
    """
    STAGE 2: Segment transcript into sentences.
    
    Maintains:
    - Timestamp mapping
    - Speaker continuity
    - Raw vs cleaned versions
    
    Args:
        audio_segments: List of Whisper segments
        sentence_ending_pattern: Regex for sentence boundaries
    
    Returns:
        List of Sentence objects
    """
    # Concatenate all segments preserving timing
    full_text = " ".join(seg.text for seg in audio_segments)
    sentences = []
    
    # Create char→segment mapping for accurate timestamps
    char_pos = 0
    char_to_segment = {}
    for seg_idx, seg in enumerate(audio_segments):
        seg_text = seg.text
        for i in range(len(seg_text)):
            char_to_segment[char_pos + i] = seg_idx
        char_pos += len(seg_text) + 1  # +1 for space
    
    # Split into sentences using NLTK when available, otherwise regex fallback.
    if HAS_NLTK:
        try:
            sentence_texts = sent_tokenize(full_text.strip())
        except Exception:
            sentence_texts = re.split(r"(?<=[.!?])\s+", full_text.strip())
    else:
        sentence_texts = re.split(r"(?<=[.!?])\s+", full_text.strip())

    # If some sentences are extremely long (e.g., long monologue without punctuation),
    # further split them on commas or after a max length to ensure every word gets a mapping.
    refined_sentences = []
    MAX_SENT_LEN = 240
    for st in sentence_texts:
        if len(st) > MAX_SENT_LEN:
            # Try splitting by commas conservatively
            parts = [p.strip() for p in st.split(',') if p.strip()]
            if len(parts) > 1:
                # Recombine into manageable chunks
                temp = []
                cur = ''
                for p in parts:
                    if len(cur) + len(p) + 2 <= MAX_SENT_LEN:
                        cur = (cur + ' ' + p).strip()
                    else:
                        if cur:
                            temp.append(cur)
                        cur = p
                if cur:
                    temp.append(cur)
                refined_sentences.extend(temp)
                continue
            else:
                # Hard split by max length
                for i in range(0, len(st), MAX_SENT_LEN):
                    refined_sentences.append(st[i:i+MAX_SENT_LEN].strip())
                continue
        refined_sentences.append(st)

    sentence_texts = refined_sentences
    
    current_pos = 0
    for sent_text in sentence_texts:
        if not sent_text.strip():
            continue
        
        # Locate in full text
        start_pos = full_text.find(sent_text, current_pos)
        if start_pos == -1:
            continue
        end_pos = start_pos + len(sent_text)
        current_pos = end_pos
        
        # Get segment range
        start_seg = char_to_segment.get(start_pos, 0)
        end_seg = char_to_segment.get(end_pos - 1, len(audio_segments) - 1)
        
        # Get timestamps from segments
        start_time = audio_segments[start_seg].start
        end_time = audio_segments[end_seg].end
        
        # Average audio confidence across contributing segments
        contrib_segs = audio_segments[start_seg:end_seg + 1]
        avg_conf = np.mean([seg.confidence_score() for seg in contrib_segs]) if contrib_segs else 0.5
        
        # Get speaker if consistent across segments
        speakers = [seg.speaker for seg in contrib_segs if seg.speaker]
        speaker = speakers[0] if speakers and all(s == speakers[0] for s in speakers) else None
        
        # Clean text: normalize whitespace, basic grammar fixes
        cleaned = sent_text.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        
        sent = Sentence(
            text=cleaned,
            start=start_time,
            end=end_time,
            speaker=speaker,
            raw_text=sent_text,
            audio_confidence=float(avg_conf),
            segment_ids=list(range(start_seg, end_seg + 1))
        )
        sentences.append(sent)
    
    return semantic_chunk_sentences(sentences)


def _semantic_similarity(a, b) -> float:
    try:
        similarity_util = _SEMANTIC_CHUNK_UTIL if _SEMANTIC_CHUNK_UTIL is not None else util
        return float(similarity_util.cos_sim(a, b)[0][0])
    except Exception:
        return 0.0


def _merge_sentences(first: Sentence, second: Sentence) -> Sentence:
    merged_text = f"{first.text.strip()} {second.text.strip()}".strip()
    merged_start = min(first.start, second.start)
    merged_end = max(first.end, second.end)
    merged_segment_ids = sorted(set(first.segment_ids + second.segment_ids))
    combined_confidence = (first.audio_confidence * len(first.text.split()) + second.audio_confidence * len(second.text.split())) / max(1, len(merged_text.split()))
    merged_speaker = first.speaker if first.speaker == second.speaker else first.speaker or second.speaker
    return Sentence(
        text=merged_text,
        start=merged_start,
        end=merged_end,
        speaker=merged_speaker,
        raw_text=f"{first.raw_text} {second.raw_text}".strip(),
        audio_confidence=float(combined_confidence),
        segment_ids=merged_segment_ids
    )


def _get_semantic_chunk_model(model_name: str = "all-MiniLM-L6-v2"):
    global _SEMANTIC_CHUNK_MODEL, _SEMANTIC_CHUNK_UTIL
    if _SEMANTIC_CHUNK_MODEL is not None:
        return _SEMANTIC_CHUNK_MODEL

    if _SENT_TRANS_SPEC:
        try:
            from sentence_transformers import SentenceTransformer as ST, util as st_util
            _SEMANTIC_CHUNK_MODEL = ST(model_name)
            _SEMANTIC_CHUNK_UTIL = st_util
            return _SEMANTIC_CHUNK_MODEL
        except Exception:
            pass

    _SEMANTIC_CHUNK_MODEL = SentenceTransformer()
    _SEMANTIC_CHUNK_UTIL = None
    return _SEMANTIC_CHUNK_MODEL


def semantic_chunk_sentences(
    sentences: List[Sentence],
    similarity_threshold: float = 0.45,
    min_merge_length: int = 40
) -> List[Sentence]:
    """Merge adjacent sentences into semantic chunks.

    This step prevents overly granular sentence splitting and preserves
    paragraph-level meaning for KT mapping.
    """
    if len(sentences) <= 1:
        return sentences

    model = _get_semantic_chunk_model()
    encodings = [model.encode(s.text, convert_to_tensor=True) for s in sentences]
    chunks: List[Sentence] = []
    current = sentences[0]
    current_emb = encodings[0]

    for idx in range(1, len(sentences)):
        candidate = sentences[idx]
        candidate_emb = encodings[idx]
        sim = _semantic_similarity(current_emb, candidate_emb)

        should_merge = False
        if sim >= similarity_threshold:
            should_merge = True
        elif len(current.text) < min_merge_length or len(candidate.text) < min_merge_length:
            should_merge = sim >= (similarity_threshold - 0.15)

        if should_merge:
            current = _merge_sentences(current, candidate)
            current_emb = model.encode(current.text, convert_to_tensor=True)
        else:
            chunks.append(current)
            current = candidate
            current_emb = candidate_emb

    chunks.append(current)
    return chunks


# ============================================================================
# STAGE 3: Context Classification Engine
# ============================================================================

@dataclass
@dataclass
class Classification:
    """Semantic classification of a sentence."""
    section_id: str
    section_title: str
    confidence: float
    reason: str  # Why this section was chosen
    similarity_score: float
    entities: Optional[Dict[str, List[str]]] = None  # Extracted entities (tool, environment, owner, etc.)


@dataclass
class ExplainabilityLog:
    """Detailed reasoning for a classification or repair decision."""
    action: str  # "classify" | "repair" | "gap_detect"
    timestamp: str
    sentence_id: int
    section_id: Optional[str]
    reasoning: str  # Detailed explanation
    confidence: float
    alternatives: Optional[List[str]] = None  # Alternative sections considered


@dataclass
class HumanFeedback:
    """Record of human override or correction."""
    sentence_id: int
    original_classification: Optional[str]
    corrected_classification: str
    feedback_timestamp: str
    user: str
    confidence_adjustment: float = 1.0


@dataclass
class ClassifiedSentence:
    """Sentence with class assignments and explainability."""
    sentence: Sentence
    primary_classification: Optional[Classification]
    secondary_classifications: List[Classification]
    multi_section_assignments: Optional[List[str]] = None  # All sections this sentence maps to
    is_unassigned: bool = False
    explainability_log: Optional[ExplainabilityLog] = None
    human_feedback: Optional[HumanFeedback] = None
    active_topic_context: Optional[Dict[str, Any]] = None  # Topic memory context (active_section, topic_confidence, topic_duration)
    
    def __post_init__(self):
        # Build multi-section assignment from primary ONLY
        # Do NOT add secondary classifications to prevent duplicate sentences across sections
        if self.multi_section_assignments is None:
            self.multi_section_assignments = []
            if self.primary_classification:
                self.multi_section_assignments.append(self.primary_classification.section_id)
            # Skip secondary classifications to avoid duplication across sections
            # (keeping this comment for clarity on design decision)
        # Explicit evidence extracted from sentence (e.g., 'connection pool', 'timeout')
        if getattr(self, 'explicit_evidence', None) is None:
            self.explicit_evidence = []
        # Whether the sentence's assigned cause is inferred (no explicit evidence)
        if getattr(self, 'is_inferred', None) is None:
            self.is_inferred = False
    

class ContextClassifier:
    """
    STAGE 3: Semantic classification against KT schema.
    
    Uses sentence-transformers for semantic similarity (embedding stage)
    followed by cross-encoder reranking (reranking stage) for high accuracy.
    
    Flow:
    1. Embed sentence and sections (BAAI/bge-large-en-v1.5)
    2. Find top 5 candidates by cosine similarity
    3. Use cross-encoder to rerank top 5 and select best match
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        similarity_threshold: float = 0.15,
        use_cross_encoder: bool = True
    ):
        """
        Args:
            model_name: HuggingFace model for embeddings (upgraded to BAAI/bge-large-en-v1.5)
            similarity_threshold: Minimum cosine similarity for assignment
            use_cross_encoder: Whether to use cross-encoder for reranking (default True)
        """
        # Lazily import heavy dependencies when sentence-transformers is installed.
        if _SENT_TRANS_SPEC:
            try:
                st = __import__("sentence_transformers")
                import numpy as _np
                globals()['np'] = _np
                globals()['SentenceTransformer'] = st.SentenceTransformer
                globals()['util'] = st.util
                if use_cross_encoder:
                    globals()['CrossEncoder'] = st.CrossEncoder
            except Exception:
                # Fall back to lightweight implementations only if import fails
                pass

        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.section_embeddings = {}
        self.section_metadata = {}
        self._section_hints = {}  # Initialize for keyword matching
        
        # Initialize cross-encoder for reranking if available
        self.cross_encoder = None
        if use_cross_encoder:
            try:
                if 'CrossEncoder' in globals() and globals()['CrossEncoder']:
                    self.cross_encoder = globals()['CrossEncoder']("cross-encoder/ms-marco-MiniLM-L-6-v2")
                    logger.info("Loaded cross-encoder for reranking")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}. Classification will proceed without reranking.")
        
        # Initialize entity extractor for GLiNER-based entity extraction
        self.entity_extractor = EntityExtractor()
    
    def _encode_texts(self, texts, normalize_embeddings: bool = True, convert_to_tensor: bool = True):
        """Encode one or more texts with the classifier model."""
        try:
            return self.model.encode(
                texts,
                convert_to_tensor=convert_to_tensor,
                normalize_embeddings=normalize_embeddings
            )
        except TypeError:
            # Fallback encoder may not support normalize_embeddings
            return self.model.encode(texts, convert_to_tensor=convert_to_tensor)

    def index_schema(self, schema_sections: List[Dict]) -> None:
        """Index schema sections for fast lookup."""
        self._section_hints = {}  # Store hints for keyword matching
        for sec in schema_sections:
            sec_id = sec.get("id")
            title = sec.get("title", sec_id)
            description = sec.get("description", "")
            keywords = sec.get("hints", [])  # Use 'hints' from schema
            
            # Store hints for keyword-based boosting in classify_sentence
            self._section_hints[sec_id] = keywords
            
            # Build rich section text
            section_text = f"{title} {description} {' '.join(keywords)}"
            embedding = self._encode_texts(section_text)
            
            self.section_embeddings[sec_id] = embedding
            self.section_metadata[sec_id] = {
                "title": title,
                "required": sec.get("required", False),
                "description": description
            }
    
    def classify_sentence(
        self,
        sentence: Sentence,
        sent_embedding=None,
        context_embedding=None,
        context_text: Optional[str] = None,
        neighbor_embeddings: List = None,
        top_k: int = 3,
        alpha: float = 0.7,
        beta: float = 0.3
    ) -> ClassifiedSentence:
        """
        Classify sentence against all schema sections.
        
        Uses semantic similarity boosted by keyword matching and surrounding context
        to prevent misplacement while allowing lower thresholds for better coverage.
        
        Returns:
            ClassifiedSentence with primary + secondary classifications
        """
        if not sentence.text.strip():
            return ClassifiedSentence(
                sentence=sentence,
                primary_classification=None,
                secondary_classifications=[],
                is_unassigned=True
            )
        
        classifications, extracted_entities, rule_match = self._score_sentence_candidates(
            sentence,
            sent_embedding=sent_embedding,
            context_embedding=context_embedding,
            context_text=context_text,
            neighbor_embeddings=neighbor_embeddings,
            alpha=alpha,
            beta=beta,
        )

        classifications = self._rerank_candidates(sentence.text, classifications, rule_match)
        
        # Filter by a relaxed threshold for secondary candidates
        primary = None
        secondary = []
        filtered = [c for c in classifications if c.confidence >= (self.similarity_threshold * 0.6)]
        if filtered:
            primary = filtered[0]
            secondary = filtered[1:top_k]
        else:
            # If none passed relaxed threshold, still keep top candidate as low-confidence primary
            if classifications:
                primary = classifications[0]
                secondary = classifications[1:top_k]

        is_unassigned = primary is None
        
        # Build explainability log
        explanation = ""
        alternatives = []
        if primary:
            explanation = f"Matched '{sentence.text[:60]}' to '{primary.section_title}' (score={primary.confidence:.3f})"
            alternatives = [c.section_id for c in secondary[:2]] if secondary else []
        else:
            explanation = f"No section matched above threshold {self.similarity_threshold} for: {sentence.text[:60]}"
            if classifications:
                alternatives = [c.section_id for c in classifications[:3]]
        
        explainability = ExplainabilityLog(
            action="classify",
            timestamp=datetime.utcnow().isoformat() + "Z",
            sentence_id=0,  # Will be set during assembly
            section_id=primary.section_id if primary else None,
            reasoning=explanation,
            confidence=primary.confidence if primary else 0.0,
            alternatives=alternatives
        )
        # Glossary ambiguity detection: do not change classification decision
        # but attach warnings for human review if ambiguous usage detected.
        try:
            warnings = detect_ambiguous_usage(sentence.text)
        except Exception:
            warnings = []

        cs = ClassifiedSentence(
            sentence=sentence,
            primary_classification=primary,
            secondary_classifications=secondary,
            is_unassigned=is_unassigned,
            explainability_log=explainability
        )

        if warnings:
            # attach warnings and note them in explainability
            cs.glossary_warnings = warnings
            # Append to reasoning for transparency
            cs.explainability_log.reasoning = cs.explainability_log.reasoning + " | GLOSSARY_WARNINGS: " + ", ".join(warnings)

        return cs

    def _score_sentence_candidates(
        self,
        sentence: Sentence,
        sent_embedding=None,
        context_embedding=None,
        context_text: Optional[str] = None,
        neighbor_embeddings: List = None,
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> Tuple[List[Classification], Any, Optional[Any]]:
        """Score a sentence against all sections before cross-encoder reranking."""
        extracted_entities = self.entity_extractor.get_context_entities(sentence.text)

        if sent_embedding is None:
            sent_embedding = self._encode_texts(sentence.text)

        classifications = []
        sent_text_lower = (sentence.text or "").lower()

        for sec_id, sec_embedding in self.section_embeddings.items():
            base_sim = float(util.cos_sim(sent_embedding, sec_embedding)[0][0])

            context_sim = 0.0
            if context_embedding is not None:
                try:
                    context_sim = float(util.cos_sim(context_embedding, sec_embedding)[0][0])
                except Exception:
                    context_sim = 0.0
            elif context_text:
                try:
                    context_embedding = self._encode_texts(context_text)
                    context_sim = float(util.cos_sim(context_embedding, sec_embedding)[0][0])
                except Exception:
                    context_sim = 0.0
            elif neighbor_embeddings:
                sims = [float(util.cos_sim(nb, sec_embedding)[0][0]) for nb in neighbor_embeddings]
                if sims:
                    context_sim = float(np.mean(sims))

            keyword_boost = 0.0
            hints = getattr(self, '_section_hints', {}).get(sec_id, [])
            if hints:
                for hint in hints:
                    hint_lower = hint.lower().strip()
                    if not hint_lower:
                        continue
                    if " " in hint_lower:
                        if hint_lower in sent_text_lower:
                            keyword_boost += 0.08
                    else:
                        if re.search(rf"\b{re.escape(hint_lower)}\b", sent_text_lower):
                            keyword_boost += 0.05
                keyword_boost = min(0.35, keyword_boost)

            overview_penalty = 0.0
            if sec_id == "system_overview":
                specialized = find_overview_reassignment(sentence.text)
                if specialized:
                    overview_penalty = 0.25

            combined = float(alpha * base_sim + beta * context_sim + keyword_boost - overview_penalty)
            classifications.append(Classification(
                section_id=sec_id,
                section_title=self.section_metadata[sec_id]["title"],
                confidence=combined,
                similarity_score=base_sim,
                reason=f"Semantic={base_sim:.3f}, Context={context_sim:.3f}, Keywords={keyword_boost:.3f}, Combined={combined:.3f}",
                entities=extracted_entities if extracted_entities else None
            ))

        rule_match = match_section_rules(sentence.text)
        if rule_match:
            meta = self.section_metadata.get(rule_match.section_id, {})
            classifications.insert(0, Classification(
                section_id=rule_match.section_id,
                section_title=meta.get("title", rule_match.section_id),
                confidence=rule_match.confidence,
                similarity_score=rule_match.confidence,
                reason=f"Rule match: {rule_match.matched_pattern}",
                entities=extracted_entities if extracted_entities else None
            ))

        classifications.sort(key=lambda x: x.confidence, reverse=True)
        return classifications, extracted_entities, rule_match

    def _rerank_candidates(
        self,
        sentence_text: str,
        classifications: List[Classification],
        rule_match=None,
        cross_scores=None,
    ) -> List[Classification]:
        """Apply cross-encoder reranking to an existing candidate list."""
        top_candidates = classifications[:5]
        if self.cross_encoder and len(top_candidates) > 0:
            try:
                if cross_scores is None:
                    pairs = [(sentence_text, c.section_title) for c in top_candidates]
                    cross_scores = self.cross_encoder.predict(pairs)
                for i, c in enumerate(top_candidates):
                    if c.reason and c.reason.startswith("Rule match:"):
                        continue
                    cross_score = float(cross_scores[i])
                    normalized_cross = 1.0 / (1.0 + np.exp(-cross_score))
                    c.confidence = 0.4 * c.confidence + 0.6 * normalized_cross
                    c.reason = f"Embedding={c.similarity_score:.3f}, CrossEncoder={cross_score:.3f}, Blended={c.confidence:.3f}"
                top_candidates.sort(key=lambda x: x.confidence, reverse=True)
                classifications = top_candidates + classifications[5:]
                logger.debug(f"Cross-encoder reranking applied. Top match: {classifications[0].section_title} ({classifications[0].confidence:.3f})")
            except Exception as e:
                logger.warning(f"Cross-encoder reranking failed: {e}. Using embedding scores only.")

        if rule_match:
            meta = self.section_metadata.get(rule_match.section_id, {})
            classifications.insert(0, Classification(
                section_id=rule_match.section_id,
                section_title=meta.get("title", rule_match.section_id),
                confidence=rule_match.confidence,
                similarity_score=rule_match.confidence,
                reason=f"Rule match: {rule_match.matched_pattern}",
                entities=None
            ))
            classifications.sort(key=lambda x: x.confidence, reverse=True)

        return classifications


# ============================================================================
# STAGE 4: Contextual Repair & Enhancement
# ============================================================================

@dataclass
class RepairAction:
    """Record of text repair applied."""
    reason: str
    original: str
    improved: str
    confidence: float


class ContextRepair:
    """
    STAGE 4: AI-assisted repair for low-confidence sentences.
    
    Rules:
    - Never hallucinate technical content
    - Preserve meaning
    - Improve grammar only
    - Maintain both original and improved versions
    """
    
    def __init__(self, nlp_model=None, llm_fallback_fn=None):
        """
        Args:
            nlp_model: Optional spaCy model for grammar fixing
            llm_fallback_fn: Optional async function for LLM-based repair
                Args: (original_text, context, section_id) -> str
        """
        self.nlp_model = nlp_model
        self.llm_fallback_fn = llm_fallback_fn  # Hook for Claude/GPT-4 fallback
    
    def should_repair(
        self,
        classified: ClassifiedSentence,
        audio_conf_threshold: float = 0.4,
        semantic_conf_threshold: float = 0.35
    ) -> bool:
        """
        Decide if sentence needs repair.
        
        Trigger conditions:
        1. Low audio confidence
        2. Low semantic confidence
        3. Grammar issues
        """
        triggers = []
        
        if classified.sentence.audio_confidence < audio_conf_threshold:
            triggers.append("low_audio_confidence")
        
        if classified.primary_classification and classified.primary_classification.confidence < semantic_conf_threshold:
            triggers.append("low_semantic_confidence")
        
        # Basic grammar check
        if self._has_grammar_issues(classified.sentence.text):
            triggers.append("grammar_issues")
        
        return len(triggers) > 0
    
    def _has_grammar_issues(self, text: str) -> bool:
        """Detect basic grammar issues."""
        # Missing punctuation at end
        if text and text[-1] not in ".!?":
            return True
        # Multiple consecutive spaces
        if "  " in text:
            return True
        # All lowercase start of sentence
        if text and text[0].islower() and len(text) > 10:
            return True
        return False
    
    def repair(
        self,
        classified: ClassifiedSentence,
        context_sentences: List[ClassifiedSentence]
    ) -> Tuple[str, Optional[RepairAction]]:
        """
        Repair low-confidence sentence using context.
        
        Args:
            classified: Sentence to potentially repair
            context_sentences: Surrounding sentences for context
        
        Returns:
            (improved_text, repair_record)
        """
        original = classified.sentence.text
        # If sentence is protected (failures, rollback, recovery, security), do not modify
        if is_protected_sentence(original):
            return original, None
        improved = original
        repair_record = None
        
        # Stage 1: Basic grammar fixes
        improved = self._basic_grammar_fix(improved)

        # Stage 1.5: Glossary-based conservative corrections (apply when audio confidence low)
        try:
            glossary_improved, did_change = apply_glossary_corrections(improved, classified.sentence.audio_confidence)
            if did_change:
                improved = glossary_improved
        except Exception:
            # Any glossary failures should not break pipeline
            pass

        # Stage 2: Context-based inference (within bounds)
        if classified.primary_classification and classified.primary_classification.confidence < 0.5:
            # Try to improve using surrounding context
            context_text = self._extract_context(classified, context_sentences)
            if context_text:
                improved = self._infer_from_context(improved, context_text)
        
        # Stage 3: LLM fallback (if enabled and needed)
        if self.llm_fallback_fn and (
            classified.sentence.audio_confidence < 0.3
            or (classified.primary_classification and classified.primary_classification.confidence < 0.45)
            or self._has_grammar_issues(original)
        ):
            try:
                section_id = classified.primary_classification.section_id if classified.primary_classification else None
                llm_result = self._try_llm_repair(original, context_sentences, section_id)
                if llm_result and llm_result != improved:
                    improved = llm_result
            except Exception:
                # LLM fallback failed, stick with current improvement
                pass
        
        if improved != original:
            repair_record = RepairAction(
                reason="multi_stage_repair",
                original=original,
                improved=improved,
                confidence=0.85 if self.llm_fallback_fn else 0.7
            )
        
        return improved, repair_record
    
    def _basic_grammar_fix(self, text: str) -> str:
        """Apply basic grammar normalization."""
        # Capitalize first letter if sentence
        result = text.strip()
        if result and result[0].islower():
            result = result[0].upper() + result[1:]
        
        # Add missing period
        if result and result[-1] not in ".!?":
            result = result.rstrip() + "."
        
        # Normalize whitespace
        result = re.sub(r"\s+", " ", result)
        
        return result
    
    def _extract_context(self, target: ClassifiedSentence, neighbors: List[ClassifiedSentence], window: int = 2) -> str:
        """Extract surrounding context for inference."""
        # Find target in neighbors
        target_idx = next((i for i, c in enumerate(neighbors) if c.sentence == target.sentence), -1)
        
        if target_idx == -1:
            return ""
        
        # Get surrounding sentences
        start = max(0, target_idx - window)
        end = min(len(neighbors), target_idx + window + 1)
        
        context_texts = []
        for i in range(start, end):
            if i != target_idx and neighbors[i].sentence.text:
                context_texts.append(neighbors[i].sentence.text)
        
        return " ".join(context_texts)
    
    def _infer_from_context(self, text: str, context: str) -> str:
        """Infer meaning from context (conservative approach)."""
        # Only fix obvious pronoun issues or missing verbs
        # Never add new information not in text or context
        
        # Simple heuristic: if text is very short and context is available,
        # try to extend with context clues
        if len(text) < 20 and context:
            # Extract key topic from context
            words = context.split()[:10]
            common_nouns = [w for w in words if w.istitle() or w.isupper()]
            if common_nouns and not any(noun in text for noun in common_nouns):
                # Could potentially improve, but stick with original to avoid hallucination
                pass
        
        return text
    
    def _try_llm_repair(self, original: str, context: List[ClassifiedSentence], section_id: Optional[str]) -> Optional[str]:
        """Hook for LLM-based repair (Claude/GPT-4)."""
        if not self.llm_fallback_fn:
            return None

        context_text = ' '.join([c.sentence.text for c in context]) if context else ''
        prompt = (
            "You are a conservative text editor for technical transcripts. "
            "Improve the sentence below only for grammar and clarity without changing any technical meaning or introducing new facts. "
            "If the sentence is already clear, return it unchanged.\n\n"
            f"Section: {section_id or 'unknown'}\n"
            f"Context: {context_text}\n"
            f"Original sentence: {original}\n"
            "Return only the repaired sentence."
        )

        try:
            # Support both legacy call signatures and prompt/metadata-based hooks.
            sig = inspect.signature(self.llm_fallback_fn)
            if len(sig.parameters) >= 4:
                return self.llm_fallback_fn(original, context_text, section_id, temperature=0.2)

            result = self.llm_fallback_fn(
                prompt,
                metadata={
                    "context": context_text,
                    "section_id": section_id,
                    "temperature": 0.2
                }
            )
            return result
        except Exception:
            return None


# ============================================================================
# STAGE 5: Gap Detection
# ============================================================================

@dataclass
class TopicBlock:
    """Contiguous group of related sentences forming a coherent topic/paragraph."""
    section_id: str
    topic_title: Optional[str]  # e.g., "Deployment Process"
    sentences: List[Sentence]  # Ordered, consecutive sentences
    start_time: float  # Timestamp of first sentence
    end_time: float  # Timestamp of last sentence
    confidence_score: float  # Block-level confidence (mean of sentences)
    duration: int  # Number of consecutive sentences in block
    speaker: Optional[str] = None  # Primary speaker for block
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "topic_title": self.topic_title,
            "sentence_count": self.duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence_score,
            "sentences": [
                {
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "speaker": s.speaker,
                    "audio_confidence": s.audio_confidence
                }
                for s in self.sentences
            ]
        }

@dataclass
class SectionCoverage:
    """Coverage metrics for a KT section."""
    section_id: str
    section_title: str
    required: bool
    status: str  # "missing" | "weak" | "covered"
    block_count: int  # Number of coherent topic blocks
    sentence_count: int  # Total sentences across all blocks
    confidence_score: float
    risk_score: float
    coverage_score: float = 0.0
    blocks: List[TopicBlock] = field(default_factory=list)  # Topic-coherent blocks instead of scattered sentences


def semantic_coverage_score(
    sentences: List[Sentence],
    block_confidences: List[float],
    section_meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute a semantic coverage score for a section.

    The score favors coherent, actionable, and complete coverage over raw sentence count.
    """
    import math

    if not sentences:
        return {
            "status": "missing",
            "scs": 0.0,
            "dimensions": {
                "confidence": 0.0,
                "depth": 0.0,
                "actionability": 0.0,
                "completeness": 0.0,
            },
        }

    strong_scores = [score for score in block_confidences if score >= 0.35]
    if not strong_scores:
        confidence = float(np.mean(block_confidences)) if block_confidences else 0.0
    else:
        confidence = float(np.mean(sorted(strong_scores, reverse=True)[:3]))

    total_words = sum(len(sentence.text.split()) for sentence in sentences)
    min_words = int(section_meta.get("min_words", 30))
    depth = min(1.0, math.log1p(total_words) / math.log1p(min_words * 3))

    combined = " ".join(sentence.text for sentence in sentences)
    actionability_signals = [
        bool(re.search(r"https?://", combined)),
        bool(re.search(r"\b\d{1,5}\s*(min|hour|sec)s?\b", combined, re.I)),
        bool(re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", combined)),
        bool(re.search(r"`[^`]+`|kubectl|helm|terraform|argocd|docker", combined, re.I)),
        bool(re.search(r"@|#[a-z\-]+", combined)),
        bool(re.search(r"\b(step \d+|first|second|third|finally)\b", combined, re.I)),
    ]
    actionability = sum(actionability_signals) / len(actionability_signals)

    sub_topics = section_meta.get("sub_topics") or section_meta.get("keywords") or []
    if sub_topics:
        covered_topics = sum(
            1 for topic in sub_topics
            if any(str(topic).lower() in sentence.text.lower() for sentence in sentences)
        )
        completeness = covered_topics / len(sub_topics)
    else:
        completeness = 1.0

    scs = (
        confidence * 0.30 +
        depth * 0.25 +
        actionability * 0.25 +
        completeness * 0.20
    )

    if scs >= 0.65:
        status = "covered"
    elif scs >= 0.40:
        status = "weak"
    else:
        status = "missing"

    return {
        "status": status,
        "scs": round(float(scs), 3),
        "dimensions": {
            "confidence": round(float(confidence), 3),
            "depth": round(float(depth), 3),
            "actionability": round(float(actionability), 3),
            "completeness": round(float(completeness), 3),
        },
    }
    

def detect_gaps(
    classified_sentences: List[ClassifiedSentence],
    schema_sections: List[Dict],
    weak_threshold: int = 1,
    required_threshold: int = 2
) -> Dict[str, SectionCoverage]:
    """
    STAGE 5: Detect coverage gaps and compute risk.
    
    NEW APPROACH: Group consecutive sentences into TopicBlocks to preserve paragraph continuity.
    
    Rules:
    - 0 blocks → missing
    - 1 block → weak
    - 2+ blocks → covered
    """
    coverage = {}
    
    # PASS 1: Group consecutive sentences by section to create candidate blocks
    candidate_blocks: Dict[str, List[TopicBlock]] = defaultdict(list)
    
    current_block_sentences: List[Sentence] = []
    current_block_section: Optional[str] = None
    current_block_indices: List[int] = []
    
    for idx, cs in enumerate(classified_sentences):
        assigned_section = None
        if cs.primary_classification:
            assigned_section = cs.primary_classification.section_id
        
        # If section changes or no assignment, finalize current block
        if assigned_section != current_block_section and current_block_sentences:
            # Finalize the block
            if current_block_section:
                confidences = []
                for idx_val in current_block_indices:
                    cs_block = classified_sentences[idx_val]
                    if cs_block.primary_classification:
                        confidences.append(cs_block.primary_classification.confidence)
                    else:
                        confidences.append(cs_block.sentence.audio_confidence)
                block_confidence = float(np.mean(confidences)) if confidences else 0.0
                
                block = TopicBlock(
                    section_id=current_block_section,
                    topic_title=None,  # Could infer from first sentence later
                    sentences=current_block_sentences.copy(),
                    start_time=current_block_sentences[0].start,
                    end_time=current_block_sentences[-1].end,
                    confidence_score=block_confidence,
                    duration=len(current_block_sentences),
                    speaker=current_block_sentences[0].speaker if current_block_sentences else None
                )
                candidate_blocks[current_block_section].append(block)
            
            # Start new block
            current_block_sentences = []
            current_block_section = assigned_section
            current_block_indices = []
        
        # Add sentence to current block
        if assigned_section:
            current_block_sentences.append(cs.sentence)
            current_block_indices.append(idx)
    
    # Finalize last block
    if current_block_sentences and current_block_section:
        confidences = []
        for idx_val in current_block_indices:
            cs_block = classified_sentences[idx_val]
            if cs_block.primary_classification:
                confidences.append(cs_block.primary_classification.confidence)
            else:
                confidences.append(cs_block.sentence.audio_confidence)
        block_confidence = float(np.mean(confidences)) if confidences else 0.0

        block = TopicBlock(
            section_id=current_block_section,
            topic_title=None,
            sentences=current_block_sentences.copy(),
            start_time=current_block_sentences[0].start,
            end_time=current_block_sentences[-1].end,
            confidence_score=block_confidence,
            duration=len(current_block_sentences),
            speaker=current_block_sentences[0].speaker if current_block_sentences else None
        )
        candidate_blocks[current_block_section].append(block)
    
    # PASS 2: Compute coverage metrics per section
    for sec in schema_sections:
        sec_id = sec.get("id")
        sec_title = sec.get("title", sec_id)
        sec_required = sec.get("required", False)
        
        blocks = candidate_blocks.get(sec_id, [])
        block_count = len(blocks)
        total_sentences = sum(b.duration for b in blocks)
        block_sentences = [sentence for block in blocks for sentence in block.sentences]
        block_confidences = [block.confidence_score for block in blocks]
        coverage_analysis = semantic_coverage_score(block_sentences, block_confidences, sec)
        
        # Compute section-level confidence as mean of block confidences
        if blocks:
            confidence = float(np.mean([b.confidence_score for b in blocks]))
        else:
            confidence = 0.0
        
        # Determine status directly from the semantic coverage score.
        status = coverage_analysis["status"]
        if status == "covered":
            risk = max(0.0, 0.15 - coverage_analysis["scs"] * 0.1)
        elif status == "weak":
            risk = 0.6 if sec_required else 0.2
        else:
            risk = 1.0 if sec_required else 0.5
        
        coverage[sec_id] = SectionCoverage(
            section_id=sec_id,
            section_title=sec_title,
            required=sec_required,
            status=status,
            block_count=block_count,
            sentence_count=total_sentences,
            confidence_score=confidence,
            risk_score=float(risk),
            coverage_score=float(coverage_analysis["scs"]),
            blocks=blocks
        )
    
    return coverage


# ============================================================================
# STAGE 6: Screenshot & URL Extraction
# ============================================================================

@dataclass
class ExtractedAsset:
    """Screenshot or URL found during processing."""
    asset_type: str  # "screenshot" | "url"
    content: str  # path or URL
    sentence_ids: List[int]  # Associated sentence timestamps
    detected_component: Optional[str] = None  # e.g., "grafana", "jenkins"
    timestamp: Optional[float] = None


KNOWN_DASHBOARDS = {
    "grafana": r"grafana|dashboard",
    "jenkins": r"jenkins|build|pipeline",
    "kubernetes": r"kubernetes|k8s|kubectl|dashboard",
    "github": r"github|repository|repo",
    "aws": r"aws|console|ec2|s3|cloudwatch",
    "datadog": r"datadog|monitoring",
}


# Evidence keywords for extracting explicit mentions of root causes
EVIDENCE_KEYWORDS = [
    r"connection pool", r"connection pool exhausted", r"connection refused", r"timeout", r"deadlock", r"out of connections",
    r"db", r"database", r"sql", r"connection leak", r"max_connections", r"too many connections"
]


def extract_explicit_evidence(text: str):
    """Return list of evidence phrases found in text (case-insensitive)."""
    found = []
    if not text:
        return found
    t = text.lower()
    for kw in EVIDENCE_KEYWORDS:
        try:
            if re.search(kw, t):
                found.append(kw)
        except Exception:
            # fallback literal check
            if kw in t:
                found.append(kw)
    return found


def is_causal_statement(text: str) -> bool:
    if not text:
        return False
    return re.search(r"\b(when|because|due to|caused by|after|once|as a result|lead to|leads to)\b", text.lower()) is not None


# Protected content keywords: never summarize or alter these
PROTECTED_KEYWORDS = [
    r"critical failure", r"failure", r"failures", r"rollback", r"rollback plan",
    r"restore", r"recovery", r"recovery procedure", r"backup", r"backup strategy",
    r"security", r"vulnerability", r"breach", r"incident response"
]


def is_protected_sentence(text: str) -> bool:
    """Return True if sentence contains terms that must be preserved verbatim."""
    if not text:
        return False
    t = text.lower()
    for kw in PROTECTED_KEYWORDS:
        try:
            if re.search(kw, t):
                return True
        except Exception:
            if kw in t:
                return True
    return False


def extract_urls_and_assets(
    sentences: List[Sentence],
    video_path: Optional[str] = None,
    enable_ocr: bool = False
) -> List[ExtractedAsset]:
    """
    STAGE 6: Extract URLs and auto-detect dashboard screenshots.
    
    Implementation options:
    - URL regex matching in text (enabled by default)
    - Component keyword detection (enabled by default)
    - OCR for burned-in URLs (optional, requires pytesseract)
    - Browser extension integration (future)
    - Headless Chromium capture (future)
    
    Args:
        sentences: List of sentences to scan
        video_path: Optional path to video file for frame extraction
        enable_ocr: Flag to enable OCR (requires pytesseract)
    """
    assets = []
    
    # Extract URLs from sentences
    url_pattern = r"https?://[^\s]+"
    for sent in sentences:
        urls = re.findall(url_pattern, sent.text)
        for url in urls:
            # Detect component type
            component = None
            for dash_name, dash_pattern in KNOWN_DASHBOARDS.items():
                if re.search(dash_pattern, url, re.IGNORECASE):
                    component = dash_name
                    break
            
            assets.append(ExtractedAsset(
                asset_type="url",
                content=url,
                sentence_ids=[],
                detected_component=component,
                timestamp=sent.start
            ))
    
    # Detect dashboard mentions in sentence text
    # NOTE: Screenshot extraction/asset capture disabled by request.
    # for sent in sentences:
    #     for dash_name, dash_pattern in KNOWN_DASHBOARDS.items():
    #         if re.search(dash_pattern, sent.text, re.IGNORECASE):
    #             assets.append(ExtractedAsset(
    #                 asset_type="screenshot_candidate",
    #                 content=f"dashboard:{dash_name}",
    #                 sentence_ids=[],
    #                 detected_component=dash_name,
    #                 timestamp=sent.start
    #             ))
    
    # OCR-based URL detection (optional, enterprise feature)
    if enable_ocr and video_path:
        try:
            # Placeholder: OCR implementation
            # In production: use pytesseract + ffmpeg frame extraction
            # assets.extend(_extract_urls_via_ocr(video_path))
            pass
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
    
    return assets


# ============================================================================
# STAGE 7: Structured KT Assembly
# ============================================================================

@dataclass
class StructuredKT:
    """Final assembled KT output."""
    job_id: str
    transcript: str
    sentences: List[Sentence]
    classified_sentences: List["ClassifiedSentence"]
    coverage: Dict[str, SectionCoverage]
    section_content: Dict[str, Dict[str, Any]]
    missing_required_sections: List[str]
    unassigned_sentences: List[Sentence]
    assets: List[ExtractedAsset]
    overall_coverage_percent: float
    overall_risk_score: float
    timestamp: str
    explainability_logs: Optional[List[ExplainabilityLog]] = None
    human_feedback: Optional[List[HumanFeedback]] = None
    parent_job_id: Optional[str] = None  # For incremental KT (session 2+)
    cross_references: Optional[List[Dict[str, Any]]] = None
    topic_memory_contexts: Optional[List[Dict[str, Any]]] = None  # Topic memory context for each sentence

def assemble_kt(
    job_id: str,
    transcript: str,
    classified_sentences: List[ClassifiedSentence],
    coverage: Dict[str, SectionCoverage],
    assets: List[ExtractedAsset],
    repaired_map: Dict[int, Tuple[str, Optional[RepairAction]]],
    topic_memory_contexts: Optional[List[Dict[str, Any]]] = None
) -> StructuredKT:
    """
    STAGE 7: Assemble final KT structure.
    
    Returns comprehensive KT object with:
    - Topic blocks per section (NEW: preserves paragraph continuity)
    - Sentence lists per section (for backwards compatibility)
    - Enhanced text (repaired versions)
    - Associated screenshots
    - Confidence scores
    - Missing section flags
    """
    section_content = {}
    unassigned = []
    missing_required = []
    cross_refs = []
    
    # Track seen sentences per section to avoid duplicates caused by transcript repetition
    seen_texts_per_section: Dict[str, set] = {}

    for idx, cs in enumerate(classified_sentences):
        # Build enhanced text
        enhanced_text, repair_action = repaired_map.get(idx, (cs.sentence.text, None))
        
        # Detect referential sentences like "as I mentioned earlier", "this connects to..."
        def is_referential_sentence(text: str) -> bool:
            if not text:
                return False
            patterns = [
                r"\bas i mentioned\b",
                r"\bas noted earlier\b",
                r"\bas stated earlier\b",
                r"\bthis connects to\b",
                r"\bsee above\b",
                r"\brefer to the previous\b",
                r"\b(as described|as explained) earlier\b",
                r"\b(as I said|as I mentioned)\b"
            ]
            t = text.lower()
            for p in patterns:
                if re.search(p, t):
                    return True
            return False

        referential = is_referential_sentence(cs.sentence.text)

        if cs.is_unassigned:
            # Deduplicate unassigned sentences as well
            norm = (cs.sentence.text or '').strip()
            if norm and norm not in [s.text for s in unassigned]:
                unassigned.append(cs.sentence)
            continue
        # Handle referential sentences: link to previous relevant section and avoid duplication
        if referential and cs.primary_classification:
            # Try to find prior sentence in same section or best match
            target_section = cs.primary_classification.section_id
            found = None
            # search backwards for a sentence already placed in that section
            for j in range(idx - 1, -1, -1):
                prev = classified_sentences[j]
                if prev.primary_classification and prev.primary_classification.section_id == target_section:
                    found = prev
                    break
            # fallback: find any earlier sentence with significant token overlap
            if not found:
                cur_words = set(re.findall(r"\w+", (cs.sentence.text or '').lower()))
                for j in range(idx - 1, -1, -1):
                    prev = classified_sentences[j]
                    prev_words = set(re.findall(r"\w+", (prev.sentence.text or '').lower()))
                    if not cur_words or not prev_words:
                        continue
                    inter = cur_words.intersection(prev_words)
                    if len(inter) >= max(1, min(5, int(0.3 * len(cur_words)))):
                        found = prev
                        break

            if found:
                # Record a cross-reference entry under the found section
                dest_sec = found.primary_classification.section_id if found.primary_classification else target_section
                entry = {
                    "from_sentence": cs.sentence.text,
                    "from_index": idx,
                    "to_section": dest_sec,
                    "to_sentence": found.sentence.text,
                    "note": "referential_link"
                }
                cross_refs.append(entry)
                # Also annotate target section content if present
                if dest_sec in section_content:
                    section_content[dest_sec].setdefault("cross_references", []).append(entry)
                continue
        elif cs.primary_classification:
            # Place sentence into all assigned sections (multi-label support)
            assigned_secs = cs.multi_section_assignments or []
            if not assigned_secs:
                # fallback to primary only
                assigned_secs = [cs.primary_classification.section_id]

            for sec_id in assigned_secs:
                if sec_id not in section_content:
                    # find section title if available
                    title = cs.primary_classification.section_title if cs.primary_classification and cs.primary_classification.section_id == sec_id else sec_id
                    section_content[sec_id] = {
                        "section_id": sec_id,
                        "section_title": title,
                        "sentences": [],
                        "enhanced_texts": [],
                        "repair_actions": [],
                        "screenshots": [],
                        "blocks": [],  # NEW: preserve topic blocks
                        "confidence": 0.0,
                        "sentence_count": 0
                    }
                    seen_texts_per_section[sec_id] = set()

                # Deduplicate based on normalized text
                norm_text = (cs.sentence.text or '').strip()
                norm_key = re.sub(r"\s+", " ", norm_text).lower()
                if norm_key in seen_texts_per_section.get(sec_id, set()):
                    # Already present — avoid duplicating content; add cross-reference
                    # Find original text index (best-effort)
                    original_text = None
                    for s in section_content.get(sec_id, {}).get("sentences", []):
                        ot = s.get("text", "").strip()
                        if re.sub(r"\s+", " ", ot).lower() == norm_key:
                            original_text = ot
                            break
                    cref = {
                        "from_sentence": cs.sentence.text,
                        "from_index": idx,
                        "to_section": sec_id,
                        "to_sentence": original_text,
                        "note": "duplicate_reference"
                    }
                    cross_refs.append(cref)
                    section_content[sec_id].setdefault("cross_references", []).append(cref)
                    continue
                seen_texts_per_section.setdefault(sec_id, set()).add(norm_key)

                section_content[sec_id]["sentences"].append({
                    "text": cs.sentence.text,
                    "start": cs.sentence.start,
                    "end": cs.sentence.end,
                    "speaker": cs.sentence.speaker,
                    "audio_confidence": cs.sentence.audio_confidence,
                    "assigned_sections": list(cs.multi_section_assignments or []),
                    "preserve_verbatim": bool(is_protected_sentence(cs.sentence.text))
                })
                section_content[sec_id]["enhanced_texts"].append(enhanced_text)

                if repair_action:
                    section_content[sec_id]["repair_actions"].append({
                        "reason": repair_action.reason,
                        "original": repair_action.original,
                        "improved": repair_action.improved
                    })
    
    # NEW: Add topic blocks to section_content to preserve paragraph continuity
    for sec_id, cov in coverage.items():
        if sec_id not in section_content:
            section_content[sec_id] = {
                "section_id": sec_id,
                "section_title": cov.section_title,
                "sentences": [],
                "enhanced_texts": [],
                "repair_actions": [],
                "screenshots": [],
                "blocks": [],
                "confidence": cov.confidence_score,
                "sentence_count": 0
            }
        
        # Add blocks to section_content - this preserves the original paragraph grouping
        section_content[sec_id]["blocks"] = [b.to_dict() for b in cov.blocks]
    
    # Compute section confidences
    for sec_id, content in section_content.items():
        if content["sentences"]:
            conf = np.mean([s["audio_confidence"] for s in content["sentences"]])
            content["confidence"] = float(conf)
            content["sentence_count"] = len(content["sentences"])
    
    # Identify missing required sections
    for sec_id, cov in coverage.items():
        if cov.required and cov.status == "missing":
            missing_required.append(sec_id)
    
    # Overall metrics
    covered = sum(1 for c in coverage.values() if c.status in ("covered", "weak"))
    overall_coverage = 100.0 * covered / len(coverage) if coverage else 0.0
    overall_risk = np.mean([c.risk_score for c in coverage.values()]) if coverage else 0.0
    
    return StructuredKT(
        job_id=job_id,
        transcript=transcript,
        sentences=[cs.sentence for cs in classified_sentences],
        classified_sentences=classified_sentences,
        coverage=coverage,
        section_content=section_content,
        missing_required_sections=missing_required,
        unassigned_sentences=unassigned,
        assets=assets,
        overall_coverage_percent=float(overall_coverage),
        overall_risk_score=float(overall_risk),
        timestamp=datetime.utcnow().isoformat() + "Z",
        cross_references=cross_refs or [],
        topic_memory_contexts=topic_memory_contexts
    )


# ============================================================================
# Topic Memory & Continuity Tracking
# ============================================================================

@dataclass
@dataclass
class TopicTransition:
    """Record of a topic/section change."""
    from_section: Optional[str]
    to_section: Optional[str]
    sentence_index: int
    confidence: float
    reason: str  # e.g., "high_confidence_switch", "related_section", "context_decay"
    timestamp: str


class TopicMemory:
    """
    Enhanced context memory for maintaining semantic continuity across sentences.
    
    Tracks:
    - Active topic with confidence and duration
    - Context window (previous sentences for reference)
    - Topic stack (nested topics)
    - Transition history for debugging
    - Related sections for topic continuity
    """
    
    def __init__(self, max_context_window: int = 5):
        self.active_section_id: Optional[str] = None
        self.topic_confidence: float = 0.0
        self.topic_duration: int = 0
        self.related_sections: List[str] = []
        
        # NEW: Context window for multi-sentence context
        self.context_window: List[Tuple[Optional[str], float]] = []  # (section_id, confidence) tuples
        self.max_context_window = max_context_window
        
        # NEW: Topic stack for nested topics
        self.topic_stack: List[Tuple[Optional[str], float]] = []  # Stack of (section_id, confidence)
        
        # NEW: Transition history for debugging
        self.transitions: List[TopicTransition] = []
        
        # NEW: Decay tracking for unclassified sentences
        self.unclassified_count: int = 0
    
    def update(
        self,
        section_id: Optional[str],
        confidence: float,
        related: List[str] = None,
        sentence_index: int = 0
    ):
        """Update topic memory with new classified sentence.
        
        Args:
            section_id: Section the sentence was classified to
            confidence: Semantic confidence of the classification
            related: Related sections that should extend topic duration
            sentence_index: Index in the full transcript (for tracking)
        """
        reason = "no_change"
        
        if section_id is None:
            # Unclassified sentence; decay topic memory slightly
            self.topic_duration = max(0, self.topic_duration - 1)
            self.topic_confidence *= 0.9
            self.unclassified_count += 1
            reason = "unclassified_decay"
            # If too many unclassified sentences, pop the topic stack
            if self.unclassified_count > 3 and self.topic_stack:
                self.active_section_id, self.topic_confidence = self.topic_stack.pop()
                reason = "stack_recovery"
        else:
            # Reset unclassified counter
            self.unclassified_count = 0
            
            if section_id == self.active_section_id:
                # Continuing in same topic
                self.topic_duration += 1
                self.topic_confidence = max(self.topic_confidence, confidence)
                reason = "topic_continuation"
            elif related and section_id in related:
                # Switching to related section (e.g., deployment -> rollback)
                # Push current topic onto stack for recovery
                if self.active_section_id:
                    self.topic_stack.append((self.active_section_id, self.topic_confidence))
                self.active_section_id = section_id
                self.topic_duration = 1
                self.topic_confidence = confidence
                reason = "related_section_switch"
            elif confidence > self.topic_confidence:
                # Switching to new topic only if confidence is significantly higher
                if self.topic_confidence > 0:
                    threshold = self.topic_confidence + 0.15
                else:
                    threshold = 0.2
                
                if confidence > threshold:
                    # High-confidence topic switch: push current to stack
                    if self.active_section_id:
                        self.topic_stack.append((self.active_section_id, self.topic_confidence))
                    self.active_section_id = section_id
                    self.topic_duration = 1
                    self.topic_confidence = confidence
                    reason = "high_confidence_switch"
                else:
                    # Stay in current topic
                    self.topic_duration += 1
                    reason = "threshold_hold"
            else:
                # Low confidence sentence; stay in current topic
                self.topic_duration += 1
                self.topic_confidence *= 0.95
                reason = "low_confidence_hold"
        
        # Update context window
        self._update_context_window(section_id, confidence)
        
        # Record transition if topic changed
        if section_id != self.active_section_id or section_id != (self.context_window[-2][0] if len(self.context_window) > 1 else None):
            trans = TopicTransition(
                from_section=self.active_section_id,
                to_section=section_id,
                sentence_index=sentence_index,
                confidence=confidence,
                reason=reason,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
            self.transitions.append(trans)
            if len(self.transitions) > 100:
                self.transitions = self.transitions[-100:]  # Keep last 100
    
    def _update_context_window(self, section_id: Optional[str], confidence: float):
        """Maintain a sliding window of recent section assignments."""
        self.context_window.append((section_id, confidence))
        if len(self.context_window) > self.max_context_window:
            self.context_window.pop(0)
    
    def get_context_sections(self, depth: int = 3) -> List[str]:
        """Get the most recent section assignments from context window."""
        sections = []
        for section_id, _ in self.context_window[-depth:]:
            if section_id and section_id not in sections:
                sections.append(section_id)
        return sections
    
    def get_topic_boost(self, section_id: Optional[str], base_confidence: float) -> float:
        """Compute confidence boost based on active topic and context.
        
        Args:
            section_id: Section being evaluated
            base_confidence: Base semantic confidence
        
        Returns:
            Boosted confidence score
        """
        if section_id is None:
            return base_confidence
        
        # Boost if matching active topic with momentum
        if section_id == self.active_section_id and self.topic_duration > 0:
            boost = min(0.20, self.topic_confidence * 0.4)
            return base_confidence + boost
        
        # Boost if in related sections
        if self.active_section_id and section_id in self.related_sections:
            boost = min(0.10, 0.05 + self.topic_confidence * 0.2)
            return base_confidence + boost
        
        # Boost if recently seen in context window
        recent_sections = self.get_context_sections(depth=2)
        if section_id in recent_sections:
            boost = 0.08
            return base_confidence + boost
        
        return base_confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for inclusion in output/logging."""
        return {
            "active_section": self.active_section_id,
            "topic_confidence": self.topic_confidence,
            "topic_duration": self.topic_duration,
            "context_window": self.get_context_sections(),
            "stack_depth": len(self.topic_stack),
            "recent_transitions": [
                {
                    "from": t.from_section,
                    "to": t.to_section,
                    "reason": t.reason
                }
                for t in self.transitions[-3:]
            ]
        }


# ============================================================================
# Main Pipeline Orchestrator
# ============================================================================

class ContextMappingPipeline:
    """Orchestrates all 7 stages of context mapping."""
    
    def __init__(
        self,
        schema_sections: List[Dict],
        similarity_threshold: float = 0.30,
        audio_confidence_threshold: float = 0.4,
        llm_fallback_fn=None,
        enterprise_mapper_enabled: bool = True
    ):
        self.schema_sections = schema_sections
        self.classifier = ContextClassifier(similarity_threshold=similarity_threshold)
        self.classifier.index_schema(schema_sections)
        self.repair = ContextRepair(llm_fallback_fn=llm_fallback_fn)
        self.audio_conf_threshold = audio_confidence_threshold
        self.enterprise_mapper_enabled = bool(enterprise_mapper_enabled)
    
    def process(
        self,
        job_id: str,
        transcript: str,
        audio_segments: List[Dict]
    ) -> StructuredKT:
        """
        Run full pipeline.
        
        Args:
            job_id: Unique identifier
            transcript: Full transcript text
            audio_segments: Whisper segments with confidence
        
        Returns:
            StructuredKT object
        """
        # STAGE 1: Convert to AudioSegment objects (already done by Whisper)
        segments = [
            AudioSegment(
                text=clean_transcript(seg.get("text", "")),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                avg_logprob=seg.get("avg_logprob", -1.0),
                speaker=seg.get("speaker")
            )
            for seg in audio_segments
        ]
        
        # STAGE 1.5: Pre-clean transcript segments before sentence segmentation
        logger.info("Stage 1.5: Cleaned transcript segments before classification")

        # STAGE 2: Segment sentences
        sentences = segment_sentences(segments)
        logger.info(f"Stage 2: Segmented into {len(sentences)} sentences")
        
        # STAGE 3: Classify sentences
        # Precompute embeddings for sentences to allow context-aware scoring
        texts = [s.text for s in sentences]
        embeddings = self.classifier._encode_texts(texts)

        # Precompute context embeddings for overlapping sentence windows
        context_texts = []
        for i, s in enumerate(sentences):
            window_size = 2
            start = max(0, i - window_size)
            end = min(len(sentences), i + window_size + 1)
            context_texts.append(" ".join([sentences[j].text for j in range(start, end)]))
        context_embeddings = self.classifier._encode_texts(context_texts)

        # Initialize topic memory for semantic continuity
        topic_memory = TopicMemory()
        
        # Build section-to-keywords map for topic-related section detection
        related_sections_map = {}  # section_id -> list of related section ids
        for sec in self.schema_sections:
            sec_id = sec.get("id")
            keywords = set((sec.get("keywords") or []))
            # Find sections with overlapping keywords
            related = []
            for other_sec in self.schema_sections:
                other_id = other_sec.get("id")
                if other_id != sec_id:
                    other_keywords = set((other_sec.get("keywords") or []))
                    if keywords & other_keywords:  # Intersection
                        related.append(other_id)
            related_sections_map[sec_id] = related

        sentence_candidates = []
        sentence_rule_matches = []
        extracted_entities_cache = []
        for i, s in enumerate(sentences):
            classifications, extracted_entities, rule_match = self.classifier._score_sentence_candidates(
                s,
                sent_embedding=embeddings[i],
                context_embedding=context_embeddings[i]
            )
            sentence_candidates.append(classifications)
            sentence_rule_matches.append(rule_match)
            extracted_entities_cache.append(extracted_entities)

        batch_cross_scores = []
        batch_pairs = []
        batch_offsets = []
        if self.classifier.cross_encoder:
            for i, classifications in enumerate(sentence_candidates):
                top_candidates = classifications[:5]
                batch_offsets.append(len(batch_pairs))
                batch_pairs.extend([(sentences[i].text, c.section_title) for c in top_candidates])
            if batch_pairs:
                try:
                    batch_cross_scores = self.classifier.cross_encoder.predict(batch_pairs)
                except Exception as e:
                    logger.warning(f"Cross-encoder batch reranking failed: {e}. Using embedding scores only.")
                    batch_cross_scores = []

        classified_sentences = []
        for i, s in enumerate(sentences):
            classifications = sentence_candidates[i]
            rule_match = sentence_rule_matches[i]
            extracted_entities = extracted_entities_cache[i]
            top_candidates = classifications[:5]
            if batch_cross_scores is not None and len(batch_cross_scores) > 0:
                start_idx = batch_offsets[i]
                end_idx = start_idx + len(top_candidates)
                relevant_scores = batch_cross_scores[start_idx:end_idx]
                classifications = self.classifier._rerank_candidates(
                    s.text,
                    classifications,
                    rule_match=rule_match,
                    cross_scores=relevant_scores
                )
            else:
                classifications = self.classifier._rerank_candidates(
                    s.text,
                    classifications,
                    rule_match=rule_match,
                    cross_scores=None
                )

            # Recreate the final ClassifiedSentence from the reranked candidates
            filtered = [c for c in classifications if c.confidence >= (self.classifier.similarity_threshold * 0.6)]
            primary = filtered[0] if filtered else (classifications[0] if classifications else None)
            secondary = filtered[1:3] if filtered else (classifications[1:3] if len(classifications) > 1 else [])
            is_unassigned = primary is None
            explanation = ""
            alternatives = []
            if primary:
                explanation = f"Matched '{s.text[:60]}' to '{primary.section_title}' (score={primary.confidence:.3f})"
                alternatives = [c.section_id for c in secondary[:2]] if secondary else []
            else:
                explanation = f"No section matched above threshold {self.classifier.similarity_threshold} for: {s.text[:60]}"
                if classifications:
                    alternatives = [c.section_id for c in classifications[:3]]

            explainability = ExplainabilityLog(
                action="classify",
                timestamp=datetime.utcnow().isoformat() + "Z",
                sentence_id=0,
                section_id=primary.section_id if primary else None,
                reasoning=explanation,
                confidence=primary.confidence if primary else 0.0,
                alternatives=alternatives
            )
            cs = ClassifiedSentence(
                sentence=s,
                primary_classification=primary,
                secondary_classifications=secondary,
                is_unassigned=is_unassigned,
                explainability_log=explainability
            )
            if extracted_entities:
                cs.primary_classification.entities = extracted_entities if cs.primary_classification else None
            
            # Apply topic memory boost to primary classification
            if cs.primary_classification:
                base_conf = cs.primary_classification.confidence
                boosted_conf = topic_memory.get_topic_boost(
                    cs.primary_classification.section_id,
                    base_conf
                )
                cs.primary_classification.confidence = boosted_conf
                # Re-sort secondary classifications by boosted confidence
                if cs.secondary_classifications:
                    for sec_class in cs.secondary_classifications:
                        sec_class.confidence = topic_memory.get_topic_boost(
                            sec_class.section_id,
                            sec_class.confidence
                        )
                    cs.secondary_classifications.sort(key=lambda x: x.confidence, reverse=True)
            
            # Update topic memory with sentence index for tracking
            classified_section = cs.primary_classification.section_id if cs.primary_classification else None
            related = related_sections_map.get(classified_section, []) if classified_section else []
            topic_memory.update(classified_section, cs.primary_classification.confidence if cs.primary_classification else 0.0, related, sentence_index=i)
            
            # Annotate with enhanced topic context
            cs.active_topic_context = topic_memory.to_dict()
            
            classified_sentences.append(cs)
        logger.info(f"Stage 3: Classified {len(classified_sentences)} sentences with topic memory")

        # Stage 3.5: deterministic routing overrides and overview cleanup
        apply_rule_overrides(classified_sentences, self.classifier.section_metadata)
        logger.info("Stage 3.5: Applied section routing rules")

        # Load runtime policy and enforce policies before repair
        policy = load_policy()
        CONFIDENCE_ACCEPT_THRESHOLD = float(policy.get("confidence_accept_threshold", 0.55))
        IMPLEMENTATION_INDICATORS = [i.lower() for i in policy.get("implementation_indicators", [])]
        CONCEPTUAL_SECTIONS = [c.lower() for c in policy.get("conceptual_sections", [])]

        def _is_implementation_step(text: str) -> bool:
            t = (text or "").lower()
            for ind in IMPLEMENTATION_INDICATORS:
                if ind in t:
                    return True
            if "`" in (text or "") or "->" in (text or ""):
                return True
            return False

        conceptual_set = set(CONCEPTUAL_SECTIONS)

        for idx, cs in enumerate(classified_sentences):
            # If no primary classification -> mark review
            if not cs.primary_classification:
                cs.is_unassigned = True
                cs.explainability_log = ExplainabilityLog(
                    action="policy",
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    sentence_id=idx,
                    section_id="review_required",
                    reasoning="No primary classification",
                    confidence=0.0
                )
                continue

            sec_title = (cs.primary_classification.section_title or "").lower()
            conf = cs.primary_classification.confidence or 0.0

            # Extract explicit evidence early so the markers survive policy gating.
            evidence = extract_explicit_evidence(cs.sentence.text)
            cs.explicit_evidence = evidence
            cs.sentence.explicit_evidence = evidence
            if is_causal_statement(cs.sentence.text) and not evidence:
                cs.is_inferred = True
                cs.sentence.is_inferred = True

            # Enforce configured confidence acceptance
            if conf < CONFIDENCE_ACCEPT_THRESHOLD:
                cs.is_unassigned = True
                cs.explainability_log = ExplainabilityLog(
                    action="policy",
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    sentence_id=idx,
                    section_id="review_required",
                    reasoning=f"Low semantic confidence ({conf:.2f})",
                    confidence=conf
                )
                continue

            # Implementation detection + conceptual placement check
            if _is_implementation_step(cs.sentence.text) and any(c in sec_title for c in conceptual_set):
                cs.is_unassigned = True
                cs.explainability_log = ExplainabilityLog(
                    action="policy",
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    sentence_id=idx,
                    section_id="review_required",
                    reasoning=f"Implementation step detected but classified to conceptual section '{cs.primary_classification.section_title}'",
                    confidence=conf
                )
                continue

            # If sentence reads as a causal statement but lacks explicit evidence,
            # flag it as inferred. Do not change classification, only mark inference.
            if is_causal_statement(cs.sentence.text) and not evidence:
                cs.is_inferred = True
                cs.sentence.is_inferred = True
                # annotate explainability log
                if cs.explainability_log:
                    cs.explainability_log.reasoning = (cs.explainability_log.reasoning or "") + " [INFERRED]"
                else:
                    cs.explainability_log = ExplainabilityLog(
                        action="inference",
                        timestamp=datetime.utcnow().isoformat() + "Z",
                        sentence_id=idx,
                        section_id=cs.primary_classification.section_id if cs.primary_classification else None,
                        reasoning="Causal statement without explicit evidence [INFERRED]",
                        confidence=conf
                    )
        
        # STAGE 4: Repair low-confidence sentences
        repaired_map = {}
        for i, cs in enumerate(classified_sentences):
            if self.repair.should_repair(cs, self.audio_conf_threshold):
                improved, repair_action = self.repair.repair(cs, classified_sentences)
                repaired_map[i] = (improved, repair_action)
            else:
                repaired_map[i] = (cs.sentence.text, None)
        logger.info(f"Stage 4: Repaired {len([r for r in repaired_map.values() if r[1]])} sentences")
        
        # STAGE 5: Detect gaps
        coverage = detect_gaps(classified_sentences, self.schema_sections)
        logger.info(f"Stage 5: Coverage analysis complete")
        
        # STAGE 6: Extract assets
        assets = extract_urls_and_assets(sentences)
        logger.info(f"Stage 6: Extracted {len(assets)} assets")
        
        # Collect topic memory contexts for serialization
        topic_contexts = [cs.active_topic_context for cs in classified_sentences if cs.active_topic_context]
        
        # STAGE 7: Assemble KT
        kt = assemble_kt(job_id, transcript, classified_sentences, coverage, assets, repaired_map, topic_contexts)
        logger.info(f"Stage 7: KT assembled ({kt.overall_coverage_percent:.1f}% coverage, {kt.overall_risk_score:.2f} risk)")
        # STAGE 7.5: Enterprise paragraph reconstruction (optional)
        try:
            # Gate enterprise mapper behind pipeline-level flag to avoid heavy loads by default
            if getattr(self, 'enterprise_mapper_enabled', True):
                # Lazy import to avoid heavy model load during module import
                from enterprise_semantic_mapper import create_semantic_mapper
                llm_refiner = self.repair.llm_fallback_fn if getattr(self, 'repair', None) else None
                mapper = create_semantic_mapper(self.schema_sections, llm_refiner=llm_refiner)
            else:
                mapper = None
            # Build sentence tuples with stable ids
            sentence_tuples = [(f"s_{i}", cs.sentence.text) for i, cs in enumerate(classified_sentences)]
            sem_res = mapper.process_transcript(sentence_tuples) if mapper else {}
            # Merge reconstructed paragraphs into kt.section_content under 'enterprise_paragraphs'
            paragraphs = sem_res.get('paragraphs', {}) if isinstance(sem_res, dict) else {}
            for sec_id, para_list in paragraphs.items():
                try:
                    kt.section_content.setdefault(sec_id, {}).setdefault('enterprise_paragraphs', para_list)
                except Exception:
                    # Non-critical: skip on any merge failure
                    continue
            # Attach metrics for visibility
            kt.enterprise_metrics = sem_res.get('metrics', {}) if isinstance(sem_res, dict) else {}
            logger.info("Enterprise semantic mapper produced paragraphs for %d sections", len(paragraphs))
        except Exception as e:
            logger.warning("Enterprise semantic mapper integration failed: %s", e)

        return kt


def serialize_kt(kt: StructuredKT) -> Dict[str, Any]:
    """Convert StructuredKT to JSON-serializable dict."""
    def _desired_order_keywords():
        return [
            ("System Overview", ["overview", "system", "summary", "introduction", "context"]),
            ("Architecture Components", ["arch", "architecture", "component", "service"]),
            ("Data Flow", ["flow", "data flow", "pipeline", "request", "message", "event"]),
            ("Deployment Process", ["deploy", "deployment", "kubernetes", "kubectl", "helm", "docker", "ci/cd"]),
            ("Monitoring Strategy", ["monitor", "monitoring", "metrics", "alert", "logging", "tracing"]),
            ("Known Issues", ["issue", "problem", "error", "fail", "bug", "troubleshoot", "troubleshooting"])
        ]

    def compute_flow_coherence(kt_obj: StructuredKT) -> Tuple[float, List[str]]:
        """Compute a simple flow coherence score (0.0-1.0) and list flow issues.

        Score = fraction of sections that appear in non-decreasing bucket order
        relative to desired flow. Provides quick signal if ordering is scrambled.
        """
        desired = _desired_order_keywords()
        bucket_names = [name for name, _ in desired]

        # Map section to bucket index
        mapping = {}
        for sec_id, content in kt_obj.section_content.items():
            title = (content.get("section_title") or sec_id).lower()
            placed = False
            for idx, (name, keywords) in enumerate(desired):
                for kw in keywords:
                    if kw in title or kw in sec_id.lower():
                        mapping[sec_id] = idx
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                # Unknown sections go to Known Issues bucket
                mapping[sec_id] = bucket_names.index("Known Issues")

        # Build sequence in original appearance order
        seq = [mapping.get(sid, len(bucket_names)-1) for sid in kt_obj.section_content.keys()]
        if not seq:
            return 1.0, []

        good = 1
        issues = []
        prev = seq[0]
        for i, cur in enumerate(seq[1:], start=1):
            if cur < prev:
                # out of order
                sid = list(kt_obj.section_content.keys())[i]
                issues.append(f"Section '{sid}' appears out of flow (bucket {cur} < prev {prev})")
            else:
                good += 1
            prev = cur

        score = float(good) / len(seq) if seq else 1.0
        return score, issues

    def export_kt_markdown(kt_obj: StructuredKT) -> str:
        """Generate a simple ordered Markdown representation of the KT.

        Sections are presented in the desired logical flow; unmatched sections
        are appended under 'Known Issues'.
        """
        def _format_section_lines(sec_id: str, content: Dict[str, Any]) -> List[str]:
            title = content.get("section_title", sec_id)
            texts = [s.get("text", "").strip() for s in content.get("sentences", []) if s.get("text", "").strip()]
            lower_id = sec_id.lower()
            lower_title = title.lower()
            lines = [f"### {title}"]

            if "deployment" in lower_id or "rollback" in lower_id or "deployment" in lower_title:
                lines.append("Deployment Process:")
                for idx, text in enumerate(texts, 1):
                    lines.append(f"{idx}. {text}")
            elif "failures" in lower_id or "issue" in lower_title or "troubleshooting" in lower_id:
                lines.append("Issue / Cause / Fix:")
                if texts:
                    if len(texts) >= 1:
                        lines.append(f"- Issue: {texts[0]}")
                    if len(texts) >= 2:
                        lines.append(f"- Cause: {texts[1]}")
                    if len(texts) >= 3:
                        lines.append(f"- Fix: {texts[2]}")
                    for extra in texts[3:]:
                        lines.append(f"- Detail: {extra}")
            elif "architecture" in lower_id or "architecture" in lower_title:
                lines.append("Architecture Notes:")
                for text in texts:
                    lines.append(f"- {text}")
            else:
                for text in texts:
                    lines.append(f"- {text}")

            lines.append("")
            return lines

        desired = _desired_order_keywords()
        buckets = {name: [] for name, _ in desired}
        unmatched = []

        for sec_id, content in kt_obj.section_content.items():
            title = (content.get("section_title") or sec_id)
            placed = False
            for name, keywords in desired:
                for kw in keywords:
                    if kw in title.lower() or kw in sec_id.lower():
                        buckets[name].append((sec_id, content))
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                unmatched.append((sec_id, content))

        markdown_lines = [f"# KT Report - {kt_obj.job_id}", ""]
        for name, _ in desired:
            items = buckets.get(name, [])
            if not items:
                continue
            markdown_lines.append(f"## {name}")
            for sec_id, content in items:
                markdown_lines.extend(_format_section_lines(sec_id, content))

        if unmatched:
            markdown_lines.append("## Known Issues")
            for sec_id, content in unmatched:
                markdown_lines.extend(_format_section_lines(sec_id, content))

        return "\n".join(markdown_lines)

    # Build ordered_section_content for consumers that want ordered sections
    desired = _desired_order_keywords()
    raw_sections = []
    for sec_id, content in kt.section_content.items():
        title = content.get("section_title") or sec_id
        raw_sections.append({
            "section_id": sec_id,
            "section_title": title,
            "content": content
        })

    buckets = {name: [] for name, _ in desired}
    unmatched = []
    for sec in raw_sections:
        placed = False
        lower_title = (sec.get("section_title") or "").lower()
        sec_id_lower = (sec.get("section_id") or "").lower()
        for name, keywords in desired:
            for kw in keywords:
                if kw in lower_title or kw in sec_id_lower:
                    buckets[name].append(sec)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            unmatched.append(sec)

    if unmatched:
        buckets["Known Issues"].extend(unmatched)

    ordered_section_content = []
    for name, _ in desired:
        for sec in buckets.get(name, []):
            sec_id = sec["section_id"]
            content = kt.section_content.get(sec_id, {})
            ordered_section_content.append({
                "section_id": sec_id,
                "section_title": sec.get("section_title"),
                "sentence_count": content.get("sentence_count", 0),
                "confidence": content.get("confidence", 0.0),
                "sentences_preview": [s["text"][:100] for s in content.get("sentences", [])[:5]],
                "sentences": content.get("sentences", [])
            })

    flow_score, flow_issues = compute_flow_coherence(kt)

    return {
        "job_id": kt.job_id,
        "parent_job_id": kt.parent_job_id,
        "timestamp": kt.timestamp,
        "transcript": kt.transcript,
        "overall_coverage_percent": kt.overall_coverage_percent,
        "overall_risk_score": kt.overall_risk_score,
        "missing_required_sections": kt.missing_required_sections,
        "section_count": len(kt.section_content),
        "sentence_count": len(kt.sentences),
        "unassigned_count": len(kt.unassigned_sentences),
        "asset_count": len(kt.assets),
        "explainability_logs_count": len(kt.explainability_logs or []),
        "human_feedback_count": len(kt.human_feedback or []),
        "coverage": {
            sec_id: {
                "section_title": cov.section_title,
                "status": cov.status,
                "required": cov.required,
                "sentence_count": kt.section_content.get(sec_id, {}).get("sentence_count", cov.sentence_count),
                "confidence": cov.confidence_score,
                "risk": cov.risk_score,
                "coverage_score": cov.coverage_score
            }
            for sec_id, cov in kt.coverage.items()
        },
        "section_content": {
            sec_id: {
                "section_id": content["section_id"],
                "section_title": content["section_title"],
                "sentence_count": content["sentence_count"],
                "confidence": content["confidence"],
                "sentences_preview": [s["text"][:100] for s in content["sentences"][:3]]
            }
            for sec_id, content in kt.section_content.items()
        },
        "ordered_section_content": ordered_section_content,
        "unassigned_sentences": [
            {"text": s.text, "start": s.start, "end": s.end}
            for s in (list({(re.sub(r"\s+"," ", u.text.strip()).lower()): u for u in kt.unassigned_sentences}.values()))[:5]
        ],
        "review_required_sentences": [
            {
                "text": s.text,
                "start": s.start,
                "end": s.end,
                "explainability": next(({
                    "action": log.action,
                    "section_id": log.section_id,
                    "reasoning": log.reasoning,
                    "confidence": log.confidence
                } for log in (kt.explainability_logs or []) if log.sentence_id == i), None)
            }
            for i, s in enumerate(kt.unassigned_sentences)
        ],
        "evidence": [
            {
                "text": s.text,
                "evidence": getattr(s, 'explicit_evidence', []) if hasattr(s, 'explicit_evidence') else []
            }
            for s in kt.sentences[:10]
        ],
        "assets": [
            {
                "type": a.asset_type,
                "content": a.content,
                "component": a.detected_component,
                "timestamp": a.timestamp
            }
            for a in kt.assets[:10]
        ],
        "top_explainability_logs": [
            {
                "action": log.action,
                "section_id": log.section_id,
                "reasoning": log.reasoning,
                "confidence": log.confidence
            }
            for log in (kt.explainability_logs or [])[:5]
        ]
        ,
        "flow_coherence_score": flow_score,
        "flow_issues": flow_issues,
        "ordered_markdown": export_kt_markdown(kt)
        ,
        "cross_references": kt.cross_references if getattr(kt, 'cross_references', None) else []
    }


def merge_incremental_kt(parent_kt: StructuredKT, child_kt: StructuredKT) -> StructuredKT:
    """
    Merge child KT (follow-up session) with parent KT.
    
    Rules:
    - Concatenate transcripts
    - Re-aggregate coverage (child takes priority if conflicting)
    - Append sentences with unique IDs
    - Merge assets
    - Keep separate explainability logs
    
    Args:
        parent_kt: Original KT session
        child_kt: Follow-up KT session
    
    Returns:
        Merged StructuredKT
    """
    # Merge transcripts
    merged_transcript = f"{parent_kt.transcript}\n\n[SESSION 2]\n{child_kt.transcript}"
    
    # Merge sentences (offset timestamps for child)
    parent_start = parent_kt.transcript.count('\n')
    child_sentences = [
        Sentence(
            text=s.text,
            start=s.start + parent_kt.sentences[-1].end if parent_kt.sentences else s.start,
            end=s.end + parent_kt.sentences[-1].end if parent_kt.sentences else s.end,
            speaker=s.speaker,
            raw_text=s.raw_text,
            audio_confidence=s.audio_confidence,
            segment_ids=s.segment_ids
        )
        for s in child_kt.sentences
    ]
    merged_sentences = parent_kt.sentences + child_sentences
    
    # Merge coverage (child wins if better)
    merged_coverage = dict(parent_kt.coverage)
    for sec_id, child_cov in child_kt.coverage.items():
        if sec_id in merged_coverage:
            parent_cov = merged_coverage[sec_id]
            # Upgrade status if child has better coverage
            if child_cov.status == "covered" or (child_cov.status == "weak" and parent_cov.status == "missing"):
                merged_coverage[sec_id] = child_cov
            # Append sentences
            merged_coverage[sec_id].sentences.extend(child_cov.sentences)
        else:
            merged_coverage[sec_id] = child_cov
    
    # Re-compute overall metrics
    covered = sum(1 for c in merged_coverage.values() if c.status in ("covered", "weak"))
    overall_coverage = 100.0 * covered / len(merged_coverage) if merged_coverage else 0.0
    overall_risk = np.mean([c.risk_score for c in merged_coverage.values()]) if merged_coverage else 0.0
    
    # Merge explainability logs
    merged_logs = (parent_kt.explainability_logs or []) + (child_kt.explainability_logs or [])
    
    # Merge human feedback
    merged_feedback = (parent_kt.human_feedback or []) + (child_kt.human_feedback or [])
    
    # Merge assets
    merged_assets = parent_kt.assets + child_kt.assets
    
    return StructuredKT(
        job_id=child_kt.job_id,
        parent_job_id=parent_kt.job_id,
        transcript=merged_transcript,
        sentences=merged_sentences,
        coverage=merged_coverage,
        section_content=parent_kt.section_content,  # TODO: merge section content properly
        missing_required_sections=[s for s in merged_coverage if merged_coverage[s].required and merged_coverage[s].status == "missing"],
        unassigned_sentences=parent_kt.unassigned_sentences + child_kt.unassigned_sentences,
        assets=merged_assets,
        overall_coverage_percent=float(overall_coverage),
        overall_risk_score=float(overall_risk),
        timestamp=child_kt.timestamp,
        explainability_logs=merged_logs,
        human_feedback=merged_feedback
    )


def apply_human_feedback(kt: StructuredKT, feedback: HumanFeedback) -> StructuredKT:
    """
    Apply human override/correction to KT.
    
    Args:
        kt: Original KT
        feedback: Human feedback (correction)
    
    Returns:
        Updated KT with human feedback applied
    """
    if not kt.human_feedback:
        kt.human_feedback = []
    
    kt.human_feedback.append(feedback)
    
    # Find and update the classified sentence in section_content
    # This would require more detailed sentence mapping...
    # For now, just record the feedback for audit trail
    
    return kt
