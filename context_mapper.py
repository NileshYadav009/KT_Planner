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
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import importlib.util
import os
from policy import (
    CONFIDENCE_ACCEPT_THRESHOLD,
    IMPLEMENTATION_INDICATORS,
    CONCEPTUAL_SECTIONS
)

from runtime_policy import load_policy

# Detect if sentence_transformers package is installed but avoid importing it at module import time.
# Only use the real heavy model when the environment variable USE_REAL_EMBEDDINGS is set to true.
_SENT_TRANS_SPEC = importlib.util.find_spec("sentence_transformers")
_USE_REAL_EMBEDDINGS = bool(_SENT_TRANS_SPEC and os.getenv("USE_REAL_EMBEDDINGS", "").lower() in ("1", "true", "yes"))

# Fallback: simple token-set embedding + Jaccard similarity to avoid heavy deps at module import time
if not _USE_REAL_EMBEDDINGS:
    import statistics as np

    class SimpleEmbedding:
        def __init__(self, text: str):
            self.words = set(re.findall(r"\w+", text.lower()))

    class SentenceTransformer:
        def __init__(self, model_name=None):
            pass
        def encode(self, texts, convert_to_tensor=False):
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
    
    def __post_init__(self):
        if self.segment_ids is None:
            self.segment_ids = []
        if self.raw_text is None:
            self.raw_text = self.text


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
    
    # Split by sentence endings
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
    
    return sentences


# ============================================================================
# STAGE 3: Context Classification Engine
# ============================================================================

@dataclass
class Classification:
    """Semantic classification of a sentence."""
    section_id: str
    section_title: str
    confidence: float
    reason: str  # Why this section was chosen
    similarity_score: float


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
    
    Uses sentence-transformers for semantic similarity.
    Never discards sentences—marks unassigned if similarity too low.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.15
    ):
        """
        Args:
            model_name: HuggingFace model for embeddings
            similarity_threshold: Minimum cosine similarity for assignment
        """
        # Lazily import heavy dependencies if configured to use real embeddings
        if _USE_REAL_EMBEDDINGS:
            try:
                st = __import__("sentence_transformers")
                # Import numpy when using real model
                import numpy as _np
                globals()['np'] = _np
                globals()['SentenceTransformer'] = st.SentenceTransformer
                globals()['util'] = st.util
            except Exception:
                # Fall back to lightweight implementations
                pass

        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.section_embeddings = {}
        self.section_metadata = {}
        self._section_hints = {}  # Initialize for keyword matching
    
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
            embedding = self.model.encode(section_text, convert_to_tensor=True)
            
            self.section_embeddings[sec_id] = embedding
            self.section_metadata[sec_id] = {
                "title": title,
                "required": sec.get("required", False),
                "description": description
            }
    
    def classify_sentence(self, sentence: Sentence, sent_embedding=None, neighbor_embeddings: List = None, top_k: int = 3, alpha: float = 0.7, beta: float = 0.3) -> ClassifiedSentence:
        """
        Classify sentence against all schema sections.
        
        Uses semantic similarity boosted by keyword matching to prevent misplacement
        while allowing lower thresholds for better coverage.
        
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
        
        # Embed sentence (reuse precomputed embedding when available)
        if sent_embedding is None:
            sent_embedding = self.model.encode(sentence.text, convert_to_tensor=True)

        # Compute similarities to all sections
        classifications = []
        sent_text_lower = (sentence.text or "").lower()
        
        for sec_id, sec_embedding in self.section_embeddings.items():
            base_sim = float(util.cos_sim(sent_embedding, sec_embedding)[0][0])

            # Incorporate neighbor context similarity when provided
            context_sim = 0.0
            if neighbor_embeddings:
                sims = [float(util.cos_sim(nb, sec_embedding)[0][0]) for nb in neighbor_embeddings]
                if sims:
                    context_sim = float(np.mean(sims))

            # Keyword matching boost: if hints match, boost the score
            keyword_boost = 0.0
            if sec_id in self.section_metadata:
                # Get hints from schema (already loaded during index_schema)
                # Compute keyword match score
                hints = getattr(self, '_section_hints', {}).get(sec_id, [])
                if hints:
                    matching_hints = sum(1 for h in hints if h.lower() in sent_text_lower)
                    if matching_hints > 0:
                        keyword_boost = min(0.3, 0.05 * matching_hints)  # Cap boost at 0.3

            # Combined score — prefer sentence-level similarity but allow context and keywords to influence
            combined = float(alpha * base_sim + beta * context_sim + keyword_boost)

            # Always consider top candidates, thresholding later
            classifications.append(Classification(
                section_id=sec_id,
                section_title=self.section_metadata[sec_id]["title"],
                confidence=combined,
                similarity_score=base_sim,
                reason=f"Semantic={base_sim:.3f}, Context={context_sim:.3f}, Keywords={keyword_boost:.3f}, Combined={combined:.3f}"
            ))
        
        # Sort by combined confidence and take top_k
        classifications.sort(key=lambda x: x.confidence, reverse=True)
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
        if self.llm_fallback_fn and classified.sentence.audio_confidence < 0.3:
            try:
                # Async hook for Claude/GPT-4
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
        
        try:
            # This would be called asynchronously in production
            # Delegate to provided LLM hook with conservative settings
            # The llm_fallback_fn should accept (text, context_text, section_id, temperature)
            context_text = ' '.join([c.sentence.text for c in context]) if context else ''
            # Low temperature to avoid hallucination
            result = self.llm_fallback_fn(original, context_text, section_id, temperature=0.2)
            return result
        except Exception:
            return None


# ============================================================================
# STAGE 5: Gap Detection
# ============================================================================

@dataclass
class SectionCoverage:
    """Coverage metrics for a KT section."""
    section_id: str
    section_title: str
    required: bool
    status: str  # "missing" | "weak" | "covered"
    sentence_count: int
    confidence_score: float
    risk_score: float
    sentences: List[Sentence]
    

def detect_gaps(
    classified_sentences: List[ClassifiedSentence],
    schema_sections: List[Dict],
    weak_threshold: int = 1,
    required_threshold: int = 2
) -> Dict[str, SectionCoverage]:
    """
    STAGE 5: Detect coverage gaps and compute risk.
    
    Rules:
    - 0 sentences → missing
    - 1 sentence → weak
    - 2+ sentences → covered
    """
    coverage = {}
    
    for sec in schema_sections:
        sec_id = sec.get("id")
        sec_title = sec.get("title", sec_id)
        sec_required = sec.get("required", False)
        
        # Find all sentences mapped to this section (use unique normalized text)
        mapped_texts = []
        unique_texts = set()
        mapped_sentences = []
        for cs in classified_sentences:
            if cs.primary_classification and cs.primary_classification.section_id == sec_id:
                text = (cs.sentence.text or '').strip()
                key = re.sub(r"\s+", " ", text).lower()
                if key not in unique_texts:
                    unique_texts.add(key)
                    mapped_texts.append(text)
                    mapped_sentences.append(cs.sentence)
        sentences = mapped_sentences
        
        # Determine status
        unique_count = len(mapped_texts)
        if unique_count == 0:
            status = "missing"
            confidence = 0.0
            risk = 1.0 if sec_required else 0.5
        elif unique_count <= weak_threshold:
            status = "weak"
            confidence = np.mean([cs.sentence.audio_confidence for cs in classified_sentences 
                                 if cs.primary_classification and cs.primary_classification.section_id == sec_id])
            risk = 0.6 if sec_required else 0.2
        else:
            status = "covered"
            confidence = np.mean([cs.sentence.audio_confidence for cs in classified_sentences 
                                 if cs.primary_classification and cs.primary_classification.section_id == sec_id])
            risk = 0.0 if confidence > 0.7 else 0.1
        
        coverage[sec_id] = SectionCoverage(
            section_id=sec_id,
            section_title=sec_title,
            required=sec_required,
            status=status,
            sentence_count=unique_count,
            confidence_score=float(confidence),
            risk_score=float(risk),
            sentences=sentences
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


def assemble_kt(
    job_id: str,
    transcript: str,
    classified_sentences: List[ClassifiedSentence],
    coverage: Dict[str, SectionCoverage],
    assets: List[ExtractedAsset],
    repaired_map: Dict[int, Tuple[str, Optional[RepairAction]]]
) -> StructuredKT:
    """
    STAGE 7: Assemble final KT structure.
    
    Returns comprehensive KT object with:
    - Section-wise content
    - Sentence lists per section
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
        coverage=coverage,
        section_content=section_content,
        missing_required_sections=missing_required,
        unassigned_sentences=unassigned,
        assets=assets,
        overall_coverage_percent=float(overall_coverage),
        overall_risk_score=float(overall_risk),
        timestamp=datetime.utcnow().isoformat() + "Z"
        ,
        cross_references=cross_refs or []
    )


# ============================================================================
# Main Pipeline Orchestrator
# ============================================================================

class ContextMappingPipeline:
    """Orchestrates all 7 stages of context mapping."""
    
    def __init__(
        self,
        schema_sections: List[Dict],
        similarity_threshold: float = 0.30,
        audio_confidence_threshold: float = 0.4
    ):
        self.schema_sections = schema_sections
        self.classifier = ContextClassifier(similarity_threshold=similarity_threshold)
        self.classifier.index_schema(schema_sections)
        self.repair = ContextRepair()
        self.audio_conf_threshold = audio_confidence_threshold
    
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
                text=seg.get("text", ""),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                avg_logprob=seg.get("avg_logprob", -1.0),
                speaker=seg.get("speaker")
            )
            for seg in audio_segments
        ]
        
        # STAGE 2: Segment sentences
        sentences = segment_sentences(segments)
        logger.info(f"Stage 2: Segmented into {len(sentences)} sentences")
        
        # STAGE 3: Classify sentences
        # Precompute embeddings for sentences to allow context-aware scoring
        texts = [s.text for s in sentences]
        embeddings = self.classifier.model.encode(texts, convert_to_tensor=True)

        classified_sentences = []
        for i, s in enumerate(sentences):
            # Build neighbor embeddings (fixed window)
            window = 3
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            neighbor_embs = [embeddings[j] for j in range(start, end) if j != i]
            cs = self.classifier.classify_sentence(s, sent_embedding=embeddings[i], neighbor_embeddings=neighbor_embs)
            classified_sentences.append(cs)
        logger.info(f"Stage 3: Classified {len(classified_sentences)} sentences")

        # Load runtime policy and enforce policies before repair
        policy = load_policy()
        CONFIDENCE_ACCEPT_THRESHOLD = float(policy.get("confidence_accept_threshold", 0.8))
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

            # Extract explicit evidence for root-cause assertions
            evidence = extract_explicit_evidence(cs.sentence.text)
            cs.explicit_evidence = evidence

            # If sentence reads as a causal statement but lacks explicit evidence,
            # flag it as inferred. Do not change classification, only mark inference.
            if is_causal_statement(cs.sentence.text) and not evidence:
                cs.is_inferred = True
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
        
        # STAGE 7: Assemble KT
        kt = assemble_kt(job_id, transcript, classified_sentences, coverage, assets, repaired_map)
        logger.info(f"Stage 7: KT assembled ({kt.overall_coverage_percent:.1f}% coverage, {kt.overall_risk_score:.2f} risk)")
        
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
                markdown_lines.append(f"### {content.get('section_title', sec_id)}")
                for s in content.get('sentences', []):
                    markdown_lines.append(f"- {s.get('text')}")
                markdown_lines.append("")

        if unmatched:
            markdown_lines.append("## Known Issues")
            for sec_id, content in unmatched:
                markdown_lines.append(f"### {content.get('section_title', sec_id)}")
                for s in content.get('sentences', []):
                    markdown_lines.append(f"- {s.get('text')}")
                markdown_lines.append("")

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
                "risk": cov.risk_score
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
