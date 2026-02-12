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

from sentence_transformers import SentenceTransformer, util
import numpy as np

logger = logging.getLogger(__name__)

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
        # Build multi-section assignment from primary + secondary
        if self.multi_section_assignments is None:
            self.multi_section_assignments = []
            if self.primary_classification:
                self.multi_section_assignments.append(self.primary_classification.section_id)
            for sec in self.secondary_classifications:
                if sec.section_id not in self.multi_section_assignments:
                    self.multi_section_assignments.append(sec.section_id)
    

class ContextClassifier:
    """
    STAGE 3: Semantic classification against KT schema.
    
    Uses sentence-transformers for semantic similarity.
    Never discards sentences—marks unassigned if similarity too low.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.30
    ):
        """
        Args:
            model_name: HuggingFace model for embeddings
            similarity_threshold: Minimum cosine similarity for assignment
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.section_embeddings = {}
        self.section_metadata = {}
    
    def index_schema(self, schema_sections: List[Dict]) -> None:
        """Index schema sections for fast lookup."""
        for sec in schema_sections:
            sec_id = sec.get("id")
            title = sec.get("title", sec_id)
            description = sec.get("description", "")
            keywords = sec.get("keywords", [])
            
            # Build rich section text
            section_text = f"{title} {description} {' '.join(keywords)}"
            embedding = self.model.encode(section_text, convert_to_tensor=True)
            
            self.section_embeddings[sec_id] = embedding
            self.section_metadata[sec_id] = {
                "title": title,
                "required": sec.get("required", False),
                "description": description
            }
    
    def classify_sentence(self, sentence: Sentence) -> ClassifiedSentence:
        """
        Classify sentence against all schema sections.
        
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
        
        # Embed sentence
        sent_embedding = self.model.encode(sentence.text, convert_to_tensor=True)
        
        # Compute similarities to all sections
        classifications = []
        for sec_id, sec_embedding in self.section_embeddings.items():
            sim = float(util.cos_sim(sent_embedding, sec_embedding)[0][0])
            
            if sim >= self.similarity_threshold:
                classifications.append(Classification(
                    section_id=sec_id,
                    section_title=self.section_metadata[sec_id]["title"],
                    confidence=sim,
                    similarity_score=sim,
                    reason=f"Semantic match (similarity={sim:.3f})"
                ))
        
        # Sort by confidence
        classifications.sort(key=lambda x: x.confidence, reverse=True)
        
        primary = classifications[0] if classifications else None
        secondary = classifications[1:] if len(classifications) > 1 else []
        is_unassigned = primary is None
        
        # Build explainability log
        explanation = ""
        alternatives = []
        if primary:
            explanation = f"Matched '{sentence.text[:60]}' to '{primary.section_title}' (similarity={primary.confidence:.3f})"
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
        
        return ClassifiedSentence(
            sentence=sentence,
            primary_classification=primary,
            secondary_classifications=secondary,
            is_unassigned=is_unassigned,
            explainability_log=explainability
        )


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
        improved = original
        repair_record = None
        
        # Stage 1: Basic grammar fixes
        improved = self._basic_grammar_fix(improved)
        
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
            # For now, just return None (no-op)
            # In production: result = await self.llm_fallback_fn(original, context, section_id)
            return None
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
        
        # Find all sentences mapped to this section
        sentences = [
            cs.sentence for cs in classified_sentences
            if cs.primary_classification and cs.primary_classification.section_id == sec_id
        ]
        
        # Determine status
        if len(sentences) == 0:
            status = "missing"
            confidence = 0.0
            risk = 1.0 if sec_required else 0.5
        elif len(sentences) <= weak_threshold:
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
            sentence_count=len(sentences),
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
    for sent in sentences:
        for dash_name, dash_pattern in KNOWN_DASHBOARDS.items():
            if re.search(dash_pattern, sent.text, re.IGNORECASE):
                assets.append(ExtractedAsset(
                    asset_type="screenshot_candidate",
                    content=f"dashboard:{dash_name}",
                    sentence_ids=[],
                    detected_component=dash_name,
                    timestamp=sent.start
                ))
    
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
    
    for cs in classified_sentences:
        # Build enhanced text
        idx = classified_sentences.index(cs)
        enhanced_text, repair_action = repaired_map.get(idx, (cs.sentence.text, None))
        
        if cs.is_unassigned:
            unassigned.append(cs.sentence)
        elif cs.primary_classification:
            sec_id = cs.primary_classification.section_id
            if sec_id not in section_content:
                section_content[sec_id] = {
                    "section_id": sec_id,
                    "section_title": cs.primary_classification.section_title,
                    "sentences": [],
                    "enhanced_texts": [],
                    "repair_actions": [],
                    "screenshots": [],
                    "confidence": 0.0,
                    "sentence_count": 0
                }
            
            section_content[sec_id]["sentences"].append({
                "text": cs.sentence.text,
                "start": cs.sentence.start,
                "end": cs.sentence.end,
                "speaker": cs.sentence.speaker,
                "audio_confidence": cs.sentence.audio_confidence
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
        classified_sentences = [self.classifier.classify_sentence(s) for s in sentences]
        logger.info(f"Stage 3: Classified {len(classified_sentences)} sentences")
        
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
                "sentence_count": cov.sentence_count,
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
        "unassigned_sentences": [
            {"text": s.text, "start": s.start, "end": s.end}
            for s in kt.unassigned_sentences[:5]
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
