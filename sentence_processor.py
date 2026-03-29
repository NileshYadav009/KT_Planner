"""
Advanced Sentence Processing Engine
- Smart segmentation with metadata
- Confusion detection
- Auto-mapping learning
- Sentence versioning
- Code linking
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import json
import hashlib
import uuid
from collections import defaultdict

class ConfidenceLevel(Enum):
    """Confidence classification levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CONFUSION = "confusion"


class SentenceStatus(Enum):
    """Status of sentence classification."""
    MAPPED = "mapped"
    PARTIAL = "partial"
    UNASSIGNED = "unassigned"
    CONFLICTING = "conflicting"
    CLARIFICATION_NEEDED = "clarification_needed"


@dataclass
class CodeReference:
    """Reference to code or log snippet."""
    id: str
    code_block: str
    file_name: str
    language: str = "text"
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    related_sentence_ids: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self):
        return asdict(self)


@dataclass
class SentenceMetadata:
    """Rich metadata for each sentence."""
    id: str
    text: str
    confidence_score: float  # 0.0-1.0
    predicted_section: str
    alternatives: List[Dict[str, float]]  # [{"section": "...", "score": 0.8}, ...]
    importance_weight: float  # 0.0-1.0
    quality_score: float  # 0.0-1.0 (transcription quality)
    status: SentenceStatus = SentenceStatus.UNASSIGNED
    
    # Original vs Edited tracking
    original_text: Optional[str] = None
    is_edited: bool = False
    edit_history: List[str] = field(default_factory=list)
    
    # Audio properties
    audio_start: float = 0.0
    audio_end: float = 0.0
    speaker: Optional[str] = None
    
    # Linking features
    code_references: List[str] = field(default_factory=list)  # CodeReference IDs
    manual_notes: str = ""
    user_tags: List[str] = field(default_factory=list)
    
    # Confusion detection
    is_confusing: bool = False
    confusion_reasons: List[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_requested: str = ""  # User question for clarification
    
    # Assignment tracking
    assigned_sections: List[str] = field(default_factory=list)
    assignment_history: List[Dict] = field(default_factory=list)
    
    # AI learning
    manual_corrections: int = 0  # Times user corrected AI
    confidence_adjustments: List[float] = field(default_factory=list)
    
    timestamp_created: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    timestamp_modified: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self):
        data = asdict(self)
        data['status'] = self.status.value
        return data

    def update_text(self, new_text: str, user: str = "system"):
        """Track text edits."""
        if new_text != self.text:
            self.edit_history.append(self.text)
            self.original_text = self.original_text or self.text
            self.text = new_text
            self.is_edited = True
            self.timestamp_modified = datetime.utcnow().isoformat()

    def assign_section(self, section_id: str, user: str = "system", confidence: float = 1.0):
        """Assign sentence to section with tracking."""
        if section_id not in self.assigned_sections:
            self.assigned_sections.append(section_id)
            self.assignment_history.append({
                "section": section_id,
                "user": user,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat()
            })
            self.status = SentenceStatus.MAPPED
            self.timestamp_modified = datetime.utcnow().isoformat()


@dataclass
class TranscriptVersion:
    """Version snapshot of entire transcript."""
    version_id: str
    job_id: str
    timestamp: str
    user: str
    sentences: List[SentenceMetadata]
    section_assignments: Dict[str, List[str]]  # section_id -> [sentence_ids]
    change_summary: str  # What changed
    is_automated: bool = False  # Auto version or manual save

    def to_dict(self):
        return {
            "version_id": self.version_id,
            "job_id": self.job_id,
            "timestamp": self.timestamp,
            "user": self.user,
            "sentence_count": len(self.sentences),
            "change_summary": self.change_summary,
            "is_automated": self.is_automated,
            "sentence_metadata": [s.to_dict() for s in self.sentences]
        }


class SentenceProcessor:
    """Advanced sentence processing with all metadata."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.sentences: Dict[str, SentenceMetadata] = {}  # sentence_id -> metadata
        self.code_references: Dict[str, CodeReference] = {}  # ref_id -> reference
        self.versions: List[TranscriptVersion] = []  # Version history
        self.undo_stack: List[Dict] = []
        self.redo_stack: List[Dict] = []
        self.mapping_patterns: Dict[str, List[str]] = defaultdict(list)  # Learn patterns

    def process_segments(self, segments: List[Dict], transcript: str) -> List[SentenceMetadata]:
        """Process Whisper segments into rich sentence metadata."""
        from nltk.tokenize import sent_tokenize
        try:
            sentences = sent_tokenize(transcript)
        except:
            sentences = transcript.split('. ')
        
        metadata_list = []
        current_pos = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sent_id = f"sent_{i}_{hashlib.md5(sentence.encode()).hexdigest()[:8]}"
            
            # Find corresponding audio segment
            audio_segment = self._find_best_segment(sentence, segments, current_pos)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(sentence, audio_segment)
            
            # Initial confidence and section prediction
            predicted_section, confidence, alternatives = self._predict_section(sentence)
            
            # Detect if confusing
            is_confusing, confusion_reasons = self._detect_confusion(sentence, confidence)
            
            # Calculate importance
            importance_weight = self._calculate_importance(sentence)
            
            metadata = SentenceMetadata(
                id=sent_id,
                text=sentence,
                confidence_score=confidence,
                predicted_section=predicted_section,
                alternatives=alternatives,
                importance_weight=importance_weight,
                quality_score=quality_score,
                audio_start=audio_segment.get('start', 0.0) if audio_segment else 0.0,
                audio_end=audio_segment.get('end', 0.0) if audio_segment else 0.0,
                speaker=audio_segment.get('speaker') if audio_segment else None,
                is_confusing=is_confusing,
                confusion_reasons=confusion_reasons,
                status=SentenceStatus.UNASSIGNED
            )
            
            self.sentences[sent_id] = metadata
            metadata_list.append(metadata)
            current_pos += len(sentence)
        
        return metadata_list

    def _find_best_segment(self, sentence: str, segments: List[Dict], pos: int) -> Optional[Dict]:
        """Find best matching Whisper segment for sentence."""
        for seg in segments:
            if sentence.lower() in seg.get('text', '').lower():
                return seg
        return segments[0] if segments else None

    def _calculate_quality_score(self, sentence: str, segment: Optional[Dict]) -> float:
        """Calculate transcription quality (0-1)."""
        score = 1.0
        
        # Penalize very short sentences
        if len(sentence.split()) < 2:
            score -= 0.3
        
        # Check for common transcription artifacts
        artifacts = ['uh', 'um', '...', '  ']
        for artifact in artifacts:
            if artifact in sentence.lower():
                score -= 0.1
        
        # Use Whisper confidence if available
        if segment and 'avg_logprob' in segment:
            whisper_conf = min(1.0, max(0.0, -segment['avg_logprob']))
            score = score * 0.5 + whisper_conf * 0.5
        
        return max(0.0, min(1.0, score))

    def _predict_section(self, sentence: str) -> Tuple[str, float, List[Dict]]:
        """Predict KT section for sentence using semantic similarity."""
        try:
            from ai import classify_with_confidence
            result = classify_with_confidence(sentence)
            
            return (
                result.get('section', 'overview'),
                result.get('confidence', 0.5),
                result.get('alternatives', [])
            )
        except:
            return ('overview', 0.5, [])

    def _detect_confusion(self, sentence: str, confidence: float) -> Tuple[bool, List[str]]:
        """Detect if sentence is confusing/ambiguous."""
        reasons = []
        is_confusing = False
        
        # Low confidence is a red flag
        if confidence < 0.4:
            reasons.append("low_confidence_prediction")
            is_confusing = True
        
        # Multiple questions/uncertainties
        question_marks = sentence.count('?')
        if question_marks > 1:
            reasons.append("multiple_questions")
            is_confusing = True
        
        # Mixed sections indicators (context switching)
        if any(x in sentence.lower() for x in ['but', 'however', 'instead', 'actually']):
            if sentence.count(',') > 2:
                reasons.append("context_switching")
                is_confusing = True
        
        # Very long complex sentences
        if len(sentence.split()) > 40:
            reasons.append("overly_complex")
            is_confusing = True
        
        return is_confusing, reasons

    def _calculate_importance(self, sentence: str) -> float:
        """Calculate importance weight (0-1)."""
        score = 0.5  # Base importance
        
        # Important keywords boost score
        important_keywords = [
            'critical', 'important', 'must', 'required', 'deploy', 'production',
            'failure', 'error', 'bug', 'fix', 'security', 'compliance'
        ]
        
        for keyword in important_keywords:
            if keyword in sentence.lower():
                score += 0.1
        
        # Longer sentences tend to be more informative
        word_count = len(sentence.split())
        if word_count > 15:
            score += 0.1
        
        return min(1.0, score)

    def add_code_reference(self, code_block: str, file_name: str, 
                          language: str = "text", line_start: Optional[int] = None,
                          line_end: Optional[int] = None) -> CodeReference:
        """Add a code reference."""
        ref_id = f"ref_{uuid.uuid4().hex[:8]}"
        ref = CodeReference(
            id=ref_id,
            code_block=code_block,
            file_name=file_name,
            language=language,
            line_start=line_start,
            line_end=line_end
        )
        self.code_references[ref_id] = ref
        return ref

    def link_code_to_sentence(self, sentence_id: str, code_ref_id: str):
        """Link code reference to a sentence."""
        if sentence_id in self.sentences and code_ref_id in self.code_references:
            self.sentences[sentence_id].code_references.append(code_ref_id)
            self.code_references[code_ref_id].related_sentence_ids.append(sentence_id)

    def request_clarification(self, sentence_id: str, question: str, user: str = "user"):
        """User requests clarification for a sentence."""
        if sentence_id in self.sentences:
            sent = self.sentences[sentence_id]
            sent.needs_clarification = True
            sent.clarification_requested = question
            sent.status = SentenceStatus.CLARIFICATION_NEEDED
            self._record_action("clarification_requested", sentence_id, user)

    def mark_confusing(self, sentence_id: str, user: str = "user"):
        """Mark sentence as confusing."""
        if sentence_id in self.sentences:
            sent = self.sentences[sentence_id]
            sent.is_confusing = True
            sent.status = SentenceStatus.CONFLICTING
            self._record_action("marked_confusing", sentence_id, user)

    def learn_mapping_pattern(self, section_id: str, keyword: str):
        """Learn pattern: when we see 'keyword', section is likely 'section_id'."""
        self.mapping_patterns[keyword].append(section_id)

    def suggest_improved_mapping(self, sentence_id: str) -> List[Dict]:
        """Suggest improved section mapping based on learned patterns."""
        if sentence_id not in self.sentences:
            return []
        
        sent = self.sentences[sentence_id]
        suggestions = []
        
        # Check learned patterns
        for word in sent.text.lower().split():
            if word in self.mapping_patterns:
                sections = self.mapping_patterns[word]
                section_counts = {}
                for sec in sections:
                    section_counts[sec] = section_counts.get(sec, 0) + 1
                
                for section, count in section_counts.items():
                    if count > 1:
                        suggestions.append({
                            "section": section,
                            "reason": f"learned_pattern_{count}_times",
                            "confidence": min(0.9, 0.5 + count * 0.1)
                        })
        
        return sorted(suggestions, key=lambda x: x['confidence'], reverse=True)

    def create_version(self, user: str = "system", change_summary: str = "", 
                      is_automated: bool = False) -> TranscriptVersion:
        """Create version snapshot."""
        # Build section assignments from current metadata
        section_assignments: Dict[str, List[str]] = defaultdict(list)
        for sent_id, sent_meta in self.sentences.items():
            for section in sent_meta.assigned_sections:
                section_assignments[section].append(sent_id)
        
        version = TranscriptVersion(
            version_id=f"v_{uuid.uuid4().hex[:8]}",
            job_id=self.job_id,
            timestamp=datetime.utcnow().isoformat(),
            user=user,
            sentences=list(self.sentences.values()),
            section_assignments=dict(section_assignments),
            change_summary=change_summary or "Automatic save",
            is_automated=is_automated
        )
        self.versions.append(version)
        return version

    def get_version_history(self) -> List[Dict]:
        """Get list of versions with metadata."""
        return [v.to_dict() for v in self.versions]

    def revert_to_version(self, version_id: str):
        """Revert to specific version."""
        for v in self.versions:
            if v.version_id == version_id:
                # Save current state to undo stack
                self.undo_stack.append({
                    "sentences": dict(self.sentences),
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Restore version
                self.sentences = {s.id: s for s in v.sentences}
                self.redo_stack.clear()
                return True
        return False

    def undo(self):
        """Undo last change."""
        if self.undo_stack:
            # Save current state
            self.redo_stack.append({
                "sentences": dict(self.sentences),
                "timestamp": datetime.utcnow().isoformat()
            })
            # Restore
            state = self.undo_stack.pop()
            self.sentences = state["sentences"]
            return True
        return False

    def redo(self):
        """Redo last undone change."""
        if self.redo_stack:
            self.undo_stack.append({
                "sentences": dict(self.sentences),
                "timestamp": datetime.utcnow().isoformat()
            })
            state = self.redo_stack.pop()
            self.sentences = state["sentences"]
            return True
        return False

    def _record_action(self, action: str, sentence_id: str, user: str):
        """Record action for audit trail."""
        self.undo_stack.append({
            "action": action,
            "sentence_id": sentence_id,
            "timestamp": datetime.utcnow().isoformat(),
            "user": user,
            "sentences": dict(self.sentences)
        })

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total = len(self.sentences)
        mapped = sum(1 for s in self.sentences.values() if s.status == SentenceStatus.MAPPED)
        confusing = sum(1 for s in self.sentences.values() if s.is_confusing)
        needs_clarification = sum(1 for s in self.sentences.values() if s.needs_clarification)
        avg_confidence = sum(s.confidence_score for s in self.sentences.values()) / total if total > 0 else 0
        
        return {
            "total_sentences": total,
            "mapped_sentences": mapped,
            "confusing_sentences": confusing,
            "clarification_needed": needs_clarification,
            "unassigned_sentences": total - mapped,
            "average_confidence": avg_confidence,
            "versions_created": len(self.versions),
            "code_references": len(self.code_references),
            "mapping_patterns_learned": len(self.mapping_patterns)
        }

    def export_as_json(self) -> Dict:
        """Export all sentence metadata as JSON."""
        return {
            "job_id": self.job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "sentences": {sid: s.to_dict() for sid, s in self.sentences.items()},
            "code_references": {rid: r.to_dict() for rid, r in self.code_references.items()},
            "version_history": [v.to_dict() for v in self.versions],
            "statistics": self.get_stats(),
            "mapping_patterns": dict(self.mapping_patterns)
        }
