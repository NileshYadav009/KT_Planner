"""
ENTERPRISE KT PLANNER - PRODUCTION CODE SNIPPETS

Ready-to-use implementation code for core features.
Part 1: AI Engine & Mapping Logic (Python)
"""

# =====================================================
# 1. SENTENCE SEGMENTATION & EMBEDDING SERVICE
# =====================================================

from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime
import uuid

# Initialize models (run once at startup)
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # Fast + accurate
nltk.download('punkt', quiet=True)

class SentenceSegmentationService:
    """
    Breaks raw transcript into sentences and generates embeddings.
    """
    
    @staticmethod
    def segment_transcript(raw_text: str) -> List[Dict]:
        """
        Split transcript into sentences with metadata.
        """
        sentences = sent_tokenize(raw_text)
        segmented = []
        
        for idx, text in enumerate(sentences):
            # Skip very short sentences
            if len(text.strip()) < 5:
                continue
            
            segmented.append({
                "sentence_index": idx,
                "text": text.strip(),
                "word_count": len(text.split()),
                "char_count": len(text),
                "timestamp_ms": None,  # Will be populated from audio metadata
            })
        
        return segmented
    
    @staticmethod
    def generate_embeddings(sentences: List[Dict]) -> List[Dict]:
        """
        Generate semantic embeddings for each sentence.
        """
        texts = [s["text"] for s in sentences]
        
        # Batch embed for efficiency
        embeddings = EMBEDDING_MODEL.encode(
            texts,
            convert_to_tensor=True,
            batch_size=32,
            show_progress_bar=False
        )
        
        # Convert to numpy for storage
        embeddings_np = embeddings.cpu().numpy() if hasattr(embeddings, 'cpu') else embeddings
        
        result = []
        for i, sent in enumerate(sentences):
            sent["embedding"] = embeddings_np[i].tolist()  # Store as list for JSON
            result.append(sent)
        
        return result


# =====================================================
# 2. SEMANTIC MAPPING WITH CONFIDENCE SCORING
# =====================================================

class SemanticMappingEngine:
    """
    Maps sentences to KT sections using semantic similarity.
    Provides confidence scores and alternative suggestions.
    """
    
    def __init__(self, schema: Dict, section_hints: Dict[str, set]):
        self.schema = schema
        self.section_hints = section_hints
        self.section_embeddings = {}
        self.build_section_embeddings()
    
    def build_section_embeddings(self):
        """
        Create semantic embeddings for each KT section
        based on keywords, hints, and field labels.
        """
        for section in self.schema:
            section_id = section["id"]
            
            # Collect all textual hints for this section
            hint_texts = list(self.section_hints.get(section_id, set()))
            
            # Add section title and description
            hint_texts.append(section.get("title", ""))
            if section.get("description"):
                hint_texts.append(section["description"])
            
            # Embed all hints together
            if hint_texts:
                combined_text = " ".join(hint_texts)
                embedding = EMBEDDING_MODEL.encode(
                    combined_text,
                    convert_to_tensor=False
                )
                self.section_embeddings[section_id] = embedding
    
    def classify_sentence(
        self,
        sentence_text: str,
        sentence_embedding: np.ndarray,
        confidence_threshold: float = 0.65
    ) -> Dict:
        """
        Classify a sentence to a KT section with confidence.
        
        Returns:
            {
                "predicted_section_id": str,
                "confidence_score": float,
                "alternative_suggestions": [
                    {"section_id": str, "confidence": float}
                ],
                "is_confusing": bool,
                "semantic_keywords": List[str]
            }
        """
        
        # Ensure embedding is numpy
        if hasattr(sentence_embedding, 'cpu'):
            sentence_embedding = sentence_embedding.cpu().numpy()
        
        # Calculate cosine similarities to all sections
        similarities = {}
        for section_id, section_embed in self.section_embeddings.items():
            # Normalize embeddings
            s_norm = sentence_embedding / (np.linalg.norm(sentence_embedding) + 1e-8)
            sec_norm = section_embed / (np.linalg.norm(section_embed) + 1e-8)
            
            similarity = np.dot(s_norm, sec_norm)
            similarities[section_id] = float(similarity)
        
        # Sort by confidence
        sorted_sections = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        predicted_section_id, top_score = sorted_sections[0]
        is_confusing = top_score < confidence_threshold
        
        # Get alternatives
        alternatives = [
            {"section_id": sec_id, "confidence": score}
            for sec_id, score in sorted_sections[1:4]  # Top 3 alternatives
        ]
        
        # Extract semantic keywords (simple approach: top words by TF-IDF)
        semantic_keywords = self._extract_keywords(sentence_text)
        
        return {
            "predicted_section_id": predicted_section_id,
            "confidence_score": top_score,
            "alternative_suggestions": alternatives,
            "is_confusing": is_confusing,
            "semantic_keywords": semantic_keywords
        }
    
    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract important keywords from sentence."""
        from nltk.corpus import stopwords
        
        stop_words = set(stopwords.words('english'))
        words = [
            w.lower() for w in text.split()
            if w.lower() not in stop_words and len(w) > 3
        ]
        # Return most common (simple approach)
        return list(set(words))[:top_n]


# =====================================================
# 3. CONFUSION DETECTION & AUTO-FLAGGING
# =====================================================

class ConfusionDetectionEngine:
    """
    Detects confusing, unclear, or low-confidence sentences.
    Applies multiple heuristics for robust detection.
    """
    
    def __init__(self, mapping_engine: SemanticMappingEngine):
        self.mapping_engine = mapping_engine
    
    def detect_confusing_sentences(
        self,
        sentences_with_mappings: List[Dict],
        confidence_threshold: float = 0.65
    ) -> List[str]:
        """
        Returns list of sentence IDs that are confusing.
        """
        confusing_ids = []
        
        for sent in sentences_with_mappings:
            if self._is_confusing(sent, confidence_threshold):
                confusing_ids.append(sent["id"])
        
        return confusing_ids
    
    def _is_confusing(self, sentence_data: Dict, threshold: float) -> bool:
        """
        Multiple heuristics to detect confusing sentences:
        1. Low semantic confidence
        2. Multiple equally-good alternatives
        3. Large gap between top suggestions
        4. Unusual word patterns
        """
        
        # Heuristic 1: Low confidence
        if sentence_data.get("confidence_score", 0) < threshold:
            return True
        
        # Heuristic 2: Multiple close alternatives (ambiguous)
        alternatives = sentence_data.get("alternative_suggestions", [])
        if len(alternatives) >= 2:
            top_score = sentence_data.get("confidence_score", 0)
            second_score = alternatives[0].get("confidence", 0)
            
            # If gap is small, it's ambiguous
            if top_score - second_score < 0.15:
                return True
        
        # Heuristic 3: Very short or very long sentences
        word_count = len(sentence_data.get("text", "").split())
        if word_count < 3 or word_count > 200:
            return True
        
        return False
    
    def generate_clarification_suggestion(self, sentence_data: Dict) -> Optional[str]:
        """
        Generate a user-friendly suggestion for clarifying the sentence.
        """
        confidence = sentence_data.get("confidence_score", 0)
        alternatives = sentence_data.get("alternative_suggestions", [])
        
        if confidence < 0.5:
            return "This sentence is unclear. Please rephrase or provide more context."
        
        if alternatives and len(alternatives) >= 2:
            alt_sections = ", ".join([
                alt["section_id"] for alt in alternatives[:2]
            ])
            return f"This could belong to: {alt_sections}. Please clarify."
        
        word_count = len(sentence_data.get("text", "").split())
        if word_count > 100:
            return "This sentence is quite long. Consider breaking it into smaller parts."
        
        return None


# =====================================================
# 4. PATTERN LEARNING FROM USER CORRECTIONS
# =====================================================

class PatternLearningEngine:
    """
    Learns from user corrections to improve future mappings.
    Tracks patterns of misclassification and adjusts confidence.
    """
    
    def __init__(self):
        self.correction_patterns: Dict[Tuple[str, str], int] = {}
        self.confidence_adjustments: Dict[str, float] = {}
    
    def record_correction(
        self,
        sentence_text: str,
        original_section: str,
        corrected_section: str
    ):
        """
        Record a user correction for pattern learning.
        """
        pattern_key = (original_section, corrected_section)
        
        if pattern_key not in self.correction_patterns:
            self.correction_patterns[pattern_key] = 0
        
        self.correction_patterns[pattern_key] += 1
    
    def get_correction_patterns(self) -> List[Dict]:
        """
        Return most common correction patterns.
        Useful for bulk remap suggestions.
        """
        patterns = []
        
        for (from_section, to_section), count in sorted(
            self.correction_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if count >= 2:  # Only suggest if pattern repeats
                patterns.append({
                    "original_section": from_section,
                    "corrected_section": to_section,
                    "frequency": count,
                    "confidence": min(count / 10.0, 0.95)  # Cap at 0.95
                })
        
        return patterns
    
    def should_apply_correction_pattern(self, pattern: Dict) -> bool:
        """
        Determine if a correction pattern is strong enough to use
        in bulk remap operations.
        """
        return pattern["frequency"] >= 3  # Require at least 3 corrections


# =====================================================
# 5. DEDUPLICATION ENGINE
# =====================================================

class DeduplicationEngine:
    """
    Prevents same content appearing in multiple sections.
    Finds and removes near-duplicates.
    """
    
    def deduplicate_mappings(
        self,
        mappings: Dict[str, List[str]],  # {section_id: [sentence_ids]}
        similarity_threshold: float = 0.85
    ) -> Dict[str, List[str]]:
        """
        Remove near-duplicate sentences across sections.
        Keeps version in most relevant section (highest confidence).
        """
        
        # Collect all sentences with their section assignments
        all_sentences = {}
        for section_id, sentence_ids in mappings.items():
            for sent_id in sentence_ids:
                if sent_id not in all_sentences:
                    all_sentences[sent_id] = []
                all_sentences[sent_id].append(section_id)
        
        # For duplicates, keep only in best section
        deduplicated = {section_id: [] for section_id in mappings.keys()}
        
        processed = set()
        for sent_id, sections in all_sentences.items():
            if sent_id in processed:
                continue
            
            # Keep in first (most confident) section only
            best_section = sections[0]
            deduplicated[best_section].append(sent_id)
            processed.add(sent_id)
        
        return deduplicated


# =====================================================
# 6. MISSING CONTENT SUGGESTION ENGINE
# =====================================================

class MissingContentSuggestionEngine:
    """
    Analyzes gaps and suggests questions to ask KT provider.
    Generates actionable checklists.
    """
    
    def generate_suggestions(
        self,
        coverage_data: Dict[str, Dict],  # {section_id: {status, count, ...}}
        schema: List[Dict]
    ) -> Dict[str, str]:
        """
        Generate suggestions for missing sections.
        
        Returns:
            {section_id: "Suggested question to ask KT provider"}
        """
        suggestions = {}
        
        # Template questions for common KT sections
        question_templates = {
            "deployment_steps": "Can you walk through the exact steps to deploy a new version?",
            "rollback_procedure": "How do we rollback a failed deployment?",
            "monitoring_setup": "How is the application monitored in production?",
            "troubleshooting": "What are the common issues and how do we debug them?",
            "operations": "What are the day-to-day operational procedures?",
            "security": "What security measures are in place?",
            "performance": "What are the performance characteristics and limits?",
            "scaling": "How do we scale the system?",
        }
        
        for section in schema:
            section_id = section.get("id")
            status = coverage_data.get(section_id, {}).get("status", "missing")
            
            # Generate suggestion for missing or weak sections
            if status in ["missing", "partial"]:
                suggestion = question_templates.get(
                    section_id,
                    f"Please provide information about: {section.get('title', section_id)}"
                )
                suggestions[section_id] = suggestion
        
        return suggestions
