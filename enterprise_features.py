"""
ENTERPRISE KT PLANNER - SENTENCE-LEVEL MAPPING & FEATURES

Extends main.py with enterprise features:
- Sentence-level granularity
- Drag & drop mapping
- Confusion detection
- Code linking
- Real-time WebSocket updates
"""

from fastapi import WebSocket, Query
from typing import List, Dict, Set, Optional
from datetime import datetime
import json
import uuid

# =====================================================
# IN-MEMORY STORES (Replace with DB later)
# =====================================================

# Store sentences with metadata
SENTENCE_STORE: Dict[str, Dict] = {}  # sentence_id -> {text, confidence, section, ...}
CODE_REFERENCES: Dict[str, List[Dict]] = {}  # sentence_id -> [{code, language, ...}]
REVIEWS: Dict[str, List[Dict]] = {}  # sentence_id -> [{type, comment, ...}]
SENTENCE_MAPPINGS: Dict[str, Dict] = {}  # sentence_id -> {section_id, confidence, status}
CONFUSED_SENTENCES: Set[str] = set()  # sentence_ids marked as confusing

# WebSocket connections per project
ACTIVE_CONNECTIONS: Dict[str, List[WebSocket]] = {}

# =====================================================
# SENTENCE SEGMENTATION & ANALYSIS
# =====================================================

def segment_and_analyze_transcript(transcript_text: str, project_id: str) -> List[Dict]:
    """
    Break transcript into sentences and analyze each.
    Returns list of sentences with metadata.
    """
    from nltk.tokenize import sent_tokenize
    import numpy as np
    from enterprise_ai_engine import SemanticMappingEngine, ConfusionDetectionEngine
    from ai import get_sentence_model
    
    try:
        nltk_import = __import__('nltk')
        nltk_import.download('punkt', quiet=True)
    except:
        pass
    
    sentences_text = sent_tokenize(transcript_text)
    result = []
    
    # Load embedding model
    model = get_sentence_model()
    if not model:
        # Fallback: return basic sentences
        for idx, text in enumerate(sentences_text):
            result.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "index": idx,
                "confidence": 0.5,
                "predicted_section": None,
                "status": "unmapped"
            })
        return result
    
    # Generate embeddings
    texts = [s.strip() for s in sentences_text if len(s.strip()) > 5]
    embeddings = model.encode(texts, convert_to_tensor=False, batch_size=32)
    
    # Initialize mapping engine
    try:
        engine = SemanticMappingEngine(
            schema=[],  # Will use from main.py SCHEMA
            section_hints={}
        )
    except:
        engine = None
    
    # Process each sentence
    for idx, text in enumerate(texts):
        if idx < len(embeddings):
            embedding = embeddings[idx]
        else:
            embedding = None
        
        sent_id = str(uuid.uuid4())
        
        # Classify to section
        predicted_section = None
        confidence = 0.5
        
        if engine and embedding is not None:
            try:
                analysis = engine.classify_sentence(
                    text,
                    embedding,
                    confidence_threshold=0.65
                )
                predicted_section = analysis.get("predicted_section_id")
                confidence = analysis.get("confidence_score", 0.5)
                is_confusing = analysis.get("is_confusing", False)
                
                if is_confusing:
                    CONFUSED_SENTENCES.add(sent_id)
            except:
                pass
        
        sentence_data = {
            "id": sent_id,
            "text": text,
            "index": idx,
            "confidence": confidence,
            "predicted_section": predicted_section,
            "assigned_section": None,
            "is_important": False,
            "is_confusing": sent_id in CONFUSED_SENTENCES,
            "status": "unmapped",
            "created_at": datetime.utcnow().isoformat(),
            "embedding": embedding.tolist() if embedding is not None else None
        }
        
        SENTENCE_STORE[sent_id] = sentence_data
        SENTENCE_MAPPINGS[sent_id] = {
            "section_id": predicted_section,
            "confidence": confidence,
            "status": "auto"
        }
        
        result.append(sentence_data)
    
    return result


# =====================================================
# DRAG & DROP MAPPING
# =====================================================

async def map_sentence_to_section(
    sentence_id: str,
    section_id: str,
    project_id: str,
    user_id: Optional[str] = None
) -> Dict:
    """
    Update sentence mapping (from drag-drop or manual assignment).
    Records pattern for learning.
    """
    if sentence_id not in SENTENCE_STORE:
        raise ValueError(f"Sentence {sentence_id} not found")
    
    sentence = SENTENCE_STORE[sentence_id]
    old_section = sentence.get("assigned_section")
    
    # Update mapping
    sentence["assigned_section"] = section_id
    sentence["status"] = "mapped"
    sentence["updated_at"] = datetime.utcnow().isoformat()
    
    SENTENCE_MAPPINGS[sentence_id] = {
        "section_id": section_id,
        "confidence": sentence.get("confidence", 0.5),
        "status": "manual",
        "assigned_at": datetime.utcnow().isoformat()
    }
    
    # Broadcast WebSocket update
    await broadcast_update(project_id, {
        "event": "sentence_mapped",
        "sentence_id": sentence_id,
        "section_id": section_id,
        "old_section": old_section
    })
    
    return sentence


# =====================================================
# CODE REFERENCE LINKING
# =====================================================

def add_code_reference(
    sentence_id: str,
    code_snippet: str,
    language: str = "text",
    description: Optional[str] = None,
    file_name: Optional[str] = None
) -> Dict:
    """
    Link code snippet to a sentence.
    """
    if sentence_id not in SENTENCE_STORE:
        raise ValueError(f"Sentence {sentence_id} not found")
    
    code_ref = {
        "id": str(uuid.uuid4()),
        "code": code_snippet,
        "language": language,
        "description": description or "",
        "file_name": file_name or "",
        "created_at": datetime.utcnow().isoformat()
    }
    
    if sentence_id not in CODE_REFERENCES:
        CODE_REFERENCES[sentence_id] = []
    
    CODE_REFERENCES[sentence_id].append(code_ref)
    
    # Update sentence to show it has code ref
    SENTENCE_STORE[sentence_id]["has_code_ref"] = True
    
    return code_ref


def get_code_references(sentence_id: str) -> List[Dict]:
    """Get all code references for a sentence."""
    return CODE_REFERENCES.get(sentence_id, [])


# =====================================================
# CONFUSION DETECTION & REVIEWS
# =====================================================

def mark_confusing(
    sentence_id: str,
    comment: Optional[str] = None,
    suggested_section: Optional[str] = None
) -> Dict:
    """
    Mark sentence as confusing or needing clarification.
    """
    if sentence_id not in SENTENCE_STORE:
        raise ValueError(f"Sentence {sentence_id} not found")
    
    sentence = SENTENCE_STORE[sentence_id]
    sentence["is_confusing"] = True
    sentence["status"] = "needs_review"
    CONFUSED_SENTENCES.add(sentence_id)
    
    review = {
        "id": str(uuid.uuid4()),
        "type": "confusing",
        "comment": comment or "",
        "suggested_section": suggested_section,
        "created_at": datetime.utcnow().isoformat()
    }
    
    if sentence_id not in REVIEWS:
        REVIEWS[sentence_id] = []
    
    REVIEWS[sentence_id].append(review)
    
    return {
        "sentence_id": sentence_id,
        "review": review,
        "status": "marked_confusing"
    }


def get_reviews(sentence_id: str) -> List[Dict]:
    """Get all reviews for a sentence."""
    return REVIEWS.get(sentence_id, [])


def get_confusing_sentences(project_id: str) -> List[Dict]:
    """Get all sentences marked as confusing."""
    result = []
    for sent_id, sentence in SENTENCE_STORE.items():
        if sentence.get("is_confusing", False):
            result.append({
                "sentence": sentence,
                "reviews": REVIEWS.get(sent_id, [])
            })
    return result


# =====================================================
# COVERAGE ANALYSIS
# =====================================================

def analyze_coverage(schema: List[Dict]) -> Dict:
    """
    Analyze coverage across all KT sections.
    """
    coverage = {}
    total_mapped = 0
    
    for section in schema:
        section_id = section.get("id")
        
        # Count sentences mapped to this section
        mapped_sentences = [
            s for sid, s in SENTENCE_STORE.items()
            if s.get("assigned_section") == section_id
        ]
        
        status = "missing"
        if len(mapped_sentences) >= 2:
            status = "covered"
        elif len(mapped_sentences) > 0:
            status = "partial"
        
        avg_conf = sum([s.get("confidence", 0) for s in mapped_sentences]) / len(mapped_sentences) if mapped_sentences else 0
        
        coverage[section_id] = {
            "section": section.get("title", section_id),
            "status": status,
            "count": len(mapped_sentences),
            "confidence": avg_conf,
            "sentences": mapped_sentences
        }
        
        if status != "missing":
            total_mapped += 1
    
    return {
        "total_sections": len(schema),
        "covered": len([c for c in coverage.values() if c["status"] == "covered"]),
        "partial": len([c for c in coverage.values() if c["status"] == "partial"]),
        "missing": len([c for c in coverage.values() if c["status"] == "missing"]),
        "coverage_percentage": (len([c for c in coverage.values() if c["status"] != "missing"]) / len(schema) * 100) if schema else 0,
        "sections": coverage,
        "confusing_count": len(CONFUSED_SENTENCES)
    }


# =====================================================
# WEBSOCKET BROADCASTING
# =====================================================

async def connect_websocket(project_id: str, websocket: WebSocket):
    """Register WebSocket connection for project."""
    await websocket.accept()
    if project_id not in ACTIVE_CONNECTIONS:
        ACTIVE_CONNECTIONS[project_id] = []
    ACTIVE_CONNECTIONS[project_id].append(websocket)


async def disconnect_websocket(project_id: str, websocket: WebSocket):
    """Unregister WebSocket connection."""
    if project_id in ACTIVE_CONNECTIONS:
        ACTIVE_CONNECTIONS[project_id].remove(websocket)
        if not ACTIVE_CONNECTIONS[project_id]:
            del ACTIVE_CONNECTIONS[project_id]


async def broadcast_update(project_id: str, message: Dict):
    """Broadcast update to all connected clients for a project."""
    if project_id not in ACTIVE_CONNECTIONS:
        return
    
    disconnected = []
    for connection in ACTIVE_CONNECTIONS[project_id]:
        try:
            await connection.send_json(message)
        except Exception as e:
            disconnected.append(connection)
    
    # Clean up disconnected clients
    for conn in disconnected:
        if conn in ACTIVE_CONNECTIONS[project_id]:
            ACTIVE_CONNECTIONS[project_id].remove(conn)


# =====================================================
# BULK OPERATIONS
# =====================================================

async def bulk_map_sentences(
    sentence_ids: List[str],
    target_section: str,
    project_id: str
) -> Dict:
    """Map multiple sentences to a section at once."""
    mapped_count = 0
    failed_count = 0
    
    for sent_id in sentence_ids:
        try:
            await map_sentence_to_section(sent_id, target_section, project_id)
            mapped_count += 1
        except Exception as e:
            failed_count += 1
    
    await broadcast_update(project_id, {
        "event": "bulk_map_complete",
        "mapped_count": mapped_count,
        "failed_count": failed_count
    })
    
    return {
        "mapped_count": mapped_count,
        "failed_count": failed_count,
        "total": len(sentence_ids)
    }


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def get_sentence(sentence_id: str) -> Optional[Dict]:
    """Get a sentence by ID."""
    return SENTENCE_STORE.get(sentence_id)


def get_all_sentences(filter_status: Optional[str] = None) -> List[Dict]:
    """Get all sentences, optionally filtered by status."""
    sentences = list(SENTENCE_STORE.values())
    if filter_status:
        sentences = [s for s in sentences if s.get("status") == filter_status]
    return sentences


def edit_sentence_text(sentence_id: str, new_text: str) -> Dict:
    """Edit sentence text."""
    if sentence_id not in SENTENCE_STORE:
        raise ValueError(f"Sentence {sentence_id} not found")
    
    SENTENCE_STORE[sentence_id]["text_edited"] = new_text
    SENTENCE_STORE[sentence_id]["updated_at"] = datetime.utcnow().isoformat()
    
    return SENTENCE_STORE[sentence_id]


def mark_important(sentence_id: str, important: bool = True) -> Dict:
    """Mark sentence as important."""
    if sentence_id not in SENTENCE_STORE:
        raise ValueError(f"Sentence {sentence_id} not found")
    
    SENTENCE_STORE[sentence_id]["is_important"] = important
    
    return SENTENCE_STORE[sentence_id]
