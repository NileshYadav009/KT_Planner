"""
ENTERPRISE API ROUTES - RESTful integration with FastAPI

Registers new endpoints in main.py for:
- GET sentences (with filtering)
- POST map sentence to section (drag-drop)
- POST code reference linking
- POST mark confusing/review
- POST bulk operations
- WebSocket real-time updates
"""

from fastapi import APIRouter, WebSocket, Depends, HTTPException, Query
from typing import List, Optional, Dict
from pydantic import BaseModel

# Import enterprise features
import enterprise_features as ent

# =====================================================
# PYDANTIC MODELS (Request/Response)
# =====================================================

class SentenceResponse(BaseModel):
    id: str
    text: str
    index: int
    confidence: float
    predicted_section: Optional[str] = None
    assigned_section: Optional[str] = None
    status: str  # unmapped, mapped, needs_review
    is_confusing: bool = False
    is_important: bool = False
    has_code_ref: bool = False
    
    class Config:
        from_attributes = True


class MapSentenceRequest(BaseModel):
    sentence_id: str
    section_id: str
    project_id: str
    user_id: Optional[str] = None


class CodeReferenceRequest(BaseModel):
    sentence_id: str
    code_snippet: str
    language: str = "text"
    description: Optional[str] = None
    file_name: Optional[str] = None


class ReviewRequest(BaseModel):
    sentence_id: str
    comment: Optional[str] = None
    suggested_section: Optional[str] = None


class BulkMapRequest(BaseModel):
    sentence_ids: List[str]
    target_section: str
    project_id: str


class EditSentenceRequest(BaseModel):
    sentence_id: str
    new_text: str


# =====================================================
# SETUP ROUTER
# =====================================================

def create_enterprise_router():
    """
    Factory function to create enterprise API router.
    Call this in main.py with: app.include_router(create_enterprise_router())
    """
    router = APIRouter(prefix="/api/v1", tags=["enterprise"])
    
    # =====================================================
    # SENTENCE ENDPOINTS
    # =====================================================
    
    @router.get("/sentences", response_model=List[SentenceResponse])
    async def list_sentences(
        status: Optional[str] = Query(None, description="Filter by status: unmapped, mapped, needs_review"),
        confusing_only: bool = Query(False, description="Only return confusing sentences")
    ):
        """Get all sentences, optionally filtered."""
        sentences = ent.get_all_sentences(filter_status=status)
        
        if confusing_only:
            sentences = [s for s in sentences if s.get("is_confusing", False)]
        
        return sentences
    
    
    @router.get("/sentences/{sentence_id}", response_model=SentenceResponse)
    async def get_sentence(sentence_id: str):
        """Get a specific sentence."""
        sentence = ent.get_sentence(sentence_id)
        if not sentence:
            raise HTTPException(status_code=404, detail="Sentence not found")
        return sentence
    
    
    @router.post("/sentences/{sentence_id}/edit")
    async def edit_sentence(sentence_id: str, request: EditSentenceRequest):
        """Edit sentence text."""
        try:
            result = ent.edit_sentence_text(sentence_id, request.new_text)
            return {"status": "success", "sentence": result}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    
    @router.post("/sentences/{sentence_id}/importance")
    async def toggle_importance(sentence_id: str, important: bool = Query(...)):
        """Mark or unmark sentence as important."""
        try:
            result = ent.mark_important(sentence_id, important)
            return {"status": "success", "sentence": result}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    
    # =====================================================
    # MAPPING ENDPOINTS (DRAG & DROP)
    # =====================================================
    
    @router.post("/sentences/map")
    async def map_sentence(request: MapSentenceRequest):
        """
        Map a sentence to a section (from drag-drop or manual assignment).
        Triggers WebSocket broadcast to all connected clients.
        """
        try:
            result = await ent.map_sentence_to_section(
                request.sentence_id,
                request.section_id,
                request.project_id,
                request.user_id
            )
            return {"status": "success", "sentence": result}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    
    @router.post("/sentences/bulk-map")
    async def bulk_map(request: BulkMapRequest):
        """Bulk map multiple sentences to a section."""
        result = await ent.bulk_map_sentences(
            request.sentence_ids,
            request.target_section,
            request.project_id
        )
        return result
    
    
    # =====================================================
    # CODE REFERENCE ENDPOINTS
    # =====================================================
    
    @router.post("/sentences/{sentence_id}/code-references")
    async def add_code_reference(sentence_id: str, request: CodeReferenceRequest):
        """Link a code snippet to a sentence."""
        try:
            code_ref = ent.add_code_reference(
                sentence_id,
                request.code_snippet,
                request.language,
                request.description,
                request.file_name
            )
            return {"status": "success", "code_reference": code_ref}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    
    @router.get("/sentences/{sentence_id}/code-references")
    async def get_code_references(sentence_id: str):
        """Get all code references for a sentence."""
        references = ent.get_code_references(sentence_id)
        return {"code_references": references}
    
    
    # =====================================================
    # REVIEW & CONFUSION ENDPOINTS
    # =====================================================
    
    @router.post("/sentences/{sentence_id}/reviews")
    async def add_review(sentence_id: str, request: ReviewRequest):
        """Mark sentence as confusing or needing review."""
        try:
            result = ent.mark_confusing(
                sentence_id,
                request.comment,
                request.suggested_section
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    
    @router.get("/sentences/{sentence_id}/reviews")
    async def get_sentence_reviews(sentence_id: str):
        """Get all reviews for a sentence."""
        reviews = ent.get_reviews(sentence_id)
        return {"reviews": reviews}
    
    
    @router.get("/confusing-sentences")
    async def list_confusing_sentences(project_id: Optional[str] = Query(None)):
        """Get all sentences marked as confusing."""
        confusing = ent.get_confusing_sentences(project_id or "default")
        return {"confusing_sentences": confusing}
    
    
    # =====================================================
    # COVERAGE ANALYSIS
    # =====================================================
    
    @router.post("/coverage-analysis")
    async def analyze_coverage(schema: List[Dict]):
        """Analyze coverage across all KT sections."""
        coverage = ent.analyze_coverage(schema)
        return coverage
    
    
    # =====================================================
    # WEBSOCKET ENDPOINT (REAL-TIME UPDATES)
    # =====================================================
    
    @router.websocket("/ws/{project_id}")
    async def websocket_endpoint(websocket: WebSocket, project_id: str):
        """
        WebSocket connection for real-time updates.
        
        Usage:
        - Client connects: ws://localhost:8000/api/v1/ws/project-123
        - Events received: sentence_mapped, bulk_map_complete, etc.
        - Messages: JSON with event type and data
        """
        await ent.connect_websocket(project_id, websocket)
        try:
            while True:
                # Keep connection open, receive any client messages
                data = await websocket.receive_text()
                # Server focuses on broadcasting updates, not processing client input
        except Exception as e:
            await ent.disconnect_websocket(project_id, websocket)
    
    
    return router


# =====================================================
# INTEGRATION INSTRUCTIONS
# =====================================================

"""
To integrate this into main.py:

1. At the top of main.py, add:
   from enterprise_api_routes import create_enterprise_router

2. After FastAPI app initialization, add:
   app.include_router(create_enterprise_router())

3. When processing transcript (in process_upload_task):
   - After transcription, call: segment_and_analyze_transcript(transcript, project_id)
   - This will populate SENTENCE_STORE with analyzed sentences

4. Add to requirements.txt:
   - All dependencies already in main.py (fastapi, pydantic, etc.)

5. Frontend integration:
   - Connect WebSocket to ws://localhost:8000/api/v1/ws/{project_id}
   - Make POST requests to new endpoints for drag-drop, code linking, reviews

Example workflow:
   1. User uploads audio
   2. Backend transcribes and creates sentences
   3. Frontend fetches sentences from GET /api/v1/sentences
   4. User drags sentence to section
   5. Frontend POSTs to POST /api/v1/sentences/map
   6. Backend broadcasts update via WebSocket
   7. All connected clients see updated mapping
"""
