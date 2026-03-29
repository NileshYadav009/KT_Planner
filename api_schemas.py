"""
ENTERPRISE KT PLANNER - PYDANTIC MODELS & API SCHEMAS

These models define the request/response contracts for the enterprise API.
"""

from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid

# =====================================================
# ENUMS
# =====================================================

class SectionStatus(str, Enum):
    COVERED = "covered"
    PARTIAL = "partial"
    MISSING = "missing"
    UNCERTAIN = "uncertain"

class MappingType(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"
    CORRECTED = "corrected"

class ReviewType(str, Enum):
    CONFUSING = "confusing"
    NEEDS_CLARIFICATION = "needs_clarification"
    MARKED_IMPORTANT = "marked_important"
    COMMENT = "comment"

class SentenceStatus(str, Enum):
    UNMAPPED = "unmapped"
    MAPPED = "mapped"
    CONFLICTING = "conflicting"
    MARKED_FOR_REVIEW = "marked_for_review"

# =====================================================
# PROJECT MODELS
# =====================================================

class KTProjectCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=255)
    description: Optional[str] = None
    schema_id: str = "default"
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    auto_map_enabled: bool = True

class KTProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    confidence_threshold: Optional[float] = None
    auto_map_enabled: Optional[bool] = None
    status: Optional[Literal["draft", "in_progress", "completed", "archived"]] = None

class KTProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    confidence_threshold: float
    auto_map_enabled: bool
    
    class Config:
        from_attributes = True

# =====================================================
# SENTENCE MODELS
# =====================================================

class SentenceCreate(BaseModel):
    text: str = Field(..., min_length=5)
    sentence_index: int
    start_time_ms: Optional[int] = None
    end_time_ms: Optional[int] = None
    predicted_section_id: Optional[str] = None
    confidence_score: float = 0.0
    importance_weight: float = 1.0

class SentenceUpdate(BaseModel):
    text_edited: Optional[str] = None
    assigned_section_id: Optional[str] = None
    is_important: Optional[bool] = None
    is_confusing: Optional[bool] = None
    needs_clarification: Optional[bool] = None
    importance_weight: Optional[float] = None

class SentenceResponse(BaseModel):
    id: str
    transcript_id: str
    text: str
    text_edited: Optional[str]
    sentence_index: int
    start_time_ms: Optional[int]
    end_time_ms: Optional[int]
    
    # AI Classification
    predicted_section_id: Optional[str]
    confidence_score: float
    
    # User Assignment
    assigned_section_id: Optional[str]
    assigned_at: Optional[datetime]
    assigned_by: Optional[str]
    
    # Flags
    is_important: bool
    is_confusing: bool
    needs_clarification: bool
    importance_weight: float
    status: str
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class SentenceDetailedResponse(SentenceResponse):
    """Extended response with related data"""
    mapping_history: List['MappingResponse'] = []
    reviews: List['ReviewResponse'] = []
    code_references: List['CodeReferenceResponse'] = []

# =====================================================
# MAPPING MODELS
# =====================================================

class MappingCreate(BaseModel):
    sentence_id: str
    section_id: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    assignment_type: MappingType = MappingType.AUTO

class MappingUpdate(BaseModel):
    section_id: Optional[str] = None
    confidence: Optional[float] = None
    status: Optional[Literal["active", "superseded", "archived"]] = None

class MappingResponse(BaseModel):
    id: str
    sentence_id: str
    section_id: str
    confidence: float
    assignment_type: str
    assigned_by: Optional[str]
    assigned_at: Optional[datetime]
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# =====================================================
# CODE REFERENCE MODELS
# =====================================================

class CodeReferenceCreate(BaseModel):
    code_snippet: str = Field(..., min_length=1)
    language: str = Field(default="text")
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    description: Optional[str] = None
    section_id: Optional[str] = None
    sentence_id: Optional[str] = None

class CodeReferenceUpdate(BaseModel):
    code_snippet: Optional[str] = None
    description: Optional[str] = None
    section_id: Optional[str] = None

class CodeReferenceResponse(BaseModel):
    id: str
    sentence_id: Optional[str]
    project_id: str
    code_snippet: str
    language: str
    file_name: Optional[str]
    file_path: Optional[str]
    description: Optional[str]
    section_id: Optional[str]
    created_at: datetime
    created_by: Optional[str]
    
    class Config:
        from_attributes = True

# =====================================================
# REVIEW / ANNOTATION MODELS
# =====================================================

class ReviewCreate(BaseModel):
    sentence_id: str
    review_type: ReviewType
    comment: Optional[str] = None
    suggested_section_id: Optional[str] = None

class ReviewUpdate(BaseModel):
    comment: Optional[str] = None
    suggested_section_id: Optional[str] = None
    status: Optional[Literal["open", "resolved", "acknowledged"]] = None

class ReviewResponse(BaseModel):
    id: str
    sentence_id: str
    review_type: str
    comment: Optional[str]
    suggested_section_id: Optional[str]
    created_by: Optional[str]
    created_at: datetime
    updated_at: datetime
    status: str
    
    class Config:
        from_attributes = True

# =====================================================
# TRANSCRIPT MODELS
# =====================================================

class TranscriptUploadResponse(BaseModel):
    id: str
    project_id: str
    file_name: str
    status: str
    created_at: datetime

class TranscriptResponse(BaseModel):
    id: str
    project_id: str
    raw_text: str
    file_name: Optional[str]
    duration_seconds: Optional[int]
    status: str
    created_at: datetime
    updated_at: datetime
    
    # Computed statistics
    total_sentences: int = 0
    mapped_sentences: int = 0
    confusing_sentences: int = 0
    coverage_percentage: float = 0.0
    
    class Config:
        from_attributes = True

# =====================================================
# COVERAGE & STATISTICS MODELS
# =====================================================

class SectionCoverageResponse(BaseModel):
    section_id: str
    section_title: str
    status: SectionStatus
    mapped_sentence_count: int
    total_expected_sentences: Optional[int]
    confidence_avg: float
    mapped_sentences: List[SentenceResponse] = []
    missing_suggestions: Optional[str] = None

class CoverageReportResponse(BaseModel):
    project_id: str
    total_sections: int
    covered_sections: int
    partial_sections: int
    missing_sections: int
    coverage_percentage: float
    
    confusing_sentences_count: int
    requires_clarification_count: int
    
    sections: List[SectionCoverageResponse]
    
    # Missing content auto-suggestions
    missing_content_checklist: Dict[str, str]  # {section_id: suggested_question}

# =====================================================
# VERSIONING MODELS
# =====================================================

class VersionResponse(BaseModel):
    id: str
    project_id: str
    version_number: int
    version_name: Optional[str]
    change_type: str
    changed_by: Optional[str]
    change_description: Optional[str]
    created_at: datetime
    
    # Summary statistics
    total_sentences: int
    mapped_count: int
    unmapped_count: int
    
    class Config:
        from_attributes = True

class VersionComparisonResponse(BaseModel):
    version_a: VersionResponse
    version_b: VersionResponse
    
    # Differences
    sentences_added: List[str]
    sentences_removed: List[str]
    mappings_changed: List[Dict[str, Any]]
    confidence_delta: float

# =====================================================
# USER CORRECTION MODELS (FOR PATTERN LEARNING)
# =====================================================

class UserCorrectionCreate(BaseModel):
    sentence_id: str
    original_section_id: str
    corrected_section_id: str

class UserCorrectionResponse(BaseModel):
    id: str
    project_id: str
    sentence_id: Optional[str]
    original_section_id: str
    corrected_section_id: str
    sentence_text: str
    original_confidence: float
    correction_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# =====================================================
# BULK OPERATIONS
# =====================================================

class BulkMapRequest(BaseModel):
    sentence_ids: List[str]
    target_section_id: str
    assignment_type: MappingType = MappingType.MANUAL

class BulkMapResponse(BaseModel):
    mapped_count: int
    failed_count: int
    failed_ids: List[str] = []

class RemapAllRequest(BaseModel):
    project_id: str
    use_corrected_patterns: bool = True
    target_confidence_threshold: Optional[float] = None

class RemapAllResponse(BaseModel):
    remapped_count: int
    unchanged_count: int
    accuracy_improvement: float
    changes_summary: Dict[str, int]  # section_id -> count of changes

# =====================================================
# EXPORT MODELS
# =====================================================

class ExportFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    YAML = "yaml"
    PDF = "pdf"

class ExportRequest(BaseModel):
    project_id: str
    format: ExportFormat
    include_code_references: bool = True
    include_version_history: bool = False
    include_reviews: bool = True

class ExportResponse(BaseModel):
    export_id: str
    project_id: str
    format: str
    file_path: str
    created_at: datetime
    size_bytes: int

# =====================================================
# AI ANALYSIS REQUEST/RESPONSE
# =====================================================

class SentenceAnalysisResponse(BaseModel):
    """Response from AI analysis endpoint"""
    sentence_id: str
    predicted_section_id: str
    confidence_score: float
    alternative_sections: List[Dict[str, float]]  # [{section_id: score}, ...]
    is_confusing: bool
    suggestion: Optional[str]
    semantic_keywords: List[str]

class BatchAnalysisRequest(BaseModel):
    transcript_id: str
    confidence_threshold: float = 0.65

class BatchAnalysisResponse(BaseModel):
    transcript_id: str
    sentences_analyzed: int
    high_confidence_count: int
    low_confidence_count: int
    confusing_count: int
    analysis_results: List[SentenceAnalysisResponse]

# =====================================================
# UTILITY MODELS
# =====================================================

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    database_connected: bool
    models_loaded: bool

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str]
    trace_id: Optional[str]

# =====================================================
# UPDATE FORWARD REFERENCES
# =====================================================

SentenceDetailedResponse.update_forward_refs()
