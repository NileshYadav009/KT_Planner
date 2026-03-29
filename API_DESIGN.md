"""
ENTERPRISE KT PLANNER - API DESIGN & ENDPOINTS

Complete API specification with all endpoints needed for enterprise features.
"""

# =====================================================
# API ENDPOINT SPECIFICATION
# =====================================================

BASE_URL = "http://localhost:8000/api/v1"

# =====================================================
# PROJECT MANAGEMENT ENDPOINTS
# =====================================================

"""
POST   /projects
GET    /projects
GET    /projects/{project_id}
PUT    /projects/{project_id}
DELETE /projects/{project_id}

POST   /projects/{project_id}/archive
POST   /projects/{project_id}/duplicate
"""

PROJECT_ENDPOINTS = {
    "create": "POST /projects",
    "list": "GET /projects?skip=0&limit=10&status=active",
    "get": "GET /projects/{project_id}",
    "update": "PUT /projects/{project_id}",
    "delete": "DELETE /projects/{project_id}",
    "archive": "POST /projects/{project_id}/archive",
    "duplicate": "POST /projects/{project_id}/duplicate",
    "coverage_report": "GET /projects/{project_id}/coverage",
}

# =====================================================
# TRANSCRIPT ENDPOINTS
# =====================================================

"""
POST   /transcripts/upload
GET    /projects/{project_id}/transcripts
GET    /transcripts/{transcript_id}
DELETE /transcripts/{transcript_id}

POST   /transcripts/{transcript_id}/process-async
GET    /transcripts/{transcript_id}/status
"""

TRANSCRIPT_ENDPOINTS = {
    "upload": "POST /transcripts/upload (multipart/form-data)",
    "list_by_project": "GET /projects/{project_id}/transcripts",
    "get": "GET /transcripts/{transcript_id}",
    "delete": "DELETE /transcripts/{transcript_id}",
    "process_async": "POST /transcripts/{transcript_id}/process-async",
    "status": "GET /transcripts/{transcript_id}/status",
}

# =====================================================
# SENTENCE ENDPOINTS - CORE FUNCTIONALITY
# =====================================================

"""
GET    /transcripts/{transcript_id}/sentences
GET    /transcripts/{transcript_id}/sentences/{sentence_id}
PUT    /transcripts/{transcript_id}/sentences/{sentence_id}
DELETE /transcripts/{transcript_id}/sentences/{sentence_id}

POST   /transcripts/{transcript_id}/sentences/bulk-map
POST   /transcripts/{transcript_id}/sentences/search
"""

SENTENCE_ENDPOINTS = {
    "list": "GET /transcripts/{transcript_id}/sentences?skip=0&limit=50&status=unmapped",
    "get": "GET /transcripts/{transcript_id}/sentences/{sentence_id}",
    "update": "PUT /transcripts/{transcript_id}/sentences/{sentence_id}",
    "delete": "DELETE /transcripts/{transcript_id}/sentences/{sentence_id}",
    "bulk_map": "POST /transcripts/{transcript_id}/sentences/bulk-map",
    "search": "POST /transcripts/{transcript_id}/sentences/search",
    "get_by_section": "GET /transcripts/{transcript_id}/sentences?assigned_section={section_id}",
    "get_confusing": "GET /transcripts/{transcript_id}/sentences?is_confusing=true",
}

# =====================================================
# MAPPING ENDPOINTS
# =====================================================

"""
POST   /sentences/{sentence_id}/mappings
GET    /sentences/{sentence_id}/mappings
PUT    /sentences/{sentence_id}/mappings/{mapping_id}
DELETE /sentences/{sentence_id}/mappings/{mapping_id}

POST   /projects/{project_id}/remap-all
GET    /projects/{project_id}/mapping-suggestions
"""

MAPPING_ENDPOINTS = {
    "create": "POST /sentences/{sentence_id}/mappings",
    "list": "GET /sentences/{sentence_id}/mappings",
    "update": "PUT /sentences/{sentence_id}/mappings/{mapping_id}",
    "delete": "DELETE /sentences/{sentence_id}/mappings/{mapping_id}",
    "remap_all": "POST /projects/{project_id}/remap-all",
    "get_suggestions": "GET /projects/{project_id}/mapping-suggestions",
}

# =====================================================
# CODE REFERENCE ENDPOINTS
# =====================================================

"""
POST   /code-references
GET    /projects/{project_id}/code-references
GET    /sentences/{sentence_id}/code-references
PUT    /code-references/{reference_id}
DELETE /code-references/{reference_id}

GET    /code-references/section/{section_id}
"""

CODE_REFERENCE_ENDPOINTS = {
    "create": "POST /code-references",
    "list_by_project": "GET /projects/{project_id}/code-references?section_id=optional",
    "list_by_sentence": "GET /sentences/{sentence_id}/code-references",
    "update": "PUT /code-references/{reference_id}",
    "delete": "DELETE /code-references/{reference_id}",
    "list_by_section": "GET /code-references/section/{section_id}",
}

# =====================================================
# REVIEW / ANNOTATION ENDPOINTS
# =====================================================

"""
POST   /reviews
GET    /sentences/{sentence_id}/reviews
PUT    /reviews/{review_id}
DELETE /reviews/{review_id}

GET    /projects/{project_id}/reviews/confusing
GET    /projects/{project_id}/reviews/pending-clarification
"""

REVIEW_ENDPOINTS = {
    "create": "POST /reviews",
    "list_by_sentence": "GET /sentences/{sentence_id}/reviews",
    "update": "PUT /reviews/{review_id}",
    "delete": "DELETE /reviews/{review_id}",
    "list_confusing": "GET /projects/{project_id}/reviews/confusing",
    "list_pending_clarification": "GET /projects/{project_id}/reviews/pending-clarification",
    "list_all": "GET /projects/{project_id}/reviews",
}

# =====================================================
# AI & ANALYSIS ENDPOINTS
# =====================================================

"""
POST   /ai/analyze-sentence
POST   /ai/analyze-batch
POST   /ai/detect-confusion
POST   /ai/suggest-corrections

GET    /ai/section-hints/{section_id}
"""

AI_ENDPOINTS = {
    "analyze_single": "POST /ai/analyze-sentence",
    "analyze_batch": "POST /ai/analyze-batch",
    "detect_confusion": "POST /ai/detect-confusion",
    "suggest_corrections": "POST /ai/suggest-corrections",
    "get_section_hints": "GET /ai/section-hints/{section_id}",
}

# =====================================================
# COVERAGE & REPORTING ENDPOINTS
# =====================================================

"""
GET    /projects/{project_id}/coverage
GET    /projects/{project_id}/coverage/by-section
GET    /projects/{project_id}/missing-content-checklist

POST   /projects/{project_id}/generate-suggestions
"""

COVERAGE_ENDPOINTS = {
    "get_coverage": "GET /projects/{project_id}/coverage",
    "by_section": "GET /projects/{project_id}/coverage/by-section",
    "missing_checklist": "GET /projects/{project_id}/missing-content-checklist",
    "generate_suggestions": "POST /projects/{project_id}/generate-suggestions",
}

# =====================================================
# VERSIONING ENDPOINTS
# =====================================================

"""
GET    /projects/{project_id}/versions
GET    /projects/{project_id}/versions/{version_id}
POST   /projects/{project_id}/versions/create-checkpoint
GET    /projects/{project_id}/versions/{v1}/compare/{v2}

POST   /projects/{project_id}/restore-version/{version_id}
"""

VERSION_ENDPOINTS = {
    "list": "GET /projects/{project_id}/versions",
    "get": "GET /projects/{project_id}/versions/{version_id}",
    "create_checkpoint": "POST /projects/{project_id}/versions/create-checkpoint",
    "compare": "GET /projects/{project_id}/versions/{v1}/compare/{v2}",
    "restore": "POST /projects/{project_id}/versions/{version_id}/restore",
}

# =====================================================
# EXPORT ENDPOINTS
# =====================================================

"""
POST   /projects/{project_id}/export/{format}
GET    /exports/{export_id}
GET    /exports/{export_id}/download
"""

EXPORT_ENDPOINTS = {
    "export": "POST /projects/{project_id}/export/{format}",
    "get_status": "GET /exports/{export_id}",
    "download": "GET /exports/{export_id}/download",
}

# =====================================================
# BULK OPERATIONS
# =====================================================

"""
POST   /projects/{project_id}/bulk-corrections
POST   /projects/{project_id}/bulk-import-mappings
"""

BULK_ENDPOINTS = {
    "bulk_corrections": "POST /projects/{project_id}/bulk-corrections",
    "bulk_import": "POST /projects/{project_id}/bulk-import-mappings",
}

# =====================================================
# WEBSOCKET ENDPOINTS (REAL-TIME UPDATES)
# =====================================================

"""
WS ws://localhost:8000/ws/projects/{project_id}
   - On connect: receive current state
   - On sentence edit: receive update
   - On mapping change: receive update
   - On review added: receive notification
"""

WEBSOCKET_ENDPOINTS = {
    "project_updates": "WS /ws/projects/{project_id}",
}

# =====================================================
# DETAILED API REQUEST/RESPONSE EXAMPLES
# =====================================================

API_EXAMPLES = {
    
    # ===== SENTENCE MAPPING =====
    "sentence_update": {
        "endpoint": "PUT /transcripts/{transcript_id}/sentences/{sentence_id}",
        "request": {
            "text_edited": "Updated sentence content",
            "assigned_section_id": "deployment_steps",
            "is_important": True,
            "needs_clarification": False
        },
        "response": {
            "id": "uuid",
            "text": "original text",
            "text_edited": "Updated sentence content",
            "assigned_section_id": "deployment_steps",
            "is_important": True,
            "updated_at": "2026-03-29T10:00:00Z"
        }
    },
    
    # ===== BULK MAPPING =====
    "bulk_map": {
        "endpoint": "POST /transcripts/{transcript_id}/sentences/bulk-map",
        "request": {
            "sentence_ids": ["id1", "id2", "id3"],
            "target_section_id": "deployment_steps",
            "assignment_type": "manual"
        },
        "response": {
            "mapped_count": 3,
            "failed_count": 0,
            "failed_ids": []
        }
    },
    
    # ===== CODE REFERENCE =====
    "add_code_reference": {
        "endpoint": "POST /code-references",
        "request": {
            "sentence_id": "uuid",
            "code_snippet": "kubectl apply -f deployment.yaml",
            "language": "bash",
            "file_name": "deployment.yaml",
            "section_id": "deployment_steps",
            "description": "Command to deploy the application"
        },
        "response": {
            "id": "uuid",
            "sentence_id": "uuid",
            "code_snippet": "kubectl apply -f deployment.yaml",
            "language": "bash",
            "section_id": "deployment_steps"
        }
    },
    
    # ===== REVIEW / ANNOTATION =====
    "mark_confusing": {
        "endpoint": "POST /reviews",
        "request": {
            "sentence_id": "uuid",
            "review_type": "confusing",
            "comment": "This sentence is unclear - needs clarification from KT provider",
            "suggested_section_id": "operations"
        },
        "response": {
            "id": "uuid",
            "sentence_id": "uuid",
            "review_type": "confusing",
            "status": "open",
            "created_at": "2026-03-29T10:00:00Z"
        }
    },
    
    # ===== REMAP ALL =====
    "remap_all": {
        "endpoint": "POST /projects/{project_id}/remap-all",
        "request": {
            "use_corrected_patterns": True,
            "target_confidence_threshold": 0.70
        },
        "response": {
            "remapped_count": 45,
            "unchanged_count": 35,
            "accuracy_improvement": 0.08,
            "changes_summary": {
                "deployment_steps": 15,
                "operations": 12,
                "troubleshooting": 18
            }
        }
    },
    
    # ===== COVERAGE REPORT =====
    "coverage_report": {
        "endpoint": "GET /projects/{project_id}/coverage",
        "response": {
            "total_sections": 8,
            "covered_sections": 5,
            "partial_sections": 2,
            "missing_sections": 1,
            "coverage_percentage": 62.5,
            "confusing_sentences_count": 8,
            "sections": [
                {
                    "section_id": "deployment_steps",
                    "status": "covered",
                    "mapped_sentence_count": 12,
                    "confidence_avg": 0.82
                }
            ],
            "missing_content_checklist": {
                "rollback_procedure": "Ask KT provider: How do we rollback a failed deployment?",
                "monitoring_setup": "Ask: How are we monitoring the deployed service?"
            }
        }
    },
    
    # ===== AI ANALYZE BATCH =====
    "analyze_batch": {
        "endpoint": "POST /ai/analyze-batch",
        "request": {
            "transcript_id": "uuid",
            "confidence_threshold": 0.65
        },
        "response": {
            "transcript_id": "uuid",
            "sentences_analyzed": 80,
            "high_confidence_count": 65,
            "low_confidence_count": 12,
            "confusing_count": 3,
            "analysis_results": [
                {
                    "sentence_id": "uuid",
                    "predicted_section_id": "deployment_steps",
                    "confidence_score": 0.87,
                    "alternative_sections": [
                        {"section_id": "operations", "score": 0.45},
                        {"section_id": "troubleshooting", "score": 0.32}
                    ],
                    "is_confusing": False,
                    "semantic_keywords": ["deploy", "kubectl", "container"]
                }
            ]
        }
    }
}

# =====================================================
# API DOCUMENTATION
# =====================================================

"""
ENTERPRISE KT PLANNER - API SUMMARY

1. AUTHENTICATION
   - JWT Bearer tokens required for all endpoints
   - Header: Authorization: Bearer {token}
   - Role-based access control (Admin, Editor, Reviewer, Viewer)

2. RATE LIMITING
   - 1000 requests/hour per user
   - 100 requests/minute per endpoint

3. PAGINATION
   - Default limit: 50
   - Max limit: 1000
   - Query params: ?skip=0&limit=50

4. REQUEST/RESPONSE FORMAT
   - Content-Type: application/json
   - All timestamps in ISO 8601 format
   - IDs are UUIDs

5. ERROR RESPONSES
   - 400: Bad Request
   - 401: Unauthorized
   - 403: Forbidden
   - 404: Not Found
   - 422: Validation Error
   - 500: Internal Server Error

6. WEBSOCKET EVENTS
   - sentence_updated: When sentence is edited/mapped
   - review_added: When review is created
   - mapping_changed: When mapping changes
   - project_status_changed: When project status changes

7. ASYNC OPERATIONS
   - Long-running ops return async job ID
   - Poll /jobs/{job_id} for status
   - Or subscribe to WebSocket for real-time updates

8. SEARCH & FILTERING
   - Full-text search on sentence content: /sentences/search
   - Filter by section: ?assigned_section={section_id}
   - Filter by status: ?status=unmapped,confusing
   - Filter by importance: ?is_important=true
"""
