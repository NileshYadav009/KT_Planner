"""
ENTERPRISE KT PLANNER - DATABASE SCHEMA
PostgreSQL with SQLAlchemy ORM

This schema supports:
- Sentence-level mapping
- Version control
- Code linking
- User corrections / pattern learning
- Confusion detection
- Full audit trail
"""

-- =====================================================
-- PROJECTS & TRANSCRIPTS
-- =====================================================

CREATE TABLE kt_projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    schema_id VARCHAR(100) NOT NULL,  -- Reference to kt_schema_new.json sections
    
    created_by UUID NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Status: draft, in_progress, completed, archived
    status VARCHAR(50) DEFAULT 'draft',
    
    -- Settings
    confidence_threshold FLOAT DEFAULT 0.65,
    auto_map_enabled BOOLEAN DEFAULT true,
    
    INDEX idx_created_by (created_by),
    INDEX idx_created_at (created_at)
);

CREATE TABLE transcripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES kt_projects(id) ON DELETE CASCADE,
    
    -- Raw data
    raw_text TEXT NOT NULL,
    file_name VARCHAR(255),
    file_path VARCHAR(500),
    duration_seconds INT,
    
    -- Processing status
    status VARCHAR(50) DEFAULT 'raw',  -- raw, processing, segmented, mapped, complete
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (id),
    FOREIGN KEY (project_id) REFERENCES kt_projects(id) ON DELETE CASCADE,
    INDEX idx_project_id (project_id),
    INDEX idx_status (status)
);

-- =====================================================
-- SENTENCES - Core Unit at Sentence Level
-- =====================================================

CREATE TABLE sentences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transcript_id UUID NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    
    -- Text Content
    text TEXT NOT NULL,
    text_edited TEXT,  -- Edited version if user modifies
    
    -- Position / Sequencing
    sentence_index INT NOT NULL,  -- Order in transcript
    start_time_ms BIGINT,  -- Start timestamp in original audio
    end_time_ms BIGINT,    -- End timestamp in original audio
    
    -- AI Classification
    predicted_section_id VARCHAR(100),  -- Which KT section it belongs to
    confidence_score FLOAT DEFAULT 0.0,  -- 0.0 - 1.0 confidence
    
    -- User Assignment (Overrides AI prediction)
    assigned_section_id VARCHAR(100),
    assigned_by UUID,
    assigned_at TIMESTAMP,
    
    -- Flags & Metadata
    importance_weight FLOAT DEFAULT 1.0,  -- 1.0 = normal, >1.0 = important
    is_important BOOLEAN DEFAULT false,
    is_confusing BOOLEAN DEFAULT false,   -- Flagged for review
    needs_clarification BOOLEAN DEFAULT false,
    
    -- Status
    status VARCHAR(50) DEFAULT 'unmapped',  -- unmapped, mapped, conflicting, marked_for_review
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (id),
    FOREIGN KEY (transcript_id) REFERENCES transcripts(id) ON DELETE CASCADE,
    INDEX idx_transcript_id (transcript_id),
    INDEX idx_sentence_index (sentence_index),
    INDEX idx_predicted_section (predicted_section_id),
    INDEX idx_assigned_section (assigned_section_id),
    INDEX idx_is_confusing (is_confusing),
    INDEX idx_status (status)
);

-- =====================================================
-- SENTENCE EMBEDDINGS - For Semantic Search & Similarity
-- =====================================================

CREATE TABLE sentence_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sentence_id UUID NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    
    -- Embedding vector (1024 dims for SentenceTransformer)
    embedding VECTOR(1024) NOT NULL,
    
    -- Metadata
    model_name VARCHAR(255) DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (id),
    FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE CASCADE,
    UNIQUE(sentence_id),
    INDEX idx_sentence_id (sentence_id)
);

-- =====================================================
-- MAPPINGS - Sentence to Section Assignment
-- =====================================================

CREATE TABLE mappings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sentence_id UUID NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    
    -- Section mapping
    section_id VARCHAR(100) NOT NULL,
    confidence FLOAT DEFAULT 0.0,
    
    -- Assignment type
    assignment_type VARCHAR(50),  -- 'auto', 'manual', 'corrected'
    assigned_by UUID,
    
    -- Status
    status VARCHAR(50) DEFAULT 'active',  -- active, superseded, archived
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (id),
    FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE CASCADE,
    INDEX idx_sentence_id (sentence_id),
    INDEX idx_section_id (section_id),
    INDEX idx_assignment_type (assignment_type),
    INDEX idx_status (status)
);

-- =====================================================
-- CODE REFERENCES - Link Code/Configs to Sentences or Sections
-- =====================================================

CREATE TABLE code_references (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Link to sentence or section
    sentence_id UUID REFERENCES sentences(id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES kt_projects(id) ON DELETE CASCADE,
    
    -- Code Content
    code_snippet TEXT NOT NULL,
    language VARCHAR(50),  -- 'python', 'yaml', 'json', 'bash', etc.
    file_name VARCHAR(255),
    file_path VARCHAR(500),
    
    -- Metadata
    description TEXT,
    section_id VARCHAR(100),  -- Which KT section this relates to
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID,
    
    PRIMARY KEY (id),
    FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE SET NULL,
    FOREIGN KEY (project_id) REFERENCES kt_projects(id) ON DELETE CASCADE,
    INDEX idx_project_id (project_id),
    INDEX idx_sentence_id (sentence_id),
    INDEX idx_section_id (section_id)
);

-- =====================================================
-- VERSION CONTROL - Track All Changes
-- =====================================================

CREATE TABLE versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES kt_projects(id) ON DELETE CASCADE,
    
    -- Version metadata
    version_number INT NOT NULL,
    version_name VARCHAR(255),
    
    -- Full snapshot
    transcript_snapshot JSONB,  -- Full transcript state
    mappings_snapshot JSONB,    -- All mappings at this version
    
    -- Change info
    change_type VARCHAR(50),  -- 'created', 'edited', 'merged', 'remapped'
    changed_by UUID,
    change_description TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (id),
    FOREIGN KEY (project_id) REFERENCES kt_projects(id) ON DELETE CASCADE,
    INDEX idx_project_id (project_id),
    INDEX idx_version_number (version_number)
);

-- =====================================================
-- REVIEWS / ANNOTATIONS - Mark Confusing Sections
-- =====================================================

CREATE TABLE reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sentence_id UUID NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    
    -- Review type
    review_type VARCHAR(50),  -- 'confusing', 'needs_clarification', 'marked_important', 'comment'
    
    -- Content
    comment TEXT,
    suggested_section_id VARCHAR(100),  -- Reviewer's suggestion
    
    -- Metadata
    created_by UUID,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Status
    status VARCHAR(50) DEFAULT 'open',  -- open, resolved, acknowledged
    
    PRIMARY KEY (id),
    FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE CASCADE,
    INDEX idx_sentence_id (sentence_id),
    INDEX idx_review_type (review_type),
    INDEX idx_status (status)
);

-- =====================================================
-- USER CORRECTIONS - For Pattern Learning
-- =====================================================

CREATE TABLE user_corrections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES kt_projects(id) ON DELETE CASCADE,
    sentence_id UUID REFERENCES sentences(id) ON DELETE CASCADE,
    
    -- Original vs Corrected
    original_section_id VARCHAR(100),
    corrected_section_id VARCHAR(100),
    
    -- Context for pattern learning
    sentence_text TEXT,
    original_confidence FLOAT,
    
    -- Learning metadata
    user_id UUID,
    correction_count INT DEFAULT 1,  -- How many times this pattern has been corrected
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (id),
    FOREIGN KEY (project_id) REFERENCES kt_projects(id) ON DELETE CASCADE,
    FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE SET NULL,
    INDEX idx_project_id (project_id),
    INDEX idx_original_section (original_section_id),
    INDEX idx_corrected_section (corrected_section_id),
    INDEX idx_correction_count (correction_count)
);

-- =====================================================
-- MISSING CONTENT DETECTOR - Generate Suggestions
-- =====================================================

CREATE TABLE missing_content_suggestions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES kt_projects(id) ON DELETE CASCADE,
    
    -- What's missing
    section_id VARCHAR(100) NOT NULL,
    section_title VARCHAR(255),
    
    -- Suggestion
    suggested_question TEXT,
    severity VARCHAR(50),  -- 'critical', 'high', 'medium', 'low'
    
    -- Status
    status VARCHAR(50) DEFAULT 'unresolved',  -- unresolved, addressed, ignored
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    
    PRIMARY KEY (id),
    FOREIGN KEY (project_id) REFERENCES kt_projects(id) ON DELETE CASCADE,
    INDEX idx_project_id (project_id),
    INDEX idx_section_id (section_id),
    INDEX idx_status (status)
);

-- =====================================================
-- USERS & PERMISSIONS
-- =====================================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    
    password_hash VARCHAR(500),  -- bcrypt hashed
    role VARCHAR(50) DEFAULT 'editor',  -- 'admin', 'editor', 'reviewer', 'viewer'
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    
    INDEX idx_email (email)
);

-- =====================================================
-- AUDIT LOG - Track All Changes
-- =====================================================

CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- What changed
    entity_type VARCHAR(100),  -- 'sentence', 'mapping', 'review', etc.
    entity_id UUID,
    action VARCHAR(50),  -- 'created', 'updated', 'deleted'
    
    -- Who and when
    user_id UUID REFERENCES users(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Details
    old_values JSONB,
    new_values JSONB,
    
    INDEX idx_entity_type (entity_type),
    INDEX idx_user_id (user_id),
    INDEX idx_timestamp (timestamp)
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Composite indexes for common queries
CREATE INDEX idx_sentences_transcript_status ON sentences(transcript_id, status);
CREATE INDEX idx_sentences_section_status ON sentences(assigned_section_id, status);
CREATE INDEX idx_mappings_section_confidence ON mappings(section_id, confidence);
CREATE INDEX idx_reviews_sentence_type ON reviews(sentence_id, review_type);

-- Full-text search index for sentence content
CREATE INDEX idx_sentences_text_fts ON sentences USING GIN(to_tsvector('english', text));
