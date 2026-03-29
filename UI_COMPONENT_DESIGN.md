"""
ENTERPRISE KT PLANNER - REACT COMPONENT STRUCTURE

Complete component hierarchy and state management design for enterprise UI.
"""

# =====================================================
# COMPONENT HIERARCHY
# =====================================================

APP_STRUCTURE = """
├── App (Root)
│   ├── AuthProvider (JWT Context)
│   ├── BrowserRouter
│   │   ├── ProjectList
│   │   │   ├── ProjectCard
│   │   │   ├── CreateProjectModal
│   │   │   └── ProjectFilterBar
│   │   │
│   │   ├── ProjectDetail
│   │   │   ├── Header (Project Name, Status, Actions)
│   │   │   ├── TabNavigation
│   │   │   │   ├── "Transcript" Tab
│   │   │   │   ├── "Coverage" Tab
│   │   │   │   ├── "Reviews" Tab
│   │   │   │   └── "Versions" Tab
│   │   │   │
│   │   │   ├── MainContentArea (Split View)
│   │   │   │   ├── LeftPanel (Transcript)
│   │   │   │   │   ├── TranscriptUploadZone
│   │   │   │   │   ├── SentenceList
│   │   │   │   │   │   ├── SentenceCard
│   │   │   │   │   │   │   ├── SentenceText
│   │   │   │   │   │   │   ├── ConfidenceBadge
│   │   │   │   │   │   │   ├── StatusIndicator (Green/Yellow/Red)
│   │   │   │   │   │   │   ├── InlineActionBar
│   │   │   │   │   │   │   │   ├── EditButton
│   │   │   │   │   │   │   │   ├── AssignButton
│   │   │   │   │   │   │   │   ├── MarkConfusingButton
│   │   │   │   │   │   │   │   ├── ImportanceToggle
│   │   │   │   │   │   │   │   └── MoreActionsMenu
│   │   │   │   │   │   │   └── SentenceDetailDrawer
│   │   │   │   │   │   │       ├── EditForm
│   │   │   │   │   │   │       ├── CodeReferencePanel
│   │   │   │   │   │   │       ├── ReviewsPanel
│   │   │   │   │   │   │       └── MappingHistory
│   │   │   │   │   │   └── [Virtualized List for Performance]
│   │   │   │   │   │
│   │   │   │   │   ├── SearchBar
│   │   │   │   │   └── FilterPanel
│   │   │   │   │       ├── FilterByStatus (mapped/unmapped/confusing)
│   │   │   │   │       ├── FilterBySection
│   │   │   │   │       ├── FilterByConfidence
│   │   │   │   │       └── FilterByImportance
│   │   │   │   │
│   │   │   │   └── RightPanel (KT Sections)
│   │   │   │       ├── SectionDropZone (DnD Container)
│   │   │   │       │   └── SectionColumn (for each section)
│   │   │   │       │       ├── SectionHeader
│   │   │   │       │       │   ├── SectionTitle
│   │   │   │       │       │   ├── MappedCount / TotalExpected
│   │   │   │       │       │   └── CoverageBar
│   │   │   │       │       ├── DroppableArea
│   │   │   │       │       │   └── MappedSentenceCard (Draggable)
│   │   │   │       │       │       ├── SentenceContent
│   │   │   │       │       │       ├── ConfidenceScore
│   │   │   │       │       │       ├── RemoveButton
│   │   │   │       │       │       └── EditButton
│   │   │   │       │       └── SectionFooter
│   │   │   │       │           ├── AddNotesButton
│   │   │   │       │           ├── ViewCodeRefsButton
│   │   │   │       │           └── CollapseButton
│   │   │   │       │
│   │   │   │       └── SpecialSection (Clarification Required)
│   │   │   │           └── ConfusingSentencesList
│   │   │   │
│   │   │   └── ControlPanel
│   │   │       ├── BulkActionsBar
│   │   │       │   ├── BulkMapButton
│   │   │       │   ├── BulkDeleteButton
│   │   │       │   └── SelectAllCheckbox
│   │   │       ├── ActionButtons
│   │   │       │   ├── RemapAllButton
│   │   │       │   ├── ExportButton
│   │   │       │   ├── VersionsButton
│   │   │       │   └── SettingsButton
│   │   │       └── ProgressIndicators
│   │   │           ├── OverallCoverageGauge
│   │   │           ├── ConfusingSentencesCount
│   │   │           └── LastUpdatedTime
│   │   │
│   │   ├── CoverageReport
│   │   │   ├── SectionCoverageTable
│   │   │   ├── MissingContentChecklist
│   │   │   ├── CoverageChart (Pie/Bar)
│   │   │   └── AutoSuggestions
│   │   │
│   │   ├── ReviewPanel
│   │   │   ├── ConfusingSentencesList
│   │   │   ├── ClarificationRequiredSection
│   │   │   └── ReviewActionsPanels
│   │   │
│   │   ├── VersionHistory
│   │   │   ├── VersionTimeline
│   │   │   ├── VersionComparisonView
│   │   │   └── RestoreButtons
│   │   │
│   │   └── SettingsPanel
│       ├── ProjectSettings
│       ├── ConfidenceThresholdSlider
│       └── ExportSettings
│
└── Modals & Overlays
    ├── EditSentenceModal
    ├── AssignToSectionModal (with AI suggestions)
    ├── MergeConflictModal
    ├── CodeReferenceModal
    └── ConfirmActionModal
"""

# =====================================================
# STATE MANAGEMENT (ZUSTAND)
# =====================================================

ZUSTAND_STORE_STRUCTURE = """
projectStore = {
  // Project State
  currentProject: {
    id, name, status, confidence_threshold, auto_map_enabled
  },
  projects: [],
  
  // Transcript State
  currentTranscript: {
    id, raw_text, status, duration_seconds, file_name
  },
  
  // Sentence State
  sentences: [
    {
      id, text, predicted_section, confidence_score,
      assigned_section, is_confusing, is_important, status
    }
  ],
  selectedSentenceIds: Set<string>,
  
  // Filter & Search
  filters: {
    search: string,
    status: string[],  // ['mapped', 'unmapped', 'confusing']
    section: string,
    confidence_min: float,
    importance_only: boolean
  },
  
  // UI State
  uiState: {
    leftPanelWidth: number,
    rightPanelWidth: number,
    activeTab: 'transcript' | 'coverage' | 'reviews' | 'versions',
    selectedSentenceDetail: string | null,
    showCodePanel: boolean,
    showFilters: boolean,
    isDragging: boolean
  },
  
  // Editing State
  editingState: {
    sentenceBeingEdited: string | null,
    editedText: string,
    isDirty: boolean
  },
  
  // Async Operations
  asyncJobs: {
    transcriptionJobId: string | null,
    remapAllJobId: string | null,
    exportJobId: string | null
  },
  
  // Cache
  sectionCache: Map<section_id, SectionData>,
  codeReferencesCache: Map<sentence_id, CodeRef[]>,
  reviewsCache: Map<sentence_id, Review[]>,
  
  // Actions
  actions: {
    setCurrentProject, updateSentence, assignToSection,
    bulkMap, remapAll, markConfusing, addCodeReference,
    setFilter, setSelectedSentences, updateUIState,
    // ... many more
  }
}
"""

# =====================================================
# KEY COMPONENTS - DETAILED SPECS
# =====================================================

COMPONENT_SPECS = {
    
    "SentenceCard": {
        "props": {
            "sentence": "Sentence object",
            "isDragging": "boolean",
            "isSelected": "boolean",
            "onSelect": "function",
            "onEdit": "function",
            "onAssign": "function",
            "onMarkConfusing": "function"
        },
        "visual_design": """
        ┌─────────────────────────────────────────────┐
        │ [Checkbox] Sentence text here...            │
        │                                             │
        │ ┌─ ┐  [Confidence: 0.82]  [Important ☆]    │
        │ │G├─ Assigned: deployment_steps           │
        │ └─ ┘  [Edit] [Assign] [Mark Confusing]     │
        └─────────────────────────────────────────────┘
        
        Color coded border:
        - Green: High confidence
        - Yellow: Low confidence
        - Red: Confusing / marked for review
        """,
        "features": [
            "Drag-enabled",
            "Click to expand details",
            "Inline action buttons",
            "Color-coded confidence",
            "Edit in-place or in modal",
            "Show mapping history on hover"
        ]
    },
    
    "SectionColumn": {
        "props": {
            "sectionId": "string",
            "sectionTitle": "string",
            "sentences": "Sentence[]",
            "isDropTarget": "boolean",
            "onDrop": "function"
        },
        "visual_design": """
        ┌─────────────────────────────────────┐
        │ Deployment Steps        [10/12]  ██ │
        ├─────────────────────────────────────┤
        │                                     │
        │  ☐ Sentence 1 [0.89] [Edit] [x]   │
        │                                     │
        │  ☐ Sentence 2 [0.76] [Edit] [x]   │
        │                                     │
        │  🟨 Sentence 3 (low conf) [?]     │
        │                                     │
        │  [+ Add Note] [Code Refs] [-]      │
        └─────────────────────────────────────┘
        """,
        "features": [
            "Drop zone for DnD",
            "Coverage progress bar",
            "Collapsible content",
            "Add notes to section",
            "Link code references",
            "Visual feedback on hover"
        ]
    },
    
    "InlineActionBar": {
        "visual_design": """
        [Edit] [Assign ▼] [★] [?] [⋯ More]
        
        More menu contains:
        - Mark Important
        - Mark Confusing
        - Add Code Reference
        - Add Comment
        - Move to Section...
        - Delete
        - View History
        """,
        "features": [
            "Edit sentence text",
            "Quick assign to section",
            "Toggle importance",
            "Mark as confusing",
            "More actions dropdown"
        ]
    },
    
    "CodeReferencePanel": {
        "visual_design": """
        CODE REFERENCES
        ┌─────────────────────────────────┐
        │ [+ Add Reference] [Filter]      │
        ├─────────────────────────────────┤
        │                                 │
        │ 📄 deployment.yaml              │
        │    kubectl apply -f ...         │
        │    [Bash] [Show Full] [x]       │
        │                                 │
        │ 📄 config.json                  │
        │    { "key": "value" }           │
        │    [JSON] [Show Full] [x]       │
        │                                 │
        └─────────────────────────────────┘
        """,
        "features": [
            "Add code snippet modal",
            "Display code with syntax highlighting",
            "Link code to multiple sections",
            "Collapsible code blocks",
            "Copy to clipboard button",
            "Language tag display"
        ]
    },
    
    "ConfusingSentencesPanel": {
        "visual_design": """
        CLARIFICATION REQUIRED (8 items)
        ┌─────────────────────────────────┐
        │ Sentence: \"unclear wording...\"   │
        │ Suggested Section: ?             │
        │ [Mark as Clarified] [Reassign]  │
        │ ─────────────────────────────────│
        │ Sentence: \"ambiguous term...\"    │
        │ [Add Comment] [Link Code]       │
        │ ─────────────────────────────────│
        │ ...                             │
        └─────────────────────────────────┘
        """,
        "features": [
            "List all confusing sentences",
            "Quick reassign",
            "Add clarification comment",
            "Link to code for context",
            "Mark as resolved",
            "Track clarification status"
        ]
    },
    
    "CoverageReport": {
        "visual_design": """
        COVERAGE ANALYSIS
        ┌─────────────────────────────────────────┐
        │ Overall Coverage: 62.5%  ███████░░░     │
        ├─────────────────────────────────────────┤
        │ Section                  Status  %      │
        │ ─────────────────────────────────────   │
        │ Deployment Steps        ✓ 12  100%    │
        │ Operations              ⊘ 5    50%    │
        │ Troubleshooting         ⚠ 3    30%    │
        │ Rollback Procedure      ✗ 0     0%    │
        ├─────────────────────────────────────────┤
        │ MISSING CONTENT CHECKLIST               │
        │ □ Ask about rollback steps              │
        │ □ Document monitoring setup             │
        │ □ Add load balancing config             │
        └─────────────────────────────────────────┘
        """,
        "features": [
            "Overall coverage percentage",
            "Per-section breakdown",
            "Auto-generated missing content checklist",
            "AI suggestions for missing items",
            "Export report as PDF/Markdown"
        ]
    }
}

# =====================================================
# DRAG & DROP IMPLEMENTATION (dnd-kit)
# =====================================================

DRAG_DROP_SPEC = """
DRAG & DROP FLOW

1. Source: SentenceCard (Draggable)
   - Can be dragged from left panel
   - Shows dragging ghost image
   - Shows "Assign to Section" hint

2. Drop Target: SectionColumn (Droppable)
   - Highlights on hover
   - Shows drop zone outline
   - Displays "Drop to assign" text

3. Drop Action:
   - Update sentence.assigned_section_id
   - Update mappings table
   - Move card to right panel
   - Trigger "remap suggestion" if multiple corrections

4. Undo/Redo:
   - Track in version history
   - User can undo last action
   - Version comparison shows before/after

5. Bulk Operations:
   - Select multiple sentences
   - Drag as batch to section
   - All get mapped together
   - Show batch operation toast
"""

# =====================================================
# WEBSOCKET INTEGRATION
# =====================================================

WEBSOCKET_SPEC = """
REAL-TIME UPDATES

Connection: ws://localhost:8000/ws/projects/{project_id}

Events:
1. SENTENCE_UPDATED
   - When user edits a sentence
   - When mapping changes
   - Broadcast to all connected clients
   
2. REVIEW_ADDED
   - When new review/annotation added
   - Show notification toast
   - Update reviews panel

3. MAPPING_CHANGED
   - Remap all in progress
   - Show progress indicator
   - Update UI when complete

4. PROJECT_STATUS_CHANGED
   - Project archived/completed
   - Show toast notification

Implementation:
- Zustand store subscribes to WebSocket
- On message, update local state
- Reconcile if offline changes conflict
"""

# =====================================================
# COLOR SCHEME
# =====================================================

COLOR_SCHEME = {
    "status": {
        "covered": "#10b981",        # Green
        "partial": "#f59e0b",        # Amber
        "missing": "#ef4444",        # Red
        "uncertain": "#eab308",      # Yellow
        "confusing": "#dc2626",      # Dark Red
        "code_linked": "#3b82f6",    # Blue
    },
    "confidence": {
        "high": "#10b981",           # Green (>0.75)
        "medium": "#f59e0b",         # Amber (0.5-0.75)
        "low": "#ef4444",            # Red (<0.5)
    },
    "ui": {
        "primary": "#2563eb",        # Blue
        "secondary": "#64748b",      # Slate
        "success": "#10b981",        # Green
        "danger": "#ef4444",         # Red
        "warning": "#f59e0b",        # Amber
        "hover_bg": "#f8fafc",       # Very light slate
        "border": "#e2e8f0",         # Light slate
    }
}

# =====================================================
# KEYBOARD SHORTCUTS
# =====================================================

KEYBOARD_SHORTCUTS = {
    "Ctrl+E": "Edit selected sentence",
    "Ctrl+A": "Assign selected to section",
    "Ctrl+Shift+C": "Mark as confusing",
    "Ctrl+S": "Save changes",
    "Ctrl+Z": "Undo",
    "Ctrl+Shift+Z": "Redo",
    "Ctrl+F": "Search sentences",
    "Ctrl+R": "Remap all",
    "Ctrl+P": "Quick project switcher",
    "Ctrl+?": "Show shortcuts help",
}
"""

# =====================================================
# RESPONSE TIMES & PERFORMANCE TARGETS
# =====================================================

PERFORMANCE_TARGETS = {
    "sentence_editing": "< 200ms (local state update)",
    "drag_drop": "< 50ms (smooth 60fps)",
    "search": "< 500ms (with debounce)",
    "bulk_mapping": "< 2s (100 sentences)",
    "remap_all": "< 10s (500 sentences)",
    "page_load": "< 1s (with code splitting)",
    "websocket_latency": "< 100ms (real-time updates)",
}

# =====================================================
# ACCESSIBILITY (A11Y)
# =====================================================

A11Y_REQUIREMENTS = {
    "keyboard_navigation": "Full keyboard support with Tab, Arrow keys",
    "screen_reader": "ARIA labels for all interactive elements",
    "color_contrast": "WCAG AA compliant (4.5:1 for text)",
    "focus_indicators": "Visible focus ring on all focusable elements",
    "semantic_html": "Proper heading hierarchy, landmarks, regions",
    "motion": "Respect prefers-reduced-motion setting",
}
