# KT Planner: Coverage Issue Fix & Recovery Guide

## Problem Summary

**Issue**: After manually assigning sentences to sections, the coverage metrics weren't updating. Sections still showed as "missing" even though sentences were assigned.

**Root Cause**: The coverage was calculated once during initial processing and cached in the job. When manual assignments were made via `/manual-assign`, the `section_content` was updated, but the cached `coverage` metrics were **not recalculated**.

**Impact**: The UI showed stale coverage data, confusing users about the actual state of the KT document.

---

## Solution: Three-Part Fix

### 1. **Fixed `/manual-assign` Endpoint**
When you manually assign a sentence to a section, the system now:
- ✅ Updates `section_content` with the assignment
- ✅ **Immediately recalculates coverage metrics** using the new helper function
- ✅ Updates the cache in the job
- ✅ Returns the new coverage percentage

**Before**: Manual assignments didn't reflect in coverage.  
**After**: Coverage updates in real-time when you make assignments.

### 2. **New `recalculate_coverage_from_section_content()` Function**
This helper function:
- Takes current `section_content` as input
- Calculates coverage status for each section:
  - **"missing"**: 0 sentences
  - **"weak"**: 1 sentence
  - **"covered"**: 2+ sentences
- Returns updated coverage dict, missing required sections, and progress percentage

---

## How to Use the New Endpoints

### Option 1: Auto-Populate Empty Sections (Recommended First Step)

**Endpoint**: `POST /populate-section/{job_id}/{section_id}`

This endpoint automatically assigns unassigned sentences to a section based on keyword/hint matching.

**Example**:
```bash
curl -X POST http://localhost:8000/populate-section/my-job-id/system_overview
```

**Response**:
```json
{
  "status": "populated",
  "section_id": "system_overview",
  "section_title": "SYSTEM OVERVIEW",
  "sentences_assigned": 3,
  "matched_candidates": 5,
  "new_coverage_percent": 25,
  "message": "Assigned 3 sentences to SYSTEM OVERVIEW"
}
```

**How it works**:
1. Scans all unassigned sentences
2. Scores each sentence against the section's hints
3. Assigns top 5 scoring sentences to the section
4. Recalculates coverage automatically

---

### Option 2: Diagnose Why Sections Are Empty

**Endpoint**: `GET /diagnose/{job_id}`

Provides detailed diagnostics about coverage issues.

**Example**:
```bash
curl http://localhost:8000/diagnose/my-job-id
```

**Response** (excerpt):
```json
{
  "job_id": "my-job-id",
  "total_unassigned_sentences": 42,
  "unassigned_preview": [
    "The system handles payment processing...",
    "We deploy to production weekly...",
    "On Fridays, deployments are blocked..."
  ],
  "section_diagnostics": {
    "system_overview": {
      "status": "missing",
      "sentence_count": 0,
      "hints": ["system name", "what does this system do", "who uses it"],
      "recommendation": "This required section is EMPTY. Look for unassigned sentences matching: system name, what does this system do, who uses it"
    },
    "deployment_and_rollback": {
      "status": "weak",
      "sentence_count": 1,
      "recommendation": "This required section has only 1 sentence. Need at least 2 for 'covered' status."
    }
  }
}
```

**Use this to**:
- Understand why coverage is low
- See which sentences exist but aren't assigned
- Get recommendations for each empty section

---

### Option 3: Manually Rebuild Coverage (If Cache Seems Stuck)

**Endpoint**: `POST /rebuild-coverage/{job_id}`

Force a complete recalculation of coverage metrics from current state.

**Example**:
```bash
curl -X POST http://localhost:8000/rebuild-coverage/my-job-id
```

**Response**:
```json
{
  "status": "rebuilt",
  "job_id": "my-job-id",
  "new_coverage_percent": 45,
  "sections_covered": 6,
  "total_sections": 13,
  "missing_required_sections": ["system_overview", "deployment_and_rollback"],
  "timestamp": "2026-03-30T10:30:00.000000Z"
}
```

**When to use**:
- Coverage seems "stuck" or outdated
- After importing external data
- For debugging purposes

---

## Step-by-Step Recovery Workflow

### For Each Missing Required Section:

1. **Diagnose** what's available
   ```bash
   curl http://localhost:8000/diagnose/{job_id}
   ```

2. **Auto-populate** the section
   ```bash
   curl -X POST http://localhost:8000/populate-section/{job_id}/section-id
   ```

3. **Verify** coverage improved
   ```bash
   curl http://localhost:8000/coverage/{job_id}
   ```

4. **Check** if result is satisfactory
   - If "weak" (1 sentence) → respond with section-id  
   - If "missing" after auto-populate → use `/reviews` endpoint to manually assign

---

## Manual Assignment Workflow (When Auto-Populate Isn't Enough)

### Using Existing `/manual-assign` Endpoint (Now Fixed!)

1. **Get review items** that need assignment:
   ```bash
   curl http://localhost:8000/reviews/{job_id}
   ```

2. **Manually assign** a sentence to a section:
   ```bash
   curl -X POST "http://localhost:8000/manual-assign/{job_id}?sentence_text=The+system+processes+payments&target_section=system_overview"
   ```

3. **Coverage auto-updates** (now fixed!)

4. **Repeat** for other sections

---

## Coverage Calculation Rules

| Criteria | Status |
|----------|--------|
| 0 sentences assigned | **missing** |
| 1 sentence assigned | **weak** |
| 2+ sentences assigned | **covered** |

For a section to be marked "complete", it needs at least 2 content sentences.

---

## Troubleshooting

### Q: I assigned a sentence but coverage still shows "missing"

**A**: This was the original bug. **You must do one of:**
1. Hit the `/rebuild-coverage` endpoint to force recalculation
2. Upgrade to the latest version (bug now fixed in `/manual-assign`)

### Q: Auto-populate assigned 0 sentences

**A**: The unassigned sentences don't match the section's hints. Try:
1. Use `/diagnose` to see what unassigned sentences exist
2. Manually assign relevant sentences using `/manual-assign`
3. Or create new sentences that match the section's requirements

### Q: A section has only 1 sentence (weak status)

**A**: Add one more sentence:
1. Use `/reviews` to find unrelated sentences
2. Use `/manual-assign` to move one to this section
3. Or create/record additional content

### Q: Coverage shows 0% but I assigned all sections

**A**: Force rebuild:
```bash
curl -X POST http://localhost:8000/rebuild-coverage/{job_id}
```

---

## What Each Required Section Needs

| Section | Hints | Min Content |
|---------|-------|------------|
| **SYSTEM OVERVIEW** | system name, what it does, users, business impact | 2+ sentences describing the system's purpose and criticality |
| **Architecture Reference** | architecture documentation, last updated | 2+ sentences with architecture links/references |
| **DAY-1 SURVIVAL CHECKLIST** | required access, tools, first safe actions | 2+ sentences with access requirements and safe first steps |
| **DEPLOYMENT & ROLLBACK** | deployment process, steps, rollback procedure | 2+ sentences covering normal deploy and rollback processes |
| **COMMON FAILURES & FIXES** | common failures, how to fix, known issues | 2+ sentences with failure scenarios and resolutions |
| **KNOWN BAD DAYS/WINDOWS** | maintenance windows, blocked times, known issues | 2+ sentences about downtime windows or deployment restrictions |
| **DANGER ZONES** | must not touch, risky areas, do not execute | 2+ sentences warning about dangerous operations |
| **OWNERSHIP & ESCALATION** | who owns what, escalation procedure | 2+ sentences defining ownership and escalation paths |
| **FIRST 30-DAY PLAN** | 30-day goals, learning curve, what to focus on | 2+ sentences outlining the new owner's first month |
| **OPEN RESPONSIBILITIES** | transition items, incomplete items, handover | 2+ sentences about pending work or incomplete transitions |
| **HANDOVER COMPLETION CHECK** | verification steps, checklist items, sign-off | 2+ sentences with checklist or verification steps |
| **Sign-off** | date, person, approval | At least 1 sentence with approval information |

---

## API Summary

### New Endpoints (Added in This Fix)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/rebuild-coverage/{job_id}` | Force recalculate coverage metrics |
| GET | `/diagnose/{job_id}` | Diagnose why sections are empty |
| POST | `/populate-section/{job_id}/{section_id}` | Auto-assign unassigned sentences to section |

### Updated Endpoints

| Method | Endpoint | Change |
|--------|----------|--------|
| POST | `/manual-assign/{job_id}` | **Now recalculates coverage immediately** |

### Existing Endpoints (Still Available)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/coverage/{job_id}` | Get coverage analysis |
| GET | `/reviews/{job_id}` | Get unassigned sentences |
| POST | `/manual-assign/{job_id}` | Manual assignment (now with auto-recalc) |

---

## Implementation Details

### Code Changes

1. **Added `recalculate_coverage_from_section_content()` function** (line ~150)
   - Calculates coverage from current section_content
   - Handles both required and optional sections
   - Returns coverage dict, missing list, and progress %

2. **Updated `/manual-assign` endpoint** (line ~720)
   - Calls recalculate function after assignment
   - Updates job cache immediately
   - Returns new coverage percentage

3. **Added 3 new endpoints**:
   - `/rebuild-coverage/{job_id}` - Force recalc
   - `/diagnose/{job_id}` - Diagnostic info
   - `/populate-section/{job_id}/{section_id}` - Auto-populate

---

## Related Documentation

- See `DEVELOPERS_GUIDE.md` for architecture details
- See `output_generator.py` for export formats
- See `kt_schema_new.json` for section definitions
