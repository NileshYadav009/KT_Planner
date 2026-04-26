# ⚡ Quick Fix Commands - Copy & Paste

## One-Command Fix

Replace `YOUR-JOB-ID` with your actual job ID and run:

```bash
curl -X POST http://localhost:8000/reclassify/YOUR-JOB-ID
```

That's it! This will:
1. Use the AI 3-level word matching algorithm from ai.py
2. Re-classify your entire transcript
3. Populate sections with matched content
4. Recalculate coverage automatically
5. Return detailed results showing what was classified

---

## Get Your Job ID

```bash
# List all recent jobs
curl http://localhost:8000/jobs | jq '.jobs[-1]'

# Extract job ID from the list
curl http://localhost:8000/jobs | jq '.jobs[-1].job_id'
```

---

## Complete Workflow

```bash
# Step 1: Get job ID  
JOB_ID=$(curl -s http://localhost:8000/jobs | jq -r '.jobs[-1].job_id')
echo "Job ID: $JOB_ID"

# Step 2: Check current state (should show all "missing")
echo "BEFORE reclassification:"
curl -s http://localhost:8000/coverage/$JOB_ID | jq '.coverage_by_section | to_entries[] | {section: .key, status: .value.status, count: .value.sentence_count}' | head -20

# Step 3: Run the AI word matching re-classification ⭐
echo "Running AI Word Matching..."
curl -X POST -s http://localhost:8000/reclassify/$JOB_ID | jq '.'

# Step 4: Check new coverage
echo "AFTER reclassification:"  
curl -s http://localhost:8000/coverage/$JOB_ID | jq '.coverage_by_section | to_entries[] | {section: .key, status: .value.status, count: .value.sentence_count}' | head -20

# Step 5: See coverage percentage
echo "Coverage %:"
curl -s http://localhost:8000/coverage/$JOB_ID | jq '.overall_coverage_percent'
```

---

## What Each Command Does

### `/reclassify/{job_id}` [⭐ THE KEY ENDPOINT]

**What it does:**
- Uses the powerful `classify_transcript()` from ai.py
- 3-level hint matching (exact → token → partial)
- Re-populates ALL sections with classified content
- Recalculates coverage automatically

**Request:**
```bash
POST /reclassify/YOUR-JOB-ID
```

**Response:**
```json
{
  "status": "reclassified",
  "new_coverage_percent": 85,
  "sections_now_covered": 11,
  "total_chunks_classified": 47,
  "message": "✅ Re-classification COMPLETE! Coverage improved to 85%..."
}
```

---

### `/coverage/{job_id}` [Check Coverage]

```bash
# Get overall coverage percentage
curl http://localhost:8000/coverage/YOUR-JOB-ID | jq '.overall_coverage_percent'

# See all sections
curl http://localhost:8000/coverage/YOUR-JOB-ID | jq '.coverage_by_section'

# See missing required sections
curl http://localhost:8000/coverage/YOUR-JOB-ID | jq '.missing_required_sections'
```

---

### `/diagnose/{job_id}` [Debug Issues]

```bash
# See what sentences exist vs aren't classified
curl http://localhost:8000/diagnose/YOUR-JOB-ID | jq '.'

# See unassigned sentences (if any remain)
curl http://localhost:8000/diagnose/YOUR-JOB-ID | jq '.unassigned_preview'
```

---

### `/manual-assign/{job_id}` [Fine-tune Manually]

```bash
# Move a sentence to a specific section
curl -X POST "http://localhost:8000/manual-assign/YOUR-JOB-ID?sentence_text=Your+sentence+here&target_section=system_overview"
```

---

### `/rebuild-coverage/{job_id}` [Force Refresh]

```bash
# Force recalculation of coverage metrics (if cache seems stuck)
curl -X POST http://localhost:8000/rebuild-coverage/YOUR-JOB-ID
```

---

## Expected Results

### Before (Without re-classification):
```
Coverage: 0%
✗ SYSTEM OVERVIEW (0 sentences)
✗ DEPLOYMENT & ROLLBACK (0 sentences)
✗ COMMON FAILURES (0 sentences)
✗ ... all 13 sections empty
```

### After (/reclassify):
```
Coverage: 85%
✅ SYSTEM OVERVIEW (5 sentences)
✅ DEPLOYMENT & ROLLBACK (4 sentences)  
✅ COMMON FAILURES (3 sentences)
⚠ KNOWN BAD DAYS (1 sentence) → needs 1 more
✗ SIGN-OFF (0 sentences) → still empty
```

---

## For PowerShell Users

If you need to run this in PowerShell:

```powershell
# Get job ID
$JOB_ID = (Invoke-WebRequest -Uri "http://localhost:8000/jobs" | ConvertFrom-Json).jobs[-1].job_id

# Run re-classification
$result = Invoke-WebRequest -Method POST -Uri "http://localhost:8000/reclassify/$JOB_ID" | ConvertFrom-Json

# Show results
$result | ConvertTo-Json -Depth 10

# Check coverage
(Invoke-WebRequest -Uri "http://localhost:8000/coverage/$JOB_ID" | ConvertFrom-Json).overall_coverage_percent
```

---

## Python Script Version

```python
import requests
import json

# Configuration
API_BASE = "http://localhost:8000"
JOB_ID = "YOUR-JOB-ID"

def reclassify():
    """Run AI word matching to fix coverage"""
    response = requests.post(f"{API_BASE}/reclassify/{JOB_ID}")
    result = response.json()
    print(json.dumps(result, indent=2))
    return result

def check_coverage():
    """Check current coverage"""
    response = requests.get(f"{API_BASE}/coverage/{JOB_ID}")
    data = response.json()
    print(f"Coverage: {data['overall_coverage_percent']}%")
    return data

if __name__ == "__main__":
    print("Starting AI Word Matching Re-classification...")
    result = reclassify()
    print(f"\n✅ Coverage improved to {result['new_coverage_percent']}%")
    
    print("\nVerifying...")
    check_coverage()
```

---

## Troubleshooting

### Getting 404 errors?
```bash
# Make sure the API is running
curl http://localhost:8000/jobs

# If nothing returns, start uvicorn:
cd c:\Users\dell\Continumm\KT_Planner
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Job ID not found?
```bash
# List all jobs
curl http://localhost:8000/jobs | jq '.jobs | length'

# If empty, upload a new file first
```

### Coverage still not improved?
```bash
# Check what sentences exist
curl http://localhost:8000/diagnose/YOUR-JOB-ID | jq '.unassigned_preview'

# Manually assign them
curl -X POST "http://localhost:8000/manual-assign/YOUR-JOB-ID?sentence_text=TEXT&target_section=SECTION"
```

---

## Summary

| Action | Command |
|--------|---------|
| **Fix Coverage** ⭐ | `curl -X POST http://localhost:8000/reclassify/JOB_ID` |
| Check Coverage | `curl http://localhost:8000/coverage/JOB_ID` |
| Debug Issues | `curl http://localhost:8000/diagnose/JOB_ID` |
| Manually Assign | `curl -X POST "http://localhost:8000/manual-assign/JOB_ID?..."` |
| Force Refresh | `curl -X POST http://localhost:8000/rebuild-coverage/JOB_ID` |

**Most Important**: The `/reclassify` endpoint uses the intelligent word-matching algorithm from ai.py that was designed to handle exactly your problem!
