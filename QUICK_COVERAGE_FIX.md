# Quick Reference: Coverage Recovery Commands

## 🚀 Quick Fix (Run These in Order)

```bash
# Step 1: Diagnose the issue
curl http://localhost:8000/diagnose/YOUR-JOB-ID

# Step 2: Auto-populate empty sections (run for each missing section)
curl -X POST http://localhost:8000/populate-section/YOUR-JOB-ID/system_overview
curl -X POST http://localhost:8000/populate-section/YOUR-JOB-ID/architecture_reference
curl -X POST http://localhost:8000/populate-section/YOUR-JOB-ID/day1_survival_checklist
curl -X POST http://localhost:8000/populate-section/YOUR-JOB-ID/deployment_and_rollback
curl -X POST http://localhost:8000/populate-section/YOUR-JOB-ID/common_failures
curl -X POST http://localhost:8000/populate-section/YOUR-JOB-ID/known_bad_days
curl -X POST http://localhost:8000/populate-section/YOUR-JOB-ID/danger_zones
curl -X POST http://localhost:8000/populate-section/YOUR-JOB-ID/ownership_escalation
curl -X POST http://localhost:8000/populate-section/YOUR-JOB-ID/first_30day_plan
curl -X POST http://localhost:8000/populate-section/YOUR-JOB-ID/open_responsibilities
curl -X POST http://localhost:8000/populate-section/YOUR-JOB-ID/handover_completion_check
curl -X POST http://localhost:8000/populate-section/YOUR-JOB-ID/sign_off

# Step 3: Check coverage
curl http://localhost:8000/coverage/YOUR-JOB-ID

# Step 4: If coverage still shows old data, force rebuild
curl -X POST http://localhost:8000/rebuild-coverage/YOUR-JOB-ID

# Step 5: Verify final coverage
curl http://localhost:8000/coverage/YOUR-JOB-ID
```

## 📝 Common Workflows

### Populate All Empty Sections at Once
```bash
JOB_ID="your-job-id"

for section in system_overview architecture_reference day1_survival_checklist deployment_and_rollback common_failures known_bad_days danger_zones ownership_escalation first_30day_plan open_responsibilities handover_completion_check sign_off; do
  echo "Populating $section..."
  curl -X POST http://localhost:8000/populate-section/$JOB_ID/$section -s | jq '.sentences_assigned'
done
```

### Check Coverage Progress
```bash
curl http://localhost:8000/coverage/YOUR-JOB-ID | jq '.coverage_by_section | to_entries[] | {key, status: .value.status, count: .value.sentence_count}'
```

### Get All Unassigned Sentences
```bash
curl http://localhost:8000/diagnose/YOUR-JOB-ID | jq '.unassigned_preview'
```

### Manually Assign a Specific Sentence
```bash
curl -X POST "http://localhost:8000/manual-assign/YOUR-JOB-ID?sentence_text=YOUR-SENTENCE-TEXT&target_section=TARGET-SECTION"
```

## ✅ Verification Checklist

- [ ] Run `/diagnose` to understand current state
- [ ] See total unassigned sentences
- [ ] Run `/populate-section` for each missing section
- [ ] Check if coverage improved to 100%
- [ ] Run `/rebuild-coverage` if cache seems stale
- [ ] Verify `/coverage` shows all sections as "covered" or "weak"
- [ ] Export final KT document

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Auto-populate assigns 0 sentences | Use `/diagnose` to see available unassigned sentences, then manually assign relevant ones |
| Coverage still shows old percentage | Run `/rebuild-coverage` endpoint |
| Section stuck as "weak" (1 sentence) | Use `/populate-section` again to add more, or manually assign via `/manual-assign` |
| Can't find sentences to assign | Check `/diagnose` → `unassigned_preview` to see available content |

## 📊 Expected Results

**Before Fix**:
- Manual assignments didn't update coverage
- Coverage appeared stuck
- ✗ 13 sections missing

**After Fix**:
- Coverage auto-updates with manual assignments
- `/populate-section` fills ~50-70% of sections instantly
- `/manual-assign` now includes coverage recalculation
- ✅ Coverage progresses to 100%
