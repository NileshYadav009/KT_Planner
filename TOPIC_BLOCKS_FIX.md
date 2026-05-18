# Topic Blocks: Coverage Extraction Fix

## Problem
After implementing topic block storage, sentences were not appearing in the coverage details. The UI was showing:

```
"No explicit sentence text is available for this section; coverage is inferred from semantic/contextual analysis."
```

## Root Cause
The API layer (`main.py`) was trying to extract sentences from `cov.sentences`, but the new topic block architecture stores sentences in `cov.blocks` instead.

```python
# OLD (broken) - tried to access non-existent .sentences attribute
for s in getattr(cov, 'sentences', []) or []:
    coverage_sentences.append(...)

# NEW (fixed) - extracts sentences from blocks
blocks = getattr(cov, 'blocks', []) or []
for block in blocks:
    for s in block.sentences:
        coverage_sentences.append(...)
```

## Solution
Updated `main.py` lines 213-238 to:

1. **Extract sentences from blocks** - Iterate through `cov.blocks` and flatten sentences
2. **Preserve ordering** - Blocks maintain consecutive sentence order
3. **Include blocks in API response** - Frontend receives both sentences AND blocks
4. **Backwards compatibility** - Fallback to old `.sentences` attribute if blocks unavailable

## Impact

### Before (Broken)
```
System Architecture (covered)
  No explicit sentence text is available for this section; coverage is inferred 
  from semantic/contextual analysis.
```

### After (Fixed)
```
System Architecture (weak)
  Blocks: 1 | Sentences: 1 | Confidence: 0.00
  
  Content:
    1. The API layer handles all incoming requests.
```

## Validation

Test Results from `test_coverage_extraction.py`:
- ✓ All sections extracted
- ✓ All sentences extracted from blocks  
- ✓ Content field populated for all non-missing sections
- ✓ Proper paragraph continuity maintained

Example output:

**Deployment Process** (covered)
```json
{
  "title": "Deployment Process",
  "status": "covered",
  "sentence_count": 4,
  "block_count": 1,
  "content": [
    "We route them to the appropriate microservices.",
    "For deployment, we use Kubernetes with continuous integration.",
    "The rollout process takes about twenty minutes.",
    "We have automated health checks that verify the deployment."
  ],
  "blocks": [
    {
      "section_id": "deployment",
      "topic_title": null,
      "sentence_count": 4,
      "start_time": 10.0,
      "end_time": 30.0,
      "confidence": 0.0,
      "sentences": [...]
    }
  ]
}
```

## Why This Works

1. **Blocks preserve structure**: Consecutive sentences remain grouped
2. **Sentences extracted for frontend**: API flattens blocks into sentences for UI display
3. **Both accessible**: Frontend gets blocks (for smart display) and sentences (for legacy support)
4. **Maintains continuity**: Order from blocks is preserved in flattened sentence list

## Files Modified
- `context_mapper.py`: Added TopicBlock dataclass and updated detect_gaps(), assemble_kt()
- `main.py`: Updated coverage extraction to use blocks (lines 213-238)

## Integration Complete
- ✅ All tests passing
- ✅ Sentences now display in coverage details
- ✅ Topic blocks preserve paragraph boundaries
- ✅ API response includes both blocks and flattened sentences
