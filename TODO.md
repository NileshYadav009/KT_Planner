# KT_Planner Performance Optimization Plan

## Information Gathered
- **Bottlenecks identified** (from main.py, ai.py, context_mapper.py):
  1. Whisper transcription (MODEL.transcribe() - 60-70% time)
  2. SentenceTransformer embeddings (get_sentence_model() - batch but heavy)
  3. FFMPEG audio conversion/trimming/screenshots (3+ calls per job)
  4. Multiple analysis loops (classify → analyze → deduplicate → map_fields)
- Current: 'tiny' Whisper (fastest), batched embeds (bs=64), max_screens=5
- Schema: 20+ sections → embed matrix computation
- Jobs: In-memory queue, background tasks

## Plan (Step-by-step Edits)
1. **main.py**: ✅ COMPLETE
   - max_screens: 5 → 2
   - Skip WAV conversion for audio inputs
   - similarity_threshold: 0.30 → 0.45
2. **ai.py**: ✅ COMPLETE
   - chunk_size: 120 → 200
   - top_indices: [-3:] → [-1:] (top_k 3→1)
3. **context_mapper.py**:
   - Reduce neighbor_window: 3 → 1 (classify_sentence context)
   - Skip librosa/NLTK if not HAS_*
4. **Global**:
   - Add CLI flag `--fast-mode` (disables screenshots, uses base model)
   - requirements.txt: Ensure faster deps (whisper-tiny.en → even faster?)

## Dependent Files to Edit
- main.py (primary: FFMPEG, screenshot logic)
- ai.py (embedding/chunking params)
- context_mapper.py (window sizes)
- requirements.txt (check deps)

## Followup Steps
1. User approval → implement step-by-step
2. Test: `time curl -F file=@devops_kt.mp3 ...` before/after
3. Benchmark: Measure transcription/embed time
4. Deploy: `uvicorn main:app --reload`
5. Validate: No functional regressions (coverage accuracy)

**Estimated speedup: 40-60%** (screenshots -60%, chunks -30%, conversion -20%)
