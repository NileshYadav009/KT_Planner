"""
New endpoint to be added to main.py:

@app.post("/reclassify/{job_id}")
async def reclassify_transcript_using_ai_matching(job_id: str):
    '''
    CRITICAL: Re-classify the entire transcript using ai.py's classify_transcript function.
    
    This uses the intelligent 3-level hint matching algorithm with weightage:
    - Level 3: Exact phrase match (highest confidence) 
    - Level 2: Token match at word boundaries
    - Level 1: Partial token match
    
    Run this when:
    - Coverage is showing all sections as "missing"
    - Initial classification didn't work properly
    - You want to use better word-based matching
    - After receiving new audio
    '''
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        kt = job.get("kt_structured")
        if not kt or not isinstance(kt, dict):
            raise HTTPException(status_code=400, detail="KT data not available")
    
    transcript = kt.get("transcript", "")
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript available")
    
    # ===== USE THE POWERFUL classify_transcript FUNCTION FROM ai.py =====
    from ai import classify_transcript
    
    # Call the intelligent word-matching classification
    classified_chunks = classify_transcript(transcript, similarity_threshold=-0.05)
    
    # Build section_content from classified chunks
    section_content = {}
    
    for sec_id, chunks in classified_chunks.items():
        # Find section in schema
        section = next((s for s in SCHEMA if s["id"] == sec_id), None)
        if not section:
            continue
        
        # Convert chunks to sentences
        sentences = []
        for chunk_idx, chunk in enumerate(chunks):
            sentences.append({
                "text": chunk,
                "original_chunk": chunk,
                "confidence": 0.75,  # Default confidence for auto-classified
                "chunk_index": chunk_idx,
                "classification_method": "ai_word_matching_v3"
            })
        
        section_content[sec_id] = {
            "section_id": sec_id,
            "section_title": section.get("title"),
            "sentences": sentences,
            "enhanced_texts": [s["text"] for s in sentences],
            "repair_actions": [],
            "screenshots": [],
            "confidence": 0.75 if sentences else 0.0,
            "sentence_count": len(sentences)
        }
    
    # Update KT with new classifications
    kt["section_content"] = section_content
    kt["classification_method"] = "ai_word_matching_3level"
    
    # Clear any previous unassigned sentences (all got classified now)
    kt["unassigned_sentences"] = []
    
    # Recalculate coverage
    new_coverage, new_missing, new_progress = recalculate_coverage_from_section_content(section_content)
    
    # Persist updates
    with JOB_LOCK:
        JOB_QUEUE[job_id]["kt_structured"] = kt
        JOB_QUEUE[job_id]["coverage"] = new_coverage
        JOB_QUEUE[job_id]["missing_required"] = new_missing
        JOB_QUEUE[job_id]["progress"] = new_progress
    
    # Calculate how many sections now have content
    covered_count = sum(1 for c in new_coverage.values() if c["status"] in {"covered", "weak"})
    
    return {
        "status": "reclassified",
        "job_id": job_id,
        "method": "AI Word Matching (3-level hint weighting)",
        "new_coverage_percent": new_progress,
        "sections_now_covered": covered_count,
        "total_sections": len(SCHEMA),
        "sections_populated": {
            sec_id: len(section_content.get(sec_id, {}).get("sentences", []))
            for sec_id in section_content.keys()
        },
        "message": f"Re-classification complete using intelligent word matching. Coverage improved to {new_progress}%"
    }
"""
