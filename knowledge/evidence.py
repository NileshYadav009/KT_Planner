from typing import Any, Dict, List


def build_evidence(sentence_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence = []
    for idx, sentence in enumerate(sentence_entries[:3]):
        evidence.append({
            "sentence_index": idx,
            "text": sentence.get("text", ""),
            "start": sentence.get("start"),
            "end": sentence.get("end"),
            "speaker": sentence.get("speaker"),
            "audio_confidence": sentence.get("audio_confidence"),
        })
    return evidence
