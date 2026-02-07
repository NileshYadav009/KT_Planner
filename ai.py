import json
import re
from sentence_transformers import SentenceTransformer, util

with open("kt_schema.json") as f:
    SCHEMA = json.load(f)["sections"]

MODEL = None
SECTION_EMBEDS = None

SECTION_HINTS = {
    s["id"]: {hint.lower() for hint in s["hints"]}
    for s in SCHEMA
}

SECTION_HINTS = {
    s["id"]: {hint.lower() for hint in s["hints"]}
    for s in SCHEMA
}

WORD_RE = re.compile(r"\b\w+\b")

def chunk_text(text, size=250):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i + size])

def classify_transcript(transcript: str, *, similarity_threshold: float = 0.35):
    coverage = {s["id"]: [] for s in SCHEMA}
    section_embeds = get_section_embeds()

    for chunk in chunk_text(transcript):
        emb = model.encode(chunk, convert_to_tensor=True)
        chunk_tokens = {token.lower() for token in WORD_RE.findall(chunk)}
        for sec_id, sec_emb in SECTION_EMBEDS.items():
            score = util.cos_sim(emb, sec_emb).item()
            has_hint = bool(chunk_tokens & SECTION_HINTS[sec_id])
            if score > similarity_threshold or has_hint:
                coverage[sec_id].append(chunk)

    return coverage
