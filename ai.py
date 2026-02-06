import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("kt_schema.json") as f:
    SCHEMA = json.load(f)["sections"]

SECTION_EMBEDS = {
    s["id"]: model.encode(" ".join(s["hints"]), convert_to_tensor=True)
    for s in SCHEMA
}

def chunk_text(text, size=300):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i+size])

def classify_transcript(transcript: str):
    coverage = {s["id"]: [] for s in SCHEMA}

    for chunk in chunk_text(transcript):
        emb = model.encode(chunk, convert_to_tensor=True)
        for sec_id, sec_emb in SECTION_EMBEDS.items():
            score = util.cos_sim(emb, sec_emb).item()
            if score > 0.35:
                coverage[sec_id].append(chunk)

    return coverage