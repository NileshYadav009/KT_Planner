#!/usr/bin/env python3
"""Download every model Continuum loads, at pinned revisions (P1-5).

    python scripts/fetch_models.py            # download into HF_HOME (the image bakes them in)
    python scripts/fetch_models.py --verify   # load each one with networking off

The app loads these by name at their default revision; this script fetches
the pinned commit and points the local "main" ref at it, so an offline
container (HF_HUB_OFFLINE=1) gets exactly these files and output cannot
change because a model was updated upstream.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# repo id -> commit. Update deliberately, then re-run the golden suite.
MODELS = {
    "BAAI/bge-large-en-v1.5": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",          # section classifier
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",  # reranker
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",  # chunking, field mapping
    "urchade/gliner_multi": "b5720abb5b8c575e626e54a7c1001761db8b08db",              # entity extraction
    "microsoft/mdeberta-v3-base": "a0484667b22365f84929a935b5e50a51f71f159d",         # GLiNER's tokenizer
    "Systran/faster-whisper-small": "536b0662742c02347bc0e980a01041f333bce120",       # transcription (CPU default)
}


def fetch() -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    for repo, revision in MODELS.items():
        path = snapshot_download(repo_id=repo, revision=revision)
        # Offline loads ask for "main": make it resolve to the pinned commit.
        refs = os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"), "refs")
        os.makedirs(refs, exist_ok=True)
        with open(os.path.join(refs, "main"), "w", encoding="ascii") as fh:
            fh.write(revision)
        print(f"{repo}@{revision[:10]} -> {path}")
    # Speaker diarisation (P1-7): two ONNX files from sherpa-onnx's releases,
    # checked against pinned SHA-256 sums.
    import diarization

    for name, path in diarization.fetch().items():
        print(f"diarisation {name} -> {path}")
    try:
        from rapidocr import RapidOCR

        RapidOCR()            # fetches its ONNX models on first use
        print("rapidocr models ready")
    except Exception as exc:  # screen capture is optional; report, don't fail the build
        print(f"rapidocr models not fetched: {exc}")


def verify() -> int:
    """Load every model the way the app does, with the network off."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from faster_whisper import WhisperModel
    from sentence_transformers import CrossEncoder, SentenceTransformer

    SentenceTransformer("BAAI/bge-large-en-v1.5")
    SentenceTransformer("all-MiniLM-L6-v2")
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    WhisperModel("small", device="cpu", compute_type="int8")
    try:
        from gliner import GLiNER

        GLiNER.from_pretrained("urchade/gliner_multi")
    except ImportError:
        pass
    import diarization

    if not diarization.available():
        print("speaker diarisation models missing: run scripts/fetch_models.py")
        return 1
    diarization._diarizer()
    print("all models load offline")
    return 0


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(verify())
    fetch()
