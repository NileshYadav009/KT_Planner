#!/usr/bin/env python3
"""Download every model Continuum loads, at pinned revisions (P1-5).

    python scripts/fetch_models.py            # download into HF_HOME (the image bakes them in)
    python scripts/fetch_models.py --verify   # load each one with networking off
    python scripts/fetch_models.py --half     # the image: also store the large ones in fp16 (_HALF)

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
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",  # field matching, if the classifier's model is missing
    "urchade/gliner_multi": "b5720abb5b8c575e626e54a7c1001761db8b08db",              # entity extraction
    "microsoft/mdeberta-v3-base": "a0484667b22365f84929a935b5e50a51f71f159d",         # GLiNER's tokenizer
    "Systran/faster-whisper-small": "536b0662742c02347bc0e980a01041f333bce120",       # transcription (CPU default)
}


# Weight formats the app never loads (TensorFlow, Flax, Rust, ONNX and
# OpenVINO exports): about half of the 10 GB the repos hold.
_SKIP = ["*.h5", "*.msgpack", "*.ot", "onnx/*", "openvino/*", "*.onnx"]
# GLiNER builds its backbone from mdeberta's config and loads its own weights
# (gliner 0.2.26: backbone_from_pretrained=False), so mdeberta's own weights
# (3.4 GB) are never read: only its config and tokenizer.
_ONLY = {"microsoft/mdeberta-v3-base": ["*.json", "spm.model"]}
# Stored in fp16, half the size (1.2 GB less), and still run in fp32:
# bge's config pins torch_dtype float32, and GLiNER copies its state dict into
# a model built in fp32 (verify() checks both). transformers 5 loads a
# checkpoint whose config names no dtype in the checkpoint's dtype, so the
# MiniLM models (no dtype in their configs, 90 MB together) stay as they are.
# With these weights rounded to fp16, the golden suite and the e2e KTs
# rendered identical documents (13 Oct, audit §9aac).
_HALF = {
    "BAAI/bge-large-en-v1.5": "model.safetensors",
    "urchade/gliner_multi": "pytorch_model.bin",
}


def _to_half(path: str) -> None:
    import torch

    target = os.path.realpath(path)  # the cache blob the snapshot links to
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file, save_file

        tensors = load_file(target)
        save_file({k: v.half() if v.is_floating_point() else v for k, v in tensors.items()},
                  target + ".half", metadata={"format": "pt"})
    else:
        state = torch.load(target, map_location="cpu", weights_only=True)
        torch.save({k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                    for k, v in state.items()}, target + ".half")
    os.replace(target + ".half", target)


def _ignore(api, repo: str, revision: str) -> list:
    """Skip the PyTorch .bin copy when the repo also has safetensors, which
    transformers loads first."""
    files = api.list_repo_files(repo, revision=revision)
    if any(f.endswith(".safetensors") for f in files):
        return _SKIP + ["pytorch_model*.bin"]
    return _SKIP


def fetch(half: bool = False) -> None:
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    api = HfApi()
    for repo, revision in MODELS.items():
        if repo in _ONLY:
            path = snapshot_download(repo_id=repo, revision=revision, allow_patterns=_ONLY[repo])
        else:
            path = snapshot_download(repo_id=repo, revision=revision, ignore_patterns=_ignore(api, repo, revision))
        if half and repo in _HALF:
            _to_half(os.path.join(path, _HALF[repo]))
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
    # The copy edit's local model (copyedit.py): one 4-bit file, 1.1 GB.
    import copyedit
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(copyedit.LOCAL_REPO, copyedit.LOCAL_FILE, revision=copyedit.LOCAL_REVISION)
    print(f"copy edit -> {path}")
    # Sentence ends for speech to text with no punctuation (punctuation.py):
    # one ONNX file, 210 MB.
    import punctuation

    print(f"punctuation -> {punctuation.fetch()}")
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
    from sentence_transformers import SentenceTransformer

    loaded = {
        "BAAI/bge-large-en-v1.5": SentenceTransformer("BAAI/bge-large-en-v1.5"),
        "all-MiniLM-L6-v2": SentenceTransformer("all-MiniLM-L6-v2"),
    }
    WhisperModel("small", device="cpu", compute_type="int8")
    try:
        from gliner import GLiNER

        loaded["gliner_multi"] = GLiNER.from_pretrained("urchade/gliner_multi")
    except ImportError:
        pass
    # Stored in fp16 (_HALF), they must still run in fp32.
    for name, model in loaded.items():
        dtype = str(next(model.parameters()).dtype)
        if dtype != "torch.float32":
            print(f"{name} loads as {dtype}, expected torch.float32")
            return 1
    import diarization

    if not diarization.available():
        print("speaker diarisation models missing: run scripts/fetch_models.py")
        return 1
    diarization._diarizer()
    import copyedit

    if copyedit.ENABLED and copyedit.local_model_path() is None and copyedit.LOCAL_MODEL.lower() != "off":
        print("copy-edit model missing: run scripts/fetch_models.py")
        return 1
    import punctuation

    if not punctuation.available():
        print("punctuation model missing: run scripts/fetch_models.py")
        return 1
    print("all models load offline")
    return 0


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(verify())
    fetch(half="--half" in sys.argv)
