"""Who said what in a recorded KT (P1-7).

Whisper does not tell speakers apart, so every sentence used to have no
speaker: the document could not say who stated a fact, and a question could
only be paired with its answer by punctuation. This module labels each
Whisper segment "Speaker 1", "Speaker 2", ... in order of first appearance.

It runs locally with sherpa-onnx and two openly published ONNX models, so no
account, token or hosted service is involved and the audio never leaves the
machine:

    segmentation  pyannote segmentation-3.0 (MIT, CNRS), as exported by sherpa-onnx
    embedding     WeSpeaker ResNet34 trained on VoxCeleb (CC BY 4.0)

`python scripts/fetch_models.py` downloads both at pinned checksums (the
image bakes them in). Without them, or without sherpa-onnx, transcription
runs exactly as before and segments have no speaker. Diarisation never fails
a KT: on any error the document is built without speakers.

Set CONTINUUM_DIARIZATION=0 to switch it off; CONTINUUM_DIARIZATION_DIR to
keep the models elsewhere.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
# name -> (url, sha256 of the download, file inside an archive or None, sha256 of that file)
MODELS = {
    "segmentation": (
        f"{_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
        "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
        "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    ),
    "embedding": (
        f"{_RELEASES}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx",
        "5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94",
        None,
        "5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94",
    ),
}
_FILE_NAMES = {"segmentation": "pyannote-segmentation-3.0.onnx", "embedding": "wespeaker-voxceleb-resnet34.onnx"}

# Speaker embeddings closer than this are one speaker. On a two-speaker test
# recording, 0.5-0.7 all found exactly two speakers with every labelled
# stretch right; lower values split one speaker in two.
CLUSTER_THRESHOLD = float(os.getenv("CONTINUUM_DIARIZATION_THRESHOLD", "0.6"))
# A Whisper segment with no diarised speech under it (a short pause) takes
# the speaker of the nearest turn within this many seconds.
_NEAREST_TURN_SECONDS = 1.0


@dataclass
class Turn:
    start: float
    end: float
    speaker: int


def model_dir() -> str:
    base = os.getenv("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.getenv("CONTINUUM_DIARIZATION_DIR") or os.path.join(base, "continuum-diarization")


def model_paths() -> Dict[str, str]:
    return {name: os.path.join(model_dir(), file_name) for name, file_name in _FILE_NAMES.items()}


def enabled() -> bool:
    return os.getenv("CONTINUUM_DIARIZATION", "1").strip().lower() not in ("0", "false", "no", "off")


def available() -> bool:
    """sherpa-onnx is installed, the models are on disk and it is switched on."""
    if not enabled():
        return False
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return all(os.path.exists(p) for p in model_paths().values())


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(dest: Optional[str] = None) -> Dict[str, str]:
    """Download both models, check their checksums and put them in `dest`
    (default model_dir()). Already present and intact: left alone."""
    dest = dest or model_dir()
    os.makedirs(dest, exist_ok=True)
    out = {}
    for name, (url, download_sha, member, file_sha) in MODELS.items():
        target = os.path.join(dest, _FILE_NAMES[name])
        if os.path.exists(target) and _sha256(target) == file_sha:
            out[name] = target
            continue
        with tempfile.TemporaryDirectory() as tmp:
            download = os.path.join(tmp, "download")
            urllib.request.urlretrieve(url, download)  # noqa: S310 (pinned https URL, checked below)
            if _sha256(download) != download_sha:
                raise RuntimeError(f"{name} model download does not match its pinned checksum: {url}")
            source = download
            if member:
                with tarfile.open(download, "r:bz2") as archive:
                    archive.extract(member, tmp, filter="data")
                source = os.path.join(tmp, member)
            if _sha256(source) != file_sha:
                raise RuntimeError(f"{name} model file does not match its pinned checksum")
            shutil.copyfile(source, target)
        out[name] = target
    return out


_DIARIZER = None


def _diarizer():
    global _DIARIZER
    if _DIARIZER is None:
        import sherpa_onnx

        paths = model_paths()
        threads = max(1, min(4, os.cpu_count() or 1))
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=paths["segmentation"]),
                num_threads=threads),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=paths["embedding"], num_threads=threads),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=CLUSTER_THRESHOLD),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not config.validate():
            raise RuntimeError("speaker diarisation models are not usable")
        _DIARIZER = sherpa_onnx.OfflineSpeakerDiarization(config)
    return _DIARIZER


def _read_wav(path: str):
    import numpy as np

    with wave.open(path) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("diarisation needs 16-bit mono audio")
        rate = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return samples, rate


def diarize(wav_path: str) -> List[Turn]:
    """Speaker turns in a 16 kHz mono WAV (the file Whisper transcribed, so
    the times match its segments)."""
    diarizer = _diarizer()
    samples, rate = _read_wav(wav_path)
    if rate != diarizer.sample_rate:
        raise ValueError(f"diarisation needs {diarizer.sample_rate} Hz audio, got {rate} Hz")
    result = diarizer.process(samples).sort_by_start_time()
    return [Turn(float(r.start), float(r.end), int(r.speaker)) for r in result]


def assign_speakers(segments: Sequence[Dict[str, Any]], turns: Sequence[Turn]) -> Dict[str, Any]:
    """Give each segment the speaker who talks most during it ("speaker":
    "Speaker N", numbered by first appearance). A segment no turn overlaps
    takes the nearest turn's speaker within a second, else stays unlabelled.
    Returns who spoke and for how long."""
    labels: Dict[int, str] = {}
    seconds: Dict[str, float] = {}
    for seg in segments:
        start, end = float(seg.get("start") or 0.0), float(seg.get("end") or 0.0)
        overlap: Dict[int, float] = {}
        for turn in turns:
            shared = min(end, turn.end) - max(start, turn.start)
            if shared > 0:
                overlap[turn.speaker] = overlap.get(turn.speaker, 0.0) + shared
        if overlap:
            speaker = max(overlap, key=overlap.get)
        else:
            distance, speaker = min(((max(turn.start - end, start - turn.end), turn.speaker) for turn in turns),
                                    default=(None, None))
            if distance is None or distance > _NEAREST_TURN_SECONDS:
                continue
        label = labels.setdefault(speaker, f"Speaker {len(labels) + 1}")
        seg["speaker"] = label
        seconds[label] = seconds.get(label, 0.0) + max(0.0, end - start)
    total = sum(seconds.values()) or 1.0
    return {
        "count": len(seconds),
        "speakers": [{"label": label, "seconds": round(secs, 1), "share": round(secs / total, 3)}
                     for label, secs in sorted(seconds.items(), key=lambda kv: -kv[1])],
    }


def label_segments(wav_path: str, segments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Diarise `wav_path` and label `segments` in place. None when diarisation
    is unavailable or switched off; raises on a failure the caller reports."""
    if not segments or not available():
        return None
    return assign_speakers(segments, diarize(wav_path))


def speaker_notice(summary: Optional[Dict[str, Any]]) -> Optional[str]:
    """One line for the document: how many speakers, and who talked most."""
    if not summary or not summary.get("count"):
        return None
    if summary["count"] == 1:
        return "One speaker was identified in the recording."
    top = summary["speakers"][0]
    return (f"{summary['count']} speakers were identified in the recording; {top['label']} spoke "
            f"{round(top['share'] * 100)}% of the time. Sources show who said each sentence.")
