"""Batched Whisper decodes each chunk once; a chunk whose text stops early is
decoded again from where it stopped (audit §9abm). No model is loaded: the
batched pipeline's own forward pass is replaced by a script."""
from types import SimpleNamespace

import numpy as np
import pytest

import pipeline
from faster_whisper import BatchedInferencePipeline, WhisperModel


def scripted(script, calls):
    """A forward pass that answers each chunk from `script` (offset -> segments
    relative to it) and records what it was given."""
    def forward(self, features, tokenizer, chunks_metadata, options):
        calls.append([(round(m["offset"], 2), round(m["duration"], 2), f.shape, float(f[0, 0]))
                      for f, m in zip(features, chunks_metadata)])
        return [[{"start": m["offset"] + a, "end": m["offset"] + b, "text": t}
                 for a, b, t in script.get(round(m["offset"], 2), [])]
                for m in chunks_metadata]
    return forward


def run(monkeypatch, script, duration, marker_frame=None):
    calls = []
    monkeypatch.setattr(BatchedInferencePipeline, "forward", scripted(script, calls))
    resuming = pipeline._ResumingBatchedPipeline(model=SimpleNamespace(frames_per_second=100))
    features = np.zeros((1, 80, 3000), dtype=np.float32)
    if marker_frame is not None:
        features[0, 0, marker_frame] = 7.0
    out = resuming.forward(features, None, [{"offset": 0.0, "duration": duration, "segments": []}], None)
    return [s["text"] for s in out[0]], calls


def test_a_chunk_that_stops_early_is_decoded_again_from_where_it_stopped(monkeypatch):
    # The synthetic W02: Whisper wrote up to 26.4 s of a 29.26 s chunk and
    # stopped; "ArgoCD deploys them to EKS." was never in the transcript.
    texts, calls = run(monkeypatch, {0.0: [(0, 12.6, " One."), (12.6, 26.4, " Two.")],
                                     26.4: [(0, 2.7, " ArgoCD deploys them to EKS.")]}, 29.26, marker_frame=2640)
    assert texts == [" One.", " Two.", " ArgoCD deploys them to EKS."]
    # The rest of the chunk, from frame 2640, padded back to a 30 s window.
    assert calls[1] == [(26.4, 2.86, (80, 3000), 7.0)]


def test_a_chunk_written_to_its_end_is_not_decoded_again(monkeypatch):
    texts, calls = run(monkeypatch, {0.0: [(0, 28.5, " All of it.")]}, 29.26)
    assert texts == [" All of it."]
    assert len(calls) == 1


def test_a_chunk_with_nothing_written_is_not_decoded_again(monkeypatch):
    texts, calls = run(monkeypatch, {}, 29.26)
    assert texts == []
    assert len(calls) == 1


def test_a_second_decode_that_writes_nothing_ends_it(monkeypatch):
    texts, calls = run(monkeypatch, {0.0: [(0, 20.0, " A.")], 20.0: [(0, 0, "  ")]}, 29.26)
    assert texts == [" A."]
    assert len(calls) == 2


def test_decoding_again_is_bounded(monkeypatch):
    # Each pass writes one more second; it stops after RESUME_PASSES passes.
    script = {float(k): [(0, 1.0, f" S{k}.")] for k in range(30)}
    texts, calls = run(monkeypatch, script, 29.0)
    assert len(calls) == 1 + pipeline._ResumingBatchedPipeline.RESUME_PASSES
    assert texts == [" S0.", " S1.", " S2.", " S3."]


@pytest.mark.parametrize("resume, kind", [(True, "_ResumingBatchedPipeline"), (False, "BatchedInferencePipeline")])
def test_the_app_resumes_unless_turned_off(monkeypatch, resume, kind):
    monkeypatch.setattr(pipeline, "MODEL", object.__new__(WhisperModel))
    monkeypatch.setattr(pipeline, "_BATCHED_MODEL", None)
    monkeypatch.setattr(pipeline, "WHISPER_BATCHED", True)
    monkeypatch.setattr(pipeline, "WHISPER_BATCHED_RESUME", resume)
    monkeypatch.setattr(BatchedInferencePipeline, "transcribe", lambda self, path, **kw: type(self).__name__)
    assert pipeline._transcribe("audio.wav", language="en") == kind
