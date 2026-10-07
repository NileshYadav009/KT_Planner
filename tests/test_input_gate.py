"""P0-7: inputs that cannot produce a KT document are refused or flagged
before mapping. Before the gate, an empty transcript, a few pleasantries and
an unpunctuated caption dump each produced a full "KT document"."""
import os
import re
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from input_gate import assess_transcript

KT = (
    "This KT is for TripWise, the booking backend behind our travel app. "
    "The API runs on Amazon ECS with Fargate and bookings are written to DynamoDB. "
    "GitHub Actions deploys to staging on merge and production needs a manual approval. "
    "If a release misbehaves, redeploy the previous task definition. "
    "Datadog is our monitoring tool and PagerDuty pages us if booking success drops below ninety five percent. "
    "One recurring problem is DynamoDB throttling during flash sales; switch the table to on-demand capacity. "
    "Never delete items from the bookings table by hand. "
    "RTO is four hours and backups use point in time recovery. "
    "Escalate to Marco Silva if PagerDuty is not acknowledged within fifteen minutes."
)


@pytest.mark.parametrize("text", [
    "",
    "Um, okay. So yeah. Right, uh.",
    "Hello everyone, thanks for joining today. Let's get started. Any questions? Great, thanks all.",
    "Yesterday I worked on the login page. Today I will continue with it. No blockers from my side. Thanks everyone for the time today, see you all tomorrow at the same time.",
])
def test_non_kt_input_is_rejected(text):
    assert assess_transcript(text)["verdict"] == "reject"


def test_normal_kt_passes():
    result = assess_transcript(KT)
    assert result["verdict"] == "ok", result
    assert result["metrics"]["fact_bearing_sentences"] >= 8


def test_unpunctuated_paste_is_rejected_with_the_real_reason():
    caption = re.sub(r"[^\w\s]", "", KT).lower()
    result = assess_transcript(caption, source="paste")
    assert result["verdict"] == "reject"
    assert "punctuation" in result["reasons"][0]


def test_unpunctuated_audio_is_a_warning_not_a_rejection():
    caption = re.sub(r"[^\w\s]", "", KT).lower()
    result = assess_transcript(caption, source="audio")
    assert result["verdict"] == "warn"
    assert "punctuation" in result["reasons"][0]


def test_thin_kt_is_built_with_a_warning():
    thin = ("This KT is for Orbitpay, our payout service on AWS Lambda with DynamoDB. "
            "The RTO is thirty minutes for the payout API. "
            "Alerts go to Opsgenie and the platform team owns the service since the reorg.")
    result = assess_transcript(thin)
    assert result["verdict"] == "warn"
    assert "follow-up" in result["reasons"][0]


def test_paste_route_refuses_and_explains(monkeypatch):
    from fastapi.testclient import TestClient
    import main
    import pipeline

    queued = []
    monkeypatch.setattr(pipeline, "run_kt_pipeline", lambda *a, **k: queued.append(k))
    client = TestClient(main.app)
    resp = client.post("/kt-from-transcript", json={"transcript": "Hello everyone, thanks for joining. Let's get started."})
    assert resp.status_code == 422
    assert "needs at least" in resp.json()["detail"]
    forced = client.post("/kt-from-transcript", json={"transcript": "Hello everyone, thanks for joining.", "force": True})
    assert forced.status_code == 200
    from worker import Worker
    Worker(worker_id="test-gate").drain(forced.json()["job_id"])         # a worker picks it up (P1-4)
    assert queued and queued[0]["warnings"]          # the reasons still reach the document


def test_upload_of_a_non_kt_recording_fails_with_the_reason(tmp_path, monkeypatch):
    import pipeline

    seg = SimpleNamespace(id=0, seek=0, start=0.0, end=3.0, text="Hello everyone, thanks for joining. Let's get started.",
                          avg_logprob=-0.2, compression_ratio=1.0, no_speech_prob=0.01)
    monkeypatch.setattr(pipeline, "MODEL", SimpleNamespace(transcribe=lambda *a, **k: ([seg], SimpleNamespace(language="en"))))
    monkeypatch.setattr(pipeline.screen_capture, "enabled", lambda: False)
    monkeypatch.setattr(pipeline, "_extract_audio", lambda path, media_format: path)   # no ffmpeg needed here
    src = tmp_path / "upload.tmp"
    src.write_bytes(b"not really audio")
    pipeline.process_upload_task("gate-upload-0001", str(src), str(src) + ".mp3")
    job = pipeline.JOB_QUEUE.pop("gate-upload-0001")
    assert job["status"] == "failed"
    assert "enough KT content" in job["error"]
