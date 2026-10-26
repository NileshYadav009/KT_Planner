"""Naming who spoke (audit 9abd): Teams and Zoom transcript files keep their
speakers (captions.py), the people in a meeting can be listed on upload, and
a finished KT's speakers can be named (POST /kt/{job_id}/speakers), which
rebuilds the document as a new version."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import captions  # noqa: E402
import sessions  # noqa: E402
from speaker_roles import rename_turn_labels  # noqa: E402

TEAMS = """WEBVTT

1a2b3c4d-0001/12-0
00:00:01.200 --> 00:00:04.900
<v Priya Sharma>This is Ledgerline, our invoicing service.</v>

1a2b3c4d-0001/13-0
00:00:05.000 --> 00:00:08.100
<v Priya Sharma>It runs on Amazon EKS.</v>

1a2b3c4d-0002/14-0
00:00:08.300 --> 00:00:10.000
<v Tom Lee>Where are the invoices stored?</v>
"""

ZOOM = """WEBVTT

1
00:00:01.000 --> 00:00:03.000
Priya Sharma: This is Ledgerline.

2
00:00:03.100 --> 00:00:05.000
Tom Lee: How do releases work?

NOTE the recording paused here

3
00:00:05.500 --> 00:00:07.000
Priya Sharma: GitHub Actions deploys it.
"""

SRT = """1
00:00:01,000 --> 00:00:03,000
Priya Sharma: Backups are kept for 14 days.

2
00:00:03,100 --> 00:00:05,000
and restored from AWS Backup.
"""


def test_a_teams_file_becomes_speaker_turns():
    assert captions.to_turns(TEAMS) == ("Priya Sharma: This is Ledgerline, our invoicing service. It runs on Amazon EKS.\n"
                                        "Tom Lee: Where are the invoices stored?")
    # Older Teams exports: short times, no closing tag.
    older = "WEBVTT\n\n0:0:0.0 --> 0:0:5.12\n<v Priya Sharma>Hello everyone.\n\n0:0:5.5 --> 0:0:7.0\n<v Tom Lee>Hi."
    assert captions.to_turns(older) == "Priya Sharma: Hello everyone.\nTom Lee: Hi."


def test_a_zoom_or_srt_file_becomes_speaker_turns():
    assert captions.to_turns(ZOOM) == ("Priya Sharma: This is Ledgerline.\nTom Lee: How do releases work?\n"
                                       "Priya Sharma: GitHub Actions deploys it.")
    # A cue with no speaker joins the turn before it.
    assert captions.to_turns(SRT) == "Priya Sharma: Backups are kept for 14 days. and restored from AWS Backup."


def test_plain_text_is_left_alone():
    assert captions.to_turns("Priya: This is Ledgerline. It runs at 10:30 every night.") is None
    assert captions.to_turns("") is None


def test_renaming_changes_the_labels_never_the_words():
    text = "Priya: Tom said Meera owns it. Meera: Yes.\nTom: Fine."
    assert rename_turn_labels(text, {"Tom": "Tom Lee", "Meera": "Meera Iyer"}) == \
        "Priya: Tom said Meera owns it. Meera Iyer: Yes.\nTom Lee: Fine."


def _finished(**job):
    return dict({"status": "completed"}, **job)


RECORDING = [{"start": 0.0, "end": 4.0, "text": "This is Ledgerline on Amazon EKS.", "speaker": "Speaker 1"},
             {"start": 4.0, "end": 6.0, "text": "Where are backups kept?", "speaker": "Speaker 2"},
             {"start": 6.0, "end": 9.0, "text": "In AWS Backup, for 14 days.", "speaker": "Speaker 1"}]


def test_the_names_are_checked():
    job = _finished(segments=RECORDING, transcript="...")
    assert sessions.speakers_said(job) == ["Speaker 1", "Speaker 2"]
    assert sessions.check_can_rename_speakers(job, {"Speaker 1": " Priya  Sharma ", "Speaker 2": "Speaker 2"}) == \
        {"Speaker 1": "Priya Sharma"}
    # A recording's speaker may take any name of up to 60 characters.
    assert sessions.check_can_rename_speakers(job, {"Speaker 1": "émile de la Cruz"})
    for bad in ({"Speaker 3": "Priya"}, {"Speaker 1": ""}, {"Speaker 1": "Priya: Sharma"},
                {"Speaker 1": "x" * 61}, {"Speaker 1": "Speaker 2"}, {"Speaker 1": "Speaker 1"}, {}, None):
        with pytest.raises(sessions.SessionError):
            sessions.check_can_rename_speakers(job, bad)
    # A pasted transcript's label must stay a label speaker_roles reads.
    pasted = _finished(transcript="Arjun: This is Ledgerline. Meera: Where are backups? Arjun: In AWS Backup.")
    assert sessions.speakers_said(pasted) == ["Arjun", "Meera"]
    with pytest.raises(sessions.SessionError):
        sessions.check_can_rename_speakers(pasted, {"Meera": "meera iyer"})
    assert sessions.check_can_rename_speakers(pasted, {"Meera": "Meera Iyer"}) == {"Meera": "Meera Iyer"}
    # Not while it is being rebuilt, and never once signed.
    for state in ({"giver_pending": True}, {"speakers_pending": True}, {"signoff": {"signed_version": 2}},
                  {"status": "processing"}):
        with pytest.raises(sessions.SessionError):
            sessions.check_can_rename_speakers(dict(job, **state), {"Speaker 1": "Priya Sharma"})


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import ai
    import main
    import pipeline
    from context_mapper import ContextMappingPipeline
    from job_queue import TaskQueue
    from job_store import JobStore, PersistentJobs
    from kt_schema_loader import SCHEMA
    from worker import Worker

    store = JobStore(str(tmp_path / "jobs.sqlite"))
    monkeypatch.setattr(pipeline, "JOB_STORE", store)
    monkeypatch.setattr(pipeline, "JOB_QUEUE", PersistentJobs(store))
    monkeypatch.setattr(pipeline, "TASKS", TaskQueue(store))
    monkeypatch.setattr(pipeline, "get_llm_provider", lambda: None)
    monkeypatch.setattr(ai, "get_llm_provider", lambda: None)
    monkeypatch.setattr(pipeline, "record_candidates", lambda *a, **k: None)
    if pipeline.MAPPER_PIPELINE is None:
        monkeypatch.setattr(pipeline, "MAPPER_PIPELINE", ContextMappingPipeline(SCHEMA, llm_fallback_fn=None))
    return TestClient(main.app), pipeline, lambda job_id: Worker(worker_id="test-speakers").drain(job_id)


def test_a_teams_file_names_its_speakers_and_a_recordings_speakers_can_be_named(client):
    client, pipeline, drain = client
    job_id = client.post("/kt-from-transcript", json={"transcript": TEAMS + ZOOM.replace("WEBVTT", "", 1),
                                                      "participants": "Priya Sharma, Tom Lee, Priya Sharma",
                                                      "force": True}).json()["job_id"]
    drain(job_id)
    status = client.get(f"/status/{job_id}").json()
    assert status["status"] in pipeline.COMPLETED_STATUSES
    assert status["speaker_names"] == ["Priya Sharma", "Tom Lee"]
    assert status["participants"] == ["Priya Sharma", "Tom Lee"]
    assert "WEBVTT" not in status["transcript"] and "-->" not in status["transcript"]

    # The same KT as a recording: its voices are Speaker 1 and Speaker 2.
    job = pipeline.JOB_QUEUE[job_id]
    job.update(segments=RECORDING, transcript=" ".join(s["text"] for s in RECORDING))
    pipeline.JOB_QUEUE[job_id] = job
    assert client.get(f"/status/{job_id}").json()["speaker_names"] == ["Speaker 1", "Speaker 2"]
    assert client.post(f"/kt/{job_id}/speakers", json={"names": {"Speaker 9": "Priya"}}).status_code == 400
    resp = client.post(f"/kt/{job_id}/speakers", json={"names": {"Speaker 1": "Priya Sharma", "Speaker 2": "Tom Lee"}})
    assert resp.status_code == 200, resp.text
    assert client.post(f"/kt/{job_id}/speakers", json={"names": {"Speaker 1": "Tom"}}).status_code == 409
    drain(job_id)
    status = client.get(f"/status/{job_id}").json()
    assert not status.get("speakers_pending") and not status.get("speakers_error")
    assert status["speaker_names"] == ["Priya Sharma", "Tom Lee"]
    assert status["speaker_roles"]["giver"] == "Priya Sharma"
    assert status["participants"] == ["Priya Sharma", "Tom Lee"]
    assert {s.get("speaker") for s in status["knowledge_object"]["sources"]} >= {"Priya Sharma"}
    assert status["document_version"] == 2
    versions = client.get(f"/documents/{job_id}/versions").json()
    summaries = " ".join(str(v.get("summary")) for v in (versions.get("versions") if isinstance(versions, dict)
                                                          else versions))
    assert "Speaker 1 as Priya Sharma" in summaries
