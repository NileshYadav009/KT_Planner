"""Speaker diarisation (P1-7): who said each segment, carried to the sources."""
import json
import os
import shutil
import subprocess

import pytest

import diarization
from diarization import Turn, assign_speakers

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "two_speaker_kt")


def test_each_segment_takes_the_speaker_who_talks_most_during_it():
    segments = [{"start": 0.0, "end": 4.0}, {"start": 4.5, "end": 6.0}, {"start": 6.2, "end": 9.0}]
    turns = [Turn(0.0, 4.2, 7), Turn(4.4, 6.1, 3), Turn(6.1, 6.4, 7), Turn(6.4, 9.0, 3)]
    summary = assign_speakers(segments, turns)
    # Numbered by first appearance, whatever the clustering called them.
    assert [s["speaker"] for s in segments] == ["Speaker 1", "Speaker 2", "Speaker 2"]
    assert summary["count"] == 2
    assert summary["speakers"][0]["label"] == "Speaker 2" and summary["speakers"][0]["share"] > 0.5


def test_a_segment_in_a_pause_takes_the_nearest_turn_only_when_it_is_close():
    segments = [{"start": 5.0, "end": 5.5}, {"start": 20.0, "end": 21.0}]
    assign_speakers(segments, [Turn(0.0, 4.6, 0)])
    assert segments[0]["speaker"] == "Speaker 1"
    assert "speaker" not in segments[1]


def test_switched_off_or_without_models_nothing_is_labelled(monkeypatch):
    monkeypatch.setenv("CONTINUUM_DIARIZATION", "0")
    segments = [{"start": 0.0, "end": 1.0, "text": "Hello."}]
    assert diarization.label_segments("missing.wav", segments) is None
    assert "speaker" not in segments[0]


def test_the_notice_says_how_many_speakers_and_who_talked_most():
    assert diarization.speaker_notice(None) is None
    line = diarization.speaker_notice({"count": 2, "speakers": [{"label": "Speaker 1", "share": 0.806}]})
    assert line.startswith("2 speakers") and "Speaker 1 spoke 81%" in line


def test_sources_say_who_said_each_sentence():
    from knowledge.evidence import transcript_sentences
    from pdf_rendering import render_sources_appendix

    segments = [
        {"start": 0.0, "end": 3.0, "text": "Where does it run?", "speaker": "Speaker 2"},
        {"start": 3.5, "end": 9.0, "text": "It runs on Azure Kubernetes Service in North Europe.",
         "speaker": "Speaker 1"},
    ]
    sentences = transcript_sentences("", segments)
    assert [s.get("speaker") for s in sentences] == ["Speaker 2", "Speaker 1"]
    html = render_sources_appendix([{"id": 1, "quote": sentences[1]["quote"], "start": 3.5, "speaker": "Speaker 1"}])
    assert "Speaker 1" in html and "who said it" in html
    # A pasted transcript has neither times nor speakers.
    assert "speaker" not in transcript_sentences("It runs on AKS. Backups are nightly.")[0]


@pytest.mark.skipif(not diarization.available() or not shutil.which("ffmpeg"),
                    reason="speaker diarisation models not installed (scripts/fetch_models.py)")
def test_two_speakers_are_told_apart_in_a_recording(tmp_path):
    wav = str(tmp_path / "two.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", FIXTURE + ".mp3", "-ac", "1", "-ar", "16000",
                    "-acodec", "pcm_s16le", wav], check=True)
    with open(FIXTURE + ".json", encoding="utf-8") as fh:
        truth = json.load(fh)["turns"]
    # One segment per spoken turn, as Whisper would cut it.
    segments = [{"start": t["start"], "end": t["end"], "text": t["text"]} for t in truth]
    summary = diarization.label_segments(wav, segments)
    assert summary["count"] == 2
    giver = segments[0]["speaker"]
    correct = sum((s["speaker"] == giver) == (t["speaker"] == "giver") for s, t in zip(segments, truth))
    assert correct / len(truth) >= 0.9
