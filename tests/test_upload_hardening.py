"""P1-14: uploads are untrusted. Only real audio/video recordings reach
ffmpeg's decoders, playlist and concat tricks cannot make ffmpeg read other
files, and size and length are capped. Before, any file was handed to
ffmpeg, and the original was passed to the Whisper decoder if conversion
failed."""
import asyncio
import io
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import media_guard

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe not available on PATH")

CANARY = "CANARY-7f3a-SERVER-FILE"


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    files = {}
    files["wav"] = d / "tone.wav"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(files["wav"]))
    files["mp4"] = d / "meeting.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=5:duration=2", "-f", "lavfi", "-i",
            "sine=frequency=300:duration=2", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(files["mp4"]))
    files["silent_video"] = d / "no_audio.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=5:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(files["silent_video"]))
    canary = d / "secret.txt"
    canary.write_text(CANARY)
    posix = canary.as_posix()
    files["hls"] = d / "playlist.m3u8"
    files["hls"].write_text("#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\nfile:///" + posix.lstrip("/")
                            + "\n#EXT-X-ENDLIST\n")
    files["concat"] = d / "list.ffconcat"
    files["concat"].write_text(f"ffconcat version 1.0\nfile '{files['wav'].as_posix()}'\nfile '{posix}'\n")
    files["text"] = d / "notes.mp3"
    files["text"].write_text("This is not audio, whatever the name says.")
    files["png"] = d / "slide.mp4"
    _ffmpeg("-f", "lavfi", "-i", "color=c=blue:size=64x64", "-frames:v", "1", "-f", "image2", "-c:v", "png",
            str(d / "slide.png"))
    shutil.copy(d / "slide.png", files["png"])
    return files


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import main
    import pipeline

    queued = []
    monkeypatch.setattr(pipeline, "process_upload_task", lambda *a, **k: queued.append((a, k)))
    c = TestClient(main.app)
    c.queued = queued
    return c


def _upload(client, path, name=None):
    with open(path, "rb") as fh:
        return client.post("/upload", files={"file": (name or os.path.basename(path), fh, "application/octet-stream")})


# --------------------------------------------------------------------------
# What is refused
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind,name", [("hls", "meeting.m3u8"), ("hls", "meeting.mp4"), ("concat", "list.ffconcat"),
                                       ("text", "notes.mp3"), ("png", "slide.mp4")])
def test_files_that_are_not_recordings_are_refused_before_decoding(client, media, kind, name):
    resp = _upload(client, media[kind], name)
    assert resp.status_code == 422, resp.text
    assert "supported recording" in resp.json()["detail"]
    assert client.queued == []                          # nothing was queued for ffmpeg or Whisper
    assert CANARY not in resp.text


def test_a_playlist_cannot_make_ffmpeg_read_a_server_file(media, tmp_path):
    """Even called directly, every ffmpeg entry point refuses the playlist and
    the concat list, so the canary file is never read into the output."""
    for kind in ("hls", "concat"):
        with pytest.raises(media_guard.MediaRejected):
            media_guard.probe(str(media[kind]))
        with pytest.raises(media_guard.MediaRejected):
            media_guard.extract_audio(str(media[kind]), None, str(tmp_path / f"{kind}.wav"))
        assert not (tmp_path / f"{kind}.wav").exists() or CANARY.encode() not in (tmp_path / f"{kind}.wav").read_bytes()
    import screen_capture
    assert screen_capture.has_video_stream(str(media["concat"])) is False


def test_a_video_without_an_audio_track_is_refused(client, media):
    resp = _upload(client, media["silent_video"])
    assert resp.status_code == 422 and "no audio track" in resp.json()["detail"]


def test_oversized_uploads_are_refused(client, media, monkeypatch):
    monkeypatch.setenv("CONTINUUM_MAX_UPLOAD_MB", "0.01")                  # 10 KB; the tone is ~64 KB
    resp = _upload(client, media["wav"])
    assert resp.status_code == 413 and "larger than" in resp.json()["detail"]
    assert client.queued == []

    # Without a declared length (chunked upload) the streamed copy stops it.
    class FakeUpload:
        def __init__(self, data):
            self._buf = io.BytesIO(data)

        async def read(self, n=-1):
            return self._buf.read(n)

    with pytest.raises(media_guard.MediaRejected) as exc:
        asyncio.run(media_guard.save_upload(FakeUpload(media["wav"].read_bytes())))
    assert exc.value.status == 413


def test_recordings_over_the_length_limit_are_refused(client, media, monkeypatch):
    monkeypatch.setenv("CONTINUUM_MAX_MEDIA_MINUTES", "0.01")              # 0.6 s; the tone is 2 s
    resp = _upload(client, media["wav"])
    assert resp.status_code == 422 and "limit is" in resp.json()["detail"]


# --------------------------------------------------------------------------
# What is accepted
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind,expected", [("wav", "wav"), ("mp4", "mov")])
def test_real_recordings_are_accepted_with_their_probed_format(client, media, kind, expected):
    resp = _upload(client, media[kind])
    assert resp.status_code == 200, resp.text
    from worker import Worker
    Worker(worker_id="test-upload").drain(resp.json()["job_id"])        # a worker picks it up (P1-4)
    (args, kwargs), = client.queued
    assert kwargs["media_format"] == expected
    os.unlink(args[1])                                   # the saved upload (the stubbed task would delete it)


def test_audio_is_extracted_to_wav_with_safe_arguments(media, tmp_path):
    out = media_guard.extract_audio(str(media["mp4"]), "mov", str(tmp_path / "audio.wav"))
    assert media_guard.probe(out)["format"] == "wav"
    args = media_guard.input_args("x.mp4", "mov")
    assert args[:2] == ["-protocol_whitelist", "file"] and "-format_whitelist" in args
    assert args[-4:] == ["-f", "mov", "-i", "x.mp4"]
    assert "-f" not in media_guard.input_args("x", "hls")                 # never forces a disallowed demuxer


def test_the_original_upload_never_reaches_whisper(tmp_path, monkeypatch):
    """If ffmpeg cannot extract audio, the job fails with the reason; the
    upload is not handed to the decoder as a fallback."""
    import pipeline

    transcribed = []
    monkeypatch.setattr(pipeline, "MODEL", SimpleNamespace(
        transcribe=lambda path, **k: transcribed.append(path) or ([], SimpleNamespace(language="en"))))
    monkeypatch.setattr(pipeline.screen_capture, "enabled", lambda: False)
    src = tmp_path / "upload.upload"
    src.write_bytes(b"RIFF....WAVEfmt this is not a real wav file")
    pipeline.JOB_QUEUE["upload-guard-0001"] = {"status": "processing", "progress": 0}
    pipeline.process_upload_task("upload-guard-0001", str(src), str(src) + ".mp3", media_format="wav")
    job = pipeline.JOB_QUEUE.pop("upload-guard-0001")
    assert job["status"] == "failed" and "No audio could be read" in job["error"]
    assert transcribed == []
    assert not src.exists()                              # the upload is deleted either way
