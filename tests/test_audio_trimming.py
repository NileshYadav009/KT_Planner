"""Tests for pipeline.trim_leading_trailing_silence().

Regression guard for a real transcription-data-loss bug: the original
silence-trimming filter chain used `silenceremove(..., stop_periods=1, ...)`
on a single forward pass. `stop_periods` does not trim trailing silence --
it stops the ENTIRE filter output at the first silence gap found anywhere
in the stream and discards everything after it. Any real recording with a
normal pause between sentences (which is virtually all real speech) had
most of its audio silently thrown away before Whisper ever saw it, so
uploads would "transcribe" only the first sentence or two. Reproduced
end-to-end with a real MP3 of ~20s of speech: the old filter chain cut it
down to 1.4s and a 241-character transcript came back as 12 characters.

These tests use synthetic tone/silence clips generated via ffmpeg's lavfi
sources rather than real speech audio, so they run fast and don't depend on
any TTS engine or the Whisper model.
"""
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline import trim_leading_trailing_silence

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
pytestmark = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not available on PATH")


def _make_tone_silence_tone_clip(path: str, tone_s: float = 1.0, silence_s: float = 1.0) -> None:
    """Build a WAV: `tone_s` sine tone, `silence_s` true silence, `tone_s` sine tone."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={tone_s}",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-filter_complex",
            f"[1:a]atrim=0:{silence_s}[sil];[0:a][sil][0:a]concat=n=3:v=0:a=1[out]",
            "-map", "[out]", "-ar", "16000", "-ac", "1", path,
        ],
        check=True, capture_output=True,
    )


def _duration_seconds(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def test_mid_stream_pause_is_not_truncated(tmp_path):
    src = str(tmp_path / "clip.wav")
    out = str(tmp_path / "trimmed.wav")
    _make_tone_silence_tone_clip(src, tone_s=1.0, silence_s=1.0)
    original_duration = _duration_seconds(src)
    assert original_duration > 2.5  # sanity check on the fixture itself

    ok = trim_leading_trailing_silence(src, out)

    assert ok
    trimmed_duration = _duration_seconds(out)
    # The old buggy stop_periods=1 chain collapsed this down to ~1.0s
    # (everything from the first silence gap onward was discarded).
    # A correct trim only removes leading/trailing silence, so most of the
    # clip -- including the second tone, past the mid-stream pause -- must
    # still be present.
    assert trimmed_duration > original_duration * 0.7, (
        f"expected most of the {original_duration:.2f}s clip to survive "
        f"trimming, got {trimmed_duration:.2f}s -- mid-stream audio was "
        f"likely truncated at the pause"
    )


def test_leading_and_trailing_silence_is_still_trimmed(tmp_path):
    src = str(tmp_path / "padded.wav")
    out = str(tmp_path / "trimmed.wav")
    # 1.5s of silence, 1s tone, 1.5s of silence
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-filter_complex",
            "[1:a]atrim=0:1.5[sil];[sil][0:a][sil]concat=n=3:v=0:a=1[out]",
            "-map", "[out]", "-ar", "16000", "-ac", "1", src,
        ],
        check=True, capture_output=True,
    )
    original_duration = _duration_seconds(src)
    assert original_duration >= 3.5

    ok = trim_leading_trailing_silence(src, out)

    assert ok
    trimmed_duration = _duration_seconds(out)
    # A meaningful chunk of the 3s of padding silence should be gone. Not
    # asserting an exact amount -- the filter's detection window means the
    # precise trimmed length is somewhat ffmpeg-version-dependent -- just
    # that leading/trailing silence trimming still works at all.
    assert trimmed_duration < original_duration - 0.5
