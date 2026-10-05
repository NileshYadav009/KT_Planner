"""Untrusted uploads: size, type and length checks before anything decodes
them, and safe ffmpeg arguments (P1-14).

Every upload comes from outside. Before this module the whole file was read
into memory, any file was handed to ffmpeg whatever it was, and if ffmpeg
could not convert it the original was passed to the Whisper decoder anyway.
ffmpeg's playlist and concat formats can make it read other files on the
server or fetch URLs, which is a standard pentest finding. Now:

  * the upload is streamed to disk and refused above CONTINUUM_MAX_UPLOAD_MB
    (default 2048, i.e. 2 GB, the limit the UI states: about two hours of
    meeting video with screen share; length is capped separately below);
  * ffprobe identifies the container first, and only the demuxers in
    ALLOWED_DEMUXERS may open it: no playlists, concat lists, image
    sequences or devices. It must have an audio track, and recordings longer
    than CONTINUUM_MAX_MEDIA_MINUTES (default 240) are refused;
  * every ffmpeg or ffprobe run on an upload reads local files only
    (-protocol_whitelist file), may use only the allowed demuxers, and has a
    time limit.

Running ffmpeg in a separate, unprivileged worker is the remaining step (P1-4).
"""
import json
import os
import subprocess
import tempfile
from typing import List, Optional

# ffprobe's demuxer names: "mov" covers MP4, MOV and M4A; "matroska" covers
# MKV and WebM.
ALLOWED_DEMUXERS = ("mov", "matroska", "mp3", "wav", "ogg", "flac", "aac")
SUPPORTED_DESCRIPTION = "MP4, MOV, M4A, MKV, WebM, MP3, WAV, OGG, FLAC or AAC"
PROBE_TIMEOUT_SECONDS = 30
CHUNK_BYTES = 1024 * 1024


class MediaRejected(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def max_upload_bytes() -> int:
    return int(float(os.getenv("CONTINUUM_MAX_UPLOAD_MB", "2048")) * 1024 * 1024)


def upload_limit_label() -> str:
    """The limit as people read it: "2 GB", "500 MB"."""
    mb = max_upload_bytes() / (1024 * 1024)
    return f"{mb / 1024:g} GB" if mb >= 1024 else f"{mb:g} MB"


def max_media_seconds() -> float:
    return float(os.getenv("CONTINUUM_MAX_MEDIA_MINUTES", "240")) * 60


def ffmpeg_timeout_seconds() -> float:
    """Long enough to convert the longest allowed recording."""
    return max(300.0, max_media_seconds())


def input_args(path: str, media_format: Optional[str] = None) -> List[str]:
    """ffmpeg arguments that open an uploaded file safely. Put them where
    `-i path` would go (after any -ss seek)."""
    args = ["-protocol_whitelist", "file", "-format_whitelist", ",".join(ALLOWED_DEMUXERS)]
    if media_format in ALLOWED_DEMUXERS:
        args += ["-f", media_format]          # no re-detection as some other format
    return args + ["-i", path]


async def save_upload(upload) -> str:
    """Stream a FastAPI UploadFile to a temporary file, refusing it once it
    passes the size limit. Returns the path; the caller deletes it."""
    limit = max_upload_bytes()
    fd, path = tempfile.mkstemp(suffix=".upload")
    written = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await upload.read(CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise MediaRejected(413, f"The file is larger than {upload_limit_label()}. "
                                             "Upload a shorter recording or the audio track only.")
                out.write(chunk)
        if written == 0:
            raise MediaRejected(400, "The uploaded file is empty.")
    except BaseException:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def probe(path: str) -> dict:
    """What the file is, decided by ffprobe with only the allowed demuxers
    enabled. Raises MediaRejected for anything that is not a supported
    recording with an audio track, or that is too long."""
    cmd = ["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-format_whitelist", ",".join(ALLOWED_DEMUXERS),
           "-show_entries", "format=format_name,duration:stream=codec_type", "-of", "json", path]
    unsupported = f"This file isn't a supported recording. Upload {SUPPORTED_DESCRIPTION} audio or video."
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=PROBE_TIMEOUT_SECONDS, check=False)
    except FileNotFoundError as exc:
        raise MediaRejected(503, "ffprobe is not installed on the server, so recordings cannot be checked.") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaRejected(422, unsupported) from exc
    if out.returncode != 0:
        raise MediaRejected(422, unsupported)
    try:
        data = json.loads(out.stdout.decode("utf-8", "replace") or "{}")
    except ValueError as exc:
        raise MediaRejected(422, unsupported) from exc
    fmt = data.get("format") or {}
    demuxer = str(fmt.get("format_name", "")).split(",")[0]
    if demuxer not in ALLOWED_DEMUXERS:
        raise MediaRejected(422, unsupported)
    streams = data.get("streams") or []
    if not any(s.get("codec_type") == "audio" for s in streams):
        raise MediaRejected(422, "This file has no audio track, so there is nothing to transcribe.")
    try:
        duration = float(fmt.get("duration")) if fmt.get("duration") not in (None, "N/A") else None
    except (TypeError, ValueError):
        duration = None
    limit = max_media_seconds()
    if duration is not None and duration > limit:
        raise MediaRejected(422, f"This recording is {duration / 60:.0f} minutes long; the limit is "
                                 f"{limit / 60:.0f} minutes. Split it into separate KT sessions.")
    return {"format": demuxer, "duration": duration,
            "has_video": any(s.get("codec_type") == "video" for s in streams)}


def extract_audio(path: str, media_format: Optional[str], out_path: str) -> str:
    """16 kHz mono WAV for transcription. Recordings without a known
    duration (some browser WebM files) are cut at the length limit."""
    cmd = (["ffmpeg", "-v", "error", "-nostdin", "-y"] + input_args(path, media_format)
           + ["-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "-t", str(int(max_media_seconds())), out_path])
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=ffmpeg_timeout_seconds(),
                       check=True)
    except FileNotFoundError as exc:
        raise MediaRejected(503, "ffmpeg is not installed on the server.") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise MediaRejected(422, "No audio could be read from this file.") from exc
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise MediaRejected(422, "No audio could be read from this file.")
    return out_path
