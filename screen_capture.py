"""
screen_capture.py
=================
Finds the useful moments of a KT giver's screen share in a recorded meeting
video and turns them into KT content: a screenshot of each dashboard that was
shown and talked about, and the URL of each useful page that was open. None
of this is in the transcript; it comes from the video frames.

It must not capture at random (the removed first version grabbed one frame
per transcript segment). A screenshot is taken only when all of these hold:

1. The screen was held still for a while (a "stable screen"), not flashed
   past while switching tabs.
2. OCR shows it is a readable screen (not the camera, not a blank slide).
3. Its content identifies it as a dashboard: a monitoring tool name and/or
   dashboard vocabulary (panels, time ranges, p95, error rate, time-axis
   labels).
4. The speaker was talking about it (pointing language such as "as you can
   see on this dashboard", or naming the tool or the metrics on screen)
   while it was visible, or it was held on screen long enough to be clearly
   deliberate.
5. It shows nothing that looks like a secret (passwords, tokens, keys). Such
   a screen is never saved.
6. It is not a repeat of a dashboard already captured.

Links are recorded for every useful page that was held on screen (dashboards,
runbooks/docs, pipelines, repositories, cloud consoles); meeting, chat, mail
and search pages are ignored, and sensitive query parameters are removed.

Pipeline: ffmpeg samples small grey frames at SCREEN_SAMPLE_FPS -> stable
screens -> one sharp full-size frame per stable screen -> OCR (RapidOCR,
CPU) -> classify -> align with the transcript -> select -> de-duplicate ->
save JPEGs. Every skipped screen is counted with its reason.

Settings: SCREEN_CAPTURE (on/off, default on), SCREEN_CAPTURE_MAX (10),
SCREEN_SAMPLE_FPS (1), SCREEN_MIN_DWELL_SECONDS (3), SCREEN_MAX_OCR (60).
"""
from __future__ import annotations

import io
import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

LOGGER = logging.getLogger(__name__)

SAMPLE_W, SAMPLE_H = 160, 90


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def enabled() -> bool:
    return os.getenv("SCREEN_CAPTURE", "on").strip().lower() not in ("0", "off", "false", "no")


# --------------------------------------------------------------------------
# Content vocabulary
# --------------------------------------------------------------------------

_DASHBOARD_TOOLS = re.compile(
    r"\b(grafana|datadog|kibana|cloudwatch|new relic|newrelic|dynatrace|prometheus|azure monitor|"
    r"application insights|app insights|splunk|honeycomb|sentry|opensearch dashboards|elastic|"
    r"signoz|chronograf|thanos|victoriametrics|appdynamics|lightstep|instana|stackdriver|"
    r"cloud monitoring|metrics explorer|log analytics)\b",
    re.IGNORECASE,
)
_DASHBOARD_TERMS = [
    r"\bdashboards?\b", r"\bpanels?\b", r"\blatency\b", r"\bp(?:50|90|95|99)\b", r"\berror rate\b",
    r"\bthroughput\b", r"\brequests?\b", r"\brps\b|req/s|requests/s", r"\bcpu\b", r"\bmemory\b",
    r"\butili[sz]ation\b", r"\bsaturation\b", r"\buptime\b", r"\bavailability\b", r"\bslo\b|\bsli\b",
    r"\bapdex\b", r"\blast\s+\d+\s*(?:m|min|minutes?|h|hours?|d|days?)\b", r"\brefresh\b",
    r"\b(?:5xx|4xx|errors?)\b", r"\bqueue depth\b|\blag\b", r"\bheap\b|\bgc\b",
]
_DASHBOARD_TERM_RES = [re.compile(t, re.IGNORECASE) for t in _DASHBOARD_TERMS]
_TIME_TICK_RE = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")
_UNIT_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|s|%|k|m|gb|mb|rps|/s)\b", re.IGNORECASE)

# Page types, by URL host/path and on-screen words. Order matters: first match wins.
_PAGE_TYPES: List[Tuple[str, re.Pattern, re.Pattern]] = [
    ("meeting", re.compile(r"teams\.microsoft|zoom\.us|meet\.google|webex|gotomeeting", re.I),
     re.compile(r"\b(?:mute|unmute|leave meeting|participants|raise hand|start video|stop video)\b", re.I)),
    ("chat", re.compile(r"slack\.com|discord|whatsapp|web\.telegram", re.I),
     re.compile(r"\b(?:direct messages|threads|mentions & reactions|channels)\b", re.I)),
    ("mail", re.compile(r"outlook\.(?:office|live)|mail\.google|gmail", re.I),
     re.compile(r"\b(?:inbox|compose|sent items|drafts)\b", re.I)),
    ("search", re.compile(r"google\.[a-z.]+/search|bing\.com/search|duckduckgo", re.I), re.compile(r"(?!x)x")),
    ("dashboard", re.compile(r"grafana|datadoghq|kibana|cloudwatch|newrelic|dynatrace|/d/|dashboards?", re.I),
     re.compile(r"(?!x)x")),
    ("pipeline", re.compile(r"jenkins|/actions|/pipelines?|circleci|argocd|argo-cd|buildkite|/-/jobs", re.I),
     re.compile(r"\b(?:pipeline|build #?\d+|stage|deploy(?:ment)?s?|workflow run|passed|failed)\b", re.I)),
    ("docs", re.compile(r"confluence|atlassian\.net/wiki|/wiki|notion\.so|sharepoint|readme|docs\.|/docs|runbook", re.I),
     re.compile(r"\b(?:runbook|playbook|on-?call guide|how to|troubleshooting|architecture|overview)\b", re.I)),
    ("repository", re.compile(r"github\.com|gitlab|bitbucket|dev\.azure\.com", re.I),
     re.compile(r"\b(?:pull requests?|merge requests?|commits?|branches|readme\.md)\b", re.I)),
    ("console", re.compile(r"console\.aws\.amazon|portal\.azure|console\.cloud\.google|cloud\.digitalocean", re.I),
     re.compile(r"(?!x)x")),
    ("diagram", re.compile(r"lucid\.app|lucidchart|draw\.io|diagrams\.net|miro\.com|excalidraw", re.I),
     re.compile(r"(?!x)x")),
]
_USEFUL_LINK_TYPES = {"dashboard", "pipeline", "docs", "repository", "console", "diagram", "other"}
# Shell prompts and commands. Not "#": chat channels ("# payments-oncall")
# start with it too.
_TERMINAL_RE = re.compile(r"(?:^|\s)(?:\$|PS [A-Z]:\\\S*>|C:\\\S*>)\s|\b(?:kubectl|helm|terraform|ssh|sudo|docker)\s", re.I)

# Anything that looks like a credential. A screen showing one is never saved.
_SECRET_RE = re.compile(
    r"(?:password|passwd|pwd|secret|api[_\- ]?key|access[_\- ]?key|token|private[_\- ]?key)\s*[:=]\s*\S{4,}"
    r"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|\bgsk_[A-Za-z0-9]{10,}|\bsk-[A-Za-z0-9]{16,}"
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}|\bxox[abpr]-[A-Za-z0-9-]{10,}|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    re.IGNORECASE,
)
_SENSITIVE_QUERY_KEYS = re.compile(r"token|key|secret|sig|signature|password|pwd|auth|code|session|credential", re.I)

# What the speaker says while a screen is visible.
_POINTING_RE = re.compile(
    r"\b(?:on (?:my|the|your) screen|you can see|as you can see|you'll see|you will see|if you look|take a look|"
    r"let me (?:show|share|open|pull up|bring up)|i'?m (?:sharing|showing)|here (?:is|we have|you can see|you see)|"
    r"this (?:dashboard|graph|panel|chart|board|page|view|link|url|runbook|pipeline|console)|"
    r"that (?:dashboard|graph|panel|chart|board|page|link)|bookmark|the (?:link|url) (?:is|for)|save this link|"
    r"go to|open (?:the|this))\b",
    re.IGNORECASE,
)
_MONITORING_SPEECH_RE = re.compile(
    r"\b(?:dashboards?|metrics?|latency|error rates?|alerts?|monitor(?:ing)?|graphs?|panels?|p95|p99|throughput|"
    r"cpu|memory|spike|trend|queue depth)\b",
    re.IGNORECASE,
)


@dataclass
class ScreenAsset:
    kind: str                      # "dashboard" (screenshot + link) or "link"
    start: float                   # seconds on the video timeline
    end: float
    page_type: str
    title: str
    url: Optional[str] = None
    tool: Optional[str] = None
    image_path: Optional[str] = None
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    spoken_context: str = ""
    section_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _Stable:
    start: float
    end: float
    edge: float
    dhash: int
    samples: List[float]

    @property
    def dwell(self) -> float:
        return self.end - self.start


# --------------------------------------------------------------------------
# Video sampling and stable screens
# --------------------------------------------------------------------------

def has_video_stream(path: str) -> bool:
    try:
        import av
        with av.open(path) as container:
            return any(s.type == "video" and (s.frames or s.duration or s.average_rate) for s in container.streams)
    except Exception:
        return False


def _iter_gray_samples(path: str, fps: float):
    """Yield (timestamp, uint8[SAMPLE_H, SAMPLE_W]) frames sampled by ffmpeg."""
    vf = (f"fps={fps},scale={SAMPLE_W}:{SAMPLE_H}:force_original_aspect_ratio=decrease,"
          f"pad={SAMPLE_W}:{SAMPLE_H}:(ow-iw)/2:(oh-ih)/2")
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    size = SAMPLE_W * SAMPLE_H
    i = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield i / fps, np.frombuffer(buf, dtype=np.uint8).reshape(SAMPLE_H, SAMPLE_W)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


def _dhash(gray: np.ndarray) -> int:
    """64-bit difference hash: the same screen shown twice hashes within a
    few bits even after re-encoding."""
    ys = np.linspace(0, gray.shape[0] - 1, 8).astype(int)
    xs = np.linspace(0, gray.shape[1] - 1, 9).astype(int)
    small = gray[np.ix_(ys, xs)].astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _edge_density(gray: np.ndarray) -> float:
    g = gray.astype(np.int16)
    grad = np.abs(np.diff(g, axis=1))[:-1, :] + np.abs(np.diff(g, axis=0))[:, :-1]
    return float((grad > 40).mean())


def _same_screen(a: np.ndarray, b: np.ndarray, mean_limit: float = 3.0, block_limit: float = 6.0) -> bool:
    """Whole-frame AND per-region change are both small. The whole-frame
    mean alone missed a switch between two dashboards in the same theme
    (Grafana -> Kibana: mean change 0.67, but 9.15 in the region where the
    title differs; an unchanged screen measures 0.00 on both)."""
    d = np.abs(a - b)
    if float(d.mean()) > mean_limit:
        return False
    bh, bw = SAMPLE_H // 6, SAMPLE_W // 8
    blocks = d[: bh * 6, : bw * 8].reshape(6, bh, 8, bw).mean(axis=(1, 3))
    return float(blocks.max()) <= block_limit


def find_stable_screens(path: str, fps: float, min_dwell: float) -> List[_Stable]:
    """Runs of consecutive samples whose picture barely changes. Scrolling,
    switching tabs or a moving camera end a run."""
    runs: List[_Stable] = []
    prev = None
    cur: Optional[_Stable] = None
    for t, gray in _iter_gray_samples(path, fps):
        if prev is not None and cur is not None and _same_screen(gray.astype(np.int16), prev):
            cur.end = t + 1.0 / fps
            cur.samples.append(t)
            cur.edge = max(cur.edge, _edge_density(gray))
        else:
            if cur is not None:
                runs.append(cur)
            cur = _Stable(start=t, end=t + 1.0 / fps, edge=_edge_density(gray), dhash=_dhash(gray), samples=[t])
        prev = gray.astype(np.int16)
    if cur is not None:
        runs.append(cur)
    return [r for r in runs if r.dwell >= min_dwell]


def _grab_frame(path: str, t: float, max_width: int = 1600):
    from PIL import Image
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.2f}", "-i", path, "-frames:v", "1",
         "-vf", f"scale='min({max_width},iw)':-2", "-f", "image2pipe", "-vcodec", "png", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    if not out.stdout:
        return None
    return Image.open(io.BytesIO(out.stdout)).convert("RGB")


def _sharpness(img) -> float:
    g = np.asarray(img.convert("L").resize((480, 270)), dtype=np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def _best_frame(path: str, run: _Stable):
    """Sharpest of up to three frames from the middle of the run (avoids a
    frame caught mid-transition or with a moving cursor blur)."""
    inner = run.samples[len(run.samples) // 4: max(len(run.samples) * 3 // 4, len(run.samples) // 4 + 1)] or run.samples
    picks = sorted({inner[0], inner[len(inner) // 2], inner[-1]})
    best, best_t, best_s = None, None, -1.0
    for t in picks:
        img = _grab_frame(path, t + 0.25)
        if img is None:
            continue
        s = _sharpness(img)
        if s > best_s:
            best, best_t, best_s = img, t, s
    return best, best_t


# --------------------------------------------------------------------------
# OCR and understanding a screen
# --------------------------------------------------------------------------

_OCR_ENGINE = None


def _ocr(img) -> List[Tuple[str, float, float]]:
    """[(text, top_y_fraction, line_height_fraction)] for confident lines."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr import RapidOCR
        _OCR_ENGINE = RapidOCR()
    w, h = img.size
    result = _OCR_ENGINE(np.asarray(img))
    lines = []
    for text, box, score in zip(result.txts or (), result.boxes if result.boxes is not None else (), result.scores or ()):
        if float(score) < 0.6 or not str(text).strip():
            continue
        ys = [p[1] for p in box]
        lines.append((str(text).strip(), min(ys) / h, (max(ys) - min(ys)) / h))
    return lines


_URL_RE = re.compile(
    r"(?:https?\s*[:;]\s*/\s*/\s*)?(?:www\.)?[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)+(?::\d+)?(?:/[^\s<>\"'|]*)?",
    re.IGNORECASE,
)
# Without a scheme or "www.", a dotted word ("error.rate", "app.config") is
# only taken as a web address when it ends in a common top-level domain and
# has a path.
_COMMON_TLD_RE = re.compile(
    r"\.(?:com|io|net|org|dev|cloud|app|ai|co|in|uk|us|de|eu|internal|local|corp|tech|site|sh)$", re.I
)


def clean_url(raw: str) -> Optional[str]:
    """Normalise an OCR'd URL; None when it is not a plausible web address."""
    s = re.sub(r"\s+", "", raw or "")
    s = re.sub(r"^(https?)[:;]/*", r"\1://", s, flags=re.I)
    s = s.rstrip(".,;:)]}>'\"")
    explicit = bool(re.match(r"https?://|www\.", s, re.I))
    if not re.match(r"https?://", s, re.I):
        s = "https://" + s
    try:
        parts = urlsplit(s)
    except ValueError:
        return None
    host = parts.hostname or ""
    if "." not in host or not re.search(r"\.[a-z]{2,24}$", host, re.I):
        return None
    if not explicit and not (_COMMON_TLD_RE.search(host) and len(parts.path) > 1):
        return None
    if "@" in parts.netloc:  # credentials embedded in the URL
        return None
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not _SENSITIVE_QUERY_KEYS.search(k)])
    # The fragment is kept: single-page apps (Kibana, Grafana Explore) put
    # the actual view there ("#/view/payments-logs").
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, query, parts.fragment))


def _find_url(lines) -> Optional[str]:
    """The page URL: prefer the address bar (top of the screen)."""
    candidates = []
    for text, top, _ in lines:
        for m in _URL_RE.finditer(text):
            url = clean_url(m.group(0))
            if url:
                has_scheme = bool(re.match(r"\s*https?", m.group(0), re.I))
                candidates.append((0 if top < 0.15 else 1, 0 if has_scheme else 1, -len(url), url))
    return sorted(candidates)[0][3] if candidates else None


def _page_type(url: Optional[str], text: str) -> str:
    if not url and len(_TERMINAL_RE.findall(text)) >= 2:
        # Before the word lists: a shell running "kubectl rollout restart
        # deployment/..." is a terminal, not a pipeline page.
        return "terminal"
    for name, url_re, text_re in _PAGE_TYPES:
        if (url and url_re.search(url)) or text_re.search(text):
            return name
    if _TERMINAL_RE.search(text):
        return "terminal"
    return "other"


def _dashboard_score(text: str, url: Optional[str]) -> Tuple[int, List[str]]:
    score, why = 0, []
    tool = _DASHBOARD_TOOLS.search(text) or (url and _DASHBOARD_TOOLS.search(url))
    if tool:
        score += 3
        why.append(f"monitoring tool on screen ({tool.group(0)})")
    terms = {r.pattern for r in _DASHBOARD_TERM_RES if r.search(text)}
    if terms:
        score += min(len(terms), 5)
        why.append(f"{len(terms)} dashboard term(s)")
    ticks = len(_TIME_TICK_RE.findall(text))
    if ticks >= 4:
        score += 2
        why.append(f"{ticks} time-axis labels")
    if len(_UNIT_NUMBER_RE.findall(text)) >= 3:
        score += 1
        why.append("metric values with units")
    return score, why


def _title(lines, url: Optional[str]) -> str:
    """The biggest text near the top that is not browser chrome or the URL."""
    best = None
    for text, top, height in lines:
        if top > 0.35 or len(text) < 4 or (url and _URL_RE.fullmatch(text.replace(" ", ""))):
            continue
        if re.search(r"https?[:;]|www\.|\.com|\.io/", text, re.I):
            continue
        key = (height, -top)
        if best is None or key > best[0]:
            best = (key, text)
    return best[1] if best else ""


# --------------------------------------------------------------------------
# Speech alignment
# --------------------------------------------------------------------------

def _spoken_during(segments: Sequence[Dict[str, Any]], start: float, end: float,
                   before: float = 5.0, after: float = 2.0) -> str:
    """What was said while the screen was up, plus a few seconds before
    ("let me pull up the dashboard" usually precedes the switch). Kept short
    so a screen does not inherit the previous screen's narration."""
    out = []
    for seg in segments or ():
        s, e = float(seg.get("start") or 0.0), float(seg.get("end") or 0.0)
        if e >= start - before and s <= end + after:
            out.append(str(seg.get("text") or "").strip())
    return " ".join(t for t in out if t)


def _speech_score(spoken: str, tool: Optional[str], page_type: str) -> Tuple[int, List[str]]:
    score, why = 0, []
    if _POINTING_RE.search(spoken):
        score += 2
        why.append("speaker refers to the screen")
    if tool and re.search(re.escape(tool), spoken, re.I):
        score += 2
        why.append(f"speaker names {tool}")
    if page_type == "dashboard" and _MONITORING_SPEECH_RE.search(spoken):
        score += 1
        why.append("speaker talks about metrics")
    return score, why


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def leading_silence_seconds(path: str, threshold_db: str = "-50dB", min_silence: float = 0.5) -> float:
    """How much leading silence pipeline.trim_leading_trailing_silence() cut
    before transcription; transcript timestamps are shifted by this much
    relative to the video."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "info", "-i", path, "-vn", "-af", f"silencedetect=noise={threshold_db}:d={min_silence}",
             "-t", "900", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False, text=True, errors="ignore",
        ).stderr
        starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", out)]
        ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", out)]
        if starts and starts[0] <= 0.05 and ends:
            return ends[0]
    except Exception as exc:
        LOGGER.warning("Leading-silence measurement failed: %s", exc)
    return 0.0


def analyze_screen_shares(
    video_path: str,
    segments: Sequence[Dict[str, Any]],
    out_dir: str,
    *,
    transcript_offset: float = 0.0,
) -> Dict[str, Any]:
    """Find dashboards and useful links shown on screen.

    `segments` are the transcript segments (start/end/text) on the trimmed
    audio timeline; `transcript_offset` is the leading silence that was cut,
    added back to map them onto the video timeline.

    Returns {"assets": [ScreenAsset dicts], "stats": {...}}.
    """
    stats: Dict[str, Any] = {"enabled": enabled(), "video": False, "stable_screens": 0, "ocr_frames": 0,
                             "screenshots": 0, "links": 0, "skipped": {}}

    def skip(reason: str) -> None:
        stats["skipped"][reason] = stats["skipped"].get(reason, 0) + 1

    if not enabled() or not video_path or not has_video_stream(video_path):
        return {"assets": [], "stats": stats}
    stats["video"] = True

    fps = _env_float("SCREEN_SAMPLE_FPS", 1.0)
    min_dwell = _env_float("SCREEN_MIN_DWELL_SECONDS", 3.0)
    max_shots = int(_env_float("SCREEN_CAPTURE_MAX", 10))
    max_ocr = int(_env_float("SCREEN_MAX_OCR", 60))

    shifted = [dict(s, start=float(s.get("start") or 0) + transcript_offset,
                    end=float(s.get("end") or 0) + transcript_offset) for s in (segments or [])]

    runs = find_stable_screens(video_path, fps, min_dwell)
    stats["stable_screens"] = len(runs)
    # OCR is the expensive step: read the most promising screens first
    # (held longest, most edges = most text/UI), up to the cap.
    order = sorted(range(len(runs)), key=lambda i: -(runs[i].dwell * (0.5 + runs[i].edge)))
    if len(order) > max_ocr:
        for _ in order[max_ocr:]:
            skip("over the OCR budget")
        order = order[:max_ocr]

    candidates: List[Tuple[ScreenAsset, Any, int]] = []
    for i in sorted(order):
        run = runs[i]
        if run.edge < 0.01:
            skip("blank or near-uniform screen")
            continue
        img, t = _best_frame(video_path, run)
        if img is None:
            skip("frame could not be decoded")
            continue
        lines = _ocr(img)
        stats["ocr_frames"] += 1
        text = "\n".join(t_ for t_, _, _ in lines)
        url = _find_url(lines)
        if len(lines) < 6 and not url:
            skip("no readable screen content (camera view or blank slide)")
            continue
        page_type = _page_type(url, text)
        if page_type in ("meeting", "chat", "mail", "search"):
            skip(f"{page_type} window")
            continue
        sensitive = bool(_SECRET_RE.search(text))
        tool_match = _DASHBOARD_TOOLS.search(text) or (url and _DASHBOARD_TOOLS.search(url))
        tool = tool_match.group(0) if tool_match else None
        spoken = _spoken_during(shifted, run.start, run.end)
        visual, vwhy = _dashboard_score(text, url)
        speech, swhy = _speech_score(spoken, tool, page_type if visual < 5 else "dashboard")
        title = _title(lines, url) or (f"{tool} dashboard" if tool else page_type.title())

        is_dashboard = visual >= 5 and (page_type in ("dashboard", "other", "console") or bool(tool))
        deliberate = speech >= 2 or (run.dwell >= 10 and visual >= 7)
        if is_dashboard and deliberate and not sensitive:
            asset = ScreenAsset(kind="dashboard", start=run.start, end=run.end, page_type="dashboard", title=title,
                                url=url, tool=tool, score=float(visual + speech + min(run.dwell / 10, 2)),
                                reasons=vwhy + swhy + [f"held on screen {run.dwell:.0f}s"], spoken_context=spoken[:400])
            candidates.append((asset, img, run.dhash))
            continue
        if sensitive:
            skip("possible secret on screen (not saved)")
        elif is_dashboard:
            skip("dashboard shown but not discussed")
        elif page_type == "terminal":
            skip("terminal window")
        # A link needs the same evidence of intent as a screenshot: the
        # speaker referred to it, or it was held on screen for a while.
        link_deliberate = speech >= 2 or run.dwell >= 10
        if url and page_type in _USEFUL_LINK_TYPES and not link_deliberate:
            skip("link shown briefly and not discussed")
        elif url and page_type in _USEFUL_LINK_TYPES and (page_type != "other" or speech >= 2):
            asset = ScreenAsset(kind="link", start=run.start, end=run.end, page_type=page_type, title=title, url=url,
                                tool=tool, score=float(speech + min(run.dwell / 10, 2)),
                                reasons=swhy + [f"open on screen {run.dwell:.0f}s"], spoken_context=spoken[:400])
            candidates.append((asset, None, run.dhash))
        elif not is_dashboard and page_type != "terminal":
            skip("not a dashboard and no useful link")

    # De-duplicate: the same dashboard shown twice (same picture or same URL),
    # and the same link opened repeatedly. Keep the best-scoring instance.
    def url_key(u: Optional[str]) -> Optional[str]:
        if not u:
            return None
        p = urlsplit(u)
        return (p.netloc + p.path).rstrip("/").lower()

    kept: List[Tuple[ScreenAsset, Any, int]] = []
    for cand in sorted(candidates, key=lambda c: -c[0].score):
        a, _, h = cand
        dup = next((k for k in kept if (k[0].kind == a.kind == "dashboard" and _hamming(k[2], h) <= 8)
                    or (url_key(a.url) and url_key(a.url) == url_key(k[0].url))), None)
        if dup is not None:
            skip("repeat of a screen already captured")
            if a.kind == "dashboard" and dup[0].kind == "link":
                kept.remove(dup)
                kept.append(cand)
            continue
        kept.append(cand)

    dashboards = [k for k in kept if k[0].kind == "dashboard"]
    if len(dashboards) > max_shots:
        for k in sorted(dashboards, key=lambda c: c[0].score)[: len(dashboards) - max_shots]:
            kept.remove(k)
            skip("over the screenshot limit")

    os.makedirs(out_dir, exist_ok=True)
    assets: List[Dict[str, Any]] = []
    for n, (a, img, _) in enumerate(sorted(kept, key=lambda c: c[0].start), start=1):
        if a.kind == "dashboard" and img is not None:
            name = f"screen_{n:02d}.jpg"
            img.save(os.path.join(out_dir, name), "JPEG", quality=82, optimize=True)
            a.image_path = os.path.join(out_dir, name)
            stats["screenshots"] += 1
        if a.url:
            stats["links"] += 1
        assets.append(a.to_dict())
    return {"assets": assets, "stats": stats}


# --------------------------------------------------------------------------
# Placing captures in the KT document
# --------------------------------------------------------------------------

# Where a capture goes when nothing was said while it was on screen.
_DEFAULT_SECTION = {
    "dashboard": "monitoring_observability",
    "pipeline": "deployment_and_rollback",
    "repository": "deployment_and_rollback",
    "console": "architecture_reference",
    "diagram": "architecture_reference",
}
SHARED_SCREENS_SECTION_ID = "shared_screens"
SHARED_SCREENS_TITLE = "Shared Screens & Links"


def _mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def assign_sections(assets: List[Dict[str, Any]], coverage: Dict[str, Any], transcript_offset: float = 0.0) -> None:
    """Put each capture in the section being discussed while it was on
    screen: the most common section among the sentences spoken in that
    window (transcript timeline = video timeline - offset). Falls back to
    the page type when nothing was said."""
    timed: List[Tuple[float, float, str]] = []
    for sid, cov in (coverage or {}).items():
        for s in (cov or {}).get("sentences") or []:
            if isinstance(s, dict) and s.get("start") is not None:
                timed.append((float(s.get("start") or 0), float(s.get("end") or s.get("start") or 0), sid))
    for a in assets:
        lo, hi = a["start"] - transcript_offset - 5.0, a["end"] - transcript_offset + 2.0
        counts: Dict[str, int] = {}
        for s, e, sid in timed:
            if e >= lo and s <= hi:
                counts[sid] = counts.get(sid, 0) + 1
        if a["page_type"] == "docs" and re.search(r"runbook|playbook|troubleshoot", a.get("title") or "", re.I):
            # A runbook is operational whatever sentence introduced it ("this
            # is the runbook in Confluence" reads as documentation).
            a["section_id"] = "common_failures"
        elif counts:
            a["section_id"] = max(counts, key=lambda k: (counts[k], k == _DEFAULT_SECTION.get(a["page_type"])))
        else:
            a["section_id"] = _DEFAULT_SECTION.get(a["page_type"], "architecture_reference")


def asset_url_path(job_id: str, image_path: str) -> str:
    return f"/kt-assets/{job_id}/{os.path.basename(image_path)}"


def attach_to_rendered_sections(rendered_sections: List[Dict[str, Any]], assets: List[Dict[str, Any]],
                                job_id: str) -> None:
    """Append an image block per captured dashboard and a link list to the
    section each capture belongs to (an appendix section when that section
    is not in the document)."""
    by_section: Dict[str, List[Dict[str, Any]]] = {}
    for a in assets:
        by_section.setdefault(a.get("section_id") or SHARED_SCREENS_SECTION_ID, []).append(a)
    index = {s.get("section_id"): s for s in rendered_sections}
    for sid, items in by_section.items():
        target = index.get(sid)
        if target is None:
            target = index.get(SHARED_SCREENS_SECTION_ID)
            if target is None:
                target = {"section_id": SHARED_SCREENS_SECTION_ID, "section_title": SHARED_SCREENS_TITLE, "blocks": []}
                rendered_sections.append(target)
                index[SHARED_SCREENS_SECTION_ID] = target
        blocks = target.setdefault("blocks", [])
        for a in items:
            if a["kind"] == "dashboard" and a.get("image_path"):
                tool = f"{a['tool'].title()} " if a.get("tool") else ""
                blocks.append({
                    "type": "ImageBlock",
                    "title": f"Shown on screen: {a['title']}",
                    "image_path": a["image_path"],
                    "src": asset_url_path(job_id, a["image_path"]),
                    "caption": f"{tool}dashboard shown at {_mmss(a['start'])} in the session recording.",
                    "url": a.get("url"),
                })
        links = [a for a in items if a.get("url")]
        if links:
            blocks.append({
                "type": "ChecklistBlock",
                "title": "Links shown on screen",
                "items": [f"{a['title']}: {a['url']} (at {_mmss(a['start'])})" for a in links],
            })


def ui_payload(assets: List[Dict[str, Any]], job_id: str, section_titles: Dict[str, str]) -> Dict[str, Any]:
    shots, links = [], []
    for a in assets:
        entry = {"title": a["title"], "url": a.get("url"), "time": _mmss(a["start"]),
                 "section": section_titles.get(a.get("section_id") or "", a.get("section_id")),
                 "page_type": a["page_type"]}
        if a["kind"] == "dashboard" and a.get("image_path"):
            shots.append(dict(entry, src=asset_url_path(job_id, a["image_path"])))
        if a.get("url"):
            links.append(entry)
    return {"screenshots": shots, "links": links}
