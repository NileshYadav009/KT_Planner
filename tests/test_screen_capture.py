"""Screen-share capture: dashboards and links are taken only when they were
held on screen, recognisably useful and discussed; never from the camera,
chat windows or a screen showing a secret."""
import os
import subprocess
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import screen_capture as sc


@pytest.mark.parametrize("raw,expected", [
    ("https://grafana.acme.io/d/abc/payments?orgId=1", "https://grafana.acme.io/d/abc/payments?orgId=1"),
    ("https;//acme.atlassian.net/wiki/spaces/OPS", "https://acme.atlassian.net/wiki/spaces/OPS"),
    ("grafana.acme.io/d/abc", "https://grafana.acme.io/d/abc"),
    ("https://kibana.acme.io/app/dashboards#/view/logs", "https://kibana.acme.io/app/dashboards#/view/logs"),
    ("https://x.io/callback?code=123&page=2", "https://x.io/callback?page=2"),  # secret-looking param removed
    ("https://user:pw@host.com/x", None),                                     # credentials in the URL
    ("error.rate", None),
    ("app.config/x", None),
    ("Payments API - Latency", None),
])
def test_clean_url(raw, expected):
    assert sc.clean_url(raw) == expected


@pytest.mark.parametrize("url,text,expected", [
    (None, "PS C:\\ops> kubectl get pods\nPS C:\\ops> kubectl rollout restart deployment/x", "terminal"),
    (None, "Threads\nMentions & reactions\n# payments-oncall\n# general", "chat"),
    ("https://acme.atlassian.net/wiki/spaces/OPS/pages/1/Runbook", "Runbook", "docs"),
    ("https://grafana.acme.io/d/abc/x", "p95 latency", "dashboard"),
    ("https://teams.microsoft.com/l/meetup", "", "meeting"),
    ("https://github.com/acme/payments", "Pull requests", "repository"),
])
def test_page_type(url, text, expected):
    assert sc._page_type(url, text) == expected


def test_dashboard_scoring_separates_dashboards_from_prose():
    dash = "Grafana / Payments API\nLast 6 hours\np95 latency (ms)\nError rate %\n10:00 11:00 12:00 13:00 14:00\n234 ms 1.7 % 3.4 %"
    prose = "Payments Consumer Runbook\n1. Check the queue\n2. Restart the consumer\nEscalate to the on-call lead"
    assert sc._dashboard_score(dash, None)[0] >= 7
    assert sc._dashboard_score(prose, None)[0] < 5


@pytest.mark.parametrize("text,secret", [
    ("export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG", True),
    ("key id AKIAIOSFODNN7EXAMPLE", True),
    ("password: hunter22", True),
    ("-----BEGIN RSA PRIVATE KEY-----", True),
    ("Use the password reset page if you are locked out.", False),
    ("Rotate the token every 90 days.", False),
])
def test_secret_detection(text, secret):
    assert bool(sc._SECRET_RE.search(text)) is secret


def test_speech_must_refer_to_the_screen():
    assert sc._speech_score("Here you can see the Grafana dashboard for payments.", "grafana", "dashboard")[0] >= 4
    assert sc._speech_score("Any questions so far?", "kibana", "dashboard")[0] < 2


def test_same_screen_detects_a_small_region_change():
    base = np.full((sc.SAMPLE_H, sc.SAMPLE_W), 30, np.int16)
    other = base.copy()
    other[2:12, 10:60] = 220  # a different dashboard title in one region
    noise = base + np.random.default_rng(0).integers(-2, 3, base.shape)
    assert sc._same_screen(base, base)
    assert sc._same_screen(base, noise)
    assert not sc._same_screen(base, other)


def _asset(kind, page_type, start, title="T", url="https://grafana.acme.io/d/x", image="x.jpg"):
    return {"kind": kind, "page_type": page_type, "start": start, "end": start + 10, "title": title,
            "url": url, "tool": "grafana" if page_type == "dashboard" else None,
            "image_path": image if kind == "dashboard" else None, "section_id": None}


def test_assign_sections_follows_the_discussion_with_fallbacks():
    coverage = {
        "monitoring_observability": {"sentences": [{"text": "a", "start": 20.0, "end": 24.0}]},
        "deployment_and_rollback": {"sentences": [{"text": "b", "start": 100.0, "end": 104.0}]},
    }
    dash = _asset("dashboard", "dashboard", 26.0)        # video time; offset 6 s -> transcript 20 s
    pipe = _asset("link", "pipeline", 500.0, url="https://ci.acme.io/job/x")
    runbook = _asset("link", "docs", 106.0, title="Payments Runbook", url="https://wiki.acme.io/runbook")
    sc.assign_sections([dash, pipe, runbook], coverage, transcript_offset=6.0)
    assert dash["section_id"] == "monitoring_observability"
    assert pipe["section_id"] == "deployment_and_rollback"   # nothing said: by page type
    assert runbook["section_id"] == "common_failures"          # runbook title wins


def test_attach_blocks_and_ui_payload():
    rendered = [{"section_id": "monitoring_observability", "section_title": "Monitoring", "blocks": []}]
    dash = dict(_asset("dashboard", "dashboard", 14.0, title="Payments latency"), section_id="monitoring_observability")
    link = dict(_asset("link", "docs", 58.0, title="Runbook", url="https://wiki.acme.io/r"), section_id="not_in_document")
    sc.attach_to_rendered_sections(rendered, [dash, link], "job-1234-5678")
    blocks = rendered[0]["blocks"]
    assert blocks[0]["type"] == "ImageBlock" and blocks[0]["src"] == "/kt-assets/job-1234-5678/x.jpg"
    assert "00:14" in blocks[0]["caption"]
    assert rendered[-1]["section_id"] == sc.SHARED_SCREENS_SECTION_ID  # appendix for an unknown section
    view = sc.ui_payload([dash, link], "job-1234-5678", {"monitoring_observability": "Monitoring"})
    assert len(view["screenshots"]) == 1 and len(view["links"]) == 2


def test_leading_silence_offset(tmp_path):
    sr = 16000
    pcm = np.concatenate([np.zeros(3 * sr), (np.sin(np.arange(2 * sr) / 5) * 8000)]).astype(np.int16)
    path = str(tmp_path / "a.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
    assert sc.leading_silence_seconds(path) == pytest.approx(3.0, abs=0.3)


def test_audio_only_upload_is_a_no_op(tmp_path):
    path = str(tmp_path / "a.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(np.zeros(16000, np.int16).tobytes())
    result = sc.analyze_screen_shares(path, [], str(tmp_path / "out"))
    assert result["assets"] == [] and result["stats"]["video"] is False


# --------------------------------------------------------------------------
# Integration: a small real video through the whole analysis
# --------------------------------------------------------------------------

def _draw_dashboard(title, url):
    from PIL import Image, ImageDraw, ImageFont
    try:
        f, fb = ImageFont.truetype("arial.ttf", 16), ImageFont.truetype("arialbd.ttf", 22)
    except OSError:
        f = fb = ImageFont.load_default()
    img = Image.new("RGB", (960, 540), (22, 24, 29))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 960, 40], fill=(235, 236, 240))
    d.text((20, 10), url, fill=(20, 20, 20), font=f)
    d.text((16, 52), title, fill=(230, 230, 230), font=fb)
    d.text((780, 56), "Last 6 hours", fill=(200, 200, 200), font=f)
    for i, p in enumerate(["p95 latency (ms)", "Error rate %", "Requests / sec", "CPU usage"]):
        x, y = 16 + (i % 2) * 470, 95 + (i // 2) * 220
        d.rectangle([x, y, x + 455, y + 205], outline=(60, 62, 70), fill=(30, 33, 39))
        d.text((x + 8, y + 6), p, fill=(220, 220, 220), font=f)
        d.line([(x + 15 + k * 24, y + 160 - ((k * 37 + i * 29) % 110)) for k in range(18)], fill=(115, 191, 105), width=3)
        for k, t in enumerate(["10:00", "11:00", "12:00", "13:00"]):
            d.text((x + 15 + k * 110, y + 182), t, fill=(150, 150, 150), font=f)
    return img


def _draw_terminal():
    from PIL import Image, ImageDraw, ImageFont
    try:
        f = ImageFont.truetype("consola.ttf", 18)
    except OSError:
        f = ImageFont.load_default()
    img = Image.new("RGB", (960, 540), (12, 12, 12))
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(["PS C:\\ops> kubectl get pods -n payments", "payments-api-5c6b9   1/1   Running   0",
                            "PS C:\\ops> kubectl get secret db -o yaml", "password: Sup3rS3cretValue",
                            "export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCY",
                            "PS C:\\ops> kubectl rollout restart deployment/payments-api", "deployment restarted"]):
        d.text((20, 30 + i * 34), ln, fill=(200, 230, 200), font=f)
    return img


@pytest.fixture(scope="module")
def meeting_video(tmp_path_factory):
    av = pytest.importorskip("av")
    pytest.importorskip("rapidocr")
    path = str(tmp_path_factory.mktemp("video") / "meeting.mp4")
    dash = _draw_dashboard("Payments API - Latency", "https://grafana.acme-pay.io/d/payments/payments-latency")
    scenes = [(0, 8, dash), (8, 14, _draw_terminal()), (14, 20, dash)]
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=4)
        s.width, s.height, s.pix_fmt = 960, 540, "yuv420p"
        for i in range(20 * 4):
            img = next(sc_ for a, b, sc_ in scenes if a <= i / 4 < b)
            for pkt in s.encode(av.VideoFrame.from_image(img)):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return path


def test_meeting_video_end_to_end(meeting_video, tmp_path):
    segments = [
        {"start": 1.0, "end": 6.0, "text": "Here you can see the Grafana dashboard with the p95 latency."},
        {"start": 9.0, "end": 12.0, "text": "Let me show the secret from the terminal."},
        {"start": 15.0, "end": 18.0, "text": "Back on this dashboard, notice the spike."},
    ]
    result = sc.analyze_screen_shares(meeting_video, segments, str(tmp_path / "assets"))
    shots = [a for a in result["assets"] if a["kind"] == "dashboard"]
    assert len(shots) == 1, result
    assert shots[0]["url"] == "https://grafana.acme-pay.io/d/payments/payments-latency"
    assert os.path.isfile(shots[0]["image_path"])
    skipped = result["stats"]["skipped"]
    assert skipped.get("possible secret on screen (not saved)") == 1
    assert skipped.get("repeat of a screen already captured") == 1
    assert len(os.listdir(tmp_path / "assets")) == 1   # nothing else written to disk
