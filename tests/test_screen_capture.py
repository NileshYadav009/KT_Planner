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


def test_a_screenshot_goes_to_the_section_its_screen_is_about():
    # Its heading names the topic.
    assert sc.screen_topic_section("Monitoring", []) == "monitoring_observability"
    assert sc.screen_topic_section("Alerts", []) == "monitoring_observability"
    assert sc.screen_topic_section("Deployments", []) == "deployment_and_rollback"
    assert sc.screen_topic_section("Security posture", []) == "security_controls"
    assert sc.screen_topic_section("Cloud cost", []) == "cost_optimization"
    assert sc.screen_topic_section("Kubernetes·relay-prod-01", []) == "architecture_reference"
    assert sc.screen_topic_section("Platform overview", []) == "system_overview"
    # No telling heading: the contents decide when they clearly lean one way.
    dora = ["Elite", "Deploy frequency", "14/day", "Lead time", "2h 10m", "Change failure", "3%", "MTTR 22min"]
    assert sc.screen_topic_section("Elite", dora) == "deployment_and_rollback"
    assert sc.screen_topic_section("Relay", ["Relay", "orders-api", "42"]) is None


def test_a_console_menu_on_every_screen_does_not_decide_the_section():
    menu = ["Overview", "Pipelines", "Deployments", "Kubernetes", "Monitoring", "Alerts", "Security", "Cost"]
    screens = [menu + ["Monitoring", "p95 latency", "Error rate"], menu + ["Cloud cost", "$2,210/mo"],
               menu + ["Deployments", "orders-api v2.4"], menu + ["Alerts", "3 active"]]
    repeated = sc._repeated_lines(screens)
    assert {"pipelines", "security", "cost"} <= repeated
    page = [ln for ln in screens[0] if ln.strip().lower() not in repeated]
    assert sc.screen_topic_section("", page) == "monitoring_observability"


def test_the_screen_topic_wins_over_what_was_said_meanwhile():
    coverage = {"architecture_reference": {"sentences": [{"text": "a", "start": 20.0, "end": 24.0}]},
                "monitoring_observability": {"sentences": []}}
    dash = dict(_asset("dashboard", "dashboard", 26.0), topic_section="monitoring_observability")
    unclear = _asset("dashboard", "dashboard", 26.0)
    sc.assign_sections([dash, unclear], coverage, transcript_offset=6.0)
    assert dash["section_id"] == "monitoring_observability"
    assert unclear["section_id"] == "architecture_reference"   # screen doesn't say: the discussion decides


def test_a_section_only_shown_on_screen_is_not_called_not_covered():
    from renderers.blocks.common import NOT_COVERED_MESSAGE
    rendered = [{"section_id": "cost_optimization", "section_title": "Cost Optimization",
                 "blocks": [{"type": "NarrativeBlock", "paragraphs": [NOT_COVERED_MESSAGE]}]}]
    shot = dict(_asset("dashboard", "dashboard", 393.0, title="Cloud cost"), section_id="cost_optimization")
    sc.attach_to_rendered_sections(rendered, [shot], "job-1234-5678")
    blocks = rendered[0]["blocks"]
    assert blocks[0]["paragraphs"] == [sc.SHOWN_NOT_DISCUSSED_MESSAGE]
    assert blocks[1]["type"] == "ImageBlock"


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


# --------------------------------------------------------------------------
# A single-page ops console (14 Oct): 30 stable screens, 0 screenshots.
# Its menu ("Pipelines", "Deployments") typed every view as a pipeline page,
# every view shared one URL so all but one were "repeats", and describing a
# view without "as you can see" did not count as discussing it.
# --------------------------------------------------------------------------

def test_describing_what_is_on_screen_counts_as_discussing_it():
    screen = "Pipelines\nRuns today 64\nSuccess rate 92%\nMedian duration 6m 12s\nBlocked by policy 2\nRepositories 7"
    said = "We have pipelines running so today we ran 64 pipelines across 7 repositories. The success rate, the median duration, the block by policy pipelines are 2."
    score, why = sc._speech_score(said, None, "dashboard", screen)
    assert score >= 2 and any("describes" in w for w in why)
    assert sc._speech_score("Any questions so far?", None, "dashboard", screen)[0] < 2


def test_console_status_views_score_as_operational_screens():
    k8s = ("Kubernetes relay-prod-01\nNodes ready 6/6\nPods running 84\nCluster CPU 47%\nCluster memory 68%\n"
           "Nodes: CPU and memory utilisation\nAutoscaling (HPA) replicas min/max, CPU vs target 70%")
    cost = "Cloud cost\nMonth to date $18,420\nForecast $23,150\nPotential savings $2,210/mo\nRecommendations"
    security = ("Security posture\nOpen vulnerabilities\n2 critical 9 high 34 medium\nCompliance score 86%\n"
                "Hardcoded secrets 0\nFindings by scanner: image, dependency, SAST")
    for text in (k8s, cost, security):
        assert sc._dashboard_score(text, None)[0] >= 5, text


def test_the_browser_bars_are_cropped_and_never_give_the_title():
    lines = [("Netflix x  YouTube x  (1) WhatsApp", 0.01, 0.02), ("claude.ai/artifact/abc123", 0.06, 0.02),
             ("All Bookmarks", 0.105, 0.015), ("Share", 0.15, 0.03), ("Kubernetes - relay-prod-01", 0.2, 0.025),
             ("22min", 0.22, 0.05)]
    chrome = sc._browser_chrome_bottom(lines, "https://claude.ai/artifact/abc123")
    assert 0.12 <= chrome <= 0.14
    assert sc._browser_chrome_bottom(lines, None) == 0.0
    assert sc._title([ln for ln in lines if ln[1] >= chrome], "https://claude.ai/artifact/abc123") == "Kubernetes - relay-prod-01"


def _draw_console_view(view, url):
    """One view of a single-page console: same URL and menu, different content."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        f, fb = ImageFont.truetype("arial.ttf", 16), ImageFont.truetype("arialbd.ttf", 22)
    except OSError:
        f = fb = ImageFont.load_default()
    img = Image.new("RGB", (960, 540), (22, 24, 29))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 960, 40], fill=(235, 236, 240))
    d.text((20, 10), url, fill=(20, 20, 20), font=f)
    for i, item in enumerate(["Overview", "Pipelines", "Deployments", "Kubernetes", "Cost"]):
        d.text((14, 70 + i * 34), item, fill=(200, 200, 200), font=f)
    if view == "k8s":
        d.text((180, 52), "Kubernetes relay-prod-01", fill=(230, 230, 230), font=fb)
        rows = ["Nodes ready 6/6", "Pods running 84", "Cluster CPU 47%", "Cluster memory 68%", "Autoscaling HPA replicas"]
        for i, r in enumerate(rows):
            d.rectangle([180, 100 + i * 80, 940, 170 + i * 80], outline=(60, 62, 70), fill=(30, 33, 39))
            d.text((196, 120 + i * 80), r, fill=(220, 220, 220), font=fb)
    else:
        d.text((180, 52), "Cloud cost", fill=(230, 230, 230), font=fb)
        for i, (k, v) in enumerate([("Month to date", "$18,420"), ("Forecast", "$23,150"), ("Savings", "$2,210/mo")]):
            x = 180 + i * 255
            d.rectangle([x, 100, x + 240, 220], outline=(60, 62, 70), fill=(30, 33, 39))
            d.text((x + 12, 112), k, fill=(200, 200, 200), font=f)
            d.text((x + 12, 150), v, fill=(240, 240, 240), font=fb)
        for i in range(6):
            h = 40 + (i * 37) % 160
            d.rectangle([200 + i * 120, 500 - h, 280 + i * 120, 500], fill=(80, 140, 220))
        d.text((200, 260), "Recommendations: rightsize 6 instances, stop dev outside hours", fill=(220, 220, 220), font=f)
    return img


def test_two_views_of_one_console_url_are_both_captured(tmp_path):
    av = pytest.importorskip("av")
    pytest.importorskip("rapidocr")
    url = "https://ops.acme.io/console"
    path = str(tmp_path / "console.mp4")
    scenes = [(0, 8, _draw_console_view("k8s", url)), (8, 16, _draw_console_view("cost", url))]
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=4)
        s.width, s.height, s.pix_fmt = 960, 540, "yuv420p"
        for i in range(16 * 4):
            img = next(sc_ for a, b, sc_ in scenes if a <= i / 4 < b)
            for pkt in s.encode(av.VideoFrame.from_image(img)):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    segments = [
        {"start": 1.0, "end": 7.0, "text": "The cluster has six nodes ready and 84 pods running, cluster CPU at 47 percent."},
        {"start": 9.0, "end": 15.0, "text": "Cloud cost month to date is 18,420, the forecast is 23,150, savings from the recommendations."},
    ]
    result = sc.analyze_screen_shares(path, segments, str(tmp_path / "assets"))
    shots = [a for a in result["assets"] if a["kind"] == "dashboard"]
    assert len(shots) == 2, result["stats"]
    assert result["stats"]["links"] == 1
    from PIL import Image
    assert all(Image.open(a["image_path"]).height < 540 for a in shots)   # address bar cropped off


def test_a_url_shown_on_every_tab_is_listed_once_and_sourced_by_the_screen():
    # The user's recordings (24 Oct): a single-page console showed one URL on
    # ten tabs, and the document listed it ten times, each "no source".
    from knowledge.evidence import block_units

    url = "https://console.example.io/app"
    rendered = [{"section_id": sid, "section_title": sid, "blocks": []}
                for sid in ("monitoring_observability", "cost_optimization")]
    shots = [dict(_asset("dashboard", "dashboard", 14.0, title="Alerts", url=url), section_id="monitoring_observability"),
             dict(_asset("dashboard", "dashboard", 90.0, title="Cloud cost", url=url), section_id="cost_optimization")]
    sc.attach_to_rendered_sections(rendered, shots, "job-1234-5678")
    lists = [b for s in rendered for b in s["blocks"] if b.get("title") == "Links shown on screen"]
    assert [b["items"] for b in lists] == [[f"Alerts: {url} (at 00:14)"]]
    assert block_units(lists[0]) is None
