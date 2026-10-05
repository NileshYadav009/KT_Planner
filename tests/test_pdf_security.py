"""P0-3: text from a transcript or an LLM must never be interpreted as HTML by
the PDF renderer, and the renderer must never read files or call URLs.

Reproduced before the fix: an <img src="http://..."> in a transcript made
WeasyPrint call that address while rendering, and <a rel="attachment"
href="file:///..."> embedded a server file in the PDF."""
import http.server
import io
import os
import socketserver
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pdf_rendering
from pdf_rendering import _render_paragraph_text, html_to_pdf_bytes, render_pdf_html

pypdf = pytest.importorskip("pypdf")
pytest.importorskip("weasyprint")


def test_tags_in_paragraph_text_are_printed_not_interpreted():
    html = _render_paragraph_text('See <a rel="attachment" href="file:///etc/passwd">runbook</a> <img src="http://x/y.png">')
    assert "<a " not in html and "<img" not in html
    assert "&lt;a rel=&quot;attachment&quot;" in html


def test_markdown_formatting_still_works():
    html = _render_paragraph_text("**Rollback:** redeploy the previous task definition & verify")
    assert "<strong>Rollback:</strong>" in html
    assert "&amp; verify" in html


@pytest.mark.parametrize("text,kept", [
    ("[dashboard](https://grafana.acme.io/d/x)", True),
    ("[mail](mailto:oncall@acme.io)", True),
    ("[local](file:///etc/passwd)", False),
    ("[js](javascript:alert(1))", False),
])
def test_markdown_links_keep_only_safe_schemes(text, kept):
    html = _render_paragraph_text(text)
    assert ("href=" in html) is kept


def test_markdown_images_are_dropped():
    assert "<img" not in _render_paragraph_text("![graph](http://127.0.0.1:9/x.png)")


def _probe_server():
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(404)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, hits


def _file_attachments(pdf_bytes):
    found = []
    for page in pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages:
        for annot in page.get("/Annots") or []:
            if annot.get_object().get("/Subtype") == "/FileAttachment":
                found.append(annot)
    return found


def test_rendered_pdf_reads_no_files_and_calls_no_urls(tmp_path):
    canary = tmp_path / "secret.txt"
    canary.write_text("CANARY-local-file-contents")
    server, hits = _probe_server()
    port = server.server_address[1]
    payload = (f'The runbook is <a rel="attachment" href="file:///{canary.as_posix()}">here</a>. '
               f'Graph <img src="http://127.0.0.1:{port}/probe.png">. '
               f'![x](http://127.0.0.1:{port}/md.png)')
    sections = [{"section_id": "monitoring_observability", "section_title": "Monitoring",
                 "blocks": [{"type": "NarrativeBlock", "title": "Monitoring", "paragraphs": [payload]}]}]
    html = render_pdf_html(title="T", job_id="sec-0001", rendered_sections=sections, coverage={}, date_str="d")
    pdf = html_to_pdf_bytes(html)
    server.shutdown()
    assert _file_attachments(pdf) == []
    assert hits == []
    text = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(pdf)).pages)
    assert "rel=" in text            # the tag is shown as text
    assert "CANARY" not in text


def test_fetcher_refuses_everything_but_data_uris():
    for url in ("file:///etc/passwd", "http://169.254.169.254/latest/meta-data/", "https://example.com/x.png"):
        with pytest.raises(ValueError):
            pdf_rendering.safe_url_fetcher(url)


def test_image_block_only_embeds_captured_screenshots(tmp_path, monkeypatch):
    assets = tmp_path / "kt_assets"
    (assets / "job-1").mkdir(parents=True)
    shot = assets / "job-1" / "screen_01.jpg"
    shot.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    outside = tmp_path / "secret.txt"
    outside.write_text("server file")
    monkeypatch.setenv("KT_ASSETS_DIR", str(assets))
    ok = pdf_rendering._render_image_block({"image_path": str(shot), "caption": "c"})
    refused = pdf_rendering._render_image_block({"image_path": str(outside), "caption": "c"})
    assert "data:image/jpeg;base64," in ok
    assert "base64" not in refused and "not available" in refused


def test_export_route_uses_the_restricted_renderer(tmp_path):
    from fastapi.testclient import TestClient
    import main
    import pipeline

    canary = tmp_path / "secret.txt"
    canary.write_text("CANARY-route")
    job_id = "sec-route-0001"
    pipeline.JOB_QUEUE[job_id] = {
        "status": "completed", "coverage": {},
        "knowledge_object": {"system_name": "T", "rendered_sections": [{
            "section_id": "s", "section_title": "S",
            "blocks": [{"type": "NarrativeBlock", "title": "S",
                        "paragraphs": [f'<a rel="attachment" href="file:///{canary.as_posix()}">x</a>']}]}]},
    }
    try:
        resp = TestClient(main.app).get(f"/export/pdf/{job_id}")
    finally:
        pipeline.JOB_QUEUE.pop(job_id, None)
    assert resp.status_code == 200
    assert _file_attachments(resp.content) == []
