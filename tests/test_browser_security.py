"""Browser security headers on every response, and the API reference for
signed-in users only (enterprise checklist; browser_security.py). Before,
no response set a Content-Security-Policy, HSTS, X-Frame-Options or
Referrer-Policy, and /docs and /openapi.json were public."""
import base64
import hashlib
import os
import sys
from html.parser import HTMLParser

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auth  # noqa: E402
import browser_security  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient
    import main

    monkeypatch.setenv("CONTINUUM_AUTH", "required")
    return TestClient(main.app)


class _InlineScripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts, self._inline = [], None

    def handle_starttag(self, tag, attrs):
        if tag == "script" and "src" not in dict(attrs):
            self._inline = ""

    def handle_data(self, data):
        if self._inline is not None:
            self._inline += data

    def handle_endtag(self, tag):
        if tag == "script" and self._inline is not None:
            self.scripts.append(self._inline)
            self._inline = None


def _script_src(policy):
    return next(d for d in policy.split(";") if d.strip().startswith("script-src"))


def test_every_response_carries_the_headers(client):
    for resp in (client.get("/healthz"), client.get("/"), client.get("/jobs"), client.get("/static/index.html")):
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["Referrer-Policy"] == "same-origin"
        policy = resp.headers["Content-Security-Policy"]
        assert "frame-ancestors 'none'" in policy and "object-src 'none'" in policy
        assert "'unsafe-inline'" not in _script_src(policy)


@pytest.mark.parametrize("page", ["index.html", "login.html"])
def test_each_inline_script_of_the_pages_is_allowed_by_its_hash(page):
    # Hashed as a browser does, from the page's parsed text: a script edited
    # without a matching hash would not run.
    with open(os.path.join(browser_security.STATIC_DIR, page), encoding="utf-8", newline="") as fh:
        html = fh.read().replace("\r\n", "\n")
    parser = _InlineScripts()
    parser.feed(html)
    assert parser.scripts
    policy = browser_security.content_security_policy()
    for script in parser.scripts:
        digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode()
        assert f"'sha256-{digest}'" in _script_src(policy)


def test_hsts_only_over_https(client, monkeypatch):
    from fastapi.testclient import TestClient
    import main

    assert "Strict-Transport-Security" not in client.get("/healthz").headers
    secure = TestClient(main.app, base_url="https://testserver")
    assert secure.get("/healthz").headers["Strict-Transport-Security"] == "max-age=31536000"
    behind_proxy = client.get("/healthz", headers={"X-Forwarded-Proto": "https"})
    assert "Strict-Transport-Security" in behind_proxy.headers
    monkeypatch.setenv("CONTINUUM_HSTS_SECONDS", "0")
    assert "Strict-Transport-Security" not in secure.get("/healthz").headers


def test_the_api_reference_is_for_signed_in_users(client, monkeypatch):
    anonymous = client.get("/docs", follow_redirects=False)
    assert anonymous.status_code == 302 and anonymous.headers["location"] == "/"
    assert client.get("/openapi.json").status_code == 401
    assert client.get("/redoc").status_code == 404

    tenant = auth.create_tenant("Docs")
    key = {"Authorization": f"Bearer {auth.create_key(tenant, 'dana')}"}
    page = client.get("/docs", headers=key)
    assert page.status_code == 200 and "swagger-ui" in page.text
    assert "https://cdn.jsdelivr.net" in page.headers["Content-Security-Policy"]
    schema = client.get("/openapi.json", headers=key)
    assert schema.status_code == 200 and "/kt-from-transcript" in schema.json()["paths"]
    assert "/docs" not in schema.json()["paths"]
    assert client.get("/me", headers=key).json()["api_docs"] is True

    monkeypatch.setenv("CONTINUUM_API_DOCS", "0")
    assert client.get("/docs", headers=key).status_code == 404
    assert client.get("/openapi.json", headers=key).status_code == 404
    assert client.get("/me", headers=key).json()["api_docs"] is False
