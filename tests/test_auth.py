"""P0-1: every job route requires a key, and a tenant only ever sees its own
jobs. Before, all routes were public and /status returned the transcript to
anyone with a job id."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import auth


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import main
    import pipeline

    monkeypatch.setenv("CONTINUUM_AUTH", "required")
    monkeypatch.setattr(pipeline, "KT_ASSETS_DIR", str(tmp_path / "assets"))
    monkeypatch.setattr(pipeline, "run_kt_pipeline", lambda *a, **k: None)   # nothing heavy in the background
    return TestClient(main.app)


@pytest.fixture
def tenants():
    a = auth.create_tenant("Acme")
    b = auth.create_tenant("Globex")
    return {"a": (a, auth.create_key(a, "alice")), "b": (b, auth.create_key(b, "bob"))}


def _h(key):
    return {"Authorization": f"Bearer {key}"}


def _job(pipeline, job_id, tenant_id):
    pipeline.JOB_QUEUE[job_id] = {
        "status": "completed", "tenant_id": tenant_id, "transcript": "SECRET TRANSCRIPT", "coverage": {
            "monitoring_observability": {"title": "M", "content": ["Datadog."], "sentences": [{"text": "Datadog."}]}},
        "screenshots": [{"src": f"/kt-assets/{job_id}/screen_01.jpg"}],
        "knowledge_object": {"system_name": "T", "rendered_sections": [
            {"section_id": "s", "section_title": "S", "blocks": [{"type": "NarrativeBlock", "title": "S", "paragraphs": ["x"]}]}]},
    }


@pytest.mark.parametrize("method,path,body", [
    ("get", "/status/any-job-0001", None),
    ("get", "/schema/any-job-0001", None),
    ("get", "/export/pdf/any-job-0001", None),
    ("get", "/jobs", None),
    ("post", "/kt-from-transcript", {"transcript": "x"}),
    ("post", "/feedback", {"job_id": "x", "sentence_id": 0, "corrected_classification": "danger_zones"}),
    ("post", "/semantic-placement", {"transcript": "x"}),
])
def test_every_job_route_requires_a_key(client, method, path, body):
    resp = getattr(client, method)(path, json=body) if body is not None else getattr(client, method)(path)
    assert resp.status_code == 401
    bad = getattr(client, method)(path, json=body, headers=_h("ckt_not-a-real-key")) if body is not None \
        else getattr(client, method)(path, headers=_h("ckt_not-a-real-key"))
    assert bad.status_code == 401


def test_public_routes_stay_public(client):
    assert client.get("/schema").status_code == 200
    assert client.get("/healthz").status_code in (200, 503)


def test_a_tenant_cannot_see_another_tenants_job(client, tenants):
    import pipeline

    (a, key_a), (_, key_b) = tenants["a"], tenants["b"]
    _job(pipeline, "auth-job-0001", a)
    try:
        assert client.get("/status/auth-job-0001", headers=_h(key_a)).status_code == 200
        for path in ("/status/auth-job-0001", "/schema/auth-job-0001", "/export/pdf/auth-job-0001"):
            resp = client.get(path, headers=_h(key_b))
            assert resp.status_code == 404 and "SECRET TRANSCRIPT" not in resp.text
        resp = client.post("/feedback", headers=_h(key_b),
                           json={"job_id": "auth-job-0001", "sentence_id": 0, "corrected_classification": "danger_zones"})
        assert resp.status_code == 404
        listed_b = [j["job_id"] for j in client.get("/jobs", headers=_h(key_b)).json()["jobs"]]
        listed_a = [j["job_id"] for j in client.get("/jobs", headers=_h(key_a)).json()["jobs"]]
        assert "auth-job-0001" in listed_a and "auth-job-0001" not in listed_b
    finally:
        pipeline.JOB_STORE.delete("auth-job-0001")


def test_new_jobs_belong_to_the_callers_tenant(client, tenants):
    import pipeline

    a, key_a = tenants["a"]
    transcript = ("This KT is for TripWise, the booking backend. Bookings are written to DynamoDB. "
                  "Datadog is our monitoring tool and PagerDuty pages us. RTO is four hours. "
                  "Never delete items from the bookings table by hand. Escalate to Marco Silva.")
    resp = client.post("/kt-from-transcript", json={"transcript": transcript}, headers=_h(key_a))
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert pipeline.JOB_STORE.tenant_of(job_id) == a
    pipeline.JOB_STORE.delete(job_id)


def test_feedback_is_attributed_to_the_key_not_the_request_body(client, tenants):
    import pipeline

    a, key_a = tenants["a"]
    _job(pipeline, "auth-fb-0001", a)
    try:
        client.post("/feedback", headers=_h(key_a), json={"job_id": "auth-fb-0001", "sentence_id": 0,
                                                          "corrected_classification": "danger_zones", "user": "ceo@victim.com"})
        user = pipeline.JOB_STORE.get("auth-fb-0001")["human_feedback"][0]["user"]
        assert user == f"{a}:alice"
    finally:
        pipeline.JOB_STORE.delete("auth-fb-0001")


def test_revoked_keys_stop_working(client, tenants):
    a, _ = tenants["a"]
    key = auth.create_key(a, "temp")
    assert client.get("/jobs", headers=_h(key)).status_code == 200
    auth.revoke_key(key[:12])
    assert client.get("/jobs", headers=_h(key)).status_code == 401


def test_screenshots_need_a_signed_url(client, tenants):
    import pipeline

    a, key_a = tenants["a"]
    _job(pipeline, "auth-img-0001", a)
    shot = os.path.join(pipeline.KT_ASSETS_DIR, "auth-img-0001", "screen_01.jpg")
    os.makedirs(os.path.dirname(shot), exist_ok=True)
    open(shot, "wb").write(b"\xff\xd8\xff\xe0jpeg")
    try:
        assert client.get("/kt-assets/auth-img-0001/screen_01.jpg").status_code == 404
        signed = client.get("/status/auth-img-0001", headers=_h(key_a)).json()["screenshots"][0]["src"]
        assert "sig=" in signed
        assert client.get(signed).status_code == 200
        assert client.get(signed.replace("sig=", "sig=0")).status_code == 404
    finally:
        pipeline.JOB_STORE.delete("auth-img-0001")


# --------------------------------------------------------------------------
# Browser sign-in: the page itself is gated, and the key becomes a session
# --------------------------------------------------------------------------

def test_the_app_page_shows_sign_in_until_signed_in(client, tenants):
    _, key_a = tenants["a"]
    page = client.get("/")
    assert page.status_code == 200 and "Sign in" in page.text and 'id="themeToggle"' not in page.text
    assert client.post("/login", json={"api_key": "ckt_wrong"}).status_code == 401
    login = client.post("/login", json={"api_key": key_a})
    assert login.status_code == 200 and login.json()["workspace"] == "Acme"
    cookie = login.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    app = client.get("/")
    assert 'id="themeToggle"' in app.text                      # the real app, now
    assert client.get("/me").json()["workspace"] == "Acme"
    assert client.get("/jobs").status_code == 200


def test_cookie_sessions_need_the_csrf_header_to_change_anything(client, tenants):
    _, key_a = tenants["a"]
    client.post("/login", json={"api_key": key_a})
    body = {"transcript": "Hello."}
    assert client.post("/kt-from-transcript", json=body).status_code == 403
    assert client.post("/kt-from-transcript", json=body, headers={"X-Continuum-CSRF": "1"}).status_code == 422


def test_sign_out_ends_the_session(client, tenants):
    _, key_a = tenants["a"]
    client.post("/login", json={"api_key": key_a})
    assert client.get("/jobs").status_code == 200
    client.post("/logout")
    assert client.get("/jobs").status_code == 401
    assert "Sign in" in client.get("/").text
