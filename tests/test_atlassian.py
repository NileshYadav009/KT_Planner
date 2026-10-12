"""P2-2: a workspace connected to Confluence and Jira publishes its KT as a
page (published again, the same page gets a new version) and its knowledge
gaps as issues (a gap already sent is not sent again). Run against a local
stand-in for the Atlassian APIs: Cloud (Confluence v2, Jira v3, basic auth)
and Data Center (Confluence v1, Jira v2, bearer token)."""
import json
import os
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import atlassian
import auth
from job_queue import TaskQueue
from job_store import JobStore, PersistentJobs
from renderers.sections.kt_coverage import KNOWLEDGE_GAPS_TITLE

GAPS = ["Disaster recovery was not covered in the session", "Cost controls were not covered in the session"]


class FakeAtlassian(BaseHTTPRequestHandler):
    requests: list = []
    pages: dict = {}
    issues = 0

    def log_message(self, *args):
        pass

    def _send(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"null") if length else None
        url = urlparse(self.path)
        FakeAtlassian.requests.append({"method": self.command, "path": url.path, "query": url.query,
                                       "auth": self.headers.get("Authorization"), "body": body})
        base = f"http://127.0.0.1:{self.server.server_port}"
        parts = url.path.strip("/").split("/")
        if url.path == "/wiki/api/v2/spaces":
            return self._send(200, {"results": [{"id": "98765", "key": "OPS"}] if "keys=OPS" in url.query else []})
        if url.path in ("/wiki/api/v2/pages", "/rest/api/content") and self.command == "POST":
            page_id = str(1000 + len(FakeAtlassian.pages) + 1)
            FakeAtlassian.pages[page_id] = {"version": 1, "body": body}
            return self._send(200, {"id": page_id, "version": {"number": 1},
                                    "_links": {"base": base + ("/wiki" if "wiki" in url.path else ""),
                                               "webui": f"/spaces/OPS/pages/{page_id}"}})
        if parts[-1] in FakeAtlassian.pages or (len(parts) > 1 and parts[-1].isdigit()):
            page = FakeAtlassian.pages.get(parts[-1])
            if page is None:
                return self._send(404, {"message": "No page"})
            if self.command == "PUT":
                assert body["version"]["number"] == page["version"] + 1
                page.update(version=body["version"]["number"], body=body)
            return self._send(200, {"id": parts[-1], "version": {"number": page["version"]},
                                    "_links": {"webui": f"/spaces/OPS/pages/{parts[-1]}"}})
        if url.path.endswith("/issue/bulk"):
            issues, errors = [], []
            for n, update in enumerate(body["issueUpdates"]):
                if "REFUSE" in update["fields"]["summary"]:
                    errors.append({"failedElementNumber": n, "status": 400,
                                   "elementErrors": {"errors": {"summary": "Summary is not allowed"}}})
                    continue
                FakeAtlassian.issues += 1
                issues.append({"id": str(FakeAtlassian.issues), "key": f"OPS-{FakeAtlassian.issues}"})
            return self._send(201, {"issues": issues, "errors": errors})
        return self._send(404, {"errorMessages": ["Not found"]})

    do_GET = do_POST = do_PUT = _handle


@pytest.fixture
def site():
    FakeAtlassian.requests, FakeAtlassian.pages, FakeAtlassian.issues = [], {}, 0
    server = HTTPServer(("127.0.0.1", 0), FakeAtlassian)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _document():
    return {"system_name": "TripWise", "rendered_sections": [
        {"section_id": "monitoring_observability", "section_title": "Monitoring", "blocks": [
            {"type": "NarrativeBlock", "title": "Monitoring", "sources": [[1]],
             "paragraphs": ["Datadog monitors the <script>alert(1)</script> booking API."]}]},
        {"section_id": "ownership_escalation", "section_title": "Ownership", "blocks": [
            {"type": "OwnershipTable", "title": "Owners", "rows": [{"role": "Bookings", "team": "Platform & SRE"}]}]},
        {"section_id": "kt_coverage", "section_title": "KT Coverage", "blocks": [
            {"type": "ChecklistBlock", "title": KNOWLEDGE_GAPS_TITLE, "items": list(GAPS)}]},
    ], "sections": [{"id": "kt_coverage", "_knowledge_gaps": list(GAPS)}],
        "sources": [{"id": 1, "quote": "Datadog monitors the booking API.", "start": 12.0}]}


@pytest.fixture
def team(tmp_path, monkeypatch):
    import pipeline
    from fastapi.testclient import TestClient
    import main

    store = JobStore(str(tmp_path / "jobs.sqlite"))
    monkeypatch.setattr(pipeline, "JOB_STORE", store)
    monkeypatch.setattr(pipeline, "JOB_QUEUE", PersistentJobs(store))
    monkeypatch.setattr(pipeline, "TASKS", TaskQueue(store))
    monkeypatch.setenv("CONTINUUM_AUTH", "required")
    monkeypatch.setenv("ATLASSIAN_TOKEN", "secret-token")
    tenant = auth.create_tenant("Publish Co " + uuid.uuid4().hex[:6])

    def person(role):
        token = auth.create_session(auth.Principal(tenant_id=tenant, role=role, key_label=f"{role}@x.example",
                                                   user_id=f"u_{role}", method="sso"))
        return {"Cookie": f"{auth.SESSION_COOKIE}={token}", "X-Continuum-CSRF": "1"}

    job_id = "pub-" + uuid.uuid4().hex[:8]
    pipeline.JOB_QUEUE[job_id] = {"status": "completed", "tenant_id": tenant, "knowledge_object": _document()}
    return TestClient(main.app), job_id, tenant, {"giver": person("giver"), "receiver": person("receiver")}, store


def test_data_center_page_is_created_then_updated_with_escaped_text(site, team):
    client, job_id, tenant, people, store = team
    atlassian.set_integration(tenant, "confluence", site, "ATLASSIAN_TOKEN", space="OPS", parent_id="42")
    assert client.get("/integrations", headers=people["receiver"]).json() == {"confluence": True, "jira": False}

    first = client.post(f"/publish/confluence/{job_id}", headers=people["giver"])
    assert first.status_code == 200, first.text
    assert first.json()["created"] and first.json()["version"] == 1
    assert first.json()["url"] == f"{site}/spaces/OPS/pages/1001"
    create = FakeAtlassian.requests[-1]
    assert (create["method"], create["path"]) == ("POST", "/rest/api/content")
    assert create["auth"] == "Bearer secret-token"
    assert create["body"]["space"] == {"key": "OPS"} and create["body"]["ancestors"] == [{"id": "42"}]
    assert create["body"]["title"] == f"TripWise - KT handover ({job_id[:8].upper()})"
    page = create["body"]["body"]["storage"]["value"]
    assert "&lt;script&gt;" in page and "<script>" not in page          # transcript text is never markup
    assert "<td>Platform &amp; SRE</td>" in page and "<h2>Sources</h2>" in page

    updated_before = store.updated_at(job_id)
    again = client.post(f"/publish/confluence/{job_id}", headers=people["giver"]).json()
    assert again == {"page_id": "1001", "url": f"{site}/spaces/OPS/pages/1001", "version": 2, "created": False}
    assert len(FakeAtlassian.pages) == 1
    assert store.updated_at(job_id) == updated_before                   # publishing is not a change to the KT


def test_cloud_page_uses_the_v2_api_and_an_api_token(site, team):
    client, job_id, tenant, people, _ = team
    atlassian.set_integration(tenant, "confluence", site + "/wiki", "ATLASSIAN_TOKEN", email="bot@x.example",
                              space="OPS", cloud=True)
    resp = client.post(f"/publish/confluence/{job_id}", headers=people["giver"])
    assert resp.status_code == 200, resp.text
    lookup, create = FakeAtlassian.requests[-2:]
    assert lookup["path"] == "/wiki/api/v2/spaces" and lookup["query"] == "keys=OPS"
    assert create["path"] == "/wiki/api/v2/pages" and create["body"]["spaceId"] == "98765"
    assert create["body"]["body"]["representation"] == "storage"
    assert create["auth"].startswith("Basic ")
    assert client.post(f"/publish/confluence/{job_id}", headers=people["giver"]).json()["version"] == 2
    assert FakeAtlassian.requests[-1]["method"] == "PUT"


def test_each_gap_becomes_one_jira_issue_once(site, team):
    client, job_id, tenant, people, _ = team
    atlassian.set_integration(tenant, "jira", site, "ATLASSIAN_TOKEN", email="bot@x.example", project="ops",
                              cloud=True)
    first = client.post(f"/publish/jira/{job_id}", headers=people["giver"]).json()
    assert [i["key"] for i in first["created"]] == ["OPS-1", "OPS-2"] and first["failed"] == []
    assert first["created"][0]["url"] == f"{site}/browse/OPS-1"
    sent = FakeAtlassian.requests[-1]
    assert sent["path"] == "/rest/api/3/issue/bulk"
    fields = sent["body"]["issueUpdates"][0]["fields"]
    assert fields["project"] == {"key": "OPS"} and fields["labels"] == ["kt-gap"]
    assert fields["summary"] == f"KT gap (TripWise): {GAPS[0]}"
    assert fields["description"]["type"] == "doc"                       # Atlassian Document Format on Cloud

    # A new gap after an edit: only it is sent.
    import pipeline
    job = pipeline.JOB_QUEUE[job_id]
    job["knowledge_object"]["sections"][0]["_knowledge_gaps"].append("REFUSE this one")
    job["knowledge_object"]["sections"][0]["_knowledge_gaps"].append("Security was not covered in the session")
    pipeline.JOB_QUEUE[job_id] = job
    second = client.post(f"/publish/jira/{job_id}", headers=people["giver"]).json()
    assert [i["key"] for i in second["already"]] == ["OPS-1", "OPS-2"]
    assert [i["key"] for i in second["created"]] == ["OPS-3"]
    assert second["failed"] == [{"gap": "REFUSE this one", "reason": "summary: Summary is not allowed"}]
    assert len(FakeAtlassian.requests[-1]["body"]["issueUpdates"]) == 2


def test_data_center_jira_gets_plain_text_and_a_bearer_token(site, team):
    client, job_id, tenant, people, _ = team
    atlassian.set_integration(tenant, "jira", site, "ATLASSIAN_TOKEN", project="OPS")
    assert client.post(f"/publish/jira/{job_id}", headers=people["giver"]).status_code == 200
    sent = FakeAtlassian.requests[-1]
    assert sent["path"] == "/rest/api/2/issue/bulk" and sent["auth"] == "Bearer secret-token"
    assert isinstance(sent["body"]["issueUpdates"][0]["fields"]["description"], str)


def test_who_may_publish_and_what_is_refused(site, team, monkeypatch):
    client, job_id, tenant, people, _ = team
    assert client.post(f"/publish/jira/{job_id}", headers=people["giver"]).status_code == 409      # not connected
    atlassian.set_integration(tenant, "jira", site, "ATLASSIAN_TOKEN", project="OPS")
    assert client.post(f"/publish/jira/{job_id}", headers=people["receiver"]).status_code == 403   # read-only role
    monkeypatch.delenv("ATLASSIAN_TOKEN")
    missing = client.post(f"/publish/jira/{job_id}", headers=people["giver"])
    assert missing.status_code == 503 and "ATLASSIAN_TOKEN" in missing.json()["detail"]
    assert "secret" not in missing.text
    other = auth.create_tenant("Other Co")
    assert atlassian.configured(other) == {"confluence": False, "jira": False}


def test_settings_are_checked_and_the_token_itself_is_never_stored(team):
    _, _, tenant, _, _ = team
    with pytest.raises(ValueError, match="https"):
        atlassian.set_integration(tenant, "jira", "http://jira.acme.com", "ATLASSIAN_TOKEN", project="OPS")
    with pytest.raises(ValueError, match="NAME of an environment variable"):
        atlassian.set_integration(tenant, "jira", "https://acme.atlassian.net", "ATATT3xFfGF0abcdefXYZ", project="OPS")
    with pytest.raises(ValueError, match="space key"):
        atlassian.set_integration(tenant, "confluence", "https://acme.atlassian.net/wiki", "ATLASSIAN_TOKEN")
    config = atlassian.set_integration(tenant, "jira", "https://acme.atlassian.net/", "ATLASSIAN_TOKEN",
                                       email="bot@acme.com", project="ops")
    assert config == {"url": "https://acme.atlassian.net", "token_env": "ATLASSIAN_TOKEN", "email": "bot@acme.com",
                      "cloud": True, "project": "OPS", "issue_type": "Task"}
    assert atlassian.remove_integration(tenant, "jira") and atlassian.get_integration(tenant, "jira") is None
