"""P3-4: the components several systems of a workspace name, from their KTs'
architecture pictures, with the sentence that says how each system uses it."""
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dependencies


def _job(system, data=(), external=(), delivery=(), edges=()):
    graph = {"central": {"id": "central", "header": "Amazon EKS", "label": system},
             "boundary": {"label": "AWS"}, "entry": [{"id": "users", "label": "Users", "caption": "customers"}],
             "data": [{"id": d.lower(), "label": d, "caption": "database"} for d in data],
             "external": [{"id": e.lower(), "label": e, "caption": "payments"} for e in external],
             "delivery": [{"id": t.lower(), "label": t, "caption": "CI/CD"} for t in delivery],
             "operations": [], "edges": [{"from": "central", "to": to, "label": label, "quote": quote}
                                         for to, label, quote in edges]}
    return {"status": "completed", "knowledge_object": {"system_name": system, "sections": [
        {"id": "architecture_reference", "_architecture_graph": graph}]}}


def test_a_component_named_by_two_systems_is_shared_with_how_each_uses_it():
    entries = [dependencies.system_entry("j1", _job("Relay", data=["Kafka", "Snowflake"], delivery=["GitLab CI"],
                                                     edges=[("kafka", "publishes to", "The producers publish to Kafka.")])),
               dependencies.system_entry("j2", _job("Beacon", data=["Kafka"], external=["Twilio"],
                                                     delivery=["GitLab CI"])),
               dependencies.system_entry("j3", _job("Quarry", data=["PostgreSQL"]))]
    result = dependencies.explore(entries)
    assert [s["system"] for s in result["systems"]] == ["Relay", "Beacon", "Quarry"]
    kafka, gitlab = result["shared"]
    assert kafka["component"] == "Kafka" and kafka["dependency"]           # dependencies before tooling
    assert kafka["systems"][0] == {"job_id": "j1", "system": "Relay", "how": "publishes to",
                                   "quote": "The producers publish to Kafka."}
    assert gitlab["component"] == "GitLab CI" and not gitlab["dependency"]
    assert all(c["component"] not in ("Users", "Twilio", "PostgreSQL") for c in result["shared"])


def test_repeat_kts_of_one_system_count_once_and_kts_without_a_picture_are_skipped():
    entries = [dependencies.system_entry("new", _job("Relay", data=["Kafka"])),
               dependencies.system_entry("old", _job("relay", data=["Kafka"]))]
    assert dependencies.explore(entries)["shared"] == []
    assert dependencies.system_entry("x", {"knowledge_object": {"sections": []}}) is None


def test_the_route_reads_only_the_callers_workspace(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import auth
    import main
    import pipeline
    from job_store import JobStore, PersistentJobs

    store = JobStore(str(tmp_path / "jobs.sqlite"))
    monkeypatch.setattr(pipeline, "JOB_STORE", store)
    monkeypatch.setattr(pipeline, "JOB_QUEUE", PersistentJobs(store))
    monkeypatch.setenv("CONTINUUM_AUTH", "required")
    mine, theirs = auth.create_tenant("Mine " + uuid.uuid4().hex[:6]), auth.create_tenant("Theirs")
    key = auth.create_key(mine, "reader", role="receiver")
    pipeline.JOB_QUEUE["a"] = dict(_job("Relay", data=["Kafka"]), tenant_id=mine)
    pipeline.JOB_QUEUE["b"] = dict(_job("Beacon", data=["Kafka"]), tenant_id=mine)
    pipeline.JOB_QUEUE["c"] = dict(_job("Spy", data=["Kafka"]), tenant_id=theirs)
    pipeline.JOB_QUEUE["d"] = dict(_job("Pending", data=["Kafka"]), tenant_id=mine, status="processing")
    body = TestClient(main.app).get("/dependencies", headers={"X-API-Key": key}).json()
    assert sorted(s["system"] for s in body["shared"][0]["systems"]) == ["Beacon", "Relay"]
    assert len(body["systems"]) == 2
