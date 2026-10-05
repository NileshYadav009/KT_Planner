"""P1-13: one classifier. POST /semantic-placement ran a second classifier
(enterprise_semantic_mapper.py plus a legacy path in ai.py) that disagreed
with the pipeline, and its response hard-coded "unclassified_sentences": 0
and "duplicate_rate": 0.0."""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_legacy_endpoint_is_gone():
    from fastapi.testclient import TestClient
    import main

    resp = TestClient(main.app).post("/semantic-placement", json={"transcript": "Datadog alerts page the on-call."})
    assert resp.status_code in (404, 405)


def test_only_the_context_mapper_classifies():
    assert importlib.util.find_spec("enterprise_semantic_mapper") is None
    import ai
    for name in ("generate_report", "analyze_transcript", "classify_transcript", "build_section_paragraphs"):
        assert not hasattr(ai, name), name
    offenders = []
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("archive", "__pycache__", "node_modules")]
        for f in files:
            if f.endswith(".py") and f != os.path.basename(__file__):
                with open(os.path.join(folder, f), encoding="utf-8", errors="ignore") as fh:
                    if "enterprise_semantic_mapper" in fh.read():
                        offenders.append(os.path.relpath(os.path.join(folder, f), ROOT))
    assert offenders == []


def test_the_project_root_holds_modules_not_scripts():
    loose = [f for f in os.listdir(ROOT) if f.endswith(".py") and f.startswith(("test_", "demo_"))]
    assert loose == []
