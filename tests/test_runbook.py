"""P3-5: one runbook per known failure, built only from the KT's incident card
(as last saved), with the rollback when a release may be the cause and the
never-do rules that name the same component."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import runbook


def _job(rows):
    return {"document_version": 2, "knowledge_object": {"system_name": "Quarry", "rendered_sections": [
        {"section_id": "quick_reference", "section_title": "Quick Reference", "blocks": [
            {"type": "DecisionTable", "title": "Incident card", "columns": ["Situation", "What to do", "Source"],
             "rows": rows}]}]}}


ROWS = [
    {"Situation": "An alert fires", "What to do": "Check the quote latency dashboard in Grafana.", "Source": "Monitoring"},
    {"Situation": "the SAP price file arriving late", "What to do": "Ask the SAP team to resend it, then rerun the import.",
     "Source": "Common Failures & Fixes"},
    {"Situation": "quotes failing after a deployment", "What to do": "No fix was stated in the KT. Ask the outgoing owner.",
     "Source": "Common Failures & Fixes"},
    {"Situation": "A release misbehaves (roll back)", "What to do": "Run the same playbook with the previous version.",
     "Source": "Deployment"},
    {"Situation": "Who to page or escalate to", "What to do": "Marta Kowalski is the person to call.", "Source": "Ownership"},
    {"Situation": "Never", "What to do": "Don't run the import script by hand during working hours.", "Source": "Danger Zones"},
    {"Situation": "Never", "What to do": "Never restart all three Postgres nodes at once.", "Source": "Danger Zones"},
    {"Situation": "Do not deploy", "What to do": "The last two weeks of March.", "Source": "Operational Calendar"},
]


def test_each_failure_gets_its_steps_rollback_escalation_and_matching_rules():
    books = runbook.runbooks(_job(ROWS)["knowledge_object"])
    sap, release = books["failures"]
    assert sap["situation"] == "the SAP price file arriving late"
    assert sap["notice"] == "Check the quote latency dashboard in Grafana."
    assert sap["escalate"] == "Marta Kowalski is the person to call."
    assert sap["rollback"] is None                                      # a late file is not a release
    assert sap["careful"] == ["Don't run the import script by hand during working hours."]   # "import"
    assert release["rollback"] == "Run the same playbook with the previous version."
    assert books["always"] == [("Never", "Never restart all three Postgres nodes at once."),
                               ("Do not deploy", "The last two weeks of March.")]


def test_the_markdown_says_where_the_kt_gave_no_fix_and_adds_nothing():
    text = runbook.runbook_markdown("abcd1234-0000", _job(ROWS), created=1_760_000_000)
    assert text.startswith("# Runbooks: Quarry\n")
    assert "job ABCD1234, version 2" in text
    assert "## 2. Quotes failing after a deployment" in text
    assert "- **What to do:** No fix was stated in the KT. Ask the outgoing owner." in text
    assert "- **If a release caused it, roll back:** Run the same playbook with the previous version." in text
    lines = [l for l in text.splitlines() if l.startswith("- **")]
    stated = {r["What to do"] for r in ROWS}
    assert all(any(s in l for s in stated) for l in lines)              # every step is a card line


def test_a_kt_without_failures_says_so():
    text = runbook.runbook_markdown("abcd1234", _job(ROWS[:1]))
    assert "named no failures" in text
