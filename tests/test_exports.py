"""P2-2: the KT document as Markdown and its gaps as a tracker CSV."""
import csv
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import markdown_export


def _job():
    return {
        "status": "completed", "warnings": ["Transcription failed for part of the recording."],
        "knowledge_object": {
            "system_name": "Payslip",
            "rendered_sections": [
                {"section_id": "deployment_and_rollback", "section_title": "Deployment & Rollback", "blocks": [
                    {"type": "ChecklistBlock", "title": "Steps", "items": ["Build in Azure DevOps", "", "Swap slots"],
                     "sources": [[1], [], [2]]},
                    {"type": "DecisionTable", "title": "Rollback", "columns": ["Trigger", "Action"],
                     "rows": [{"Trigger": "Login errors | 5xx", "Action": ""}], "sources": [[2]]},
                ]},
            ],
            "sources": [{"id": 1, "quote": "A pull request into main runs the build.", "start": 75.0, "speaker": "Speaker 1"},
                        {"id": 2, "quote": "Rolling back is swapping the slots again.", "start": 3700.0}],
            "sections": [{"id": "kt_coverage", "_knowledge_gaps": [
                "Disaster Recovery: RPO, DR testing", "Cost Optimization: not covered in the KT session"]}],
        },
    }


def test_markdown_keeps_structure_sources_and_speakers():
    text = markdown_export.job_markdown("abcd1234-0000", _job(), created=1759800000)
    assert text.startswith("# Payslip\n")
    assert "> **Review before relying on this document:** Transcription failed" in text
    assert "## 1. Deployment & Rollback" in text
    assert "- Build in Azure DevOps [1]" in text and "- Swap slots [2]" in text and "\n- \n" not in text
    # Table cells stay on one line, pipes escaped, empty cells say so.
    assert r"| Login errors \| 5xx | Not covered [2] |" in text
    assert "1. 01:15 Speaker 1: “A pull request into main runs the build.”" in text
    assert "2. 1:01:40 “Rolling back is swapping the slots again.”" in text


def test_gaps_become_one_tracker_row_each():
    rows = list(csv.DictReader(io.StringIO(markdown_export.gaps_csv("abcd1234-0000", _job()))))
    assert [r["Summary"] for r in rows] == ["KT gap (Payslip): Disaster Recovery: RPO, DR testing",
                                            "KT gap (Payslip): Cost Optimization: not covered in the KT session"]
    assert all(r["Issue Type"] == "Task" and r["Labels"] == "kt-gap" and "ABCD1234" in r["Description"] for r in rows)
