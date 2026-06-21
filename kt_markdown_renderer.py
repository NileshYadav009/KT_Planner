"""Structured Markdown renderer for KT output.

This module turns the serialized KT structure into a more professional
ordered Markdown report without changing the JSON payload shape.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def _sentence_texts(content: Dict[str, Any]) -> List[str]:
    return [
        s.get("text", "").strip()
        for s in content.get("sentences", [])
        if s.get("text", "").strip()
    ]


def _section_title(sec_id: str, content: Dict[str, Any]) -> str:
    title = (content.get("section_title") or sec_id or "KT Section").strip()
    return title or "KT Section"


def _label_sentence(text: str) -> str:
    if ":" not in text:
        return text

    label, value = text.split(":", 1)
    label = label.strip()
    value = value.strip()

    if not label or len(label.split()) > 6:
        return text

    return f"**{label}:** {value}" if value else f"**{label}:**"


def _render_numbered_sentences(sentences: Iterable[str]) -> List[str]:
    lines: List[str] = []
    for idx, text in enumerate(sentences, 1):
        lines.append(f"{idx}. {_label_sentence(text)}")
    return lines


def _render_bullets(sentences: Iterable[str]) -> List[str]:
    return [f"- {_label_sentence(text)}" for text in sentences]


def _render_deployment_section(sec_id: str, content: Dict[str, Any]) -> List[str]:
    sentences = _sentence_texts(content)
    lines = [f"### {_section_title(sec_id, content)}", "Deployment Process:"]
    lines.extend(_render_numbered_sentences(sentences) if sentences else ["1. No deployment details captured."])
    return lines + [""]


def _render_monitoring_section(sec_id: str, content: Dict[str, Any]) -> List[str]:
    sentences = _sentence_texts(content)
    lines = [f"### {_section_title(sec_id, content)}", "Monitoring Stack:"]
    lines.extend(_render_bullets(sentences) if sentences else ["- No monitoring details captured."])
    return lines + [""]


def _render_architecture_section(sec_id: str, content: Dict[str, Any]) -> List[str]:
    sentences = _sentence_texts(content)
    lines = [f"### {_section_title(sec_id, content)}", "Architecture Notes:"]
    lines.extend(_render_bullets(sentences) if sentences else ["- No architecture details captured."])
    return lines + [""]


def _render_issue_section(sec_id: str, content: Dict[str, Any]) -> List[str]:
    sentences = _sentence_texts(content)
    lines = [f"### {_section_title(sec_id, content)}", "Issue / Cause / Fix:"]

    if sentences:
        for text in sentences:
            lines.append(f"- {_label_sentence(text)}")
    else:
        lines.append("- No issue details captured.")

    return lines + [""]


def _render_generic_section(sec_id: str, content: Dict[str, Any]) -> List[str]:
    sentences = _sentence_texts(content)
    lines = [f"### {_section_title(sec_id, content)}"]
    lines.extend(_render_bullets(sentences) if sentences else ["- No section details captured."])
    return lines + [""]


def _section_renderer(sec_id: str, content: Dict[str, Any]):
    title = _section_title(sec_id, content).lower()
    lower_id = sec_id.lower()

    if any(token in lower_id or token in title for token in ("deploy", "rollback", "release")):
        return _render_deployment_section
    if any(token in lower_id or token in title for token in ("monitor", "observability", "alert", "metrics")):
        return _render_monitoring_section
    if any(token in lower_id or token in title for token in ("architecture", "arch", "component", "flow")):
        return _render_architecture_section
    if any(token in lower_id or token in title for token in ("issue", "troubleshoot", "problem", "failure", "known")):
        return _render_issue_section

    return _render_generic_section


def render_ordered_markdown(kt_obj: Any) -> str:
    """Render a structured Markdown report from a KT object."""
    section_content = getattr(kt_obj, "section_content", {}) or {}
    job_id = getattr(kt_obj, "job_id", "kt-report")

    desired_order = [
        ("System Overview", ["overview", "system", "summary", "introduction", "context"]),
        ("Architecture Components", ["architecture", "component", "service", "design"]),
        ("Data Flow", ["flow", "pipeline", "request", "message", "event"]),
        ("Deployment & Rollback", ["deploy", "deployment", "rollback", "release", "rollout", "kubectl", "helm", "docker"]),
        ("Monitoring & Observability", ["monitor", "observability", "metrics", "alert", "logging", "tracing", "grafana"]),
        ("Troubleshooting & Known Issues", ["issue", "problem", "error", "fail", "bug", "troubleshoot", "troubleshooting"]),
    ]

    grouped: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {name: [] for name, _ in desired_order}
    grouped["Known Issues"] = []

    for sec_id, content in section_content.items():
        title = _section_title(sec_id, content).lower()
        lower_id = sec_id.lower()
        placed = False
        for group_name, keywords in desired_order:
            if any(token in lower_id or token in title for token in keywords):
                grouped[group_name].append((sec_id, content))
                placed = True
                break
        if not placed:
            grouped["Known Issues"].append((sec_id, content))

    lines = [f"# KT Report - {job_id}", ""]

    for group_name, _ in desired_order:
        items = grouped.get(group_name, [])
        if not items:
            continue
        lines.append(f"## {group_name}")
        for sec_id, content in items:
            renderer = _section_renderer(sec_id, content)
            lines.extend(renderer(sec_id, content))

    known_issues = grouped.get("Known Issues", [])
    if known_issues:
        lines.append("## Known Issues")
        for sec_id, content in known_issues:
            renderer = _section_renderer(sec_id, content)
            lines.extend(renderer(sec_id, content))

    return "\n".join(lines).rstrip() + "\n"