"""
field_populator.py
==================
Populate KT fields from coverage content using pattern, semantic, and LLM passes.
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

PATTERN_EXTRACTORS = {
    "url": re.compile(r"https?://[^\s\)\"']+", re.IGNORECASE),
    "tools": re.compile(
        r"\b(Prometheus|Grafana|CloudWatch|PagerDuty|OpsGenie|Jenkins|"
        r"GitLab\s*CI|GitHub\s*Actions|ArgoCD|Flux|Helm|Terraform|Ansible|"
        r"Kubernetes|Docker|Rancher|Vault|Consul|Nexus|Artifactory|"
        r"SonarQube|Trivy|Veracode|Datadog|Splunk|ELK|Elasticsearch|"
        r"Logstash|Kibana|Redis|Kafka|RabbitMQ|PostgreSQL|MySQL|MongoDB|"
        r"Amazon\s+EKS|Amazon\s+RDS|Amazon\s+ECR|CloudFront|S3)\b",
        re.IGNORECASE,
    ),
    "duration": re.compile(r"\b(\d+)\s*(minutes?|mins?|hours?|hrs?|days?|seconds?|secs?)\b", re.IGNORECASE),
    "team": re.compile(r"\b([A-Z][a-z]+ (?:Team|Engineering|Lead|Manager|Owner))\b"),
    "slack": re.compile(r"#[a-z][a-z0-9\-_]+"),
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "environments": re.compile(r"\b(prod(?:uction)?|staging|qa|dev(?:elopment)?|sandbox|uat|pre-?prod)\b", re.IGNORECASE),
    "region": re.compile(r"\b(us-east-[12]|us-west-[12]|eu-west-[123]|eu-central-1|ap-south-1|ap-southeast-[12]|ap-northeast-[123])\b"),
    "orders_per_day": re.compile(r"(\d[\d,]+)\s*orders?\s*(?:per|a)\s*day", re.IGNORECASE),
}


def _extract_by_pattern(field: Dict[str, Any], section_text: str) -> Optional[Any]:
    field_type = field.get("type", "text")
    field_id = field.get("id", "")

    if field_type == "url":
        match = PATTERN_EXTRACTORS["url"].search(section_text)
        return match.group(0) if match else None

    if field_type == "boolean":
        text_lower = section_text.lower()
        positives = ["yes", "confirmed", "verified", "done", "complete", "✓"]
        if any(p in text_lower for p in positives):
            return True
        return None

    if field_type in ("single_select", "multi_select"):
        options = [o.lower() for o in field.get("options", [])]
        matched = []
        text_lower = section_text.lower()
        for opt in options:
            if opt in text_lower:
                matched.append(opt.title())
        if field_type == "single_select":
            return matched[0] if matched else None
        return matched if matched else None

    if "orders" in field_id or "transactions" in field_id:
        match = PATTERN_EXTRACTORS["orders_per_day"].search(section_text)
        if match:
            return f"{match.group(1)} orders per day"

    # Special-case oncall tool extraction to avoid matching generic 'tool' patterns
    if field_id == "oncall_tool":
        oncall_pattern = re.compile(r"\b(PagerDuty|OpsGenie|VictorOps|Splunk\s+On-Call)\b", re.IGNORECASE)
        match = oncall_pattern.search(section_text)
        if match:
            return match.group(1)

    if field_id != "oncall_tool" and ("tool" in field_id or "technolog" in field_id or "stack" in field_id):
        tools = PATTERN_EXTRACTORS["tools"].findall(section_text)
        if tools:
            return ", ".join(dict.fromkeys(tools))

    if "environment" in field_id:
        envs = PATTERN_EXTRACTORS["environments"].findall(section_text)
        if envs:
            return ", ".join(dict.fromkeys(e.capitalize() for e in envs))

    if "channel" in field_id or "slack" in field_id:
        channels = PATTERN_EXTRACTORS["slack"].findall(section_text)
        if channels:
            return ", ".join(channels)

    if field_id == "system_name":
        match = re.search(
            r"(?:handing over|this is|for|about)\s+(?:the\s+)?([A-Za-z0-9][A-Za-z0-9\s\-]{2,50}?)(?:\s+platform|\s+system|\s+application|\s+service|\.|\,)",
            section_text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip().title()

    if "escalation" in field_id or "chain" in field_id:
        pattern = re.compile(
            r"(?:starts with|first|initially|beginning with)\s+(.+?)\s+(?:followed by|then|and then)\s+(.+?)(?:\s+and then\s+(.+?))?[,\.]",
            re.IGNORECASE,
        )
        match = pattern.search(section_text)
        if match:
            chain = [g.strip() for g in match.groups() if g]
            return " → ".join(chain)

    if "blackout" in field_id or "bad_day" in field_id or "avoid" in field_id:
        events = re.findall(
            r"\b(Black Friday|month[\s-]end|end[\s-]of[\s-]month|promotional campaign|peak sale|quarter[\s-]end|EOD)\b",
            section_text,
            re.IGNORECASE,
        )
        if events:
            return ", ".join(dict.fromkeys(events))

    if "rollback" in field_id and ("time" in field_id or "duration" in field_id):
        match = re.search(
            r"(?:rollback|roll back|roll-back).*?(?:within|in|under)\s+(\d+\s*minutes?)",
            section_text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)

    if "duration" in field_id or "time" in field_id or "minutes" in field_id:
        match = PATTERN_EXTRACTORS["duration"].search(section_text)
        if match:
            return f"{match.group(1)} {match.group(2)}"

    if field_type == "table":
        lines = [line.strip() for line in section_text.splitlines() if line.strip()]
        if not lines:
            return None

        pipe_rows = [line for line in lines if "|" in line]
        if pipe_rows:
            return "\n".join(pipe_rows)

        numbered_rows = [line for line in lines if re.match(r"^(?:\d+\.|step\b)", line, re.IGNORECASE)]
        if numbered_rows:
            return "\n".join(numbered_rows)

        return "\n".join(lines[:10])

    return None


def _extract_by_semantic(field: Dict[str, Any], sentences: List[str], model=None) -> Optional[str]:
    if not sentences or model is None:
        return sentences[0] if sentences else None

    from sentence_transformers import util as st_util

    field_text = f"{field.get('label', '')} {field.get('description', '')}"
    field_emb = model.encode(field_text, convert_to_tensor=True, normalize_embeddings=True)
    sent_embs = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
    scores = st_util.cos_sim(field_emb, sent_embs)[0]
    best_idx = int(scores.argmax().item())
    best_score = float(scores[best_idx].item())
    if best_score >= 0.35:
        return sentences[best_idx]
    return None


def _build_llm_gap_fill_prompt(section_title: str, field: Dict[str, Any], available_text: str) -> str:
    field_label = field.get("label", field.get("id", ""))
    field_desc = field.get("description", "")
    options = field.get("options", [])
    options_str = f"\nValid options: {', '.join(options)}" if options else ""
    return (
        f"You are filling a Knowledge Transfer document field.\n\n"
        f"Section: {section_title}\n"
        f"Field: {field_label}\n"
        f"Description: {field_desc}{options_str}\n\n"
        f"Available transcript content:\n{available_text}\n\n"
        f"Extract ONLY the value for '{field_label}' from the transcript above.\n"
        f"Rules:\n"
        f"- If the information is present, extract it exactly.\n"
        f"- If it is NOT present, respond with exactly: NOT_MENTIONED\n"
        f"- Do NOT invent information.\n"
        f"- Return only the extracted value, no explanation.\n\n"
        f"Value:"
    )


def populate_fields(
    dynamic_schema: List[Dict],
    coverage: Dict[str, Any],
    llm_provider=None,
    embedding_model=None,
) -> Dict[str, Dict[str, Any]]:
    result = {}
    for section in dynamic_schema:
        section_id = section.get("id")
        section_title = section.get("title", section_id)
        fields = section.get("fields") or []
        if not fields:
            continue

        cov_entry = coverage.get(section_id, {})
        content = cov_entry.get("content", [])
        if isinstance(content, list):
            section_text = " ".join(content)
            sentences = [c for c in content if c.strip()]
        else:
            section_text = str(content)
            sentences = [section_text]

        result[section_id] = {}
        _populate_fields_recursive(
            fields=fields,
            section_id=section_id,
            section_title=section_title,
            section_text=section_text,
            sentences=sentences,
            output=result[section_id],
            llm_provider=llm_provider,
            embedding_model=embedding_model,
        )

    return result


def _populate_fields_recursive(
    fields: List[Dict[str, Any]],
    section_id: str,
    section_title: str,
    section_text: str,
    sentences: List[str],
    output: Dict[str, Any],
    llm_provider=None,
    embedding_model=None,
):
    for field in fields:
        field_id = field.get("id")
        field_type = field.get("type", "text")

        if field_type == "group":
            output[field_id] = {}
            _populate_fields_recursive(
                fields=field.get("fields", []),
                section_id=section_id,
                section_title=section_title,
                section_text=section_text,
                sentences=sentences,
                output=output[field_id],
                llm_provider=llm_provider,
                embedding_model=embedding_model,
            )
            continue

        value = _extract_by_pattern(field, section_text)
        if value is not None:
            output[field_id] = {"value": value, "confidence": 0.90, "source": "pattern"}
            continue

        if field_type == "text" and sentences:
            value = _extract_by_semantic(field, sentences, embedding_model)
            if value is not None:
                output[field_id] = {"value": value, "confidence": 0.65, "source": "semantic"}
                continue

        if llm_provider and section_text.strip() and field_type != "table":
            try:
                prompt = _build_llm_gap_fill_prompt(section_title=section_title, field=field, available_text=section_text[:1500])
                raw = llm_provider.generate(prompt, temperature=0.1, max_output_tokens=256, stop_sequences=["\n"])
                val = raw.strip() if isinstance(raw, str) else ""
                if val and val.upper() != "NOT_MENTIONED" and len(val) < 500:
                    output[field_id] = {"value": val, "confidence": 0.75, "source": "llm"}
                    continue
            except Exception as exc:
                logger.warning("LLM field fill failed for %s.%s: %s", section_id, field_id, exc)

        output[field_id] = {"value": "", "confidence": 0.0, "source": "unfilled"}
