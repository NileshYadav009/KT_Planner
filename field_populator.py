"""
field_populator.py
==================
Populate KT fields from coverage content using pattern, semantic, and LLM passes.
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip()).lower()

PATTERN_EXTRACTORS = {
    "url": re.compile(r"https?://[^\s\)\"']+", re.IGNORECASE),
    "tools": re.compile(
        r"\b(Prometheus|Grafana|CloudWatch|PagerDuty|OpsGenie|Jenkins|"
        r"GitLab\s*CI|GitHub\s*Actions|ArgoCD|Flux|Helm|Terraform|Ansible|"
        r"Kubernetes|Docker|Rancher|Vault|Consul|Nexus|Artifactory|"
        r"SonarQube|Trivy|Veracode|Datadog|Splunk|ELK|Elasticsearch|"
        r"Logstash|Kibana|Redis|Kafka|RabbitMQ|PostgreSQL|MySQL|MongoDB|"
        r"Amazon\s+EKS|Amazon\s+RDS|Amazon\s+ECR|CloudFront|S3|"
        r"React|Angular|Vue(?:\.js)?|Fast\s*API|Django|Flask|Node(?:\.js)?|"
        r"Express|Application\s+Load\s+Balancer|ALB|Load\s+Balancer)\b",
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


def _extract_escalation_chain(text: str) -> Optional[str]:
    """
    Extract escalation chain with robust multi-stage split approach.
    Handles phrases like "Escalation path starts with an on-call engineer 
    followed by the platform engineering manager and then the head of engineering."
    
    Returns steps joined with ' -> ' separator, matching the convention used by
    the LLM structured-extraction prompt for this same field (see
    llm/prompts.py) and parsed by knowledge/relationships.py.
    """
    match = re.search(
        r"escalation\s+(?:path|chain)?\s*(?:starts with|starting with)?\s*(.+?)(?:\.|$)",
        text, re.IGNORECASE,
    )
    if not match:
        return None

    clause = match.group(1)
    # Split on explicit transition keywords
    steps = re.split(r"\bfollowed by\b|\band then\b|\bthen\b", clause, flags=re.IGNORECASE)
    steps = [s.strip(" .") for s in steps if s.strip(" .")]

    return " -> ".join(steps) if len(steps) >= 2 else None


def _extract_ownership(text: str) -> dict:
    """
    Extract ownership assignments from text.
    Looks for specific patterns indicating application vs infrastructure ownership.
    
    Returns dict with "application_ownership" and/or "infrastructure_ownership" keys.
    """
    result = {}
    
    # Application ownership pattern
    app = re.search(
        r"\b(developers?|dev team)\b[^.]*?\bown[s]?\b[^.]*?\b(application code|app code|codebase)\b",
        text, re.IGNORECASE
    )
    if app:
        result["application_ownership"] = "Developers"
    
    # Infrastructure ownership pattern
    infra = re.search(
        r"\b(platform engineers?|platform engineering|infra(?:structure)? team)\b[^.]*?\bown[s]?\b[^.]*?\b(infrastructure|kubernetes)\b",
        text, re.IGNORECASE
    )
    if infra:
        result["infrastructure_ownership"] = "Platform Engineers"
    
    return result


def _extract_by_pattern(
    field: Dict[str, Any],
    section_text: str,
    exclude_line_keys: Optional[Set[str]] = None,
) -> Optional[Any]:
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
        # Use robust 3-stage extraction for escalation chains
        return _extract_escalation_chain(section_text)

    if field_id in ("application_ownership", "infrastructure_ownership"):
        # Extract ownership from text
        ownership_dict = _extract_ownership(section_text)
        return ownership_dict.get(field_id)

    if "oncall_tool" in field_id or "oncall" in field_id:
        oncall_pattern = re.compile(r"\b(PagerDuty|OpsGenie|VictorOps|Splunk\s+On-Call)\b", re.IGNORECASE)
        match = oncall_pattern.search(section_text)
        if match:
            return match.group(1)

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
        if exclude_line_keys:
            lines = [line for line in lines if _normalize_line(line) not in exclude_line_keys]
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


def _extract_by_semantic(
    field: Dict[str, Any],
    sentences: List[str],
    model=None,
    exclude_line_keys: Optional[Set[str]] = None,
) -> Optional[str]:
    if exclude_line_keys:
        sentences = [s for s in sentences if _normalize_line(s) not in exclude_line_keys]
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


def _build_llm_gap_fill_prompt(
    section_title: str,
    field: Dict[str, Any],
    available_text: str,
    already_captured: Optional[List[str]] = None,
) -> str:
    field_label = field.get("label", field.get("id", ""))
    field_desc = field.get("description", "")
    field_type = field.get("type", "text")
    options = field.get("options", [])
    options_str = f"\nValid options: {', '.join(options)}" if options else ""

    # The "paraphrase counts as explicit" leniency below is what lets
    # business_criticality normalize "most business critical system" -> a
    # High/Medium/Low category — appropriate for open-ended/categorical
    # fields. Applied to a structurally-typed field (a real URL, an actual
    # calendar date, an explicit yes/no) it does the opposite of what's
    # intended: it lets the model treat a loosely-related mention (e.g. "the
    # diagram is in Confluence") as if it satisfied "link to the
    # architecture documentation", producing a value that isn't actually a
    # URL at all. Those types get a strict literal-presence rule instead.
    _STRUCTURAL_TYPE_RULES = {
        "url": "an actual URL (starting with http:// or https://, or a clearly identifiable link)",
        "date": "an actual calendar date or timestamp",
        "boolean": "an explicit yes/no, true/false, or clearly confirmed/denied statement",
    }
    structural_hint = _STRUCTURAL_TYPE_RULES.get(field_type)
    strictness_rule = (
        f"- '{field_label}' requires {structural_hint}. Only extract a value if "
        f"the transcript literally contains one — a mention of a related tool, "
        f"system, or topic (e.g. naming where something is stored, without "
        f"giving the actual {field_type}) does NOT count. If no literal "
        f"{field_type} is present, respond with exactly: NOT_MENTIONED\n"
        if structural_hint else
        f"- If the transcript specifically discusses '{field_label}' (even in "
        f"different words), extract or normalize that value.\n"
    )

    # Sibling fields in the same section (e.g. Production / Staging /
    # Non-production "characteristics", or per-entity rows generally) often
    # share almost all of a sparse section's text. Without seeing what other
    # fields already claimed, the model has no way to tell "this text is
    # specifically about THIS field" from "this is the only text in the
    # section, so I'll reuse it" — which silently copies one entity's facts
    # onto an unrelated sibling entity. Generic: works for any set of
    # sibling field labels, not just environment names.
    already_str = ""
    if already_captured:
        joined = "\n".join(f"- {item}" for item in already_captured if item)
        if joined:
            already_str = (
                f"\nAlready captured for OTHER fields in this section (do not "
                f"repeat these as if they were specifically about "
                f"'{field_label}' unless the transcript explicitly says this "
                f"applies to '{field_label}' too):\n{joined}\n"
            )

    return (
        f"You are filling a Knowledge Transfer document field.\n\n"
        f"Section: {section_title}\n"
        f"Field: {field_label}\n"
        f"Description: {field_desc}{options_str}\n"
        f"{already_str}\n"
        f"Available transcript content:\n{available_text}\n\n"
        f"Extract ONLY the value for '{field_label}' from the transcript above.\n"
        f"Rules:\n"
        f"{strictness_rule}"
        f"- If the only relevant text is actually about a DIFFERENT, similar "
        f"item (e.g. a different environment/entity/team than '{field_label}') "
        f"and the transcript never says it also applies to '{field_label}', "
        f"respond with exactly: NOT_MENTIONED\n"
        f"- If nothing relevant is present at all, respond with exactly: NOT_MENTIONED\n"
        f"- Do NOT invent information that has no basis in the transcript above.\n\n"
        f"Respond with EXACTLY two lines and nothing else:\n"
        f"Line 1: the extracted value (or NOT_MENTIONED).\n"
        f"Line 2: EXPLICIT if the transcript directly states this fact (even in "
        f"different wording), or INFERRED if you had to reason/guess beyond "
        f"what the transcript actually says."
    )


def populate_fields(
    dynamic_schema: List[Dict],
    coverage: Dict[str, Any],
    llm_provider=None,
    embedding_model=None,
    section_content: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    section_content: kt.section_content (raw, timestamped sentences per section,
    as produced by context_mapper) — optional, but without it every extracted
    field's evidence falls back to knowledge_builder's generic "first 3
    sentences of the section" instead of the one sentence that actually
    supports it. This is the SAME list knowledge_builder._collect_evidence()
    indexes into via source_chunk_index, so the index computed here must refer
    to it specifically, not to `coverage[...]['content']` (a different,
    polished/reordered view of the section) — using the wrong list would
    silently attach the wrong evidence rather than just the generic fallback.
    """
    result = {}
    for section in dynamic_schema:
        section_id = section.get("id")
        section_title = section.get("title", section_id)
        fields = section.get("fields") or []
        if not fields:
            continue

        cov_entry = coverage.get(section_id, {})
        sc_entry = (section_content or {}).get(section_id, {}) or {}
        raw_sentences = sc_entry.get("sentences", [])
        if not raw_sentences:
            # context_mapper.py populates section_content[id]['sentences'] and
            # section_content[id]['blocks'] via two independent mechanisms
            # (a multi-label classification loop vs. detect_gaps()'s
            # single-label topic-block grouping) that can disagree — a section
            # can have real coverage (blocks) while ['sentences'] stays empty.
            # Flatten blocks' sentences as a fallback so this doesn't silently
            # fall through to the coarse polished-content path below.
            raw_sentences = [
                s for block in (sc_entry.get("blocks") or [])
                for s in (block.get("sentences") or [])
            ]
        raw_sentence_texts = [
            s.get("text", "") for s in raw_sentences
            if isinstance(s, dict) and s.get("text", "").strip()
        ]

        if raw_sentence_texts:
            # Real per-sentence transcript text — gives the type:"table"
            # fallback and _extract_by_semantic() fine-grained candidates
            # instead of a handful of large LLM-polished paragraph blocks
            # (coverage[id]['content']), which caused different fields in the
            # same section to collide onto the same broad text (e.g.
            # day1_survival_checklist's required_access ending up with
            # first_safe_actions' text). Joined with "\n" (not " ") so the
            # line-based table-extraction heuristic below treats each
            # sentence as its own candidate row.
            section_text = "\n".join(raw_sentence_texts)
            sentences = raw_sentence_texts
        else:
            # Fallback for the rare case section_content wasn't threaded
            # through for this section.
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
            raw_sentence_texts=raw_sentence_texts,
            output=result[section_id],
            llm_provider=llm_provider,
            embedding_model=embedding_model,
            # Shared across every field (including nested group fields) in
            # this section, so a second type:"table" field can't fall back
            # onto the exact same generic lines[:10] slice the first one
            # already claimed — see _populate_fields_recursive's _emit table
            # handling below.
            used_line_keys=set(),
        )

    return result


def find_source_sentence_index(value: Any, raw_sentence_texts: List[str]) -> Optional[int]:
    """Best-effort: which raw sentence a field's extracted value most likely
    came from, for precise per-fact evidence (see populate_fields docstring).
    Returns None (not a guess) when nothing lines up well enough — a wrong
    index would be worse than knowledge_builder's generic fallback, not better.

    Public (no leading underscore) — also used by ai.wrap_structured_as_fields()
    for the structured-extraction sections (security_controls, disaster_recovery,
    ownership_escalation, monitoring_observability), which never go through
    populate_fields() at all since they have no "fields" array in the schema.
    """
    if not value or not raw_sentence_texts:
        return None
    value_str = str(value).strip().lower()
    if len(value_str) < 3:
        return None
    # Pattern extractors sometimes join multiple matches with ", " (e.g. a
    # tool list) — no single sentence may contain the whole joined string, so
    # also try each part on its own.
    candidates = [value_str] + [v.strip() for v in value_str.split(",") if len(v.strip()) >= 3]
    for idx, text in enumerate(raw_sentence_texts):
        text_lower = text.lower()
        if any(c in text_lower for c in candidates):
            return idx
    return None


def _populate_fields_recursive(
    fields: List[Dict[str, Any]],
    section_id: str,
    section_title: str,
    section_text: str,
    sentences: List[str],
    output: Dict[str, Any],
    llm_provider=None,
    embedding_model=None,
    raw_sentence_texts: Optional[List[str]] = None,
    used_line_keys: Optional[Set[str]] = None,
    already_captured: Optional[List[str]] = None,
):
    raw_sentence_texts = raw_sentence_texts or []
    used_line_keys = used_line_keys if used_line_keys is not None else set()
    # Shared across every field (including nested group fields) in this
    # section, so the LLM gap-fill prompt for a later field can see what
    # earlier sibling fields already claimed and avoid copying it onto a
    # different, unrelated entity/field. See _build_llm_gap_fill_prompt.
    already_captured = already_captured if already_captured is not None else []

    def _emit(value: Any, confidence: float, source: str):
        entry = {"value": value, "confidence": confidence, "source": source}
        idx = find_source_sentence_index(value, raw_sentence_texts)
        if idx is not None:
            entry["source_chunk_index"] = idx
        return entry

    for field in fields:
        field_id = field.get("id")
        field_type = field.get("type", "text")
        field_label = field.get("label", field_id)

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
                raw_sentence_texts=raw_sentence_texts,
                used_line_keys=used_line_keys,
                already_captured=already_captured,
            )
            continue

        if field_type == "table":
            value = _extract_by_pattern(field, section_text, exclude_line_keys=used_line_keys)
            if value is not None:
                used_line_keys.update(_normalize_line(line) for line in value.splitlines())
                output[field_id] = _emit(value, 0.90, "pattern")
                already_captured.append(f"{field_label}: {value}")
                continue
        else:
            value = _extract_by_pattern(field, section_text)
            if value is not None:
                output[field_id] = _emit(value, 0.90, "pattern")
                already_captured.append(f"{field_label}: {value}")
                continue

        if field_type == "text" and sentences:
            value = _extract_by_semantic(field, sentences, embedding_model, exclude_line_keys=used_line_keys)
            if value is not None:
                used_line_keys.add(_normalize_line(value))
                output[field_id] = _emit(value, 0.65, "semantic")
                already_captured.append(f"{field_label}: {value}")
                continue

        if llm_provider and section_text.strip() and field_type != "table":
            try:
                prompt = _build_llm_gap_fill_prompt(
                    section_title=section_title,
                    field=field,
                    available_text=section_text[:1500],
                    already_captured=already_captured,
                )
                raw = llm_provider.generate(prompt, temperature=0.1, max_output_tokens=128)
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()] if isinstance(raw, str) else []
                val = lines[0] if lines else ""
                basis = lines[1].upper() if len(lines) > 1 else "INFERRED"
                if val and val.upper() != "NOT_MENTIONED" and len(val) < 500:
                    # The gap-fill prompt forbids inventing ungrounded values,
                    # so a returned value is always an extraction — but only
                    # tag it "llm" (which knowledge_builder marks as
                    # *(inferred...)* in the rendered doc) when the model
                    # itself says it had to reason beyond what's literally
                    # stated. A directly-stated-but-paraphrased fact (e.g.
                    # "most business critical system" -> "High") is grounded,
                    # not inferred.
                    source = "llm" if basis.startswith("INFER") else "llm_explicit"
                    output[field_id] = _emit(val, 0.75, source)
                    already_captured.append(f"{field_label}: {val}")
                    continue
            except Exception as exc:
                logger.warning("LLM field fill failed for %s.%s: %s", section_id, field_id, exc)

        output[field_id] = {"value": "", "confidence": 0.0, "source": "unfilled"}
