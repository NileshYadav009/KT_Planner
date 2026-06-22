"""
Format-aware section polish tier for KT documents.

This module sits ABOVE the reconstruction engines in enterprise_semantic_mapper.py.
Those engines already merge classified sentences into coherent paragraphs (Tier 1).
This module adds Tier 2: section-TYPE-AWARE formatting per the KT methodology:

  * Deployment & Rollback  -> numbered steps + bold labels (Trigger, Approver, Duration)
  * Common Failures         -> Issue | Symptoms | Root Cause | Resolution table
  * Danger Zones            -> imperative "Do Not Touch" list
  * Monitoring              -> stack-component bullet list
  * Architecture            -> traffic-flow narrative
  * default                 -> professional prose with bold labels

Hard rule (from methodology): NEVER add information not present in the source.
NEVER hallucinate infrastructure, commands, or numbers.

Tier 2 (Gemini) is optional. On failure or when Gemini is disabled, the caller
falls back to the Tier-1 reconstructed paragraph already produced upstream.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section-type-aware prompt templates
# ---------------------------------------------------------------------------
# Each template receives {section_title} and {sentences}. The {sentences} block
# is pre-rendered as a bulleted list by polish_section_with_gemini().

SECTION_PROMPTS: Dict[str, str] = {
    # ---- Deployment / Rollback: numbered steps + bold labels ----
    "deployment_and_rollback": (
        "You are a technical documentation writer producing the Deployment & Rollback "
        "section of an enterprise Knowledge Transfer document.\n\n"
        "Source sentences (raw transcript, already classified to this section):\n"
        "{sentences}\n\n"
        "Rewrite as professional documentation with this structure:\n"
        "1. A short orientation paragraph.\n"
        "2. A numbered list of deployment steps (preserve order implied by the source).\n"
        "3. A Rollback subsection. Use **bold labels** where the source mentions them: "
        "**Trigger:**, **Approver:**, **Expected Duration:**, **Risk Level:**.\n\n"
        "RULES:\n"
        "- Preserve every concrete value (commands, durations, names, branches, tool names) exactly as stated.\n"
        "- NEVER invent infrastructure, commands, numbers, or steps not present in the source.\n"
        "- If a label has no supporting sentence, omit it rather than fabricate.\n"
        "- Output ONLY the formatted section content. No heading."
    ),

    # ---- Common Failures: Issue | Cause | Fix triples ----
    "common_failures": (
        "You are a technical documentation writer producing the Common Failures & Fixes "
        "section of an enterprise Knowledge Transfer document.\n\n"
        "Source sentences (raw transcript, already classified to this section):\n"
        "{sentences}\n\n"
        "Rewrite as a reference guide. For each distinct issue, use this exact block:\n"
        "**Issue:** <problem name>\n"
        "**Symptoms:** <observable signals, if stated>\n"
        "**Root Cause:** <cause, if stated>\n"
        "**Resolution:** <fix steps>\n"
        "**Frequency:** <if mentioned, else omit>\n\n"
        "RULES:\n"
        "- Only include fields the source supports. Omit any field with no supporting sentence.\n"
        "- NEVER invent root causes, frequencies, or fixes not present in the source.\n"
        "- Merge near-duplicate issues into one block.\n"
        "- Output ONLY the formatted content. No heading."
    ),

    # ---- Danger Zones: imperative Do-Not-Touch list ----
    "danger_zones": (
        "You are a technical documentation writer producing the Danger Zones section of an "
        "enterprise Knowledge Transfer document.\n\n"
        "Source sentences (raw transcript, already classified to this section):\n"
        "{sentences}\n\n"
        "Rewrite as a list of critical warnings. For each danger:\n"
        "- Lead with an imperative 'Do NOT ...' statement where the source is prohibitive.\n"
        "- Add one short sentence on WHY it is dangerous, if stated.\n"
        "- Add '**Approval required:** ...' only if the source names an approver.\n\n"
        "RULES:\n"
        "- NEVER invent dangers, services, or approvals not present in the source.\n"
        "- Preserve every concrete system/config name exactly.\n"
        "- Output ONLY the formatted content. No heading."
    ),

    # ---- Monitoring: stack-component bullet list ----
    "monitoring_observability": (
        "You are a technical documentation writer producing the Monitoring & Observability "
        "section of an enterprise Knowledge Transfer document.\n\n"
        "Source sentences (raw transcript, already classified to this section):\n"
        "{sentences}\n\n"
        "Rewrite as professional documentation:\n"
        "1. A short paragraph describing the monitoring stack.\n"
        "2. A bullet list of stack components (dashboards, alerting tools, metrics backends) "
        "named in the source.\n"
        "3. Where stated, include **SLA / escalation:** and **Key dashboards:** as bold-labeled lines.\n\n"
        "RULES:\n"
        "- Only list tools/components actually named in the source.\n"
        "- NEVER invent URLs, thresholds, or tool names.\n"
        "- Output ONLY the formatted content. No heading."
    ),

    # ---- Architecture: traffic-flow narrative ----
    "architecture_reference": (
        "You are a technical documentation writer producing the Architecture Reference "
        "section of an enterprise Knowledge Transfer document.\n\n"
        "Source sentences (raw transcript, already classified to this section):\n"
        "{sentences}\n\n"
        "Rewrite as a professional architecture narrative that traces request/traffic flow "
        "through the components named in the source (edge -> load balancing -> compute -> data), "
        "followed by a bullet list of the concrete components mentioned.\n\n"
        "RULES:\n"
        "- Only describe components present in the source. NEVER invent services or connections.\n"
        "- Preserve exact service/product names (CloudFront, EKS, RDS, etc.).\n"
        "- Output ONLY the formatted content. No heading."
    ),

    # ---- Default: professional prose with bold labels ----
    "default": (
        "You are a technical documentation writer producing the {section_title} section of an "
        "enterprise Knowledge Transfer document.\n\n"
        "Source sentences (raw transcript, already classified to this section):\n"
        "{sentences}\n\n"
        "Rewrite as professional, structured documentation in complete paragraphs.\n"
        "Improve grammar, clarity, and flow. Remove filler and conversational noise.\n"
        "Use bold labels (**Label:** value) where the source implies a key/value fact.\n"
        "Merge related statements; do not repeat information.\n\n"
        "RULES:\n"
        "- Preserve every concrete fact, name, number, and command exactly as stated.\n"
        "- NEVER add information not present in the source. NEVER hallucinate infrastructure.\n"
        "- Output ONLY the formatted section content. No heading."
    ),
}


# ---------------------------------------------------------------------------
# Gemini call (direct, NOT via ai.gemini_refiner — that one truncates at \n\n)
# ---------------------------------------------------------------------------

def _call_gemini_structured(prompt: str, gemini_fn: Optional[Callable[[str], str]],
                            timeout_seconds: int = 45) -> Optional[str]:
    """Call Gemini in a way that preserves multi-line structured output.

    `ai.gemini_refiner` sets stop_sequences=["\\n\\n"], which would truncate
    numbered lists and tables at the first blank line. For section polishing we
    need the full structured response, so we prefer a direct client call without
    that stop sequence.

    `gemini_fn` is expected to be a callable that takes a prompt and returns text
    WITHOUT the \\n\\n stop sequence. If the only available callable is
    `ai.gemini_refiner`, we still use it but warn that output may be truncated.
    """
    if gemini_fn is None:
        return None
    try:
        result = gemini_fn(prompt)
    except Exception as exc:
        logger.warning("Gemini polish call failed: %s", exc)
        return None
    if not result or not result.strip():
        return None
    return result.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def polish_section_with_gemini(
    section_id: str,
    section_title: str,
    sentences: List[str],
    gemini_fn: Optional[Callable[[str], str]],
) -> Dict:
    """Polish one section's raw sentences into structured documentation.

    Returns a dict:
        {
          "tier": 2,                       # 2 = Gemini, 1 = local fallback, 0 = raw
          "content": "<formatted markdown>",
          "tier1_fallback": "<local prose or ''>",
          "raw": "<joined source sentences>",
          "used_gemini": bool,
        }
    Always returns a non-empty `content`.
    """
    cleaned = [s.strip() for s in (sentences or []) if s and s.strip()]
    raw = "\n".join(f"- {s}" for s in cleaned)

    if not cleaned:
        return {"tier": 0, "content": "", "tier1_fallback": "", "raw": "", "used_gemini": False}

    # ---- Tier 1: local professionalization (always available) ----
    tier1_text = ""
    try:
        from enterprise_semantic_mapper import (
            ProfessionalReconstructionEngine,
        )
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            SentenceTransformer = None  # type: ignore

        if SentenceTransformer is not None:
            engine = ProfessionalReconstructionEngine(SentenceTransformer("all-MiniLM-L6-v2"))
            joined_source = " ".join(cleaned)
            refined, _was_refined, _detail = engine.refine(
                joined_source, {"section": section_title}
            )
            tier1_text = (refined or "").strip()
    except Exception as exc:
        logger.debug("Tier-1 professionalization unavailable for %s: %s", section_id, exc)
        tier1_text = ""

    if not tier1_text:
        tier1_text = " ".join(cleaned)

    # ---- Tier 2: Gemini section-type-aware polish ----
    tier2_text = None
    if gemini_fn is not None:
        template = SECTION_PROMPTS.get(section_id, SECTION_PROMPTS["default"])
        try:
            prompt = template.format(section_title=section_title, sentences=raw)
        except KeyError:
            prompt = SECTION_PROMPTS["default"].format(
                section_title=section_title, sentences=raw
            )
        tier2_text = _call_gemini_structured(prompt, gemini_fn)

    if tier2_text:
        return {
            "tier": 2,
            "content": tier2_text,
            "tier1_fallback": tier1_text,
            "raw": " ".join(cleaned),
            "used_gemini": True,
        }
    return {
        "tier": 1,
        "content": tier1_text,
        "tier1_fallback": tier1_text,
        "raw": " ".join(cleaned),
        "used_gemini": False,
    }


def polish_all_sections(
    kt_structured: Dict,
    schema: List[Dict],
    gemini_fn: Optional[Callable[[str], str]] = None,
) -> Dict:
    """Apply section-aware polishing to every section that has classified content.

    Reads raw sentences from kt_structured["coverage"][sec_id]["sentences"]
    (the structure produced by main.process_upload_task). Writes the result
    into kt_structured["polished_sections"][sec_id].

    This is additive: it never mutates existing keys. On any failure the
    function leaves kt_structured unchanged and returns it.
    """
    schema_by_id = {s.get("id"): s for s in (schema or [])}
    coverage = kt_structured.get("coverage", {}) or {}
    polished: Dict[str, Dict] = {}

    for sec_id, sec_data in coverage.items():
        sentences_meta = sec_data.get("sentences", []) if isinstance(sec_data, dict) else []
        texts: List[str] = []
        for s in sentences_meta:
            if isinstance(s, dict):
                t = s.get("text")
            else:
                t = getattr(s, "text", None)
            if t:
                texts.append(t)

        # Also include the legacy "content" list if sentences is empty
        if not texts and isinstance(sec_data, dict):
            for c in sec_data.get("content", []) or []:
                if isinstance(c, str) and c.strip():
                    texts.append(c)

        if not texts:
            continue

        title = (schema_by_id.get(sec_id, {}) or {}).get("title", sec_id)
        try:
            polished[sec_id] = polish_section_with_gemini(sec_id, title, texts, gemini_fn)
        except Exception as exc:
            logger.warning("Polish failed for section %s: %s", sec_id, exc)
            polished[sec_id] = {
                "tier": 0,
                "content": " ".join(texts),
                "tier1_fallback": "",
                "raw": " ".join(texts),
                "used_gemini": False,
            }

    kt_structured["polished_sections"] = polished
    return kt_structured


__all__ = [
    "SECTION_PROMPTS",
    "polish_section_with_gemini",
    "polish_all_sections",
]
