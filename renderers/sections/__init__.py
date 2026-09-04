import logging

from renderers.sections.system_overview import render as render_system_overview
from renderers.sections.architecture import render as render_architecture
from renderers.sections.architecture_reference import render as render_architecture_reference
from renderers.sections.deployment import render as render_deployment
from renderers.sections.common import (
    render_monitoring as render_common_monitoring,
    render_danger_zones,
    render_generic,
)
from renderers.sections.common_failures import render as render_common_failures
from renderers.sections.disaster_recovery import render as render_disaster_recovery
from renderers.sections.first_30_day_ownership import render as render_first_30_day_ownership
from renderers.sections.handover_completion import render as render_handover_completion
from renderers.sections.known_bad_days import render as render_known_bad_days
from renderers.sections.open_responsibilities import render as render_open_responsibilities
from renderers.sections.cost_optimization import render as render_cost_optimization
from renderers.sections.security import render as render_security
from renderers.sections.ownership import render as render_ownership
from renderers.sections.ownership_escalation import render as render_ownership_escalation
from renderers.sections.day1 import render as render_day1
from renderers.sections.default import render as render_default

logger = logging.getLogger(__name__)

RENDERER_REGISTRY = {
    "system_overview": render_system_overview,
    "architecture_reference": render_architecture_reference,
    "deployment_and_rollback": render_deployment,
    "monitoring_observability": render_common_monitoring,
    "common_failures": render_common_failures,
    "security_controls": render_security,
    "ownership_escalation": render_ownership_escalation,
    "day1_survival_checklist": render_day1,
    "plain_english_notes": render_generic,
    "disaster_recovery": render_disaster_recovery,
    "cost_optimization": render_cost_optimization,
    "danger_zones": render_danger_zones,
    "known_bad_days": render_known_bad_days,
    "open_responsibilities": render_open_responsibilities,
    "handover_completion": render_handover_completion,
    "first_30_day_ownership": render_first_30_day_ownership,
    "signoff": render_generic,
}


def get_renderer(section_id: str):
    return RENDERER_REGISTRY.get(section_id, render_default)


def validate_renderer_registry(schema_sections: list) -> None:
    """
    Startup validation: Assert that every schema section id with a dedicated 
    renderer file has a matching key in RENDERER_REGISTRY.
    
    This catches silent-fallback bugs where get_renderer() returns None/default
    instead of the intended specialized renderer.
    
    Args:
        schema_sections: List of section dicts from kt_schema_new.json
        
    Raises:
        AssertionError: If any schema section id is missing from registry
    """
    registry_keys = set(RENDERER_REGISTRY.keys())
    schema_ids = {sec["id"] for sec in schema_sections}

    # Sections where we intentionally built a dedicated renderer file.
    # These MUST also be section ids that actually exist in the live schema —
    # a mismatch here means get_renderer() silently falls back to render_default
    # for a section every KT document generates.
    sections_with_renderers = {
        "system_overview", "architecture_reference", "deployment_and_rollback",
        "monitoring_observability", "common_failures", "security_controls",
        "ownership_escalation", "day1_survival_checklist", "disaster_recovery",
        "cost_optimization", "danger_zones", "known_bad_days",
        "open_responsibilities", "handover_completion", "first_30_day_ownership"
    }

    for section_id in sections_with_renderers:
        if section_id not in registry_keys:
            raise AssertionError(
                f"[RENDERER REGISTRY] Section '{section_id}' has a dedicated "
                f"renderer file but is NOT in RENDERER_REGISTRY. This will cause "
                f"silent fallback to default narrative rendering. "
                f"Add entry: '{section_id}': render_{section_id.replace('_', '_')}"
            )
        if section_id not in schema_ids:
            raise AssertionError(
                f"[RENDERER REGISTRY] '{section_id}' is in RENDERER_REGISTRY with a "
                f"dedicated renderer, but no section in kt_schema_new.json has that id "
                f"(schema ids: {sorted(schema_ids)}). The dedicated renderer will never "
                f"fire for this section — fix the schema id or the registry key."
            )

    logger.info(
        f"✓ Renderer registry validated: {len(sections_with_renderers)} "
        f"specialized renderers registered and matched to schema sections."
    )
