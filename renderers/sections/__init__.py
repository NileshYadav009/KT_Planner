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
from renderers.sections.day1 import render as render_day1
from renderers.sections.default import render as render_default

RENDERER_REGISTRY = {
    "system_overview": render_system_overview,
    "architecture_reference": render_architecture_reference,
    "deployment_and_rollback": render_deployment,
    "monitoring_observability": render_common_monitoring,
    "common_failures": render_common_failures,
    "security_controls": render_security,
    "ownership_escalation": render_ownership,
    "day1_survival_checklist": render_day1,
    "plain_english_notes": render_generic,
    "disaster_recovery": render_disaster_recovery,
    "cost_optimization": render_cost_optimization,
    "danger_zones": render_danger_zones,
    "known_bad_days": render_known_bad_days,
    "open_responsibilities": render_open_responsibilities,
    "handover_completion": render_handover_completion,
    "first_30_day_ownership": render_first_30_day_ownership,
    "architecture_reference": render_architecture_reference,
    "signoff": render_generic,
}


def get_renderer(section_id: str):
    return RENDERER_REGISTRY.get(section_id, render_default)
