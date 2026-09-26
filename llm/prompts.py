"""Prompt templates for LLM-driven section extraction and polishing.

Pure data, no logic — the functions that build and send these prompts
(`_extract_structured_section`, `polish_coverage_sections`, etc.) live in ai.py
and import from here. Extracted out of ai.py as part of the Phase 3
architecture split (see REPOSITORY_AUDIT.md).
"""

# Sections that use structured JSON extraction instead of plain text polishing.
#
# Keys MUST be real kt_schema_new.json section ids — this dict used to be keyed
# "monitoring" instead of "monitoring_observability", which meant the whole
# feature silently never fired (see REPOSITORY_AUDIT.md's Phase 6 notes; same
# class of bug as the first_30_day_plan/first_30_day_ownership mismatch fixed
# earlier this session).
#
# For security_controls/disaster_recovery/ownership_escalation, values are
# returned as flat strings (not arrays) — multi-item answers joined with "\n"
# (or " -> " for escalation_chain) directly by the model, because that's
# exactly the shape renderers/sections/{security,disaster_recovery,
# ownership_escalation}.py already expect from `fields[id]["value"]`
# (e.g. `str(fields["rto_steps"]["value"]).split("\n")`). This avoids any
# per-section join/format logic in ai.py — the prompt does that work.
SECTION_STRUCTURED_PROMPTS = {
    "monitoring_observability": (
        "Return ONLY valid JSON (no markdown, no extra text):\n"
        "{{\"tools\": [string], \"first_response_steps\": [string], \"alert_routing\": string|null}}\n\n"
        "Extract from these fragments:\n{fragments}\n\n"
        "JSON:"
    ),
    "security_controls": (
        "Return ONLY valid JSON (no markdown, no extra text). Use null for anything not mentioned:\n"
        "{{\"security_scan_config\": string|null, \"vault_configuration\": string|null, \"security_issues\": string|null}}\n\n"
        "- security_scan_config: the scanning tool and what it scans (e.g. \"Trivy for container image scanning\")\n"
        "- vault_configuration: how secrets are managed (e.g. \"HashiCorp Vault\")\n"
        "- security_issues: known security concerns or gaps, if any\n"
        "Do NOT invent tools or issues not present in the fragments.\n\n"
        "Extract from these fragments:\n{fragments}\n\n"
        "JSON:"
    ),
    "disaster_recovery": (
        "Return ONLY valid JSON (no markdown, no extra text). Use null for anything not mentioned:\n"
        "{{\"rto_steps\": string|null, \"rpo_steps\": string|null, \"known_failure_scenarios\": string|null, "
        "\"recovery_contact\": string|null, \"dr_testing_frequency\": string|null}}\n\n"
        "- rto_steps: recovery time objective steps/procedure, each step on its own line (\\n-separated)\n"
        "- rpo_steps: recovery point objective / backup details (schedule, retention), each item on its "
        "own line (\\n-separated)\n"
        "- known_failure_scenarios: known DR failure scenarios, each on its own line (\\n-separated)\n"
        "- recovery_contact: who to contact for recovery, if mentioned\n"
        "- dr_testing_frequency: how often DR/recovery testing itself is performed (e.g. \"quarterly\", "
        "\"twice per year\") — this is separate from rpo_steps' backup schedule; do not merge the two or "
        "drop this if the fragments state it alongside a backup/retention fact in the same sentence\n"
        "Do NOT invent RTO/RPO values or contacts not present in the fragments.\n\n"
        "Extract from these fragments:\n{fragments}\n\n"
        "JSON:"
    ),
    "ownership_escalation": (
        "Return ONLY valid JSON (no markdown, no extra text). Use null for anything not mentioned:\n"
        "{{\"oncall_tool\": string|null, \"escalation_chain\": string|null, \"application_ownership\": string|null, "
        "\"infrastructure_ownership\": string|null, \"operational_escalation_guidance\": string|null}}\n\n"
        "- oncall_tool: the NAME of the on-call/paging tool itself (e.g. \"PagerDuty\"), never a sentence\n"
        "- escalation_chain: the escalation steps in order, joined as \"Step 1 -> Step 2 -> Step 3\"\n"
        "- application_ownership: who owns the application code (e.g. \"Developers\")\n"
        "- infrastructure_ownership: who owns the infrastructure (e.g. \"Platform Engineers\")\n"
        "- operational_escalation_guidance: a general instruction for WHEN/HOW to reach out that is not "
        "itself a tool name, an escalation chain, or a stated owner (e.g. \"If you are unsure about a "
        "change, involve the appropriate platform or application owner.\") — this is guidance about what "
        "to do, not a statement of who formally owns the system; never put this kind of sentence into "
        "oncall_tool just because it mentions contacting someone\n"
        "Do NOT invent owners or escalation steps not present in the fragments.\n\n"
        "Extract from these fragments:\n{fragments}\n\n"
        "JSON:"
    ),
    "cost_optimization": (
        "Return ONLY valid JSON (no markdown, no extra text):\n"
        "{{\"levers\": [{{\"lever\": string, \"detail\": string}}]}}\n\n"
        "- lever: the cost optimization technique (e.g. \"Spot instances\", \"Scheduled scaling\")\n"
        "- detail: what it applies to and any specifics mentioned\n"
        "One entry per distinct lever actually discussed. Do NOT invent levers not "
        "present in the fragments.\n\n"
        "Extract from these fragments:\n{fragments}\n\nJSON:"
    ),
    "open_responsibilities": (
        "Return ONLY valid JSON (no markdown, no extra text):\n"
        "{{\"open_tasks\": [{{\"task\": string, \"type\": string|null, \"status\": string|null, "
        "\"business_impact\": string|null, \"kt_done\": string|null, \"recommendation\": string|null, "
        "\"owner_decision\": string|null}}], "
        "\"recurring_responsibilities\": [{{\"activity\": string, \"frequency\": string|null, "
        "\"trigger\": string|null, \"owner_before\": string|null, \"owner_after\": string|null}}]}}\n\n"
        "A row belongs in \"open_tasks\" ONLY if it describes a concrete, "
        "specific piece of unfinished or in-progress work being handed over "
        "(e.g. a project, a pending decision, an unresolved ticket). A row "
        "belongs in \"recurring_responsibilities\" ONLY if it describes an "
        "ongoing operational duty done on a schedule or trigger.\n"
        "Do NOT include general safety warnings, escalation/contact "
        "instructions, danger-zone cautions, or other advice that isn't a "
        "specific task or recurring duty — those belong elsewhere in the "
        "document, not here. If nothing in the fragments is actually a task "
        "or recurring responsibility, return empty arrays for both.\n"
        "Do NOT invent tasks not present in the fragments.\n\n"
        "Extract from these fragments:\n{fragments}\n\nJSON:"
    ),
    "common_failures": (
        "Return ONLY valid JSON (no markdown, no extra text). Use null for anything not mentioned:\n"
        "{{\"failures\": [{{\"symptom\": string, \"condition\": string|null, "
        "\"cause\": string|null, "
        "\"first_checks\": string|null, \"fix\": string|null, "
        "\"frequency\": string|null, \"ticket\": string|null, \"when\": string|null, "
        "\"impact\": string|null, \"resolution\": string|null, "
        "\"preventive_action\": string|null}}]}}\n\n"
        "\"condition\" is WHEN the failure occurs — the triggering circumstance "
        "stated alongside it (\"during high traffic\", \"after a deployment\", "
        "\"during market-open bursts\", \"at month end\"). Keep it OUT of "
        "\"symptom\": the symptom is the failure itself. Never discard the "
        "condition — knowing when a failure strikes is often the most useful "
        "part of the entry, and it is routinely stated in the same sentence.\n"
        "\"first_checks\" is what to LOOK AT first to diagnose this specific issue "
        "(metrics, dashboards, logs, health indicators) when the transcript states "
        "them — e.g. \"Azure SQL connection exhaustion during high traffic. Check "
        "active connections and connection pool metrics.\" -> "
        "{{\"symptom\": \"Azure SQL connection exhaustion\", \"condition\": \"During "
        "high traffic\", \"first_checks\": \"Active "
        "connections; connection pool metrics\", \"fix\": null}}. Multiple checks are "
        "joined with \"; \". A diagnostic check is NOT a fix: record it here and leave "
        "\"fix\" null unless an actual remediation was stated. Never drop a stated "
        "first check just because no remediation followed it.\n"
        "One entry per distinct failure/issue described. \"frequency\" is how often it "
        "happens if mentioned (e.g. \"weekly\", \"during deploys\"). \"ticket\" is a "
        "KEDB/ticket reference if mentioned. \"when\" is a specific past OCCURRENCE — "
        "an actual time reference (e.g. \"last year\", \"in March\", \"during the Black "
        "Friday launch\") — for one-off historical incidents rather than recurring "
        "issues; never a description of severity or how notable the incident was (a "
        "phrase like \"a previous major incident\" or \"a major outage\" is NOT a time "
        "reference — that belongs in \"impact\", not \"when\", and if no actual time "
        "reference was stated, \"when\" must be null). \"impact\" is the concrete "
        "EFFECT the incident had (e.g. \"major outage\", \"customers could not place "
        "orders for 2 hours\") — never just a restatement of the symptom/incident name "
        "itself. Example: transcript says \"A previous major incident was caused by "
        "Redis memory saturation\" with no further detail -> "
        "{{\"symptom\": \"Redis memory saturation\", \"when\": null, \"impact\": null, "
        "\"cause\": null}} (severity language alone doesn't establish a time or a "
        "concrete effect — leave both null rather than reusing the incident's own "
        "description). \"cause\", \"fix\", \"resolution\" and \"preventive_action\" are "
        "each only set if the transcript explicitly states that specific thing — a "
        "stated symptom does NOT imply a stated cause or fix. Never fill any of these "
        "four with a plausible-sounding root cause or remediation step drawn from your "
        "own general troubleshooting knowledge; leave the field null instead. It is "
        "normal and expected for most of these four fields to be null. Do NOT invent "
        "failures not present in the fragments.\n\n"
        "Extract from these fragments:\n{fragments}\n\nJSON:"
    ),
}

SECTION_POLISH_PROMPTS = {
    "deployment_and_rollback": (
        "You are a senior technical writer producing a KT document.\n"
        "Section: {title}\n\n"
        "Rewrite these transcript fragments into a structured deployment reference.\n"
        "Format rules:\n"
        "- Deployment steps: numbered list (1. 2. 3.)\n"
        "- Use BOLD labels: **Trigger:**, **Window:**, **Approver:**, **Duration:**\n"
        "- Rollback: separate subsection with **Rollback trigger:**, **Action:**, "
        "**Target time:**\n"
        "- Keep all specific values (tool names, times, branch names) exactly as given.\n"
        "- Do NOT invent information not present in the input.\n"
        "Return only the formatted section content, no heading.\n\n"
        "Source fragments:\n{fragments}\n\nFormatted output:"
    ),
    "common_failures": (
        "You are a senior technical writer producing a KT document.\n"
        "Section: {title}\n\n"
        "Rewrite these fragments into a structured failure reference.\n"
        "For each distinct issue use this format:\n"
        "**Issue:** [name]\n"
        "**Cause:** [root cause]\n"
        "**Fix:** [resolution steps]\n"
        "**Frequency:** [if mentioned]\n\n"
        "Group related fragments into one issue block.\n"
        "Do NOT add issues not present in the input.\n"
        "Return only the formatted content, no heading.\n\n"
        "Source fragments:\n{fragments}\n\nFormatted output:"
    ),
    "ownership_escalation": (
        "You are a senior technical writer producing a KT document.\n"
        "Section: {title}\n\n"
        "Rewrite these fragments into an ownership and escalation reference.\n"
        "Format:\n"
        "**Application ownership:** [team]\n"
        "**Infrastructure ownership:** [team]\n\n"
        "**Escalation chain:**\n1. [first contact]\n2. [second]\n3. [third]\n\n"
        "**Contact channel:** [Slack / PagerDuty / email if mentioned]\n\n"
        "Do NOT add contacts not present in the input.\n"
        "Return only the formatted content, no heading.\n\n"
        "Source fragments:\n{fragments}\n\nFormatted output:"
    ),
    "monitoring_observability": (
        "You are a senior technical writer producing a KT document.\n"
        "Section: {title}\n\n"
        "Rewrite these fragments into a monitoring reference.\n"
        "Use this EXACT format with each item on its own line:\n\n"
        "**Monitoring stack:**\n"
        "- [tool name]: [what it monitors]\n"
        "- [tool name]: [what it monitors]\n\n"
        "**First response steps:**\n"
        "1. [step]\n"
        "2. [step]\n\n"
        "**Alert routing:** [tool and channel if mentioned]\n\n"
        "Do NOT put multiple items on the same line.\n"
        "Do NOT add information not in the input.\n"
        "Return only the formatted content, no heading.\n\n"
        "Source fragments:\n{fragments}\n\nFormatted output:"
    ),
    "disaster_recovery": (
        "You are a senior technical writer producing a KT document.\n"
        "Section: {title}\n\n"
        "Rewrite these fragments into a DR reference.\n"
        "Format:\n"
        "**Recovery procedure:**\n1. [step]\n\n"
        "**Backup policy:**\n"
        "- Schedule: [if mentioned]\n"
        "- Retention: [if mentioned]\n\n"
        "**DR testing:** [frequency if mentioned]\n\n"
        "**RTO / RPO:** [if mentioned]\n\n"
        "Do NOT add facts not in the input.\n"
        "Return only the formatted content, no heading.\n\n"
        "Source fragments:\n{fragments}\n\nFormatted output:"
    ),
    "security_controls": (
        "You are a senior technical writer producing a KT document.\n"
        "Section: {title}\n\n"
        "Rewrite these fragments into a security controls reference.\n"
        "Format:\n"
        "**Security controls:**\n"
        "- [Tool]: [what it does]\n\n"
        "Group by type if possible: Secret Management, Scanning, Access Control.\n"
        "Do NOT add controls not in the input.\n"
        "Return only the formatted content, no heading.\n\n"
        "Source fragments:\n{fragments}\n\nFormatted output:"
    ),
    "day1_survival_checklist": (
        "You are a senior technical writer producing a KT document.\n"
        "Section: {title}\n\n"
        "Rewrite these fragments into a Day-1 checklist.\n"
        "Format:\n"
        "**Access to request:**\n- [system]\n\n"
        "**Safe first actions (read-only):**\n- [action]\n\n"
        "**Do NOT do on Day 1:**\n- [action]\n\n"
        "Only include groups that have content from the input.\n"
        "Do NOT add items not present in the input.\n"
        "Return only the formatted content, no heading.\n\n"
        "Source fragments:\n{fragments}\n\nFormatted output:"
    ),
    "default": (
        "You are a senior technical writer for a Knowledge Transfer document.\n"
        "Section: {title}\n\n"
        "Rewrite these transcript fragments into clean, professional prose.\n"
        "Rules:\n"
        "- Fix grammar, casing, punctuation.\n"
        "- Preserve all facts, names, numbers exactly.\n"
        "- Do NOT add information not present in the input.\n"
        "- Return only the rewritten content, no heading.\n\n"
        "Source fragments:\n{fragments}\n\nPolished content:"
    ),
}
