"""Tests for field_populator.py — in particular the cross-field-contamination
regression (REPOSITORY_AUDIT.md §9n): populate_fields() must source real
per-sentence transcript text (falling back to flattened topic blocks when the
primary sentence list is empty), not the coarse LLM-polished paragraph blocks
that caused different fields in the same section to collide onto identical
text.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from field_populator import (
    populate_fields, find_source_sentence_index, _build_llm_gap_fill_prompt, _extract_by_semantic,
    _extract_by_pattern, extract_rto_rpo, PATTERN_EXTRACTORS,
)


SCHEMA = [
    {
        "id": "day1_survival_checklist",
        "title": "Day 1",
        "fields": [
            {"id": "required_access", "type": "table"},
            {"id": "first_safe_actions", "label": "Safe first actions", "type": "text"},
        ],
    }
]


def test_populate_fields_uses_raw_sentences_not_polished_content():
    # coverage['content'] deliberately carries the kind of LLM-polish markdown
    # this bug produced verbatim in the past — if the fix regresses, this
    # exact string would leak into populated field values.
    coverage = {
        "day1_survival_checklist": {
            "content": ["**Safe first actions (read-only):**\n- Reviewing Grafana dashboards"],
        }
    }
    section_content = {
        "day1_survival_checklist": {
            "sentences": [
                {"text": "Request access to the cloud console and git repository.", "start": 0, "end": 3, "speaker": None, "audio_confidence": 0.9},
                {"text": "Safe first actions include reviewing Grafana dashboards.", "start": 3, "end": 6, "speaker": None, "audio_confidence": 0.9},
            ],
        }
    }

    result = populate_fields(SCHEMA, coverage, llm_provider=None, embedding_model=None, section_content=section_content)
    fields = result["day1_survival_checklist"]

    for field_id in ("required_access", "first_safe_actions"):
        value = str(fields[field_id]["value"])
        assert "**" not in value, f"{field_id} leaked polished markdown: {value!r}"


def test_populate_fields_does_not_collide_two_distinct_fields():
    coverage = {"day1_survival_checklist": {"content": []}}
    section_content = {
        "day1_survival_checklist": {
            "sentences": [
                {"text": "Request access to the cloud console and git repository.", "start": 0, "end": 3, "speaker": None, "audio_confidence": 0.9},
                {"text": "Safe first actions include reviewing Grafana dashboards.", "start": 3, "end": 6, "speaker": None, "audio_confidence": 0.9},
            ],
        }
    }

    result = populate_fields(SCHEMA, coverage, llm_provider=None, embedding_model=None, section_content=section_content)
    fields = result["day1_survival_checklist"]
    # required_access (type: table) pattern-fallback joins all available raw
    # lines; first_safe_actions (semantic, no model) falls back to the first
    # sentence. They should not both end up as the identical polished blob
    # this bug produced — at minimum required_access must contain real
    # sentence text, not a duplicate of a differently-sourced value.
    assert "Request access" in fields["required_access"]["value"]


def test_populate_fields_falls_back_to_flattened_blocks_when_sentences_empty():
    # The deeper bug found underneath the first one: section_content[id]
    # ['sentences'] can be empty while ['blocks'] (topic-block grouping) has
    # real content for the same section — must not silently fall through to
    # the coarse coverage['content'] path when that happens.
    coverage = {"day1_survival_checklist": {"content": ["**should not be used**"]}}
    section_content = {
        "day1_survival_checklist": {
            "sentences": [],
            "blocks": [
                {
                    "sentences": [
                        {"text": "Request access to the cloud console.", "start": 0, "end": 3, "speaker": None, "audio_confidence": 0.9},
                    ]
                }
            ],
        }
    }

    result = populate_fields(SCHEMA, coverage, llm_provider=None, embedding_model=None, section_content=section_content)
    value = str(result["day1_survival_checklist"]["required_access"]["value"])
    assert "should not be used" not in value
    assert "Request access to the cloud console." in value


def test_populate_fields_falls_back_to_coverage_content_when_nothing_else_available():
    coverage = {"day1_survival_checklist": {"content": ["Only polished content is available here."]}}
    result = populate_fields(SCHEMA, coverage, llm_provider=None, embedding_model=None, section_content=None)
    value = str(result["day1_survival_checklist"]["required_access"]["value"])
    assert "Only polished content is available here." in value


def test_populate_fields_sets_source_chunk_index_when_value_matches_a_sentence():
    coverage = {"day1_survival_checklist": {"content": []}}
    section_content = {
        "day1_survival_checklist": {
            "sentences": [
                {"text": "Request access to the cloud console.", "start": 0, "end": 3, "speaker": None, "audio_confidence": 0.9},
            ],
        }
    }
    result = populate_fields(SCHEMA, coverage, llm_provider=None, embedding_model=None, section_content=section_content)
    field = result["day1_survival_checklist"]["required_access"]
    assert field.get("source_chunk_index") == 0


def test_find_source_sentence_index_matches_substring():
    sentences = ["We use Prometheus and Grafana.", "Rollback takes 15 minutes."]
    assert find_source_sentence_index("Prometheus", sentences) == 0
    assert find_source_sentence_index("15 minutes", sentences) == 1


def test_find_source_sentence_index_returns_none_for_no_match():
    sentences = ["We use Prometheus and Grafana."]
    assert find_source_sentence_index("Kubernetes", sentences) is None


def test_find_source_sentence_index_returns_none_for_short_or_empty_value():
    sentences = ["We use Prometheus and Grafana."]
    assert find_source_sentence_index("", sentences) is None
    assert find_source_sentence_index("Pr", sentences) is None
    assert find_source_sentence_index(None, sentences) is None


TWO_TABLE_FIELDS_SCHEMA = [
    {
        "id": "open_responsibilities",
        "title": "Open Responsibilities",
        "fields": [
            {"id": "open_tasks", "type": "table"},
            {"id": "recurring_responsibilities", "type": "table"},
        ],
    }
]


def test_two_table_fields_in_one_section_do_not_get_identical_fallback_content():
    # Regression test: before the fix, every type:"table" field independently
    # re-derived candidate lines from the WHOLE section text and fell back to
    # the same generic lines[:10] slice — so a second table field in the same
    # section (e.g. "Recurring responsibilities" alongside "Open tasks")
    # always duplicated the first one's content verbatim instead of getting
    # its own data or an honest "not mentioned".
    section_content = {
        "open_responsibilities": {
            "sentences": [
                {"text": "Contact platform engineering before making changes.", "start": 0, "end": 3, "speaker": None, "audio_confidence": 0.9},
                {"text": "Review the on-call runbook weekly.", "start": 3, "end": 6, "speaker": None, "audio_confidence": 0.9},
                {"text": "Renew the TLS certificate every quarter.", "start": 6, "end": 9, "speaker": None, "audio_confidence": 0.9},
            ],
        }
    }
    coverage = {"open_responsibilities": {"content": []}}

    result = populate_fields(
        TWO_TABLE_FIELDS_SCHEMA, coverage, llm_provider=None, embedding_model=None, section_content=section_content
    )
    fields = result["open_responsibilities"]

    open_tasks_value = fields["open_tasks"]["value"]
    recurring_value = fields["recurring_responsibilities"]["value"]

    assert open_tasks_value, "first table field should still get the available lines"
    assert open_tasks_value != recurring_value, (
        "second table field must not silently duplicate the first field's content"
    )
    # With only 3 short lines total and the first field consuming all of
    # them, the honest outcome for the second field is "not mentioned", not
    # a repeat of the same content.
    assert fields["recurring_responsibilities"]["source"] == "unfilled"


class _StubLLM:
    """Records every prompt it's called with and returns queued responses
    in order — lets a test assert both what the model was told and what
    happens with what it says back."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self._responses.pop(0)


EXPLICIT_INFERRED_SCHEMA = [
    {
        "id": "system_overview",
        "title": "System Overview",
        "fields": [
            {"id": "business_criticality", "label": "Business Criticality", "type": "single_select", "options": ["High", "Medium", "Low"]},
            {"id": "customer_reach", "label": "Customer Reach", "type": "text"},
        ],
    }
]


def test_llm_gap_fill_tags_explicit_basis_as_llm_explicit():
    # business_criticality's options ("high"/"medium"/"low") never appear
    # verbatim in the transcript, so pattern extraction misses it and this
    # falls to the LLM — which should recognize "most business critical
    # system" as a directly-stated (if paraphrased) fact, not a guess.
    coverage = {"system_overview": {"content": []}}
    section_content = {
        "system_overview": {
            "sentences": [
                {"text": "This is one of the company's most business critical systems.", "start": 0, "end": 3, "speaker": None, "audio_confidence": 0.9},
            ],
        }
    }
    stub = _StubLLM(["High\nEXPLICIT", "global\nEXPLICIT"])
    result = populate_fields(
        EXPLICIT_INFERRED_SCHEMA, coverage, llm_provider=stub, embedding_model=None, section_content=section_content
    )
    field = result["system_overview"]["business_criticality"]
    assert field["value"] == "High"
    assert field["source"] == "llm_explicit"


def test_llm_gap_fill_tags_inferred_basis_as_llm():
    coverage = {"system_overview": {"content": []}}
    section_content = {
        "system_overview": {
            "sentences": [
                {"text": "The team ships fairly often.", "start": 0, "end": 3, "speaker": None, "audio_confidence": 0.9},
            ],
        }
    }
    stub = _StubLLM(["Medium\nINFERRED", "NOT_MENTIONED\nINFERRED"])
    result = populate_fields(
        EXPLICIT_INFERRED_SCHEMA, coverage, llm_provider=stub, embedding_model=None, section_content=section_content
    )
    field = result["system_overview"]["business_criticality"]
    assert field["value"] == "Medium"
    assert field["source"] == "llm"


def test_llm_gap_fill_defaults_to_inferred_when_basis_line_missing():
    # If the model doesn't follow the two-line format, err on the safe
    # (pre-existing) side rather than silently treating an unparseable
    # response as explicit.
    coverage = {"system_overview": {"content": []}}
    section_content = {
        "system_overview": {
            "sentences": [
                {"text": "Some unrelated sentence.", "start": 0, "end": 3, "speaker": None, "audio_confidence": 0.9},
            ],
        }
    }
    stub = _StubLLM(["High", "NOT_MENTIONED"])
    result = populate_fields(
        EXPLICIT_INFERRED_SCHEMA, coverage, llm_provider=stub, embedding_model=None, section_content=section_content
    )
    field = result["system_overview"]["business_criticality"]
    assert field["value"] == "High"
    assert field["source"] == "llm"


SIBLING_ENV_SCHEMA = [
    {
        "id": "environments",
        "title": "Environments",
        "fields": [
            {"id": "production_notes", "label": "Production characteristics", "type": "text"},
            {"id": "staging_notes", "label": "Staging characteristics", "type": "text"},
            {"id": "non_production_notes", "label": "Non-production characteristics", "type": "text"},
        ],
    }
]


def test_llm_gap_fill_prompt_requires_literal_url_for_url_fields():
    # Regression test: architecture_reference's architecture_link (type
    # "url") once absorbed "The architecture diagram is maintained in
    # Confluence..." as if "Confluence" were a URL, because the general
    # paraphrase-friendly rule ("extract or normalize, even in different
    # words") applied to every field type — collapsing the whole rendered
    # section down to one line (see REPOSITORY_AUDIT.md). A url/date/
    # boolean field must get the strict literal-presence rule instead.
    field = {"id": "architecture_link", "label": "Link to detailed architecture documentation", "type": "url"}
    prompt = _build_llm_gap_fill_prompt("Architecture Reference", field, "The architecture diagram is maintained in Confluence.")
    assert "an actual URL" in prompt
    assert "does NOT count" in prompt
    assert "even in different words), extract or normalize" not in prompt


def test_llm_gap_fill_prompt_keeps_paraphrase_leniency_for_open_ended_fields():
    field = {"id": "business_criticality", "label": "Business Criticality", "type": "single_select", "options": ["High", "Medium", "Low"]}
    prompt = _build_llm_gap_fill_prompt("System Overview", field, "One of the company's most business critical systems.")
    assert "extract or normalize that value" in prompt
    assert "an actual URL" not in prompt


def test_llm_gap_fill_prompt_lists_already_captured_sibling_values():
    # Only one real sentence exists, and its subject is staging (it only
    # mentions "production" as a comparison target). production_notes
    # (first in schema order) must NOT claim it just by running first —
    # staging_notes should resolve it via semantic match instead, and the
    # non_production_notes LLM prompt (which fires after staging_notes has
    # already captured a value) must be told about that value so it doesn't
    # blindly re-attribute staging-specific text to a different environment.
    coverage = {"environments": {"content": []}}
    section_content = {
        "environments": {
            "sentences": [
                {"text": "The staging environment closely mirrors production.", "start": 0, "end": 3, "speaker": None, "audio_confidence": 0.9},
            ],
        }
    }
    stub = _StubLLM(["NOT_MENTIONED\nEXPLICIT", "NOT_MENTIONED\nEXPLICIT"])
    result = populate_fields(
        SIBLING_ENV_SCHEMA, coverage, llm_provider=stub, embedding_model=None, section_content=section_content
    )
    fields = result["environments"]
    assert fields["staging_notes"]["value"] == "The staging environment closely mirrors production."
    assert fields["staging_notes"]["source"] == "semantic"
    # production_notes and non_production_notes both correctly fall through
    # to LLM gap-fill instead of claiming a sentence that's really about
    # staging.
    assert len(stub.prompts) == 2
    assert any(
        "Already captured for OTHER fields" in p and "closely mirrors production" in p
        for p in stub.prompts
    )


def test_extract_by_semantic_identity_filter_picks_correct_sibling_by_first_mention():
    # Direct unit test of the disambiguation logic (no embedding model
    # needed — the identity filter runs before the model is even
    # consulted). The sentence names both "staging" and "production", but
    # "staging" is mentioned first, so it's the real subject.
    field = {"id": "production_notes", "label": "Production characteristics", "type": "text"}
    sentence = "The staging environment closely mirrors production."
    result = _extract_by_semantic(
        field, [sentence], model=None,
        own_identity_word="production", other_identity_words=["staging", "non-production"],
    )
    assert result is None  # disqualified — not claimed by production_notes

    staging_field = {"id": "staging_notes", "label": "Staging characteristics", "type": "text"}
    result = _extract_by_semantic(
        staging_field, [sentence], model=None,
        own_identity_word="staging", other_identity_words=["production", "non-production"],
    )
    assert result == sentence  # correctly claimed by staging_notes


def test_extract_by_semantic_identity_filter_is_noop_when_own_word_mentioned_first():
    field = {"id": "production_notes", "label": "Production characteristics", "type": "text"}
    sentence = "Production runs multi-AZ, unlike staging."
    result = _extract_by_semantic(
        field, [sentence], model=None,
        own_identity_word="production", other_identity_words=["staging", "non-production"],
    )
    assert result == sentence


# Regression tests for the system_name pattern extractor, found via a
# 4-transcript adversarial test pass: it either missed real names entirely
# (compound intro phrasing like "this is the handover for the X platform")
# or, worse, confidently grabbed the wrong thing (any "for the X." sentence
# anywhere in the section, not just a real name introduction, since bare
# "." / "," used to count as a valid terminator).
SYSTEM_NAME_FIELD = {"id": "system_name", "label": "System Name", "type": "text"}


def test_system_name_extracts_simple_handing_over_phrasing():
    text = "Hi, I'm handing over the Ledger Analytics platform today."
    assert _extract_by_pattern(SYSTEM_NAME_FIELD, text) == "Ledger Analytics"


def test_system_name_extracts_compound_handover_for_phrasing():
    # "this is the handover for the X platform" has 4 structural/filler
    # words between the trigger and the real name — a single optional
    # "the" wasn't enough, so this used to match nothing at all.
    text = "Good morning, this is the handover for the CorePay payments processing platform."
    assert _extract_by_pattern(SYSTEM_NAME_FIELD, text) == "Corepay Payments Processing"


def test_system_name_does_not_false_positive_on_unrelated_for_the_x_sentence():
    # "for the site." is an ordinary sentence fragment, not a name
    # introduction — must not be mistaken for one just because "for the X."
    # happens to match the old, too-permissive terminator list.
    text = "The platform trains and serves the product-recommendation models for the site."
    assert _extract_by_pattern(SYSTEM_NAME_FIELD, text) is None


def test_system_name_returns_none_when_no_proper_name_is_stated():
    text = "It's the old claims processing system, it's been around forever."
    assert _extract_by_pattern(SYSTEM_NAME_FIELD, text) is None


# Regression tests for a live-app bug: "The system name is cloud nat order
# processing platform." (no trigger phrase this regex recognized at all)
# fell all the way to _infer_system_name()'s fallback, which grabbed "is
# named the Cloud NAT Order Processing" as the document TITLE — a real
# capture containing substantive words, so it passed the stopword-only
# rejection guard, but with leading verb/filler words ("is named the")
# baked in because nothing trimmed them off a capture that wasn't rejected
# outright.
def test_system_name_extracts_direct_system_name_is_x_statement():
    text = "The system name is cloud nat order processing platform."
    assert _extract_by_pattern(SYSTEM_NAME_FIELD, text) == "Cloud Nat Order Processing"


def test_system_name_extracts_is_named_x_without_trailing_descriptor():
    text = "The system is named Meridian."
    assert _extract_by_pattern(SYSTEM_NAME_FIELD, text) == "Meridian"


def test_system_name_is_named_pattern_does_not_false_positive_on_unrelated_is_sentence():
    # "The service is down" must never be mistaken for a name introduction
    # just because it matches "<noun> is <word>" — "is" alone (without
    # "named"/"called") is far too generic a trigger to allow a bare
    # punctuation terminator.
    text = "The service is down right now, we are investigating."
    assert _extract_by_pattern(SYSTEM_NAME_FIELD, text) is None


# Regression tests for a bug found via a real-world critique of 3 generated
# KT PDFs (AWS/Azure/GCP): day1_survival_checklist's required_access field
# declares 5 fixed row labels (Cloud Console/Git Repository/CI-CD Tool/
# Monitoring/Secrets Location) in kt_schema_new.json, but real speech states
# the required tools as ONE comma-joined sentence ("For new team members,
# review Grafana dashboards, GitHub repositories, Kubernetes namespaces, and
# pipelines.") rather than one sentence per row. The old table-type pattern
# fallback just joined up to 10 raw lines verbatim, so that whole sentence
# landed in a single row's first cell instead of becoming one row per item.
REQUIRED_ACCESS_FIELD = {
    "id": "required_access",
    "type": "table",
    "rows": ["Cloud Console", "Git Repository", "CI/CD Tool", "Monitoring", "Secrets Location"],
}


def test_required_access_table_splits_enumerated_sentence_into_rows():
    text = (
        "For new team members, review Grafana dashboards, GitHub repositories, "
        "Kubernetes namespaces, and pipelines."
    )
    value = _extract_by_pattern(REQUIRED_ACCESS_FIELD, text)
    lines = value.split("\n")
    assert lines == ["Grafana dashboards", "GitHub repositories", "Kubernetes namespaces", "pipelines"]


def test_required_access_table_does_not_split_a_real_non_enumerated_sentence():
    # A normal instruction sentence with a single comma and no "and"-joined
    # list must survive intact, not get mangled by the enumeration splitter.
    text = "If you are unsure about an ongoing production activity, contact platform engineering before proceeding."
    value = _extract_by_pattern(REQUIRED_ACCESS_FIELD, text)
    assert value.split("\n") == [text]


def test_required_access_table_without_declared_rows_keeps_old_behavior():
    # A table-type field that does NOT declare fixed row labels (every
    # other table field in the schema) must be unaffected by this fix.
    field = {"id": "some_other_table", "type": "table"}
    text = "For new team members, review Grafana dashboards, GitHub repositories, and pipelines."
    value = _extract_by_pattern(field, text)
    assert value == text


def test_required_access_table_does_not_fuse_across_sentence_boundaries():
    # Regression test for a real bug found in a live GCP transcript: the
    # enumeration text handed to the splitter isn't always a single
    # sentence — it can be three ("review X, Y, Z. Remember A, B, C. The
    # danger zones are D, E.") glued together because upstream chunking
    # didn't break on sentence boundaries. Without cutting at the first
    # sentence boundary, the comma-only split fuses the tail of one
    # sentence onto the head of the next (no comma between "pager-duty."
    # and "Remember" -- just a period) into one garbled row like
    # "pager-duty. Remember the main failure scenarios", which is exactly
    # what a real generated KT's Day-1 checklist showed.
    text = (
        "For new team members, review pub slash sub, Dataflow, BigQuery, GKE, "
        "Airflow, Vertex AI, Terraform, Argo CD, Secret Manager, and pager-duty. "
        "Remember the main failure scenarios, pub slash sub backlog, unexpected "
        "BigQuery query cost, Airflow DAG failures, and schema changes. The "
        "danger zones are raw event deletion, BigQuery retention, and "
        "partitioning, pub Slash sub retention, and manual GKE changes."
    )
    value = _extract_by_pattern(REQUIRED_ACCESS_FIELD, text)
    lines = value.split("\n")
    assert lines == [
        "pub slash sub", "Dataflow", "BigQuery", "GKE", "Airflow",
        "Vertex AI", "Terraform", "Argo CD", "Secret Manager", "pager-duty",
    ]
    assert not any("Remember" in line or "danger zones" in line for line in lines)


def test_required_access_table_does_not_shred_a_non_enumerated_troubleshooting_sentence():
    # Regression test for a real bug found via a live LLM-enabled pipeline
    # run on a GCP transcript: "For Kubernetes issues, check GKE and
    # ArgoCD." is a troubleshooting tip, not a list of access items -- it
    # has no enumeration-introducing verb ("review"/"access"/"requires"/...)
    # anywhere in it, but the caller's trigger heuristic (>=1 comma + "and"
    # present) still handed it to the splitter, which used to split on the
    # comma and "and" regardless of whether a real enumeration was ever
    # found, producing three bogus "access items": "For Kubernetes issues",
    # "check GKE", "ArgoCD". Must now stay as a single, unsplit item.
    text = "For Kubernetes issues, check GKE and ArgoCD."
    value = _extract_by_pattern(REQUIRED_ACCESS_FIELD, text)
    lines = value.split("\n")
    # Preserved verbatim (including the trailing period), same as the
    # no-split precedent above -- the fix's job is only to stop it being
    # shredded into bogus fragments, not to reformat it.
    assert lines == [text]


def test_extract_rto_rpo_captures_an_explicit_sentence_with_no_llm_involved():
    # disaster_recovery has no "fields" array in kt_schema_new.json, so
    # before this function existed, an explicit RTO/RPO duration depended
    # entirely on the structured LLM prompt (llm/prompts.py) -- meaning a
    # plainly-stated "RTO is 2 hours and RPO is 15 minutes" was silently
    # lost whenever get_llm_provider() returned None. extract_rto_rpo() is
    # a pure regex function called with no LLM provider anywhere in the
    # picture. Writes to rto_metric/rpo_metric specifically -- NOT
    # rto_steps/rpo_steps, which are a different fact (the LLM prompt's own
    # recovery PROCEDURE narrative) that this function must never collide
    # with or overwrite (regression found via live verification: it used
    # to destroy a real "restore from backups" procedure).
    text = "RTO is 2 hours and RPO is 15 minutes."
    result = extract_rto_rpo(text)
    assert result == {"rto_metric": "2 hours", "rpo_metric": "15 minutes"}


def test_extract_rto_rpo_handles_spelled_out_recovery_objective_phrasing():
    text = "Our recovery time objective is 4 hours and the recovery point objective is 30 minutes."
    result = extract_rto_rpo(text)
    assert result == {"rto_metric": "4 hours", "rpo_metric": "30 minutes"}


def test_extract_rto_rpo_handles_comma_set_off_acronym_callout():
    # Regression test for a real bug found via a live end-to-end pipeline
    # run: a transcript phrasing the metric out loud and then naming its
    # acronym in a comma-set-off aside ("the recovery time objective, RTO,
    # is 2 hours") is common spoken-transcript style, but the original
    # regex only tolerated a bare "(rto)" immediately before "is", so this
    # phrasing matched nothing and silently dropped an explicitly stated
    # RTO/RPO.
    text = (
        "The recovery time objective, RTO, is 2 hours and the recovery "
        "point objective, RPO, is 15 minutes."
    )
    result = extract_rto_rpo(text)
    assert result == {"rto_metric": "2 hours", "rpo_metric": "15 minutes"}


def test_extract_rto_rpo_returns_only_whichever_one_is_present():
    assert extract_rto_rpo("RTO is 2 hours.") == {"rto_metric": "2 hours"}
    assert extract_rto_rpo("RPO is 15 minutes.") == {"rpo_metric": "15 minutes"}


def test_extract_rto_rpo_returns_empty_when_neither_is_stated():
    assert extract_rto_rpo("") == {}
    assert extract_rto_rpo("We back up the database nightly.") == {}


def test_extract_rto_rpo_never_collides_with_the_llm_procedure_fields():
    # Regression test for a real bug found via live PDF comparison: an
    # earlier version wrote the plain duration into "rto_steps"/"rpo_steps"
    # -- the same field ids the LLM structured prompt (llm/prompts.py) uses
    # for the recovery PROCEDURE narrative ("restore database from backups",
    # "recreate infrastructure using infra-as-code") -- silently destroying
    # that real procedure and replacing it with a bare, unlabeled number.
    # Metric and procedure are different facts; this must never regress.
    result = extract_rto_rpo("RTO is 2 hours and RPO is 15 minutes.")
    assert "rto_steps" not in result
    assert "rpo_steps" not in result
    assert set(result) <= {"rto_metric", "rpo_metric"}


def test_dotnet_microservices_recognized_as_a_component():
    # ".NET microservices" is the standard way an Azure KT names its backend
    # tier, but the leading "." can't sit inside the tools regex's outer
    # \b(...)\b -- so the entire backend was invisible to the component
    # list, Technology summary and architecture diagram alike. Matched via
    # its "NET microservices" tail and canonicalized back to ".NET".
    from knowledge.knowledge_builder import _canonicalize_component_term
    found = PATTERN_EXTRACTORS["tools"].findall("The back end consists of .NET microservices.")
    assert found, "no component matched for '.NET microservices'"
    assert ".NET" in [_canonicalize_component_term(t) for t in found]


class _Idx:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _ScoreRow(list):
    """Minimal stand-in for the 1-D tensor sentence_transformers returns."""

    def argmax(self):
        return _Idx(max(range(len(self)), key=lambda i: list.__getitem__(self, i)))

    def __getitem__(self, index):
        return _Idx(list.__getitem__(self, index))


def test_semantic_match_requires_a_named_anchor_when_the_field_names_one(monkeypatch):
    # Regression test for a real, published mapping error: System Overview's
    # "Cache Layer" field (description "Redis / ElastiCache configuration
    # and sizing") was filled with "Flux synchronizes the new version into
    # AKS. Rollback is performed by reverting the Git deployment
    # configuration..." -- a GitOps sentence with nothing to do with a
    # cache, which cleared the flat 0.35 similarity threshold. A field whose
    # own definition names a concrete technology must only be matched
    # against a sentence that actually mentions it.
    field = {
        "id": "cache_layer",
        "label": "Cache Layer",
        "description": "Redis / ElastiCache configuration and sizing",
        "type": "text",
    }
    sentences = [
        "Flux synchronizes the new version into AKS. Rollback is performed by reverting the Git deployment configuration.",
        "Azure Cache for Redis provides caching for the order platform.",
    ]
    captured = {}

    class _Model:
        def encode(self, value, convert_to_tensor=False, normalize_embeddings=False):
            if isinstance(value, list):
                captured["candidates"] = value
            return value

    class _FakeUtil:
        @staticmethod
        def cos_sim(a, b):
            # Uniform score for every candidate: without the anchor filter
            # argmax would pick the FIRST (wrong) sentence.
            return [_ScoreRow([0.9] * len(b))]

    import sys as _sys
    import types as _types
    stub = _types.ModuleType("sentence_transformers")
    stub.util = _FakeUtil
    monkeypatch.setitem(_sys.modules, "sentence_transformers", stub)

    result = _extract_by_semantic(field, sentences, model=_Model())
    assert captured["candidates"] == [sentences[1]], "non-anchor sentences must be filtered out"
    assert result == sentences[1]


def test_semantic_match_unaffected_when_field_names_no_technology(monkeypatch):
    # Fields with no concrete technology in their definition (most fields)
    # must keep the original behaviour -- the anchor rule only narrows a
    # field that actually names something to look for.
    field = {"id": "business_criticality", "label": "Business Criticality", "description": "How critical is it"}
    sentences = ["This is one of the most business critical systems.", "Unrelated filler sentence."]
    captured = {}

    class _Model:
        def encode(self, value, convert_to_tensor=False, normalize_embeddings=False):
            if isinstance(value, list):
                captured["candidates"] = value
            return value

    class _FakeUtil:
        @staticmethod
        def cos_sim(a, b):
            return [_ScoreRow([0.9, 0.1])]

    import sys as _sys
    import types as _types
    stub = _types.ModuleType("sentence_transformers")
    stub.util = _FakeUtil
    monkeypatch.setitem(_sys.modules, "sentence_transformers", stub)

    result = _extract_by_semantic(field, sentences, model=_Model())
    assert captured["candidates"] == sentences
    assert result == sentences[0]


def test_customer_reach_extracted_from_the_same_sentence_as_business_volume():
    # One sentence genuinely carries two independent facts. Whichever field
    # claimed the line first excluded it from the other, so on a live run
    # the volume was captured and "web and mobile applications" was lost
    # outright -- the Coverage matrix then reported Customer Reach as simply
    # "missing" when the transcript plainly stated it.
    from field_populator import extract_customer_reach
    text = ("The platform processes around 120,000 customer orders per day "
            "through web and mobile applications.")
    assert extract_customer_reach(text) == "web and mobile applications"
    assert _extract_by_pattern(
        {"id": "customer_reach", "label": "Customer Reach", "type": "text"}, text
    ) == "web and mobile applications"
    # The volume extractor still works off the very same sentence.
    assert _extract_by_pattern(
        {"id": "orders_per_day", "label": "Business Volume", "type": "text"}, text
    ) == "120,000 orders per day"


def test_customer_reach_handles_audience_phrasing_and_stays_silent_otherwise():
    from field_populator import extract_customer_reach
    assert extract_customer_reach("Used by B2B partners and internal teams.") == "B2B partners, internal teams"
    assert extract_customer_reach("Nothing about the audience here.") is None
    assert extract_customer_reach("") is None
