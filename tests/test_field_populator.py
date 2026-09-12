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

from field_populator import populate_fields, find_source_sentence_index, _build_llm_gap_fill_prompt


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
    # Only one real sentence exists; production_notes (first in schema
    # order) will claim it via semantic match. The staging/non_production
    # LLM prompts must be told what production_notes already captured, so
    # the model can avoid blindly re-attributing the same staging-specific
    # text to a different, unrelated environment.
    coverage = {"environments": {"content": []}}
    section_content = {
        "environments": {
            "sentences": [
                {"text": "The staging environment closely mirrors production.", "start": 0, "end": 3, "speaker": None, "audio_confidence": 0.9},
            ],
        }
    }
    stub = _StubLLM(["NOT_MENTIONED\nEXPLICIT", "NOT_MENTIONED\nEXPLICIT"])
    populate_fields(
        SIBLING_ENV_SCHEMA, coverage, llm_provider=stub, embedding_model=None, section_content=section_content
    )
    # production_notes should have resolved via semantic match (no LLM call
    # needed for it), and the two LLM prompts that did fire (for staging and
    # non_production) should reference what production_notes already has.
    assert len(stub.prompts) == 2
    for prompt in stub.prompts:
        assert "Already captured for OTHER fields" in prompt
        assert "closely mirrors production" in prompt
