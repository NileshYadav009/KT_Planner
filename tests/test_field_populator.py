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

from field_populator import populate_fields, find_source_sentence_index


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
