"""End-to-end KT suite (tests/e2e_kts, scored by tests/e2e_eval.py).

Four short synthetic handovers (AWS e-commerce, Azure payments, GCP data
platform, an ambiguous incomplete KT) and the user's QuickBite KT (AWS food
delivery, sentences that state several facts), each run through the orchestration
POST /kt-from-transcript queues (pipeline.run_kt_pipeline) and the real PDF
renderer, then checked on the PDF text: nothing the speaker said is lost,
nothing they did not say is added, safety warnings reach Danger Zones,
stated connections are drawn and unstated ones are not, values the speaker
called unknown stay unknown, incomplete procedures are flagged where they
appear, and the coverage report agrees with what is printed.

No LLM is called (zero external requests); the LLM path is exercised by
scripts/e2e_kts.py --llm live|replay and reported separately. Placement
floors per dataset are in tests/e2e_kts/floors.json.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from e2e_eval import load_datasets, load_floors, norm, run_all  # noqa: E402

DATASETS = load_datasets()
IDS = [d["id"] for d in DATASETS]


def test_five_datasets_of_100_to_300_words_with_no_secrets():
    from e2e_eval import _SECRET

    # E (QuickBite) is the user's own KT, added with the fact-level mapping
    # change: sentences holding several facts for several sections.
    assert len(DATASETS) == 5
    for ds in DATASETS:
        assert 100 <= len(ds["transcript"].split()) <= 300, ds["id"]
        assert not _SECRET.search(ds["transcript"]), ds["id"]


def test_every_checklist_quote_is_in_the_cleaned_transcript():
    """Including 'ArgoCD deploys them' in D, which the transcript says as
    'Argo City deploys them': the cleaner must correct it."""
    from devops_transcription import clean_transcript

    for ds in DATASETS:
        text = norm(clean_transcript(ds["transcript"]))
        for fact in ds["facts"]:
            assert norm(fact["quote"]) in text, (ds["id"], fact["id"])


@pytest.fixture(scope="module")
def reports():
    import pipeline

    if pipeline.MAPPER_PIPELINE is None:
        pipeline.load_models()
    return {r["id"]: r for r in run_all(DATASETS)}


@pytest.mark.parametrize("ds_id", IDS)
def test_the_pdf_opens_and_no_llm_was_called(reports, ds_id):
    r = reports[ds_id]
    assert r["status"] in ("completed", "completed_with_warnings")
    assert r["pdf"]["opens"] and r["pdf"]["pages"] > 2
    assert not r["llm"]["llm_calls"]


@pytest.mark.parametrize("ds_id", IDS)
def test_no_fact_is_lost(reports, ds_id):
    r = reports[ds_id]
    assert r["false_negatives"]["lost_facts"] == [], r["placement"]
    assert r["retained_in_pdf"] == r["facts"], r["false_negatives"]["facts_not_in_pdf_text"]


@pytest.mark.parametrize("ds_id", IDS)
def test_safety_warnings_and_incidents_are_in_their_sections(reports, ds_id):
    r = reports[ds_id]
    assert r["false_negatives"]["safety_warnings_misplaced"] == []
    assert r["false_positives"]["incidents_in_danger_zones"] == []


@pytest.mark.parametrize("ds_id", IDS)
def test_nothing_unsupported_is_added(reports, ds_id):
    fp = reports[ds_id]["false_positives"]
    for check in ("undocumented_given_a_value", "forbidden_claims", "durations_not_in_transcript",
                  "products_not_in_transcript", "false_signoff", "session_framing_printed"):
        assert fp[check] == [], (check, fp[check])


@pytest.mark.parametrize("ds_id", IDS)
def test_stated_connections_are_drawn_and_unstated_ones_are_not(reports, ds_id):
    r = reports[ds_id]
    assert r["false_negatives"]["relations_missing"] == []
    assert r["false_positives"]["wrong_relations"] == []


@pytest.mark.parametrize("ds_id", IDS)
def test_incomplete_procedures_are_flagged_where_they_appear(reports, ds_id):
    assert reports[ds_id]["false_negatives"]["incomplete_procedures_not_flagged"] == []


@pytest.mark.parametrize("ds_id", IDS)
def test_coverage_agrees_with_what_is_printed(reports, ds_id):
    r = reports[ds_id]
    assert r["false_negatives"]["false_missing_sections"] == []
    assert r["false_positives"]["false_covered_sections"] == []
    assert r["false_positives"]["coverage_disagrees_with_content"] == []
    assert r["knowledge_coverage"].get("lost", 0) == 0


@pytest.mark.parametrize("ds_id", IDS)
def test_every_fact_shown_has_its_source_and_is_printed_once(reports, ds_id):
    fp = reports[ds_id]["false_positives"]
    assert fp["facts_without_source"] == []
    assert fp["repeated_sentences"] == []


@pytest.mark.parametrize("ds_id", IDS)
def test_misheard_terms_are_corrected(reports, ds_id):
    assert reports[ds_id]["false_negatives"]["asr_not_fixed"] == []


@pytest.mark.parametrize("ds_id", IDS)
def test_placement_does_not_fall_below_its_floor(reports, ds_id):
    floor = load_floors()[ds_id]
    r = reports[ds_id]
    assert r["placement_accuracy"] >= floor["placement_accuracy"], {
        k: v for k, v in r["placement"].items() if v["status"] != "correct"}
    assert len(r["false_positives"]["misplaced_facts"]) <= floor["misplaced"]
    assert r["pdf"]["pages"] <= floor["max_pages"]


# --------------------------------------------------------------------------
# LLM mode, offline: the recorded Groq answers (tests/fixtures), nothing sent
# --------------------------------------------------------------------------

RECORDING = os.path.join(os.path.dirname(__file__), "fixtures", "e2e_kts_llm_recording.json")


@pytest.fixture(scope="module")
def replay_reports():
    import pipeline
    from e2e_eval import run_replay

    if pipeline.MAPPER_PIPELINE is None:
        pipeline.load_models()
    return {r["id"]: r for r in run_replay(RECORDING)}


@pytest.mark.parametrize("ds_id", IDS)
def test_llm_mode_from_the_recording_sends_nothing_and_adds_nothing(replay_reports, ds_id):
    """The LLM path (polish, structured extraction, gap-fill, classification
    checks) on the answers Groq gave on 12 Oct. Not a live model test: a
    changed prompt is an unrecorded request, skipped and never sent."""
    r = replay_reports[ds_id]
    assert r["llm"]["llm_calls"] == 0 and r["llm"]["cache_mode"] == "replay"
    if not r["llm"]["cache_hits"]:
        pytest.skip("no recorded answer matches the current prompts: re-record with scripts/e2e_kts.py --api "
                    "and --export-recording")
    fn, fp = r["false_negatives"], r["false_positives"]
    for check in ("lost_facts", "safety_warnings_misplaced", "relations_missing", "asr_not_fixed",
                  "incomplete_procedures_not_flagged"):
        assert fn[check] == [], (check, fn[check])
    for check in ("undocumented_given_a_value", "forbidden_claims", "durations_not_in_transcript",
                  "products_not_in_transcript", "false_signoff", "session_framing_printed", "wrong_relations",
                  "facts_without_source", "incidents_in_danger_zones"):
        assert fp[check] == [], (check, fp[check])
