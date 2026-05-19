from devops_transcription import clean_transcript


def test_dedupes_repeated_paragraphs_and_fixes_devops_terms():
    block = (
        "hi everyone. this kt is about devops. "
        "this system handles order intake, validation, pavement orchestration and fulfillment triggers. "
        "approval isrequired and actions are emergency only."
    )
    raw = f"{block}\n\n{block}"
    cleaned = clean_transcript(raw)

    assert cleaned.lower().count("this kt is about devops") == 1
    assert "payment orchestration" in cleaned.lower()
    assert "is required" in cleaned.lower()


def test_polishes_capitalization_and_read_only_phrase():
    raw = "on day 1 there are safe to use asread only. kubernetes cluster autoscroller setting."
    cleaned = clean_transcript(raw)

    assert "as read-only" in cleaned.lower()
    assert "autoscaler" in cleaned.lower()
    assert cleaned[0].isupper()
