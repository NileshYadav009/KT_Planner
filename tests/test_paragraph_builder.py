from context_mapper import build_section_paragraph


def test_build_section_paragraph_orders_and_formats():
    sents = [
        {"text": "step two jenkins triggers automatically", "start": 5.0, "end": 6.0, "speaker": "A"},
        {"text": "code is merged into main branch", "start": 1.0, "end": 2.0, "speaker": "A"},
        {"text": "artifacts are deployed in kubernetes", "start": 7.0, "end": 8.0, "speaker": "B"},
    ]
    paragraph = build_section_paragraph(sents, [])
    assert paragraph.startswith("Code is merged into main branch.")
    assert "Step two jenkins triggers automatically." in paragraph
    assert "Additionally," in paragraph
    assert paragraph.endswith(".")
