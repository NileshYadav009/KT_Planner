"""The web page tells screen-reader users what is happening: a KT's progress
and every toast ("Processing failed: ...") are live regions, and the
progress bar is a progressbar with its value (audit §9aaz: neither was
announced)."""
import os
import re

PAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "index.html")


def _page():
    with open(PAGE, encoding="utf-8") as fh:
        return fh.read()


def test_toasts_and_progress_are_announced():
    html = _page()
    assert re.search(r'<div id="toasts"[^>]*aria-live="polite"', html)
    assert re.search(r'<p id="progressHint"[^>]*aria-live="polite"', html)
    track = re.search(r'<div class="progress-track"[^>]*>', html, re.DOTALL).group(0)
    assert 'role="progressbar"' in track and 'aria-valuemin="0"' in track and 'aria-valuemax="100"' in track
    # The value follows the bar.
    assert "setAttribute('aria-valuenow'" in html


def test_every_element_the_script_looks_up_is_on_the_page():
    # One missing id stops the whole script: getElementById(...) returns null
    # and the first addEventListener on it throws, so no button works.
    html = _page()
    markup = html.split("<script>\n/* ── Theme")[0]
    ids = set(re.findall(r'\bid="([A-Za-z][\w-]*)"', markup))
    looked_up = set(re.findall(r"getElementById\('([\w-]+)'\)", html))
    made_by_the_script = {"diffAgainst", "docDiff", "signOutBtn"}
    assert looked_up - made_by_the_script <= ids, sorted(looked_up - made_by_the_script - ids)


def test_the_kt_views_are_tabs_each_with_its_panel():
    html = _page()
    tabs = re.findall(r'role="tab" id="tab-(\w+)" aria-controls="panel-(\w+)"', html)
    assert [t for t, _ in tabs] == ["doc", "gaps", "signoff", "transcript", "details"]
    assert all(t == p for t, p in tabs)
    for t, _ in tabs:
        assert re.search(rf'id="panel-{t}" role="tabpanel" aria-labelledby="tab-{t}"', html)


def test_coverage_on_the_page_is_the_coverage_matrix_the_pdf_uses():
    # The page counted the classifier's per-section statuses: it called 13
    # sections of a KT "weak" that the PDF's matrix called covered well. It
    # now reads the matrix and its follow-ups, as handover_glance does.
    html = _page()
    assert "_coverage_rows" in html and "_knowledge_gaps" in html
    assert "status==='weak'" not in html
