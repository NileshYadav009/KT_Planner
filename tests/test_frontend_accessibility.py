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
