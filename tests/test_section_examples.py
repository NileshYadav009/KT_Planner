"""section_examples.json: the classifier's labelled examples (scripts/build_section_examples.py)."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_section_examples  # noqa: E402
from golden_eval import load_goldens, norm  # noqa: E402


def test_the_examples_file_matches_the_goldens():
    with open(build_section_examples.OUT_PATH, encoding="utf-8") as fh:
        assert fh.read() == build_section_examples.render(build_section_examples.build()), \
            "run python scripts/build_section_examples.py"


def test_no_holdout_or_customer_transcript_is_used():
    """Holdouts must stay unseen to measure accuracy on new KTs, and a
    customer's KT never ships inside the product."""
    with open(build_section_examples.OUT_PATH, encoding="utf-8") as fh:
        examples = {norm(t) for texts in json.load(fh)["sections"].values() for t in texts}
    for golden in load_goldens():
        if build_section_examples.usable(golden):
            continue
        leaked = [f["id"] for f in golden["facts"] if norm(f["quote"]) in examples]
        assert not leaked, (golden["id"], leaked)


def test_the_classifier_loads_every_section_it_knows():
    from context_mapper import load_section_examples
    from kt_schema_loader import SCHEMA

    ids = {s["id"] for s in SCHEMA}
    examples = load_section_examples()
    assert examples and set(examples) <= ids
    assert load_section_examples(os.path.join(ROOT, "no-such-file.json")) == {}
