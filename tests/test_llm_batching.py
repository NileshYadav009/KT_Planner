"""Fewer LLM round trips (P1-12): batched classification checks and gap-fills
give each item the same answer handling as one call each."""
import re

from context_mapper import Classification, ContextClassifier
from field_populator import populate_fields


def _classifier(fn):
    clf = ContextClassifier.__new__(ContextClassifier)   # no models needed for the LLM check
    clf.llm_fallback_fn = fn
    clf.similarity_threshold = 0.15
    return clf


def _cand(sid, conf):
    return Classification(section_id=sid, section_title=sid.replace("_", " ").title(), confidence=conf,
                          similarity_score=conf, reason="test")


def _pick(ids):
    return sorted(ids)[-1]


class _Fake:
    """Answers one sentence or a batch the same way per sentence."""

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, **kwargs):
        self.calls += 1
        if "Answers:" in prompt:
            return "\n".join(f"{n}: {_pick([c.split(' (')[0] for c in cands.split('; ')])}"
                             for n, cands in re.findall(r"^(\d+)\. Sentence:.*?\n(?:   Context.*?\n)?   Candidates: (.*?)$",
                                                        prompt, re.M | re.S))
        return _pick(re.findall(r"^- ([a-z0-9_]+): ", prompt, re.M))


ITEMS = [
    ("Backups run nightly.", _cand("disaster_recovery", 0.20), [_cand("security_controls", 0.19)], "", False),
    ("Bump the limit and it catches up.", _cand("disaster_recovery", 0.18), [_cand("common_failures", 0.17)], "ctx", False),
    ("Clear decision.", _cand("monitoring_observability", 0.90), [_cand("common_failures", 0.20)], "", False),
    ("Never delete the table.", _cand("architecture_reference", 0.15), [_cand("danger_zones", 0.14)], "", True),
] * 5


def test_batched_checks_reach_the_same_verdicts_in_fewer_calls(monkeypatch):
    monkeypatch.setenv("LLM_VERIFY_BATCH_SIZE", "15")
    single_fn, batch_fn = _Fake(), _Fake()
    single = [_classifier(single_fn)._maybe_verify_with_llm(t, p, s, context_text=c, rule_decided=r)
              for t, p, s, c, r in ITEMS]
    batched = _classifier(batch_fn).verify_in_batches(ITEMS)
    assert [(p.section_id, [x.section_id for x in s]) for p, s, _ in single] == \
        [(p.section_id, [x.section_id for x in s]) for p, s, _ in batched]
    # 15 borderline sentences (the clear one is never asked): 15 calls -> 1.
    assert single_fn.calls == 15 and batch_fn.calls == 1


def test_a_batched_answer_outside_its_own_candidates_is_ignored():
    clf = _classifier(lambda prompt, **kw: "1: danger_zones\n2: not_a_section\n")
    checks = [("A.", [_cand("disaster_recovery", .2), _cand("danger_zones", .19)], ""),
              ("B.", [_cand("disaster_recovery", .2), _cand("common_failures", .19)], ""),
              ("C.", [_cand("environments", .2)], "")]
    assert clf._verify_batch_with_llm(checks) == ["danger_zones", None, None]


def test_a_failed_batch_keeps_the_classifier_picks():
    def boom(prompt, **kwargs):
        raise RuntimeError("rate limited")

    items = ITEMS[:2]
    assert [p.section_id for p, _, _ in _classifier(boom).verify_in_batches(items)] == \
        [p.section_id for _, p, _, _, _ in items]


SCHEMA = [{
    "id": "system_overview",
    "title": "System Overview",
    "fields": [
        {"id": "business_criticality", "label": "Business Criticality", "type": "single_select",
         "options": ["High", "Medium", "Low"]},
        {"id": "data_sensitivity", "label": "Data Sensitivity", "type": "single_select",
         "options": ["Public", "Internal", "Confidential"]},
        {"id": "customer_reach", "label": "Customer Reach", "type": "text"},
        {"id": "doc_link", "label": "Documentation Link", "type": "url"},
    ],
}]
SECTION = {"system_overview": {"sentences": [
    {"text": "Leadership treats this system as the company's top operational priority.", "start": 0, "end": 3},
    {"text": "Customers in Europe and Asia use it every day.", "start": 3, "end": 6},
]}}


class _FillStub:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if prompt.startswith("You are filling the empty fields"):
            return "1. High | EXPLICIT\n2. Internal | INFERRED\n3. Europe and Asia | EXPLICIT"
        if "Field: Business Criticality" in prompt:
            return "High\nEXPLICIT"
        return "Internal\nINFERRED" if "Field: Data Sensitivity" in prompt else "Europe and Asia\nEXPLICIT"


def _values(result):
    return {k: (v["value"], v["source"]) for k, v in result["system_overview"].items()}


def test_batched_gap_fill_gives_the_same_values_in_one_call(monkeypatch):
    one = _FillStub()
    monkeypatch.setenv("LLM_BATCH_CALLS", "0")
    single = populate_fields(SCHEMA, {"system_overview": {"content": []}}, llm_provider=one, section_content=SECTION)
    batch = _FillStub()
    monkeypatch.setenv("LLM_BATCH_CALLS", "1")
    batched = populate_fields(SCHEMA, {"system_overview": {"content": []}}, llm_provider=batch, section_content=SECTION)
    assert _values(single) == _values(batched)
    assert _values(batched)["business_criticality"] == ("High", "llm_explicit")
    assert _values(batched)["data_sensitivity"] == ("Internal", "llm")
    # Customer reach is stated and filled without the LLM; the URL field has
    # no literal URL in the text, so neither path asks for either.
    assert len(one.prompts) == 2 and len(batch.prompts) == 1


def test_a_batched_value_not_in_the_transcript_is_discarded(monkeypatch):
    class Inventor(_FillStub):
        def generate(self, prompt, **kwargs):
            return "1. High | EXPLICIT\n2. Restricted, per the Brazil data office | EXPLICIT"

    monkeypatch.setenv("LLM_BATCH_CALLS", "1")
    result = populate_fields(SCHEMA, {"system_overview": {"content": []}}, llm_provider=Inventor(),
                             section_content=SECTION)
    assert result["system_overview"]["business_criticality"]["value"] == "High"
    assert result["system_overview"]["data_sensitivity"]["source"] == "unfilled"
