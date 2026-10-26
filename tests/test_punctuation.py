"""Punctuation for speech to text that has none (punctuation.py, ledger X90).
A stand-in model: tests never download the real one."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import punctuation  # noqa: E402

RUN = ("in the first week get an account on the weblogic console and read-only access to the database by the end "
       "of the month run a thursday deployment yourself the database is oracle 19c on a two-node rac cluster.")


class StandIn:
    """Returns fixed sentences, spelled as the real model spells them."""

    def __init__(self, sentences):
        self.sentences = sentences
        self.calls = 0

    def infer(self, texts, apply_sbd=True):
        self.calls += 1
        return [self.sentences for _ in texts]


@pytest.fixture
def model(monkeypatch):
    # A short run stands in for a long one (MIN_WORDS is 120).
    monkeypatch.setattr(punctuation, "MIN_WORDS", 30)
    monkeypatch.setenv("PUNCTUATION_MODEL", "pcs_en")
    stand_in = StandIn([
        "In the first week, get an account on the Weblogic console and read<unk>only access to the database.",
        "By the end of the month, run a Thursday deployment yourself.",
        "The database is Oracle 19C on a two<unk>node RAC cluster.",
        ".",
    ])
    monkeypatch.setattr(punctuation, "_model", stand_in)
    monkeypatch.setattr(punctuation, "_unavailable", False)
    return stand_in


def test_only_a_long_paragraph_with_almost_no_punctuation_needs_it():
    # A paragraph-length run is left to fact_units' cutting; a whole
    # transcript with no breaks is not.
    assert not punctuation.needs_punctuation(RUN)
    assert punctuation.needs_punctuation(" ".join([RUN.rstrip(".")] * 4))
    assert not punctuation.needs_punctuation("get an account on the console and run a deployment")
    punctuated = ("In the first week, get an account on the WebLogic console. By the end of the month, run a "
                  "Thursday deployment yourself. The database is Oracle 19c on a two-node RAC cluster.")
    assert not punctuation.needs_punctuation(punctuated)


def test_the_words_stay_as_said_and_only_the_sentence_ends_are_taken(model):
    out = punctuation.restore_punctuation(RUN)
    assert out == ("In the first week get an account on the weblogic console and read-only access to the database. "
                   "By the end of the month run a thursday deployment yourself. "
                   "The database is oracle 19c on a two-node rac cluster.")
    # The model's spelling ("read<unk>only", "Weblogic", "19C") never reaches the text.
    assert "<unk>" not in out and "19c" in out


def test_commas_only_when_asked(model, monkeypatch):
    monkeypatch.setenv("PUNCTUATION_COMMAS", "1")
    assert punctuation.restore_punctuation(RUN).startswith("In the first week, get an account")


def test_a_break_that_leaves_one_or_two_words_is_not_made():
    words = "the registry team signs it off before it goes live and deployments happen on thursday".split()
    out = punctuation.apply_marks(words, ["The registry team signs it off before it goes.", "Live.",
                                          "And deployments happen on Thursday."])
    assert out == "The registry team signs it off before it goes live. And deployments happen on thursday."


def test_words_that_do_not_line_up_leave_the_paragraph_alone(monkeypatch):
    monkeypatch.setattr(punctuation, "MIN_WORDS", 30)
    monkeypatch.setenv("PUNCTUATION_MODEL", "pcs_en")
    monkeypatch.setattr(punctuation, "_model", StandIn(["In the first week.", "Something else entirely."]))
    monkeypatch.setattr(punctuation, "_unavailable", False)
    assert punctuation.restore_punctuation(RUN) == RUN


def test_a_turn_label_stays_and_punctuated_text_is_not_sent(model):
    text = "Ellen: What should I do first?\nMarcus: " + RUN
    out = punctuation.restore_punctuation(text)
    assert out.startswith("Ellen: What should I do first?\nMarcus: In the first week get an account")
    assert model.calls == 1


def test_runs_are_judged_between_sentence_ends_not_by_line(monkeypatch):
    """The cleaner runs again on its own output, one line with its paragraphs
    joined by full stops: dev d12 (paragraphs of 20 to 66 words) was sent
    whole to the model, and placed 4 facts fewer."""
    stand_in = StandIn(["x"])
    monkeypatch.setenv("PUNCTUATION_MODEL", "pcs_en")
    monkeypatch.setattr(punctuation, "_model", stand_in)
    monkeypatch.setattr(punctuation, "_unavailable", False)
    joined = " ".join([RUN] * 5)
    assert len(joined.split()) >= punctuation.MIN_WORDS
    assert punctuation.restore_punctuation(joined) == joined and stand_in.calls == 0


def test_off_or_unavailable_the_text_is_unchanged(monkeypatch):
    monkeypatch.setattr(punctuation, "MIN_WORDS", 30)
    monkeypatch.setenv("PUNCTUATION_MODEL", "0")
    assert punctuation.restore_punctuation(RUN) == RUN
    monkeypatch.setenv("PUNCTUATION_MODEL", "pcs_en")
    monkeypatch.setattr(punctuation, "_model", None)
    monkeypatch.setattr(punctuation, "_unavailable", True)
    assert punctuation.restore_punctuation(RUN) == RUN
