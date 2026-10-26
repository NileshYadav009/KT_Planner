"""Punctuation for speech to text that has none (ledger X90, audit §9abf).

Some transcription tools return text with no full stops: "in the first week
get an account on the weblogic console ... by the end of the month run a
thursday deployment yourself". Every later stage cuts facts at sentence
ends and commas, so a paragraph like that was one fact, cut by length.

A paragraph of at least 120 words with fewer than one punctuation mark per
30 words is given the sentence ends (and, with PUNCTUATION_COMMAS=1, commas) of
a small English model (`punctuators`, "pcs_en": punctuation, capitals and
sentence boundaries; ONNX, 210 MB, at a pinned revision, fetched into the
image by scripts/fetch_models.py and otherwise on first use). Only the marks are taken, and a capital at each sentence start: the
words stay exactly as said, since the model spells hyphens and slashes
"<unk>" and guesses capitals for tool names. A break that would leave a
sentence of one or two words is not made ("... before it goes. | Live.").
Text with punctuation, short Whisper segments and every failure (package
missing, no network for the download, a word count that does not line up)
are left as they are. PUNCTUATION_MODEL=0 turns it off.
"""
import logging
import os
import re
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)

# Only a long run. On the visible KTs without punctuation (noisy_copies.py,
# audit §9abf), sending runs of a paragraph too (30 words or more) placed no
# better than no model (1048 of 1085 facts either way, and dev d12, really
# unpunctuated, 4 fewer): it ends sentences mid-phrase in halting speech ("we
# get you know around. | Fifty searches a second"), where fact_units' cutting
# does as well. From 120 words: 1057 of 1085, the normal KTs unchanged, and a
# whole transcript with no line breaks (391 words) from 46% to 83%.
MIN_WORDS = int(os.getenv("PUNCTUATION_MIN_WORDS", "120"))
WORDS_PER_MARK = 30
_MIN_SENTENCE_WORDS = 3
_MARKS_RE = re.compile(r"[.,;:!?]")
_ENDS = ".?!"
_RUN_SPLIT_RE = re.compile(r"((?<=[.!?])[ \t]+|\n)")
# A pasted turn's label stays in front: "Marcus: sure quayside is ..."
_LABEL_RE = re.compile(r"^([A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,3}:\s+)")

# Pinned like the other models (scripts/fetch_models.py fetches it into the
# image, which runs offline).
LOCAL_REPO = "1-800-BAD-CODE/punctuation_fullstop_truecase_english"
LOCAL_REVISION = "b26fd1c40e88678859048898218ea4edcc24c84a"
_FILES = {"model_filename": "punct_cap_seg_en.onnx", "spe_filename": "spe_32k_lc_en.model",
          "config_filename": "config.yaml"}

_lock = threading.Lock()
_model = None
_unavailable = False


def _enabled() -> bool:
    return os.getenv("PUNCTUATION_MODEL", "on").lower() not in ("", "0", "off", "false")


def _with_commas() -> bool:
    return os.getenv("PUNCTUATION_COMMAS", "0") == "1"


def fetch() -> str:
    """The model's directory, downloaded at the pinned revision if needed."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=LOCAL_REPO, revision=LOCAL_REVISION, allow_patterns=list(_FILES.values()))


def _load():
    global _model, _unavailable
    if _model is not None or _unavailable:
        return _model
    with _lock:
        if _model is None and not _unavailable:
            try:
                from punctuators.models.punc_cap_seg_model import PunctCapSegConfigONNX, PunctCapSegModelONNX
                _model = PunctCapSegModelONNX(PunctCapSegConfigONNX(directory=fetch(), **_FILES))
            except Exception as exc:  # package missing, no network, bad file
                _unavailable = True
                logger.warning("Punctuation model unavailable, text left as is: %s", exc)
    return _model


def available() -> bool:
    """The model loads (fetch_models.py --verify, offline)."""
    return _load() is not None


def needs_punctuation(paragraph: str) -> bool:
    """A paragraph of speech to text with (almost) no punctuation."""
    words = len(paragraph.split())
    return words >= MIN_WORDS and words / max(1, len(_MARKS_RE.findall(paragraph))) >= WORDS_PER_MARK


def apply_marks(words: List[str], sentences: List[str], commas: bool = False) -> Optional[str]:
    """`words` (as said) with the sentence ends, question marks and, if
    `commas`, the commas of the model's `sentences`; None when the two do
    not line up word for word."""
    # A mark the model writes as a word of its own (a "." after text that
    # already ended with one) is not a word.
    said = [[w for w in s.split() if w.strip(".,;:!?")] for s in sentences]
    said = [s for s in said if s]
    if sum(len(s) for s in said) != len(words):
        return None
    # A sentence of one or two words joins the one before it.
    merged: List[List[str]] = []
    for sentence in said:
        if merged and len(sentence) < _MIN_SENTENCE_WORDS:
            merged[-1] = merged[-1][:-1] + [merged[-1][-1].rstrip(_ENDS)] + sentence
        else:
            merged.append(list(sentence))
    if merged and len(merged) > 1 and len(merged[0]) < _MIN_SENTENCE_WORDS:
        merged[1] = [w.rstrip(_ENDS) for w in merged[0]] + merged[1]
        merged.pop(0)
    out, i = [], 0
    for sentence in merged:
        for j, marked in enumerate(sentence):
            word = words[i]
            i += 1
            if j == 0 and word[:1].islower():
                word = word[0].upper() + word[1:]
            mark = marked[-1] if marked[-1:] in ("." , "?", "!", ",") else ""
            last = j == len(sentence) - 1
            if word[-1:] not in ".,;:!?":
                if last:
                    word += mark if mark in _ENDS else "."
                elif mark == "," and commas:
                    word += ","
            out.append(word)
    return " ".join(out)


def restore_punctuation(text: str) -> str:
    """`text` with punctuation for each paragraph that has none (see the
    module docstring); unchanged when none needs it or the model is off."""
    if not text or not _enabled():
        return text
    # Runs between sentence ends and line breaks, with what separates them:
    # the cleaner runs again on its own output, where paragraphs are one line
    # joined by full stops, and each of them is a run of its own.
    parts = _RUN_SPLIT_RE.split(text)
    todo = [i for i in range(0, len(parts), 2) if needs_punctuation(_LABEL_RE.sub("", parts[i].strip()))]
    if not todo:
        return text
    model = _load()
    if model is None:
        return text
    commas = _with_commas()
    for i in todo:
        run = parts[i]
        lead = run[:len(run) - len(run.lstrip())]
        label = _LABEL_RE.match(run.strip())
        body = run.strip()[label.end():] if label else run.strip()
        try:
            sentences = model.infer([body], apply_sbd=True)[0]
        except Exception as exc:
            logger.warning("Punctuation model failed on a run, left as is: %s", exc)
            continue
        marked = apply_marks(body.split(), sentences, commas)
        if marked:
            parts[i] = lead + (label.group(1) if label else "") + marked + run[len(run.rstrip()):]
    return "".join(parts)
