"""Copy-editing spoken sentences into written English, without changing a fact.

A KT is built from speech, and a recording's sentences read like speech:
"It processes around 15,000 orders daily Customers uses a react application
... to services Running on Amazon EKS". Each sentence that reads like speech
(`needs_edit`) is rewritten by a language model, and a rewrite is kept only
when `problems()` finds nothing:

- every tool name and number in the sentence is still there;
- negations and obligations are still there ("never", "not", "must",
  "only", "still") and none were added;
- at least 85% of its content words are still there, at most a few new
  ones are added, and nothing new is stated (grounding.added_specifics);
- the length is about the same, and it is not a chat reply ("Sure! ...").

Otherwise the speaker's sentence is kept. A safety rule ("Never modify
Terraform code manually") is never rewritten: a trial turned it into
"Terraform code cannot be modified manually".

The model, in order:
1. the configured LLM provider (Groq/Gemini) when the workspace allows one:
   fastest, and no memory on this machine. Sentences it could not take (a
   dead key, a rate limit mid-KT, no network, the per-KT call limit) go to
   the local model;
2. else a local 4-bit model through llama.cpp (about 1.1 GB on disk, about
   1.5 GB of RAM while it runs). It is loaded for the copy edit only and
   freed straight after, runs on half the CPU cores so the machine stays
   usable, is skipped when less than CONTINUUM_COPYEDIT_MIN_FREE_GB of
   memory is free, and stops after CONTINUUM_COPYEDIT_MAX_SECONDS (later
   sentences stay as said). It is never downloaded during a KT
   (scripts/fetch_models.py fetches it);
3. else nothing is rewritten.

The Sources appendix quotes the transcript itself, so it keeps the speaker's
own words either way (knowledge/evidence.py).
"""
from __future__ import annotations

import gc
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

# Qwen2.5-1.5B-Instruct in 4 bits (Apache-2.0): it wrote 8 of 8 trial
# sentences acceptably, where the 0.5B model managed 8 of 20. In full
# precision through PyTorch it needed 6 GB of RAM and 16 s a sentence here;
# the grammar-correction models (CoEdIT, grammar-synthesis) are licensed for
# non-commercial use only.
LOCAL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
LOCAL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LOCAL_REVISION = "91cad51170dc346986eccefdc2dd33a9da36ead9"
# A path to another .gguf file, or "off" for no local model.
LOCAL_MODEL = os.getenv("CONTINUUM_COPYEDIT_MODEL", "").strip()
# "0" turns the copy edit off whatever the model.
ENABLED = os.getenv("CONTINUUM_COPYEDIT", "1").strip().lower() not in ("0", "false", "off", "no")
MAX_SECONDS = float(os.getenv("CONTINUUM_COPYEDIT_MAX_SECONDS", "300") or 300)
THREADS = int(os.getenv("CONTINUUM_COPYEDIT_THREADS", "0") or 0) or max(1, (os.cpu_count() or 2) // 2)
# The local model adds about 1.5 GB while it runs (0.8 GB committed; the rest
# is the model file, mapped, which the OS can drop under pressure). With less
# free memory than this the KT keeps its rules-only sentences instead of
# pushing the machine into swap.
MIN_FREE_GB = float(os.getenv("CONTINUUM_COPYEDIT_MIN_FREE_GB", "1.5") or 1.5)
# Sentences per LLM call (one numbered list per call).
_BATCH = 15

SYSTEM_PROMPT = (
    "You are a copy editor for technical knowledge-transfer notes made from a spoken recording. "
    "Rewrite the spoken text as clear, grammatical written English.\n"
    "Rules:\n"
    "- Keep every fact, name, number, unit and technical term exactly as given.\n"
    "- Do not add any information that is not in the text.\n"
    "- Remove filler and screen-tour talk (yeah, yep, right, so, like, you know, as you can see here).\n"
    "- Fix grammar, capitalisation and punctuation; split run-on speech into sentences.\n"
    "- Output only the rewritten text."
)
# Generic examples (not from any KT the suite scores).
_EXAMPLES = [
    ("so yeah the payments service Runs on ECS right and you can see here it uses Aurora for the ledger",
     "The payments service runs on ECS and uses Aurora for the ledger."),
    ("then we have the backups which runs every 6 hours Snapshots are kept in S3 for 30 days",
     "Backups run every 6 hours. Snapshots are kept in S3 for 30 days."),
]

# --------------------------------------------------------------------------
# Which sentences to rewrite, and which rewrites to keep
# --------------------------------------------------------------------------

# Words a copy edit may drop or change without losing content.
_LIGHT_WORDS = frozenset("""yeah yep yup like just here there right okay basically actually really very so then now
you your we our they them this that these those what which whatever with from into onto have has had
been being will would could should able also more much many some such than when where while
about after before again even ever going gonna want wanna thing things stuff kind sort""".split())
_POLARITY = ("never", "not", "no", "cannot", "must", "always", "only", "still", "without", "until", "unless")
# A rule is never reworded. "No changes in the last week of December." came
# back as "There have been no changes in the last week of December.": the
# rule became a statement about the past.
_SAFETY_RE = re.compile(r"^\s*(?:never|do\s+not|don't|always|must|avoid|no)\b", re.IGNORECASE)
# "Nadia: And my first week? Owen: Get access to ..." came back as "Nadia: My
# first week involves getting access to ...": Owen's answer became Nadia's.
_SPEAKER_LABEL_RE = re.compile(r"(?:^|(?<=[.?!]\s))([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)?)\s*:\s")
# Tense the speaker did not use: "No changes ..." -> "There have been no
# changes ..." turns a rule into history.
_TENSE_WORDS = frozenset({"have", "has", "had", "was", "were", "been"})
# Words that name a KT topic. The section a sentence goes to is decided on
# the rewritten sentence, so a word the speaker did not say moves it: "Java
# Spring Boot services on Red Hat OpenShift" came back as "... services are
# deployed on Red Hat OpenShift" and read as a deployment fact.
_TOPIC_STEMS = ("deploy", "releas", "rollb", "backup", "restor", "recover", "monitor", "alert", "dashboard",
                "secur", "encrypt", "cost", "budget", "escalat", "owner", "owns", "incident", "outage", "fail",
                "migrat", "access", "approv")
_CHAT_RE = re.compile(r"^\s*(?:sure|certainly|here(?:'s|\s+is)|of\s+course)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d[\d,.]*\d|\d")
_MIN_CONTENT_KEPT = 0.85
_MAX_NEW_WORDS = 2

# Signs a sentence reads as speech; a sentence without any is left alone,
# which saves the model's time ("Check connection pool metrics before
# restarting services." needs nothing).
_SPOKEN_RE = re.compile(
    r"\b(?:so|like|right|okay|ok|basically|actually|just|whatever|suppose|gonna|wanna|kind\s+of|sort\s+of"
    r"|you\s+can\s+see|you\s+will\s+see|as\s+you\s+can|then\s+we\s+have|then\s+comes|let\s+me|i\s+will"
    r"|we\s+are\s+having|which\s+have|is\s+there)\b",
    re.IGNORECASE)
_MID_CAPITAL_RE = re.compile(r"(?<=[a-z,])\s+([A-Z][a-z]+)\b")
_LONG_SENTENCE_WORDS = 25


def needs_edit(sentence: str) -> bool:
    """Whether a sentence reads as speech and is worth a rewrite."""
    s = sentence.strip()
    if len(s.split()) < 4 or _SAFETY_RE.match(s):
        return False
    if s[-1] not in ".!?" or len(s.split()) > _LONG_SENTENCE_WORDS or _SPOKEN_RE.search(s):
        return True
    # A capital mid-sentence that is not a product name ("services Running on").
    tools = " ".join(_tool_matches(s))
    return any(m.group(1) not in tools for m in _MID_CAPITAL_RE.finditer(s))


def _norm(text: str) -> str:
    return text.lower().replace("n't", " not").replace("cannot", "can not")


def _tool_matches(text: str) -> List[str]:
    from field_populator import PATTERN_EXTRACTORS
    return list(PATTERN_EXTRACTORS["tools"].findall(text))


def _tools(text: str) -> set:
    return {m.lower().replace(" ", "") for m in _tool_matches(text)}


def _stems(text: str) -> set:
    return {w[:5] for w in re.findall(r"[a-z]{4,}", _norm(text)) if w not in _LIGHT_WORDS}


def problems(source: str, rewrite: str) -> List[str]:
    """Why `rewrite` may not replace `source` (empty: it may)."""
    from grounding import added_specifics

    if not rewrite or not rewrite.strip():
        return ["empty"]
    out: List[str] = []
    if _SAFETY_RE.match(source):
        out.append("safety rule")
    if _CHAT_RE.match(rewrite):
        out.append("chat reply")
    added = added_specifics(rewrite, [source])
    if added:
        out.append("added " + ", ".join(added[:3]))
    lost_tools = _tools(source) - _tools(rewrite)
    if lost_tools:
        out.append("lost " + ", ".join(sorted(lost_tools)))
    lost_numbers = set(_NUMBER_RE.findall(source)) - set(_NUMBER_RE.findall(rewrite))
    if lost_numbers:
        out.append("lost " + ", ".join(sorted(lost_numbers)))
    src_words = set(re.findall(r"[a-z]+", _norm(source)))
    new_words = set(re.findall(r"[a-z]+", _norm(rewrite)))
    lost = [w for w in _POLARITY if w in src_words and w not in new_words]
    gained = [w for w in _POLARITY if w in new_words and w not in src_words]
    if lost:
        out.append("lost " + ", ".join(lost))
    if gained:
        out.append("added " + ", ".join(gained))
    stems, new_stems = _stems(source), _stems(rewrite)
    if stems and len(stems & new_stems) / len(stems) < _MIN_CONTENT_KEPT:
        out.append(f"kept {len(stems & new_stems)}/{len(stems)} content words")
    # Words the speaker never said: "we can see the alerts end to end" became
    # "we can see the entire process, including the end-to-end alerts", and
    # a fact reworded that far no longer matches its source sentence.
    added_words = new_stems - stems
    if len(added_words) > max(_MAX_NEW_WORDS, len(stems) // 5):
        out.append(f"{len(added_words)} new content words")
    n_src, n_new = len(source.split()), len(rewrite.split())
    if n_new < 0.5 * n_src or n_new > 1.5 * n_src + 4:
        out.append("length changed")
    if _SPEAKER_LABEL_RE.findall(source) != _SPEAKER_LABEL_RE.findall(rewrite):
        out.append("speakers changed")
    tense = sorted((new_words & _TENSE_WORDS) - src_words)
    if tense:
        out.append("tense changed: " + ", ".join(tense))
    said = re.findall(r"[a-z]+", source.lower())
    topics = sorted({stem for w in re.findall(r"[a-z]+", rewrite.lower()) for stem in _TOPIC_STEMS
                     if w.startswith(stem) and not any(s.startswith(stem) for s in said)})
    if topics:
        out.append("added topic words: " + ", ".join(topics))
    return out


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def local_model_path() -> Optional[str]:
    """The local model file when it is on this machine, else None (never a
    download during a KT)."""
    if LOCAL_MODEL.lower() in ("off", "none", "0"):
        return None
    if LOCAL_MODEL:
        return LOCAL_MODEL if os.path.isfile(LOCAL_MODEL) else None
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(LOCAL_REPO, LOCAL_FILE, revision=LOCAL_REVISION, local_files_only=True)
    except Exception:
        return None


def available_memory_gb() -> Optional[float]:
    """Physical memory free for a new allocation, or None when unknown."""
    try:
        if os.name == "nt":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in (
                        "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                        "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys / 2**30
            return None
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except Exception:
        return None
    return None


def _messages(sentence: str) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for spoken, written in _EXAMPLES:
        msgs += [{"role": "user", "content": spoken}, {"role": "assistant", "content": written}]
    msgs.append({"role": "user", "content": sentence})
    return msgs


def _rewrite_local(path: str, sentences: Sequence[str], deadline: float) -> List[Optional[str]]:
    """One rewrite per sentence until the deadline; the model is freed on
    return. llama.cpp reuses the shared prompt prefix (instructions and
    examples) between sentences, so each costs only its own words."""
    from llama_cpp import Llama

    out: List[Optional[str]] = [None] * len(sentences)
    llm = Llama(model_path=path, n_ctx=2048, n_threads=THREADS, verbose=False)
    try:
        for i, s in enumerate(sentences):
            if time.time() > deadline:
                LOGGER.info("Copy edit stopped at its time limit; %d sentences left as said", len(sentences) - i)
                break
            try:
                reply = llm.create_chat_completion(messages=_messages(s), temperature=0.0,
                                                   max_tokens=int(len(s.split()) * 2.5) + 24)
                out[i] = str(reply["choices"][0]["message"]["content"]).strip()
            except Exception as exc:
                LOGGER.warning("Copy edit failed for one sentence: %s", exc)
    finally:
        del llm
        gc.collect()
    return out


_NUMBERED_RE = re.compile(r"^\s*(\d+)[.)]\s*(.+?)\s*$")


def _rewrite_remote(provider: Any, sentences: Sequence[str]) -> Tuple[List[Optional[str]], List[int]]:
    """One numbered list per call; a line that comes back unnumbered or
    missing leaves its sentence unchanged. Also the positions whose call
    failed (a dead key, a rate limit, no network) or was never made (the
    per-KT call limit), for the local model to take."""
    out: List[Optional[str]] = [None] * len(sentences)
    failed: List[int] = []
    examples = "\n".join(f"Spoken: {a}\nWritten: {b}" for a, b in _EXAMPLES)
    # It runs first, so under a per-KT call limit it takes at most a quarter
    # of it and leaves the rest to extraction; later sentences stay as said.
    try:
        cap = int(os.getenv("LLM_MAX_CALLS_PER_KT", "0") or 0)
    except ValueError:
        cap = 0
    last = len(sentences) if cap <= 0 else min(len(sentences), max(0, cap // 4) * _BATCH)
    for at in range(0, last, _BATCH):
        chunk = sentences[at:at + _BATCH]
        listing = "\n".join(f"{i}. {s}" for i, s in enumerate(chunk, start=1))
        prompt = (f"Examples:\n{examples}\n\nRewrite each numbered item separately. Answer with the same "
                  f"numbers, one item per line, and nothing else.\n\n{listing}")
        try:
            reply = provider.generate(prompt, temperature=0.0, max_output_tokens=2048, stop_sequences=[],
                                      system_prompt=SYSTEM_PROMPT)
        except Exception as exc:
            failed.extend(range(at, at + len(chunk)))
            LOGGER.warning("Copy-edit call failed: %s", exc)
            continue
        for line in str(reply or "").splitlines():
            m = _NUMBERED_RE.match(line)
            if m and 1 <= int(m.group(1)) <= len(chunk):
                out[at + int(m.group(1)) - 1] = m.group(2)
    return out, failed + list(range(last, len(sentences)))


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def _usable_local_model(stats: Dict[str, Any]) -> Optional[str]:
    """The local model's path when it can run now: downloaded, llama.cpp
    installed, and enough free memory (else stats["skipped"] says why)."""
    path = local_model_path()
    if path is None:
        return None
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        LOGGER.warning("llama-cpp-python is not installed; sentences are not copy-edited.")
        return None
    free = available_memory_gb()
    if free is not None and free < MIN_FREE_GB:
        LOGGER.warning("Copy edit skipped: %.1f GB of memory free, %.1f GB needed.", free, MIN_FREE_GB)
        stats["skipped"] = "low memory"
        return None
    return path


def copyedit_sentences(sentences: Sequence[str]) -> Tuple[List[str], Dict[str, Any]]:
    """Each sentence rewritten as written English where it reads as speech
    and the rewrite passes problems(), else unchanged; with stats {"model",
    "considered", "rewritten", "rejected", "seconds"}."""
    stats: Dict[str, Any] = {"model": None, "considered": 0, "rewritten": 0, "rejected": 0, "seconds": 0.0}
    texts = list(sentences)
    if not ENABLED or not texts:
        return texts, stats
    started = time.time()
    todo = [i for i, s in enumerate(texts) if needs_edit(s)]
    if not todo:
        return texts, stats
    from llm_provider import get_llm_provider
    provider = get_llm_provider()
    rewrites: List[Optional[str]] = [None] * len(todo)
    left = list(range(len(todo)))           # positions in `todo` no model has taken yet
    models: List[str] = []
    if provider is not None:
        rewrites, left = _rewrite_remote(provider, [texts[i] for i in todo])
        if len(left) < len(todo):
            models.append("llm")
        if left:
            # A dead key, a rate limit mid-KT or the per-KT call limit: the
            # local model takes the sentences the LLM could not.
            LOGGER.warning("Copy edit: %d sentences the LLM could not take go to the local model.", len(left))
    if left:
        path = _usable_local_model(stats)
        if path is not None:
            for k, new in zip(left, _rewrite_local(path, [texts[todo[k]] for k in left], started + MAX_SECONDS)):
                rewrites[k] = new
            models.append(os.path.basename(path))
    stats["model"] = " + ".join(models) or None
    if not models:
        return texts, stats
    stats["considered"] = len(todo)
    for i, new in zip(todo, rewrites):
        if new is None or new.strip() == texts[i].strip():
            continue
        if problems(texts[i], new):
            stats["rejected"] += 1
            continue
        texts[i] = new.strip()
        stats["rewritten"] += 1
    stats["seconds"] = round(time.time() - started, 1)
    LOGGER.info("Copy edit (%s): %d of %d sentences rewritten, %d rewrites rejected, %.0fs",
                stats["model"], stats["rewritten"], len(todo), stats["rejected"], stats["seconds"])
    return texts, stats
