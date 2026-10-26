"""Teams and Zoom transcript files: speaker turns from WebVTT and SRT.

Microsoft Teams exports a meeting transcript as WebVTT with a voice tag on
each cue ("<v Priya Sharma>We run on AKS.</v>"); Zoom writes the speaker
before the cue text ("Priya Sharma: We run on AKS."), and an SRT file looks
the same with numbered cues. Pasted as they are, the cue numbers, timings and
tags became part of the KT's text and nobody was named. to_turns() rewrites
such a file in the pasted form the pipeline reads ("Priya Sharma: We run on
AKS. ..."), one line per turn, consecutive cues of one speaker joined, so who
said what reaches the document (speaker_roles.turn_segments) and the giver
can be detected. Timings are dropped: a pasted transcript is untimed.
"""
import re
from typing import List, Optional, Tuple

# "00:00:01.200 --> 00:00:04.900", SRT's "00:00:01,200", and older Teams
# exports' "0:0:1.2 --> 0:0:4.9".
_TIME = r"\d{1,2}(?::\d{1,2}){1,2}[.,]\d{1,3}"
_TIMING_RE = re.compile(rf"^\s*{_TIME}\s+-->\s+{_TIME}")
_VOICE_RE = re.compile(r"<v(?:\.[\w.-]+)*\s+([^>]+)>(.*?)(?:</v>|$)", re.S)
_TAG_RE = re.compile(r"</?[^>]+>")
# A speaker label written before the cue text, as Zoom does.
_SPEAKER_RE = re.compile(r"^([A-Z][\w'.-]*(?:\s+[A-Za-z][\w'.-]*){0,3})\s*:\s+(.+)$", re.S)
_HEADER_BLOCKS = ("WEBVTT", "NOTE", "STYLE", "REGION")


def is_caption_file(text: str) -> bool:
    """A WebVTT file, or an SRT file (two or more cue timings)."""
    lines = (text or "").lstrip("﻿").strip().splitlines()
    if not lines:
        return False
    if lines[0].strip().upper().startswith("WEBVTT"):
        return True
    return sum(1 for line in lines[:40] if _TIMING_RE.match(line)) >= 2


def _cues(text: str) -> List[str]:
    """Each cue's text, in order: cue numbers and identifiers, timings and
    NOTE, STYLE and REGION blocks left out."""
    cues = []
    for block in re.split(r"\n[ \t]*\n", text.replace("\r\n", "\n").lstrip("﻿")):
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines or lines[0].strip().upper().startswith(_HEADER_BLOCKS):
            continue
        timing = next((i for i, line in enumerate(lines) if _TIMING_RE.match(line)), None)
        if timing is None:
            continue
        body = " ".join(line.strip() for line in lines[timing + 1:])
        if body:
            cues.append(body)
    return cues


def _speaker_and_words(cue: str) -> Tuple[Optional[str], str]:
    voice = _VOICE_RE.search(cue)
    if voice:
        return " ".join(voice.group(1).split()), " ".join(_TAG_RE.sub("", voice.group(2)).split())
    plain = " ".join(_TAG_RE.sub("", cue).split())
    labelled = _SPEAKER_RE.match(plain)
    if labelled:
        return labelled.group(1).strip(), labelled.group(2).strip()
    return None, plain


def to_turns(text: str) -> Optional[str]:
    """The transcript as "Name: ..." turns, one per line, or None when it is
    not a caption file. A cue with no speaker joins the turn before it."""
    if not is_caption_file(text):
        return None
    turns: List[Tuple[Optional[str], List[str]]] = []
    for cue in _cues(text):
        speaker, words = _speaker_and_words(cue)
        if not words:
            continue
        if turns and (speaker is None or speaker == turns[-1][0]):
            turns[-1][1].append(words)
        else:
            turns.append((speaker, [words]))
    if not turns:
        return None
    return "\n".join(f"{speaker}: {' '.join(words)}" if speaker else " ".join(words) for speaker, words in turns)
