#!/usr/bin/env python3
"""Transcription speed and accuracy (P2-3): word error rate, DevOps-term
recall and real-time factor for each Whisper setting, on recordings whose
words are known.

    python scripts/asr_benchmark.py DIR [--configs current,batched]

DIR holds pairs NAME.wav + NAME.txt (or NAME.ref): the recording and what
was said. Both sides go through the same transcript cleaning the product
applies, so product-name normalisation ("Argo CD" -> "ArgoCD") is not
counted as an error.

Settings:
    current   the product's settings (pipeline.py: WHISPER_MODEL, beam,
              Silero VAD), one segment after another
    batched   the same model through faster-whisper's BatchedInferencePipeline
              (VAD chunks transcribed in batches of 8, with timestamps so
              segments stay sentence-sized)

Real-time factor = processing seconds / audio seconds (0.5 means a one-hour
KT takes 30 minutes). Measure on an otherwise idle machine.
"""
import argparse
import os
import re
import sys
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def words(text: str):
    from devops_transcription import clean_transcript

    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", clean_transcript(text or "").lower())


def wer(ref, hyp) -> float:
    """Word-level edit distance / reference length."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / max(1, len(ref))


def term_recall(reference: str, hypothesis: str):
    """Known DevOps terms said in the reference, and how many the transcript has."""
    from devops_vocabulary import DEVOPS_VOCABULARY

    ref, hyp = " ".join(words(reference)), " ".join(words(hypothesis))
    said = [t for t in DEVOPS_VOCABULARY if len(t) > 2 and re.search(rf"\b{re.escape(t)}\b", ref)]
    heard = [t for t in said if re.search(rf"\b{re.escape(t)}\b", hyp)]
    return len(heard), len(said)


def seconds(path: str) -> float:
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--configs", default="current,batched")
    args = parser.parse_args(argv)

    from faster_whisper import BatchedInferencePipeline, WhisperModel
    import pipeline

    model = WhisperModel(pipeline.DEFAULT_WHISPER_MODEL, device="cpu", compute_type=pipeline.DEFAULT_WHISPER_COMPUTE_TYPE)
    batched = BatchedInferencePipeline(model=model)
    common = {"language": "en", "beam_size": pipeline.DEFAULT_WHISPER_BEAM_SIZE, "task": "transcribe",
              "vad_filter": pipeline.WHISPER_VAD_FILTER}
    runners = {
        "current": lambda path: model.transcribe(path, **common),
        "batched": lambda path: batched.transcribe(path, batch_size=8, without_timestamps=False, **common),
    }
    pairs = []
    for name in sorted(os.listdir(args.dir)):
        if name.endswith(".wav"):
            base = os.path.join(args.dir, name[:-4])
            ref = next((base + ext for ext in (".txt", ".ref") if os.path.exists(base + ext)), None)
            if ref:
                with open(ref, encoding="utf-8") as fh:
                    pairs.append((base + ".wav", fh.read()))
    print(f"model {pipeline.DEFAULT_WHISPER_MODEL}, beam {pipeline.DEFAULT_WHISPER_BEAM_SIZE}, "
          f"VAD {pipeline.WHISPER_VAD_FILTER}, {len(pairs)} recordings")
    for config in args.configs.split(","):
        total_audio = total_time = 0.0
        errors = ref_words = heard = said = 0.0
        for path, reference in pairs:
            started = time.time()
            segments, _ = runners[config](path)
            hypothesis = " ".join(s.text for s in segments)
            elapsed = time.time() - started
            ref, hyp = words(reference), words(hypothesis)
            e = wer(ref, hyp)
            h, s = term_recall(reference, hypothesis)
            audio = seconds(path)
            print(f"  {config:8} {os.path.basename(path):28} WER {e:6.1%}  terms {h}/{s}  "
                  f"{elapsed:6.1f}s for {audio:5.0f}s audio (RTF {elapsed / audio:.2f})")
            total_audio += audio
            total_time += elapsed
            errors += e * len(ref)
            ref_words += len(ref)
            heard += h
            said += s
        print(f"{config:10} WER {errors / max(1, ref_words):6.1%}  DevOps terms {int(heard)}/{int(said)}  "
              f"RTF {total_time / max(1, total_audio):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
