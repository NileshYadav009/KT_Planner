# Continuum KT Planner (P1-5): pinned dependencies, models baked in at
# pinned revisions, no network needed at run time, runs as a non-root user.
#
#   docker build -t continuum .
#   docker run -p 8000:8000 -v continuum-data:/app/data \
#       -e CONTINUUM_SECRET_KEY=... -e GROQ_API_KEY=... continuum
#
# That runs the API with one worker thread inside it. For the API and
# separate worker processes (P1-4), use docker-compose.yml.
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/models/hf

# ffmpeg: audio/video; pango, harfbuzz, fonts: PDF rendering (WeasyPrint);
# libgl1, libglib: OpenCV under the screen-capture OCR.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 fonts-dejavu-core fonts-liberation2 \
      libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (cached layer). The CPU build of torch comes from its
# own index; everything else is pinned in requirements.lock. The packages'
# own test suites and torch's C++ headers (about 250 MB) are never used at
# run time. They are deleted in this same layer, which is the only way they
# stay out of the image. The imports at the end check nothing needed went.
COPY requirements.lock .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "$(grep '^torch==' requirements.lock)" \
 && pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements.lock \
 && cd /usr/local/lib/python3.13/site-packages \
 && find . -type d \( -name tests -o -name test \) -prune -exec rm -rf {} + \
 && rm -rf torch/include \
 && python -c "import torch, transformers, sentence_transformers, gliner, faster_whisper, ctranslate2, sklearn, scipy, networkx, sympy, cv2, rapidocr, onnxruntime, sherpa_onnx, av, weasyprint, jinja2, llama_cpp"

# The runtime user exists before the models arrive, so they are handed to it
# in the layer that downloads them: a `chown -R` in a later layer stores a
# second copy of every model file (it added 3.2 GB on 13 Oct).
RUN useradd --create-home --uid 10001 continuum

# Models at pinned revisions, then verified to load with the network off.
# Plain HTTPS downloads: the Xet client stalled at ~40 KB/s inside a Docker
# Desktop build on 13 Oct while HTTPS from the same host ran at ~4.6 MB/s.
COPY scripts/fetch_models.py scripts/fetch_models.py
# The modules fetch_models.py imports for their pinned files (standard
# library only at import time).
COPY diarization.py copyedit.py punctuation.py ./
RUN HF_HUB_DISABLE_XET=1 python scripts/fetch_models.py --half && python scripts/fetch_models.py --verify \
 && chown -R continuum:continuum /opt/models

COPY . .

RUN mkdir -p /app/data && chown continuum:continuum /app/data
USER continuum

# Everything the app needs is in the image; it never downloads at run time.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CONTINUUM_SECURE_COOKIES=1 \
    CONTINUUM_VOCAB_CANDIDATES_PATH=/app/data/glossary_candidates.json \
    CONTINUUM_UPLOAD_DIR=/app/data/uploads \
    CONTINUUM_LOG_FORMAT=json

EXPOSE 8000
VOLUME ["/app/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
