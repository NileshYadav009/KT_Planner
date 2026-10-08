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
# own index; everything else is pinned in requirements.lock.
COPY requirements.lock .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "$(grep '^torch==' requirements.lock)" \
 && pip install -r requirements.lock

# Models at pinned revisions, then verified to load with the network off.
COPY scripts/fetch_models.py scripts/fetch_models.py
COPY diarization.py diarization.py
RUN python scripts/fetch_models.py && python scripts/fetch_models.py --verify

COPY . .

RUN useradd --create-home --uid 10001 continuum \
 && mkdir -p /app/data \
 && chown -R continuum:continuum /app/data /opt/models
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
