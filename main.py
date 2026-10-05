from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

import logging
logger = logging.getLogger(__name__)

from api import router
import media_guard
import pipeline

app = FastAPI()

# CORS configuration - restrict to allowed origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Continuum-CSRF"],
)



@app.middleware("http")
async def refuse_oversized_uploads(request: Request, call_next):
    """Refuse an upload that declares a size over the limit before its body
    is read (media_guard.save_upload also stops one that does not declare
    it). Set the same limit on the reverse proxy in production."""
    if request.method == "POST" and request.url.path == "/upload":
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > media_guard.max_upload_bytes() + 64 * 1024:   # + multipart overhead
            detail = (f"The file is larger than {media_guard.upload_limit_label()}. "
                      "Upload a shorter recording or the audio track only.")
            return JSONResponse({"detail": detail}, status_code=413)
    return await call_next(request)


# Serve the frontend static files
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)


@app.on_event("startup")
def load_models():
    """Load heavy models on startup so endpoints can use them."""
    interrupted = pipeline.JOB_STORE.mark_interrupted()
    if interrupted:
        logger.warning("%d job(s) were interrupted by the last shutdown and marked failed", interrupted)
    expired = pipeline.apply_retention()
    if expired:
        logger.info("Retention: deleted %d job(s) older than CONTINUUM_RETENTION_DAYS", expired)
    pipeline.load_models()
