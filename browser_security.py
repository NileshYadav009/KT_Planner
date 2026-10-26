"""Browser security headers on every response (enterprise checklist, P0).

- Content-Security-Policy: scripts only from this origin, the one CDN the
  page loads (cdnjs, for marked) and the pages' own inline scripts, each
  named by its hash. Text that slips past escaping into the page cannot run,
  and no other site's script can be loaded. Styles may be inline (the page
  sets style attributes); images and media only from this origin, data: and
  blob: (a recording played from the user's own file). No site may frame
  the app.
- Strict-Transport-Security on HTTPS (directly or behind a proxy that says
  so), for a year; CONTINUUM_HSTS_SECONDS=0 leaves it out.
- X-Frame-Options, X-Content-Type-Options, Referrer-Policy (signed screenshot
  links never reach another site in a Referer), Permissions-Policy and
  Cross-Origin-Opener-Policy.

The API reference (/docs, signed-in users only) is Swagger UI, which loads
its script and styles from jsDelivr with an inline start-up script: it gets
a policy of its own.
"""
import base64
import functools
import glob
import hashlib
import os
import re
from typing import Dict, List, Tuple

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
DOCS_PATHS = ("/docs",)

_INLINE_SCRIPT_RE = re.compile(r"<script(?![^>]*\bsrc\s*=)[^>]*>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL)

_POLICY = (
    "default-src 'self'; script-src 'self' https://cdnjs.cloudflare.com {hashes}; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; "
    "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)
_DOCS_POLICY = (
    "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com; "
    "connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
)
_ALWAYS = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}


@functools.lru_cache(maxsize=32)
def _hashes_in(path: str, mtime: float) -> Tuple[str, ...]:
    with open(path, encoding="utf-8") as fh:
        page = fh.read()
    # Browsers hash a script's text after turning CR LF into LF.
    page = page.replace("\r\n", "\n").replace("\r", "\n")
    return tuple("'sha256-%s'" % base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode("ascii")
                 for body in _INLINE_SCRIPT_RE.findall(page))


def inline_script_hashes(static_dir: str = STATIC_DIR) -> List[str]:
    """The hash of every inline script of the app's pages, read again when a
    page changes."""
    hashes: List[str] = []
    for path in sorted(glob.glob(os.path.join(static_dir, "*.html"))):
        for h in _hashes_in(path, os.path.getmtime(path)):
            if h not in hashes:
                hashes.append(h)
    return hashes


def content_security_policy(static_dir: str = STATIC_DIR) -> str:
    return _POLICY.format(hashes=" ".join(inline_script_hashes(static_dir)))


def _hsts_seconds() -> int:
    try:
        return max(0, int(os.getenv("CONTINUUM_HSTS_SECONDS", str(365 * 24 * 3600))))
    except ValueError:
        return 365 * 24 * 3600


def headers_for(path: str, https: bool) -> Dict[str, str]:
    """The headers for a response to `path`, served over HTTPS or not."""
    headers = dict(_ALWAYS)
    headers["Content-Security-Policy"] = _DOCS_POLICY if path in DOCS_PATHS else content_security_policy()
    seconds = _hsts_seconds()
    if https and seconds:
        headers["Strict-Transport-Security"] = f"max-age={seconds}"
    return headers
