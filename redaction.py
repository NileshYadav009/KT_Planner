"""Remove credentials from transcript text before anything else sees it (P0-9).

A KT giver reading out a password or pasting a key into a transcript used to
send it to the LLM provider, the response cache, the job database and the
PDF. Redaction runs first, so none of them ever receives it.

Deliberately limited to secrets: names, emails and phone numbers of
escalation contacts are operational knowledge a KT must keep. A value after
"password is" / "token:" is only redacted when it looks like a credential
(digits, mixed case or symbols, or a known token format), so "the token is
rotated every 90 days" stays readable.
"""
import re
from typing import Tuple

REDACTED = "[REDACTED]"

# Formats that are a secret wherever they appear.
_TOKEN_RE = re.compile(
    r"\bAKIA[0-9A-Z]{16}\b"                                   # AWS access key id
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"
    r"|\bgsk_[A-Za-z0-9]{10,}|\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}|\bsk_(?:live|test)_[A-Za-z0-9]{10,}"
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}|\bxox[abpr]-[A-Za-z0-9-]{10,}"
    r"|\bAIza[0-9A-Za-z_-]{30,}|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"
    r"|\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*",
)
# user:password@host in connection strings and URLs.
_URL_CREDENTIALS_RE = re.compile(r"(\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)([^\s@/]+)(@)", re.IGNORECASE)
# Environment-variable style: AWS_SECRET_ACCESS_KEY=..., DB_PASSWORD: ...
# A variable named like a secret holds one, whatever the value looks like.
_ENV_ASSIGNMENT_RE = re.compile(
    r"\b([A-Z0-9_]*(?:SECRET|PASSWORD|PASSWD|TOKEN|API_KEY|ACCESS_KEY|PRIVATE_KEY)[A-Z0-9_]*\s*[=:]\s*)"
    r"(['\"]?)([^\s'\"]{4,})(\2)"
)
# "<secret word> is/=/: <value>"
_ASSIGNMENT_RE = re.compile(
    r"(\b(?:password|passphrase|passwd|pwd|secret(?:\s+key)?|client\s+secret|api[\s_-]?key|access[\s_-]?key|"
    r"secret[\s_-]?access[\s_-]?key|token|private[\s_-]?key)\b\s*(?:is|was|=|:|equals)\s*)"
    r"(['\"]?)([^\s'\",;]{4,})(\2)",
    re.IGNORECASE,
)


def _looks_like_credential(value: str) -> bool:
    if len(value) < 6:
        return False
    has_digit = any(c.isdigit() for c in value)
    has_symbol = any(not c.isalnum() for c in value)
    mixed_case = any(c.islower() for c in value) and any(c.isupper() for c in value)
    return has_digit or has_symbol or mixed_case


def redact_secrets(text: str) -> Tuple[str, int]:
    """(text with credentials replaced by [REDACTED], number replaced)."""
    if not text:
        return text, 0
    count = 0

    def _sub_token(m):
        nonlocal count
        count += 1
        return REDACTED

    text = _TOKEN_RE.sub(_sub_token, text)

    def _sub_url(m):
        nonlocal count
        count += 1
        return m.group(1) + REDACTED + m.group(3)

    text = _URL_CREDENTIALS_RE.sub(_sub_url, text)

    def _sub_env(m):
        nonlocal count
        if REDACTED in m.group(3):
            return m.group(0)
        count += 1
        return m.group(1) + REDACTED

    text = _ENV_ASSIGNMENT_RE.sub(_sub_env, text)

    def _sub_assign(m):
        nonlocal count
        value = m.group(3)
        quoted = bool(m.group(2))
        if REDACTED in value or not (quoted or _looks_like_credential(value)):
            return m.group(0)
        count += 1
        return m.group(1) + REDACTED

    text = _ASSIGNMENT_RE.sub(_sub_assign, text)
    return text, count
