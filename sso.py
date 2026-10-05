"""Single sign-on with the customer's identity provider (OpenID Connect), P0-1.

Works with any OIDC provider: Microsoft Entra ID, Okta, Google Workspace,
Auth0, Keycloak. Each workspace registers one provider with
`python -m auth sso-set` (see auth.py). In the provider, register a web
application whose redirect URI is

    <CONTINUUM_PUBLIC_URL>/sso/callback        e.g. https://kt.example.com/sso/callback

Flow: authorization code with PKCE (RFC 7636).
  1. /sso/start?email=alice@acme.com finds the workspace from the email domain
     and sends the browser to its provider with a random state, nonce and PKCE
     challenge. The state is also put in a short-lived cookie, so a sign-in can
     only finish in the browser that started it (no login CSRF).
  2. /sso/callback exchanges the code, with the PKCE verifier, for an ID token
     and verifies it: the signature against the provider's published keys (RSA
     and EC algorithms only; never "none" or a shared secret), the issuer, the
     audience, the expiry and the nonce. The email domain must be one the
     workspace claimed. A server-side session is then created, the same kind an
     API-key sign-in gets.

People are identified by the token's issuer and subject, not by the email
claim. The client secret is never stored: the configuration names the
environment variable that holds it.
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
import urllib.parse
from typing import Optional, Tuple

import auth

logger = logging.getLogger(__name__)

STATE_COOKIE = "continuum_sso_state"
REQUEST_TTL_SECONDS = 600
ALLOWED_ALGORITHMS = ("RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512")
CLOCK_SKEW_SECONDS = 120
GOOGLE_ISSUER = "https://accounts.google.com"
_METADATA_TTL_SECONDS = 3600
_KEY_REFRESH_MIN_SECONDS = 60
_metadata_cache: dict = {}            # url -> (fetched_at, parsed JSON)

_DOMAIN_RE = re.compile(r"^(?=.{4,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@([^@\s]+\.[^@\s]+)$")
_SAFE_NEXT_RE = re.compile(r"^/(?![/\\])[^\s\\]{0,500}$")


class SSOError(Exception):
    """A sign-in that must not complete. `code` is shown to the person (the
    sign-in page maps it to a message); `reason` is for the server log only."""

    def __init__(self, code: str, reason: str = ""):
        super().__init__(reason or code)
        self.code = code
        self.reason = reason or code


# --------------------------------------------------------------------------
# Workspace configuration
# --------------------------------------------------------------------------

def _parse_domains(domains) -> list:
    if isinstance(domains, str):
        domains = domains.split(",")
    cleaned = sorted({d.strip().lower().lstrip("@") for d in domains if d and d.strip()})
    for d in cleaned:
        if not _DOMAIN_RE.match(d):
            raise ValueError(f"'{d}' is not an email domain")
    if not cleaned:
        raise ValueError("At least one email domain is required (--domains acme.com)")
    return cleaned


def parse_role_map(text: Optional[str]) -> dict:
    """'Continuum.Admin=admin,Continuum.Reviewer=reviewer' -> dict."""
    mapping = {}
    for pair in (text or "").split(","):
        if pair.strip():
            value, _, role = pair.partition("=")
            if not value.strip() or not role.strip():
                raise ValueError(f"Role map entry '{pair}' must look like <IdP value>=<role>")
            mapping[value.strip()] = auth.normalise_role(role)
    return mapping


def set_provider(tenant_id: str, issuer: str, client_id: str, domains, client_secret_env: Optional[str] = None,
                 default_role: str = "receiver", invite_only: bool = False, role_claim: Optional[str] = None,
                 role_map: Optional[dict] = None, enforce: bool = False) -> dict:
    issuer = (issuer or "").strip()
    if not issuer.startswith("https://"):
        raise ValueError("The issuer must be an https:// URL")
    if not (client_id or "").strip():
        raise ValueError("A client id is required")
    if client_secret_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", client_secret_env):
        raise ValueError("--client-secret-env takes the NAME of an environment variable, not the secret itself")
    domains = _parse_domains(domains)
    default_role = auth.normalise_role(default_role)
    role_map = {k: auth.normalise_role(v) for k, v in (role_map or {}).items()}
    with auth._lock, auth._connect() as db:
        auth._require_tenant(db, tenant_id)
        for other, other_domains in db.execute("SELECT tenant_id, domains FROM sso_providers WHERE tenant_id != ?",
                                               (tenant_id,)).fetchall():
            clash = set(domains) & set(other_domains.split(","))
            if clash:
                raise ValueError(f"{', '.join(sorted(clash))} already signs in to workspace {other}")
        db.execute(
            """INSERT INTO sso_providers (tenant_id, issuer, client_id, client_secret_env, domains, default_role,
                                          invite_only, role_claim, role_map, enforce, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(tenant_id) DO UPDATE SET issuer=excluded.issuer, client_id=excluded.client_id,
                   client_secret_env=excluded.client_secret_env, domains=excluded.domains,
                   default_role=excluded.default_role, invite_only=excluded.invite_only,
                   role_claim=excluded.role_claim, role_map=excluded.role_map, enforce=excluded.enforce""",
            (tenant_id, issuer, client_id.strip(), client_secret_env, ",".join(domains), default_role,
             1 if invite_only else 0, role_claim, json.dumps(role_map) if role_map else None, 1 if enforce else 0,
             time.time()))
    auth.audit(tenant_id, "cli", "sso_configured", {"issuer": issuer, "domains": domains, "enforce": enforce})
    return get_provider(tenant_id)


def remove_provider(tenant_id: str) -> bool:
    with auth._lock, auth._connect() as db:
        removed = db.execute("DELETE FROM sso_providers WHERE tenant_id = ?", (tenant_id,)).rowcount > 0
    if removed:
        auth.audit(tenant_id, "cli", "sso_removed")
    return removed


_PROVIDER_COLUMNS = ("tenant_id", "issuer", "client_id", "client_secret_env", "domains", "default_role",
                     "invite_only", "role_claim", "role_map", "enforce")


def _provider_from_row(row) -> dict:
    p = dict(zip(_PROVIDER_COLUMNS, row))
    p["domains"] = p["domains"].split(",")
    p["invite_only"], p["enforce"] = bool(p["invite_only"]), bool(p["enforce"])
    p["role_map"] = json.loads(p["role_map"]) if p["role_map"] else {}
    return p


def get_provider(tenant_id: str) -> Optional[dict]:
    with auth._connect() as db:
        row = db.execute(f"SELECT {', '.join(_PROVIDER_COLUMNS)} FROM sso_providers WHERE tenant_id = ?",
                         (tenant_id,)).fetchone()
    return _provider_from_row(row) if row else None


def provider_for_email(email: str) -> Optional[dict]:
    match = _EMAIL_RE.match((email or "").strip().lower())
    if not match:
        return None
    domain = match.group(1)
    with auth._connect() as db:
        rows = db.execute(f"SELECT {', '.join(_PROVIDER_COLUMNS)} FROM sso_providers").fetchall()
    for row in rows:
        provider = _provider_from_row(row)
        if domain in provider["domains"]:
            return provider
    return None


def sso_enforced(tenant_id: str) -> bool:
    """When on, people of this workspace must sign in through SSO: an API
    key can still call the API, but cannot open a browser session."""
    provider = get_provider(tenant_id)
    return bool(provider and provider["enforce"])


# --------------------------------------------------------------------------
# Talking to the identity provider
# --------------------------------------------------------------------------

def _http_get_json(url: str) -> dict:
    import httpx

    resp = httpx.get(url, timeout=10, follow_redirects=False, headers={"Accept": "application/json"})
    resp.raise_for_status()
    return resp.json()


def _http_post_form(url: str, data: dict, basic_auth: Optional[Tuple[str, str]] = None) -> dict:
    import httpx

    resp = httpx.post(url, data=data, auth=basic_auth, timeout=10, follow_redirects=False,
                      headers={"Accept": "application/json"})
    if resp.status_code >= 400:
        # The body can carry the provider's error code; never tokens.
        try:
            error = resp.json().get("error", "")
        except Exception:
            error = ""
        raise SSOError("sso_failed", f"token endpoint returned {resp.status_code} {error}".strip())
    return resp.json()


def _cached_json(url: str, refresh: bool = False) -> dict:
    hit = _metadata_cache.get(url)
    now = time.time()
    if hit and not refresh and now - hit[0] < _METADATA_TTL_SECONDS:
        return hit[1]
    if hit and refresh and now - hit[0] < _KEY_REFRESH_MIN_SECONDS:
        return hit[1]                        # an unknown key id cannot make us hammer the provider
    try:
        data = _http_get_json(url)
    except SSOError:
        raise
    except Exception as exc:
        raise SSOError("sso_failed", f"could not fetch {url}: {type(exc).__name__}") from exc
    _metadata_cache[url] = (now, data)
    return data


def discover(issuer: str) -> dict:
    meta = _cached_json(issuer.rstrip("/") + "/.well-known/openid-configuration")
    if meta.get("issuer") != issuer:
        raise SSOError("sso_failed", f"the provider's discovery document names issuer {meta.get('issuer')!r}, "
                                     f"not the configured {issuer!r}")
    for field in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
        if not str(meta.get(field, "")).startswith("https://"):
            raise SSOError("sso_failed", f"discovery document has no https {field}")
    return meta


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def redirect_uri_for(request_base_url: str) -> str:
    """The callback URL registered with the provider. CONTINUUM_PUBLIC_URL
    should be set behind a proxy, where the request's own URL is internal."""
    base = os.getenv("CONTINUUM_PUBLIC_URL") or request_base_url
    return base.rstrip("/") + "/sso/callback"


def safe_next(path: Optional[str]) -> str:
    """Only a path on this site may follow a sign-in (no open redirect)."""
    return path if path and _SAFE_NEXT_RE.match(path) else "/"


def begin(provider: dict, redirect_uri: str, next_path: str = "/", login_hint: Optional[str] = None) -> Tuple[str, str]:
    """Return (authorization URL, state). The state must also go into the
    STATE_COOKIE of the browser being redirected."""
    meta = discover(provider["issuer"])
    state, nonce, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(32), secrets.token_urlsafe(64)
    now = time.time()
    with auth._lock, auth._connect() as db:
        db.execute("DELETE FROM sso_requests WHERE expires_at < ?", (now,))
        db.execute("INSERT INTO sso_requests (state_hash, tenant_id, nonce, verifier, redirect_uri, next_path, expires_at) "
                   "VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (auth._hash(state), provider["tenant_id"], nonce, verifier, redirect_uri, safe_next(next_path),
                    now + REQUEST_TTL_SECONDS))
    params = {
        "response_type": "code",
        "client_id": provider["client_id"],
        "redirect_uri": redirect_uri,
        "scope": "openid email profile",
        "state": state,
        "nonce": nonce,
        "code_challenge": _b64url(hashlib.sha256(verifier.encode("ascii")).digest()),
        "code_challenge_method": "S256",
    }
    if login_hint:
        params["login_hint"] = login_hint
    endpoint = meta["authorization_endpoint"]
    return endpoint + ("&" if "?" in endpoint else "?") + urllib.parse.urlencode(params), state


def _take_request(state: str) -> Optional[dict]:
    """A pending sign-in, at most once: the row is deleted as it is read."""
    with auth._lock, auth._connect() as db:
        row = db.execute("SELECT tenant_id, nonce, verifier, redirect_uri, next_path, expires_at FROM sso_requests "
                         "WHERE state_hash = ?", (auth._hash(state),)).fetchone()
        db.execute("DELETE FROM sso_requests WHERE state_hash = ?", (auth._hash(state),))
    if not row or row[5] < time.time():
        return None
    return dict(zip(("tenant_id", "nonce", "verifier", "redirect_uri", "next_path"), row[:5]))


def _exchange_code(provider: dict, meta: dict, code: str, request: dict) -> dict:
    data = {"grant_type": "authorization_code", "code": code, "redirect_uri": request["redirect_uri"],
            "client_id": provider["client_id"], "code_verifier": request["verifier"]}
    basic = None
    if provider.get("client_secret_env"):
        secret = os.getenv(provider["client_secret_env"])
        if not secret:
            raise SSOError("sso_failed", f"environment variable {provider['client_secret_env']} (client secret) is not set")
        methods = meta.get("token_endpoint_auth_methods_supported") or ["client_secret_basic"]
        if "client_secret_basic" in methods:          # form-encoded first, RFC 6749 section 2.3.1
            basic = (urllib.parse.quote(provider["client_id"], safe=""), urllib.parse.quote(secret, safe=""))
        else:
            data["client_secret"] = secret
    tokens = _http_post_form(meta["token_endpoint"], data, basic)
    if not isinstance(tokens, dict) or not tokens.get("id_token"):
        raise SSOError("sso_failed", "token response has no id_token")
    return tokens


def _signing_keys(jwks_uri: str, kid: Optional[str], alg: str) -> dict:
    kty = "EC" if alg.startswith("ES") else "RSA"

    def matching(jwks):
        return [k for k in (jwks or {}).get("keys", []) if isinstance(k, dict) and k.get("kty") == kty
                and k.get("use", "sig") == "sig" and (kid is None or k.get("kid") == kid)]

    keys = matching(_cached_json(jwks_uri))
    if not keys:                                    # the provider may have rotated its keys
        keys = matching(_cached_json(jwks_uri, refresh=True))
    if not keys:
        raise SSOError("sso_failed", f"signing key {kid!r} is not published by the provider")
    return {"keys": keys}


def verify_id_token(id_token: str, provider: dict, meta: dict, nonce: str,
                    access_token: Optional[str] = None) -> dict:
    from jose import jwt
    from jose.exceptions import JOSEError

    try:
        header = jwt.get_unverified_header(id_token)
    except JOSEError as exc:
        raise SSOError("sso_failed", "malformed ID token") from exc
    alg = header.get("alg")
    if alg not in ALLOWED_ALGORITHMS:
        raise SSOError("sso_failed", f"ID token algorithm {alg!r} is not accepted")
    keys = _signing_keys(meta["jwks_uri"], header.get("kid"), alg)
    try:
        claims = jwt.decode(
            id_token, keys, algorithms=[alg], audience=provider["client_id"], issuer=provider["issuer"],
            access_token=access_token,
            options={"leeway": CLOCK_SKEW_SECONDS, "require_exp": True, "require_iat": True, "require_sub": True,
                     "verify_at_hash": access_token is not None},
        )
    except JOSEError as exc:
        raise SSOError("sso_failed", f"ID token rejected: {exc}") from exc
    audience = claims.get("aud")
    if isinstance(audience, list) and len(audience) > 1 and claims.get("azp") != provider["client_id"]:
        raise SSOError("sso_failed", "ID token was issued to another client (azp)")
    if not isinstance(claims.get("nonce"), str) or not hmac.compare_digest(claims["nonce"], nonce):
        raise SSOError("sso_failed", "ID token nonce does not match this sign-in")
    if not isinstance(claims.get("sub"), str) or not claims["sub"]:
        raise SSOError("sso_failed", "ID token has no subject")
    return claims


def _email_from(claims: dict) -> Optional[str]:
    for name in ("email", "preferred_username", "upn"):
        value = claims.get(name)
        if isinstance(value, str) and _EMAIL_RE.match(value.strip().lower()):
            return value.strip().lower()
    return None


def _check_allowed(claims: dict, email: Optional[str], provider: dict) -> None:
    if not email:
        raise SSOError("sso_not_allowed", "ID token carries no email address")
    if claims.get("email_verified") in (False, "false"):
        raise SSOError("sso_not_allowed", f"{email} is not verified at the provider")
    if email.rsplit("@", 1)[1] not in provider["domains"]:
        raise SSOError("sso_not_allowed", f"{email} is outside the workspace's domains")
    # Every Google account shares one issuer: only the hosted-domain claim
    # proves the account belongs to the customer's Google Workspace.
    if provider["issuer"].rstrip("/") == GOOGLE_ISSUER and claims.get("hd") not in provider["domains"]:
        raise SSOError("sso_not_allowed", "Google account is not part of the workspace's Google Workspace domain")


def _role_from_claims(claims: dict, provider: dict) -> Optional[str]:
    """With a role claim configured, the provider's groups decide the role at
    every sign-in (someone removed from the admin group loses admin)."""
    claim, mapping = provider.get("role_claim"), provider.get("role_map") or {}
    if not claim or not mapping:
        return None
    values = claims.get(claim)
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        values = []
    return auth.highest_role(mapping.get(str(v)) for v in values) or provider["default_role"]


def complete(code: Optional[str], state: Optional[str], cookie_state: Optional[str]) -> Tuple[auth.Principal, str]:
    """Finish a sign-in: returns the Principal to open a session for, and the
    path to send the browser to. Raises SSOError."""
    if not state or not cookie_state or not hmac.compare_digest(state, cookie_state):
        raise SSOError("sso_expired", "state missing or not from this browser")
    request = _take_request(state)
    if request is None:
        raise SSOError("sso_expired", "unknown, used or expired state")
    if not code:
        raise SSOError("sso_failed", "no authorization code")
    provider = get_provider(request["tenant_id"])
    if provider is None:
        raise SSOError("sso_failed", "the workspace no longer uses SSO")
    meta = discover(provider["issuer"])
    tokens = _exchange_code(provider, meta, code, request)
    claims = verify_id_token(tokens["id_token"], provider, meta, request["nonce"], tokens.get("access_token"))
    email = _email_from(claims)
    _check_allowed(claims, email, provider)
    try:
        user = auth.sso_sign_in(provider["tenant_id"], provider["issuer"], claims["sub"], email,
                                claims.get("name") if isinstance(claims.get("name"), str) else None,
                                _role_from_claims(claims, provider), provider["default_role"], provider["invite_only"])
    except PermissionError as exc:
        raise SSOError(str(exc), f"{email}: {exc}") from exc
    principal = auth.Principal(provider["tenant_id"], user["role"], user["email"],
                               auth.tenant_llm_policy(provider["tenant_id"]), user_id=user["user_id"], method="sso")
    return principal, request["next_path"] or "/"


def tenant_for_state(state: Optional[str]) -> Optional[str]:
    """For the audit log of a failed sign-in (does not consume the state)."""
    if not state:
        return None
    with auth._connect() as db:
        row = db.execute("SELECT tenant_id FROM sso_requests WHERE state_hash = ?", (auth._hash(state),)).fetchone()
    return row[0] if row else None


# --------------------------------------------------------------------------
# CLI (python -m auth sso-set / sso-show / sso-remove)
# --------------------------------------------------------------------------

def cli(cmd: str, args: list) -> int:
    tenant_id = args.pop(0)
    if cmd == "sso-show":
        provider = get_provider(tenant_id)
        print(json.dumps(provider, indent=2) if provider else "SSO is not configured for this workspace.")
        return 0
    if cmd == "sso-remove":
        print("removed" if remove_provider(tenant_id) else "SSO was not configured for this workspace.")
        return 0
    issuer = auth._option(args, "--issuer")
    client_id = auth._option(args, "--client-id")
    domains = auth._option(args, "--domains")
    if not (issuer and client_id and domains):
        print("usage: python -m auth sso-set <tenant_id> --issuer URL --client-id ID --domains acme.com "
              "[--client-secret-env VAR] [--default-role receiver] [--invite-only] "
              "[--role-claim roles --role-map Value=role,...] [--enforce]")
        return 2
    provider = set_provider(
        tenant_id, issuer, client_id, domains,
        client_secret_env=auth._option(args, "--client-secret-env"),
        default_role=auth._option(args, "--default-role", "receiver"),
        role_claim=auth._option(args, "--role-claim"),
        role_map=parse_role_map(auth._option(args, "--role-map")),
        invite_only=auth._flag(args, "--invite-only"),
        enforce=auth._flag(args, "--enforce"),
    )
    print(json.dumps(provider, indent=2))
    base = os.getenv("CONTINUUM_PUBLIC_URL", "https://<your Continuum host>")
    print(f"\nRegister this redirect URI with the identity provider: {base.rstrip('/')}/sso/callback")
    try:
        discover(provider["issuer"])
        print("Discovery document found and the issuer matches.")
    except SSOError as exc:
        print(f"WARNING: {exc.reason}. Check the issuer URL before people try to sign in.")
    return 0
