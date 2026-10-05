"""P0-1: single sign-on (OpenID Connect) and roles.

A fake identity provider (its own RSA signing key, discovery document, key
set and token endpoint) stands in for Entra ID / Okta / Google, so every
check runs without network access: signature, issuer, audience, expiry,
nonce, PKCE, state binding, email domain, roles and session lifetime."""
import base64
import hashlib
import json
import os
import secrets
import sys
import time
import urllib.parse

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import auth
import sso

ISSUER = "https://idp.example.test/acme/v2.0"
CLIENT_ID = "continuum-test-app"


def _b64url_uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _pem(key) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


class FakeIdP:
    """Just enough of an OpenID provider: discovery, JWKS, and a token
    endpoint that checks the PKCE verifier and signs an ID token."""

    def __init__(self, issuer=ISSUER):
        self.issuer = issuer
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.kid = "key-1"
        self.codes = {}
        self.token_requests = []
        self.discovery = {
            "issuer": issuer,
            "authorization_endpoint": "https://idp.example.test/authorize",
            "token_endpoint": "https://idp.example.test/token",
            "jwks_uri": "https://idp.example.test/keys",
            "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
        }

    def jwks(self):
        numbers = self.key.public_key().public_numbers()
        return {"keys": [{"kty": "RSA", "use": "sig", "alg": "RS256", "kid": self.kid,
                          "n": _b64url_uint(numbers.n), "e": _b64url_uint(numbers.e)}]}

    def get_json(self, url):
        if url == self.issuer.rstrip("/") + "/.well-known/openid-configuration":
            return self.discovery
        if url == self.discovery["jwks_uri"]:
            return self.jwks()
        raise AssertionError(f"unexpected fetch {url}")

    def authorize(self, location, claims=None, token=None):
        """The person signs in at the provider; it redirects back with a code."""
        query = urllib.parse.parse_qs(urllib.parse.urlparse(location).query)
        code = secrets.token_urlsafe(16)
        self.codes[code] = {"nonce": query["nonce"][0], "challenge": query["code_challenge"][0],
                            "method": query["code_challenge_method"][0], "claims": claims or {}, "token": token}
        return code, query["state"][0], query

    def sign(self, claims, key=None, kid=None, alg="RS256"):
        return jwt.encode(claims, _pem(key or self.key), algorithm=alg, headers={"kid": kid or self.kid})

    def post_form(self, url, data, basic_auth=None):
        self.token_requests.append({"url": url, "data": dict(data), "basic_auth": basic_auth})
        entry = self.codes.pop(data.get("code"), None)
        verifier = data.get("code_verifier", "")
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        if entry is None or entry["method"] != "S256" or challenge != entry["challenge"]:
            raise sso.SSOError("sso_failed", "token endpoint returned 400 invalid_grant")
        now = int(time.time())
        claims = {"iss": self.issuer, "aud": CLIENT_ID, "sub": "sub-alice", "email": "alice@acme.com",
                  "name": "Alice Example", "nonce": entry["nonce"], "iat": now, "exp": now + 600}
        claims.update(entry["claims"])
        claims = {k: v for k, v in claims.items() if v is not None}
        token = entry["token"](claims) if callable(entry["token"]) else self.sign(claims)
        return {"id_token": token, "access_token": "access-token-123", "token_type": "Bearer"}


@pytest.fixture
def idp(monkeypatch):
    fake = FakeIdP()
    monkeypatch.setattr(sso, "_http_get_json", fake.get_json)
    monkeypatch.setattr(sso, "_http_post_form", fake.post_form)
    sso._metadata_cache.clear()
    yield fake
    sso._metadata_cache.clear()


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import main
    import pipeline

    monkeypatch.setenv("CONTINUUM_AUTH", "required")
    monkeypatch.setattr(pipeline, "KT_ASSETS_DIR", str(tmp_path / "assets"))
    monkeypatch.setattr(pipeline, "run_kt_pipeline", lambda *a, **k: None)
    return TestClient(main.app)


@pytest.fixture
def acme(idp):
    tenant = auth.create_tenant("Acme")
    sso.set_provider(tenant, ISSUER, CLIENT_ID, "acme.com", default_role="receiver")
    yield tenant
    sso.remove_provider(tenant)


def _sign_in(client, idp, email="alice@acme.com", claims=None, token=None, next_path="/"):
    start = client.get("/sso/start", params={"email": email, "next": next_path}, follow_redirects=False)
    assert start.status_code == 302, start.text
    code, state, query = idp.authorize(start.headers["location"], claims, token)
    return client.get("/sso/callback", params={"code": code, "state": state}, follow_redirects=False), query


def _error(resp):
    assert resp.status_code == 302
    return urllib.parse.parse_qs(urllib.parse.urlparse(resp.headers["location"]).query).get("sso_error", [None])[0]


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------

def test_sso_sign_in_opens_a_session_for_the_person(client, idp, acme):
    resp, query = _sign_in(client, idp)
    assert resp.status_code == 302 and resp.headers["location"] == "/"
    cookie = resp.headers.get("set-cookie", "").lower()
    assert auth.SESSION_COOKIE in cookie and "httponly" in cookie
    # The authorization request used PKCE, a nonce and the registered callback.
    assert query["code_challenge_method"] == ["S256"] and query["nonce"][0] and query["state"][0]
    assert query["redirect_uri"][0].endswith("/sso/callback") and query["client_id"] == [CLIENT_ID]
    assert query["login_hint"] == ["alice@acme.com"]
    assert idp.token_requests[-1]["data"]["code_verifier"]           # verifier sent, checked by the fake IdP

    me = client.get("/me").json()
    assert me["user"] == "alice@acme.com" and me["method"] == "sso" and me["workspace"] == "Acme"
    assert me["role"] == "receiver" and me["permissions"] == ["kt:read"]
    assert client.get("/jobs").status_code == 200
    assert any(e["action"] == "sign_in" and e["detail"]["method"] == "sso" for e in auth.audit_events(acme))


def test_unknown_email_domain_is_sent_back_with_a_message(client, idp, acme):
    resp = client.get("/sso/start", params={"email": "bob@unknown-company.com"}, follow_redirects=False)
    assert _error(resp) == "sso_unknown_domain"
    page = client.get("/")
    assert "Continue with SSO" in page.text and "sso_unknown_domain" in page.text   # message table in the page


# --------------------------------------------------------------------------
# Tokens that must be refused
# --------------------------------------------------------------------------

def _other_key_token(idp):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return lambda claims: idp.sign(claims, key=other)                 # same kid, wrong key


def _alg_none_token(claims):
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT","kid":"key-1"}').rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"{header}.{body}."


def _hmac_with_public_key_token(idp):
    """The classic algorithm-confusion forgery: HS256 keyed with the
    provider's public key, which anyone can download. Built by hand, the way
    an attacker would (JWT libraries refuse to make it)."""
    import hmac as hmac_lib

    public_pem = idp.key.public_key().public_bytes(serialization.Encoding.PEM,
                                                   serialization.PublicFormat.SubjectPublicKeyInfo)

    def forge(claims):
        def b64(data):
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode()
        signing_input = b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": idp.kid}).encode()) + "." + \
            b64(json.dumps(claims).encode())
        signature = hmac_lib.new(public_pem, signing_input.encode(), hashlib.sha256).digest()
        return signing_input + "." + b64(signature)
    return forge


@pytest.mark.parametrize("case", ["wrong_audience", "wrong_issuer", "expired", "wrong_nonce", "no_subject",
                                  "wrong_key", "alg_none", "alg_confusion", "garbage"])
def test_forged_or_stale_id_tokens_are_refused(client, idp, acme, case):
    now = int(time.time())
    claims, token = None, None
    if case == "wrong_audience":
        claims = {"aud": "some-other-app"}
    elif case == "wrong_issuer":
        claims = {"iss": "https://evil.example.test/"}
    elif case == "expired":
        claims = {"iat": now - 7200, "exp": now - 3600}
    elif case == "wrong_nonce":
        claims = {"nonce": "nonce-from-another-sign-in"}
    elif case == "no_subject":
        claims = {"sub": None}
    elif case == "wrong_key":
        token = _other_key_token(idp)
    elif case == "alg_none":
        token = _alg_none_token
    elif case == "alg_confusion":
        token = _hmac_with_public_key_token(idp)
    elif case == "garbage":
        token = lambda claims: "not-a-jwt"
    resp, _ = _sign_in(client, idp, claims=claims, token=token)
    assert _error(resp) == "sso_failed"
    assert auth.SESSION_COOKIE not in resp.headers.get("set-cookie", "")
    assert client.get("/jobs").status_code == 401
    assert any(e["action"] == "sign_in_failed" for e in auth.audit_events(acme))


def test_a_wrong_pkce_verifier_is_refused(client, idp, acme, monkeypatch):
    original = idp.post_form

    def tampered(url, data, basic_auth=None):
        return original(url, dict(data, code_verifier="attacker-guess"), basic_auth)
    monkeypatch.setattr(sso, "_http_post_form", tampered)
    resp, _ = _sign_in(client, idp)
    assert _error(resp) == "sso_failed"


def test_the_sign_in_state_is_single_use_and_bound_to_the_browser(client, idp, acme):
    from fastapi.testclient import TestClient
    import main

    start = client.get("/sso/start", params={"email": "alice@acme.com"}, follow_redirects=False)
    code, state, _ = idp.authorize(start.headers["location"])
    # Another browser (no state cookie) cannot finish this sign-in: login CSRF.
    other_browser = TestClient(main.app)
    assert _error(other_browser.get("/sso/callback", params={"code": code, "state": state},
                                    follow_redirects=False)) == "sso_expired"
    assert other_browser.get("/jobs").status_code == 401
    # The browser that started it still can, once.
    first = client.get("/sso/callback", params={"code": code, "state": state}, follow_redirects=False)
    assert first.headers["location"] == "/"
    client.post("/logout")
    client.cookies.set(sso.STATE_COOKIE, state, path="/sso")              # even with the cookie put back
    assert _error(client.get("/sso/callback", params={"code": code, "state": state},
                             follow_redirects=False)) == "sso_expired"


def test_provider_errors_are_reported_without_echoing_them(client, idp, acme):
    start = client.get("/sso/start", params={"email": "alice@acme.com"}, follow_redirects=False)
    _, state, _ = idp.authorize(start.headers["location"])
    resp = client.get("/sso/callback", params={"error": "access_denied<script>", "state": state},
                      follow_redirects=False)
    assert resp.headers["location"] == "/?sso_error=sso_denied"


# --------------------------------------------------------------------------
# Who may sign in, and with which role
# --------------------------------------------------------------------------

@pytest.mark.parametrize("claims", [{"email": "mallory@evil.example"}, {"email_verified": False},
                                    {"email": None, "preferred_username": None}])
def test_people_outside_the_workspace_domain_are_refused(client, idp, acme, claims):
    resp, _ = _sign_in(client, idp, claims=claims)
    assert _error(resp) == "sso_not_allowed"


def test_google_accounts_must_belong_to_the_customers_google_workspace():
    provider = {"issuer": sso.GOOGLE_ISSUER, "domains": ["acme.com"]}
    with pytest.raises(sso.SSOError):
        sso._check_allowed({"email": "alice@acme.com", "email_verified": True}, "alice@acme.com", provider)
    sso._check_allowed({"email": "alice@acme.com", "email_verified": True, "hd": "acme.com"}, "alice@acme.com",
                       provider)


def test_roles_follow_the_identity_providers_groups(client, idp):
    tenant = auth.create_tenant("Acme Roles")
    sso.set_provider(tenant, ISSUER, CLIENT_ID, "acme.com", role_claim="roles",
                     role_map={"Continuum.Admin": "admin", "Continuum.Reviewer": "reviewer"})
    try:
        _sign_in(client, idp, claims={"roles": ["Continuum.Reviewer", "Continuum.Admin"]})
        assert client.get("/me").json()["role"] == "admin"
        client.post("/logout")
        _sign_in(client, idp, claims={"roles": []})           # removed from the groups at the IdP
        assert client.get("/me").json()["role"] == "receiver"
    finally:
        sso.remove_provider(tenant)


def test_invite_only_workspaces_admit_only_pre_approved_people(client, idp):
    tenant = auth.create_tenant("Acme Invite")
    sso.set_provider(tenant, ISSUER, CLIENT_ID, "acme.com", invite_only=True)
    try:
        assert _error(_sign_in(client, idp)[0]) == "sso_not_allowed"
        auth.add_user(tenant, "alice@acme.com", "reviewer")
        _sign_in(client, idp)
        assert client.get("/me").json()["role"] == "reviewer"
        client.post("/logout")
        # A different identity claiming the same email is not Alice.
        assert _error(_sign_in(client, idp, claims={"sub": "someone-else"})[0]) == "sso_not_allowed"
    finally:
        sso.remove_provider(tenant)


def test_disabling_a_person_ends_their_session_immediately(client, idp, acme):
    _sign_in(client, idp)
    assert client.get("/jobs").status_code == 200
    auth.set_user_disabled(acme, "alice@acme.com", True)
    assert client.get("/jobs").status_code == 401
    assert _error(_sign_in(client, idp)[0]) == "sso_disabled"
    auth.set_user_disabled(acme, "alice@acme.com", False)
    _sign_in(client, idp)
    assert client.get("/jobs").status_code == 200


def test_a_role_change_applies_to_an_open_session(client, idp, acme):
    _sign_in(client, idp)
    assert client.get("/me").json()["role"] == "receiver"
    auth.set_user_role(acme, "alice@acme.com", "giver")
    assert client.get("/me").json()["role"] == "giver"


def test_enforced_sso_blocks_key_sign_in_but_not_machine_api_calls(client, idp):
    tenant = auth.create_tenant("Acme Enforced")
    sso.set_provider(tenant, ISSUER, CLIENT_ID, "acme.com", enforce=True)
    try:
        key = auth.create_key(tenant, "ci-bot")
        resp = client.post("/login", json={"api_key": key})
        assert resp.status_code == 403 and "single sign-on" in resp.json()["detail"]
        assert client.get("/jobs", headers={"Authorization": f"Bearer {key}"}).status_code == 200
    finally:
        sso.remove_provider(tenant)


def test_after_sign_in_the_browser_only_goes_to_a_page_on_this_site(client, idp, acme):
    resp, _ = _sign_in(client, idp, next_path="/?job=abc-123")
    assert resp.headers["location"] == "/?job=abc-123"
    for evil in ("//evil.example/x", "https://evil.example/", "/\\evil.example", "javascript:alert(1)"):
        client.post("/logout")
        resp, _ = _sign_in(client, idp, next_path=evil)
        assert resp.headers["location"] == "/", evil


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_configuration_is_validated(idp):
    tenant = auth.create_tenant("Acme Config")
    try:
        with pytest.raises(ValueError):
            sso.set_provider(tenant, "http://idp.example.test", CLIENT_ID, "acme.com")      # not https
        with pytest.raises(ValueError):
            sso.set_provider(tenant, ISSUER, CLIENT_ID, "acme.com", client_secret_env="s3cr3t-value!")
        with pytest.raises(ValueError):
            sso.set_provider(tenant, ISSUER, CLIENT_ID, "not a domain")
        sso.set_provider(tenant, ISSUER, CLIENT_ID, "acme-config.com")
        other = auth.create_tenant("Squatter")
        with pytest.raises(ValueError):                    # a domain signs in to one workspace only
            sso.set_provider(other, ISSUER, CLIENT_ID, "acme-config.com")
    finally:
        sso.remove_provider(tenant)


def test_client_secret_comes_from_the_environment_and_is_never_stored(client, idp, monkeypatch):
    tenant = auth.create_tenant("Acme Secret")
    sso.set_provider(tenant, ISSUER, CLIENT_ID, "acme.com", client_secret_env="ACME_SSO_SECRET")
    try:
        assert _error(_sign_in(client, idp)[0]) == "sso_failed"            # variable not set
        monkeypatch.setenv("ACME_SSO_SECRET", "the-real-secret")
        resp, _ = _sign_in(client, idp)
        assert resp.headers["location"] == "/"
        assert idp.token_requests[-1]["basic_auth"] == (CLIENT_ID, "the-real-secret")
        with auth._connect() as db:
            stored = " ".join(str(v) for row in db.execute("SELECT * FROM sso_providers") for v in row)
        assert "the-real-secret" not in stored
    finally:
        sso.remove_provider(tenant)


def test_a_discovery_document_for_another_issuer_is_refused(client, idp, acme):
    idp.discovery["issuer"] = "https://evil.example.test/"
    resp = client.get("/sso/start", params={"email": "alice@acme.com"}, follow_redirects=False)
    assert _error(resp) == "sso_failed"


def test_rotated_signing_keys_are_picked_up(client, idp, acme, monkeypatch):
    monkeypatch.setattr(sso, "_KEY_REFRESH_MIN_SECONDS", 0)
    _sign_in(client, idp)
    client.post("/logout")
    idp.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    idp.kid = "key-2"
    resp, _ = _sign_in(client, idp)
    assert resp.headers["location"] == "/"


# --------------------------------------------------------------------------
# Roles on the API
# --------------------------------------------------------------------------

TRANSCRIPT = ("This KT is for TripWise, the booking backend. Bookings are written to DynamoDB. "
              "Datadog is our monitoring tool and PagerDuty pages us. RTO is four hours. "
              "Never delete items from the bookings table by hand. Escalate to Marco Silva.")


def _h(key):
    return {"Authorization": f"Bearer {key}"}


def test_roles_decide_who_may_create_correct_and_delete(client):
    import pipeline

    tenant = auth.create_tenant("Role Co")
    admin, giver, other_giver, receiver = (auth.create_key(tenant, n, role=r) for n, r in
                                           (("adm", "admin"), ("gina", "giver"), ("gus", "giver"), ("rita", "receiver")))
    # A receiver can read but not create or correct.
    assert client.post("/kt-from-transcript", json={"transcript": TRANSCRIPT}, headers=_h(receiver)).status_code == 403
    job_id = client.post("/kt-from-transcript", json={"transcript": TRANSCRIPT}, headers=_h(giver)).json()["job_id"]
    other_job = client.post("/kt-from-transcript", json={"transcript": TRANSCRIPT}, headers=_h(other_giver)).json()["job_id"]
    try:
        assert client.get(f"/status/{job_id}", headers=_h(receiver)).status_code == 200
        feedback = {"job_id": job_id, "sentence_id": 0, "corrected_classification": "danger_zones"}
        assert client.post("/feedback", json=feedback, headers=_h(receiver)).status_code == 403
        assert client.post("/feedback", json=feedback, headers=_h(giver)).status_code == 200

        listed = {j["job_id"]: j["can_delete"] for j in client.get("/jobs", headers=_h(giver)).json()["jobs"]}
        assert listed[job_id] is True and listed[other_job] is False
        assert all(j["can_delete"] is False for j in client.get("/jobs", headers=_h(receiver)).json()["jobs"])

        # A giver deletes their own KT, not a colleague's; an admin deletes any.
        assert client.delete(f"/jobs/{other_job}", headers=_h(giver)).status_code == 403
        assert client.delete(f"/jobs/{job_id}", headers=_h(receiver)).status_code == 403
        assert client.delete(f"/jobs/{job_id}", headers=_h(giver)).status_code == 200
        assert client.delete(f"/jobs/{other_job}", headers=_h(admin)).status_code == 200
        assert [e["action"] for e in auth.audit_events(tenant)].count("kt_deleted") == 2
    finally:
        pipeline.JOB_STORE.delete(job_id)
        pipeline.JOB_STORE.delete(other_job)


def test_revoking_a_key_ends_browser_sessions_opened_with_it(client):
    tenant = auth.create_tenant("Revoke Co")
    key = auth.create_key(tenant, "laptop")
    client.post("/login", json={"api_key": key})
    assert client.get("/jobs").status_code == 200
    auth.revoke_key(key[:12])
    assert client.get("/jobs").status_code == 401


def test_keys_created_before_roles_keep_working_as_givers():
    tenant = auth.create_tenant("Legacy Co")
    key = auth.create_key(tenant, "old")
    with auth._connect() as db:
        db.execute("UPDATE api_keys SET role = 'member' WHERE key_prefix = ?", (key[:12],))
    principal = auth.principal_for_key(key)
    assert principal.role == "giver" and principal.can("kt:create") and not principal.can("kt:delete_any")
