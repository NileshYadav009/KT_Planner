"""Publish a KT to Confluence and its knowledge gaps to Jira (P2-2).

The Markdown and CSV exports need someone to paste or import them. A
workspace can instead be connected to its Atlassian site once, by an operator:

    python -m auth confluence-set <tenant_id> --url https://acme.atlassian.net/wiki --space OPS \\
        --token-env CONFLUENCE_TOKEN [--email bot@acme.com] [--parent-id 123456]
    python -m auth jira-set <tenant_id> --url https://acme.atlassian.net --project OPS \\
        --token-env JIRA_TOKEN [--email bot@acme.com] [--issue-type Task]
    python -m auth integrations <tenant_id>
    python -m auth integration-remove <tenant_id> confluence|jira

A giver or reviewer then publishes from the app. The document becomes one
Confluence page; publishing again gives the same page a new version. Each
knowledge gap becomes one Jira issue, and a gap already sent is not sent
again.

The token stays in the named environment variable: the database keeps only
the variable's name. With --email the token is an Atlassian Cloud API token
(basic auth); without it, a Data Center personal access token (bearer). Sites
on atlassian.net use Confluence's v2 page API and Jira's v3 API; others the
Confluence v1 content API and Jira v2, which Data Center serves.
"""
import hashlib
import html
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import auth

logger = logging.getLogger(__name__)

KINDS = ("confluence", "jira")
TIMEOUT_SECONDS = 20
JIRA_BULK_LIMIT = 50
# An environment variable's name, in the usual upper case. Atlassian tokens
# have lower-case letters, so pasting the token itself is refused.
_ENV_NAME_RE = re.compile(r"[A-Z_][A-Z0-9_]{0,63}")
_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")


class PublishError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------
# Configuration (auth database, table integrations)
# --------------------------------------------------------------------------

def _check_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(url)
    local = parsed.hostname in ("localhost", "127.0.0.1")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
        raise ValueError("The URL must start with https:// (the token is sent with every request)")
    if not parsed.hostname:
        raise ValueError("The URL has no host name")
    return url


def set_integration(tenant_id: str, kind: str, url: str, token_env: str, email: Optional[str] = None,
                    space: Optional[str] = None, parent_id: Optional[str] = None, project: Optional[str] = None,
                    issue_type: str = "Task", cloud: Optional[bool] = None) -> Dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"Unknown integration '{kind}'. Use one of: {', '.join(KINDS)}")
    url = _check_url(url)
    if not token_env or not _ENV_NAME_RE.fullmatch(token_env):
        raise ValueError("--token-env takes the NAME of an environment variable, not the token itself")
    config: Dict[str, Any] = {"url": url, "token_env": token_env, "email": (email or "").strip() or None,
                              "cloud": urllib.parse.urlparse(url).hostname.endswith(".atlassian.net")
                              if cloud is None else bool(cloud)}
    if kind == "confluence":
        if not space or not _KEY_RE.fullmatch(space):
            raise ValueError("--space takes the Confluence space key (for example OPS)")
        if parent_id and not str(parent_id).isdigit():
            raise ValueError("--parent-id takes the numeric id of the parent page")
        config.update(space=space, parent_id=str(parent_id) if parent_id else None)
    else:
        if not project or not _KEY_RE.fullmatch(project):
            raise ValueError("--project takes the Jira project key (for example OPS)")
        config.update(project=project.upper(), issue_type=(issue_type or "Task").strip())
    with auth._lock, auth._connect() as db:
        auth._require_tenant(db, tenant_id)
        db.execute("INSERT INTO integrations (tenant_id, kind, config, created_at) VALUES (?, ?, ?, ?) "
                   "ON CONFLICT(tenant_id, kind) DO UPDATE SET config = excluded.config",
                   (tenant_id, kind, json.dumps(config), time.time()))
    auth.audit(tenant_id, "cli", "integration_configured", {"kind": kind, "url": url})
    return config


def get_integration(tenant_id: str, kind: str) -> Optional[Dict[str, Any]]:
    with auth._connect() as db:
        row = db.execute("SELECT config FROM integrations WHERE tenant_id = ? AND kind = ?",
                         (tenant_id, kind)).fetchone()
    return json.loads(row[0]) if row else None


def remove_integration(tenant_id: str, kind: str) -> bool:
    with auth._lock, auth._connect() as db:
        removed = db.execute("DELETE FROM integrations WHERE tenant_id = ? AND kind = ?",
                             (tenant_id, kind)).rowcount > 0
    if removed:
        auth.audit(tenant_id, "cli", "integration_removed", {"kind": kind})
    return removed


def configured(tenant_id: str) -> Dict[str, bool]:
    """Which integrations this workspace has (for the app's buttons)."""
    with auth._connect() as db:
        kinds = {r[0] for r in db.execute("SELECT kind FROM integrations WHERE tenant_id = ?", (tenant_id,))}
    return {kind: kind in kinds for kind in KINDS}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _auth_header(config: Dict[str, Any]) -> str:
    token = os.getenv(config["token_env"] or "")
    if not token:
        raise PublishError(f"The {config['token_env']} environment variable (the Atlassian token) is not set on "
                           "the server. Ask your Continuum operator to set it.", 503)
    if config.get("email"):
        import base64
        return "Basic " + base64.b64encode(f"{config['email']}:{token}".encode("utf-8")).decode("ascii")
    return "Bearer " + token


def _error_text(body: bytes) -> str:
    try:
        data = json.loads(body.decode("utf-8") or "{}")
    except ValueError:
        return body.decode("utf-8", "replace")[:200]
    messages = list(data.get("errorMessages") or [])
    messages += [f"{k}: {v}" for k, v in (data.get("errors") or {}).items()] if isinstance(data.get("errors"), dict) else []
    messages += [data["message"]] if isinstance(data.get("message"), str) else []
    return "; ".join(str(m) for m in messages)[:300] or "no details"


def _call(method: str, url: str, config: Dict[str, Any], body: Optional[dict] = None) -> Tuple[int, dict]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": _auth_header(config), "Accept": "application/json", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:  # noqa: S310 (operator-configured URL)
            raw = resp.read()
            return resp.status, (json.loads(raw.decode("utf-8")) if raw else {})
    except urllib.error.HTTPError as exc:
        status = exc.code
        detail = _error_text(exc.read() or b"")
        if status == 404:
            return 404, {}
        hint = {401: "the token was not accepted", 403: "the token's account may not do this"}.get(status, "")
        raise PublishError(f"Atlassian answered {status}{' (' + hint + ')' if hint else ''}: {detail}",
                           502 if status >= 500 else 400)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PublishError(f"Could not reach {urllib.parse.urlparse(url).hostname}: {exc}", 502)


# --------------------------------------------------------------------------
# Confluence
# --------------------------------------------------------------------------

def _t(value: Any) -> str:
    return html.escape(re.sub(r"\s+", " ", str(value if value is not None else "")).strip(), quote=False)


def _refs(block: Dict[str, Any], i: int) -> str:
    refs = block.get("sources") or []
    ids = refs[i] if i < len(refs) and isinstance(refs[i], list) else []
    return (" " + "".join(f"[{int(n)}]" for n in ids)) if ids else ""


def _table(headers: List[str], rows: List[List[str]]) -> str:
    head = "".join(f"<th>{_t(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tbody><tr>{head}</tr>{body}</tbody></table>"


def _block_html(block: Dict[str, Any], section_title: str) -> str:
    """One rendered block in Confluence storage format. Every text is
    escaped: transcript words never become markup."""
    kind = block.get("type")
    out: List[str] = []
    title = _t(block.get("title"))
    if title and title.lower() != _t(section_title).lower() and kind != "WarningBlock":
        out.append(f"<h3>{title}</h3>")
    if kind == "NarrativeBlock":
        out += [f"<p>{_t(p)}{_refs(block, i)}</p>" for i, p in enumerate(block.get("paragraphs") or []) if _t(p)]
    elif kind == "ChecklistBlock":
        items = [f"<li>{_t(x)}{_refs(block, i)}</li>" for i, x in enumerate(block.get("items") or []) if _t(x)]
        out.append(f"<ul>{''.join(items)}</ul>" if items else "")
    elif kind == "TroubleshootingBlock":
        steps = [f"<li>{_t(x)}{_refs(block, i)}</li>" for i, x in enumerate(block.get("steps") or []) if _t(x)]
        out.append(f"<ol>{''.join(steps)}</ol>" if steps else "")
    elif kind == "WarningBlock":
        out += [f"<p><strong>Warning:</strong> {_t(w)}{_refs(block, i)}</p>"
                for i, w in enumerate(block.get("warnings") or []) if _t(w)]
    elif kind == "TechnologyGrid":
        out.append(_table(["Item", "Value"], [[_t(r.get("label")), _t(r.get("value")) + _refs(block, i)]
                                              for i, r in enumerate(block.get("rows") or [])]))
    elif kind == "OwnershipTable":
        out.append(_table(["Responsibility", "Owner"], [[_t(r.get("role")), _t(r.get("team")) + _refs(block, i)]
                                                        for i, r in enumerate(block.get("rows") or [])]))
    elif kind == "DeploymentTimeline":
        entries = [f"<li><strong>{_t(e.get('label'))}</strong>: {_t(e.get('description'))}{_refs(block, i)}</li>"
                   for i, e in enumerate(block.get("entries") or [])]
        out.append(f"<ol>{''.join(entries)}</ol>" if entries else "")
    elif kind == "DecisionTable":
        columns = list(block.get("columns") or [])
        rows = []
        for i, row in enumerate(block.get("rows") or []):
            cells = [_t(row.get(c)) or "Not covered" for c in columns]
            if cells:
                cells[-1] += _refs(block, i)
            rows.append(cells)
        if columns:
            out.append(_table(columns, rows))
    elif kind == "CodeBlock":
        out.append(f"<pre>{html.escape(str(block.get('code') or '').rstrip(), quote=False)}</pre>")
    elif kind == "DiagramBlock":
        out.append("<p><em>Architecture diagram: see the PDF export.</em></p>")
    elif kind == "ImageBlock":
        out.append(f"<p><em>Screenshot: {_t(block.get('caption')) or 'see the PDF export'}.</em></p>")
    elif _t(block.get("description")):
        out.append(f"<p>{_t(block.get('description'))}</p>")
    return "".join(out)


def page_title(job_id: str, job: Dict[str, Any]) -> str:
    """Unique within a space (Confluence refuses a second page with the same
    title), so the job id is part of it."""
    ko = job.get("knowledge_object") or {}
    system = re.sub(r"\s+", " ", str(ko.get("system_name") or job.get("title") or "KT")).strip()[:200]
    return f"{system} - KT handover ({job_id[:8].upper()})"


def storage_body(job_id: str, job: Dict[str, Any]) -> str:
    """The document in Confluence storage format: the same sections, sign-off
    and numbered sources as the PDF and the Markdown export."""
    import signoff
    from knowledge.evidence import format_time

    ko = job.get("knowledge_object") or {}
    sections = signoff.apply_to_document(ko.get("rendered_sections") or [], job, int(job.get("document_version") or 1))
    out = [f"<p><em>Knowledge transfer and handover, version {int(job.get('document_version') or 1)}, "
           f"published from Continuum (job {_t(job_id[:8].upper())}). Edits belong in Continuum: publishing "
           "again replaces this page.</em></p>"]
    out += [f"<p><strong>Review before relying on this document:</strong> {_t(w)}</p>" for w in job.get("warnings") or []]
    for number, section in enumerate(sections, start=1):
        title = section.get("section_title") or section.get("section_id") or f"Section {number}"
        out.append(f"<h2>{number}. {_t(title)}</h2>")
        out += [_block_html(b, title) for b in section.get("blocks") or []]
    sources = ko.get("sources") or []
    if sources:
        items = []
        for src in sources:
            when = format_time(src.get("start"))
            if src.get("session"):
                when = f"S{int(src['session'])}" + (f" · {when}" if when else "")
            who = _t(src.get("speaker"))
            lead = " ".join(p for p in (_t(when), who + ":" if who else "") if p)
            items.append(f"<li>{lead + ' ' if lead else ''}&ldquo;{_t(src.get('quote'))}&rdquo;</li>")
        out.append("<h2>Sources</h2><p>Each fact above is followed by the number of the transcript sentence it "
                   f"comes from.</p><ol>{''.join(items)}</ol>")
    return "".join(out)


def _page_url(config: Dict[str, Any], data: dict) -> str:
    links = data.get("_links") or {}
    return (links.get("base") or config["url"]).rstrip("/") + (links.get("webui") or "")


def publish_confluence(job_id: str, job: Dict[str, Any], config: Dict[str, Any],
                       previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create the KT's page, or give the page published before a new version.
    Returns {"page_id", "url", "version", "created"}."""
    base, title, body = config["url"], page_title(job_id, job), storage_body(job_id, job)
    same_place = previous and previous.get("url_base") == base and previous.get("space") == config["space"]
    if config.get("cloud"):
        if same_place:
            status, page = _call("GET", f"{base}/api/v2/pages/{previous['page_id']}", config)
            if status != 404:
                number = int((page.get("version") or {}).get("number") or 1) + 1
                _, page = _call("PUT", f"{base}/api/v2/pages/{previous['page_id']}", config, {
                    "id": previous["page_id"], "status": "current", "title": title,
                    "body": {"representation": "storage", "value": body},
                    "version": {"number": number, "message": "Published again from Continuum"}})
                return {"page_id": str(page.get("id") or previous["page_id"]), "url": _page_url(config, page),
                        "version": number, "created": False}
        status, spaces = _call("GET", f"{base}/api/v2/spaces?keys={urllib.parse.quote(config['space'])}", config)
        results = spaces.get("results") or []
        if status == 404 or not results:
            raise PublishError(f"Confluence has no space with the key {config['space']} (or the token cannot see it)", 400)
        request = {"spaceId": str(results[0]["id"]), "status": "current", "title": title,
                   "body": {"representation": "storage", "value": body}}
        if config.get("parent_id"):
            request["parentId"] = config["parent_id"]
        _, page = _call("POST", f"{base}/api/v2/pages", config, request)
    else:
        if same_place:
            status, page = _call("GET", f"{base}/rest/api/content/{previous['page_id']}?expand=version", config)
            if status != 404:
                number = int((page.get("version") or {}).get("number") or 1) + 1
                _, page = _call("PUT", f"{base}/rest/api/content/{previous['page_id']}", config, {
                    "id": previous["page_id"], "type": "page", "title": title,
                    "version": {"number": number, "message": "Published again from Continuum"},
                    "body": {"storage": {"value": body, "representation": "storage"}}})
                return {"page_id": str(page.get("id") or previous["page_id"]), "url": _page_url(config, page),
                        "version": number, "created": False}
        request = {"type": "page", "title": title, "space": {"key": config["space"]},
                   "body": {"storage": {"value": body, "representation": "storage"}}}
        if config.get("parent_id"):
            request["ancestors"] = [{"id": config["parent_id"]}]
        _, page = _call("POST", f"{base}/rest/api/content", config, request)
    if not page.get("id"):
        raise PublishError("Confluence did not return the new page", 502)
    return {"page_id": str(page["id"]), "url": _page_url(config, page),
            "version": int((page.get("version") or {}).get("number") or 1), "created": True}


# --------------------------------------------------------------------------
# Jira
# --------------------------------------------------------------------------

def gap_key(gap: str) -> str:
    return hashlib.sha1(re.sub(r"[^a-z0-9]+", " ", gap.lower()).strip().encode("utf-8")).hexdigest()[:16]


def _description(text: str, cloud: bool):
    if not cloud:
        return text
    return {"type": "doc", "version": 1, "content": [
        {"type": "paragraph", "content": [{"type": "text", "text": para}]}
        for para in text.split("\n\n") if para.strip()]}


def create_jira_issues(job_id: str, job: Dict[str, Any], config: Dict[str, Any],
                       sent: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """One issue per knowledge gap not sent before (`sent`: gap key -> issue
    key). Returns {"created": [...], "already": [...], "failed": [...]}."""
    from markdown_export import knowledge_gaps

    sent = dict(sent or {})
    ko = job.get("knowledge_object") or {}
    system = re.sub(r"\s+", " ", str(ko.get("system_name") or job.get("title") or "KT")).strip()
    base, api = config["url"], "3" if config.get("cloud") else "2"
    gaps = knowledge_gaps(job)
    already = [{"gap": g, "key": sent[gap_key(g)], "url": f"{base}/browse/{sent[gap_key(g)]}"}
               for g in gaps if gap_key(g) in sent]
    todo = [g for g in gaps if gap_key(g) not in sent]
    created, failed = [], []
    for start in range(0, len(todo), JIRA_BULK_LIMIT):
        chunk = todo[start:start + JIRA_BULK_LIMIT]
        updates = []
        for gap in chunk:
            summary = f"KT gap ({system}): {gap}"
            if len(summary) > 250:
                summary = summary[:249].rsplit(" ", 1)[0] + "…"
            text = (f"The knowledge transfer for {system} (Continuum job {job_id[:8].upper()}) did not cover "
                    f"this: {gap}\n\nClose it with the outgoing owner or in a follow-up KT session.")
            updates.append({"fields": {"project": {"key": config["project"]}, "summary": summary,
                                       "issuetype": {"name": config.get("issue_type") or "Task"},
                                       "labels": ["kt-gap"], "description": _description(text, bool(config.get("cloud")))}})
        status, data = _call("POST", f"{base}/rest/api/{api}/issue/bulk", config, {"issueUpdates": updates})
        if status == 404:
            raise PublishError("Jira's issue API was not found at this URL; check the site URL", 400)
        # Jira lists the issues it created in order and the ones it refused
        # by their position in the request.
        refused = {int(e.get("failedElementNumber", -1)): _error_text(json.dumps(e.get("elementErrors") or {}).encode())
                   for e in data.get("errors") or []}
        issues = iter(data.get("issues") or [])
        for i, gap in enumerate(chunk):
            if i in refused:
                failed.append({"gap": gap, "reason": refused[i]})
                continue
            issue = next(issues, None)
            if not issue or not issue.get("key"):
                failed.append({"gap": gap, "reason": "Jira did not return the issue"})
                continue
            sent[gap_key(gap)] = issue["key"]
            created.append({"gap": gap, "key": issue["key"], "url": f"{base}/browse/{issue['key']}"})
    return {"created": created, "already": already, "failed": failed, "sent": sent}


# --------------------------------------------------------------------------
# CLI (python -m auth ...)
# --------------------------------------------------------------------------

def cli(cmd: str, args: list) -> int:
    tenant_id = args.pop(0)
    if cmd == "integrations":
        for kind in KINDS:
            config = get_integration(tenant_id, kind)
            print(f"{kind}: " + (json.dumps(config) if config else "not configured"))
        return 0
    if cmd == "integration-remove":
        kind = args[0] if args else ""
        print("removed" if remove_integration(tenant_id, kind) else f"{kind or 'that integration'} was not configured")
        return 0
    kind = "confluence" if cmd == "confluence-set" else "jira"
    url, token_env = auth._option(args, "--url"), auth._option(args, "--token-env")
    try:
        config = set_integration(
            tenant_id, kind, url, token_env, email=auth._option(args, "--email"),
            space=auth._option(args, "--space"), parent_id=auth._option(args, "--parent-id"),
            project=auth._option(args, "--project"), issue_type=auth._option(args, "--issue-type", "Task"))
    except ValueError as exc:
        print(f"error: {exc}")
        print("usage: python -m auth confluence-set <tenant_id> --url URL --space KEY --token-env VAR "
              "[--email ADDRESS] [--parent-id ID]\n"
              "       python -m auth jira-set <tenant_id> --url URL --project KEY --token-env VAR "
              "[--email ADDRESS] [--issue-type Task]")
        return 2
    print(json.dumps(config, indent=2))
    if not os.getenv(config["token_env"]):
        print(f"\nSet {config['token_env']} in the environment of the Continuum API before publishing.")
    return 0
