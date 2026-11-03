"""The user journey in a real browser: sign in, paste a transcript, read,
ask, edit, move a sentence, export, refresh, restart, sign out.

    .venv-tools/Scripts/python evaluation/browser_journey.py LABEL [--llm-down]

Runs with Playwright from the tools environment (.venv-tools: Playwright and
Chromium, kept out of the app's requirements). It starts the app itself, with
`.venv-1`, on a throwaway database with sign-in required and no LLM (with
--llm-down, an LLM that refuses every connection: a closed local port, so
nothing leaves the machine), and never touches data/ or the Groq quota.

Writes evaluation/results/browser-LABEL.json (one row per check: what was
expected, what happened, pass or fail) and, for a failed check, a screenshot
beside it.
"""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

from playwright.sync_api import sync_playwright

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PY = os.path.join(REPO, ".venv-1", "Scripts", "python.exe")
if not os.path.exists(APP_PY):
    APP_PY = os.path.join(REPO, ".venv-1", "bin", "python")
LABEL = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "journey"
LLM_DOWN = "--llm-down" in sys.argv
OUT = os.path.join(REPO, "evaluation", "results", f"browser-{LABEL}")
KT_TIMEOUT_MS = 15 * 60 * 1000


def _kt(name):
    with open(os.path.join(REPO, "tests", "e2e_kts", name), encoding="utf-8") as fh:
        return json.load(fh)


QUICKBITE = _kt("e_quickbite_food_delivery.json")
OTHERS = [_kt(n)["transcript"] for n in ("b_azure_payments.json", "c_gcp_data_platform.json",
                                         "f_online_learning.json")]

results = []
# With the LLM down every KT also waits out the local copy-edit fallback, so
# that run keeps to building, reading and exporting one KT.
LLM_DOWN_CASES = ("J01", "J04", "J05", "J10", "J19", "J20")
CURRENT = ["start"]                     # the check running, for the console messages it causes


def check(case, expected, fn, page=None):
    """Run one check; a failure is recorded with a screenshot, never raised."""
    if LLM_DOWN and not case.startswith(LLM_DOWN_CASES):
        return True
    CURRENT[0] = case.split()[0]
    started = time.time()
    try:
        actual = fn()
        ok = actual is True or (isinstance(actual, tuple) and actual[0] is True)
        detail = actual[1] if isinstance(actual, tuple) else ("as expected" if ok else repr(actual))
    except Exception as exc:                                    # noqa: BLE001 (recorded, not raised)
        ok, detail = False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
    shot = None
    if not ok and page is not None:
        shot = os.path.join(OUT, re.sub(r"\W+", "_", case)[:60] + ".png")
        try:
            page.screenshot(path=shot, full_page=True)
        except Exception:                                       # noqa: BLE001
            shot = None
    results.append({"case": case, "expected": expected, "actual": detail, "pass": ok,
                    "seconds": round(time.time() - started, 1), "screenshot": shot})
    print(f"{'PASS' if ok else 'FAIL'}  {case}: {detail}", flush=True)
    return ok


# --------------------------------------------------------------------------
# The app, on scratch data
# --------------------------------------------------------------------------

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class App:
    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="continuum-journey-")
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.env = dict(os.environ, CONTINUUM_DB_PATH=os.path.join(self.dir, "continuum.sqlite"),
                        KT_ASSETS_DIR=os.path.join(self.dir, "kt_assets"), CONTINUUM_AUTH="required",
                        CONTINUUM_WORKERS="1", HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8",
                        LLM_CACHE="off", LLM_CACHE_PATH=os.path.join(self.dir, "llm_cache.sqlite"),
                        GEMINI_API_KEY="", GROQ_BASE_URL="http://127.0.0.1:9/openai/v1")
        if LLM_DOWN:
            self.env.update(LLM_PROVIDER="groq", GROQ_API_KEY="not-a-real-key")
        else:
            self.env.update(LLM_PROVIDER="", GROQ_API_KEY="")
        self.proc = None
        self.log = open(os.path.join(OUT, "server.log"), "a", encoding="utf-8")

    def cli(self, *args):
        out = subprocess.run([APP_PY, "auth.py", *args], cwd=REPO, env=self.env, capture_output=True,
                             text=True, encoding="utf-8", timeout=120)
        if out.returncode:
            raise RuntimeError(out.stderr[-400:])
        return out.stdout

    def start(self):
        self.proc = subprocess.Popen([APP_PY, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port",
                                      str(self.port)], cwd=REPO, env=self.env, stdout=self.log, stderr=self.log)
        deadline = time.time() + 1200          # models load in 1-3 minutes on an idle laptop
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(self.base + "/readyz", timeout=5) as resp:
                    if resp.status == 200:
                        return
            except Exception:                                   # noqa: BLE001 (still starting)
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("the server exited; see server.log")
            time.sleep(2)
        raise RuntimeError("the server was not ready in 20 minutes")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def api(self, method, path, key, body=None, raw=False):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"X-API-Key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = resp.read()
                return resp.status, (payload if raw else json.loads(payload or b"{}"))
        except urllib.error.HTTPError as err:
            payload = err.read()
            try:
                return err.code, json.loads(payload)
            except ValueError:
                return err.code, payload.decode("utf-8", "replace")


def _wait_done(app, key, job_id, minutes=15):
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        status, job = app.api("GET", f"/status/{job_id}", key)
        if status == 200 and job.get("status") not in ("queued", "processing"):
            return job
        time.sleep(3)
    raise TimeoutError(f"{job_id} not finished in {minutes} min")


def _pdf_text(path):
    """Page count and text, read with pypdf from the app's environment."""
    code = ("import json,sys;from pypdf import PdfReader;r=PdfReader(sys.argv[1]);"
            "print(json.dumps({'pages':len(r.pages),'text':'\\n'.join(p.extract_text() or '' for p in r.pages)}))")
    out = subprocess.run([APP_PY, "-c", code, path], capture_output=True, text=True, encoding="utf-8", timeout=120)
    if out.returncode:
        raise RuntimeError(out.stderr[-300:])
    return json.loads(out.stdout)


# --------------------------------------------------------------------------
# The journey
# --------------------------------------------------------------------------

def sign_in(page, app, key):
    page.goto(app.base + "/")
    page.locator("#keyDetails summary").click()
    page.fill("#apiKey", key)
    page.click("#keyBtn")
    page.wait_for_selector("#newKtBtn", state="visible", timeout=30000)


def journey():
    os.makedirs(OUT, exist_ok=True)
    app = App()
    out = app.cli("create-tenant", "Journey Co")
    tenant = re.search(r"tenant: (\S+)", out).group(1)
    admin = re.search(r"api key[^:]*: (\S+)", out).group(1)
    reader = app.cli("create-key", tenant, "reader", "--role", "receiver").strip()
    try:
        app.start()
    except BaseException:
        app.stop()                      # never leave a server behind
        raise
    console, page_errors = [], []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(accept_downloads=True, viewport={"width": 1280, "height": 900})
        page = ctx.new_page()
        page.on("console", lambda m: console.append((m.type, f"[{CURRENT[0]}] {m.text}"))
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: page_errors.append(f"[{CURRENT[0]}] {e}"))
        job = {}

        # J01 sign-in
        def j01():
            page.goto(app.base + "/")
            if not page.locator("#keyDetails").count():
                return False, "no sign-in page for a signed-out visitor"
            page.locator("#keyDetails summary").click()
            page.fill("#apiKey", "ckt_wrong-key")
            page.click("#keyBtn")
            page.locator("#error", has_text=re.compile(r"\w")).wait_for(timeout=10000)
            wrong = page.locator("#error").inner_text().strip()
            page.fill("#apiKey", admin)
            page.click("#keyBtn")
            page.wait_for_selector("#newKtBtn", state="visible", timeout=30000)
            return True, f"wrong key refused ({wrong!r}); right key opens the app"
        check("J01 Sign in with an API key", "Signed-out visitors get the sign-in page; a wrong key is refused "
              "with a message; the right key opens the app", j01, page)

        # J02 empty paste
        def j02():
            page.click("#newKtBtn")
            page.click("#modeTranscript")
            page.fill("#transcriptInput", "   ")
            page.click("#upload")
            page.get_by_text("Paste a transcript first").wait_for(timeout=5000)
            status, jobs = app.api("GET", "/jobs", admin)
            return (len(jobs["jobs"]) == 0, f"message shown, {len(jobs['jobs'])} KT created")
        check("J02 Submit an empty transcript", "A message asks for a transcript; no KT is created", j02, page)

        # J03 not a KT
        def j03():
            page.fill("#transcriptInput", "Hi everyone. How was the weekend? Great, thanks. See you tomorrow.")
            page.click("#upload")
            toast = page.locator(".toast", has_text="Not a usable KT transcript").first
            toast.wait_for(timeout=30000)
            status, jobs = app.api("GET", "/jobs", admin)
            return (len(jobs["jobs"]) == 0, f"{toast.inner_text().strip()[:120]!r}; {len(jobs['jobs'])} KT created")
        check("J03 Submit small talk", "Refused with the reason (not enough operational content); no KT", j03, page)

        # J04 the real KT, with a double click
        def j04():
            if not page.locator("#transcriptInput").is_visible():          # J02 opens the form; --llm-down skips it
                page.click("#newKtBtn")
                page.click("#modeTranscript")
            page.fill("#transcriptInput", QUICKBITE["transcript"])
            page.dblclick("#upload")
            page.wait_for_url(re.compile(r"[?&]job="), timeout=60000)
            job["id"] = re.search(r"job=([0-9a-f-]{36})", page.url).group(1)
            page.wait_for_selector("#downloadPdfBtn:not([disabled])", timeout=KT_TIMEOUT_MS)
            status, jobs = app.api("GET", "/jobs", admin)
            data = _wait_done(app, admin, job["id"])
            job["data"] = data
            return (len(jobs["jobs"]) == 1 and data["status"].startswith("completed"),
                    f"{data['status']}; {len(jobs['jobs'])} KT for a double click")
        if not check("J04 Paste a 284-word KT and double-click Build", "One KT (not two) is built and opens "
                     "when done", j04, page):
            return finish(app, browser, console, page_errors)
        jid = job["id"]

        def j05():
            doc = page.locator("#renderedDocument")
            text = doc.inner_text()
            titles = [" ".join(t.split()) for t in page.locator("#renderedDocument section.doc-section > h3").all_inner_texts()]
            first = titles[0] if titles else ""
            return ("QuickBite" in text and len(titles) >= 10 and "quick reference" in first.lower(),
                    f"{len(titles)} sections, first {first!r}, names QuickBite: {'QuickBite' in text}")
        check("J05 Read the document", "The document names QuickBite, has every section, the incident card first",
              j05, page)

        def j06():
            page.click("#tab-gaps")
            panel = page.locator("#panel-gaps")
            panel.wait_for(state="visible")
            text = panel.inner_text()
            missing = {"Cost": "cost" in text.lower(), "Environments": "environments" in text.lower(),
                       "Security": "security" in text.lower()}
            follow = page.locator("#followUps li").count()
            return (all(missing.values()) and follow > 0, f"follow-up items {follow}; undiscussed named {missing}")
        check("J06 Gaps & coverage tab", "Shows coverage per section and a follow-up list naming what was not "
              "discussed (cost, environments, security in this KT)", j06, page)

        def j07():
            page.click("#tab-doc")
            page.fill("#docAskInput", "Who do I call when dispatch fails?")
            page.press("#docAskInput", "Enter")
            result = page.locator("#docAskResult")
            result.wait_for(state="visible", timeout=15000)
            answers = page.locator("#docAskResult .ask-answers li").count()
            page.fill("#docAskInput", "What is our Snowflake warehouse size?")
            page.press("#docAskInput", "Enter")
            page.wait_for_timeout(1500)
            note = page.locator("#docAskResult").inner_text()
            return (answers > 0 and ("never mentions" in note.lower() or "does not say" in note.lower()),
                    f"{answers} answer(s) to a covered question; unknown topic: {note.strip()[:100]!r}")
        check("J07 Ask the KT", "A covered question gets statements with sources; a topic never discussed is "
              "said to be missing, not answered", j07, page)

        def j08():
            page.click("#docEditBtn")
            # A table row opens one input per column; a line of text, a textarea.
            buttons = page.locator("#renderedDocument [data-act='edit']")
            for i in range(buttons.count()):
                buttons.nth(i).click()
                if page.locator(".unit-editor textarea").count():
                    break
                page.locator(".unit-editor [data-ed='cancel']").first.click()
            editor = page.locator(".unit-editor textarea").first
            editor.fill("EDITED BY THE JOURNEY: orders are checked every morning.")
            page.locator(".unit-editor [data-ed='ok']").first.click()
            page.click("#docSaveBtn")
            # A locator, not wait_for_function: the page's CSP forbids eval, as it should.
            page.locator("#docSaveBtn").wait_for(state="hidden", timeout=30000)
            status, versions = app.api("GET", f"/documents/{jid}/versions", admin)
            n = len(versions.get("versions", []))
            job["edited"] = n == 2
            return n == 2, f"{n} versions after one save"
        check("J08 Edit a line and save", "The edit is saved as version 2; version 1 is kept", j08, page)

        def j09():
            page.click("#tab-transcript")
            rows = page.locator("#assignments .sent-row")
            rows.first.wait_for(timeout=10000)
            tags = [r.locator(".sec-tag").first.inner_text() for r in rows.all()]
            # A sentence after the first section's: the case the old index matching got wrong.
            index = next(i for i, t in enumerate(tags) if t != tags[0])
            text = rows.nth(index).locator(".sent-text").inner_text().split("\n")[0].strip()
            rows.nth(index).get_by_role("button", name="Move").click()
            options = page.locator("#sectionSelect option").all()
            target = next(o.get_attribute("value") for o in options
                          if o.get_attribute("value") not in ("", None) and o.inner_text() != tags[index])
            page.select_option("#sectionSelect", target)
            page.click("#editConfirm")
            page.wait_for_timeout(2500)
            _, now = app.api("GET", f"/status/{jid}", admin)
            moved = [s["text"] for s in now["coverage"].get(target, {}).get("sentences", [])]
            elsewhere = sum(1 for sec, info in now["coverage"].items() if sec != target
                            for s in info.get("sentences", []) if s["text"] == text)
            return (text in moved and elsewhere == 0,
                    f"sentence {index + 1} ({text[:50]!r}) to {target}: there {text in moved}, left behind {elsewhere}")
        check("J09 Move a sentence from the Transcript tab", "The sentence picked (not one from another section) "
              "moves, and is listed once", j09, page)

        def j10():
            page.click("#tab-doc")
            with page.expect_download(timeout=180000) as dl:
                page.click("#downloadPdfBtn")
            path = os.path.join(OUT, "kt.pdf")
            dl.value.save_as(path)
            pdf = _pdf_text(path)
            flat = " ".join(pdf["text"].split())
            carried = "EDITED BY THE JOURNEY" in flat
            return (pdf["pages"] >= 3 and "QuickBite" in flat and carried == bool(job.get("edited")),
                    f"{pdf['pages']} pages, names QuickBite, carries the edit: {carried} (edit saved: "
                    f"{bool(job.get('edited'))})")
        check("J10 Download the PDF", "A PDF that opens, names the system and carries the saved edit", j10, page)

        def j11():
            got = {}
            for button, ext in (("#downloadMdBtn", ".md"), ("#downloadGapsBtn", ".csv"), ("#downloadRunbookBtn", ".md")):
                page.click("#exportMenuBtn")
                with page.expect_download(timeout=60000) as dl:
                    page.click(button)
                path = os.path.join(OUT, "export" + button.strip("#") + ext)
                dl.value.save_as(path)
                got[button] = os.path.getsize(path)
            text = open(os.path.join(OUT, "exportdownloadMdBtn.md"), encoding="utf-8").read()
            return (all(v > 100 for v in got.values()) and "QuickBite" in text, f"bytes {got}")
        check("J11 Export Markdown, gaps CSV and runbooks", "Each downloads with content", j11, page)

        def j12():
            page.reload()
            page.wait_for_selector("#downloadPdfBtn:not([disabled])", timeout=60000)
            same = "QuickBite" in page.locator("#renderedDocument").inner_text()
            page.click("#crumbHome")
            page.wait_for_selector("#newKtBtn", state="visible")
            page.go_back()
            page.wait_for_selector("#downloadPdfBtn:not([disabled])", timeout=60000)
            back = jid in page.url
            return (same and back, f"after reload the KT shows: {same}; back button returns to it: {back}")
        check("J12 Refresh, home, browser back", "The KT link survives a refresh and the back button", j12, page)

        def j13():
            app.stop()
            app.start()
            page.reload()
            page.wait_for_selector("#downloadPdfBtn:not([disabled])", timeout=120000)
            status, pdf = app.api("GET", f"/export/pdf/{jid}", admin, raw=True)
            return (status == 200 and pdf[:4] == b"%PDF", f"after a restart: signed in, PDF {status}")
        check("J13 Restart the server", "Sign-in, the KT, its versions and the PDF survive a restart", j13, page)

        def j14():
            other = browser.new_context(viewport={"width": 1280, "height": 900})
            rp = other.new_page()
            try:
                sign_in(rp, app, reader)
                rp.goto(f"{app.base}/?job={jid}")
                rp.wait_for_selector("#downloadPdfBtn:not([disabled])", timeout=60000)
                can_edit = rp.locator("#docEditBtn").is_visible()
                status, _ = app.api("POST", f"/documents/{jid}/edits", reader, {"base_version": 2, "edits": []})
                create, _ = app.api("POST", "/kt-from-transcript", reader, {"transcript": QUICKBITE["transcript"]})
                return (not can_edit and status == 403 and create == 403,
                        f"Edit button shown: {can_edit}; edit API {status}; create API {create}")
            finally:
                other.close()
        check("J14 A receiver opens the KT", "Can read and download; cannot edit or create (no button, API 403)",
              j14, page)

        def j15():
            status, body = app.api("POST", "/kt-from-transcript", admin, {"transcript": "word " * 40000})
            return (status == 413 and "limit" in json.dumps(body), f"{status}: {str(body)[:120]}")
        check("J15 Paste a transcript over the limit", "Refused (413) with the limit and what to do", j15)

        def j16():
            started = time.time()
            ids, errors = [], []

            def post(text):
                status, body = app.api("POST", "/kt-from-transcript", admin, {"transcript": text})
                (ids if status == 200 else errors).append(body.get("job_id") if status == 200 else status)
            threads = [threading.Thread(target=post, args=(t,)) for t in OTHERS]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            done = [_wait_done(app, admin, i, minutes=30)["status"] for i in ids]
            return (len(ids) == 3 and all(s.startswith("completed") for s in done),
                    f"3 sent at once: {done}, errors {errors}, {time.time() - started:.0f} s")
        check("J16 Three KTs at once", "All are queued and all finish", j16)

        def j17():
            phone = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True)
            pp = phone.new_page()
            try:
                sign_in(pp, app, admin)
                widths = {}
                for name, url in (("start", app.base + "/"), ("kt", f"{app.base}/?job={jid}")):
                    pp.goto(url)
                    pp.wait_for_timeout(2500)
                    widths[name] = pp.evaluate("[document.documentElement.scrollWidth, document.documentElement.clientWidth]")
                ok = all(sw <= cw + 1 for sw, cw in widths.values())
                if not ok:
                    pp.screenshot(path=os.path.join(OUT, "J17_phone.png"), full_page=True)
                return ok, f"scroll/client widths {widths}"
            finally:
                phone.close()
        check("J17 Phone width (390 px)", "No sideways scrolling on the start page or a KT", j17)

        def j18():
            page.click("#signOutBtn")
            page.wait_for_selector("#keyDetails", timeout=15000)
            page.go_back()
            page.wait_for_timeout(1500)
            leaked = page.locator("#renderedDocument").count() and page.locator("#renderedDocument").is_visible()
            status, _ = app.api("GET", f"/status/{jid}", "")
            return (not leaked and status == 401, f"app shown after back: {bool(leaked)}; API without a key {status}")
        check("J18 Sign out", "Back to the sign-in page; the back button does not show the KT", j18, page)

        def j19():
            csp = [t for kind, t in console if "Content Security Policy" in t or "Refused to" in t]
            # J13 stops the server on purpose: a request in flight then fails.
            restart = [t for kind, t in console if t.startswith("[J13]") and "Failed to fetch" in t]
            js = [t for kind, t in console if kind == "error" and "Failed to load resource" not in t and t not in restart]
            return (not csp and not js and not page_errors,
                    f"CSP violations {len(csp)}, console errors {len(js)}, page errors {len(page_errors)}, "
                    f"network errors while the server was stopped {len(restart)}"
                    + (f": {(csp + js + page_errors)[0][:160]}" if csp or js or page_errors else ""))
        check("J19 No script errors or blocked scripts", "No console error, uncaught exception or CSP violation "
              "during the journey", j19)
        if LLM_DOWN:
            def j20():
                usage = job["data"].get("llm_usage") or {}
                missed = (usage.get("failed") or 0) + (usage.get("skipped") or 0)
                return (job["data"]["status"].startswith("completed") and missed > 0,
                        f"{job['data']['status']}; failed {usage.get('failed')}, skipped {usage.get('skipped')}, "
                        f"calls {usage.get('llm_calls')}")
            check("J20 LLM unreachable", "The KT still finishes with the rules-only result, and the LLM failures "
                  "are counted", j20)
        return finish(app, browser, console, page_errors)


def finish(app, browser, console, page_errors):
    try:
        browser.close()
    finally:
        app.stop()
        app.log.close()
    report = {"label": LABEL, "llm_down": LLM_DOWN, "passed": sum(r["pass"] for r in results),
              "checks": len(results), "results": results}
    with open(os.path.join(REPO, "evaluation", "results", f"browser-{LABEL}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n{report['passed']}/{report['checks']} checks passed; results in evaluation/results/browser-{LABEL}.json")
    shutil.rmtree(app.dir, ignore_errors=True)
    return report


if __name__ == "__main__":
    journey()
