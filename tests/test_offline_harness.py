"""The no-LLM evaluation runners (tests/golden_eval.py, tests/e2e_eval.py)
must send nothing even when a key is configured. They switched off the
provider lookups in pipeline and ai, but the copy edit (copyedit.py) asks
llm_provider directly, so with a key in .env a "no-LLM" run would still have
sent copy-edit requests (16 Oct). Likewise an ordinary pytest run picked the
key up from .env; tests/conftest.py now keeps it out."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import context_mapper  # noqa: E402
import e2e_eval  # noqa: E402
import golden_eval  # noqa: E402
import llm_provider  # noqa: E402


class _LiveProvider:
    def generate(self, *args, **kwargs):
        raise AssertionError("a no-LLM run sent a request")


def _lookups_seen_inside(monkeypatch, runner_module, step_name, run):
    live = _LiveProvider()
    monkeypatch.setattr(llm_provider, "get_llm_provider", lambda: live)
    monkeypatch.setattr(context_mapper, "ContextMappingPipeline", lambda *a, **k: None)
    seen = []

    def step(*args, **kwargs):
        from llm_provider import get_llm_provider  # as copyedit.py looks it up
        seen.append(get_llm_provider())
        return ({}, b"") if step_name == "run_dataset" else {}

    monkeypatch.setattr(runner_module, step_name, step)
    run()
    return seen, llm_provider.get_llm_provider() is live


def test_e2e_no_llm_runner_switches_off_the_direct_provider_lookup(monkeypatch):
    monkeypatch.setattr(e2e_eval, "evaluate", lambda *a, **k: {})
    seen, restored = _lookups_seen_inside(monkeypatch, e2e_eval, "run_dataset",
                                          lambda: e2e_eval.run_all(datasets=[{"id": "x"}]))
    assert seen == [None]
    assert restored


def test_golden_no_llm_runner_switches_off_the_direct_provider_lookup(monkeypatch):
    seen, restored = _lookups_seen_inside(monkeypatch, golden_eval, "run_golden",
                                          lambda: golden_eval.run_all(goldens=[{"id": "x"}]))
    assert seen == [None]
    assert restored


_KEYS = ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY")


def _key_after_dotenv(tmp_path, run_conftest):
    """GROQ_API_KEY as llm_provider would see it, in a fresh process whose
    .env holds a (fake) key and no key was exported."""
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("GROQ_API_KEY=gsk_" + "x" * 52 + "\n", encoding="utf-8")
    conftest = os.path.join(os.path.dirname(__file__), "conftest.py")
    code = ("import os, runpy\n"
            + (f"runpy.run_path({conftest!r})\n" if run_conftest else "")
            + "from dotenv import load_dotenv\n"
            + f"load_dotenv({str(dotenv_file)!r})\n"
            + "print(len(os.environ.get('GROQ_API_KEY') or ''))\n")
    env = {k: v for k, v in os.environ.items() if k not in _KEYS}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    return int(out.stdout.strip())


def test_conftest_keeps_a_dotenv_key_out_of_tests(tmp_path):
    # Without conftest.py's empty defaults, python-dotenv loads the key.
    assert _key_after_dotenv(tmp_path, run_conftest=False) == 56
    assert _key_after_dotenv(tmp_path, run_conftest=True) == 0
