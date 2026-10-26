"""Test isolation: the job database and the vocabulary-candidates file are
real state files. Point both at a throwaway directory before any test module
imports pipeline (which opens the job store at import time); the audit found
the golden tests rewriting the repo's own glossary_candidates.json."""
import os
import tempfile

_STATE_DIR = tempfile.mkdtemp(prefix="continuum-tests-")
os.environ.setdefault("CONTINUUM_DB_PATH", os.path.join(_STATE_DIR, "continuum.sqlite"))
os.environ.setdefault("CONTINUUM_VOCAB_CANDIDATES_PATH", os.path.join(_STATE_DIR, "glossary_candidates.json"))
# Route tests that are not about authentication run as the local tenant;
# tests/test_auth.py switches authentication on and checks every boundary.
os.environ.setdefault("CONTINUUM_AUTH", "disabled")
# The copy edit rewrites sentences with a local model when one is downloaded
# (copyedit.py): tests score the pipeline's own output, so it is off except
# in tests/test_copyedit.py.
os.environ.setdefault("CONTINUUM_COPYEDIT", "0")
# The punctuation model (punctuation.py) is downloaded on first use; tests
# give it a stand-in (tests/test_punctuation.py) and run without it.
os.environ.setdefault("PUNCTUATION_MODEL", "0")
# No live LLM in tests: llm_provider loads .env at import, and a key there
# would send real (quota-spending) requests from an ordinary pytest run. An
# empty value counts as set, so python-dotenv leaves it alone. Export a key
# explicitly to run against a live provider on purpose.
for _key in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
    os.environ.setdefault(_key, "")
