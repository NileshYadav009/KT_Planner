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
