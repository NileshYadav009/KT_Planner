"""Single source of truth for the KT schema definition.

Previously `main.py` and `ai.py` each loaded `kt_schema_new.json` independently,
producing two separate SCHEMA globals that could diverge. Both now import SCHEMA
from here instead.
"""

import json
import os

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "kt_schema_new.json")

with open(_SCHEMA_PATH) as f:
    SCHEMA = json.load(f)["sections"]
