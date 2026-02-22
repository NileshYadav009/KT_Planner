"""
Policy configuration for KT pipeline.

Defines heuristics and thresholds used to detect implementation steps
and to decide when to route sentences to review-required.
"""
from typing import List

# Confidence threshold for accepting automatic section assignment (0.0 - 1.0)
CONFIDENCE_ACCEPT_THRESHOLD = 0.80

# Implementation step indicators (lowercase strings matched against sentence text)
IMPLEMENTATION_INDICATORS: List[str] = [
    "kubectl",
    "docker",
    "compose",
    "helm",
    "apply -f",
    "systemctl",
    "service",
    "install ",
    "pip install",
    "npm install",
    "sh ",
    "bash ",
    "curl ",
    ".yaml",
    ".yml",
    "deployment.yaml",
    "step ",
    "1.",
    "2.",
    "run ",
    "execute ",
    "sudo ",
]

# Conceptual section names (lowercase substrings) — implementations should not be placed here
CONCEPTUAL_SECTIONS = ["overview", "architecture", "data flow", "monitoring", "risks", "faqs"]
