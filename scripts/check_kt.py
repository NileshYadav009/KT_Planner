#!/usr/bin/env python3
"""Wrapper to run top-level `check_kt.py` from `scripts/`.
"""
from importlib import import_module

if __name__ == '__main__':
    try:
        mod = import_module('check_kt')
    except Exception:
        # fallback: call requests-based runner
        mod = None
    if mod and hasattr(mod, '__name__'):
        # the top-level module prints on import; nothing else to do
        pass
    else:
        print('Wrapper: ensure top-level check_kt.py exists and is importable')
