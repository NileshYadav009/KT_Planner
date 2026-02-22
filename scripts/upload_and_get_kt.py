#!/usr/bin/env python3
"""Wrapper to run the top-level `upload_and_get_kt.py` from `scripts/`.
This keeps helpers discoverable under `scripts/` while preserving the original file locations.
"""

from importlib import import_module

if __name__ == '__main__':
    # Import and call the main from the project root script
    mod = import_module('upload_and_get_kt')
    if hasattr(mod, 'main'):
        mod.main()
    else:
        print('Wrapper: upload_and_get_kt.main() not found')
