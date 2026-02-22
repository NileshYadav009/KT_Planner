#!/usr/bin/env python3
"""Wrapper to run the top-level `generate_audio.py` from `scripts/`.
"""
from importlib import import_module

if __name__ == '__main__':
    mod = import_module('generate_audio')
    # If the original exposes a CLI guard, just run it
    if hasattr(mod, 'generate_audio') and hasattr(mod, 'transcript'):
        # call generate_audio using transcript defined in the top-level script
        mod.generate_audio(mod.transcript)
    elif hasattr(mod, 'main'):
        mod.main()
    else:
        print('Wrapper: could not find a runnable entry in generate_audio.py')
