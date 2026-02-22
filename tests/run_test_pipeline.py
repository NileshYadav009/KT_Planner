#!/usr/bin/env python3
"""Run the integration test runner from `tests/`.
This delegates to the top-level `test_pipeline.py` to avoid duplicating code.
"""
import subprocess
import sys

cmd = [sys.executable, '-u', 'test_pipeline.py']
print('Running integration tests...')
ret = subprocess.call(cmd)
if ret != 0:
    print('Tests failed')
    sys.exit(ret)
print('Tests completed successfully')
