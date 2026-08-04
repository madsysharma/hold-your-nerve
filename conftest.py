"""
pytest configuration for the hold-your-nerve repo.

Placing this file at the repo root does two things pytest cares about:
  1. It anchors pytest's rootdir here.
  2. pytest adds this directory to sys.path, so tests can import top-level
     modules (error_injection, cpn, cpn_epoch_cpn, experiment, ...) no matter
     which directory pytest is invoked from.

The explicit insert below makes that independent of pytest's import mode, so
`pytest -v tests/` works without any per-file sys.path juggling.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
