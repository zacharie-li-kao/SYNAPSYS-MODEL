"""Pytest configuration for the SYNAPSYS verification harness.

Pins imports to the canonical modules at the project root. Several historical
copies of `fitting.py`, `synapse_engine.py` and `data_converter.py` live under
`archive/`, `test_export/` and the `Save *` directories; without an explicit
path the harness can end up verifying one of those instead.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Deliberately position 0: prepending is not enough if the invoking shell put
# another copy earlier on the path.
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)


def pytest_report_header(config):
    return f"SYNAPSYS project root: {PROJECT_ROOT}"
