"""Path bootstrap for the una_10x harness negative tests (same idiom as
tests/incidents/incident_runner.py: explicit repo insert)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
