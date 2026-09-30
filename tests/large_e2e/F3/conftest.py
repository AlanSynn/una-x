"""F3 (task F3I) test-package environment pins.

Mirrors the F1 convention (tests/large_e2e/F1/conftest.py):

* NUMBA_CACHE_DIR is routed OUTSIDE both source trees, to a dedicated
  F3 root (separate from nbc_f1 / nbc_oracle / nbc_a1: the suites must
  not invalidate each other's caches).
* L1_REUSE_DIR is removed so no prior campaign cache can be honored.
* NUMBA_NUM_THREADS defaults to 2 (irrelevant to the F3 mechanism —
  the gradient precompute is single-threaded SciPy — but pinned for
  reproducibility).

Override hooks:
UNA_F3_NUMBA_CACHE     cache root instead of the default.
UNA_F3_NUM_THREADS     numba thread cap instead of the default 2.
UNA_F3_CANDIDATE_ROOT  candidate import root instead of this worktree's src.
"""
from __future__ import annotations

import os
import sys

from _legacy_paths import LEGACY_WS  # portable-path fix (HARNESS 2026-09-30)
DEFAULT_CACHE_ROOT = str(LEGACY_WS) + "/campaign_data/nbc_f3"
CANDIDATE_ROOT = os.environ.get(
    "UNA_F3_CANDIDATE_ROOT",
    str(LEGACY_WS) + "/wt-large-e2e/src",
)

os.environ["NUMBA_CACHE_DIR"] = os.environ.get("UNA_F3_NUMBA_CACHE", DEFAULT_CACHE_ROOT)
os.environ.pop("L1_REUSE_DIR", None)
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("UNA_F3_NUM_THREADS", "2"))

# The H03 oracle package (fixtures, comparator, b0_import, stub_topology)
# is imported, never modified.
ORACLE_DIR = str(LEGACY_WS) + "/wt-large-e2e/tests/large_e2e/oracle"
if ORACLE_DIR not in sys.path:
    sys.path.insert(0, ORACLE_DIR)


def pytest_sessionfinish(session, exitstatus):
    """Persist the suite's collected evidence tail even on failures."""
    try:
        import harness_f3
        harness_f3.flush_artifacts()
    except Exception:
        pass
