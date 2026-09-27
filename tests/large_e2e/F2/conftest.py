"""F2 (task F2I) test-package environment pins.

Mirrors the F1/F3 convention (tests/large_e2e/F1/conftest.py):

* NUMBA_CACHE_DIR is routed OUTSIDE both source trees, to a dedicated
  F2 root (separate from nbc_f1 / nbc_f3 / nbc_oracle / nbc_a1: the
  suites must not invalidate each other's caches).
* L1_REUSE_DIR is removed so no prior campaign cache can be honored.
* NUMBA_NUM_THREADS defaults to 2 (the F2 kernel is njit-compiled but
  not parallel; pinned for reproducibility).

Override hooks:
UNA_F2_NUMBA_CACHE     cache root instead of the default.
UNA_F2_NUM_THREADS     numba thread cap instead of the default 2.
UNA_F2_ARTIFACTS       artifact JSON path instead of the evidence default.
"""
from __future__ import annotations

import os
import sys

WT = "/Users/alansynn/orca/workspaces/una-x/wt-large-e2e"
DEFAULT_CACHE_ROOT = "/Users/alansynn/orca/workspaces/una-x/campaign_data/nbc_f2"
CANDIDATE_ROOT = os.environ.get("UNA_F2_CANDIDATE_ROOT", f"{WT}/src")

os.environ["NUMBA_CACHE_DIR"] = os.environ.get("UNA_F2_NUMBA_CACHE", DEFAULT_CACHE_ROOT)
os.environ.pop("L1_REUSE_DIR", None)
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("UNA_F2_NUM_THREADS", "2"))

# This worktree's src (the F2I implementation under test) is the single
# import arm; the differential inside these tests is the ROUTE toggle
# (baseline kernel vs local kernel) within the one module.
if CANDIDATE_ROOT not in sys.path:
    sys.path.insert(0, CANDIDATE_ROOT)

# The H03 oracle package (fixtures, stub_topology) is imported, never
# modified.
ORACLE_DIR = f"{WT}/tests/large_e2e/oracle"
if ORACLE_DIR not in sys.path:
    sys.path.insert(0, ORACLE_DIR)


def pytest_sessionfinish(session, exitstatus):
    """Persist the suite's collected evidence tail even on failures."""
    try:
        import harness_f2
        harness_f2.flush_artifacts()
    except Exception:
        pass
