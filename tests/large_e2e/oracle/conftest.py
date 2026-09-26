"""H03 oracle test-package environment pins.

This conftest MUST run before any test module imports numba or the B0
package, so it pins the numerical environment at import time:

* NUMBA_CACHE_DIR is routed OUTSIDE both source trees (never inside the
  B0 worktree or a candidate tree; helper changes must not be masked by
  stale caches living next to the sources).
* L1_REUSE_DIR is removed so no prior campaign cache directory can be
  honored.
* NUMBA_NUM_THREADS is capped (small graphs; be a polite co-tenant).

Override hooks (used by benchmarks/large_e2e/oracle/compare_arms.py):
UNA_ORACLE_NUMBA_CACHE   cache root to use instead of the default.
UNA_ORACLE_NUM_THREADS   numba thread cap instead of the default 2.
"""
from __future__ import annotations

import os

DEFAULT_CACHE_ROOT = "/Users/alansynn/orca/workspaces/una-x/campaign_data/nbc_oracle"

os.environ["NUMBA_CACHE_DIR"] = os.environ.get(
    "UNA_ORACLE_NUMBA_CACHE", DEFAULT_CACHE_ROOT
)
os.environ.pop("L1_REUSE_DIR", None)
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("UNA_ORACLE_NUM_THREADS", "2"))
