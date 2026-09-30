"""A3 (task A3I) test-package environment pins.

Mirrors the A1 package pins (clone discipline; the two suites stay
independent):

* NUMBA_CACHE_DIR is routed OUTSIDE both source trees, to a dedicated
  A3 root (separate from nbc_a1 and nbc_oracle: the tail-free kernel's
  cache entries must be enumerable on their own and the suites must
  not invalidate each other).
* L1_REUSE_DIR is removed so no prior campaign cache can be honored.
* NUMBA_NUM_THREADS defaults to 2 (small graphs, polite co-tenant);
  the concurrency test overrides it in its bounded child only.

Override hooks:
UNA_A3_NUMBA_CACHE   cache root instead of the default.
UNA_A3_NUM_THREADS   numba thread cap instead of the default 2.
UNA_A3_CANDIDATE_ROOT  candidate import root instead of this worktree's src.
"""
from __future__ import annotations

import os
import sys

from _legacy_paths import LEGACY_WS  # portable-path fix (HARNESS 2026-09-30)
DEFAULT_CACHE_ROOT = str(LEGACY_WS) + "/campaign_data/nbc_a3"
CANDIDATE_ROOT = os.environ.get(
    "UNA_A3_CANDIDATE_ROOT",
    str(LEGACY_WS) + "/wt-large-e2e/src",
)

os.environ["NUMBA_CACHE_DIR"] = os.environ.get("UNA_A3_NUMBA_CACHE", DEFAULT_CACHE_ROOT)
os.environ.pop("L1_REUSE_DIR", None)
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("UNA_A3_NUM_THREADS", "2"))

# The H03 oracle package (fixtures, comparator, b0_import, support) is
# imported, never modified (prereg review pin P5).
ORACLE_DIR = str(LEGACY_WS) + "/wt-large-e2e/tests/large_e2e/oracle"
if ORACLE_DIR not in sys.path:
    sys.path.insert(0, ORACLE_DIR)
