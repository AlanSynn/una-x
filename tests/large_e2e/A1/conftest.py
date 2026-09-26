"""A1 (task A1I) test-package environment pins.

Runs before any test module imports numba or a UNA package:

* NUMBA_CACHE_DIR is routed OUTSIDE both source trees, to a dedicated
  A1 root (separate from the oracle suite's nbc_oracle: helper changes
  must not be masked by stale caches and the two suites must not
  invalidate each other).
* L1_REUSE_DIR is removed so no prior campaign cache can be honored.
* NUMBA_NUM_THREADS defaults to 2 (small graphs, polite co-tenant);
  the T12 multi-thread test overrides it in its bounded children.

Override hooks:
UNA_A1_NUMBA_CACHE   cache root instead of the default.
UNA_A1_NUM_THREADS   numba thread cap instead of the default 2.
UNA_A1_CANDIDATE_ROOT  candidate import root instead of this worktree's src.
"""
from __future__ import annotations

import os
import sys

DEFAULT_CACHE_ROOT = "/Users/alansynn/orca/workspaces/una-x/campaign_data/nbc_a1"
CANDIDATE_ROOT = os.environ.get(
    "UNA_A1_CANDIDATE_ROOT",
    "/Users/alansynn/orca/workspaces/una-x/wt-large-e2e/src",
)

os.environ["NUMBA_CACHE_DIR"] = os.environ.get("UNA_A1_NUMBA_CACHE", DEFAULT_CACHE_ROOT)
os.environ.pop("L1_REUSE_DIR", None)
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("UNA_A1_NUM_THREADS", "2"))

# The H03 oracle package (fixtures, comparator, b0_import, observed_replay)
# is imported, never modified.
ORACLE_DIR = "/Users/alansynn/orca/workspaces/una-x/wt-large-e2e/tests/large_e2e/oracle"
if ORACLE_DIR not in sys.path:
    sys.path.insert(0, ORACLE_DIR)
