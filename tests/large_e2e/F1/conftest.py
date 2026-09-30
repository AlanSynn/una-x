"""F1 (task F1I) test-package environment pins.

Runs before any test module imports numba or a UNA package:

* NUMBA_CACHE_DIR is routed OUTSIDE both source trees, to a dedicated
  F1 root (separate from the oracle suite's nbc_oracle and A1's
  nbc_a1: the suites must not invalidate each other's caches).
* L1_REUSE_DIR is removed so no prior campaign cache can be honored.
* NUMBA_NUM_THREADS defaults to 2 (irrelevant to the F1 mechanism —
  the concurrency locus is the engine's ThreadPoolExecutor, proof
  section 5c — but pinned for reproducibility); bounded children
  override it explicitly.

Override hooks:
UNA_F1_NUMBA_CACHE     cache root instead of the default.
UNA_F1_NUM_THREADS     numba thread cap instead of the default 2.
UNA_F1_CANDIDATE_ROOT  candidate import root instead of this worktree's src.
"""
from __future__ import annotations

import os
import sys

from _legacy_paths import LEGACY_WS  # portable-path fix (HARNESS 2026-09-30)
DEFAULT_CACHE_ROOT = str(LEGACY_WS) + "/campaign_data/nbc_f1"
CANDIDATE_ROOT = os.environ.get(
    "UNA_F1_CANDIDATE_ROOT",
    str(LEGACY_WS) + "/wt-large-e2e/src",
)

os.environ["NUMBA_CACHE_DIR"] = os.environ.get("UNA_F1_NUMBA_CACHE", DEFAULT_CACHE_ROOT)
os.environ.pop("L1_REUSE_DIR", None)
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("UNA_F1_NUM_THREADS", "2"))

# The H03 oracle package (fixtures, comparator, b0_import, stub_topology,
# observed_replay) is imported, never modified.
ORACLE_DIR = str(LEGACY_WS) + "/wt-large-e2e/tests/large_e2e/oracle"
if ORACLE_DIR not in sys.path:
    sys.path.insert(0, ORACLE_DIR)


def pytest_sessionfinish(session, exitstatus):
    """Persist the suite's collected evidence tail (warm-up times,
    first-divergence records, census results) even on failures."""
    try:
        import harness_f1
        harness_f1.flush_artifacts()
    except Exception:
        pass
