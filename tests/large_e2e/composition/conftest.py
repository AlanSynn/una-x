"""I00 composition test-package environment pins.

New independent suite package (clone discipline: mirrors the per-track
pins of A1/A3/F1/F2/F3; the suites stay independent):

* NUMBA_CACHE_DIR is routed OUTSIDE both source trees, to a dedicated
  composition root under the campaign custody root (campaign_data/,
  outside the repo) — separate from nbc_a1/nbc_a3/nbc_f1/nbc_f2/nbc_f3/
  nbc_oracle so no suite invalidates another's cache entries.
* L1_REUSE_DIR is removed so no prior campaign cache can be honored.
* NUMBA_NUM_THREADS is SET explicitly (ambient values never win; h04
  I00 plan item R5 — the per-track setdefault idiom is hardened here).

DECLARED-WRITE FRAME (h04 I00 plan item R1, superseding the earlier
"zero-write" phrasing): a suite run writes ONLY to
campaign_data/nbc_composition/** (numba cache; __pycache__/ is
gitignored, and bytecode suppression is additionally enabled below).
No other campaign_data path, no venvs, no campaigns/** evidence path
receives a write from this package: the only file-write site in the
package is the artifacts flush in harness_composition.flush_artifacts(),
which is env-gated (UNA_COMPOSITION_ARTIFACTS) and writes to exactly
the path that variable names. Override hooks:

UNA_COMPOSITION_NUMBA_CACHE    cache root instead of the default.
UNA_COMPOSITION_NUM_THREADS    numba thread cap instead of the default 2.
UNA_COMPOSITION_CANDIDATE_ROOT candidate import root instead of this
                               worktree's src.
UNA_COMPOSITION_ARTIFACTS      artifacts JSON path; UNSET = no write.
"""
from __future__ import annotations

import os
import sys

WT = "/Users/alansynn/orca/workspaces/una-x/wt-large-e2e"
DEFAULT_CACHE_ROOT = ("/Users/alansynn/orca/workspaces/una-x"
                      "/campaign_data/nbc_composition")
CANDIDATE_ROOT = os.environ.get("UNA_COMPOSITION_CANDIDATE_ROOT", f"{WT}/src")

os.environ["NUMBA_CACHE_DIR"] = os.environ.get(
    "UNA_COMPOSITION_NUMBA_CACHE", DEFAULT_CACHE_ROOT)
os.environ.pop("L1_REUSE_DIR", None)
# R5: SET, not setdefault — an ambient NUMBA_NUM_THREADS must not win.
os.environ["NUMBA_NUM_THREADS"] = os.environ.get(
    "UNA_COMPOSITION_NUM_THREADS", "2")

# Self-defense (h04 I00 plan, recommended): the candidate root stays
# inside this worktree and the cache root is the declared root unless
# explicitly overridden.
if not os.path.abspath(CANDIDATE_ROOT).startswith(f"{WT}/src"):
    raise RuntimeError(
        f"UNA_COMPOSITION_CANDIDATE_ROOT escaped the worktree: "
        f"{CANDIDATE_ROOT}")

# Suppress bytecode writes for every module imported after conftest
# (test modules, harnesses, fixtures). For the conftest module itself a
# .pyc may already exist from collection; the documented run command
# sets PYTHONDONTWRITEBYTECODE=1 to cover that too.
sys.dont_write_bytecode = True

# The H03 oracle package (fixtures, stub_topology, comparator, support,
# b0_import) is imported, never modified (per-track pin P5 idiom). The
# CANDIDATE root is deliberately NOT put on sys.path: the composed
# candidate is imported ONCE under the fixed alias
# `una_composition_cand` (cand_load_composition), so no module is ever
# imported twice under two names.
ORACLE_DIR = f"{WT}/tests/large_e2e/oracle"
if ORACLE_DIR not in sys.path:
    sys.path.insert(0, ORACLE_DIR)


def pytest_sessionfinish(session, exitstatus):
    """Persist the suite's collected evidence tail — ONLY when
    UNA_COMPOSITION_ARTIFACTS names a path; unset = no write at all
    (the declared-write proof lives in
    harness_composition.flush_artifacts)."""
    if os.environ.get("UNA_COMPOSITION_ARTIFACTS") is None:
        return
    try:
        import harness_composition
        harness_composition.flush_artifacts()
    except Exception:
        pass
