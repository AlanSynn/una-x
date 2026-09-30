"""Portable path resolution for the historical large_e2e suites
(HARNESS relocatability repair, campaign una-platform-2026-09).

The prior campaign recorded absolute macOS-workspace paths
(`/Users/alansynn/orca/workspaces/una-x/...`) inside these suites.  Every
such path now resolves through LEGACY_WS:

* env override      UNA_LEGACY_WORKSPACE=<old workspace root>
* default           the parent of this checkout (same layout as the old
                    machine: <workspace>/una-x, <workspace>/wt-large-e2e,
                    <workspace>/campaign_data, <workspace>/venvs)

When the historical layout is absent, suites that require it skip with an
explicit reason instead of erroring on missing directories.  Suites never
write into committed evidence: new artifact output goes to ARTIFACTS_OUT
(env UNA_LARGE_E2E_ARTIFACTS, default a per-user temp directory outside the
repository).

Read-only historical evidence lives in-repo under campaigns/history/ and is
never a write target.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # tests/large_e2e
_REPO = _HERE.parents[1]                           # checkout root

LEGACY_WS = Path(os.environ.get(
    "UNA_LEGACY_WORKSPACE", str(_REPO.parent)))

ARTIFACTS_OUT = Path(os.environ.get(
    "UNA_LARGE_E2E_ARTIFACTS",
    str(Path(tempfile.gettempdir()) / "una_large_e2e_artifacts")))

# Sub-suites whose arms require the historical worktree/venv layout.
_LEGACY_SUITES = ("A1", "A3", "F1", "F2", "F3", "composition", "installed",
                  "oracle", "scheduling")


def legacy_layout_present() -> bool:
    """True when the historical campaign layout exists under LEGACY_WS."""
    return (LEGACY_WS / "wt-large-e2e" / "src").is_dir() and \
        (LEGACY_WS / "campaign_data").is_dir()


def skip_without_legacy_layout(request) -> None:
    """Skip (per-test) when the suite needs the historical layout."""
    import pytest
    if not legacy_layout_present():
        suite = Path(request.node.path).parent.name
        pytest.skip(
            f"suite '{suite}' requires the historical large_e2e layout "
            f"(wt-large-e2e worktree + campaign_data) under {LEGACY_WS}; "
            f"set UNA_LEGACY_WORKSPACE to the old workspace to enable it")
