"""large_e2e suite conftest (HARNESS relocatability repair, 2026-09-30).

Fixes two historical defects without touching goldens/committed evidence:

1. Bare helper imports (`import fixtures`, `from comparator import ...`,
   `import observed_replay`) resolved only when pytest was launched from
   inside tests/large_e2e/oracle.  The oracle directory and this directory
   are now placed on sys.path repo-relatively, so every sub-suite collects
   from any cwd.

2. Absolute macOS-workspace paths.  All such paths now resolve through
   _legacy_paths.LEGACY_WS (env UNA_LEGACY_WORKSPACE).  Sub-suites whose
   arms require the historical worktree/venv layout skip with an explicit
   reason when it is absent; tests/large_e2e/harness (stub-based) is exempt
   and runs wherever its own dependencies resolve.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORACLE_DIR = str(HERE / "oracle")
for _p in (str(HERE), ORACLE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _legacy_paths import _LEGACY_SUITES, skip_without_legacy_layout  # noqa: E402


def pytest_collection_modifyitems(session, config, items):
    # keep collection intact; the per-test skip lives in the autouse fixture
    return None


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _require_legacy_layout_when_needed(request):
    node_path = Path(request.node.path).resolve()
    parts = set(node_path.parts)
    if parts & set(_LEGACY_SUITES):
        skip_without_legacy_layout(request)
    yield
