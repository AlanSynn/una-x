"""tests/tenx/gpu conftest: repo-root import frame + hardware laws.

These tests are CPU-ONLY.  A device-gated case skips with an explicit
hardware_unavailable marker and can never mark hardware validation passed
(TASKS_CLAUDE.yaml G10).  No test here imports src/urban_network_analysis;
the engine-facing comparison lives in the gate harness
(gpu/kernels/validate_device_bits.py --emit-host-battery) and runs outside
pytest with a pinned NUMBA_CACHE_DIR.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
