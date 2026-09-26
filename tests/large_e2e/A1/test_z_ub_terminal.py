"""UB-terminal refusal class in a bounded child (proof.md section 1
UB note + section 9 T13 routing).

An origin terminal index outside [0, V) refuses the private route; the
fallback then executes the ORIGINAL kernel, which writes
o_scope_weights[o_terminal_idxs[o, 0]] with numba bounds checking OFF
— B0's true undefined behavior. Per the H04 hazard policy this input
runs only in bounded children, never in-process. The pin is FAILURE
IDENTITY: both arms die with a fatal memory fault (SIGSEGV/SIGBUS/
SIGILL class), never a Python exception and never a surviving process.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

CANDIDATE_ROOT = os.environ.get(
    "UNA_A1_CANDIDATE_ROOT",
    "/Users/alansynn/orca/workspaces/una-x/wt-large-e2e/src")
B0_ROOT = "/Users/alansynn/orca/workspaces/una-x/wt-b0/src"
CAMPAIGN_PYTHON = ("/Users/alansynn/orca/workspaces/una-x/venvs/campaign/"
                   "bin/python")
CHILD_TIMEOUT_S = 900
# Fatal memory-fault signals, resolved per platform (SIGBUS is 10 on
# macOS, 7 on Linux).
import signal as _signal
_FATAL_SIGNALS = {int(_signal.SIGILL), int(_signal.SIGBUS),
                  int(_signal.SIGSEGV)}

_CHILD_SRC = """
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import urban_network_analysis.Engines.AccessibilityWElevation as acwe

V = 60
ptr = np.arange(V + 1, dtype=np.int64)          # degree 1 per node
nbr = np.array([(i + 1) % V for i in range(V)], dtype=np.int64)
wgt = np.full(V, 1.0)
flg = np.ones(V, dtype=np.bool_)
oti = np.array([[0, 0]], dtype=np.int64)
oti[0, 0] = 10 ** 9                             # OOB origin terminal
otw = np.zeros((1, 2))
dti = np.array([[1, 2]], dtype=np.int64)
dtw = np.zeros((1, 2))
d_w = np.ones(1)
acwe.integrated_scope_access(
    oti, otw, ptr, nbr, wgt, flg, dti, dtw, d_w,
    0.05, 0.0, 5.0, float(np.log(99.0) / 5.0), "logistic",
    np.array([1.0, 0.6, 0.3]), 30.0)
print("NO_CRASH")                               # must not be reached
"""


def _run_arm(root):
    env = dict(os.environ)
    env["NUMBA_NUM_THREADS"] = env.get("UNA_A1_NUM_THREADS", "2")
    env.pop("L1_REUSE_DIR", None)
    with tempfile.TemporaryDirectory(prefix="a1_ub_") as workdir:
        # fresh per-arm cache: cache entries written by the suite's
        # alias-loaded candidate pickle the alias module name, and a
        # real-name import reading them fails with ModuleNotFoundError
        # (test-infra collision, not an implementation defect)
        child = os.path.join(workdir, "ub_child.py")
        with open(child, "w") as fh:
            fh.write(_CHILD_SRC)
        env["NUMBA_CACHE_DIR"] = os.path.join(workdir, "cache")
        return subprocess.run(
            [CAMPAIGN_PYTHON, child, root],
            cwd="/tmp", env=env, capture_output=True, text=True,
            timeout=CHILD_TIMEOUT_S)


def test_ub_terminal_refusal_failure_identity():
    procs = [(root, _run_arm(root)) for root in (B0_ROOT, CANDIDATE_ROOT)]
    for root, proc in procs:
        assert proc.returncode < 0, (
            f"{root}: expected a fatal memory fault, got returncode "
            f"{proc.returncode}; stdout={proc.stdout[-500:]!r} "
            f"stderr={proc.stderr[-500:]!r}")
        sig = -proc.returncode
        assert sig in _FATAL_SIGNALS, (
            f"{root}: non-memory-fault signal {sig}")
        assert "NO_CRASH" not in proc.stdout
