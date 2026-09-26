"""T12: multi-thread candidate-vs-B0 byte equality (bounded children,
NUMBA_NUM_THREADS=4, unique cache roots per arm — compare_arms
pattern)."""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile

import numpy as np

CANDIDATE_ROOT = os.environ.get(
    "UNA_A1_CANDIDATE_ROOT",
    "/Users/alansynn/orca/workspaces/una-x/wt-large-e2e/src")
B0_ROOT = "/Users/alansynn/orca/workspaces/una-x/wt-b0/src"
CAMPAIGN_PYTHON = ("/Users/alansynn/orca/workspaces/una-x/venvs/campaign/"
                   "bin/python")
CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mt_child.py")
CHILD_TIMEOUT_S = 1200


def _run_arm(root, workdir):
    cache = os.path.join(workdir, "cache")  # unique per arm
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = cache
    env["NUMBA_NUM_THREADS"] = "4"
    env.pop("L1_REUSE_DIR", None)
    out_npz = os.path.join(workdir, "out.npz")
    proc = subprocess.run(
        [CAMPAIGN_PYTHON, CHILD, root, out_npz],
        cwd="/tmp", env=env, capture_output=True, text=True,
        timeout=CHILD_TIMEOUT_S)
    return proc, out_npz


def _digest(arr):
    h = hashlib.sha256()
    h.update(arr.dtype.str.encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def test_t12_multithread_four_threads_byte_equal():
    with tempfile.TemporaryDirectory(prefix="a1_t12_") as work:
        work_b0 = os.path.join(work, "b0")
        work_c = os.path.join(work, "cand")
        os.mkdir(work_b0)
        os.mkdir(work_c)
        proc_b0, npz_b0 = _run_arm(B0_ROOT, work_b0)
        assert proc_b0.returncode == 0, proc_b0.stderr[-3000:]
        proc_c, npz_c = _run_arm(CANDIDATE_ROOT, work_c)
        assert proc_c.returncode == 0, proc_c.stderr[-3000:]
        za = np.load(npz_b0)
        zb = np.load(npz_c)
    assert set(za.files) == set(zb.files)
    for key in sorted(za.files):
        assert _digest(np.ascontiguousarray(za[key])) == \
            _digest(np.ascontiguousarray(zb[key])), key
