"""A3 bounded-children battery: T6 concurrency (4-thread pool digests
equal the parent's 2-thread digests) and the positive-boundsafety run
(NUMBA_BOUNDSCHECK=1 across every origin; no BoundsError, same
digest).

Both children are fresh interpreters of the same venv python
(sys.executable), with env overrides set by the parent before spawn;
each child has a bounded timeout.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
from harness_a3 import cns, run_integrated, run_od

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_ROOT = os.environ["UNA_A3_CANDIDATE_ROOT"] \
    if "UNA_A3_CANDIDATE_ROOT" in os.environ \
    else os.path.join(HERE, "../../../src")
CANDIDATE_ROOT = os.path.abspath(CANDIDATE_ROOT)

TIMEOUT_S = 540


def digest(arr):
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _run_child(script, extra_env, out_path, cache_dir):
    env = dict(os.environ)
    env.update(extra_env)
    # fresh per-child cache: entries written by the suite's alias-loaded
    # candidate pickle the alias module name, which a real-name import
    # in the child cannot resolve (A1 test-infra collision precedent)
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, script), CANDIDATE_ROOT,
         str(out_path)],
        env=env, cwd=HERE, timeout=TIMEOUT_S, capture_output=True,
        text=True)
    return proc


def _parent_graph_arrays(seed=20260925, v=120):
    """Same arrays the children build (duplicated by design: the child
    is a standalone interpreter)."""
    rng = np.random.default_rng(seed)
    pointer = [0]
    nbrs, weights, flags = [], [], []
    for node in range(v):
        targets = {(node + 1) % v, (node + 7) % v, (node + 29) % v}
        deg = int(rng.integers(1, 4))
        for t in sorted(targets)[:deg]:
            nbrs.append(t)
            weights.append(round(float(rng.uniform(0.5, 9.0)), 6))
            flags.append(bool(t % 5 != 3))
        pointer.append(len(nbrs))
    o_count, d_count = 5, 6
    oti = np.stack(
        [(np.arange(o_count) * 11) % v, (np.arange(o_count) * 11 + 1) % v],
        axis=1).astype(np.int64)
    otw = np.round(rng.uniform(0.0, 2.0, (o_count, 2)), 6)
    dti = np.stack(
        [(np.arange(d_count) * 13 + 3) % v,
         (np.arange(d_count) * 13 + 5) % v], axis=1).astype(np.int64)
    dtw = np.round(rng.uniform(0.0, 2.0, (d_count, 2)), 6)
    return dict(
        adjacency_pointer=np.asarray(pointer, np.int64),
        adjacency_vector=np.asarray(nbrs, np.int64),
        adjacency_vector_weights=np.asarray(weights, np.float64),
        adjacynct_vector_network_node=np.asarray(flags, np.bool_),
        o_terminal_idxs=oti, o_terminal_weights=otw,
        d_count=d_count, d_terminal_idxs=dti, d_terminal_weights=dtw,
        d_weights=np.round(rng.uniform(0.5, 3.0, d_count), 6),
        gravity_beta=0.05, metric_plateau=0.0, metric_midpoint=5.0,
        knn_decay="logistic", knn_weights=np.array([1.0, 0.6, 0.3]),
        cutoff=12.5,
    )


def test_four_thread_pool_byte_equal(tmp_path):
    arrays = _parent_graph_arrays()
    out = tmp_path / "a3_mt_child.json"
    proc = _run_child("a3_mt_child.py",
                      {"NUMBA_NUM_THREADS": "4"}, out,
                      tmp_path / "cache_mt")
    assert proc.returncode == 0, proc.stderr[-4000:]
    child = json.loads(out.read_text())
    assert child["numba_num_threads"] == 4

    ns = cns()
    od = np.asarray(run_od(ns, arrays, family="acce"))
    assert digest(od) == child["od"], "od digest differs at 4 threads"
    outs = run_integrated(ns, arrays, family="acce")
    for name, arr in zip(("reach", "gravity_exponential", "gravity_logistic",
                          "knn_access"), outs):
        assert digest(arr) == child[name], f"{name} digest differs at 4 threads"


def test_boundscheck_run_is_clean_and_equal(tmp_path):
    arrays = _parent_graph_arrays()
    out = tmp_path / "a3_bounds_child.json"
    proc = _run_child("a3_bounds_child.py",
                      {"NUMBA_BOUNDSCHECK": "1"}, out,
                      tmp_path / "cache_bounds")
    assert proc.returncode == 0, proc.stderr[-4000:]
    child = json.loads(out.read_text())
    assert child["boundscheck"] == "1"
    assert child["origins_run"] == 10  # 5 origins x 2 terminal sets

    # Same kernel runs in-process (bounds off): identical bytes.
    ns = cns()
    h = hashlib.sha256()
    pointer = arrays["adjacency_pointer"]
    max_degree = int(np.diff(pointer).max())
    for dti in (arrays["d_terminal_idxs"],
                np.stack([np.zeros(6, np.int64),
                          np.full(6, arrays["adjacency_pointer"].shape[0] - 2,
                                  np.int64)], axis=1)):
        assert ns._a3_tail_admits(dti, dti.shape[0], pointer.shape[0] - 1)
        for o in range(arrays["o_terminal_idxs"].shape[0]):
            eligible_offset = np.empty(max_degree, dtype=np.int64)
            eligible_weight = np.empty(max_degree, dtype=np.float64)
            labels, _ = ns._a3_scope_search_tailless(
                arrays["o_terminal_idxs"][o],
                arrays["o_terminal_weights"][o],
                pointer,
                arrays["adjacency_vector"],
                arrays["adjacency_vector_weights"],
                arrays["adjacynct_vector_network_node"],
                arrays["cutoff"],
                eligible_offset, eligible_weight)
            h.update(np.ascontiguousarray(labels).tobytes())
    assert h.hexdigest() == child["labels_digest"]
