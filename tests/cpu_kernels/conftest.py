"""Fixtures for the CPU_ALGORITHMS conformance suite.

``build_case`` emits a deterministic adversarial CSR: duplicate arcs,
self-loops, a leaf push-gate arc, optional NaN/negative/oversized seeds,
float32 boundary variants and a fixed destination battery with
negative/zero terminal weights.  Every test module consumes the same
battery so a single seed pins a single topology across all oracles.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))


def build_case(seed, v=14, extra_arcs=3, dup=True, self_loop=True,
               weights_f32=False, terminals_f32=False,
               nan_seed=False, neg_seed=False, big_seed=False,
               same_terminal=False, cutoff=25.0, nan_seed_start=False):
    """Deterministic adversarial CSR case (see module docstring)."""
    r = np.random.default_rng(seed)
    pointer = [0]
    vector = []
    weights = []
    for v_id in range(v):
        degree = int(r.integers(0, 4))
        for _ in range(degree):
            vector.append(int(r.integers(0, v)))
            weights.append(float(r.uniform(0.1, 6.0)))
        if dup and v_id % 4 == 1 and vector:
            vector.append(vector[-1])
            weights.append(float(r.uniform(0.1, 2.0)))
        if self_loop and v_id % 5 == 2:
            vector.append(v_id)
            weights.append(float(r.uniform(0.1, 3.0)))
        pointer.append(len(vector))
    for _ in range(extra_arcs):
        vector.append(int(r.integers(0, v)))
        weights.append(float(r.uniform(0.1, 6.0)))
        pointer[-1] += 1
    ptr = np.asarray(pointer, dtype=np.int64)
    vec = np.asarray(vector, dtype=np.int64)
    if weights_f32:
        wts = np.asarray(weights, dtype=np.float32)
    else:
        wts = np.asarray(weights, dtype=np.float64)
    flag = r.random(len(vector)) < 0.85
    flag[pointer[1]:pointer[1] + 1] = True  # leaf push-gate case
    if same_terminal:
        o = np.array([[2, 2]], dtype=np.int64)
    else:
        o = np.array([[0, 3]], dtype=np.int64)
    ow = np.array([[1.5, 0.75]], dtype=np.float64)
    if nan_seed:
        ow[0, 1] = np.nan
    if nan_seed_start:
        ow[0, 0] = np.nan
    if neg_seed:
        ow[0, 1] = -3.0
    if big_seed:
        ow[0, 1] = cutoff + 10.0
    if terminals_f32:
        ow = ow.astype(np.float32)
    d = np.array([[1, 4], [4, 7], [7, 2], [2, 9], [9, 9], [12, 0]],
                 dtype=np.int64)
    dw = np.array([[0.5, 0.25], [1.0, 1.0], [0.0, 0.5], [2.0, -1.0],
                   [0.25, 0.25], [1.5, 0.5]], dtype=np.float64)
    dwt = np.array([1.0, 0.5, 2.0, 0.75, 1.25, 0.3], dtype=np.float64)
    return dict(ptr=ptr, vec=vec, wts=wts, flag=flag, o=o, ow=ow,
                d=d, dw=dw, dwt=dwt, cutoff=cutoff)


CASE_PARAMS = [
    "base",
    "nan_seed_end",
    "neg_seed",
    "big_seed",
    "same_terminal",
    "zero_cutoff",
    "weights_f32",
]
# "nan_seed_start" (NaN on BOTH origin seeds) is deliberately NOT in the
# battery: the compiled engines DO NOT TERMINATE on it (label oscillation
# — see SPEC.md section 3 and the bounded-cap test in
# test_schedule_pinning).  build_case keeps the flag for that test.


def case_kwargs(name):
    """The battery name -> build_case kwargs mapping (single source)."""
    if name == "nan_seed_end":
        return dict(nan_seed=True)
    if name == "neg_seed":
        return dict(neg_seed=True)
    if name == "big_seed":
        return dict(big_seed=True)
    if name == "same_terminal":
        return dict(same_terminal=True)
    if name == "zero_cutoff":
        return dict(cutoff=0.0)
    if name == "weights_f32":
        return dict(weights_f32=True)
    return {}


@pytest.fixture(params=CASE_PARAMS, ids=CASE_PARAMS)
def case(request):
    """The standard case battery; ids are the canonical case names."""
    name = request.param
    return build_case(3 + CASE_PARAMS.index(name),
                      **case_kwargs(name))


def bits(a):
    """Byte identity of an array's values (dtype included implicitly)."""
    return np.ascontiguousarray(a).tobytes()
