"""Fixtures for the N00 native producer suite (campaign una_10x).

The native route is the AOT C++17 twin of the trace module's producer
region; every parity test here runs BOTH routes over identical inputs
and demands bit-exact records.  The adversarial case battery is
imported verbatim from its single source (the cpu_kernels conftest,
same as the trace suite) — never retyped.  The numba cache directory is
forced to the owner-owned scratch so ``cache=True`` jitting can never
write into frozen trees, and the extension is built/stale-checked by
native/build.py's own manifest (sources + output digests).

Run in the DEDICATED venv (venv-n00, pybind11 + numba pinned like
venv-p01m), per-directory, never combined with other test dirs:

    venv-n00/bin/python -m pytest tests/native -q
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Force (not setdefault): every cached engine module must jit into
# owner-owned scratch, never into site data or frozen trees.
os.environ["NUMBA_CACHE_DIR"] = str(
    Path(__file__).resolve().parents[2] / "campaign_runs"
    / "_numba_cache_owner")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

REPO = Path(__file__).resolve().parents[2]
REPO_SRC = REPO / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
NATIVE_DIR = REPO / "native"
if str(NATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(NATIVE_DIR))

# The adversarial battery, from its single source (never retyped).
_CK_PATH = REPO / "tests" / "cpu_kernels" / "conftest.py"
_ck_spec = importlib.util.spec_from_file_location(
    "cpu_kernels_conftest", _CK_PATH)
ck = importlib.util.module_from_spec(_ck_spec)
_ck_spec.loader.exec_module(ck)

build_case = ck.build_case
CASE_PARAMS = ck.CASE_PARAMS
case_kwargs = ck.case_kwargs
bits = ck.bits


@pytest.fixture(scope="session")
def native_pair():
    """(numba build_trace, native twin) with the extension built and
    stale-checked; skips the whole suite if pybind11 is unavailable
    (the core engine still runs numba-only without this suite)."""
    pytest.importorskip("pybind11")
    from urban_network_analysis.Engines import _trace_access_scratch as trm
    import trace_build_native as tbn
    tbn._load_extension()
    return trm.build_trace, tbn.native_build_trace


def record_diffs(ref, nat):
    """Every field on which two TraceRecords must agree exactly.

    The records are indistinguishable by law: identity, geometry,
    refusal text, digest over packed bytes, and every counter EXCEPT
    tile_temp_bytes, which each route measures over its own scratch by
    design (numba: per-tile capture buffers; native: measured arena
    high-water).  Both figures are honest measurements, compared
    against their own declared bounds, never against each other.
    """
    diffs = []
    for f in ("trace_id", "schema_version", "origin_count", "dest_count",
              "cutoff_value", "cutoff_dtype", "reach_dtype", "metric_dtype",
              "complete", "route", "refusal", "content_digest"):
        rv, nv = getattr(ref, f), getattr(nat, f)
        if rv != nv:
            diffs.append((f, rv, nv))
    for f in ("indptr", "destination_ids"):
        ra, na = getattr(ref, f), getattr(nat, f)
        if ra.dtype != na.dtype or not np.array_equal(ra, na):
            diffs.append((f, "numba", "native"))
    if ref.distance_bits.tobytes() != nat.distance_bits.tobytes():
        diffs.append(("distance_bits", "numba", "native"))
    rc, nc = ref.counters, nat.counters
    for f in rc.__dataclass_fields__:
        if f == "tile_temp_bytes":
            continue
        rv, nv = getattr(rc, f), getattr(nc, f)
        if rv != nv:
            diffs.append(("counters." + f, rv, nv))
    return diffs


def assert_parity(ref, nat):
    diffs = record_diffs(ref, nat)
    assert not diffs, "native record diverges from numba: %r" % (diffs,)


def ring_case(v=400, extra=300, origins=40, d_count=100, cutoff=8.0,
              seed=23, equal_weights=False):
    """Larger ring-plus-chords case (multi-tile at default tile size
    only when origins exceed 64 is NOT required; tiling is forced via
    tile_origins).  ``equal_weights`` makes all edge weights identical
    so distances tie pervasively."""
    rng = np.random.default_rng(seed)
    src = np.concatenate([np.arange(v, dtype=np.int64),
                          rng.integers(0, v, extra).astype(np.int64)])
    vec = np.concatenate([((np.arange(v) + 1) % v).astype(np.int64),
                          rng.integers(0, v, extra).astype(np.int64)])
    order = np.argsort(src, kind="stable")
    src, vec = src[order], vec[order]
    ptr = np.zeros(v + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(src, minlength=v))
    if equal_weights:
        wts = np.full(vec.shape[0], 0.5)
    else:
        wts = rng.uniform(0.1, 1.0, vec.shape[0])
    flag = np.ones(vec.shape[0], dtype=bool)
    oi = np.arange(origins, dtype=np.int64)
    o = np.stack([oi, (oi + 11) % v], axis=1).astype(np.int64)
    ow = np.full((origins, 2), 0.1)
    d = np.stack([rng.integers(0, v, d_count),
                  rng.integers(0, v, d_count)], axis=1).astype(np.int64)
    dw = rng.uniform(0.0, 0.5, (d_count, 2))
    dwt = rng.uniform(0.25, 1.0, d_count)
    return dict(o=o, ow=ow, ptr=ptr, vec=vec, wts=wts, flag=flag,
                d=d, dw=dw, dwt=dwt, cutoff=cutoff)


def run_pair(pair, case, **kw):
    """Run both routes over one case with the trace suite's positional
    law (d_weights is admission-only and last)."""
    args = (case["o"], case["ow"], case["ptr"], case["vec"], case["wts"],
            case["flag"], case["d"], case["dw"], case["dwt"], case["cutoff"])
    ref = pair[0](*args, **kw)
    nat = pair[1](*args, **kw)
    return ref, nat
