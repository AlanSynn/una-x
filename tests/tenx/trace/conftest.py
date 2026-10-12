"""Fixtures for the C11 trace suite (campaign una_10x).

The case battery is imported verbatim from its single sources (the
cpu_kernels battery via the S11 sparse conftest).  The preserved packet
reference chain (``campaigns/una_10x/prior/una_10x_investigation/
probe.py``) is loaded from its IMMUTABLE bytes by importlib — never
retyped — with its runtime sha256 recorded; the numba cache directory
is forced to the owner-owned scratch so ``cache=True`` jitting can
never write into the immutable packet.
"""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Force (not setdefault): the packet and every cached engine module
# must jit into owner-owned scratch, never into prior/ or site data.
os.environ["NUMBA_CACHE_DIR"] = str(
    Path(__file__).resolve().parents[3] / "campaign_runs"
    / "_numba_cache_owner")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

_TENX = Path(__file__).resolve().parents[1]
_CK_PATH = _TENX.parents[0] / "cpu_kernels" / "conftest.py"
_spec = importlib.util.spec_from_file_location(
    "cpu_kernels_conftest", _CK_PATH)
ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck)

_S11_PATH = _TENX / "sparse" / "conftest.py"
_s11spec = importlib.util.spec_from_file_location(
    "sparse_conftest", _S11_PATH)
s11 = importlib.util.module_from_spec(_s11spec)
_s11spec.loader.exec_module(s11)

build_case = ck.build_case
CASE_PARAMS = ck.CASE_PARAMS
case_kwargs = ck.case_kwargs
bits = ck.bits

admitted_case = s11.admitted_case
driver_args = s11.driver_args
metric_args = s11.metric_args
RESULT_FIELDS = s11.RESULT_FIELDS
METRIC_PARAMS = s11.METRIC_PARAMS

PROBE_PATH = (Path(__file__).resolve().parents[3] / "campaigns"
              / "una_10x" / "prior" / "una_10x_investigation"
              / "probe.py")


def load_probe():
    """Importlib-load the preserved reference chain from its bytes.

    The module is registered in sys.modules under its own name AND
    under the literal name the owner numba cache recorded for these
    chain functions ('<dynamic>', from the earlier probe runner):
    on a cache hit numba rebuilds the recorded environment via
    importlib, and the cache contract (keyed by bytecode, types,
    compiler and CPU environment) then yields the same bits a fresh
    compile would.  No cached bytes are rewritten or removed.
    """
    raw = PROBE_PATH.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    name = "una10x_probe_reference"
    spec = importlib.util.spec_from_file_location(name, PROBE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.modules.setdefault("<dynamic>", mod)
    spec.loader.exec_module(mod)
    mod._runtime_sha256 = sha
    return mod


def reference_candidates(prb, case):
    """Per-origin (candidate ids, distance bits) from the preserved
    chain: make_reverse_index -> epoch_search -> local_candidates ->
    collect_local.

    Fresh single-lane arenas per origin (the probe's local_driver
    calling convention; epoch = o+1, no cross-origin state).
    Returns lists of (np.array ascending int64, np.array float64).
    """
    ptr = case["ptr"]
    v = int(ptr.shape[0]) - 1
    d_count = int(case["d"].shape[0])
    deg = int(np.max(ptr[1:] - ptr[:-1])) if v else 0
    eo = np.empty(deg, np.int64)
    ew = np.empty(deg, np.float64)
    ip, rids = prb.make_reverse_index(v, case["d"])
    out = []
    for o in range(int(case["o"].shape[0])):
        epoch = o + 1
        labels = np.empty(v, np.float64)
        stamp = np.zeros(v, np.int64)
        touched = np.empty(v, np.int64)
        marks = np.zeros(d_count, np.int64)
        scratch = np.empty(d_count, np.int64)
        nt = int(prb.epoch_search(
            case["o"][o], case["ow"][o], ptr, case["vec"], case["wts"],
            case["flag"], case["cutoff"], eo, ew, labels, stamp, epoch,
            touched))
        cands = prb.local_candidates(touched, nt, ip, rids, marks,
                                     epoch, scratch)
        ds, _ws = prb.collect_local(cands, labels, stamp, epoch,
                                    case["cutoff"], case["d"],
                                    case["dw"], case["dwt"])
        out.append((np.asarray(cands), np.asarray(ds)))
    return out


def ring_case(v=400, extra=300, origins=40, d_count=100, cutoff=8.0,
              seed=23):
    """A medium ring graph with random extra arcs, multi-origin and
    multi-destination (beyond the toy battery; admitted domain)."""
    rng = np.random.default_rng(seed)
    src = np.concatenate([np.arange(v, dtype=np.int64),
                          rng.integers(0, v, extra).astype(np.int64)])
    vec = np.concatenate(
        [((np.arange(v) + 1) % v).astype(np.int64),
         rng.integers(0, v, extra).astype(np.int64)])
    order = np.argsort(src, kind="stable")
    src, vec = src[order], vec[order]
    ptr = np.zeros(v + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(src, minlength=v))
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


@pytest.fixture
def tr():
    from urban_network_analysis.Engines import _trace_access_scratch
    return _trace_access_scratch


@pytest.fixture
def sp():
    from urban_network_analysis.Engines import _sparse_access_scratch
    return _sparse_access_scratch


@pytest.fixture
def baseline():
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        integrated_scope_access)
    return integrated_scope_access


@pytest.fixture
def probe():
    return load_probe()


def build(trm, case):
    """build_trace over an admitted battery case (positional law)."""
    return trm.build_trace(
        case["o"], case["ow"], case["ptr"], case["vec"], case["wts"],
        case["flag"], case["d"], case["dw"], case["dwt"],
        case["cutoff"])


def replay(trm, rec, case, knn_decay="exponential", d_weights=None,
           cutoff=None):
    """replay_trace over a built record with the suite metric bundle."""
    return trm.replay_trace(
        rec, case["dwt"] if d_weights is None else d_weights,
        1.5, 2.0, 3.0, 0.8, knn_decay, METRIC_PARAMS[4],
        case["cutoff"] if cutoff is None else cutoff)


def packed_equal(a, b):
    """Bytewise equality of the three packed trace arrays."""
    return (np.asarray(a.indptr).tobytes()
            == np.asarray(b.indptr).tobytes()
            and np.asarray(a.destination_ids).tobytes()
            == np.asarray(b.destination_ids).tobytes()
            and np.asarray(a.distance_bits).tobytes()
            == np.asarray(b.distance_bits).tobytes())


def result_equal(res, ref):
    """Bitwise equality of the four fold outputs against either a
    baseline 4-tuple or an attribute-bearing result object."""
    items = [getattr(ref, f, None) for f in RESULT_FIELDS]
    if any(v is None for v in items):
        items = list(ref)
    return all(bits(getattr(res, f)) == bits(g)
               for f, g in zip(RESULT_FIELDS, items))
