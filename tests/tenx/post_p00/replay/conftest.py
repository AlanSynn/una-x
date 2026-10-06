"""Fixtures for the P03C compiled-replay suite (campaign una_10x
post_p00, designs/03).

Reuses the C11 trace battery machinery verbatim (admitted cases,
build/replay helpers, bit comparators) by importlib-loading
``tests/tenx/trace/conftest.py`` from its bytes — the same pattern that
conftest itself uses to load the S11 battery.  The third parity arm is
the frozen pre-P03C Python replay (``legacy_replay_reference.py``), a
byte-faithful copy of ``replay_trace`` at commit a7281c8, also loaded
from its bytes.  NUMBA_CACHE_DIR follows the C11 owner-owned scratch
convention: ``cache=True`` jitting never writes beside the sources.
"""
import dataclasses
import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

# Owner-owned scratch (C11 convention): engine modules must jit into
# the campaign's own cache directory, never into src/ __pycache__.
os.environ["NUMBA_CACHE_DIR"] = str(
    Path(__file__).resolve().parents[4] / "campaign_runs"
    / "_numba_cache_owner")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

_TENX = Path(__file__).resolve().parents[2]

_SPEC = importlib.util.spec_from_file_location(
    "p03c_trace_conftest", _TENX / "trace" / "conftest.py")
tck = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tck)

_LSPEC = importlib.util.spec_from_file_location(
    "p03c_legacy_reference",
    Path(__file__).resolve().parent / "legacy_replay_reference.py")
_legacy = importlib.util.module_from_spec(_LSPEC)
_LSPEC.loader.exec_module(_legacy)

# Re-exported C11 battery helpers (single source, no drift).
admitted_case = tck.admitted_case
build_case = tck.build_case
case_kwargs = tck.case_kwargs
driver_args = tck.driver_args
CASE_PARAMS = tck.CASE_PARAMS
RESULT_FIELDS = tck.RESULT_FIELDS
METRIC_PARAMS = tck.METRIC_PARAMS
bits = tck.bits
result_equal = tck.result_equal
legacy_replay_trace = _legacy.legacy_replay_trace

# The suite metric bundle (C11 replay() law): gravity_beta 1.5,
# plateau 2.0, logistic midpoint 3.0, growth rate 0.8, KNN weights
# METRIC_PARAMS[4].
BUNDLE = (1.5, 2.0, 3.0, 0.8)


@pytest.fixture
def tr():
    from urban_network_analysis.Engines import _trace_access_scratch
    return _trace_access_scratch


@pytest.fixture
def baseline():
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        integrated_scope_access)
    return integrated_scope_access


@pytest.fixture
def legacy():
    return legacy_replay_trace


def replay_live(trm, rec, case, knn_decay="exponential", d_weights=None,
                cutoff=None):
    """replay_trace on the LIVE (compiled) path with the bundle."""
    return trm.replay_trace(
        rec, case["dwt"] if d_weights is None else d_weights,
        BUNDLE[0], BUNDLE[1], BUNDLE[2], BUNDLE[3], knn_decay,
        METRIC_PARAMS[4],
        case["cutoff"] if cutoff is None else cutoff)


def replay_legacy(rec, case, knn_decay="exponential", d_weights=None,
                  cutoff=None):
    """replay_trace on the FROZEN pre-P03C bytes with the bundle."""
    return legacy_replay_trace(
        rec, case["dwt"] if d_weights is None else d_weights,
        BUNDLE[0], BUNDLE[1], BUNDLE[2], BUNDLE[3], knn_decay,
        METRIC_PARAMS[4],
        case["cutoff"] if cutoff is None else cutoff)


def first_difference(res_a, res_b, fields=RESULT_FIELDS):
    """First bitwise difference between two replay results.

    Debug law (designs/03): retain the first differing origin, the
    field, and both parties' bits.  None means bitwise equality.
    """
    for f in fields:
        xa = np.asarray(getattr(res_a, f))
        xb = np.asarray(getattr(res_b, f))
        if xa.tobytes() != xb.tobytes():
            if xa.shape != xb.shape:
                return {"field": f, "origin": None,
                        "shape_a": xa.shape, "shape_b": xb.shape,
                        "bits_a": xa.tobytes()[:32].hex(),
                        "bits_b": xb.tobytes()[:32].hex()}
            for i in range(xa.shape[0]):
                if xa[i].tobytes() != xb[i].tobytes():
                    return {"field": f, "origin": i,
                            "bits_a": xa[i].tobytes().hex(),
                            "bits_b": xb[i].tobytes().hex()}
    return None


def assert_three_way(new, old, ref=None):
    """Compiled replay must be bitwise the frozen legacy replay (the
    pre-change bytes) and, when given, the integrated engine run."""
    assert new.route == old.route == "trace_replay"
    assert new.refusal == old.refusal == ""
    assert new.folds_done == old.folds_done
    assert first_difference(new, old) is None, \
        first_difference(new, old)
    assert first_difference(old, new) is None
    if ref is not None:
        items = [getattr(ref, f, None) for f in RESULT_FIELDS]
        if any(v is None for v in items):
            items = list(ref)
        for f, g in zip(RESULT_FIELDS, items):
            assert bits(getattr(new, f)) == bits(g), f


def synth_record(trm, slices, dists, dest_count, d_weights=None,
                 cutoff=25.0, trace_id="p03c-synthetic"):
    """A COMPLETE synthetic TraceRecord over explicit per-origin slices.

    ``slices`` is a list of ascending original-id lists (may be empty);
    ``dists`` is the packed float64 distance array (len == total nnz).
    The packed bytes are digest-bound exactly as the producer binds
    them, so the record passes every packed gate honestly.
    """
    origin_count = len(slices)
    indptr = np.zeros(origin_count + 1, np.int64)
    np.cumsum([len(s) for s in slices], out=indptr[1:])
    if indptr[-1]:
        ids_all = np.concatenate(
            [np.asarray(s, np.int64) for s in slices])
    else:
        ids_all = np.zeros(0, np.int64)
    dist_all = np.asarray(dists, np.float64)
    assert dist_all.shape[0] == ids_all.shape[0]
    digest = hashlib.sha256(indptr.tobytes() + ids_all.tobytes()
                            + dist_all.tobytes()).hexdigest()
    return trm.TraceRecord(
        trace_id=trace_id, schema_version=trm.TRACE_SCHEMA_VERSION,
        origin_count=origin_count, dest_count=dest_count,
        cutoff_value=float(cutoff), cutoff_dtype="float64",
        reach_dtype="int64", metric_dtype="float64",
        indptr=trm._read_only(indptr),
        destination_ids=trm._read_only(ids_all),
        distance_bits=trm._read_only(dist_all),
        complete=True, route=trm.ROUTE_TRACE, refusal="",
        counters=trm.TraceCounters(
            origins=origin_count, tiles=0, tile_origins=0,
            candidates_total=int(ids_all.shape[0]), touched_total=0,
            indptr_bytes=indptr.nbytes, ids_bytes=ids_all.nbytes,
            dist_bytes=dist_all.nbytes, metadata_bytes=0,
            trace_bytes=ids_all.nbytes * 8 + dist_all.nbytes * 8,
            tile_temp_bytes=0, stamps_zeroed=0),
        content_digest=digest)


def synth_weights(dest_count, seed=7):
    """Fractional destination metric weights (exercises the reach
    int64 truncation on assignment) in the dossier-02 value domain."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.25, 2.75, dest_count)
    w[0] = 1.5  # deterministic fractional lead
    return w


def retamper(trm, rec, mutate):
    """A forged record whose packed bytes were mutated by ``mutate``
    (operating on writable copies) and whose content_digest honestly
    binds the TAMPERED bytes — so the record passes the digest gate and
    reaches the gates AFTER it (structure/ids/slice-order)."""
    indptr = np.array(rec.indptr)
    ids_all = np.array(rec.destination_ids)
    dist_all = np.array(rec.distance_bits)
    mutate(indptr, ids_all, dist_all)
    digest = hashlib.sha256(indptr.tobytes() + ids_all.tobytes()
                            + dist_all.tobytes()).hexdigest()
    return dataclasses.replace(
        rec, indptr=trm._read_only(indptr),
        destination_ids=trm._read_only(ids_all),
        distance_bits=trm._read_only(dist_all),
        content_digest=digest)
