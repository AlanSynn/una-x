"""C20-MAJOR-2 integration pin (campaign una_10x): a REAL build_trace
record flows through the residency with its OWN identity — no int
surrogate, no retyped fields.

The independent review's probe (rv6) proved the chain works after the
repair; this pin commits it: publish the record's true meta (string
schema version, the record's own bound content digest), fault it back
in through a FRESH residency over the same store, reconstruct the
record and replay it — bitwise identical to direct replay of the
original record.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from conftest import (METRIC_PARAMS, build, packed_equal,
                      result_equal, ring_case)  # trace-suite helpers

from urban_network_analysis.Engines import _trace_access_scratch as trm
from urban_network_analysis.Execution import CacheOptions
from urban_network_analysis.cache.trace_residency import (
    TraceResidency, packed_digest)
from urban_network_analysis.cache.store import ContentStore

PRODUCER = hashlib.sha256(
    (REPO_SRC / "urban_network_analysis" / "Engines"
     / "_trace_access_scratch.py").read_bytes()).hexdigest()


def _store(tmp_path):
    return ContentStore(CacheOptions(
        mode="disk", directory=str(tmp_path / "store")))


def _real_meta(rec):
    """The record's OWN identity fields, verbatim (rv6 mapping)."""
    return {"trace_id": rec.trace_id,
            "trace_schema_version": rec.schema_version,
            "dest_count": int(rec.dest_count),
            "cutoff_bits": np.float64(rec.cutoff_value).tobytes().hex(),
            "cutoff_dtype": rec.cutoff_dtype,
            "reach_dtype": rec.reach_dtype,
            "metric_dtype": rec.metric_dtype}


def test_real_trace_record_publishes_faults_in_and_replays_bitwise(
        tmp_path):
    tr = trm
    case = ring_case()
    rec = build(tr, case)
    assert rec.complete
    # the record's bound digest is the honest function of its bytes
    assert rec.content_digest == packed_digest(
        np.asarray(rec.indptr).tobytes(),
        np.asarray(rec.destination_ids).tobytes(),
        np.asarray(rec.distance_bits).tobytes())

    res1 = TraceResidency(_store(tmp_path), max_resident_bytes=1 << 22,
                          producer_sha256=PRODUCER)
    key = res1.publish(np.asarray(rec.indptr),
                       np.asarray(rec.destination_ids),
                       np.asarray(rec.distance_bits),
                       _real_meta(rec),
                       bound_digest=rec.content_digest)
    res1.close()

    # resume: a FRESH residency over the same store
    res2 = TraceResidency(_store(tmp_path), max_resident_bytes=1 << 22,
                          producer_sha256=PRODUCER)
    assert res2.resident_bytes() == 0
    lease = res2.acquire(key)
    assert lease is not None
    ip, ids, dist = lease.arrays()
    try:
        rebuilt = tr.TraceRecord(
            trace_id=rec.trace_id, schema_version=rec.schema_version,
            origin_count=rec.origin_count, dest_count=rec.dest_count,
            cutoff_value=rec.cutoff_value, cutoff_dtype=rec.cutoff_dtype,
            reach_dtype=rec.reach_dtype, metric_dtype=rec.metric_dtype,
            indptr=ip, destination_ids=ids, distance_bits=dist,
            complete=True, route=rec.route, refusal="",
            counters=rec.counters, content_digest=rec.content_digest)
        assert packed_equal(rebuilt, rec)
        replayed = trm.replay_trace(
            rebuilt, case["dwt"], 1.5, 2.0, 3.0, 0.8, "exponential",
            METRIC_PARAMS[4], case["cutoff"])
        direct = trm.replay_trace(
            rec, case["dwt"], 1.5, 2.0, 3.0, 0.8, "exponential",
            METRIC_PARAMS[4], case["cutoff"])
        assert result_equal(replayed, direct)
    finally:
        lease.release()

    # the stored identity is the TRUE schema string, never a surrogate
    stored = res2._entries[key]["meta"]["trace_schema_version"]
    assert stored == rec.schema_version == trm.TRACE_SCHEMA_VERSION


def test_real_record_rejects_wrong_bound_digest(tmp_path):
    tr = trm
    rec = build(trm, ring_case())
    res = TraceResidency(_store(tmp_path), max_resident_bytes=1 << 22,
                         producer_sha256=PRODUCER)
    with pytest.raises(Exception) as ei:
        res.publish(np.asarray(rec.indptr),
                    np.asarray(rec.destination_ids),
                    np.asarray(rec.distance_bits),
                    _real_meta(rec),
                    bound_digest="0" * 64)
    assert "digest does not match" in str(ei.value)
    assert res.counters["publishes"] == 0
