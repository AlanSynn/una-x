"""Device region wiring (G11): execute_fold_region through the twin port.

Scope law: these tests judge the WIRING — fail-closed order, per-row
device-fold flow, K3 composition order, cancellation, record counters —
with silicon replaced by the host twins.  Bit-equality claims here are
twin-vs-twin and prove composition order, not silicon parity; parity vs
the real device is exclusively the gate harness's durable verdict.

The twin runtime lives in test_device_gate_plumbing (same directory, no
package): import it from there so there is exactly one twin."""

import numpy as np
import pytest

from test_device_gate_plumbing import _TwinRuntime

from gpu.kernels.validate_device_bits import _COHORT_CASE
from gpu.runtime.errors import RegionCanceled
from gpu.runtime.region import (CancelFlag, FoldCell, FoldRegionRequest,
                                FoldRowScalars, execute_fold_region,
                                run_host_stage_composition)


class _FakeProbe:
    def __init__(self, available=True):
        self.available = available
        self.reason = "" if available else "no_cuda_driver"
        self.detail = "fake probe (test only)"
        self.device_count = 1 if available else 0
        self.fp64_capable = available
        self.device_names = ("fake",) if available else ()


def _scalars(**over):
    base = dict(cutoff=400.0, gravity_beta=0.001, gravity_plateau=0.0,
                gravity_logistic_midpoint=500.0, gravity_growth_rate=1.0,
                knn_decay="logistic", knn_weights=np.ones(4))
    base.update(over)
    return FoldRowScalars(**base)


def _gated_request(tmp_path, monkeypatch, rows, cell_id="wired-cell",
                   association="seq", **scalar_over):
    from gpu.backends.gpu import record_gate
    evidence = tmp_path / "ev.json"
    evidence.write_text('{"fixture": true}\n')
    store = tmp_path / "store.json"
    record_gate(str(store), cell_id, passed=True, association=association,
                evidence_ref=str(evidence))
    monkeypatch.setattr("gpu.runtime.region.probe_driver",
                        lambda: _FakeProbe())
    rng = np.random.default_rng(11)
    request = FoldRegionRequest(
        distance_rows=tuple(rows),
        d_weights=rng.random(rows[0].shape[0]) + 0.1,
        scalars=_scalars(**scalar_over),
        cell=FoldCell(cell_id=cell_id, association=association,
                      bit_validated=True),
        tile_origins=2, gate_store=str(store))
    return request


def test_device_wiring_matches_host_tail_bit_for_bit(tmp_path, monkeypatch):
    rows = [np.linspace(1.0, 900.0, 64),
            np.full(64, 50.0),                     # every element kept
            np.full(64, 5000.0)]                   # nothing kept (m == 0)
    request = _gated_request(tmp_path, monkeypatch, rows,
                             knn_decay="linear")
    rec = execute_fold_region(request, context=_TwinRuntime())
    host = run_host_stage_composition(
        tuple(rows), request.d_weights, request.scalars, "seq")
    for name in ("reach", "gravity_exponential", "gravity_logistic",
                 "knn_access"):
        got = getattr(rec, name)
        want = getattr(host, name)
        assert got.tobytes() == want.tobytes(), name
    # The nothing-kept row reduces to additive identity, not a crash.
    assert rec.reach[2] == 0.0
    assert rec.context_kind.startswith("device:")
    assert rec.staging_high_water_bytes > 0


def test_device_wiring_cohort_decay_logistic(tmp_path, monkeypatch):
    c = _COHORT_CASE
    rows = [np.asarray(c["distances"], dtype=np.float64)]
    request = _gated_request(tmp_path, monkeypatch, rows,
                             cutoff=float(c["cutoff"]),
                             gravity_beta=float(c["gravity_beta"]),
                             gravity_plateau=float(c["plateau"]),
                             gravity_logistic_midpoint=float(c["midpoint"]),
                             gravity_growth_rate=float(c["growth_rate"]),
                             knn_decay=c["knn_decay"])
    import numpy as _np
    request = FoldRegionRequest(
        distance_rows=request.distance_rows,
        d_weights=_np.asarray(c["weights"], dtype=_np.float64),
        scalars=request.scalars, cell=request.cell,
        tile_origins=request.tile_origins, gate_store=request.gate_store)
    rec = execute_fold_region(request, context=_TwinRuntime())
    host = run_host_stage_composition(
        request.distance_rows, request.d_weights, request.scalars, "seq")
    assert rec.reach.tobytes() == host.reach.tobytes()
    assert rec.knn_access.tobytes() == host.knn_access.tobytes()


def test_cancellation_refuses_at_second_tile_barrier(tmp_path, monkeypatch):
    rows = [np.linspace(float(i), float(i + 63), 64) for i in range(4)]
    request = _gated_request(tmp_path, monkeypatch, rows)
    cancel = CancelFlag()
    cancel.cancel()   # canceled BEFORE the first barrier
    with pytest.raises(RegionCanceled, match="tile barrier"):
        execute_fold_region(request, context=_TwinRuntime(), cancel=cancel)


def test_port_without_launch_surface_is_typed_refusal(tmp_path, monkeypatch):
    rows = [np.linspace(1.0, 100.0, 16)]
    request = _gated_request(tmp_path, monkeypatch, rows)
    with pytest.raises(TypeError, match="device context port lacks"):
        execute_fold_region(request, context=object())


def test_tampered_evidence_closes_device_dispatch(tmp_path, monkeypatch):
    # End to end at the WIRED layer: evidence bytes changing after the
    # verdict closes the cell even though the FoldCell object is carried.
    from gpu.backends.gpu import record_gate
    evidence = tmp_path / "ev.json"
    evidence.write_text('{"fixture": true}\n')
    store = tmp_path / "store.json"
    record_gate(str(store), "tamper-wired", passed=True, association="seq",
                evidence_ref=str(evidence))
    evidence.write_text('{"fixture": true, "tampered": 1}\n')
    monkeypatch.setattr("gpu.runtime.region.probe_driver",
                        lambda: _FakeProbe())
    request = FoldRegionRequest(
        distance_rows=(np.linspace(1.0, 100.0, 16),),
        d_weights=np.ones(16), scalars=_scalars(),
        cell=FoldCell(cell_id="tamper-wired", association="seq",
                      bit_validated=True),
        tile_origins=2, gate_store=str(store))
    from gpu.runtime.errors import ParityGateNotRun
    with pytest.raises(ParityGateNotRun, match="gate store does not open"):
        execute_fold_region(request, context=_TwinRuntime())
