"""Region fail-closed ladder, cancellation contract, and host-tail
composition.  Refusal order is contractual: validate -> probe -> gate ->
context; every refusal precedes any compute."""

import numpy as np
import pytest

import gpu.runtime.region as region
from gpu.runtime.errors import HardwareUnavailable, ParityGateNotRun
from gpu.runtime.region import (CancelFlag, FoldCell, FoldRegionRequest,
                                FoldRowScalars, execute_fold_region,
                                plan_tiles, run_host_stage_composition,
                                validate_request)


class _FakeProbe:
    def __init__(self, available):
        self.available = available
        self.reason = "" if available else "no_cuda_driver"
        self.detail = "fake probe (test only)"
        self.device_count = 1 if available else 0
        self.fp64_capable = available
        self.device_names = ("fake",) if available else ()


def _scalars(**over):
    base = dict(cutoff=400.0, gravity_beta=0.001, gravity_plateau=0.0,
                gravity_logistic_midpoint=500.0, gravity_growth_rate=1.0,
                knn_decay="logistic", knn_weights=np.array([1.0]))
    base.update(over)
    return FoldRowScalars(**base)


def _request(rows=2, d=8, cell=None):
    rng = np.random.default_rng(2)
    return FoldRegionRequest(
        distance_rows=tuple(rng.random(d) * 500.0 for _ in range(rows)),
        d_weights=rng.random(d) + 0.1,
        scalars=_scalars(),
        cell=cell or FoldCell(cell_id="c0"),
        tile_origins=2)


def test_validation_refusals_precede_hardware_probe(monkeypatch):
    # Even with a (faked) available device, a malformed request must fail
    # at validation — refusal order is part of the contract.
    monkeypatch.setattr(region, "probe_driver",
                        lambda: _FakeProbe(available=True))
    bad = FoldRegionRequest(
        distance_rows=(np.array([1.0, 2.0], dtype=np.float32),),
        d_weights=np.ones(2), scalars=_scalars(),
        cell=FoldCell(cell_id="c0"))
    with pytest.raises(ValueError,
                       match=r"distance_rows\[0\] must be float64"):
        execute_fold_region(bad)


def test_no_device_refuses_hardware_unavailable():
    # This host has no CUDA device (G00 audit); the REAL probe refuses.
    with pytest.raises(HardwareUnavailable, match="no_cuda_driver"):
        execute_fold_region(_request())


def test_gate_refusal_precedes_context(monkeypatch):
    monkeypatch.setattr(region, "probe_driver",
                        lambda: _FakeProbe(available=True))
    with pytest.raises(ParityGateNotRun, match="bit-validation gate"):
        execute_fold_region(_request(), context=object())


def test_forged_cell_cannot_bypass_gate_store(monkeypatch):
    # A caller-constructed cell object with bit_validated=True is not
    # sufficient: the gate STORE is authoritative (only the device-side
    # gate harness opens cells).
    monkeypatch.setattr(region, "probe_driver",
                        lambda: _FakeProbe(available=True))
    forged = FoldCell(cell_id="never-gated", association="seq",
                      bit_validated=True)
    with pytest.raises(ParityGateNotRun, match="gate store"):
        execute_fold_region(_request(cell=forged), context=object())


def test_context_refusal_after_gate(monkeypatch):
    from gpu.backends.gpu import cell_gate
    cell_gate("gated-cell", passed=True, association="seq",
              evidence_ref="test fixture")
    monkeypatch.setattr(region, "probe_driver",
                        lambda: _FakeProbe(available=True))
    cell = FoldCell(cell_id="gated-cell", association="seq",
                    bit_validated=True)
    with pytest.raises(HardwareUnavailable, match="device context"):
        execute_fold_region(_request(cell=cell), context=None)


def test_all_gates_pass_stop_is_designed_not_fallback(monkeypatch):
    from gpu.backends.gpu import cell_gate
    cell_gate("gated-cell2", passed=True, association="seq",
              evidence_ref="test fixture")
    monkeypatch.setattr(region, "probe_driver",
                        lambda: _FakeProbe(available=True))
    cell = FoldCell(cell_id="gated-cell2", association="seq",
                    bit_validated=True)
    with pytest.raises(NotImplementedError, match="G11"):
        execute_fold_region(_request(cell=cell), context=object())


def test_cancel_flag_contract():
    flag = CancelFlag()
    assert flag.canceled() is False
    flag.cancel()
    assert flag.canceled() is True


def test_plan_tiles_bounds():
    assert plan_tiles(5, 2) == ((0, 2), (2, 4), (4, 5))
    assert plan_tiles(0, 4) == ()
    with pytest.raises(ValueError):
        plan_tiles(5, 0)


def test_host_tail_composition_matches_twin_reference():
    rng = np.random.default_rng(9)
    rows = tuple(rng.random(16) * 500.0 for _ in range(3))
    w = rng.random(16) + 0.1
    rec = run_host_stage_composition(rows, w, _scalars(), "seq")
    assert rec.origins == 3 and rec.tiles == 1
    assert rec.hardware_validated is False
    assert "not device execution" in rec.context_kind
    assert rec.reach.shape == (3,)


def test_records_never_claim_hardware_validation():
    rec = run_host_stage_composition(
        (np.array([1.0, 2.0]),), np.array([1.0, 1.0]), _scalars(), "seq")
    assert rec.hardware_validated is False


def test_tiny_positive_budget_passes_validation_without_preallocating():
    # Validation must not allocate: a 1-byte budget is constructible and
    # valid; the staging pool refuses at ALLOCATION time (see
    # test_buffer_ownership.py::test_staging_pool_bounds_and_high_water).
    req = FoldRegionRequest(
        distance_rows=(np.array([1.0, 2.0]),), d_weights=np.ones(2),
        scalars=_scalars(), cell=FoldCell(cell_id="c"),
        staging_budget_bytes=1)
    validate_request(req)
