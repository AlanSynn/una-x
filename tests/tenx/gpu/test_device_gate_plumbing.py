"""Gate-chain PLUMBING test: run_device_gate end to end with silicon
replaced by the host twins.

Scope law: this judges WIRING ONLY — call arity, kernel param plumbing,
report shape, evidence/store recording, exit codes.  It exists because
attempts 1-3 class defects (wchar read, wrapper kwarg name, a 7-vs-6-arg
call) were all unreachable on device-less hosts and each cost a real GPU
job to discover.  A green result here is NOT a gate verdict and never
touches a durable store outside tmp_path; parity vs silicon is judged
exclusively by the real device gate on the qualification node."""

import json

import numpy as np
import pytest

import gpu.kernels.device_exec as dx
import gpu.kernels.validate_device_bits as vdb
from gpu.backends.gpu import gate_state
from gpu.kernels.device_exec import DeviceIdentity
from gpu.kernels.host_twins import (REDUCTION_TREES, fold_host_reference,
                                    k0_compact, k1_gravity_exponential_coeffs,
                                    k1_gravity_logistic_coeffs, k1_knn_coeffs)
from gpu.kernels.validate_device_bits import (EXIT_HOST_BATTERY,
                                              EXIT_PASSED, main)


class _TwinRuntime:
    """DeviceRuntime stand-in: buffers live in a dict keyed by int handle;
    every launch executes the corresponding host twin on stored arrays."""

    def __init__(self, device_index: int = 0):
        self._bufs = {}
        self._next = 1
        self.identity = DeviceIdentity(
            name="TWIN (plumbing only; not silicon)",
            compute_capability=(8, 0), sm_name="sm_80",
            driver_version=(12, 9), device_index=0)

    def h2d(self, arr):
        handle = self._next
        self._next += 1
        self._bufs[handle] = np.ascontiguousarray(arr).copy()
        return handle

    def d2h(self, dev, like):
        out = np.empty_like(like)
        flat = out.reshape(-1)
        flat[:] = self._bufs[dev].reshape(-1)[:flat.size]
        return out

    def sync(self):
        pass

    def close(self):
        self._bufs.clear()

    def load_ptx(self, ptx_path):
        text = ptx_path.read_text()
        for k in ("una_g10_fold_compact",
                  "una_g10_fold_coefficients_exp",
                  "una_g10_fold_knn_coeffs_linear"):
            assert k in text

    def launch(self, kname, grid, block, params, shared_bytes=0):
        vals = [p if isinstance(p, int) else p.value for p in params]
        if kname == "una_g10_fold_compact":
            dist, count, cutoff, oidx, ocnt = vals
            d = self._bufs[dist].reshape(-1)
            idx, n = k0_compact(d, cutoff)
            buf = np.zeros(d.shape[0], dtype=np.int64)
            buf[:n] = idx
            self._bufs[oidx] = buf.reshape(1, -1)
            self._bufs[ocnt] = np.array([n], dtype=np.uint64)
        elif "coefficients_exp" in kname:
            dn = self._bufs[vals[0]].reshape(-1)
            wn = self._bufs[vals[1]].reshape(-1)
            self._bufs[vals[-1]] = k1_gravity_exponential_coeffs(
                dn, wn, vals[3], vals[4])
        elif "coefficients_log" in kname:
            dn = self._bufs[vals[0]].reshape(-1)
            wn = self._bufs[vals[1]].reshape(-1)
            self._bufs[vals[-1]] = k1_gravity_logistic_coeffs(
                dn, wn, vals[3], vals[4], vals[5])
        elif "knn_coeffs_exp" in kname:
            dn_s = self._bufs[vals[0]].reshape(-1)
            wn_s = self._bufs[vals[1]].reshape(-1)
            kn = self._bufs[vals[2]].reshape(-1)
            self._bufs[vals[-1]] = k1_knn_coeffs(
                dn_s, wn_s, kn, "exponential", vals[4], 1.0, vals[5], 0.0)
        elif "knn_coeffs_log" in kname:
            dn_s = self._bufs[vals[0]].reshape(-1)
            wn_s = self._bufs[vals[1]].reshape(-1)
            kn = self._bufs[vals[2]].reshape(-1)
            self._bufs[vals[-1]] = k1_knn_coeffs(
                dn_s, wn_s, kn, "logistic", 0.0, vals[4], vals[5], vals[6])
        elif "knn_coeffs_linear" in kname:
            wn_s = self._bufs[vals[0]].reshape(-1)
            kn = self._bufs[vals[1]].reshape(-1)
            self._bufs[vals[-1]] = k1_knn_coeffs(
                np.empty(0), wn_s, kn, "linear", 0.0, 0.0, 0.0, 0.0)
        else:  # pragma: no cover - unknown kernel must not pass silently
            raise AssertionError(f"twin runtime: unknown kernel {kname}")


def _fake_engine(d, w, cutoff, beta, plateau, midpoint, _x1, kn_w, _x2,
                 decay, _x3, _x4, _x5):
    """Engine stand-in: the host reference under a FIXED tree (seq).  The
    point is a live consumer of the device-produced coefficient arrays, so
    leg B's tree-hit logic is exercised end to end — not engine parity."""
    r, ge, gl, kn = fold_host_reference(
        np.asarray(d, dtype=np.float64), np.asarray(w, dtype=np.float64),
        float(cutoff), float(beta), float(plateau), float(midpoint),
        1.0, decay, kn_w, "seq")
    return float(r), float(ge), float(gl), float(kn)


@pytest.fixture()
def twin_gate(monkeypatch, tmp_path):
    ptx = tmp_path / "fake.ptx"
    ptx.write_text(
        "// twin stand-in PTX; load_ptx asserts the kernel names:\n"
        "// una_g10_fold_compact una_g10_fold_coefficients_exp\n"
        "// una_g10_fold_coefficients_log una_g10_fold_knn_coeffs_exp\n"
        "// una_g10_fold_knn_coeffs_log una_g10_fold_knn_coeffs_linear\n")

    class _Nvcc:
        on_path = True
        version = "twin-fake"
        detail = "/twin/fake/nvcc"

    from gpu.runtime.availability import DriverProbe
    monkeypatch.setattr(vdb, "probe_driver", lambda: DriverProbe(
        available=True, reason="available",
        detail="twin fixture (no hardware behind this probe)",
        device_count=1, fp64_capable=True, device_names=("TWIN",)))
    monkeypatch.setattr(dx, "DeviceRuntime", _TwinRuntime)
    monkeypatch.setattr(dx, "compile_ptx",
                        lambda src, arch, path, out: ptx)
    monkeypatch.setattr(vdb, "detect_nvcc", lambda: _Nvcc())
    monkeypatch.setattr(vdb, "_load_engine", lambda: _fake_engine)
    return tmp_path


def test_gate_chain_wiring_reaches_passed_store(twin_gate):
    run = twin_gate
    rc = main(["--gate-store", str(run / "store.json"),
               "--evidence", str(run / "evidence.json")])
    assert rc == EXIT_PASSED
    state = gate_state("una_legacy.fold.default", str(run / "store.json"))
    assert state["open"] is True
    assert state["association"] in REDUCTION_TREES
    evidence = json.loads((run / "evidence.json").read_text())
    assert evidence["status"] == "gate_passed"
    assert set(evidence["k0_battery"]) >= {
        "k0_random_row_a", "k0_random_row_b", "k0_nan_inf_mix", "k0_all_kept"}
    assert all(v["k0_ok"] for v in evidence["k0_battery"].values())
    assert evidence["exp_boundary"]["all_equal"] is True
    assert evidence["tie_law"]["mismatches"] == 0
    for name, res in evidence["fold_battery"].items():
        assert res["leg_a_ok"], f"{name}: leg A device-vs-twin failed"
        assert res["all_outputs_certified"], f"{name}: leg B unclosed"


def test_gate_chain_refusal_path_records_closed_cell(twin_gate, monkeypatch):
    # Force a leg-A mismatch by corrupting one twin result in flight: the
    # refused verdict must be recorded durably and the store must stay
    # closed (bit_validated=False, no association).
    run = twin_gate
    orig = _TwinRuntime.launch

    def poisoned(self, kname, grid, block, params, shared_bytes=0):
        orig(self, kname, grid, block, params, shared_bytes)
        if "coefficients_exp" in kname:
            handle = params[-1] if isinstance(params[-1], int) \
                else params[-1].value
            self._bufs[handle] = np.full(self._bufs[handle].shape, np.nan)
    monkeypatch.setattr(_TwinRuntime, "launch", poisoned)
    rc = main(["--gate-store", str(run / "store2.json"),
               "--evidence", str(run / "evidence2.json")])
    assert rc == 4  # EXIT_BATTERY_MISMATCH: designed refusal, recorded
    state = gate_state("una_legacy.fold.default", str(run / "store2.json"))
    assert state["open"] is False
    assert "bit_validated=False" in state["reason"]
    evidence = json.loads((run / "evidence2.json").read_text())
    assert evidence["status"] == "battery_mismatch"


def test_host_battery_dispatch_still_exits_5(twin_gate, tmp_path):
    out = tmp_path / "hb.json"
    rc = main(["--emit-host-battery", str(out)])
    assert rc == EXIT_HOST_BATTERY == 5
    battery = json.loads(out.read_text())
    assert set(battery) >= {"magnitude_cases", "cohort_case", "tie_law"}
