"""Per-cell bit-validation gate harness for the fold region (G11).

THE gate from gpu_region.json ("device-bit-validation"): before any cell
may dispatch K1/K3 on a device, the DEVICE's coefficients and certified
reduction trees must reproduce the pinned una_legacy fold bits on this
characterization battery, ON THAT DEVICE, with the evidence recorded.

Exit-code contract (designed refusals are reported, never faked):
  0   gate PASSED on a real device — evidence JSON written AND recorded in
      the durable gate store (backends.gpu.record_gate); the cell is open
  2   hardware_unavailable — no CUDA-class FP64 device (designed refusal)
  3   device present but the toolchain/context is unusable (nvcc missing,
      compile failure, driver-API failure) — no verdict is recorded
  4   battery mismatch — device bits != pinned bits, no certified
      association for some output, or tie-law drift vs the installed
      numba build (the gate doing its job; the cell stays CLOSED and the
      refusal is durably recorded)
  5   --emit-host-battery: the pinned host expected-bits table was
      written; NO device claim and NO gate verdict (G10R F4: 5 is
      deliberately distinct from the gate-passed 0)

Modes:
  --emit-host-battery PATH
      Host half only: runs the ENGINE fold (the parity artifact,
      Engines.AccessibilityWElevation.reach_gravity_knn_access) on this
      host over the characterization battery and writes the pinned
      expected-bits table plus, per case, which K3 association trees
      reproduce the engine bits.  This imports the engine and JIT-compiles
      it: pin a fresh NUMBA_CACHE_DIR and PYTHONDONTWRITEBYTECODE=1.
  (default, no flags)
      Full device gate: probe -> compile the COMMITTED
      gpu/kernels/fold_kernels.cu (contractual flags, arch from the
      device) -> K0/K1/K2/K3 battery on the device -> engine-parity
      certification on this host's build -> durable gate-store record +
      evidence JSON.
      Options: --gate-store PATH (default $UNA_GPU_GATE_STORE; required
      for a pass — a verdict with nowhere durable to live is refused),
      --evidence PATH (default: beside the gate store), --cell ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..runtime.availability import detect_nvcc, probe_driver

EXIT_PASSED = 0
EXIT_HARDWARE_UNAVAILABLE = 2
EXIT_TOOLCHAIN_UNUSABLE = 3
EXIT_BATTERY_MISMATCH = 4
EXIT_HOST_BATTERY = 5

# Characterization magnitude cases, verbatim from the committed instrument
# campaign_runs/una_10x/tenx-20261005T2331Z/n00/instruments/probe_order.py.
# These separate the vectorization classes of the fastmath sums.
_BIG = np.float64(2.0) ** 53
_TWO30 = np.float64(2.0) ** 30
MAGNITUDE_CASES = {
    "big_at_1": np.array([1.0, _BIG] + [1.0] * 6),
    "big_at_7": np.array([1.0] * 7 + [_BIG]),
    "big_at_0": np.array([_BIG] + [1.0] * 7),
    "tiny_then_big": np.array([2.0 ** -30] * 8 + [_BIG]),
    "mixed": np.array([1.0, _TWO30, 1.0, 1.0, _TWO30, 1.0, 1.0, 1.0]),
}

# A cohort-realistic case (local_audit_manifest row 0 family: gravity_beta
# 0.001, logistic midpoint 500, knn_decay logistic, knn_weights [1.0]) that
# pins exp-bearing bits in the domain the 10x primary actually runs.
_COHORT_CASE = {
    "distances": [0.0, 1.0, 2.5, 10.0, 100.0, 399.5, 400.0, 401.0],
    "weights": [2.0, 1.0, 0.5, 1.25, 3.0, 0.75, 2.5, 1.0],
    "cutoff": 400.0,
    "gravity_beta": 0.001,
    "plateau": 0.0,
    "midpoint": 500.0,
    "growth_rate": 1.0,
    "knn_decay": "logistic",
    "knn_weights": [1.0],
}

# exp boundary inputs for the per-element K1 gate battery (device exp must
# reproduce HOST exp bits on these before anything dispatches).
EXP_BOUNDARY_VALUES = [
    0.0, -0.0, 2.0 ** -1074, 2.0 ** -1037, 1e-308, 1e-16, 1.0, 709.7827,
    -745.0, 1e308, 2.0 ** 52,
]

# Tie-law battery spec: heavy-tie arrays; the transcription is re-checked
# against a locally compiled probe of the INSTALLED numba on the gate host
# (cache=False) — numba-version drift fails the gate (rc=4), which is the
# designed response.
TIE_BATTERY_TRIALS = 64
TIE_BATTERY_SEED = 7

# Extra K0 device cases (K0 order/count parity incl. NaN exclusion and the
# multi-chunk block loop); seed recorded in evidence.
_K0_RNG_SEED = 11

_CU_SOURCE = Path(__file__).resolve().parent / "fold_kernels.cu"


def _bits(x: float) -> str:
    """float64 bit identity as hex — the only comparable form here."""
    return np.float64(x).tobytes().hex()


def _load_engine():
    """Import the parity artifact from the repo checkout (sys.path insert)."""
    repo_src = str(Path(__file__).resolve().parents[3] / "src")
    if repo_src not in sys.path:
        _sys_path_prepend(repo_src)
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        reach_gravity_knn_access as engine_fold,
    )
    return engine_fold


def _sys_path_prepend(repo_src: str) -> None:
    sys.path.insert(0, repo_src)


def _tree_hits_for(coeffs_by_output, engine_bits_by_output):
    """Which association trees reproduce each engine output's bits.

    ``coeffs_by_output`` maps output name -> float64 coefficient array to
    reduce (``reach`` reduces the kept weights themselves).  Returns
    {output: [trees...]} preserving REDUCTION_TREES order.
    """
    from ..kernels.host_twins import REDUCTION_TREES, reduce_with
    hits = {}
    for out, arr in coeffs_by_output.items():
        want = engine_bits_by_output[out]
        hits[out] = [t for t in REDUCTION_TREES
                     if _bits(reduce_with(t, arr)) == want]
    return hits


def host_battery_tables():
    """Compute the pinned host-side expected bits (engine fold on host).

    Returns the battery dict.  Imports the engine (numba JIT, cache=True):
    call only with a pinned NUMBA_CACHE_DIR.
    """
    engine_fold = _load_engine()
    from ..kernels.host_twins import (REDUCTION_TREES, fold_host_reference,
                                      k0_compact)

    battery = {"magnitude_cases": {}, "cohort_case": {}, "tie_law": {}}

    for name, w in MAGNITUDE_CASES.items():
        d = np.zeros(w.shape[0], dtype=np.float64)
        reach, ge, gl, kn = engine_fold(
            d, w, np.float64(10.0), 0.0, 0.0, 1.0, 0.0,
            np.ones(4, dtype=np.float64), 0.0, "linear", 1.0, 1.0, 1.0)
        entry = {
            "engine_bits": {"reach": _bits(reach), "gravity_exponential": _bits(ge),
                            "gravity_logistic": _bits(gl), "knn_access": _bits(kn)},
            "weights": [float(x) for x in w],
        }
        hits = {}
        for tree in REDUCTION_TREES:
            r2, ge2, gl2, kn2 = fold_host_reference(
                d, w, 10.0, 0.0, 0.0, 1.0, 1.0, "linear",
                np.ones(4, dtype=np.float64), tree)
            # Engine outputs are scalars (row-level sums); compare bit
            # identities, never float equality.
            matches = [m for m, a, b in
                       (("reach", r2, reach), ("gravity_exponential", ge2, ge),
                        ("gravity_logistic", gl2, gl), ("knn_access", kn2, kn))
                       if _bits(a) == _bits(b)]
            hits[tree] = matches
        entry["tree_hits"] = hits
        battery["magnitude_cases"][name] = entry

    c = _COHORT_CASE
    d = np.array(c["distances"], dtype=np.float64)
    w = np.array(c["weights"], dtype=np.float64)
    kn_w = np.array(c["knn_weights"], dtype=np.float64)
    reach, ge, gl, kn = engine_fold(
        d, w, np.float64(c["cutoff"]), c["gravity_beta"], c["plateau"],
        c["midpoint"], 0.0, kn_w, 0.0, c["knn_decay"], 1.0, 1.0, 1.0)
    entry = {
        "engine_bits": {"reach": _bits(reach), "gravity_exponential": _bits(ge),
                        "gravity_logistic": _bits(gl), "knn_access": _bits(kn)},
        "inputs": {k: (v if not isinstance(v, list) else v) for k, v in c.items()},
    }
    hits = {}
    for tree in REDUCTION_TREES:
        r2, ge2, gl2, kn2 = fold_host_reference(
            d, w, c["cutoff"], c["gravity_beta"], c["plateau"], c["midpoint"],
            c["growth_rate"], c["knn_decay"], kn_w, tree)
        matches = [m for m, a, b in
                   (("reach", r2, reach), ("gravity_exponential", ge2, ge),
                    ("gravity_logistic", gl2, gl), ("knn_access", kn2, kn))
                   if _bits(a) == _bits(b)]
        hits[tree] = matches
    entry["tree_hits"] = hits
    entry["k0_indices_hex_or_list"] = [int(x) for x in k0_compact(d, c["cutoff"])[0]]
    battery["cohort_case"] = entry

    # K2 tie-law battery: local numba probe (cache=False; no disk cache).
    import numba as nb

    @nb.njit(cache=False)
    def _argsort_probe(a):
        return np.argsort(a)

    rng = np.random.default_rng(TIE_BATTERY_SEED)
    trials = []
    numpy_kind_conflicts = 0
    for trial in range(TIE_BATTERY_TRIALS):
        n = int(rng.integers(2, 60))
        a = np.round(rng.random(n) * 3, 1)
        idx_nb = [int(x) for x in _argsort_probe(a)]
        kinds = {k: [int(x) for x in np.argsort(a, kind=k)]
                 for k in ("quicksort", "stable", "heapsort")}
        conflicts = not any(idx_nb == v for v in kinds.values())
        numpy_kind_conflicts += int(conflicts)
        trials.append({"n": n, "keys": [float(x) for x in a],
                       "numba_perm": idx_nb, "differs_from_all_numpy_kinds": conflicts})
    battery["tie_law"] = {"trials": trials, "seed": TIE_BATTERY_SEED,
                          "numba_version": nb.__version__,
                          "conflicts_vs_all_numpy_kinds": numpy_kind_conflicts}
    return battery


# ---------------------------------------------------------------------------
# Device gate (the G11 deliverable this module now wires)
# ---------------------------------------------------------------------------

def _k0_device_cases():
    rng = np.random.default_rng(_K0_RNG_SEED)
    cases = {name: (w.astype(np.float64), 10.0)
             for name, w in MAGNITUDE_CASES.items()}
    cases["k0_random_row_a"] = (rng.uniform(0.0, 500.0, 256), 400.0)
    cases["k0_random_row_b"] = (rng.uniform(0.0, 500.0, 1000), 400.0)
    cases["k0_nan_inf_mix"] = (np.array(
        [np.nan, 1.0, np.inf, -np.inf, 0.0, np.nan, 399.5, 401.0]), 400.0)
    cases["k0_all_kept"] = (rng.uniform(0.0, 100.0, 64), 400.0)
    return cases


def _fold_case_inputs():
    """The fold battery: (name, d, w, scalars-dict) per case."""
    cases = []
    for name, w in MAGNITUDE_CASES.items():
        d = np.zeros(w.shape[0], dtype=np.float64)
        cases.append((name, d, w, {
            "cutoff": 10.0, "gravity_beta": 0.0, "plateau": 0.0,
            "midpoint": 1.0, "growth_rate": 1.0, "knn_decay": "linear",
            "knn_weights": np.ones(4, dtype=np.float64),
            "engine_args": (np.float64(10.0), 0.0, 0.0, 1.0, 0.0,
                            np.ones(4, dtype=np.float64), 0.0, "linear",
                            1.0, 1.0, 1.0),
        }))
    c = _COHORT_CASE
    cases.append(("cohort", np.array(c["distances"]),
                  np.array(c["weights"]), {
        "cutoff": c["cutoff"], "gravity_beta": c["gravity_beta"],
        "plateau": c["plateau"], "midpoint": c["midpoint"],
        "growth_rate": c["growth_rate"], "knn_decay": c["knn_decay"],
        "knn_weights": np.array(c["knn_weights"], dtype=np.float64),
        "engine_args": (np.float64(c["cutoff"]), c["gravity_beta"],
                        c["plateau"], c["midpoint"], 0.0,
                        np.array(c["knn_weights"], dtype=np.float64),
                        0.0, c["knn_decay"], 1.0, 1.0, 1.0),
    }))
    return cases


def _device_fold_case(rt, d, w, sc, engine_fold, k0_tw):
    """Run one fold case through device K0/K1 + host K2/K3.

    Returns a result dict with the two parity legs:
      leg A (device-vs-twin): K0 indices/count and K1 coefficient arrays
      must be BIT-EQUAL to the host twins.
      leg B (device-fold-vs-engine): host reduce_with over the DEVICE
      coefficient arrays must reproduce the engine fold's bits under some
      certified tree, per output.
    """
    import ctypes
    from ..kernels.host_twins import (argsort_tie_law, k1_gravity_exponential_coeffs,
                                      k1_gravity_logistic_coeffs, k1_knn_coeffs)

    cutoff = float(sc["cutoff"])
    # --- K0 on device (one block per row; this battery is single-row) -----
    dist_buf = np.ascontiguousarray(d, dtype=np.float64).reshape(1, -1)
    dev_dist = rt.h2d(dist_buf)
    idx_like = np.empty(dist_buf.shape, dtype=np.int64)
    cnt_like = np.empty(1, dtype=np.uint64)
    dev_idx = rt.h2d(idx_like)          # capacity buffer, contents replaced
    dev_cnt = rt.h2d(cnt_like)
    rt.launch("una_g10_fold_compact", (1, 1, 1), (256, 1, 1),
              (dev_dist, ctypes.c_longlong(dist_buf.shape[1]),
               ctypes.c_double(cutoff), dev_idx, dev_cnt))
    rt.sync()
    idx2d = rt.d2h(dev_idx, idx_like)
    cnt = int(rt.d2h(dev_cnt, cnt_like)[0])
    idx = idx2d[0][:cnt]

    h_idx, h_cnt = k0_tw(d, cutoff)
    k0_ok = (cnt == h_cnt and cnt >= 0
             and np.array_equal(idx.astype(np.int64), h_idx[:h_cnt]))

    dn = np.ascontiguousarray(d[h_idx[:h_cnt]], dtype=np.float64)
    wn = np.ascontiguousarray(w[h_idx[:h_cnt]], dtype=np.float64)
    m = int(dn.shape[0])

    res = {"k0_ok": k0_ok, "kept": m, "cases": {}}

    def _launch_elementwise(kname, params, m, out_like):
        out_like = np.ascontiguousarray(out_like)
        if m == 0:
            return np.empty(0, dtype=np.float64)
        dev_out = rt.h2d(out_like)
        rt.launch(kname, ((m + 255) // 256, 1, 1), (256, 1, 1),
                  tuple(params) + (dev_out,))
        rt.sync()
        return rt.d2h(dev_out, np.empty(m, dtype=np.float64))

    # --- K1 gravity exponential -------------------------------------------
    dev_dist2 = rt.h2d(dn)
    dev_w = rt.h2d(wn)
    c_exp = _launch_elementwise(
        "una_g10_fold_coefficients_exp",
        (dev_dist2, dev_w, ctypes.c_longlong(m),
         ctypes.c_double(float(sc["gravity_beta"])),
         ctypes.c_double(float(sc["plateau"]))), m, np.empty(m))
    twin_exp = k1_gravity_exponential_coeffs(dn, wn, sc["gravity_beta"],
                                             sc["plateau"])
    res["cases"]["gravity_exponential"] = bool(
        m == 0 or (c_exp.tobytes() == twin_exp.tobytes()))

    # --- K1 gravity logistic ----------------------------------------------
    c_log = _launch_elementwise(
        "una_g10_fold_coefficients_log",
        (dev_dist2, dev_w, ctypes.c_longlong(m),
         ctypes.c_double(float(sc["growth_rate"])),
         ctypes.c_double(float(sc["plateau"])),
         ctypes.c_double(float(sc["midpoint"]))), m, np.empty(m))
    twin_log = k1_gravity_logistic_coeffs(dn, wn, sc["growth_rate"],
                                          sc["plateau"], sc["midpoint"])
    res["cases"]["gravity_logistic"] = bool(
        m == 0 or (c_log.tobytes() == twin_log.tobytes()))

    # --- K2 (host tie-law sort) + K1 knn on the sorted/cropped prefix -----
    knn = min(m, int(sc["knn_weights"].shape[0]))
    if knn > 0:
        order = argsort_tie_law(dn)
        dn_s = np.ascontiguousarray(dn[order[:knn]])
        wn_s = np.ascontiguousarray(wn[order[:knn]])
        kn = np.ascontiguousarray(sc["knn_weights"][:knn])
        dev_s = rt.h2d(dn_s)
        dev_ws = rt.h2d(wn_s)
        dev_kn = rt.h2d(kn)
        if sc["knn_decay"] == "exponential":
            c_knn = _launch_elementwise(
                "una_g10_fold_knn_coeffs_exp",
                (dev_s, dev_ws, dev_kn, ctypes.c_longlong(knn),
                 ctypes.c_double(float(sc["gravity_beta"])),
                 ctypes.c_double(float(sc["plateau"]))), knn, np.empty(knn))
        elif sc["knn_decay"] == "logistic":
            c_knn = _launch_elementwise(
                "una_g10_fold_knn_coeffs_log",
                (dev_s, dev_ws, dev_kn, ctypes.c_longlong(knn),
                 ctypes.c_double(float(sc["growth_rate"])),
                 ctypes.c_double(float(sc["plateau"])),
                 ctypes.c_double(float(sc["midpoint"]))), knn, np.empty(knn))
        else:
            c_knn = _launch_elementwise(
                "una_g10_fold_knn_coeffs_linear",
                (dev_ws, dev_kn, ctypes.c_longlong(knn)), knn, np.empty(knn))
        twin_knn = k1_knn_coeffs(dn_s, wn_s, kn, sc["knn_decay"],
                                 sc["gravity_beta"], sc["growth_rate"],
                                 sc["plateau"], sc["midpoint"])
        res["cases"]["knn_access"] = bool(c_knn.tobytes() == twin_knn.tobytes())
    else:
        c_knn = np.empty(0, dtype=np.float64)
        res["cases"]["knn_access"] = True

    # --- leg B: certified associations vs the engine fold ------------------
    if engine_fold is None:
        # K0-only battery case: leg A is the entire verdict here.
        res["leg_a_ok"] = res["k0_ok"] and all(res["cases"].values())
        return res

    reach_engine, ge_engine, gl_engine, kn_engine = engine_fold(
        d, w, *sc["engine_args"])
    engine_bits = {"reach": _bits(reach_engine),
                   "gravity_exponential": _bits(ge_engine),
                   "gravity_logistic": _bits(gl_engine),
                   "knn_access": _bits(kn_engine)}
    coeffs = {"reach": wn, "gravity_exponential": c_exp,
              "gravity_logistic": c_log, "knn_access": c_knn}
    res["engine_bits"] = engine_bits
    res["certified"] = _tree_hits_for(coeffs, engine_bits)
    res["all_outputs_certified"] = all(
        res["certified"][o] for o in ("reach", "gravity_exponential",
                                      "gravity_logistic", "knn_access"))
    res["leg_a_ok"] = (res["k0_ok"] and all(res["cases"].values()))
    return res


def _tie_law_check():
    """argsort_tie_law vs a locally compiled probe of the INSTALLED numba."""
    import numba as nb
    from ..kernels.host_twins import argsort_tie_law

    @nb.njit(cache=False)
    def _probe(a):
        return np.argsort(a)

    rng = np.random.default_rng(TIE_BATTERY_SEED)
    mismatches = 0
    for _ in range(TIE_BATTERY_TRIALS):
        n = int(rng.integers(2, 60))
        a = np.round(rng.random(n) * 3, 1)
        if not np.array_equal(argsort_tie_law(a), _probe(a)):
            mismatches += 1
    return {"trials": TIE_BATTERY_TRIALS, "mismatches": mismatches,
            "numba_version": nb.__version__}


def _exp_boundary_check(rt):
    """K1 exp-kernel vs the host twin expression on boundary magnitudes."""
    import ctypes
    from ..kernels.host_twins import k1_gravity_exponential_coeffs

    vals = np.array(EXP_BOUNDARY_VALUES, dtype=np.float64)
    dn = vals.copy()
    wn = np.ones(vals.shape[0], dtype=np.float64)
    m = vals.shape[0]
    dev_d = rt.h2d(dn)
    dev_w = rt.h2d(wn)
    dev_out = rt.h2d(np.empty(m, dtype=np.float64))
    rt.launch("una_g10_fold_coefficients_exp",
              ((m + 255) // 256, 1, 1), (256, 1, 1),
              (dev_d, dev_w, ctypes.c_longlong(m), ctypes.c_double(1.0),
               ctypes.c_double(0.0), dev_out))
    rt.sync()
    got = rt.d2h(dev_out, np.empty(m, dtype=np.float64))
    want = k1_gravity_exponential_coeffs(dn, wn, 1.0, 0.0)
    per_val = [{"x_repr": repr(v), "device": _bits(g), "host": _bits(t),
                "equal": _bits(g) == _bits(t)}
               for v, g, t in zip(vals, got, want)]
    return {"n": m, "all_equal": all(p["equal"] for p in per_val),
            "per_value": per_val}


def run_device_gate(cell_id: str, gate_store: str, evidence_path: Path,
                    ) -> tuple:
    """The full gate sequence on a probed-available device.

    Returns (exit_code, report dict).  Exit 0 records a PASSED entry in
    the durable store; exit 4 records a refused entry; exit 3 (toolchain/
    context) records NOTHING (no verdict exists to record).
    """
    from ..backends.gpu import record_gate
    from .device_exec import (DeviceGateError, DeviceRuntime, compile_ptx,
                              nvcc_flags)

    report = {"cell": cell_id, "mode": "device_gate",
              "started_utc": datetime.now(timezone.utc).strftime(
                  "%Y-%m-%dT%H:%M:%SZ")}

    nvcc = detect_nvcc()
    report["nvcc"] = {"on_path": nvcc.on_path, "version": nvcc.version,
                      "path": nvcc.detail}
    if not nvcc.on_path:
        report["status"] = "toolchain_absent"
        return EXIT_TOOLCHAIN_UNUSABLE, report

    if not gate_store:
        report["status"] = "gate_store_unconfigured"
        report["error"] = ("a device verdict with nowhere durable to live "
                           f"is refused; pass --gate-store or set "
                           "UNA_GPU_GATE_STORE")
        return EXIT_TOOLCHAIN_UNUSABLE, report

    try:
        rt = DeviceRuntime(device_index=0)
    except DeviceGateError as exc:
        report["status"] = "context_unusable"
        report["error"] = str(exc)
        return EXIT_TOOLCHAIN_UNUSABLE, report

    try:
        idn = rt.identity
        arch = idn.compute_capability[0] * 10 + idn.compute_capability[1]
        report["device"] = {
            "name": idn.name, "compute_capability": list(idn.compute_capability),
            "sm": idn.sm_name, "driver_version": list(idn.driver_version),
            "device_index": idn.device_index,
            "context": "cuCtxCreate(flags=0) explicit; the primary context "
                       "is retained-but-inactive on this node class "
                       "(cuMemAlloc rc=201), see diag_ctx2.py consoles",
        }
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        ptx = compile_ptx(_CU_SOURCE, arch, nvcc.detail, evidence_path.parent)
        ptx_sha = hashlib.sha256(ptx.read_bytes()).hexdigest()
        report["compile"] = {"ptx": ptx.name, "ptx_sha256": ptx_sha,
                             "flags": " ".join(nvcc_flags(arch)),
                             "cu_source_sha256": hashlib.sha256(
                                 _CU_SOURCE.read_bytes()).hexdigest()}
        rt.load_ptx(ptx)

        from ..kernels.host_twins import k0_compact

        # --- K0 device battery (order/count parity incl. NaN/inf/multi-chunk)
        k0_scalars = {"cutoff": 0.0, "gravity_beta": 0.0, "plateau": 0.0,
                      "midpoint": 1.0, "growth_rate": 1.0,
                      "knn_decay": "linear",
                      "knn_weights": np.ones(1, dtype=np.float64)}
        k0_results = {}
        for name, (row, cutoff) in _k0_device_cases().items():
            sc = dict(k0_scalars, cutoff=cutoff)
            # K0-only case (engine_fold=None): leg A is the whole verdict.
            case_res = _device_fold_case(
                rt, row, np.ones(row.shape[0], dtype=np.float64), sc,
                None, k0_compact)
            k0_results[name] = {"k0_ok": case_res["k0_ok"],
                                "kept": case_res["kept"]}
        report["k0_battery"] = k0_results
        k0_all = all(v["k0_ok"] for v in k0_results.values())

        # --- K1 exp boundary battery --------------------------------------
        exp_res = _exp_boundary_check(rt)
        report["exp_boundary"] = {"all_equal": exp_res["all_equal"],
                                  "n": exp_res["n"]}

        # --- fold battery: device K0/K1 + host K2/K3 vs engine -------------
        engine_fold = _load_engine()
        from ..kernels.host_twins import k0_compact as k0_tw
        fold_results = {}
        for name, d, w, sc in _fold_case_inputs():
            fold_results[name] = _device_fold_case(
                rt, np.asarray(d, dtype=np.float64),
                np.asarray(w, dtype=np.float64), sc, engine_fold, k0_tw)
        report["fold_battery"] = {
            name: {"leg_a_ok": r["leg_a_ok"],
                   "leg_a_details": r["cases"],
                   "all_outputs_certified": r["all_outputs_certified"],
                   "certified": r["certified"],
                   "engine_bits": r["engine_bits"]}
            for name, r in fold_results.items()}

        # --- K2 tie law vs the installed numba build -----------------------
        tie = _tie_law_check()
        report["tie_law"] = tie

        legs_ok = (k0_all and exp_res["all_equal"] and tie["mismatches"] == 0
                   and all(r["leg_a_ok"] and r["all_outputs_certified"]
                           for r in fold_results.values()))

        report["numba"] = {"version": tie["numba_version"]}
        report["finished_utc"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        if legs_ok:
            cohort = fold_results["cohort"]["certified"]
            association = (cohort["gravity_exponential"][0]
                           if cohort["gravity_exponential"]
                           else next(h[0] for h in cohort.values() if h))
            report["status"] = "gate_passed"
            report["gate"] = ("PASSED on real device; evidence recorded in "
                              f"{gate_store}")
            report["cell_association_recorded"] = association
            report["association_note"] = (
                "the store's association is the cohort case's certified "
                "gravity_exponential tree (the strongest discriminator); "
                "the full per-case/per-output matrix is this evidence file")
            evidence_path.write_text(json.dumps(report, indent=2) + "\n")
            record_gate(gate_store, cell_id, passed=True,
                        association=association,
                        evidence_ref=str(evidence_path))
            return EXIT_PASSED, report

        report["status"] = "battery_mismatch"
        report["gate"] = ("NOT PASSED (designed gate refusal; cell stays "
                          "closed; refusal durably recorded)")
        evidence_path.write_text(json.dumps(report, indent=2) + "\n")
        record_gate(gate_store, cell_id, passed=False,
                    evidence_ref=str(evidence_path))
        return EXIT_BATTERY_MISMATCH, report
    except DeviceGateError as exc:
        report["status"] = "context_unusable"
        report["error"] = str(exc)
        return EXIT_TOOLCHAIN_UNUSABLE, report
    finally:
        rt.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="fold-region per-cell bit-validation gate harness")
    parser.add_argument("--emit-host-battery", metavar="PATH",
                        help="compute the pinned host expected-bits table "
                             "(no hardware needed; engine JIT on host); "
                             f"exits {EXIT_HOST_BATTERY}, never 0 — a "
                             "host-only artifact is not a gate verdict")
    parser.add_argument("--gate-store", metavar="PATH",
                        default=None,
                        help="durable gate store (default: env "
                             "UNA_GPU_GATE_STORE; required for a "
                             "device-gate verdict")
    parser.add_argument("--evidence", metavar="PATH", default=None,
                        help="evidence JSON output (default: "
                             "gate_evidence.json beside the store)")
    parser.add_argument("--cell", default="una_legacy.fold.default",
                        help="cell id the gate verdict applies to")
    args = parser.parse_args(argv)

    if args.emit_host_battery:
        battery = host_battery_tables()
        out = Path(args.emit_host_battery)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(battery, indent=2) + "\n")
        print(json.dumps({"status": "host_battery_written", "path": str(out),
                          "exit_code": EXIT_HOST_BATTERY,
                          "note": "host half only; no gate verdict — "
                                  "device half requires real hardware"},
                         indent=2))
        return EXIT_HOST_BATTERY

    probe = probe_driver()
    if not probe.available:
        print(json.dumps({
            "status": "hardware_unavailable",
            "cell": args.cell,
            "gate": "NOT PASSED (designed refusal; never simulated)",
            "driver": {"reason": probe.reason, "detail": probe.detail},
            "nvcc": {"on_path": detect_nvcc().on_path},
        }, indent=2))
        return EXIT_HARDWARE_UNAVAILABLE

    gate_store = args.gate_store
    if gate_store is None:
        gate_store = os.environ.get("UNA_GPU_GATE_STORE")
    if args.evidence:
        evidence_path = Path(args.evidence)
    elif gate_store:
        evidence_path = Path(gate_store).parent / "gate_evidence.json"
    else:
        evidence_path = Path("gate_evidence.json")
    rc, report = run_device_gate(args.cell, gate_store or "", evidence_path)
    print(json.dumps(report, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
