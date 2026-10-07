"""Per-cell bit-validation gate harness for the fold region (G10).

THE gate from gpu_region.json ("device-bit-validation"): before any cell
may dispatch K1/K3 on a device, the DEVICE's coefficients and certified
reduction trees must reproduce the pinned una_legacy fold bits on this
characterization battery, on that device, with the evidence recorded.

Exit-code contract (designed refusals are reported, never faked):
  0   gate PASSED on a real device (G11 deliverable; evidence JSON written)
  2   hardware_unavailable — no CUDA-class FP64 device (THIS HOST today);
      a designed refusal, evidence JSON printed to stdout
  3   device present but the G11 device launcher/context is absent
  4   battery mismatch — device bits != pinned host bits (the gate doing
      its job; the cell stays gated closed)

Host-side modes that run WITHOUT hardware:
  --emit-host-battery PATH
      Runs the ENGINE fold (the parity artifact,
      Engines.AccessibilityWElevation.reach_gravity_knn_access) on this
      host over the characterization battery and writes the pinned
      expected-bits table plus, per case, which K3 association trees
      reproduce the engine bits (probe_order.py's method, as a reusable
      harness).  This imports the engine and JIT-compiles it: pin a fresh
      NUMBA_CACHE_DIR and PYTHONDONTWRITEBYTECODE=1, e.g.
        NUMBA_CACHE_DIR=<run>/g10/numba_cache PYTHONDONTWRITEBYTECODE=1 \
          python -m gpu.kernels.validate_device_bits --emit-host-battery \
          <run>/g10/host_battery.json
      The device comparison half is UNRUN by construction here (no device);
      a CPU-only run can never mark a cell bit_validated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..runtime.availability import detect_nvcc, probe_driver

EXIT_PASSED = 0
EXIT_HARDWARE_UNAVAILABLE = 2
EXIT_LAUNCHER_ABSENT = 3
EXIT_BATTERY_MISMATCH = 4

# Characterization magnitude cases, verbatim from the committed instrument
# campaign_runs/una_10x/tenx-20261005T2331Z/n00/instruments/probe_order.py
# (read this task).  These separate the vectorization classes of the
# fastmath sums.
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

# Tie-law battery spec (the gate re-checks K2 on the qualification host):
# heavy-tie arrays; expected permutations recorded from the LOCAL numba
# probe (cache=False) and cross-checked against every numpy kind.
TIE_BATTERY_TRIALS = 64
TIE_BATTERY_SEED = 7


def _bits(x: float) -> str:
    """float64 bit identity as hex — the only comparable form here."""
    return np.float64(x).tobytes().hex()


def host_battery_tables():
    """Compute the pinned host-side expected bits (engine fold on host).

    Returns (battery dict, tree-classification dict).  Imports the engine
    (numba JIT, cache=True): call only with a pinned NUMBA_CACHE_DIR.
    """
    import sys as _sys
    repo_src = str(Path(__file__).resolve().parents[3] / "src")
    if repo_src not in _sys.path:
        _sys.path.insert(0, repo_src)
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        reach_gravity_knn_access as engine_fold,
    )

    from ..kernels.host_twins import (REDUCTION_TREES, argsort_tie_law,
                                      fold_host_reference, k0_compact)

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="fold-region per-cell bit-validation gate harness")
    parser.add_argument("--emit-host-battery", metavar="PATH",
                        help="compute the pinned host expected-bits table "
                             "(no hardware needed; engine JIT on host)")
    parser.add_argument("--cell", default="una_legacy.fold.default",
                        help="cell id the gate verdict applies to")
    args = parser.parse_args(argv)

    if args.emit_host_battery:
        battery = host_battery_tables()
        out = Path(args.emit_host_battery)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(battery, indent=2) + "\n")
        print(json.dumps({"status": "host_battery_written", "path": str(out),
                          "note": "host half only; device half requires "
                                  "real hardware and stays unrun"},
                         indent=2))
        return EXIT_PASSED

    probe = probe_driver()
    nvcc = detect_nvcc()
    if not probe.available:
        print(json.dumps({
            "status": "hardware_unavailable",
            "cell": args.cell,
            "gate": "NOT PASSED (designed refusal; never simulated)",
            "driver": {"reason": probe.reason, "detail": probe.detail},
            "nvcc": {"on_path": nvcc.on_path, "version": nvcc.version},
        }, indent=2))
        return EXIT_HARDWARE_UNAVAILABLE

    print(json.dumps({
        "status": "gate_launcher_not_wired",
        "cell": args.cell,
        "device": {"names": list(probe.device_names),
                   "fp64_capable": probe.fp64_capable},
        "note": "device execution lands with the G11 launcher; the battery "
                "is pinned host-side via --emit-host-battery",
    }, indent=2))
    return EXIT_LAUNCHER_ABSENT


if __name__ == "__main__":
    sys.exit(main())
