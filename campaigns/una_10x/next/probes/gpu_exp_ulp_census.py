"""Direction-1 step-1 probe: exp bit census, device libdevice vs host authorities.

Measures, over an explicitly recorded argument sweep, where the device
libdevice ``exp`` (numba.cuda ``math.exp``, float64) differs in bits from:

  - ``numpy.exp`` (the host NumPy authority the G11 receipt cited), and
  - a numba ``njit(fastmath=True)`` scalar ``exp`` (the delivered fold's
    host lowering proxy — the actual parity authority per G11).

This is a census instrument, not an admission gate: admission (Direction 1,
PLAN_GPU.md §3) requires 0 differing arguments on a declared domain AND the
extended battery matrix; this probe only maps the domain.

Without a CUDA device this exits with a typed ``device_unavailable`` status —
a recorded blocker, never a simulated pass.

Run (device node):
    PYTHONPATH=src python probes/gpu_exp_ulp_census.py \
        --manifest <cohort_manifest.csv> --out <out.json>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


def _ulp_view(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64).view(np.int64)


def _diff_stats(x: np.ndarray, dev: np.ndarray, host: np.ndarray):
    dv, hv = _ulp_view(dev), _ulp_view(host)
    same_sign = (np.signbit(dev) == np.signbit(host))
    ulp = np.where(same_sign, np.abs(dv - hv), -1)
    n_diff = int(np.count_nonzero(ulp != 0))
    finite = np.isfinite(ulp)
    max_ulp = int(ulp[finite].max()) if n_diff and finite.any() else None
    return {"n": int(x.size), "n_diff": n_diff,
            "max_ulp": max_ulp,
            "n_sign_flips": int(np.count_nonzero(~same_sign))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path(
        "/storage/scratch1/1/dsynn6/campaign_data/inputs/"
        "upstream_Boston/Boston/cohort_manifest.csv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dense-per-decade", type=int, default=2000)
    args = ap.parse_args()

    result = {"schema": "una10x-gpu-exp-ulp-census-1",
              "purpose": "Direction 1 step 1 (PLAN_GPU.md §3); census only",
              "python": sys.executable}

    # --- host authorities (run anywhere) --------------------------------
    from numba import njit

    @njit(cache=False, fastmath=True)
    def exp_fast(x):
        return math.exp(x)

    # --- sweep construction ---------------------------------------------
    manifest_rows = []
    with open(args.manifest, newline="") as fh:
        for row in csv.DictReader(fh):
            manifest_rows.append(row)
    betas = sorted({float(r["gravity_beta"]) for r in manifest_rows})
    mids = sorted({float(r["gravity_logistic_midpoint"])
                   for r in manifest_rows})
    radii = sorted({float(r["search_radius"]) for r in manifest_rows})
    result["manifest_derived"] = {"n_rows": len(manifest_rows),
                                  "betas": betas, "mids": mids,
                                  "radii": radii}

    # ge branch args: -beta * clipped, clipped in [0, radius]
    ge_args = np.concatenate([
        -b * np.linspace(0.0, rad, 4096)
        for b in betas for rad in radii])
    # gl/kn-logistic branch args: -gr * (pos - mid); the manifest does not
    # carry a growth-rate column, so a gr grid brackets the delivered
    # default neighborhood; pos in [0, radius]. One 1-D array per (gr, mid).
    gr_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    gl_args = np.concatenate([
        -g * (np.linspace(0.0, max(radii), 1024) - m)
        for g in gr_grid for m in mids])
    # dense monotone sweep for coverage
    dense = np.concatenate([
        np.logspace(-40, 40, 41 * args.dense_per_decade),
        -np.logspace(-40, 40, 41 * args.dense_per_decade),
        np.linspace(-1.0, 1.0, 8192)])
    x = np.concatenate([ge_args, gl_args, dense]).astype(np.float64)
    result["sweep"] = {"n": int(x.size),
                       "ge_args_n": int(ge_args.size),
                       "gl_args_n": int(gl_args.size),
                       "dense_n": int(dense.size),
                       "gr_grid": gr_grid}

    host_np = np.exp(x)
    host_fast = np.array([exp_fast(v) for v in x], dtype=np.float64)

    # --- device leg (gated) ----------------------------------------------
    try:
        from numba import cuda
        if not cuda.is_available():
            raise RuntimeError("no CUDA device / driver init failed")
        d_x = cuda.to_device(x)
        d_y = cuda.device_array_like(x)

        @cuda.jit
        def _k(inp, outp):
            i = cuda.grid(1)
            if i < inp.size:
                outp[i] = math.exp(inp[i])

        n = x.size
        _k[(n + 127) // 128, 128](d_x, d_y)
        cuda.synchronize()
        dev = d_y.copy_to_host()
        result["device"] = {"name": cuda.get_current_device().name
                            .decode() if isinstance(
                                cuda.get_current_device().name, bytes)
                            else str(cuda.get_current_device().name)}
    except Exception as exc:  # typed blocker, never simulated
        result["device"] = {"status": "device_unavailable",
                            "error": repr(exc)}
        result["host_only"] = {
            "np_vs_fastmath": _diff_stats(x, host_np, host_fast)}
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps({"out": str(args.out),
                          "device": "unavailable"}))
        return 0

    result["device_vs_np"] = _diff_stats(x, dev, host_np)
    result["device_vs_fastmath"] = _diff_stats(x, dev, host_fast)
    result["np_vs_fastmath"] = _diff_stats(x, host_np, host_fast)
    # exact-match sub-domain where device == delivered-lowering proxy
    dv, hv = _ulp_view(dev), _ulp_view(host_fast)
    eq = (dv == hv)
    result["device_eq_fastmath_subdomain"] = {
        "n": int(np.count_nonzero(eq)),
        "share": float(eq.mean()),
    }
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(args.out),
                      "device_vs_fastmath": result["device_vs_fastmath"],
                      "eq_share":
                          result["device_eq_fastmath_subdomain"]["share"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
