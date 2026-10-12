"""Direction-1 road-(b) probe: fold-relevant exp-argument box census + the
rep4000 structure's ACTUAL derived exp arguments, device vs delivered
authority.

Two manifest-bounded boxes (PLAN_GPU.md §3):

  - ge box: x = -b * c for every cohort manifest beta b and clip c on a
    dense 4096-point grid of [0, radius].
  - gl box: x = -gr * (p - mid) for gr in [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    p on a 4096-point grid over [0, max radius], mid from the manifest.

For EACH box: host ``np.exp``, the delivered-authority exp (njit
fastmath ``math.exp``, cache=False, same proxy as the committed census
probe), and device libdevice ``math.exp``; per-box divergence stats and
a 16-bin histogram of |x| over the divergent arguments.

Beyond the boxes, the probe reads the delivered host fold
(``Accessibility.reach_gravity_knn_access``, njit fastmath - the parity
authority per G11; the G10 bridge mirrors it) and constructs, in
float64, the ACTUAL exp arguments the rep4000 structure feeds exp per
decay (ge / gl / kn), then classifies those exact arguments
device-vs-fastmath.  ``derivation_certified`` is set true ONLY if BOTH:
(i) the derived arguments are bit-identical to the delivered fold's own
inline fastmath construction, AND (ii) the derived-arg construction, fed
through a faithful re-implementation of the fold's exp-consuming
statements, reproduces the delivered fold's final ge/gl/kn outputs
bit-exactly on small cases (host leg - runs with or without a device);
otherwise it stays false with named gaps.  The certification also
decomposes any failure: a verbatim inline re-implementation of the
delivered statements shows whether a divergence lives in the argument
derivation or in the compiled statement lowering itself.

Census instrument, not an admission gate.  Without a CUDA device the
device leg exits with a typed ``device_unavailable`` status; host legs
and the host-only certification still run and are recorded honestly.

Run (device node):
    PYTHONPATH=src python probes/gpu_fold_box_census.py --out <out.json>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from numba import njit


def _ulp_view(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64).view(np.int64)


def _diff_stats(x: np.ndarray, a: np.ndarray, b: np.ndarray):
    av, bv = _ulp_view(a), _ulp_view(b)
    same_sign = (np.signbit(a) == np.signbit(b))
    ulp = np.where(same_sign, np.abs(av - bv), -1)
    n_diff = int(np.count_nonzero(ulp != 0))
    finite = np.isfinite(ulp)
    max_ulp = int(ulp[finite].max()) if n_diff and finite.any() else None
    return {"n": int(x.size), "n_diff": n_diff,
            "max_ulp": max_ulp,
            "n_sign_flips": int(np.count_nonzero(~same_sign))}


def _divergence_census(x, dev, fast):
    """device-vs-fastmath divergence census for one argument box/group."""
    dv, hv = _ulp_view(dev), _ulp_view(fast)
    xv = _ulp_view(x)
    same_sign = (np.signbit(dev) == np.signbit(fast))
    ulp = np.where(same_sign, np.abs(dv - hv), -1)
    div = (ulp != 0)
    n_diff = int(np.count_nonzero(div))
    finite = np.isfinite(ulp)
    ax = np.abs(x)
    out = {"n": int(x.size),
           "n_diff_device_vs_fastmath": n_diff,
           "max_ulp": int(ulp[finite].max())
           if n_diff and finite.any() else None,
           "n_sign_flips": int(np.count_nonzero(~same_sign)),
           "share_divergent": (n_diff / x.size) if x.size else 0.0}
    # unique-argument-bit view: exp is deterministic, so each distinct
    # argument bit carries one device and one fastmath output bit
    ux, first = np.unique(xv, return_index=True)
    out["n_unique_args"] = int(ux.size)
    out["n_unique_args_diff"] = \
        int(np.count_nonzero(dv[first] != hv[first]))
    hi = float(ax.max()) if x.size else 0.0
    edges = np.linspace(0.0, hi, 17)
    if n_diff:
        counts = np.histogram(ax[div], bins=16, range=(0.0, hi))[0]
        out["divergent_absx_min"] = float(ax[div].min())
        out["divergent_absx_max"] = float(ax[div].max())
    else:
        counts = np.zeros(16, dtype=np.int64)
        out["divergent_absx_min"] = None
        out["divergent_absx_max"] = None
    out["divergent_absx_hist_16bins"] = {
        "range_absx": [0.0, hi],
        "edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts]}
    return out


def _rep4000(seed: int = 20261012):
    # identical builder to probes/gpu_phase_decomposition.py (same seed ->
    # the structure whose phase walls this campaign already recorded)
    rng = np.random.default_rng(seed)
    origins, target_nnz = 4000, 238387
    base = target_nnz // origins
    extra = target_nnz - base * origins
    counts = np.full(origins, base, dtype=np.int64)
    counts[:extra] += 1
    indptr = np.zeros(origins + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])
    ids = np.empty(nnz, dtype=np.int64)
    dist = np.empty(nnz, dtype=np.float64)
    for o in range(origins):
        lo, hi = int(indptr[o]), int(indptr[o + 1])
        ids[lo:hi] = np.sort(rng.choice(20000, hi - lo, replace=False))
        dist[lo:hi] = np.sort(rng.uniform(1.0, 2000.0, hi - lo))
    d_weights = rng.uniform(0.0, 1.0, 20000)
    return indptr, ids, dist, d_weights


def _derived_args(d, w_slice, cutoff, beta, plateau, mid, gr, knn_n):
    """The exp arguments the delivered fold feeds exp for one origin
    slice, constructed in float64 with the source's association
    (Accessibility.py:83/85/104/106; bridge mirror _gpu_fold.py)."""
    keep = d <= cutoff
    nd = d[keep]
    ge_args = beta * np.maximum(0.0, nd - plateau)
    gl_args = -gr * (nd - plateau - mid)
    order = np.argsort(nd)
    ndc = nd[order][:knn_n]
    kn_exp_args = -beta * np.maximum(0.0, ndc - plateau)
    kn_log_args = -gr * (ndc - plateau - mid)
    n_weights = w_slice[keep]
    knn_w_cropped = n_weights[order][:knn_n]
    return (ge_args, gl_args, kn_exp_args, kn_log_args,
            n_weights, knn_w_cropped)


# --- delivered fold's exp-consuming statements, verbatim (only the exp
# argument is pre-derived; Accessibility.py:83/85 and :104/:106) -----------

@njit(cache=False, fastmath=True)
def _reimpl_exponential(n_weights, ge_args, gl_args, knn_coeff,
                        knn_w_cropped, kn_args):
    gravity_exponential = (n_weights / np.exp(ge_args)).sum()
    gravity_logistic = (n_weights * (1 - 1 / (1 + np.exp(gl_args)))).sum()
    knn_access = ((knn_coeff * knn_w_cropped)
                  * np.exp(kn_args)).sum()
    return gravity_exponential, gravity_logistic, knn_access


@njit(cache=False, fastmath=True)
def _reimpl_logistic(n_weights, ge_args, gl_args, knn_coeff,
                     knn_w_cropped, kn_args):
    gravity_exponential = (n_weights / np.exp(ge_args)).sum()
    gravity_logistic = (n_weights * (1 - 1 / (1 + np.exp(gl_args)))).sum()
    knn_access = ((knn_coeff * knn_w_cropped)
                  * (1 - 1 / (1 + np.exp(kn_args)))).sum()
    return gravity_exponential, gravity_logistic, knn_access


@njit(cache=False, fastmath=True)
def _inline_args(d, cutoff, beta, plateau, mid, gr, knn_n):
    """The delivered fold's exp arguments built INLINE under the same
    fastmath lowering - the arg-bit reference the derivation must match."""
    keep = np.where(d <= cutoff)[0]
    nd = d[keep]
    ge_arg = beta * np.maximum(0, nd - plateau)
    gl_arg = -gr * (nd - plateau - mid)
    order = np.argsort(nd)
    ndc = nd[order][:knn_n]
    kn_exp_arg = -beta * np.maximum(0, ndc - plateau)
    kn_log_arg = -gr * (ndc - plateau - mid)
    return ge_arg, gl_arg, kn_exp_arg, kn_log_arg


@njit(cache=False, fastmath=True)
def _inline_fold_exponential(d, w_slice, cutoff, beta, plateau, mid, gr,
                             knn_coeff):
    """Verbatim delivered statements, kn exponential branch, args inline.
    Sensitivity diagnostic only: if even this diverges from the delivered
    fold, the divergence lives in the compiled statement lowering."""
    n_reach_filter = np.where(d <= cutoff)[0]
    n_distances = d[n_reach_filter]
    n_weights = w_slice[n_reach_filter]
    gravity_exponential = (n_weights
                           / np.exp(beta
                                    * np.maximum(0,
                                                 n_distances - plateau))).sum()
    gravity_logistic = (n_weights
                        * (1 - 1 / (1 + np.exp(-gr
                                               * (n_distances - plateau
                                                  - mid))))).sum()
    knn = min(n_distances.shape[0], knn_coeff.shape[0])
    idx = np.argsort(n_distances)
    n_distances_cropped = n_distances[idx][:knn]
    n_weights_cropped = n_weights[idx][:knn]
    knn_access = ((knn_coeff[:knn] * n_weights_cropped)
                  * np.exp(-beta
                           * np.maximum(0,
                                        n_distances_cropped
                                        - plateau))).sum()
    return gravity_exponential, gravity_logistic, knn_access


@njit(cache=False, fastmath=True)
def _inline_fold_logistic(d, w_slice, cutoff, beta, plateau, mid, gr,
                          knn_coeff):
    n_reach_filter = np.where(d <= cutoff)[0]
    n_distances = d[n_reach_filter]
    n_weights = w_slice[n_reach_filter]
    gravity_exponential = (n_weights
                           / np.exp(beta
                                    * np.maximum(0,
                                                 n_distances - plateau))).sum()
    gravity_logistic = (n_weights
                        * (1 - 1 / (1 + np.exp(-gr
                                               * (n_distances - plateau
                                                  - mid))))).sum()
    knn = min(n_distances.shape[0], knn_coeff.shape[0])
    idx = np.argsort(n_distances)
    n_distances_cropped = n_distances[idx][:knn]
    n_weights_cropped = n_weights[idx][:knn]
    knn_access = ((knn_coeff[:knn] * n_weights_cropped)
                  * (1 - 1 / (1 + np.exp(-gr
                                         * (n_distances_cropped
                                            - plateau - mid))))).sum()
    return gravity_exponential, gravity_logistic, knn_access


def _certify(reach_gravity_knn_access, cases_info):
    """Certify the derived-arg construction against the delivered fold:
    (i) derived args bit-identical to the delivered fold's inline
    fastmath arg construction, AND (ii) derived args through the
    re-implemented exp-consuming statements bit-equal to the delivered
    ge/gl/kn outputs.  A verbatim inline re-implementation decomposes
    any (ii) failure into arg-derivation vs statement-lowering causes."""
    C0, B1, P2, M3, G4 = 12.0, 0.35, 0.5, 0.75, 1.2
    KW = np.array([1.0, 0.8, 0.6])
    out = {"cases": [], "gaps": []}
    args_ok = True
    outputs_ok = True
    n_checks = 0

    def bits(a):
        return _ulp_view(np.atleast_1d(np.float64(a)))

    for tag, d, w_slice in cases_info:
        rec = {"case": tag, "n": int(d.size), "arg_bit_diffs": {},
               "decay_summaries": {}}
        if d.size == 0:
            gap = f"{tag}: empty case - excluded"
            out["gaps"].append(gap)
            rec["excluded"] = gap
            args_ok = outputs_ok = False
            out["cases"].append(rec)
            continue
        if np.unique(_ulp_view(d)).size != d.size:
            gap = f"{tag}: duplicate distance bits " \
                  f"(argsort tie domain) - excluded"
            out["gaps"].append(gap)
            rec["excluded"] = gap
            args_ok = outputs_ok = False
            out["cases"].append(rec)
            continue
        dc = np.ascontiguousarray(d)
        wsl = np.ascontiguousarray(w_slice)
        (ge_args, gl_args, kn_exp, kn_log,
         n_weights, knn_w_c) = _derived_args(
            dc, wsl, C0, B1, P2, M3, G4, KW.shape[0])
        i_ge, i_gl, i_ke, i_kl = _inline_args(
            dc, C0, B1, P2, M3, G4, KW.shape[0])
        for name, a, b in (("ge_positive", ge_args, i_ge),
                           ("gl", gl_args, i_gl),
                           ("kn_exponential", kn_exp, i_ke),
                           ("kn_logistic", kn_log, i_kl)):
            n_bad = int(np.count_nonzero(_ulp_view(a) != _ulp_view(b)))
            rec["arg_bit_diffs"][name] = n_bad
            if n_bad:
                out["gaps"].append(
                    f"{tag} {name}: {n_bad} derived-arg bits differ from "
                    f"the delivered fold's inline fastmath construction")
                args_ok = False
        for decay in ("exponential", "logistic"):
            ge_d, gl_d, kn_d = reach_gravity_knn_access(
                d.copy(), wsl.copy(), C0, B1, P2, M3, 0.0, KW,
                0.2, decay, 0.5, G4, 0.9)[1:]
            fn = _reimpl_exponential if decay == "exponential" \
                else _reimpl_logistic
            kn_args = kn_exp if decay == "exponential" else kn_log
            ge_r, gl_r, kn_r = fn(n_weights, ge_args, gl_args,
                                  KW[:knn_w_c.shape[0]], knn_w_c, kn_args)
            ifn = _inline_fold_exponential if decay == "exponential" \
                else _inline_fold_logistic
            ge_i, gl_i, kn_i = ifn(dc, wsl, C0, B1, P2, M3, G4, KW)
            summ = {"param_reimpl_bit_equal": {},
                    "param_reimpl_max_ulp": {},
                    "inline_reimpl_bit_equal": {}}
            for name, a, b in (("ge", ge_d, ge_r), ("gl", gl_d, gl_r),
                               ("kn", kn_d, kn_r)):
                av, bv = bits(a), bits(b)
                eq = bool(np.array_equal(av, bv))
                summ["param_reimpl_bit_equal"][name] = eq
                summ["param_reimpl_max_ulp"][name] = \
                    int(np.abs(av - bv).max()) if not eq else 0
                if not eq:
                    out["gaps"].append(
                        f"{tag} {decay} {name}: bit-verified derived args "
                        f"through the exp-consuming statements differ from "
                        f"the delivered fold by "
                        f"{int(np.abs(av - bv).max())} ULP (compiled "
                        f"statement lowering, not the arg derivation)")
                    outputs_ok = False
            for name, a, b in (("ge", ge_d, ge_i), ("gl", gl_d, gl_i),
                               ("kn", kn_d, kn_i)):
                summ["inline_reimpl_bit_equal"][name] = \
                    bool(np.array_equal(bits(a), bits(b)))
            rec["decay_summaries"][decay] = summ
            n_checks += 1
        out["cases"].append(rec)
    out["n_checks"] = n_checks
    out["arg_bits_identical"] = bool(args_ok and n_checks > 0)
    out["outputs_bit_equal"] = bool(outputs_ok and n_checks > 0)
    out["certified"] = out["arg_bits_identical"] and out["outputs_bit_equal"]
    out["method"] = (
        "certified requires BOTH: derived args bit-identical to the "
        "delivered fold's inline fastmath arg construction, AND derived "
        "args fed through verbatim re-implementations of the fold's "
        "exp-consuming statements (njit fastmath=True, cache=False) "
        "reproducing delivered ge/gl/kn bit-exactly on every small case "
        "and both decay branches; a verbatim inline re-implementation "
        "separates arg-derivation failures from statement-lowering "
        "failures; reach/kept are exp-free and flat decay consumes no "
        "exp - both out of scope")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path(
        "/storage/scratch1/1/dsynn6/campaign_data/inputs/"
        "upstream_Boston/Boston/cohort_manifest.csv"))
    ap.add_argument("--points", type=int, default=4096)
    args = ap.parse_args()

    result = {"schema": "una10x-gpu-fold-box-census-1",
              "purpose": "Direction 1 road (b) input (PLAN_GPU.md §3); "
                         "box census + derived-arg classification; "
                         "census only, no admission",
              "python": sys.executable,
              "numba": __import__("numba").__version__,
              "numpy": np.__version__}

    # --- host authority proxy (identical to the committed census probe) ---
    @njit(cache=False, fastmath=True)
    def exp_fast(x):
        return math.exp(x)

    # --- manifest ----------------------------------------------------------
    manifest_rows = []
    with open(args.manifest, newline="") as fh:
        for row in csv.DictReader(fh):
            manifest_rows.append(row)
    betas = sorted({float(r["gravity_beta"]) for r in manifest_rows})
    mids = sorted({float(r["gravity_logistic_midpoint"])
                   for r in manifest_rows})
    radii = sorted({float(r["search_radius"]) for r in manifest_rows})
    result["manifest_derived"] = {"n_rows": len(manifest_rows),
                                  "betas_n": len(betas),
                                  "mids": mids, "radii": radii}
    print(json.dumps({"stage": "manifest", "n_rows": len(manifest_rows),
                      "betas_n": len(betas), "mids": mids,
                      "radii": radii}), flush=True)

    # --- boxes (per-array construction, per the census-probe fix) ----------
    work = {}
    work["ge_box"] = {
        "x": np.concatenate([
            -b * np.linspace(0.0, rad, args.points)
            for b in betas for rad in radii]).astype(np.float64),
        "construction":
            "x = -b * c; b in manifest betas (32); "
            "c = linspace(0, radius, 4096); radius from manifest"}
    gr_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    work["gl_box"] = {
        "x": np.concatenate([
            -g * (np.linspace(0.0, max(radii), args.points) - m)
            for g in gr_grid for m in mids]).astype(np.float64),
        "construction":
            "x = -gr * (p - mid); gr in [1e-4,1e-3,1e-2,1e-1,1.0]; "
            "p = linspace(0, max_radius, 4096); mid from manifest; "
            "note: p <= radius < mid over this grid, so the spec box "
            "covers positive args only - the negative side of actual gl "
            "arguments is covered by the derived-arg groups"}
    result["gr_grid"] = gr_grid
    for name, rec in work.items():
        x = rec["x"]
        rec["host_np"] = np.exp(x)
        rec["host_fast"] = np.array([exp_fast(v) for v in x],
                                    dtype=np.float64)
        rec["host_np_vs_fastmath"] = _diff_stats(
            x, rec["host_np"], rec["host_fast"])
        print(json.dumps({"stage": "box_host", "box": name,
                          "n": int(x.size),
                          "np_vs_fastmath_diff":
                              rec["host_np_vs_fastmath"]["n_diff"]}),
              flush=True)
        # seed the result entry so host facts survive a device-unavailable
        # path (device fields are added below when a device is present)
        result[name] = {"construction": rec["construction"],
                        "n": int(x.size),
                        "host_np_vs_fastmath": rec["host_np_vs_fastmath"]}

    # --- derived actual exp arguments of the rep4000 structure -------------
    indptr, ids, dist, d_weights = _rep4000()
    cutoff, beta, plateau, mid, gr = 1500.0, 0.001, 0.0, 500.0, 0.01
    knn_n = 1
    acc = {k: [] for k in
           ("ge_positive", "gl", "kn_exponential", "kn_logistic")}
    for o in range(indptr.shape[0] - 1):
        lo, hi = int(indptr[o]), int(indptr[o + 1])
        ga, gla, ke, kl, _nw, _kw = _derived_args(
            dist[lo:hi], d_weights[ids[lo:hi]], cutoff, beta, plateau,
            mid, gr, knn_n)
        acc["ge_positive"].append(ga)
        acc["gl"].append(gla)
        acc["kn_exponential"].append(ke)
        acc["kn_logistic"].append(kl)
    groups = {k: np.concatenate(v).astype(np.float64)
              for k, v in acc.items()}
    for k, x in groups.items():
        work[f"group:{k}"] = {
            "x": x,
            "host_fast": np.array([exp_fast(v) for v in x],
                                  dtype=np.float64)}
    result["rep4000_derived_args"] = {
        "structure": {"origins": int(indptr.size - 1),
                      "nnz": int(ids.size), "cutoff": cutoff,
                      "filtered_nnz": int(np.count_nonzero(dist <= cutoff)),
                      "seed": 20261012,
                      "builder": "probes/gpu_phase_decomposition.py "
                                 "_rep4000 (same seed)"},
        "fold_params": {"gravity_beta": beta, "gravity_plateau": plateau,
                        "gravity_logistic_midpoint": mid,
                        "gravity_growth_rate": gr,
                        "knn_weights_n": knn_n},
        "source_ref": "delivered host fold "
                      "src/urban_network_analysis/Engines/Accessibility.py "
                      "reach_gravity_knn_access (njit fastmath); device "
                      "mirror _gpu_fold.py:120-121,164-169",
        "constructions": {
            "ge_positive": "+beta * maximum(0, d - plateau); the fold "
                           "computes w / exp(arg)",
            "gl": "-gr * ((d - plateau) - mid)",
            "kn_exponential": "-beta * maximum(0, d_cropped - plateau)",
            "kn_logistic": "-gr * ((d_cropped - plateau) - mid)"},
        "groups": {k: {"n": int(x.size),
                       "n_unique_bits":
                           int(np.unique(_ulp_view(x)).size)}
                   for k, x in groups.items()},
    }
    print(json.dumps({"stage": "derived_groups",
                      "sizes": {k: int(x.size)
                                for k, x in groups.items()}}), flush=True)

    # --- host-only certification leg (runs with or without a device) -------
    try:
        from urban_network_analysis.Engines.Accessibility import (  # noqa: E402
            reach_gravity_knn_access)
        rng = np.random.default_rng(20261012)
        cases = [
            ("small3_distinct",
             np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
             np.array([1.0, 0.9, 0.8, 1.0, 0.9, 0.8])),
            ("rand96_unrounded", rng.uniform(0.05, 20.0, 96),
             np.round(rng.uniform(0.25, 2.0, 96), 3)),
        ]
        cert = _certify(reach_gravity_knn_access, cases)
    except Exception as exc:  # recorded gap, never a silent pass
        cert = {"certified": False, "n_checks": 0,
                "arg_bits_identical": False, "outputs_bit_equal": False,
                "gaps": [f"certification leg failed: "
                         f"{type(exc).__name__}: {exc}"],
                "cases": [], "method": "not established"}
    cert["scope_note"] = ("host-only: validates the derived-arg "
                          "construction against the DELIVERED host fold, "
                          "not the device")
    result["derivation_certification"] = cert
    print(json.dumps({"stage": "certification",
                      "certified": cert["certified"],
                      "arg_bits_identical": cert["arg_bits_identical"],
                      "outputs_bit_equal": cert["outputs_bit_equal"],
                      "n_checks": cert.get("n_checks", 0),
                      "gaps": cert["gaps"]}), flush=True)

    # --- device leg (gated; typed blocker, never simulated) -----------------
    try:
        from numba import cuda
        if not cuda.is_available():
            raise RuntimeError("no CUDA device / driver init failed")

        @cuda.jit
        def _k(inp, outp):
            i = cuda.grid(1)
            if i < inp.size:
                outp[i] = math.exp(inp[i])

        def dev_exp(x):
            d_x = cuda.to_device(x)
            d_y = cuda.device_array_like(x)
            n = x.size
            _k[(n + 127) // 128, 128](d_x, d_y)
            cuda.synchronize()
            return d_y.copy_to_host()

        dev = cuda.get_current_device()
        try:
            uuid = str(dev.uuid)
        except Exception:
            uuid = None
        result["device"] = {
            "name": dev.name.decode() if isinstance(dev.name, bytes)
            else str(dev.name),
            "uuid": uuid}

        for name, rec in work.items():
            devy = dev_exp(rec["x"])
            st = _divergence_census(rec["x"], devy, rec["host_fast"])
            if name.startswith("group:"):
                st["device_vs_np"] = None  # group law: exp feed is the
                # fastmath-host lowering; np.exp context already in boxes
            else:
                st["device_vs_np"] = _diff_stats(
                    rec["x"], devy, rec["host_np"])
            if name.startswith("group:"):
                result["rep4000_derived_args"]["groups"][
                    name.split(":", 1)[1]]["device_vs_fastmath"] = st
            else:
                result[name]["device_vs_fastmath"] = st
            print(json.dumps({"stage": "device_eval", "target": name,
                              "n_diff":
                                  st["n_diff_device_vs_fastmath"],
                              "n_unique_args_diff":
                                  st["n_unique_args_diff"]}), flush=True)
    except Exception as exc:  # typed blocker, never simulated
        result["status"] = "device_unavailable"
        result["device"] = {"status": "device_unavailable",
                            "error": repr(exc)}
        result["note"] = ("device-dependent classifications absent; the "
                          "host legs and the host-only certification "
                          "above are real and stand alone")

    result["law"] = ("census only; no admission domain declared; 1 ULP "
                     "divergence is a recorded fact, not a tolerance; "
                     "never fed to the CPU 10x gate")
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(args.out),
                      "certified":
                          result["derivation_certification"]["certified"],
                      "status": result.get("status", "device_ok")}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
