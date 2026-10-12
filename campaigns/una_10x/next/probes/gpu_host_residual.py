"""Direction-4 probe: attribute the host residual of the G10 full-service bridge.

PLAN_GPU.md §5/§6: at rep4000 the full-service bridge wall is 0.13718 s
median while the measured device phases total 0.00345 s — ~133 ms is
unattributed host-side bridge work.  This probe re-executes the per-call
host work of ``gpu_replay_fold`` (Engines/_gpu_fold.py, the G10 bridge)
SECTION BY SECTION, bracketing each section with ``perf_counter`` in the
exact order the source executes it, and reports per-section medians, their
sum, the full-service bridge median, and the still-unattributed remainder.
Sections are named after what the source actually does; no source file is
edited — private helpers (``_req``, ``_phase_b``, ``_kernel``) are the very
code the bridge runs.

Walls only: not a route claim and never fed to the CPU 10x gate (hardware
addition, recorded separately per gpu_region.json).  Without a CUDA device
this exits with a typed ``device_unavailable`` status — a recorded blocker,
never a simulated pass.

Run (device node):
    PYTHONPATH=src python probes/gpu_host_residual.py --out <out.json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _rep4000(seed: int = 20261012):
    # identical builder to probes/gpu_phase_decomposition.py (same seed ->
    # same structure; the phase numbers this probe attributes were on it)
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


def _section_replay(g, cuda, indptr, ids, dist, d_weights, knn_weights,
                    cutoff, gravity_beta, gravity_plateau,
                    gravity_logistic_midpoint, gravity_growth_rate,
                    knn_decay):
    """One re-execution of gpu_replay_fold's per-call sequence, each source
    section bracketed.  Returns the per-section seconds list (source order)."""
    t = []
    # section 1: input layout normalization (5x ascontiguousarray)
    t0 = time.perf_counter()
    indptr = np.ascontiguousarray(indptr)
    ids_all = np.ascontiguousarray(ids)
    dist_all = np.ascontiguousarray(dist)
    d_w = np.ascontiguousarray(d_weights)
    knn_w = np.ascontiguousarray(knn_weights)
    t1 = time.perf_counter()
    t.append(t1 - t0)
    # section 2: input dtype law checks (5x _req)
    g._req(indptr, np.int64, "indptr")
    g._req(ids_all, np.int64, "ids_all")
    g._req(dist_all, np.float64, "dist_all")
    g._req(d_w, np.float64, "d_weights")
    g._req(knn_w, np.float64, "knn_weights")
    t2 = time.perf_counter()
    t.append(t2 - t1)
    # section 3: shape / endpoint validation
    if indptr.ndim != 1 or indptr.shape[0] < 1:
        raise g.GpuRefused("indptr must be a non-empty 1-D int64 array")
    if indptr[0] != 0 or indptr[-1] != ids_all.shape[0]:
        raise g.GpuRefused("indptr endpoints must bound ids_all")
    origin_count = indptr.shape[0] - 1
    nnz = ids_all.shape[0]
    if dist_all.shape[0] != nnz:
        raise g.GpuRefused("dist_all must be packed to ids_all")
    t3 = time.perf_counter()
    t.append(t3 - t2)
    # section 4: decay code + scalar normalization / finiteness
    decay = g._DECAY_CODES.get(knn_decay, 2)
    cutoff = float(cutoff)
    for v in (cutoff, gravity_beta, gravity_plateau,
              gravity_logistic_midpoint, gravity_growth_rate):
        fv = float(v)
        if not np.isfinite(fv):
            raise g.GpuRefused("fold scalars must be finite")
    t4 = time.perf_counter()
    t.append(t4 - t3)
    # section 5: Phase B host structural scan (pure-Python O(origins+nnz))
    rc = g._phase_b(indptr, ids_all, d_w.shape[0], nnz)
    if rc == 2:
        raise g.GpuRefused("slice-order corruption (Phase B refusal)")
    if rc != 0:
        raise g.GpuRefused("structural (Phase B refusal)")
    t5 = time.perf_counter()
    t.append(t5 - t4)
    # section 6: device availability gate
    g.require_device()
    t6 = time.perf_counter()
    t.append(t6 - t5)
    # section 7: kernel lookup (compiled once, cached global)
    kernel = g._kernel()
    t7 = time.perf_counter()
    t.append(t7 - t6)
    # section 8: device input upload (5x to_device)
    d_indptr = cuda.to_device(indptr)
    d_dist = cuda.to_device(dist_all)
    d_ids = cuda.to_device(ids_all)
    d_wg = cuda.to_device(d_w)
    d_knn = cuda.to_device(knn_w)
    t8 = time.perf_counter()
    t.append(t8 - t7)
    # section 9: device output + scratch allocation (10x device_array)
    d_reach = cuda.device_array(origin_count, dtype=np.float64)
    d_ge = cuda.device_array(origin_count, dtype=np.float64)
    d_gl = cuda.device_array(origin_count, dtype=np.float64)
    d_kn = cuda.device_array(origin_count, dtype=np.float64)
    d_kept = cuda.device_array(origin_count, dtype=np.int64)
    d_pairs_d = cuda.device_array(nnz, dtype=np.float64)
    d_pairs_i = cuda.device_array(nnz, dtype=np.int64)
    d_scd = cuda.device_array(nnz, dtype=np.float64)
    d_scw = cuda.device_array(nnz, dtype=np.float64)
    t9 = time.perf_counter()
    t.append(t9 - t8)
    # section 10: kernel launch
    blocks = (origin_count + 127) // 128
    kernel[blocks, 128](
        d_indptr, d_dist, d_ids, d_wg, d_knn,
        cutoff, float(gravity_beta), float(gravity_plateau),
        float(gravity_logistic_midpoint), float(gravity_growth_rate),
        int(decay),
        d_reach, d_ge, d_gl, d_kn, d_kept,
        d_pairs_d, d_pairs_i, d_scd, d_scw)
    t10 = time.perf_counter()
    t.append(t10 - t9)
    # section 11: synchronize BEFORE any result is published (region law)
    cuda.synchronize()
    t11 = time.perf_counter()
    t.append(t11 - t10)
    # section 12: result download (5x copy_to_host)
    reach = d_reach.copy_to_host()
    ge = d_ge.copy_to_host()
    gl = d_gl.copy_to_host()
    kn = d_kn.copy_to_host()
    kept = d_kept.copy_to_host()
    t12 = time.perf_counter()
    t.append(t12 - t11)
    assert reach.shape[0] == origin_count and kept.shape[0] == origin_count
    return t


_SECTIONS = [
    ("input_layout_normalization_ascontiguousarray_x5",
     "_gpu_fold.py:213-217"),
    ("input_dtype_law_checks_req_x5", "_gpu_fold.py:218-222"),
    ("shape_endpoint_validation", "_gpu_fold.py:223-230"),
    ("decay_code_scalar_normalization_finiteness",
     "_gpu_fold.py:234-240"),
    ("phase_b_structural_scan", "_gpu_fold.py:241-245 + _phase_b:185-199"),
    ("device_availability_gate", "_gpu_fold.py:246 + require_device:76-87"),
    ("kernel_lookup_cached", "_gpu_fold.py:253 + _kernel:90-175"),
    ("device_input_upload_h2d_x5", "_gpu_fold.py:254-258"),
    ("device_output_scratch_alloc_x10", "_gpu_fold.py:259-269"),
    ("kernel_launch", "_gpu_fold.py:270-278"),
    ("device_synchronize", "_gpu_fold.py:279-280"),
    ("result_download_d2h_x5", "_gpu_fold.py:281-285"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-reps", type=int, default=6)
    ap.add_argument("--section-reps", type=int, default=9)
    args = ap.parse_args()

    result = {"schema": "una10x-gpu-host-residual-1",
              "purpose": "Direction 4 prerequisite (PLAN_GPU.md §6); "
                         "host residual attribution; walls only",
              "python": sys.executable}
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

    indptr, ids, dist, d_weights = _rep4000()
    knn_weights = np.array([1.0], dtype=np.float64)
    cutoff = 1500.0
    result["shape"] = {"origins": int(indptr.size - 1),
                       "nnz": int(ids.size), "cutoff": cutoff}

    from urban_network_analysis.Engines import _gpu_fold as g

    if not g.cuda_available():
        result["status"] = "device_unavailable"
        result["note"] = ("typed hardware blocker; no residual numbers "
                          "produced, nothing simulated")
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps({"status": "device_unavailable"}), flush=True)
        return 0
    g.require_device()
    from numba import cuda

    dev = cuda.get_current_device()
    try:
        uuid = str(dev.uuid)
    except Exception:
        uuid = None
    result["device"] = {"name": dev.name.decode()
                        if isinstance(dev.name, bytes) else str(dev.name),
                        "uuid": uuid}
    scalars = (cutoff, 0.001, 0.0, 500.0, 0.01)
    result["protocol"] = {
        "full_service_reps": args.full_reps,
        "full_service_warmup": False,
        "section_reps": args.section_reps,
        "section_warmup_passes": 1,
        "section_order_mirrors_source": True,
        "source_ref": "src/urban_network_analysis/Engines/_gpu_fold.py "
                      "gpu_replay_fold",
        "caveat": "sections are timed in isolated re-execution on warm "
                  "state; the sum of isolated sections need not equal the "
                  "in-context cost (caches/allocator/dispatch) — the "
                  "remainder quantifies exactly that gap",
    }

    # --- reference full-service wall (phase-probe protocol: no warmup) ---
    full = []
    for i in range(args.full_reps):
        t0 = time.perf_counter()
        g.gpu_replay_fold(indptr, ids, dist, d_weights, cutoff,
                          0.001, 0.0, 500.0, 0.01, knn_weights, "logistic")
        full.append(time.perf_counter() - t0)
        print(json.dumps({"stage": "full_service", "rep": i,
                          "s": full[-1]}), flush=True)
    full.sort()
    result["full_service_bridge_s"] = {"median": full[len(full) // 2],
                                       "min": full[0], "max": full[-1]}

    # --- section-by-section host re-execution ----------------------------
    for _ in range(1):  # one untimed warmup pass
        _section_replay(g, cuda, indptr, ids, dist, d_weights, knn_weights,
                        *scalars, "logistic")
    per_section = [[] for _ in _SECTIONS]
    per_rep_total = []
    for r in range(args.section_reps):
        ts = _section_replay(g, cuda, indptr, ids, dist, d_weights,
                             knn_weights, *scalars, "logistic")
        for i, v in enumerate(ts):
            per_section[i].append(v)
        per_rep_total.append(sum(ts))
        print(json.dumps({"stage": "sections", "rep": r, "sum_s": sum(ts),
                          "phase_b_s": ts[4]}), flush=True)

    rows = []
    for (name, ref), vals in zip(_SECTIONS, per_section):
        vs = sorted(vals)
        rows.append({"section": name, "source_ref": ref,
                     "median_s": vs[len(vs) // 2], "min_s": vs[0],
                     "max_s": vs[-1]})
    result["host_sections"] = rows
    total_med = sorted(per_rep_total)
    sections_sum = sum(r["median_s"] for r in rows)
    result["replayed_sequence_median_s"] = total_med[len(total_med) // 2]
    result["host_sections_sum_median_s"] = sections_sum
    fs_med = result["full_service_bridge_s"]["median"]
    result["unattributed_residual_s"] = fs_med - sections_sum
    result["unattributed_share_of_full_service"] = \
        (fs_med - sections_sum) / fs_med
    result["law"] = ("walls only; hardware addition recorded separately; "
                     "never fed to the CPU 10x gate; not a route claim")

    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({
        "out": str(args.out),
        "full_service_median_s": fs_med,
        "phase_b_median_s": rows[4]["median_s"],
        "sections_sum_median_s": sections_sum,
        "unattributed_residual_s": result["unattributed_residual_s"]}),
        flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
