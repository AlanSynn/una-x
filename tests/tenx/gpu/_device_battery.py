"""G11 device battery: real-hardware bit parity + full-service walls.

Run ONLY on a CUDA node (via salloc):
    module load cuda/<ver>
    ./_py tests/tenx/gpu/_device_battery.py [out.json]

Without a device this prints NO result and raises GpuUnavailable - a
blocker, never a simulated pass.  (A CPU diagnostic mode exists for
plumbing checks only: env UNA_GPU_BATTERY_CPU_SUBSTITUTE=1 substitutes
the NATIVE STRICT bridge for the device call and writes a JSON marked
cpu_substitute; it must NEVER be read as device evidence.)

Bit-comparison targets:

  - native strict fold (N00 C++ port, sequential order): expected
    bit-equal on reach everywhere (IEEE adds, same order); ge/gl/kn
    additionally require libdevice == glibc exp on the tested inputs
    - that is the G11 question, measured here.
  - delivered numba fastmath fold (the production authority): the N00
    map already showed strict-sequential divergence at general trip
    counts; duplicate-distance kn rows add the np.argsort quicksort
    tie domain (documented refusal domain).  The public call does not
    expose kept, so kept is validated only against native.

Walls: cold JIT (first device call: context+compile+launch+sync),
then warm full-service device calls (upload+launch+sync+download)
vs the delivered numba per-origin driver on the SAME node at C=1,
alternating AB/BA blocks.  GPU walls never feed the CPU 10x gate
(hardware addition, recorded separately per gpu_region.json).
"""
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from urban_network_analysis.Engines import _gpu_fold as g  # noqa: E402
from urban_network_analysis.Engines.Accessibility import (  # noqa: E402
    reach_gravity_knn_access)

C0, B1, P2, M3, G4 = 12.0, 0.35, 0.5, 0.75, 1.2
KW = np.array([1.0, 0.8, 0.6])
NAMES = ("reach", "ge", "gl", "kn", "kept")


def build_csr(origins, dest_mean, seed, dist_quant=None):
    """Cohort-representative packed CSR (trace law: strictly ascending
    DISTINCT ids); dist_quant forces duplicate distances for ties."""
    rng = np.random.default_rng(seed)
    lens = []
    id_chunks = []
    nxt = 0
    for _ in range(origins):
        k = int(rng.integers(dest_mean // 2, dest_mean + dest_mean // 2))
        id_chunks.append(np.arange(nxt, nxt + k, dtype=np.int64))
        nxt += k
        lens.append(k)
    ids = np.concatenate(id_chunks)
    indptr = np.zeros(origins + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(lens, dtype=np.int64)
    nnz = int(indptr[-1])
    dist = np.round(rng.uniform(0.05, 20.0, nnz), 4)
    if dist_quant is not None:
        dist = np.round(dist / dist_quant) * dist_quant
    w = np.round(rng.uniform(0.25, 2.0, nxt), 3)
    return indptr, ids, dist, w


def numba_driver(indptr, ids, dist, w, decay):
    """The production authority: per-origin fresh-copy public calls.

    Returns (reach, ge, gl, kn, None): the public API does not expose
    kept, so kept is validated only against the native target."""
    r_o, ge_o, gl_o, kn_o = [], [], [], []
    lo = 0
    for o in range(indptr.shape[0] - 1):
        hi = int(indptr[o + 1])
        if hi > lo:
            r, ge, gl, kn = reach_gravity_knn_access(
                dist[lo:hi].copy(), w[ids[lo:hi]].copy(),
                C0, B1, P2, M3, 0.0, KW, 0.2, decay, 0.5, G4, 0.9)
            r_o.append(r)
            ge_o.append(ge)
            gl_o.append(gl)
            kn_o.append(kn)
        else:
            r_o.append(0.0)
            ge_o.append(0.0)
            gl_o.append(0.0)
            kn_o.append(0.0)
        lo = hi
    z = lambda xs: np.array(xs, dtype=np.float64)  # noqa: E731
    return z(r_o), z(ge_o), z(gl_o), z(kn_o), None


def compare(name, a, b):
    if a.shape != b.shape:
        return {"target": name, "equal": False, "note": "shape"}
    if a.dtype == b.dtype and a.tobytes() == b.tobytes():
        return {"target": name, "equal": True}
    if a.dtype != b.dtype:
        return {"target": name, "equal": False, "note": "dtype"}
    ia = a.view(np.int64)  # both f64/i64 views are 8-byte reinterprets
    ib = b.view(np.int64)
    bad = ia != ib
    ulp = np.abs(ia[bad] - ib[bad])
    return {"target": name, "equal": False,
            "n_diff": int(bad.sum()), "n": int(a.size),
            "max_ulp": int(ulp.max()) if ulp.size else 0}


def fold_native(indptr, ids, dist, w, decay):
    """(outputs tuple) or raise - the N00 strict reference."""
    from urban_network_analysis.Engines import _native_fold as nf
    nf.ensure_built()
    return nf.native_replay_fold(indptr, ids, dist, w, C0, B1, P2, M3,
                                 G4, KW, decay)


def fold_device(indptr, ids, dist, w, decay):
    """The device call - or its CPU substitute (diagnostics only)."""
    if os.environ.get("UNA_GPU_BATTERY_CPU_SUBSTITUTE") == "1":
        from urban_network_analysis.Engines import _native_fold as nf
        nf.ensure_built()
        return nf.native_replay_fold(indptr, ids, dist, w, C0, B1, P2,
                                     M3, G4, KW, decay)
    return g.gpu_replay_fold(indptr, ids, dist, w, C0, B1, P2, M3, G4,
                             KW, decay)


def matrix_row(tag, indptr, ids, dist, w, decay):
    dev = fold_device(indptr, ids, dist, w, decay)
    row = {"case": tag, "decay": decay,
           "origins": int(indptr.shape[0] - 1), "nnz": int(ids.shape[0])}
    num = numba_driver(indptr, ids, dist, w, decay)
    row["vs_numba"] = [compare(NAMES[i], dev[i], num[i])
                       for i in range(4)]
    try:
        nat = fold_native(indptr, ids, dist, w, decay)
    except Exception as e:  # recorded blocker, not a silent skip
        row["vs_native"] = {"blocked": f"{type(e).__name__}: {e}"}
    else:
        row["vs_native"] = [compare(NAMES[i], dev[i], nat[i])
                            for i in range(5)]
    return row


def main():
    substitute = os.environ.get("UNA_GPU_BATTERY_CPU_SUBSTITUTE") == "1"
    if substitute:
        fp = ("cpu-substitute-native-bridge", (), "diagnostic")
    else:
        g.require_device()  # hard gate: no device -> no result at all
        fp = g.device_fingerprint()
    result = {
        "mode": "cpu_substitute" if substitute else "device",
        "fingerprint": {
            "device": fp[0], "compute_capability": [int(x) for x in fp[1]],
            "stack": fp[2], "python": platform.python_version(),
            "numba": __import__("numba").__version__,
            "numpy": np.__version__,
            "cpu_model": [l.split(":", 1)[1].strip()
                          for l in open("/proc/cpuinfo")
                          if "model name" in l][0],
            "budget_C": 1,
        },
        "bit_matrix": [],
        "walls": {},
    }

    # cold JIT: the FIRST device call (context init + compile + launch)
    ip, ids, dist, w = build_csr(64, 12, 3)
    t0 = time.perf_counter()
    fold_device(ip, ids, dist, w, "exponential")
    result["walls"]["cold_jit_s"] = time.perf_counter() - t0

    rep = build_csr(4000, 60, 11)
    rep_dup = build_csr(4000, 60, 11, dist_quant=0.5)
    s_ip = np.array([0, 3, 3, 6], dtype=np.int64)
    s_ids = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    s_dist = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    s_w = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    small = (s_ip, s_ids, s_dist, s_w)
    cases = [
        ("rep4000", rep, "exponential"),
        ("rep4000", rep, "logistic"),
        ("rep4000", rep, "flat"),
        ("rep4000_dupdist", rep_dup, "exponential"),
        ("small3_emptymid", small, "exponential"),
        ("small3_emptymid", small, "logistic"),
        ("small3_emptymid", small, "flat"),
    ]
    for tag, pack, decay in cases:
        result["bit_matrix"].append(
            matrix_row(tag, pack[0], pack[1], pack[2], pack[3], decay))

    # warm full-service walls: device vs numba on the same node, C=1
    ip, ids, dist, w = rep
    fold_device(ip, ids, dist, w, "exponential")  # untimed warmup
    blocks = []
    for b in range(6):
        t0 = time.perf_counter()
        numba_driver(ip, ids, dist, w, "exponential")
        ta = time.perf_counter() - t0
        t0 = time.perf_counter()
        fold_device(ip, ids, dist, w, "exponential")
        tb = time.perf_counter() - t0
        blocks.append(("AB" if b % 2 == 0 else "BA", ta, tb))
    a_med = statistics.median(x[1] for x in blocks)
    b_med = statistics.median(x[2] for x in blocks)
    result["walls"]["numba_driver_median_s"] = a_med
    result["walls"]["fold_median_s"] = b_med
    result["walls"]["ratio_fold_over_numba"] = \
        b_med / a_med if a_med else None
    result["walls"]["blocks"] = [
        {"order": o, "numba_s": ta, "fold_s": tb}
        for o, ta, tb in blocks]

    out = sys.argv[1] if len(sys.argv) > 1 else "_device_battery_result.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print("WROTE", out, "| mode:", result["mode"])
    print("device:", fp[0], "cc", list(fp[1]), "| cold_s",
          round(result["walls"]["cold_jit_s"], 4))
    for row in result["bit_matrix"]:
        nat = row["vs_native"]
        num = row["vs_numba"]
        tag = f"{row['case']} [{row['decay']}]"
        if isinstance(nat, list):
            ne = sum(1 for c in nat if c["equal"])
            bad = ",".join(c["target"] for c in nat if not c["equal"])
            print(f"{tag:>34} native {ne}/5 equal ({bad or 'all'})")
        else:
            print(f"{tag:>34} native BLOCKED {nat['blocked'][:60]}")
        ne = sum(1 for c in num if c["equal"])
        bad = ",".join(c["target"] for c in num if not c["equal"])
        print(f"{'':>34} numba  {ne}/4 equal ({bad or 'all'})")
    print("walls: numba", a_med, "fold", b_med, "ratio",
          result["walls"]["ratio_fold_over_numba"])


if __name__ == "__main__":
    main()
