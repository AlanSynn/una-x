"""G11-Delta fold-cell crossover driver (preregistration Amendment A1).

Both arms run in ONE job on the same A100 node (matched host class,
matched core allocation, same I/O root):

  CPU arm: the INSTALLED engine fold reach_gravity_knn_access (the exact
    per-row service RunBatch w4t1 q64 cache-off invokes), driven by 4
    spawn worker processes x 1 numba thread, bounded outstanding-queue
    depth 64, nothing memoized across rows or repetitions (cache off).
  GPU arm: execute_fold_region through gpu.kernels.device_exec.DeviceRuntime
    (K0/K1 device, K2/K3 host tail), PTX REUSED from the rc=0 gate verdict
    (durable gate artifact), durable gate store una_legacy.fold.default
    (open; evidence re-verified per dispatch), association "seq".

Timing law (A1): per (arm, W, rep), wall around [input preparation ->
full cell service for all 32 rows -> outputs resident and synchronized]
(+ runtime close() on the GPU arm).  Repetition 1 is cold and INCLUDED
in the median of 7; nothing is subtracted.  Import/setup times are
recorded separately, never inside a wall.

Integrity law (A1): a serial engine-fold reference per W is computed
outside both arms' walls; BOTH arms' rep-1 outputs must match it
bit-for-bit per row, else that (arm, W) comparison is INVALID (recorded,
never averaged in).

Admission law (A2): admission into CapabilityRegistryEntry is PROPOSED
only if EVERY frozen width is inside the cell's engine-bit-parity domain
(untimed probe: region vs engine bit-equal on all rows/outputs) AND the GPU
median-of-7 is strictly below the CPU median-of-7 at every timed width.
Out-of-domain widths are a designed refusal recorded with probe evidence;
the gate default stays CPU.  No 10x claim arises from this driver.
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")   # w4t1: one thread per worker

import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

RUN = Path(__file__).resolve().parent
REPO = "/u/dsynn/projects/una-x/repo"
PTX = RUN / "fold_kernels_compute_80.ptx"
GATE_STORE = RUN / "gate_store.json"
RESULTS = RUN / "crossover_results.json"

sys.path.insert(0, REPO)

from urban_network_analysis.Engines.AccessibilityWElevation import (  # noqa: E402
    reach_gravity_knn_access as engine_fold,
)

WIDTHS = (64, 4096, 65536, 262144)
ROWS = 32
REPS = 7
SEED = 20261003
QUEUE_DEPTH = 64
KNW = np.linspace(1.0, 0.25, 8)          # frozen knn weight vector
# (decay, start, stop, cutoff) — frozen row groups
GROUPS = (("exponential", 0, 11, 400.0),
          ("logistic", 11, 22, 800.0),
          ("linear", 22, 32, 1600.0))
BETA, PLATEAU, MIDPOINT, GROWTH = 0.001, 0.0, 500.0, 1.0
_KEYS = ("reach", "gravity_exponential", "gravity_logistic", "knn_access")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bits(x):
    return np.float64(x).tobytes().hex()


def group_of(i):
    return next(g for g in GROUPS if g[1] <= i < g[2])


def make_rows(width):
    """Frozen workload bytes (identical on both arms, identical per rep)."""
    rng = np.random.default_rng([SEED, width])
    rows = np.ascontiguousarray(rng.uniform(1.0, 2000.0, (ROWS, width)))
    weights = np.ascontiguousarray(rng.uniform(0.5, 2.0, width))
    return rows, weights


def engine_call(row, weights, cutoff, decay):
    """The mirrored engine call (Amendment A1 scalar semantics)."""
    return engine_fold(row, weights, np.float64(cutoff), BETA, PLATEAU,
                       MIDPOINT, 0.0, KNW, BETA, decay, MIDPOINT, GROWTH,
                       GROWTH)


def _cpu_row(task):
    row, weights, cutoff, decay = task
    return engine_call(row, weights, cutoff, decay)


def reference_values(rows, weights):
    """Serial engine fold over all rows (integrity reference; untimed)."""
    return [tuple(bits(v) for v in
                  engine_call(rows[i], weights, group_of(i)[3],
                              group_of(i)[0]))
            for i in range(ROWS)]


def cpu_arm(rows, weights, pool):
    """Full-service CPU wall: preparation + 4-worker batch of 32 rows."""
    t0 = time.perf_counter()
    tasks = [(rows[i], weights, group_of(i)[3], group_of(i)[0])
             for i in range(ROWS)]
    # bounded outstanding queue (depth 64): never more than 64 in flight
    inflight = []
    values = [None] * ROWS
    for k, task in enumerate(tasks):
        while len(inflight) >= QUEUE_DEPTH:
            idx, fut = inflight.pop(0)
            values[idx] = fut.result()
        inflight.append((k, pool.submit(_cpu_row, task)))
    for idx, fut in inflight:
        values[idx] = fut.result()
    values = [tuple(bits(v) for v in row) for row in values]
    return time.perf_counter() - t0, values


def gpu_arm(rows, weights):
    """Full-service GPU wall: preparation + context/module + dispatches +
    outputs resident + runtime close()."""
    from gpu.kernels.device_exec import DeviceRuntime
    from gpu.runtime.region import (FoldCell, FoldRegionRequest,
                                    FoldRowScalars, execute_fold_region)
    t0 = time.perf_counter()
    rows_c = np.ascontiguousarray(rows)
    weights_c = np.ascontiguousarray(weights)
    rt = DeviceRuntime()
    try:
        rt.load_ptx(PTX)
        flat = {k: [] for k in _KEYS}
        for decay, s, e, cutoff in GROUPS:
            scalars = FoldRowScalars(
                cutoff=cutoff, gravity_beta=BETA, gravity_plateau=PLATEAU,
                gravity_logistic_midpoint=MIDPOINT,
                gravity_growth_rate=GROWTH, knn_decay=decay, knn_weights=KNW)
            request = FoldRegionRequest(
                distance_rows=tuple(rows_c[i] for i in range(s, e)),
                d_weights=weights_c, scalars=scalars,
                cell=FoldCell(cell_id="una_legacy.fold.default",
                              association="seq", bit_validated=True),
                tile_origins=8, gate_store=str(GATE_STORE))
            rec = execute_fold_region(request, context=rt)
            for k in _KEYS:
                flat[k].extend(np.asarray(getattr(rec, k)).tolist())
        values = [tuple(bits(flat[k][i]) for k in _KEYS) for i in range(ROWS)]
    finally:
        rt.close()
    return time.perf_counter() - t0, values


def compare(ref, got):
    """Per-row bit comparison; returns (equal, mismatch_detail)."""
    if len(got) != ROWS:
        return False, f"row count {len(got)} != {ROWS}"
    bad = [(i, k, ref[i][j], got[i][j])
           for i in range(ROWS) for j, k in enumerate(_KEYS)
           if ref[i][j] != got[i][j]]
    return (not bad), bad[:4]


def preflight(rows, weights):
    """Host twins (seq) vs engine fold, one row per decay, W=4096, untimed."""
    from gpu.runtime.region import (FoldRowScalars, run_host_stage_composition)
    out = {}
    for decay, s, _e, cutoff in GROUPS:
        row = np.ascontiguousarray(rows[s])
        scalars = FoldRowScalars(
            cutoff=cutoff, gravity_beta=BETA, gravity_plateau=PLATEAU,
            gravity_logistic_midpoint=MIDPOINT, gravity_growth_rate=GROWTH,
            knn_decay=decay, knn_weights=KNW)
        rec = run_host_stage_composition((row,), weights, scalars, "seq")
        twin = [bits(getattr(rec, k)[0]) for k in _KEYS]
        eng = [bits(v) for v in engine_call(row, weights, cutoff, decay)]
        out[decay] = {"twin": twin, "engine": eng,
                      "match": twin == eng}
    return out


def tree_sweep(rows, weights):
    """A2 (c): per decay group, first row — which association trees
    reproduce the engine's bits per output on THIS host's codegen."""
    from gpu.kernels.host_twins import REDUCTION_TREES, fold_host_reference
    out = {}
    for decay, s, _e, cutoff in GROUPS:
        row = np.ascontiguousarray(rows[s])
        eng = [bits(v) for v in engine_call(row, weights, cutoff, decay)]
        hits = {k: [] for k in _KEYS}
        for tree in REDUCTION_TREES:
            outs = fold_host_reference(row, weights, cutoff, BETA, PLATEAU,
                                       MIDPOINT, GROWTH, decay, KNW, tree)
            for k, v in zip(_KEYS, outs):
                if bits(v) == eng[_KEYS.index(k)]:
                    hits[k].append(tree)
        out[decay] = hits
    return out


def parity_probe(rows, weights, ref):
    """A2 (b): region on the device (gate store open) over all 32 rows,
    compared bit-for-bit against the engine reference.  Returns
    (in_domain, mismatch_detail)."""
    _wall, values = gpu_arm(rows, weights)
    return compare(ref, values)


def pins(reps):
    import gpu
    import numba
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    return {
        "repo_head": head,
        "gpu_package": getattr(gpu, "__file__", "?"),
        "engine_module": engine_fold.__module__,
        "python": platform.python_version(),
        "numba": numba.__version__,
        "numpy": np.__version__,
        "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS", "<unset>"),
        "ptx_sha256": sha256_file(PTX),
        "gate_store_sha256": sha256_file(GATE_STORE),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "seed": SEED, "rows": ROWS, "reps": reps,
        "queue_depth": QUEUE_DEPTH,
        "widths": list(WIDTHS),
        "knn_weights": KNW.tolist(),
        "groups": [list(g) for g in GROUPS],
        "scalars": {"beta": BETA, "plateau": PLATEAU, "midpoint": MIDPOINT,
                    "growth": GROWTH, "knn_plateau": 0.0, "knn_beta": BETA},
    }


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="*",
                    default=[int(w) for w in WIDTHS])
    ap.add_argument("--reps", type=int, default=REPS)
    return ap.parse_args()


def main():
    ns = parse_args()
    reps = ns.reps
    result = {"schema": "una-g11-xover-1",
              "law": "PREREGISTRATION.md Amendment A1",
              "pins": None, "preflight": None, "widths": {},
              "verdict": None}

    rows4k, w4k = make_rows(4096)
    result["preflight"] = preflight(rows4k, w4k)
    result["pins"] = pins(reps)
    print("== pins ==", json.dumps(result["pins"], indent=1))
    print("== preflight (host twins vs engine, untimed) ==",
          json.dumps({k: v["match"] for k, v in
                      result["preflight"].items()}))

    ctx = __import__("multiprocessing").get_context("spawn")
    for width in ns.widths:
        rows, weights = make_rows(width)
        entry = result["widths"].setdefault(
            str(width), {"cpu": {"rep_walls": []}, "gpu": {"rep_walls": []}})

        t0 = time.perf_counter()
        ref = reference_values(rows, weights)
        entry["reference_s"] = time.perf_counter() - t0

        # A2: untimed parity probe (region on device vs engine reference)
        in_domain, detail = parity_probe(rows, weights, ref)
        entry["parity_domain"] = {"in_domain": in_domain}
        if not in_domain:
            entry["parity_domain"]["probe_mismatch_sample"] = detail
            entry["parity_domain"]["tree_sweep"] = tree_sweep(rows, weights)
            entry["skipped_by_parity_domain"] = True
            print(f"== W={width}: OUT OF PARITY DOMAIN "
                  f"(region/seq vs engine mismatch; sample {detail}); "
                  "timed arms skipped (A2 designed refusal)")
            continue

        t0 = time.perf_counter()
        pool = ProcessPoolExecutor(max_workers=4, mp_context=ctx)
        entry["cpu"]["setup_s"] = time.perf_counter() - t0
        for r in range(reps):
            wall, values = cpu_arm(rows, weights, pool)
            entry["cpu"]["rep_walls"].append(wall)
            if r == 0:
                ok, detail = compare(ref, values)
                entry["integrity"] = {"cpu_rep1_match": ok}
                if not ok:
                    entry["integrity"]["cpu_rep1_mismatch"] = detail
        pool.shutdown()

        for r in range(reps):
            wall, values = gpu_arm(rows, weights)
            entry["gpu"]["rep_walls"].append(wall)
            if r == 0:
                ok, detail = compare(ref, values)
                entry["integrity"] = entry.get("integrity", {})
                entry["integrity"]["gpu_rep1_match"] = ok
                if not ok:
                    entry["integrity"]["gpu_rep1_mismatch"] = detail

        entry["cpu"]["median_s"] = statistics.median(entry["cpu"]["rep_walls"])
        entry["gpu"]["median_s"] = statistics.median(entry["gpu"]["rep_walls"])
        valid = (entry["integrity"].get("cpu_rep1_match")
                 and entry["integrity"].get("gpu_rep1_match"))
        entry["valid"] = valid
        entry["gpu_win"] = bool(valid and
                                entry["gpu"]["median_s"]
                                < entry["cpu"]["median_s"])
        print(f"== W={width}: cpu_med={entry['cpu']['median_s']:.4f}s "
              f"gpu_med={entry['gpu']['median_s']:.4f}s "
              f"valid={valid} gpu_win={entry['gpu_win']}")

    widths = result["widths"]
    in_domain_all = all(w["parity_domain"]["in_domain"]
                        for w in widths.values())
    timed = [w for w in widths.values() if "valid" in w]
    all_valid = bool(timed) and all(w["valid"] for w in timed)
    all_win = bool(timed) and all(w["gpu_win"] for w in timed)
    result["verdict"] = {
        "all_widths_in_parity_domain": in_domain_all,
        "timed_widths": [k for k, v in widths.items() if "valid" in v],
        "skipped_widths": [k for k, v in widths.items()
                           if v.get("skipped_by_parity_domain")],
        "timed_comparisons_valid": all_valid,
        "gpu_median_win_at_every_timed_width": all_win,
        "admission_proposed": bool(in_domain_all and all_valid and all_win),
        "law": "Amendment A2: admission into CapabilityRegistryEntry is "
               "PROPOSED only if EVERY frozen width is inside the cell's "
               "engine-bit-parity domain (region vs engine bit-equal on all "
               "rows/outputs) AND the GPU median-of-7 beats the CPU "
               "median-of-7 at every timed width. Out-of-domain widths are "
               "a designed refusal recorded with probe evidence; the gate "
               "default stays CPU. No 10x claim arises from this crossover.",
    }
    RESULTS.write_text(json.dumps(result, indent=1) + "\n")
    print("== verdict ==", json.dumps(result["verdict"], indent=1))
    print("RESULTS_WRITTEN", RESULTS)
    return 0




if __name__ == "__main__":
    sys.exit(main())
