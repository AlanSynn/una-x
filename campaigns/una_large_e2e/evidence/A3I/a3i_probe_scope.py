"""A3I phase-1 admission probe (campaign una_large_e2e, task A3I).

Measurement only. Descriptive record for h04-reviewer's admission
ruling; this probe issues NO gate verdicts and promotes NO candidate.

Legs (per the routed A3I admission plan):
  B  32-origin expansion probe: an instrumented copy of the real
     `_a1_scope_search` kernel (counters only) run on the REAL captured
     graph-engine arrays for the 32 sampled origins recorded by the
     h05_profile.py warm session.
  C  compiled init micro-measure: replica of the exact per-origin
     private-scope init expression at the observed nd_node_count.

Provenance of the instrumented copy: derived from
src/urban_network_analysis/Engines/_large_access_scratch.py
`_a1_scope_search` (:178-248 at content sha256
2d3b77363a5d42050024e335b5e2bedf8c8088a312d6e163666596ce42894e23).
Diff vs source: one added signature line (`counters,`), eight
counter-increment lines (`counters[k] += 1` after each heappop/heappush,
each phase-one incidence scan, each staged eligible pair, and each
phase-two label assignment), and 13 documentation lines omitted (the
source function's 8-line docstring and 5 comment lines); the 46
executable lines are otherwise identical (zero executable drift).
Amendment history vs the a3i_t1 bytes (sha256 f964e38c..., custody copy
campaign_data/a3i_census/interim_f964e38c_a3i_probe_scope.py): the
admission-guard call moved behind a minimal njit wrapper (the guard's
@overload resolves only in nopython context; the t1 failure), the
window key read fixed to the producer's record layout, and thread-count
provenance made explicit. Full line census is in the routed amendment
declaration.

Validation per sampled origin: the instrumented output must satisfy
array-equality with the REAL wheel kernel's output for the same inputs,
and every untouched entry of the real output must equal 1 + cutoff
exactly. Candidate/retained-destination and gradient fields named in
BENCHMARKS.md are A2/F-probe fields this kernel does not produce; they
are recorded as explicit nulls with a mapping note.

Writes: ONLY the --out record. The probe's own dispatchers are
cache=False; the REAL wheel kernel it imports is cache=True and
compiles into NUMBA_CACHE_DIR (the fire declaration names a fresh
dedicated root). No other writes, no reads outside the declared inputs.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from heapq import heappop, heappush

from urban_network_analysis.Engines._large_access_scratch import (
    _a1_scope_admits,
    _a1_scope_search,
)

import numba as nb
import numpy as np

# Same JIT settings as the engine modules (NUMBA_PARALLEL False: the
# per-origin kernel is serial; the engine parallelizes across origins in
# its own prange caller, which the probe does not replicate).
NUMBA_PARALLEL = False
NUMBA_NOGIL = True
NUMBA_FASTMATH = True


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Leg B: instrumented copy of _a1_scope_search (counter increments only)
# ---------------------------------------------------------------------------
@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=False,
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
)
def _a3i_scope_search_instrumented(
    o_terminal_idxs,
    o_terminal_weights,
    adjacency_pointer,
    adjacency_vector,
    adjacency_vector_weights,
    adjacynct_vector_network_node,
    cutoff,
    d_count,
    eligible_offset,
    eligible_weight,
    counters,
):
    nd_node_count = d_count + adjacency_pointer.shape[0] - 1
    o_idx_start = o_terminal_idxs[0]
    o_idx_end = o_terminal_idxs[1]
    o_idx_start_weight = o_terminal_weights[0]
    o_idx_end_weight = o_terminal_weights[1]

    o_scope_weights = np.ones(nd_node_count, dtype=o_terminal_weights.dtype) + cutoff
    o_scope_pred = np.empty(0, dtype=o_terminal_idxs.dtype)

    o_scope_weights[o_idx_start] = o_idx_start_weight
    o_scope_weights[o_idx_end] = o_idx_end_weight

    queue = [(o_idx_start_weight, o_idx_start)]
    weight, node = heappop(queue)
    counters[0] += 1

    if o_idx_end_weight < cutoff:
        heappush(queue, (o_idx_end_weight, o_idx_end))
        counters[1] += 1

    if o_idx_start_weight < cutoff:
        heappush(queue, (o_idx_start_weight, o_idx_start))
        counters[1] += 1

    while queue:
        weight, node = heappop(queue)
        counters[0] += 1

        node_start_pointer = adjacency_pointer[node]
        node_end_pointer = adjacency_pointer[node + 1]

        n_eligible = 0
        for i in range(node_end_pointer - node_start_pointer):
            counters[2] += 1
            neighbor_weight = adjacency_vector_weights[node_start_pointer + i] + weight
            neighbor_node = adjacency_vector[node_start_pointer + i]
            if neighbor_weight <= cutoff and neighbor_weight < o_scope_weights[neighbor_node]:
                eligible_offset[n_eligible] = node_start_pointer + i
                eligible_weight[n_eligible] = neighbor_weight
                n_eligible += 1
                counters[3] += 1

        for j in range(n_eligible):
            offset = eligible_offset[j]
            neighbor_weight = eligible_weight[j]
            neighbor_node = adjacency_vector[offset]
            o_scope_weights[neighbor_node] = neighbor_weight
            counters[4] += 1
            if adjacynct_vector_network_node[offset]:
                if adjacency_pointer[neighbor_node + 1] - adjacency_pointer[neighbor_node] > 1:
                    heappush(queue, (neighbor_weight, neighbor_node))
                    counters[1] += 1

    return o_scope_weights, o_scope_pred


# ---------------------------------------------------------------------------
# Leg C: compiled replica of the exact init expression
# ---------------------------------------------------------------------------
@nb.njit(parallel=NUMBA_PARALLEL, cache=False, nogil=NUMBA_NOGIL, fastmath=NUMBA_FASTMATH)
def _a3i_init_replica(nd_node_count, cutoff):
    return np.ones(nd_node_count, dtype=np.float64) + cutoff


# ---------------------------------------------------------------------------
# Amendment (a): the engine's admission guard resolves via
# numba.extending.overload ONLY inside a nopython caller; from the
# interpreter the `pass` stub runs and returns None (the a3i_t1 leg-3
# failure). The probe calls it through this minimal njit wrapper so the
# overload resolves, mirroring the engine's integrated-route call frame
# and JIT flags.
# ---------------------------------------------------------------------------
@nb.njit(parallel=NUMBA_PARALLEL, cache=False, nogil=NUMBA_NOGIL, fastmath=NUMBA_FASTMATH)
def _a3i_scope_admits_via_numba(
    adjacency_pointer,
    adjacency_vector,
    adjacency_vector_weights,
    adjacynct_vector_network_node,
    o_terminal_idxs,
    o_terminal_weights,
    cutoff,
):
    return _a1_scope_admits(
        adjacency_pointer,
        adjacency_vector,
        adjacency_vector_weights,
        adjacynct_vector_network_node,
        o_terminal_idxs,
        o_terminal_weights,
        cutoff,
    )


def _init_micro(nd, cutoff, reps, budget_deadline):
    times = []
    for _ in range(3):
        _a3i_init_replica(nd, cutoff)
    for _ in range(reps):
        if time.perf_counter() > budget_deadline:
            return times, True
        t0 = time.perf_counter()
        _a3i_init_replica(nd, cutoff)
        times.append(time.perf_counter() - t0)
    return times, False


def _stats(xs):
    if not xs:
        return None
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean_s": float(a.mean()),
        "median_s": float(np.median(a)),
        "p10_s": float(np.percentile(a, 10)),
        "p90_s": float(np.percentile(a, 90)),
        "min_s": float(a.min()),
        "max_s": float(a.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", required=True)
    ap.add_argument("--w1-record", required=True)
    ap.add_argument("--reps-init", type=int, default=2000)
    ap.add_argument("--wall-budget-s", type=float, default=240.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_start = time.perf_counter()
    deadline = t_start + args.wall_budget_s
    REC = {
        "task": "A3I",
        "record": "admission_probe_scope",
        "role": "implementation-owner",
        "descriptive_only": "measurement for h04-reviewer's admission ruling; NO gate verdicts",
        "probe_source": os.path.abspath(__file__),
        "inputs": {},
        "validation_policy": (
            "instrumented copy vs REAL wheel kernel array-equality per origin; "
            "untouched entries == 1 + cutoff exactly; no metric recomputation "
            "(engine metric values for the sample are carried from the h05 record)"
        ),
    }
    try:
        z = np.load(args.arrays)
        ARR = {k: np.ascontiguousarray(z[k]) for k in z.files}
        adjacency_pointer = ARR["adjacency_pointer"]
        adjacency_vector = ARR["adjacency_vector"]
        adjacency_vector_weights = ARR["adjacency_vector_weights"]
        adjacynct_vector_network_node = ARR["adjacynct_vector_network_node"]
        o_terminal_idxs = ARR["o_terminal_idxs"]
        o_terminal_weights = ARR["o_terminal_weights"]
        d_terminal_idxs = ARR["d_terminal_idxs"]

        with open(args.w1_record) as f:
            W1 = json.load(f)
        metrics_path = W1["sampled_origin_metrics"]["path"]
        with open(metrics_path) as f:
            ENGINE_METRICS = json.load(f)
        sample_rows = sorted(int(k) for k in ENGINE_METRICS.keys())

        cutoff = float(W1["settings_after_window"]["search_radius"])
        d_count = int(d_terminal_idxs.shape[0])
        node_count = int(adjacency_pointer.shape[0]) - 1
        nd_node_count = d_count + node_count
        o_count = int(o_terminal_idxs.shape[0])

        REC["inputs"] = {
            "arrays_path": os.path.abspath(args.arrays),
            "arrays_sha256": _sha256_file(args.arrays),
            "w1_record_path": os.path.abspath(args.w1_record),
            "w1_record_sha256": _sha256_file(args.w1_record),
            "engine_metrics_path": os.path.abspath(metrics_path),
            "engine_metrics_sha256": _sha256_file(metrics_path),
            "w1_window_s": W1["window"]["application_window_ns"] / 1e9,
            "w1_run_id": W1.get("run_id"),
        }
        REC["shape"] = {
            "node_count": node_count,
            "d_count": d_count,
            "nd_node_count": nd_node_count,
            "o_count": o_count,
            "cutoff": cutoff,
            "sample_rows": sample_rows,
            "sample_size": len(sample_rows),
            "o_terminal_weights_dtype": str(o_terminal_weights.dtype),
        }
        pool_numba = W1.get("pool_inventory_post_window", {}).get("numba", {})
        threads_pool = pool_numba.get("get_num_threads")
        threads_process = (W1.get("process") or {}).get("numba_num_threads")
        REC["environment"] = {
            "python": sys.version.split()[0],
            "numba": nb.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "numba_num_threads_provenance": {
                "pool_inventory_post_window.numba.get_num_threads": threads_pool,
                "process.numba_num_threads": threads_process,
            },
            "probe_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
        }

        # ---- Leg C first (independent, cheap) --------------------------------
        init_dtype = str(o_terminal_weights.dtype)
        init_times, init_truncated = _init_micro(
            nd_node_count, cutoff, args.reps_init, deadline)
        REC["leg_c_init"] = {
            "expression": "np.ones(nd_node_count, dtype=o_terminal_weights.dtype) + cutoff",
            "nd_node_count": nd_node_count,
            "dtype_observed": init_dtype,
            "dtype_replica": "float64",
            "dtype_match": init_dtype == "float64",
            "reps_requested": int(args.reps_init),
            "truncated_by_budget": init_truncated,
            "stats": _stats(init_times),
            "serial_equivalent_total_s": (
                float(np.mean(init_times)) * o_count if init_times else None),
        }

        # ---- Admission guard on the captured domain --------------------------
        t0 = time.perf_counter()
        a1_admitted, a1_max_degree = _a3i_scope_admits_via_numba(
            adjacency_pointer,
            adjacency_vector,
            adjacency_vector_weights,
            adjacynct_vector_network_node,
            o_terminal_idxs,
            o_terminal_weights,
            cutoff,
        )
        REC["a1_guard_on_capture"] = {
            "admitted": bool(a1_admitted),
            "max_degree": int(a1_max_degree),
            "wall_s": time.perf_counter() - t0,
        }

        if not a1_admitted:
            REC["probe_status"] = "refused_real_route_not_admitted_on_capture"
            REC["per_origin"] = []
        else:
            REC["probe_status"] = "ok"
            eligible_offset = np.empty(int(a1_max_degree), dtype=np.int64)
            eligible_weight = np.empty(int(a1_max_degree), dtype=np.float64)
            # Bit-exact init value from the same compiled expression the
            # kernel uses (ones + cutoff under identical flags); used as
            # the untouched-entry sentinel for the validation masks.
            init_replica = _a3i_init_replica(nd_node_count, cutoff)
            per_origin = []
            stopped = None
            for row in sample_rows:
                if time.perf_counter() > deadline:
                    stopped = "wall_budget_exhausted_between_origins"
                    break
                ot = o_terminal_idxs[row]
                ow = o_terminal_weights[row]

                t0 = time.perf_counter()
                real_w, _ = _a1_scope_search(
                    ot, ow,
                    adjacency_pointer, adjacency_vector,
                    adjacency_vector_weights, adjacynct_vector_network_node,
                    cutoff, d_count, eligible_offset, eligible_weight,
                )
                real_s = time.perf_counter() - t0
                real_w = np.array(real_w)  # own copy before the next call

                counters = np.zeros(5, dtype=np.int64)
                t0 = time.perf_counter()
                inst_w, _ = _a3i_scope_search_instrumented(
                    ot, ow,
                    adjacency_pointer, adjacency_vector,
                    adjacency_vector_weights, adjacynct_vector_network_node,
                    cutoff, d_count, eligible_offset, eligible_weight,
                    counters,
                )
                inst_s = time.perf_counter() - t0

                untouched_mask = real_w == init_replica
                written = real_w[~untouched_mask]
                per_origin.append({
                    "row": int(row),
                    "real_wall_s": real_s,
                    "instrumented_wall_s": inst_s,
                    "array_equal_real_vs_instrumented": bool(np.array_equal(real_w, inst_w)),
                    "written_labels_le_cutoff": (
                        bool(np.all(written <= cutoff)) if written.size else True),
                    "written_label_count": int(written.size),
                    "touched_unique_nodes": int(written.size),
                    "touched_in_destination_segment": int(
                        np.count_nonzero(~untouched_mask[:d_count])),
                    "heap_pops": int(counters[0]),
                    "heap_pushes": int(counters[1]),
                    "neighbor_incidences_scanned": int(counters[2]),
                    "eligible_pairs_staged": int(counters[3]),
                    "labels_written": int(counters[4]),
                    "engine_metrics": ENGINE_METRICS[str(row)],
                })

            REC["per_origin"] = per_origin
            if stopped:
                REC["probe_status"] = f"truncated_{stopped}"
            REC["aggregate"] = {
                "origins_probed": len(per_origin),
                "real_wall_s_stats": _stats([p["real_wall_s"] for p in per_origin]),
                "instrumented_wall_s_stats": _stats(
                    [p["instrumented_wall_s"] for p in per_origin]),
                "heap_pops_total": int(sum(p["heap_pops"] for p in per_origin)),
                "heap_pushes_total": int(sum(p["heap_pushes"] for p in per_origin)),
                "neighbor_incidences_total": int(
                    sum(p["neighbor_incidences_scanned"] for p in per_origin)),
                "eligible_pairs_total": int(
                    sum(p["eligible_pairs_staged"] for p in per_origin)),
                "labels_written_total": int(sum(p["labels_written"] for p in per_origin)),
                "touched_unique_nodes_max": int(
                    max((p["touched_unique_nodes"] for p in per_origin), default=0)),
                "touched_in_destination_segment_max": int(
                    max((p["touched_in_destination_segment"] for p in per_origin), default=0)),
                "all_array_equal": bool(all(
                    p["array_equal_real_vs_instrumented"] for p in per_origin)),
                "all_written_labels_le_cutoff": bool(all(
                    p["written_labels_le_cutoff"] for p in per_origin)),
            }
            # BENCHMARKS.md fields this kernel does not produce: explicit nulls.
            REC["not_produced_by_this_kernel"] = {
                "candidate_destinations": None,
                "retained_destinations": None,
                "od_overlap_sizes": None,
                "gradient_finite_entry_counts": None,
                "note": ("A2 destination-scan and F-probe fields; the A1 search "
                         "kernel exposes destinations only as the scope-vector "
                         "segment [0, d_count), reported as "
                         "touched_in_destination_segment"),
            }
            # Descriptive share candidates (numerator from leg C at the mean).
            if REC["leg_c_init"]["stats"] and REC["aggregate"]["real_wall_s_stats"]:
                t_init = REC["leg_c_init"]["stats"]["mean_s"]
                window = REC["inputs"].get("w1_window_s")
                threads = threads_pool
                mean_origin = REC["aggregate"]["real_wall_s_stats"]["mean_s"]
                shares = {"t_init_mean_s": t_init}
                if window:
                    serial = t_init * o_count / float(window)
                    shares["init_share_serial_equiv_of_w1_window"] = serial
                    if isinstance(threads, (int, float)) and threads >= 1:
                        shares["init_share_wall_spread_over_recorded_threads"] = (
                            t_init * o_count / (float(threads) * float(window)))
                if mean_origin > 0:
                    shares["init_share_of_mean_per_origin_search"] = (
                        t_init / mean_origin)
                shares["note"] = (
                    "descriptive only: three readings delivered raw; the "
                    "decisive interpretation of 'complete-job service' is "
                    "h04-reviewer's ruling")
                REC["share_candidates"] = shares

        REC["wall_budget_s"] = float(args.wall_budget_s)
        REC["elapsed_total_s"] = time.perf_counter() - t_start
    except Exception:
        REC["error"] = traceback.format_exc()
        with open(args.out, "w") as f:
            json.dump(REC, f, indent=1, default=str)
        print("A3I PROBE ERROR — record written; stop-and-route", file=sys.stderr)
        return 1

    with open(args.out, "w") as f:
        json.dump(REC, f, indent=1, default=str)
    print(f"A3I probe ok: status={REC.get('probe_status')} "
          f"origins={REC.get('aggregate', {}).get('origins_probed')} "
          f"elapsed={REC['elapsed_total_s']:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
