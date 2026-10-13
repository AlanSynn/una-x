"""Phase-B bit-identity battery: ``_phase_b_fast`` vs the ``_phase_b`` oracle.

The G10 bridge's Phase-B structural scan was re-expressed under njit
(``_phase_b_fast``, parallel=False) to remove the per-call pure-Python
scan cost the gpuprobe2 residual attribution measured at 95.8% of the
full-service wall.  Only the returned CODE must be identical on every
input (early exit is not preserved); this battery asserts exactly that,
case by case, against the retained Python oracle, and pins the
precedence SPEC with hand-built adversarial cases:

per origin ascending: (a) hi<lo or lo<0 -> 1; (b) hi>nnz -> 1; then per
j ascending in [lo,hi): (c) j>lo and ids_all[j] <= ids_all[j-1] -> 2,
(d) ids_all[j] < 0 or >= node_count -> 1.  (c) precedes (d) at the same
j (source check order), so at one j code 2 wins; across j the earlier
j's class wins; origin order precedes both.

Families: >=20k randomized adversarial structures (arbitrary indptr
cuts -> lo<0 / hi<lo / hi>nnz / empty slices all occur; ids drawn from
an out-of-range-and-duplicate-rich alphabet), a valid-CSR family
(expected 0 explicitly), ~20 hand cases (all four violation classes and
every precedence tie), and seeded rep4000-scale structures with deep
injected violations.  A FAILURE here is the gate: the device
re-measurement must not run against an unvalidated scan.

Runs anywhere (no GPU; the bridge module imports device-gated).
    PYTHONPATH=src python probes/gpu_phase_b_battery.py --out <out.json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _rep4000(seed: int = 20261012):
    # identical builder to probes/gpu_phase_decomposition.py (same seed)
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
    return indptr, ids, dist


def _check(oracle, fast, indptr, ids, node_count):
    """Run both scans; returns (oracle_code, fast_code, equal)."""
    nnz = int(ids.shape[0])
    a = int(oracle(indptr, ids, node_count, nnz))
    b = int(fast(indptr, ids, node_count, nnz))
    return a, b, a == b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--random-cases", type=int, default=24000)
    ap.add_argument("--valid-cases", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261012)
    args = ap.parse_args()

    result = {"schema": "una10x-gpu-phase-b-battery-1",
              "purpose": "bit-identity battery for the bridge Phase-B "
                         "fast scan; gate for the device re-measurement",
              "python": sys.executable,
              "numba": __import__("numba").__version__,
              "numpy": np.__version__,
              "seed": args.seed}
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
    from urban_network_analysis.Engines import _gpu_fold as g

    oracle = g._phase_b
    fast = g._phase_b_fast
    result["oracle"] = "Engines/_gpu_fold.py _phase_b (Python, retained)"
    result["candidate"] = ("Engines/_gpu_fold.py _phase_b_fast "
                           "(njit cache=False nogil parallel=False)"
                           if fast is not oracle else
                           "FALLBACK: _phase_b_fast is the Python oracle "
                           "(numba njit import failed)")
    print(json.dumps({"stage": "setup", "candidate":
                      "njit" if fast is not oracle else "fallback"}),
          flush=True)

    rng = np.random.default_rng(args.seed)
    n_fail = 0
    failures = []
    code_counts = {0: 0, 1: 0, 2: 0}

    def record(tag, indptr, ids, node_count, expected=None):
        nonlocal n_fail
        a, b, eq = _check(oracle, fast, indptr, ids, node_count)
        code_counts[a] = code_counts.get(a, 0) + 1
        ok = eq and (expected is None or a == expected)
        if not ok and n_fail < 20:
            failures.append({
                "family": tag, "oracle": a, "fast": b,
                "expected": expected,
                "indptr": [int(x) for x in indptr],
                "ids": [int(x) for x in ids][:64],
                "node_count": int(node_count)})
        if not ok:
            n_fail += 1
        return ok

    # --- family 1: randomized adversarial structures ----------------------
    n_rand = 0
    for i in range(args.random_cases):
        n_o = int(rng.integers(1, 60))
        nnz = int(rng.integers(0, 400))
        node_count = int(rng.integers(0, 50))
        # duplicates + out-of-range values (negative and >= node_count):
        # both violation classes fire at high density
        ids = rng.integers(-3, node_count + 3, nnz).astype(np.int64)
        # arbitrary cuts: hi<lo, lo<0, hi>nnz, empty slices all occur
        cuts = rng.integers(-2, nnz + 2, n_o + 1).astype(np.int64)
        if not record(f"random_{i}", cuts, ids, node_count):
            pass
        n_rand += 1
    print(json.dumps({"stage": "random_adversarial", "cases": n_rand,
                      "fails_so_far": n_fail}), flush=True)

    # --- family 2: valid CSRs (expected 0 explicitly) ---------------------
    n_valid = 0
    for i in range(args.valid_cases):
        n_o = int(rng.integers(1, 40))
        node_count = int(rng.integers(1, 200))
        # cap at node_count FIRST so indptr and the packed data agree
        lens = np.minimum(rng.integers(0, 12, n_o),
                          node_count).astype(np.int64)
        indptr = np.zeros(n_o + 1, dtype=np.int64)
        np.cumsum(lens, out=indptr[1:])
        nnz = int(indptr[-1])
        ids = np.empty(nnz, dtype=np.int64)
        pos = 0
        for o in range(n_o):
            k = int(lens[o])
            vals = np.sort(rng.choice(node_count, k, replace=False)) \
                if k else np.empty(0, dtype=np.int64)
            ids[pos:pos + k] = vals
            pos += k
        record(f"valid_{i}", indptr, ids, node_count, expected=0)
        n_valid += 1
    print(json.dumps({"stage": "valid_csr", "cases": n_valid,
                      "fails_so_far": n_fail}), flush=True)

    # --- family 3: hand adversarial cases (spec pinned independently) -----
    hand = [
        ("empty_structure", [0], [], 10, 0),
        ("single_origin_valid", [0, 3], [1, 2, 5], 10, 0),
        ("hi_lt_lo", [0, 5, 3], [1, 2, 3], 10, 1),
        ("lo_negative", [0, -1, 3], [1, 2, 3], 10, 1),
        ("hi_gt_nnz", [0, 3, 9], [1, 2, 3], 10, 1),
        ("hi_gt_nnz_before_monotonic", [0, 2, 6], [0, 5, 1], 10, 1),
        ("same_j_two_beats_one", [0, 3], [5, 3, -1], 10, 2),
        # pure earlier range violation: ids[1] > ids[0] AND out of range
        # (ids[1]=-1 would ALSO violate (c) at the same j and return 2)
        ("earlier_j_range_beats_later_monotonic", [0, 4],
         [0, 12, 7, 8], 10, 1),
        ("earlier_j_monotonic_beats_later_range", [0, 4],
         [1, 1, 99, -2], 10, 2),
        ("origin_order_range_then_monotonic", [0, 2, 4],
         [-1, 5, 1, 0], 10, 1),
        ("origin_order_monotonic_then_range", [0, 2, 4],
         [3, 1, -1, 4], 10, 2),
        ("empty_origins", [0, 0, 0, 2], [1, 2], 10, 0),
        ("node_count_zero", [0, 1], [0], 0, 1),
        ("id_eq_node_count_minus_1_valid", [0, 2], [3, 4], 5, 0),
        ("id_eq_node_count_invalid", [0, 1], [5], 5, 1),
        ("duplicate_pair", [0, 2], [4, 4], 10, 2),
        ("full_valid_multi_origin", [0, 2, 5, 5, 8],
         [0, 3, 4, 7, 9, 9, 10, 11], 12, 0),
        ("hi_eq_nnz_boundary", [0, 2], [1, 2], 10, 0),
        ("hi_gt_nnz_by_one", [0, 3], [1, 2], 10, 1),
        ("negative_lo_with_hi_violation", [0, -2, 1], [1, 2], 10, 1),
    ]
    hand_n = 0
    for tag, ip, ids, nc, expected in hand:
        record(tag, np.array(ip, dtype=np.int64),
               np.array(ids, dtype=np.int64), nc, expected=expected)
        hand_n += 1
    print(json.dumps({"stage": "hand_cases", "cases": hand_n,
                      "fails_so_far": n_fail}), flush=True)

    # --- family 4: rep4000 scale + deep injections ------------------------
    indptr, ids, _dist = _rep4000()
    scale_n = 0
    record("scale_rep4000_valid", indptr, ids, 20000, expected=0)
    scale_n += 1
    nnz = ids.shape[0]
    # origin start for a PURE range violation: a negative mid-origin
    # would also satisfy ids[k] <= ids[k-1], so (c) fires first (2)
    o150 = int(np.searchsorted(indptr, 150000, side="right") - 1)
    k_start = int(indptr[o150])
    for tag, k, kind, expected in [
            ("scale_dup_at_150000", 150000, "dup", 2),
            ("scale_negative_at_150000_monotonic_wins",
             150000, "neg", 2),  # (c) precedes (d) at the same j
            ("scale_negative_at_origin_start_pure_range",
             k_start, "neg", 1),
            ("scale_oob_at_150000", 150000, "oob", 1),
            ("scale_range100k_monotonic200k", None, "combo_rm", 1),
            ("scale_monotonic100k_range200k", None, "combo_mr", 2)]:
        ids2 = ids.copy()
        if kind == "dup":
            ids2[k] = ids2[k - 1]
        elif kind == "neg":
            ids2[k] = -1
        elif kind == "oob":  # pure range: > ids[k-1] AND >= node_count
            ids2[k] = 20000
        elif kind == "combo_rm":  # earlier PURE range violation wins
            ids2[100000] = 20000
            ids2[200000] = ids2[199999]
        elif kind == "combo_mr":  # earlier monotonic violation wins
            ids2[100000] = ids2[99999]
            ids2[200000] = -7
        record(tag, indptr, ids2, 20000, expected=expected)
        scale_n += 1
    print(json.dumps({"stage": "scale_cases", "cases": scale_n,
                      "fails_so_far": n_fail}), flush=True)

    # --- CPU wall of the scan on rep4000 (informational, this host) -------
    t = []
    for _ in range(5):
        t0 = time.perf_counter()
        oracle(indptr, ids, 20000, int(ids.shape[0]))
        t.append(time.perf_counter() - t0)
    t.sort()
    result["phase_b_oracle_rep4000_s"] = {"median": t[2], "runs": 5}
    fast(indptr, ids, 20000, int(ids.shape[0]))  # compile outside timing
    t = []
    for _ in range(5):
        t0 = time.perf_counter()
        fast(indptr, ids, 20000, int(ids.shape[0]))
        t.append(time.perf_counter() - t0)
    t.sort()
    result["phase_b_fast_rep4000_s"] = {"median": t[2], "runs": 5}

    total = n_rand + n_valid + hand_n + scale_n
    result["cases"] = {"random_adversarial": n_rand, "valid_csr": n_valid,
                       "hand_adversarial": hand_n, "scale": scale_n,
                       "total": total}
    result["oracle_code_distribution"] = {str(k): v for k, v in
                                          code_counts.items()}
    result["n_failures"] = n_fail
    result["failures_first_20"] = failures
    result["status"] = "pass" if n_fail == 0 else "fail"
    result["law"] = ("bit-identity battery only; early exit intentionally "
                     "not preserved (code equality is the contract); "
                     "CPU-only, no device facts, no claim")
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(args.out), "status": result["status"],
                      "cases": total, "n_failures": n_fail,
                      "oracle_rep4000_median_s":
                          result["phase_b_oracle_rep4000_s"]["median"],
                      "fast_rep4000_median_s":
                          result["phase_b_fast_rep4000_s"]["median"]}),
          flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
