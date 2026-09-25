"""CSR micro-benchmark: original O(V*E) builder vs ordered-CSR candidate.

Numpy-only (no UNA import, no JIT). Builds grid-network endpoint arrays at
several scales and times both builders end-to-end (best-of-R). Reports raw
walls so the campaign can state the CSR share and its scale sensitivity.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS = HERE.parents[1] / "tests" / "perf_contract"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grid_arrays(gx, gy, seed=101):
    """Endpoint arrays for the same deterministic grid workload_gen builds."""
    rng = np.random.default_rng(seed)
    n = gx * gy
    idx = lambda i, j: j * gx + i
    edges = []
    for j in range(gy):
        for i in range(gx):
            if i < gx - 1:
                edges.append((idx(i, j), idx(i + 1, j)))
            if j < gy - 1:
                edges.append((idx(i, j), idx(i, j + 1)))
            if i < gx - 1 and j < gy - 1 and rng.random() < 0.12:
                edges.append((idx(i, j), idx(i + 1, j + 1)))
    arr = np.array(edges, dtype=np.int64)
    return arr[:, 0].copy(), arr[:, 1].copy(), n


def best_of(fn, reps):
    walls = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        walls.append(time.perf_counter_ns() - t0)
    return walls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--scales", default="7360,29440,117760")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cand = _load("cand_csr",
                 HERE.parent.parent / "src" / "urban_network_analysis"
                 / "Engines" / "_ordered_csr.py")
    orig = _load("orig_csr", TESTS / "ref_original_csr.py")

    results = []
    for node_count in (int(s) for s in args.scales.split(",")):
        gy = max(2, int(node_count ** 0.5))
        gx = node_count // gy
        node_count = gx * gy
        s, e, n = grid_arrays(gx, gy)
        w = np.random.default_rng(5).random(len(s)) * 50.0 + 1.0
        ab = w + 0.1
        ba = w - 0.05

        o_walls = best_of(lambda: orig.original_builder_core(
            n, s, e, ab, ba), args.reps)
        c_walls = best_of(lambda: cand._try_build_ordered_csr(
            n, s, e, ab, ba), args.reps)

        # bit equality of the two builders on this input
        a = orig.original_builder_core(n, s, e, ab, ba)
        b = cand._try_build_ordered_csr(n, s, e, ab, ba)
        equal = all(np.array_equal(x, y) and x.dtype == y.dtype
                    for x, y in zip(a, b))

        results.append({
            "node_count": int(n), "edges": int(len(s)),
            "original_walls_ns": o_walls,
            "candidate_walls_ns": c_walls,
            "original_best_ms": min(o_walls) / 1e6,
            "candidate_best_ms": min(c_walls) / 1e6,
            "bit_equal": bool(equal),
        })
        print(f"n={n:7d} E={len(s):7d} "
              f"orig={min(o_walls)/1e6:9.3f}ms cand={min(c_walls)/1e6:8.3f}ms "
              f"equal={equal}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"reps": args.reps, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
