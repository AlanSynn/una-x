#!/usr/bin/env python3
"""F00 diagnostic probe: bound the dense-output INIT cost inside scipy's
dijkstra call at the EXACT (rows x cols) shapes recorded by the diag leg.

This is NOT a claimed stage wall and NOT part of any timed leg.  It reads
leg_C_diag.json's scipy call records and times, per distinct shape:
  * np.empty alloc only
  * np.full alloc+fill at the init values scipy must produce before the
    traversal can write finite entries (inf for dist, -9999 for pred;
    scipy documents unreached entries as inf, i.e. at least one full
    fill pass over the dense output per call)
  * a full-row np.isfinite sweep at the matched row (cross-check of the
    in-vivo isfinite wall recorded by the leg)

Diagnostic estimate only: bounds the allocation/initialization share of
the scipy call wall; the remainder is attributed to the bounded
traversal itself (which any sparse-output producer still performs).
"""
from __future__ import annotations

import collections
import json
import sys
import time

import numpy as np


def main() -> int:
    leg_path = sys.argv[1]
    with open(leg_path, "r", encoding="utf-8") as f:
        leg = json.load(f)
    # Shapes are stage-stable: the gradient calls share one (rows, cols)
    # (fixed F3 chunk x csr width); the forward calls share (1, cols).
    # The leg record persists the first 10 calls, which cover both.
    calls = leg["scipy_calls_first10"]
    shapes = collections.Counter((c["rows"], c["cols"]) for c in calls)
    out = {
        "leg": leg["leg"],
        "note": ("diagnostic allocation/INIT bound at shapes observed in "
                 "the first persisted scipy calls; totals in the leg "
                 "record give the full-call walls"),
        "shapes": {},
    }
    for (k, n), count in sorted(shapes.items()):
        reps = 5
        t0 = time.perf_counter()
        for _ in range(reps):
            a = np.empty((k, n), dtype=np.float64)
        t_empty = (time.perf_counter() - t0) / reps

        t0 = time.perf_counter()
        for _ in range(reps):
            a = np.full((k, n), np.inf, dtype=np.float64)
        t_full_dist = (time.perf_counter() - t0) / reps

        t0 = time.perf_counter()
        for _ in range(reps):
            a = np.full((k, n), -9999, dtype=np.int32)
        t_full_pred = (time.perf_counter() - t0) / reps

        row = np.full(n, np.inf, dtype=np.float64)
        t0 = time.perf_counter()
        for _ in range(reps):
            m = np.isfinite(row)
        t_isfinite_row = (time.perf_counter() - t0) / reps

        out["shapes"][f"{k}x{n}"] = {
            "observed_calls_of_shape": count,
            "empty_alloc_s": t_empty,
            "full_fill_dist_s": t_full_dist,
            "full_fill_pred_s": t_full_pred,
            "isfinite_one_row_s": t_isfinite_row,
            "dense_mb_per_call": k * n * 12.0 / 1e6,
        }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
