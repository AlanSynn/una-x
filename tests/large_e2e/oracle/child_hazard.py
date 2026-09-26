"""Bounded child-process hazard probes (never run in the main test
process).

Modes characterize the documented hazard/refusal domains of the
compiled B0 scope kernel; each child is run with a wall-clock timeout
by the parent test and its outcome recorded as evidence:

  negative_cycle  — negative edge weight: expected NONTERMINATION
                    (strictly decreasing labels around a cycle).
  nan_edge        — NaN edge weight: record actual behavior.
  nan_origin      — NaN origin terminal weight: record actual behavior.
  inf_cutoff      — cutoff = +inf: record actual behavior (potential
                    unbounded label growth around cycles).

The child prints one JSON line: {mode, outcome, scope_bits?, exception?}.
outcome is one of: completed | timeout_kill | error.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading

import numpy as np


def run_probe(mode):
    from urban_network_analysis.Engines.Accessibility import (
        compact_vector_node_view_scope,
    )

    if mode == "negative_cycle":
        # 2-cycle 0 -> 1 (w 1.0), 1 -> 0 (w -2.0), padded to degree 2 at
        # every node via node 2 so every improved label is re-pushed.
        # Labels strictly decrease around the cycle: the scope kernel
        # has no negative-cycle guard and must NOT terminate.
        ptr = np.array([0, 2, 4, 6], dtype=np.int64)
        nbr = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        w = np.array([1.0, 1.0, -2.0, 1.0, 1.0, 1.0], dtype=np.float64)
        flags = np.ones(6, dtype=np.bool_)
        o_idx = np.array([0, 0], dtype=np.int64)
        o_w = np.array([0.0, 0.0], dtype=np.float64)
        cutoff, d_count = 1e6, 1
    elif mode == "nan_edge":
        ptr = np.array([0, 1, 2], dtype=np.int64)
        nbr = np.array([1, 0], dtype=np.int64)
        w = np.array([np.nan, 1.0], dtype=np.float64)
        flags = np.ones(2, dtype=np.bool_)
        o_idx = np.array([0, 0], dtype=np.int64)
        o_w = np.array([0.0, 0.0], dtype=np.float64)
        cutoff, d_count = 10.0, 1
    elif mode == "nan_origin":
        ptr = np.array([0, 1, 2], dtype=np.int64)
        nbr = np.array([1, 0], dtype=np.int64)
        w = np.array([1.0, 1.0], dtype=np.float64)
        flags = np.ones(2, dtype=np.bool_)
        o_idx = np.array([0, 0], dtype=np.int64)
        o_w = np.array([np.nan, 0.0], dtype=np.float64)
        cutoff, d_count = 10.0, 1
    elif mode == "inf_cutoff":
        ptr = np.array([0, 2, 4, 6], dtype=np.int64)
        nbr = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        w = np.ones(6, dtype=np.float64)
        flags = np.ones(6, dtype=np.bool_)
        o_idx = np.array([0, 0], dtype=np.int64)
        o_w = np.array([0.0, 0.0], dtype=np.float64)
        cutoff, d_count = float(np.inf), 1
    else:
        raise ValueError(f"unknown mode {mode}")

    scope, pred = compact_vector_node_view_scope(
        o_idx, o_w, ptr, nbr, w, flags, cutoff, d_count)
    return {
        "mode": mode,
        "outcome": "completed",
        "scope_bits": scope.tobytes().hex(),
        "scope_dtype": str(scope.dtype),
        "scope_shape": list(scope.shape),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--timeout-s", type=float, default=15.0)
    args = ap.parse_args()

    result = {"mode": args.mode, "outcome": None}
    holder = {}

    def target():
        try:
            holder["payload"] = run_probe(args.mode)
        except Exception as exc:  # noqa: BLE001 - probe records type only
            holder["payload"] = {
                "mode": args.mode, "outcome": "error",
                "exception_type": type(exc).__name__,
            }

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=args.timeout_s)
    if thread.is_alive():
        # Non-daemon child of a nonterminating numba loop: the process
        # must be killed by the parent's subprocess timeout; emit
        # nothing and exit hard so no partial result is mistaken for a
        # numerical outcome.
        result["outcome"] = "timeout_kill"
        print(json.dumps(result))
        sys.stdout.flush()
        import os
        os._exit(9)
    print(json.dumps(holder["payload"]))


if __name__ == "__main__":
    main()
