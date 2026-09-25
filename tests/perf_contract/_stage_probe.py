"""Bounded stage probe: which engine stage hangs on nonfinite edge weights?

Prints a STAGE marker (flushed) before each stage, then runs it. A watcher
kills the process at the hang; the last marker names the stuck stage.
Usage: python _stage_probe.py --case nonfinite_weights --origin 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

import numpy as np  # noqa: E402

from stub_topology import StubSettings, StubTopology  # noqa: E402
from _arm_runner import build_cases  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="nonfinite_weights")
    ap.add_argument("--origin", type=int, default=0)
    ap.add_argument("--src-tree", default="baseline",
                    choices=["baseline", "candidate"])
    args = ap.parse_args()

    cases = {name: spec for name, spec in build_cases()}
    spec = cases[args.case]
    if "reuse_from" in spec:
        src = cases[spec["reuse_from"]]
        weights, o, d = src["weights"], src["o"], src["d"]
    else:
        weights, o, d = spec["weights"], spec["o"], spec["d"]

    topo = StubTopology(spec["start"], spec["end"], weights, spec["node_count"],
                        o[0], o[1], o[2], o[3], d[0], d[1], d[2], d[3], d[4],
                        z=spec["z"], start_dtype=spec.get("start_dtype"))
    st = StubSettings(**spec["settings"])

    print("STAGE build starting", flush=True)
    from urban_network_analysis.Engines.Accessibility import Accessibility
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        AccessibilityWElevation,
    )
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        compact_vector_node_view_scope,
    )
    eng_e = AccessibilityWElevation(topo, st)
    ge = eng_e.graph_engine
    print("STAGE build done (CSR built OK)", flush=True)

    oi = min(args.origin, len(ge.o_terminal_idxs) - 1)
    print(f"STAGE scope origin={oi} starting", flush=True)
    sw, _ = compact_vector_node_view_scope(
        ge.o_terminal_idxs[oi], ge.o_terminal_weights[oi],
        ge.adjacency_pointer, ge.adjacency_vector,
        ge.adjacency_vector_weights, ge.adjacynct_vector_network_node,
        st.search_radius, ge.d_count)
    print(f"STAGE scope done nonfinite={int(np.sum(~np.isfinite(sw)))}",
          flush=True)

    print("STAGE centrality starting", flush=True)
    eng_e.Centrality(st)
    print("STAGE centrality done", flush=True)

    print("STAGE od starting", flush=True)
    eng_e.OD_Matrix(search_radius=st.search_radius)
    print("STAGE od done", flush=True)


if __name__ == "__main__":
    main()
