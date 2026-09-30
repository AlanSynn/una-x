"""Proof section 7.3: IR side-by-side capture into evidence/A1I/.

Compiles BOTH search kernels at the observed kernel signature
(o_terminal rows int64/float64, CSR int64/float64/bool, float64
cutoff, int64 d_count) and dumps full LLVM IR for each, plus the
floating-point instruction census the A1R reviewer needs (every fadd
site with its fast-math flags; any fmuladd/fma contract intrinsics).
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

A1 = ("/Users/alansynn/orca/workspaces/una-x/wt-large-e2e/"
      "tests/large_e2e/A1")
ORACLE = ("/Users/alansynn/orca/workspaces/una-x/wt-large-e2e/"
          "tests/large_e2e/oracle")
OUT = ("/Users/alansynn/orca/workspaces/una-x/wt-large-e2e/"
       "campaigns/una_large_e2e/evidence/A1I")

sys.path.insert(0, ORACLE)
sys.path.insert(0, A1)

import numba as nb  # noqa: E402

from support import b0 as _b0  # noqa: E402
import cand_load  # noqa: E402

SIG = (nb.types.int64[:],        # o_terminal_idxs row
       nb.types.float64[:],      # o_terminal_weights row
       nb.types.int64[:],        # adjacency_pointer
       nb.types.int64[:],        # adjacency_vector
       nb.types.float64[:],      # adjacency_vector_weights
       nb.types.boolean[:],      # adjacynct_vector_network_node
       nb.types.float64,         # cutoff
       nb.types.int64)           # d_count

CAND_SIG = SIG + (nb.types.int64[:],   # eligible_offset scratch
                  nb.types.float64[:])  # eligible_weight scratch

FP_RE = re.compile(r"(fmuladd|llvm\.fma|fadd|fmul|fsub|fdiv|fcmp)")


def fp_census(ir):
    lines = []
    for line in ir.splitlines():
        if FP_RE.search(line):
            lines.append(line.strip())
    return lines


def main():
    b = _b0()
    c = cand_load.cand()
    # the oracle namespace binds the elevation DRIVERS but not the
    # elevation KERNEL; the B0 modules are imported under their real
    # names, so take the kernel from sys.modules
    b0_kernel = sys.modules[
        "urban_network_analysis.Engines.AccessibilityWElevation"
    ].compact_vector_node_view_scope

    targets = [
        ("ir_b0_compact_vector_node_view_scope.ll",
         "B0 kernel compact_vector_node_view_scope (elevation module)",
         b0_kernel, SIG),
        ("ir_cand_a1_scope_search.ll",
         "candidate kernel _a1_scope_search (helper module)",
         c._a1_scope_search, CAND_SIG),
    ]
    census = []
    for fname, title, disp, sig in targets:
        disp.compile(sig)
        ir = disp.inspect_llvm(sig)
        with open(os.path.join(OUT, fname), "w") as fh:
            fh.write(ir)
        lines = fp_census(ir)
        census.append((title, fname, lines))
        print(f"{fname}: {len(lines)} FP lines, "
              f"{sum('fadd' in l for l in lines)} fadd, "
              f"{sum('fmuladd' in l or 'llvm.fma' in l for l in lines)} "
              f"fma/fmuladd")

    with open(os.path.join(OUT, "ir_fp_census.txt"), "w") as fh:
        for title, fname, lines in census:
            fh.write(f"==== {title} ({fname}) ====\n")
            for line in lines:
                fh.write(line + "\n")
            fh.write("\n")
    print("census written")


if __name__ == "__main__":
    main()
