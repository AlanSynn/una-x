"""Compiler/numerical profile capture for the H03 oracle evidence.

Records the exact frozen profile under which golden outputs are valid:
Python, numpy/scipy/numba/llvmlite versions, threading layer (after a
parallel kernel has warmed the pool), numba thread count, ISA, GIL
mode, kernel targetoptions and the numba cache root.
"""
from __future__ import annotations

import os
import platform
import sys


def capture_profile(b0, kernels_parallel_warmed: bool) -> dict:
    import numba
    import llvmlite
    import numpy
    import scipy

    profile = {
        "python": {
            "version": sys.version,
            "version_short": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "gil_free_enabled": bool(getattr(sys, "_is_gil_enabled", lambda: False)()),
        },
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "numba_version": numba.__version__,
        "llvmlite_version": llvmlite.__version__,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "env": {
            "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR"),
            "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
            "NUMBA_PARALLEL_DIAGNOSTICS": os.environ.get("NUMBA_PARALLEL_DIAGNOSTICS"),
            "L1_REUSE_DIR": os.environ.get("L1_REUSE_DIR"),
        },
        "numba_threading_layer": None,
        "kernel_targetoptions": {
            "reach_gravity_knn_access": dict(b0.reach_gravity_knn_access.targetoptions),
            "compact_vector_node_view_scope": dict(
                b0.compact_vector_node_view_scope.targetoptions
            ),
            "adjust_destination_distances": dict(
                b0.adjust_destination_distances.targetoptions
            ),
            "od_compact_vector_node_view_scope": dict(
                b0.od_compact_vector_node_view_scope.targetoptions
            ),
            "integrated_scope_access": dict(b0.integrated_scope_access.targetoptions),
            "_accumulate_od_flow": dict(b0._accumulate_od_flow.targetoptions),
            "_find_arc": dict(b0._find_arc.targetoptions),
            "_decay": dict(b0._decay.targetoptions),
        },
        "kernel_n_signatures": {
            name: len(getattr(b0, name).signatures)
            for name in (
                "reach_gravity_knn_access",
                "compact_vector_node_view_scope",
                "adjust_destination_distances",
                "od_compact_vector_node_view_scope",
                "integrated_scope_access",
                "_accumulate_od_flow",
                "_find_arc",
                "_decay",
            )
        },
        "verified_live_facts": {
            "_accumulate_od_flow_fastmath": bool(
                b0._accumulate_od_flow.targetoptions.get("fastmath")
            ),
            "_accumulate_od_flow_nogil": "nogil" in b0._accumulate_od_flow.targetoptions
            and bool(b0._accumulate_od_flow.targetoptions.get("nogil")),
            "accessibility_kernels_nogil_fastmath": bool(
                b0.compact_vector_node_view_scope.targetoptions.get("nogil")
                and b0.compact_vector_node_view_scope.targetoptions.get("fastmath")
            ),
            "parallel_variants_use_parallel": bool(
                b0.od_compact_vector_node_view_scope.targetoptions.get("parallel")
                and b0.integrated_scope_access.targetoptions.get("parallel")
            ),
        },
    }
    if kernels_parallel_warmed:
        try:
            profile["numba_threading_layer"] = numba.threading_layer()
        except Exception as exc:  # pragma: no cover - record, never fabricate
            profile["numba_threading_layer"] = f"unavailable: {type(exc).__name__}"
    return profile
