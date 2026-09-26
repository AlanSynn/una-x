"""Shared invocation helpers for the A1 tests (single B0 singleton via
the oracle package; single candidate alias via cand_load)."""
from __future__ import annotations

import numpy as np

from comparator import assert_array_bytes_equal
from support import b0 as _b0
import cand_load


def b0ns():
    return _b0()


def cns():
    return cand_load.cand()


def run_kernel(ns, case, family="acc"):
    """One per-origin kernel call; returns (labels, pred)."""
    if family == "acc":
        kernel = ns.compact_vector_node_view_scope
    else:
        kernel = ns.compact_vector_node_view_scope_elevation
    return kernel(
        case["o_terminal_idxs"][0],
        case["o_terminal_weights"][0],
        case["adjacency_pointer"],
        case["adjacency_vector"],
        case["adjacency_vector_weights"],
        case["adjacynct_vector_network_node"],
        case["cutoff"],
        case["d_count"],
    )


def run_scratch_kernel(ns, case):
    """Candidate private kernel with per-origin scratch (the exact
    buffers the driver branch allocates)."""
    max_degree = int(np.diff(case["adjacency_pointer"]).max()) \
        if case["adjacency_pointer"].shape[0] > 1 else 0
    eligible_offset = np.empty(max_degree, dtype=np.int64)
    eligible_weight = np.empty(max_degree, dtype=np.float64)
    return ns._a1_scope_search(
        case["o_terminal_idxs"][0],
        case["o_terminal_weights"][0],
        case["adjacency_pointer"],
        case["adjacency_vector"],
        case["adjacency_vector_weights"],
        case["adjacynct_vector_network_node"],
        case["cutoff"],
        case["d_count"],
        eligible_offset,
        eligible_weight,
    )


def run_integrated(ns, arrays, family="acce"):
    """Full integrated driver; family 'acce' = elevation (the observed
    O2 family), 'acc' = base Accessibility family."""
    kernel = (ns.integrated_scope_access_elevation if family == "acce"
              else ns.integrated_scope_access)
    return kernel(
        arrays["o_terminal_idxs"],
        arrays["o_terminal_weights"],
        arrays["adjacency_pointer"],
        arrays["adjacency_vector"],
        arrays["adjacency_vector_weights"],
        arrays["adjacynct_vector_network_node"],
        arrays["d_terminal_idxs"],
        arrays["d_terminal_weights"],
        arrays["d_weights"],
        arrays["gravity_beta"],
        arrays["metric_plateau"],
        arrays["metric_midpoint"],
        float(np.log(99.0) / arrays["metric_midpoint"]),
        arrays["knn_decay"],
        arrays["knn_weights"],
        arrays["cutoff"],
    )


def run_od(ns, arrays, family="acce"):
    kernel = (ns.od_compact_vector_node_view_scope_elevation
              if family == "acce"
              else ns.od_compact_vector_node_view_scope)
    return kernel(
        arrays["o_terminal_idxs"],
        arrays["o_terminal_weights"],
        arrays["adjacency_pointer"],
        arrays["adjacency_vector"],
        arrays["adjacency_vector_weights"],
        arrays["adjacynct_vector_network_node"],
        arrays["cutoff"],
        arrays["d_count"],
        arrays["d_terminal_idxs"],
        arrays["d_terminal_weights"],
    )


INTEGRATED_OUTPUT_NAMES = (
    "reach", "gravity_exponential", "gravity_logistic", "knn_access")


def assert_integrated_bytes(arrays, tag, families=("acce", "acc")):
    """Candidate vs B0 integrated driver, byte equality per output, for
    each requested engine family (PR-N5: both mirrors)."""
    for family in families:
        outs_b0 = run_integrated(b0ns(), arrays, family=family)
        outs_c = run_integrated(cns(), arrays, family=family)
        for i, name in enumerate(INTEGRATED_OUTPUT_NAMES):
            assert_array_bytes_equal(outs_c[i], outs_b0[i],
                                     f"{tag}/{family}/{name}")


def assert_od_bytes(arrays, tag, families=("acce", "acc")):
    for family in families:
        out_b0 = np.asarray(run_od(b0ns(), arrays, family=family))
        out_c = np.asarray(run_od(cns(), arrays, family=family))
        assert_array_bytes_equal(out_c, out_b0, f"{tag}/{family}/od")


def engine_case_kernel_arrays(case):
    """kernel signature arrays + terminal arrays for one oracle
    EngineCase, plus the metric fields the drivers need."""
    arrays = dict(case.kernel_arrays())
    arrays.update(case.terminal_arrays())
    arrays["cutoff"] = case.cutoff
    arrays["gravity_beta"] = case.gravity_beta
    arrays["metric_plateau"] = case.metric_plateau
    arrays["metric_midpoint"] = case.metric_midpoint
    arrays["knn_decay"] = case.knn_decay
    arrays["knn_weights"] = case.knn_weights
    return arrays
