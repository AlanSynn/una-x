"""Shared invocation/compare helpers for the I00 composition tests.

Single B0 singleton via the oracle package's support.b0(); single
composed-candidate alias via cand_load_composition.

COMPARATOR FRAME (pin, h04 I00 plan item R2 disposition pending): this
harness follows the ACCEPTED per-track differential frame — B0 loaded
in-process ONCE under the real module name `urban_network_analysis`
from wt-b0/src @ 361928e (support.b0 -> b0_import.load_arm, the pinned
baseline root), and the composed candidate loaded in-process ONCE under
the fixed alias `una_composition_cand` from this worktree's src. One
import root per module name; the candidate root never enters sys.path.
The oracle package's subprocess-per-root note governs its own
arm-vs-arm loads under one module name (compare_arms.py), not this
alias-differential frame (A3/F1/F3 precedents). If h04 rules the
subprocess baseline frame required for composition, this module is the
single place to restructure.

DECLARED-WRITE FRAME (h04 I00 plan item R1): the artifacts tail is
collected in-memory; flush_artifacts() writes ONLY to the path named by
UNA_COMPOSITION_ARTIFACTS and is invoked only from the env-gated
sessionfinish hook. With the variable unset the suite's complete write
set is the numba cache root campaign_data/nbc_composition/** declared
in conftest (bytecode suppressed in conftest). Nothing else — no
campaigns/** evidence path, no venvs, no other campaign_data path.

CELL RECEIPTS (h04 I00 plan items R3/R4): every matrix cell records a
G5-style receipt {variant, seed, family, a1_admitted, a1_max_degree,
a3_admitted} via record_cell_receipt(); the tailfree-V (A3 route fired)
and full-nd (A1 fallback or original loop) populations are tallied by
population_tally() and asserted > 0 across the matrix.
"""
from __future__ import annotations

import datetime
import json
import os

import numba as nb
import numpy as np

from comparator import assert_array_bytes_equal
from support import b0 as _b0
import cand_load_composition

ARTIFACTS = {
    "task": "I00",
    "suite": "tests/large_e2e/composition",
    "opened_utc": datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "records": [],
    "cell_receipts": [],
}


def record(label, **fields):
    ARTIFACTS["records"].append({"label": label, **fields})


def record_cell_receipt(variant, seed, family, a1_ok, a1_max, a3_ok):
    """G5-style engagement receipt for one matrix cell (R3)."""
    ARTIFACTS["cell_receipts"].append({
        "variant": variant, "seed": seed, "family": family,
        "a1_admitted": bool(a1_ok), "a1_max_degree": int(a1_max),
        "a3_admitted": None if a3_ok is None else bool(a3_ok),
    })


def population_tally():
    """Kernel-population counts across recorded receipts (R4):
    tailfree_V = cells whose receipts show the A3 route admitted;
    full_nd = cells served by the A1 fallback route or the original
    loop (A3 refused or unreachable)."""
    tailfree_V = full_nd = 0
    for r in ARTIFACTS["cell_receipts"]:
        if r["a1_admitted"] and r["a3_admitted"]:
            tailfree_V += 1
        else:
            full_nd += 1
    return {"tailfree_V": tailfree_V, "full_nd": full_nd}


def effective_thread_pin():
    """The numba thread pin actually in force (R5: recorded per run)."""
    return {
        "env_NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
        "numba_config": nb.config.NUMBA_NUM_THREADS,
    }


def flush_artifacts():
    """Env-gated artifacts flush (see module docstring): returns None
    without writing anything when UNA_COMPOSITION_ARTIFACTS is unset."""
    path = os.environ.get("UNA_COMPOSITION_ARTIFACTS")
    if path is None:
        return None
    ARTIFACTS["effective_thread_pin"] = effective_thread_pin()
    b0_ns, cand_ns = _b0(), cns()
    ARTIFACTS["comparator_frame"] = {
        "b0": {"root": b0_ns.root,
               "module_files": dict(b0_ns.module_files)},
        "cand": {"root": cand_ns.root,
                 "alias": cand_ns.alias,
                 "module_files": dict(cand_ns.module_files)},
    }
    ARTIFACTS["flushed_utc"] = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(ARTIFACTS, fh, indent=1)
    return path


def b0ns():
    return _b0()


def cns():
    return cand_load_composition.cand()


# ======================================================================
# Accessibility driver invocation (both engine families)
# ======================================================================

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


def run_integrated(ns, arrays, family="acce"):
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


INTEGRATED_OUTPUT_NAMES = (
    "reach", "gravity_exponential", "gravity_logistic", "knn_access")


def assert_od_bytes(arrays, tag, families=("acce", "acc")):
    for family in families:
        out_b0 = np.asarray(run_od(b0ns(), arrays, family=family))
        out_c = np.asarray(run_od(cns(), arrays, family=family))
        assert_array_bytes_equal(out_c, out_b0, f"{tag}/{family}/od")


def assert_integrated_bytes(arrays, tag, families=("acce", "acc")):
    for family in families:
        outs_b0 = run_integrated(b0ns(), arrays, family=family)
        outs_c = run_integrated(cns(), arrays, family=family)
        for i, name in enumerate(INTEGRATED_OUTPUT_NAMES):
            assert_array_bytes_equal(outs_c[i], outs_b0[i],
                                     f"{tag}/{family}/{name}")


# ======================================================================
# Guard-cell probe (njit wrapper: @overload guards resolve ONLY under
# nopython — pure-Python calls would run the pass-stub)
# ======================================================================

_PROBES = {}


def guard_probe(ns):
    """Compile (once per namespace) an njit probe that resolves BOTH
    composed guards against the admitted typing and returns
    (a1_admitted, a1_max_degree, a3_admitted)."""
    key = ns.alias
    if key in _PROBES:
        return _PROBES[key]

    a1_admits = ns._a1_scope_admits
    a3_admits = ns._a3_tail_admits

    @nb.njit(cache=False)
    def probe(adjacency_pointer, adjacency_vector,
              adjacency_vector_weights, adjacynct_vector_network_node,
              o_terminal_idxs, o_terminal_weights, cutoff,
              d_terminal_idxs, d_count, node_count):
        a1_ok, a1_max = a1_admits(
            adjacency_pointer, adjacency_vector,
            adjacency_vector_weights, adjacynct_vector_network_node,
            o_terminal_idxs, o_terminal_weights, cutoff)
        a3_ok = a3_admits(d_terminal_idxs, d_count, node_count)
        return a1_ok, a1_max, a3_ok

    _PROBES[key] = probe
    return probe


def guard_cell(arrays):
    """Probe the composed dispatch cell for one fixture: returns
    (a1_admitted, a1_max_degree, a3_admitted). node_count mirrors the
    driver's computation (pointer length minus one)."""
    ns = cns()
    probe = guard_probe(ns)
    a1_ok, a1_max, a3_ok = probe(
        arrays["adjacency_pointer"],
        arrays["adjacency_vector"],
        arrays["adjacency_vector_weights"],
        arrays["adjacynct_vector_network_node"],
        arrays["o_terminal_idxs"],
        arrays["o_terminal_weights"],
        arrays["cutoff"],
        arrays["d_terminal_idxs"],
        arrays["d_count"],
        arrays["adjacency_pointer"].shape[0] - 1,
    )
    return bool(a1_ok), int(a1_max), bool(a3_ok)


# ======================================================================
# Flow engine invocation
# ======================================================================

def build_flow_arm(ns, spec, num_threads, extra_overrides=None):
    """Fresh engine + Settings on `spec` for ONE namespace."""
    import fixtures as oracle_fixtures
    import stub_topology
    overrides = oracle_fixtures.flow_settings_overrides(node_flow=True)
    if extra_overrides:
        overrides.update(extra_overrides)
    topo = stub_topology.StubFlowTopology(spec)
    settings = stub_topology.make_settings(ns.Settings,
                                           accessibility=False,
                                           **overrides)
    eng = ns.AggregateFlow(topo)
    eng.num_threads = int(num_threads)
    return eng, settings


def run_flow_arm(ns, spec, num_threads, extra_overrides=None):
    eng, settings = build_flow_arm(ns, spec, num_threads, extra_overrides)
    eng.Centrality(settings)
    return eng


def flow_output_arrays(eng):
    return {
        "edge_flow_AB": eng.edge_flow_AB,
        "edge_flow_BA": eng.edge_flow_BA,
        "edge_flow": eng.edge_flow,
        "node_flow": eng.node_flow,
    }


def assert_flow_bytes(a, b, label):
    for k in a:
        va, vb = a[k], b[k]
        if va is None or vb is None:
            assert (va is None) == (vb is None), f"{label}/{k}: None mismatch"
            continue
        assert va.tobytes() == vb.tobytes(), \
            f"{label}/{k}: engine outputs diverge"


def assert_flow_arms_equal(spec, num_threads, tag, extra_overrides=None):
    """Composed candidate vs B0 engine, byte equality per output."""
    eng_b = run_flow_arm(b0ns(), spec, num_threads, extra_overrides)
    eng_c = run_flow_arm(cns(), spec, num_threads, extra_overrides)
    assert_flow_bytes(flow_output_arrays(eng_c),
                      flow_output_arrays(eng_b), tag)
    return eng_b, eng_c


# ======================================================================
# F3 gradient invocation
# ======================================================================

def run_gradient(ns, stub):
    return ns.AggregateFlow._precompute_dest_gradients(stub, {})


def assert_gradient_bytes(out_c, out_b, tag):
    for i, name in enumerate(("indptr", "nodes", "dist", "pred")):
        assert_array_bytes_equal(np.asarray(out_c[i]), np.asarray(out_b[i]),
                                 f"{tag}/{name}")
