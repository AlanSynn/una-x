"""F3 observed-derived battery (proof sections 4 / 6 / 10; spec
tests_plan 'complete RunFlow' at L1-lite scale).

Real engines built from the oracle's adversarial flow spec (sorted
input / parallel arcs derived from the actual flow CSR construction
path via the engine's own _build_csr), two arms:

* full Centrality: B0 vs candidate — all engine output arrays
  byte-compared (complete RunFlow);
* across caps: the candidate under the frozen {64, 128, 256(default)}
  MiB caps — outputs byte-identical (C6 at the complete-run scale);
* function-level gradient call on the real engine CSRs, byte-equal to
  B0 and to the in-process original-formula replica;
* warning counts recorded per arm, none swallowed (spec
  admission_and_warnings).

No throughput claim of any kind is made here (DECISION.md line 11).
"""
from __future__ import annotations

import warnings

import numpy as np

import fixtures
import fixtures_f3
import stub_topology
from comparator import assert_array_bytes_equal
from harness_f3 import (b0ns, cns, record_note, record_schedule,
                        record_warnings)
from test_f3_chunk_equivalence import parse_f3_tail


def _engine_outputs(eng):
    return {
        "edge_flow_AB": eng.edge_flow_AB,
        "edge_flow_BA": eng.edge_flow_BA,
        "edge_flow": eng.edge_flow,
        "node_flow": eng.node_flow,
    }


def _build_and_run(ns, cap_patch=None, num_threads=2):
    topo = stub_topology.StubFlowTopology(fixtures.flow_network_spec())
    overrides = fixtures.flow_settings_overrides(node_flow=True)
    settings = stub_topology.make_settings(ns.Settings, accessibility=False,
                                           **overrides)
    eng = ns.AggregateFlow(topo)
    eng.num_threads = int(num_threads)
    if cap_patch is not None:
        # candidate policy module only (B0 has no cap)
        cns().lfws.DEFAULT_CAP_BYTES = int(cap_patch)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            eng.Centrality(settings)
    finally:
        if cap_patch is not None:
            cns().lfws.DEFAULT_CAP_BYTES = cns().lfws.CAP_256
    return eng, caught


def test_complete_runflow_two_arm_and_across_caps():
    """B0 vs candidate at default cap, and candidate at 64/128 — all
    byte-equal on every engine output array."""
    b0 = b0ns()
    c = cns()
    runs = {}
    warn_counts = {}
    eng_b0, caught_b0 = _build_and_run(b0)
    warn_counts["b0_default"] = len(caught_b0)
    runs["b0_default"] = _engine_outputs(eng_b0)
    for label, cap in (("cand_default", None),
                       ("cand_cap64", c.lfws.CAP_64),
                       ("cand_cap128", c.lfws.CAP_128)):
        eng_c, caught_c = _build_and_run(c, cap_patch=cap)
        warn_counts[label] = len(caught_c)
        runs[label] = _engine_outputs(eng_c)
    for label in ("cand_default", "cand_cap64", "cand_cap128"):
        for name, arr_b in runs["b0_default"].items():
            arr_c = runs[label][name]
            assert arr_c is not None and arr_b is not None, (
                f"{label}/{name}: engine surface missing")
            assert_array_bytes_equal(np.asarray(arr_c),
                                     np.asarray(arr_b),
                                     f"runflow/{label}/{name}")
    record_warnings("complete_runflow", warn_counts)
    record_note("runflow_outputs", {
        name: {"shape": list(np.asarray(runs["b0_default"][name]).shape),
               "dtype": str(np.asarray(runs["b0_default"][name]).dtype)}
        for name in runs["b0_default"]})


def test_engine_gradient_function_two_arm():
    """Function-level on the real engine CSRs (post-Centrality state):
    candidate byte-equal to B0 and to the original-formula replica;
    the [F3] tail parses.  A capture shim replaces the engine logger
    for the gradient call only (the function's sole logger contact is
    .log)."""
    b0 = b0ns()
    c = cns()
    for ns, tag in ((b0, "b0"), (c, "cand")):
        eng, _ = _build_and_run(ns)
        eng._gradient_limit = lambda ns_dict: np.inf  # instance shadow
        eng.logger = fixtures_f3.CaptureLogger()
        if tag == "cand":
            stub_c = eng
            out_c = c.AggregateFlow._precompute_dest_gradients(stub_c, {})
        else:
            stub_b = eng
            out_b = b0.AggregateFlow._precompute_dest_gradients(stub_b, {})
    for i, name in enumerate(("indptr", "nodes", "dist", "pred")):
        assert_array_bytes_equal(np.asarray(out_c[i]),
                                 np.asarray(out_b[i]),
                                 f"engine_gradient/{name}")
    out_r = fixtures_f3.reference_gradient(stub_c, np.inf)
    for i, name in enumerate(("indptr", "nodes", "dist", "pred")):
        assert_array_bytes_equal(np.asarray(out_c[i]),
                                 np.asarray(out_r[i]),
                                 f"engine_gradient_vs_replica/{name}")
    lines_c = stub_c.logger.v2_lines()
    lines_b = stub_b.logger.v2_lines()
    assert len(lines_c) >= 1 and len(lines_b) >= 1
    cand_grad = [ln for ln in lines_c if "[F3]" in ln]
    assert len(cand_grad) == 1, f"expected ONE [F3] line, got {len(cand_grad)}"
    parsed = parse_f3_tail(cand_grad[0])
    record_schedule("engine_gradient", parsed)


def test_warning_counts_none_swallowed():
    """Warning parity across arms on the admitted profile (recorded
    per arm; no warning classified as harmless timing metadata)."""
    b0 = b0ns()
    c = cns()
    _, caught_b = _build_and_run(b0)
    _, caught_c = _build_and_run(c)
    rec_b = sorted({str(w.message) for w in caught_b})
    rec_c = sorted({str(w.message) for w in caught_c})
    record_warnings("parity_check", {
        "b0": {"count": len(caught_b), "messages": rec_b},
        "cand": {"count": len(caught_c), "messages": rec_c},
    })
    assert len(caught_c) == len(caught_b), (
        f"warning count changed: B0 {len(caught_b)} vs cand {len(caught_c)}")
