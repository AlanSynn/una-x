"""F3 refusal / fallback battery (proof sections 3 / 7 / 10).

Entry NO_FIT and forced mid-run NO_FIT, each proving (i) output
byte-identity to B0 and (ii) via CALL-SEQUENCE instrumentation that
the fallback execution's slice boundaries equal the original formula's
(floor(1e8/V') widths) — the restart-vs-hybrid discriminator
(ruling extension 4; byte-identity alone cannot distinguish a restart
from a hybrid).  The dtype-guard trigger (sizing-contract deviation =
non-admitted profile, spec admission_and_warnings) is exercised by
simulating a first call that returns non-contract dtypes.

No OOM anywhere: the tiny-cap forcing is selector-level arithmetic on
small fixtures; the main process is never driven toward memory
pressure (spec refusal_test_safety).
"""
from __future__ import annotations

import numpy as np

import fixtures_f3
from harness_f3 import (assert_gradient_bytes, b0ns, cns, record_calls,
                        record_note, record_schedule)
from test_f3_chunk_equivalence import parse_f3_tail


def _formula_bounds(stub):
    n_total = stub._csr_indptr.shape[0] - 1
    n_dest = stub._n_destinations
    chunk = max(1, int(1e8 // max(n_total, 1)))
    return [(s, min(s + chunk, n_dest)) for s in range(0, n_dest, chunk)]


def test_entry_no_fit(monkeypatch):
    """Cap one byte below c=1 fit: the ORIGINAL formula is the ONLY
    result-producing loop; boundaries equal the formula's; markers
    entry/none; outputs byte-identical."""
    c = cns()
    b0 = b0ns()
    case = fixtures_f3.uneven_tail_case()      # 7 destinations
    stub_b0 = fixtures_f3.case_stub(case)
    stub_c = fixtures_f3.case_stub(case)
    monkeypatch.setattr(c.lfws, "DEFAULT_CAP_BYTES",
                        int(fixtures_f3.cap_entry_no_fit(stub_c, c.lfws)))
    rec = fixtures_f3.CallRecorder().install(c.flow_module)
    try:
        out_b = b0.AggregateFlow._precompute_dest_gradients(stub_b0, {})
        out_c = c.AggregateFlow._precompute_dest_gradients(stub_c, {})
    finally:
        rec.uninstall(c.flow_module)
    assert_gradient_bytes(out_c, out_b, "entry_no_fit")
    expected = _formula_bounds(stub_c)
    got = rec.position_bounds(case["n_net"])
    assert got == expected, (
        f"entry-NO_FIT call sequence {got} != formula {expected}")
    lines = stub_c.logger.v2_lines()
    assert len(lines) == 1
    parsed = parse_f3_tail(lines[0])
    assert parsed["fallback"] == "entry" and parsed["floor_fired"] == "none"
    assert parsed["schedule"] == [e - s for s, e in expected]
    record_calls("entry_no_fit", {"bounds": got,
                                  "parsed": parsed})


def test_midrun_no_fit_restart(monkeypatch):
    """Retained parts drive budget_k negative mid-schedule: free +
    restart on the ORIGINAL formula; the RESTART segment's boundaries
    equal the formula's exactly; marker midrun with where-it-fired."""
    c = cns()
    b0 = b0ns()
    # Fully-connected 12-node graph, 6 destinations: every row fully
    # finite (retained grows by ~V'*20 B per row), cap forced to fit
    # exactly two rows — slice 2 must NO_FIT mid-schedule.
    n = 12
    arcs = [(i, j, 1.0 + 0.01 * (i + j))
            for i in range(n) for j in range(n) if i != j]
    for d in range(6):
        arcs += [(t, n + d, 1.0) for t in range(n)]
    case = dict(name="midrun_forcing", arcs=arcs, n_net=n, n_dest=6,
                limit=np.inf)
    stub_b0 = fixtures_f3.case_stub(case)
    stub_c = fixtures_f3.case_stub(case)
    monkeypatch.setattr(c.lfws, "DEFAULT_CAP_BYTES",
                        int(fixtures_f3.cap_for_first_slice_c(
                            stub_c, c.lfws, 2)))
    rec = fixtures_f3.CallRecorder().install(c.flow_module)
    try:
        out_b = b0.AggregateFlow._precompute_dest_gradients(stub_b0, {})
        out_c = c.AggregateFlow._precompute_dest_gradients(stub_c, {})
    finally:
        rec.uninstall(c.flow_module)
    assert_gradient_bytes(out_c, out_b, "midrun_no_fit")
    lines = stub_c.logger.v2_lines()
    assert len(lines) == 1
    parsed = parse_f3_tail(lines[0])
    assert parsed["fallback"] == "midrun", parsed
    assert parsed["floor_slice"] is not None
    assert parsed["floor_chunks_done"] >= 1
    # restart-vs-hybrid discriminator: the LAST len(formula) calls are
    # exactly the formula's boundaries (a hybrid would keep the prefix
    # and only cover the remainder).
    expected = _formula_bounds(stub_c)
    positions = rec.position_bounds(case["n_net"])
    L = len(expected)
    assert positions[-L:] == expected, (
        f"restart segment {positions[-L:]} != formula {expected}")
    record_calls("midrun_no_fit", {"bounds": positions, "parsed": parsed,
                                   "formula_bounds": expected})


def test_dtype_guard_reverts_to_formula(monkeypatch):
    """A returned dtype deviating from the sizing contract (float32
    dist simulated on the FIRST call) is a non-admitted profile: the
    code reverts via the ruled restart; the final output (produced by
    real contract-conformant calls) is byte-identical to B0."""
    c = cns()
    b0 = b0ns()
    case = fixtures_f3.uneven_tail_case()
    stub_b0 = fixtures_f3.case_stub(case)
    stub_c = fixtures_f3.case_stub(case)
    real = c.flow_module._scipy_dijkstra
    calls = {"n": 0}

    def float32_first_call(*a, **k):
        d, p = real(*a, **k)
        calls["n"] += 1
        if calls["n"] == 1:
            return d.astype(np.float32), p.astype(np.int64)
        return d, p

    monkeypatch.setattr(c.flow_module, "_scipy_dijkstra",
                        float32_first_call)
    out_b = b0.AggregateFlow._precompute_dest_gradients(stub_b0, {})
    out_c = c.AggregateFlow._precompute_dest_gradients(stub_c, {})
    assert_gradient_bytes(out_c, out_b, "dtype_guard")
    parsed = parse_f3_tail(stub_c.logger.v2_lines()[0])
    assert parsed["fallback"] == "dtype", parsed
    assert calls["n"] > 1, "restart did not re-execute"
    record_calls("dtype_guard", {"calls": calls["n"], "parsed": parsed})


def test_no_oom_note():
    record_note("refusal_safety", {
        "forced_by": "selector-level cap arithmetic on tiny fixtures",
        "main_process_memory_pressure": "none",
        "spec_refusal_test_safety": "honored",
    })
