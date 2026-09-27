"""F3 cleanup / lifetime battery (proof sections 5 / 3(d) / 10).

1. del-before-next-call: weakref-finalizer instrumentation on the
   instrumented _scipy_dijkstra wrapper — at the entry of every call
   after the first, BOTH arrays of every prior call must already be
   finalized (CPython refcount granularity; instrumentation OUTSIDE
   any measured window, per dossier line 32).
2. Non-aliasing: every returned array is a fresh allocation (base is
   None) and byte-equal to the reference — appended parts demonstrably
   do not alias deleted dense memory (dossier line 28).
3. Margin gap gate (3(d)(ii)): a fresh CHILD process measures one
   budgeted call's realized-transient-minus-payload via ru_maxrss on a
   representative sparse-row fixture; pass requires gap <= margin/2.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import fixtures_f3
from harness_f3 import (assert_gradient_bytes, b0ns, cns, record_note,
                        record_schedule)
from test_f3_chunk_equivalence import parse_f3_tail

CANDIDATE_ROOT = ("/Users/alansynn/orca/workspaces/una-x/wt-large-e2e/src")
CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "f3_mem_child.py")


def test_del_before_next_call(monkeypatch):
    """Forced multi-slice schedule (c=2 on the 7-destination fixture,
    4 calls): at every call after the first, all prior calls' arrays
    are finalized — the dense triple does not survive into the next
    call (the load-bearing change, proof section 1 vs 5)."""
    c = cns()
    case = fixtures_f3.uneven_tail_case()
    stub = fixtures_f3.case_stub(case)
    monkeypatch.setattr(c.lfws, "DEFAULT_CAP_BYTES",
                        int(fixtures_f3.cap_for_first_slice_c(stub, c.lfws, 2)))
    rec = fixtures_f3.CallRecorder(track_finalizers=True).install(
        c.flow_module)
    try:
        c.AggregateFlow._precompute_dest_gradients(stub, {})
    finally:
        rec.uninstall(c.flow_module)
    n_calls = len(rec.bounds)
    assert n_calls >= 2, f"fixture produced {n_calls} calls; need >= 2"
    for k, ok in enumerate(rec.entry_checks, start=1):
        assert ok, (
            f"call {k} started while slice {k - 1}'s dense arrays were "
            f"still live (counters {[c[0] for c in rec.counters]})")
    # and by the END, everything is finalized
    assert all(cnt[0] == 2 for cnt in rec.counters)
    record_note("del_before_next_call", {
        "calls": n_calls,
        "entry_checks_all_true": True,
        "instrument": "weakref.finalize on each call's dist+preds; "
                      "checked at next-call entry (refcount granularity)",
    })


def test_outputs_fresh_copies():
    """Every returned array is an independent allocation (base is
    None) and byte-equal to the B0 arm — appended parts demonstrably
    do not alias deleted dense memory."""
    c = cns()
    b0 = b0ns()
    case = fixtures_f3.base_case()
    stub_b0 = fixtures_f3.case_stub(case)
    stub_c = fixtures_f3.case_stub(case)
    out_b = b0.AggregateFlow._precompute_dest_gradients(stub_b0, {})
    out_c = c.AggregateFlow._precompute_dest_gradients(stub_c, {})
    for i, name in enumerate(("indptr", "nodes", "dist", "pred")):
        assert out_c[i].base is None, f"{name} aliases another buffer"
    assert_gradient_bytes(out_c, out_b, "fresh_copies")
    record_note("non_aliasing", {
        "base_is_none": ["indptr", "nodes", "dist", "pred"],
        "plus": "byte-equality to B0; parts are fancy-index + astype "
                "copies (both arms), never views of the dense slices",
    })


def test_margin_gap_gate():
    """3(d)(ii): realized-transient-minus-payload <= margin/2, measured
    by a fresh child process (ru_maxrss high-water, not tracemalloc)."""
    sys_exe = sys.executable
    proc = subprocess.run(
        [sys_exe, CHILD, CANDIDATE_ROOT],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, NUMBA_CACHE_DIR="/Users/alansynn/orca"
                                             "/workspaces/una-x/"
                                             "campaign_data/nbc_f3"),
    )
    assert proc.returncode == 0, (
        f"margin child failed rc={proc.returncode}\n"
        f"stderr:\n{proc.stderr[-4000:]}")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["gate_pass"], (
        f"margin gap {payload['gap_bytes']} B exceeds margin/2 "
        f"({payload['margin_half']} B) — stop-and-re-derive with "
        f"reviewer approval before freezing the proof")
    record_schedule("margin_gap_child", payload)
