"""I00 composition: F2/F3 buffer lifetimes in the composed AggregateFlow.

F3's byte-budgeted gradient workspace chunks `_precompute_dest_gradients`
slices through `_lfws.select_chunk` (freeing views before each next
chunk; on NO_FIT the partial schedule is freed and the unchanged
element-count fallback formula produces the result). F2's local-overlap
route keeps reusable OD buffers with bounded touched lists in the same
module. Composition risks: a chunk schedule's frees or a NO_FIT unwind
disturbing F2's reusable scratch, and vice versa on interleaved calls.

Pinned here, in the composed tree:

* gradient outputs byte-identical to B0 at the production cap, at a
  forced-small cap (real multi-chunk schedules), and at a forced-tiny
  cap (NO_FIT fallback);
* the ascending-slice precondition forced False serves every OD on the
  unchanged baseline kernel (fallback engagement: fast_calls == 0,
  outputs byte-equal);
* an interleaved gradient → flow-Centrality → gradient sequence is
  byte-stable per stage against fresh B0 arms of the same stages.
"""
from __future__ import annotations

import pytest

from fixtures_composition import (flow_spec_long, grad_arcs_chain,
                                  make_gradient_stub)
from harness_composition import (assert_flow_bytes, assert_gradient_bytes,
                                 b0ns, build_flow_arm, cns,
                                 flow_output_arrays, record, run_flow_arm,
                                 run_gradient)

FORCED_CHUNK_CAP = 400      # bytes: forces real multi-chunk schedules
FORCED_NOFIT_CAP = 1        # bytes: nothing fits -> NO_FIT fallback


def _gradient_pair(cap_value=None):
    """Run one gradient call per arm on twin stubs; optionally force the
    candidate's workspace cap (B0 keeps its original formula)."""
    stub_b = make_gradient_stub(grad_arcs_chain(), n_net=8, n_dest=2,
                                n_extra=1)
    stub_c = make_gradient_stub(grad_arcs_chain(), n_net=8, n_dest=2,
                                n_extra=1)
    c = cns()
    real_cap = c.lfws.DEFAULT_CAP_BYTES
    if cap_value is not None:
        c.lfws.DEFAULT_CAP_BYTES = int(cap_value)
    try:
        out_b = run_gradient(b0ns(), stub_b)
        out_c = run_gradient(cns(), stub_c)
    finally:
        c.lfws.DEFAULT_CAP_BYTES = real_cap
    return stub_b, stub_c, out_b, out_c


@pytest.mark.parametrize("cap_name,cap_value", (
    ("production_cap", None),
    ("forced_chunks", FORCED_CHUNK_CAP),
    ("forced_nofit", FORCED_NOFIT_CAP),
))
def test_gradient_byte_equal_per_schedule(cap_name, cap_value):
    """Gradient indptr/nodes/dist/pred byte-identical to B0 across the
    schedule axis (production cap / forced chunks / NO_FIT fallback)."""
    stub_b, stub_c, out_b, out_c = _gradient_pair(cap_value)
    assert_gradient_bytes(out_c, out_b, f"f2f3/gradient/{cap_name}")
    tail_c = stub_c.logger.v2_lines()
    tail_b = stub_b.logger.v2_lines()
    if cap_name == "forced_nofit":
        # trigger tokens: "none" | "entry" | "midrun" | "dtype"
        # (source pin :1317-1321, :1336-1338)
        assert any("fallback=entry" in line or "fallback=midrun" in line
                   for line in tail_c), \
            f"expected a NO_FIT fallback marker, got: {tail_c!r}"
    record("gradient_schedule", cap=cap_name,
           cand_v2_lines=len(tail_c), b0_v2_lines=len(tail_b))


def test_ascending_precondition_fallback():
    """`_gradient_slices_strictly_ascending` forced False: every OD is
    served on the unchanged baseline kernel (fast_calls == 0) and the
    engine outputs stay byte-identical to the unpatched composed arm."""
    spec = flow_spec_long()
    overrides = {"search_radius": 4.0}
    flow_mod = cns().flow_module
    real_prec = flow_mod._gradient_slices_strictly_ascending
    try:
        flow_mod._gradient_slices_strictly_ascending = (
            lambda gi, gn: False)
        try:
            eng = run_flow_arm(cns(), spec, 1, extra_overrides=overrides)
        finally:
            flow_mod._gradient_slices_strictly_ascending = real_prec
        stats = getattr(eng, "_f2_stats", None)
        assert stats is not None
        assert stats["fast_calls"] == 0
        assert stats["fallback_calls"] == 0
    finally:
        flow_mod._gradient_slices_strictly_ascending = real_prec
    eng_b = run_flow_arm(b0ns(), spec, 1, extra_overrides=overrides)
    assert_flow_bytes(flow_output_arrays(eng),
                      flow_output_arrays(eng_b), "f2f3/precondition_fallback")
    record("precondition_fallback")


def test_gradient_flow_gradient_interleave():
    """F3 gradient call → F2/F1 flow Centrality → F3 gradient call again
    in one process, composed arm; every stage byte-equal to a fresh B0
    arm of the same stage (workspace frees disturb nothing across
    tracks)."""
    # stage 1: gradient (production cap)
    _, _, out_b1, out_c1 = _gradient_pair(None)
    assert_gradient_bytes(out_c1, out_b1, "f2f3/interleave/grad_1")

    # stage 2: flow Centrality on the admitting spec
    spec = flow_spec_long()
    overrides = {"search_radius": 4.0}
    eng_c, _settings = build_flow_arm(cns(), spec, 1,
                                      extra_overrides=overrides)
    eng_c.Centrality(_settings)
    eng_b = run_flow_arm(b0ns(), spec, 1, extra_overrides=overrides)
    assert_flow_bytes(flow_output_arrays(eng_c),
                      flow_output_arrays(eng_b), "f2f3/interleave/flow")

    # stage 3: gradient again (forced chunks this time)
    stub_b, stub_c, out_b3, out_c3 = _gradient_pair(FORCED_CHUNK_CAP)
    assert_gradient_bytes(out_c3, out_b3, "f2f3/interleave/grad_3")

    # stage-1 repeat must reproduce stage-1 bytes exactly
    _, _, out_b1r, out_c1r = _gradient_pair(None)
    assert_gradient_bytes(out_c1r, out_b1r, "f2f3/interleave/grad_1_repeat")
    for i, name in enumerate(("indptr", "nodes", "dist", "pred")):
        assert out_c1r[i].tobytes() == out_c1[i].tobytes(), \
            f"f2f3/interleave/grad_1_repeat/{name}: not byte-stable"
    record("gradient_flow_gradient_interleave")
