"""I00 composition: F1×F2 flow scratch ownership.

In the composed AggregateFlow, F1's nogil fixed-stripes execution wraps
F2's local-overlap OD route: stripes are distributed to the sequential
path or the thread-pool futures path, and each stripe's ODs route
through `_use_local_route` into the shared local scratch with bounded
touched lists. Composition risks: scratch reset semantics under stripe
parallelism, and route-contract retention (forced-fast vs forced-
baseline vs default still agree byte-for-byte).

Pinned here, in the composed tree:

* composed-vs-B0 byte equality on the crossover-refusing oracle spec
  and the admitting long-path spec, at K=1 (sequential
  `_process_stripe`) and K=2 (futures path);
* route triad (forced-fast / forced-baseline / default) byte-identical
  within the composed module on both specs;
* engagement observability: `_f2_stats` fast_calls > 0 exactly on the
  admitting spec, == 0 on the refusing spec;
* sequential reuse and engine interleaving leave no residue.
"""
from __future__ import annotations

import pytest

from fixtures_composition import flow_spec_long, flow_spec_oracle
from harness_composition import (assert_flow_arms_equal, assert_flow_bytes,
                                 b0ns, build_flow_arm, cns,
                                 flow_output_arrays, record, run_flow_arm)

SPECS = {"oracle_refusing": flow_spec_oracle, "long_admitting": flow_spec_long}


def _long_overrides():
    return {"search_radius": 4.0}


@pytest.mark.parametrize("k", (1, 2))
@pytest.mark.parametrize("spec_name", sorted(SPECS))
def test_composed_vs_b0_byte_equal(spec_name, k):
    """Composed AggregateFlow (F1 striping + F2 local route at their
    production defaults) vs B0: byte-equal outputs at both execution
    paths."""
    spec = SPECS[spec_name]()
    overrides = _long_overrides() if spec_name == "long_admitting" else None
    assert_flow_arms_equal(spec, k, f"f1f2/{spec_name}/k{k}",
                           extra_overrides=overrides)
    record("composed_vs_b0", spec=spec_name, k=k)


@pytest.mark.parametrize("spec_name", sorted(SPECS))
def test_route_triad_byte_identical(spec_name):
    """Forced-fast vs forced-baseline vs default: byte-identical within
    the composed module (the route toggle bypasses only the threshold,
    never the kernel — retained under composition)."""
    spec = SPECS[spec_name]()
    overrides = _long_overrides() if spec_name == "long_admitting" else None
    flow_mod = cns().flow_module
    real_route = flow_mod._use_local_route
    try:
        outs = {}
        for name, route in (("forced_fast", True), ("forced_baseline", False),
                            ("default", None)):
            if route is not None:
                flow_mod._use_local_route = (lambda ok, l, n, r=route: r)
            try:
                eng = run_flow_arm(cns(), spec, 1, extra_overrides=overrides)
            finally:
                flow_mod._use_local_route = real_route
            outs[name] = flow_output_arrays(eng)
        assert_flow_bytes(outs["forced_fast"], outs["forced_baseline"],
                          f"f1f2/triad/{spec_name}/fast_vs_baseline")
        assert_flow_bytes(outs["default"], outs["forced_baseline"],
                          f"f1f2/triad/{spec_name}/default_vs_baseline")
    finally:
        flow_mod._use_local_route = real_route
    record("route_triad", spec=spec_name)


@pytest.mark.parametrize("spec_name", sorted(SPECS))
def test_f2_engagement_stats(spec_name):
    """`_f2_stats` observability in the composed tree: fast_calls > 0
    exactly on the admitting spec, == 0 on the refusing spec."""
    spec = SPECS[spec_name]()
    overrides = _long_overrides() if spec_name == "long_admitting" else None
    eng = run_flow_arm(cns(), spec, 1, extra_overrides=overrides)
    stats = getattr(eng, "_f2_stats", None)
    assert stats is not None, "composed Centrality set no _f2_stats"
    if spec_name == "long_admitting":
        assert stats["fast_calls"] > 0
        assert stats["overflow_events"] == 0
        assert stats["fallback_calls"] == 0
    else:
        assert stats["fast_calls"] == 0
    record("f2_stats", spec=spec_name, **stats)


def test_sequential_reuse_same_engine():
    """Second Centrality on the SAME composed engine reproduces the
    first run byte-exactly (F2 scratch reset + F1 stripe state leave no
    residue), and equals a fresh B0 run of the same spec."""
    spec = flow_spec_long()
    eng, settings = build_flow_arm(cns(), spec, 1,
                                   extra_overrides=_long_overrides())
    eng.Centrality(settings)
    first = flow_output_arrays(eng)
    eng.Centrality(settings)
    second = flow_output_arrays(eng)
    assert_flow_bytes(second, first, "f1f2/reuse/cand_second_vs_first")
    eng_b = run_flow_arm(b0ns(), spec, 1, extra_overrides=_long_overrides())
    assert_flow_bytes(first, flow_output_arrays(eng_b),
                      "f1f2/reuse/cand_first_vs_b0")
    record("sequential_reuse", spec="long_admitting")


def test_interleaved_engines_no_residue():
    """Two composed engines alternating on different specs: each final
    run equals the byte pattern of a fresh same-spec run (module-level
    scratch ownership holds across engines)."""
    spec_long = flow_spec_long()
    spec_small = flow_spec_oracle()
    eng_a, settings_a = build_flow_arm(cns(), spec_long, 1,
                                       extra_overrides=_long_overrides())
    eng_a.Centrality(settings_a)
    first_a = flow_output_arrays(eng_a)

    eng_b, settings_b = build_flow_arm(cns(), spec_small, 1)
    eng_b.Centrality(settings_b)

    eng_a2, settings_a2 = build_flow_arm(cns(), spec_long, 1,
                                         extra_overrides=_long_overrides())
    eng_a2.Centrality(settings_a2)
    assert_flow_bytes(flow_output_arrays(eng_a2), first_a,
                      "f1f2/interleave/repeat_a_after_b")

    fresh_a = run_flow_arm(cns(), spec_long, 1,
                           extra_overrides=_long_overrides())
    assert_flow_bytes(flow_output_arrays(fresh_a), first_a,
                      "f1f2/interleave/repeat_a_vs_fresh_a")
    record("interleaved_engines")
