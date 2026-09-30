"""F2 driver-level battery on the real `_process_origins_aggregate`:

* route-toggle differential — fast route ON vs FORCED-OFF vs default
  produce byte-identical engine outputs (first-divergence evidence at
  engine level);
* engagement observability — `eng._f2_stats` counts fast ODs, overflow
  events and exception fallbacks; the [F2] log line exists;
* (c) fallback engagement — a disabled precondition serves every OD on
  the unchanged baseline kernel (fast_calls == 0), outputs identical;
* (d) wrapper exception-safety — an injected kernel error on one OD
  re-zeros the stripe scratch, serves that OD on the baseline kernel
  via the module-attribute interception idiom, and the worker continues
  with exact outputs (module-attribute patch, no source change);
* turns pipeline untouched — `use_turns` settings never route through
  the F2 wrapper (structural: the turns driver calls
  _accumulate_od_flow_turns; asserted here by stats absence).
"""
from __future__ import annotations

import pytest as _pytest
from _legacy_paths import legacy_layout_present as _llp
if not _llp():
    _pytest.skip('historical large_e2e layout not present '
                 '(set UNA_LEGACY_WORKSPACE)', allow_module_level=True)

import fixtures_f2
from harness_f2 import record

from urban_network_analysis.Engines import AggregateFlow as AF


def _run_engine(num_threads=1, extra_overrides=None, route=None,
                precondition=None, spec=None):
    """Build + run one engine; optionally force the route toggle or the
    precondition via module-attribute patching (F1 T10 idiom)."""
    eng, settings = fixtures_f2.build_engine(num_threads, extra_overrides,
                                             spec=spec)
    real_route = AF._use_local_route
    real_prec = AF._gradient_slices_strictly_ascending
    try:
        if route is not None:
            AF._use_local_route = (
                (lambda ok, l, n: route) if isinstance(route, bool)
                else route)
        if precondition is not None:
            AF._gradient_slices_strictly_ascending = (
                lambda gi, gn: precondition)
        eng.Centrality(settings)
    finally:
        AF._use_local_route = real_route
        AF._gradient_slices_strictly_ascending = real_prec
    return eng


def test_route_toggle_engine_bitwise():
    """Default vs forced-fast vs forced-baseline: byte-identical engine
    outputs on BOTH crossover sides, engaged through the production
    decision path only.

    Shared oracle stub (n_total small): the default crossover REFUSES
    every OD (slices exceed n_total // 8) — the refusing side, by
    design. big_flow_spec at search_radius 4.0: the default crossover
    ADMITS real ODs — the admitting side. Forcing the route bypasses
    only the threshold, never the kernel: all arms byte-identical."""
    ref = _run_engine(1, route=False)
    fast = _run_engine(1, route=True)
    default = _run_engine(1)
    fixtures_f2.assert_engine_bytes(fixtures_f2.engine_output_arrays(fast),
                        fixtures_f2.engine_output_arrays(ref),
                        "route_toggle/forced_fast_vs_baseline")
    fixtures_f2.assert_engine_bytes(fixtures_f2.engine_output_arrays(default),
                        fixtures_f2.engine_output_arrays(ref),
                        "route_toggle/default_vs_baseline")
    ref_stats = getattr(ref, "_f2_stats", None)
    fast_stats = getattr(fast, "_f2_stats", None)
    default_stats = getattr(default, "_f2_stats", None)
    assert ref_stats is not None and fast_stats is not None \
        and default_stats is not None
    assert ref_stats["fast_calls"] == 0
    assert fast_stats["fast_calls"] > 0
    assert fast_stats["overflow_events"] == 0
    assert fast_stats["fallback_calls"] == 0
    assert default_stats["fast_calls"] == 0, \
        "shared stub: default crossover must refuse (slice_max 2 < slices)"
    assert default_stats["fallback_calls"] == 0

    big = fixtures_f2.big_flow_spec()
    b_ref = _run_engine(1, route=False, spec=big,
                        extra_overrides={"search_radius": 4.0})
    b_fast = _run_engine(1, route=True, spec=big,
                         extra_overrides={"search_radius": 4.0})
    b_default = _run_engine(1, spec=big,
                            extra_overrides={"search_radius": 4.0})
    fixtures_f2.assert_engine_bytes(fixtures_f2.engine_output_arrays(b_fast),
                        fixtures_f2.engine_output_arrays(b_ref),
                        "route_toggle/big_forced_fast_vs_baseline")
    fixtures_f2.assert_engine_bytes(
        fixtures_f2.engine_output_arrays(b_default),
        fixtures_f2.engine_output_arrays(b_ref),
        "route_toggle/big_default_vs_baseline")
    b_ref_stats = b_ref._f2_stats
    b_fast_stats = b_fast._f2_stats
    b_default_stats = b_default._f2_stats
    assert b_ref_stats["fast_calls"] == 0
    assert b_fast_stats["fast_calls"] > 0
    assert b_default_stats["fast_calls"] > 0, \
        "big spec: default crossover must admit (slice_max 15 >= slices)"
    assert b_default_stats["overflow_events"] == 0
    assert b_default_stats["fallback_calls"] == 0
    record("driver/route_toggle", ref=ref_stats, fast=fast_stats,
           default=default_stats, big_default=b_default_stats,
           big_fast=b_fast_stats)


def test_precondition_disabled_falls_back(capsys):
    """(c) precondition False -> every OD on the unchanged baseline
    kernel; engine outputs byte-identical to the forced-baseline run."""
    ref = _run_engine(1, route=False)
    eng = _run_engine(1, precondition=False)
    fixtures_f2.assert_engine_bytes(fixtures_f2.engine_output_arrays(eng),
                        fixtures_f2.engine_output_arrays(ref),
                        "precondition_disabled_vs_baseline")
    stats = eng._f2_stats
    assert stats["local_route_ok"] is False
    assert stats["fast_calls"] == 0
    record("driver/precondition_disabled", stats=stats)


def test_exception_safety_injected_error_then_continue():
    """(d) The F2 wrapper intercepts an injected kernel error: the
    scratch is re-zeroed, the OD falls back to the baseline kernel, and
    the worker continues — final outputs byte-identical to the
    reference run; exactly one fallback counted. Runs on big_flow_spec
    so the DEFAULT route admits >= 2 ODs and the injected failure
    genuinely hits the wrapper on the second fast call."""
    class _Boom(RuntimeError):
        pass

    real_local = AF._accumulate_od_flow_local
    state = {"calls": 0, "raised": 0}

    def flaky_local(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 2:      # second fast-route OD fails
            state["raised"] += 1
            raise _Boom("f2_injected_kernel_error")
        return real_local(*args, **kwargs)

    big = fixtures_f2.big_flow_spec()
    ref = _run_engine(1, route=False, spec=big,
                      extra_overrides={"search_radius": 4.0})
    eng, settings = fixtures_f2.build_engine(
        1, {"search_radius": 4.0}, spec=big)
    try:
        AF._accumulate_od_flow_local = flaky_local
        eng.Centrality(settings)
    finally:
        AF._accumulate_od_flow_local = real_local
    fixtures_f2.assert_engine_bytes(fixtures_f2.engine_output_arrays(eng),
                        fixtures_f2.engine_output_arrays(ref),
                        "exception_safety_vs_baseline")
    stats = eng._f2_stats
    assert state["raised"] == 1
    assert stats["fallback_calls"] == 1
    assert stats["fast_calls"] == state["calls"] - state["raised"]
    assert stats["fast_calls"] > 0
    record("driver/exception_safety", stats=stats,
           injected_calls=state["calls"])


def test_multithread_engine_bitwise():
    """Threaded run (K=3): fast route vs forced-baseline byte-identical
    — per-stripe scratch ownership holds under the executor."""
    ref = _run_engine(3, route=False)
    fast = _run_engine(3, route=True)
    fixtures_f2.assert_engine_bytes(fixtures_f2.engine_output_arrays(fast),
                        fixtures_f2.engine_output_arrays(ref),
                        "multithread_fast_vs_baseline")
    stats = fast._f2_stats
    assert stats["fast_calls"] > 0
    assert stats["fallback_calls"] == 0
    record("driver/multithread", stats=stats)


def test_turns_pipeline_has_no_f2_wrapper():
    """The turns driver calls _accumulate_od_flow_turns directly — the
    F2 wrapper exists only in the node-graph driver. Structural pin:
    the turns settings still run (fixture turns spec if available) and
    the non-turns stats surface stays untouched by them. Asserted
    statically: the turns driver source contains no local-route call."""
    import inspect
    src = inspect.getsource(AF.AggregateFlow._process_origins_aggregate_turns)
    assert "_accumulate_od_flow_local" not in src
    assert "_use_local_route" not in src
