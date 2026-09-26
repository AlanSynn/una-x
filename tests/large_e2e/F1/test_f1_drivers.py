"""F1 engine/driver-level tests (proof 10): T2 engine battery incl.
sequential reuse, T9 path routing (fast path vs executor), T10
exception semantics via module-attribute interception, T12 progress-log
content determinism."""
from __future__ import annotations

import re
import sys
import time

import numpy as np
import pytest

import fixtures_f1
from harness_f1 import (assert_engine_bytes, b0ns, cns, record_warmup)


CASE_NAMES = sorted(fixtures_f1.ENGINE_CASES)


def _run(ns, case, k):
    eng, settings = fixtures_f1.build_engine(ns, case, k)
    t0 = time.perf_counter()
    eng.Centrality(settings)
    if (case, k) == ("base", 1):
        # first full engine assembly of the session per arm — record it
        arm = "b0" if ns is not cns() else "cand"
        record_warmup(f"engine_warmup/{arm}/base_k1",
                      time.perf_counter() - t0)
    return eng


def test_t2_engine_battery_two_arm():
    """Candidate AggregateFlow vs B0 engine on every engine case at the
    sequential K=1 profile: edge_flow, edge_flow_AB, edge_flow_BA,
    node_flow and observer flows byte-equal."""
    for case in CASE_NAMES:
        eng_b = _run(b0ns(), case, 1)
        eng_c = _run(cns(), case, 1)
        assert_engine_bytes(fixtures_f1.engine_output_arrays(eng_c),
                            fixtures_f1.engine_output_arrays(eng_b),
                            f"t2/{case}")


def test_t2_sequential_reuse_same_engine():
    """Second Centrality on the SAME engine object reproduces the first
    run byte-exactly in both arms (no cross-run contamination)."""
    for ns, tag in ((b0ns(), "b0"), (cns(), "cand")):
        eng, settings = fixtures_f1.build_engine(ns, "base", 1)
        eng.Centrality(settings)
        first = fixtures_f1.engine_output_arrays(eng)
        eng.Centrality(settings)
        second = fixtures_f1.engine_output_arrays(eng)
        assert_engine_bytes(second, first, f"t2_reuse/{tag}")


@pytest.mark.parametrize("k", [1, 3])
def test_t9_path_routing(k):
    """n_threads==1 fast path (k=1) and executor path (k=3) both
    byte-equal candidate-vs-B0 — the unchanged-fallback reachability
    obligation. K is never compared across."""
    eng_b = _run(b0ns(), "base", k)
    eng_c = _run(cns(), "base", k)
    assert_engine_bytes(fixtures_f1.engine_output_arrays(eng_c),
                        fixtures_f1.engine_output_arrays(eng_b),
                        f"t9/k{k}")


def test_t10_exception_semantics():
    """Module-attribute interception (no source change): a wrapper that
    raises on the chosen origin's first kernel call makes Centrality
    propagate the exception in BOTH arms; post-failure result arrays are
    still zeros (reduction not reached); after restoring the attribute a
    fresh Centrality on the same object produces the correct output."""
    chosen_origin_vn = 9 + 3  # n_net + n_dest: origin o0's virtual node

    class _Boom(RuntimeError):
        pass

    def make_wrapper():
        def wrapper(*args, **kwargs):
            if int(args[9]) == chosen_origin_vn:
                raise _Boom("f1_t10_injected")
            return real_kernel(*args, **kwargs)
        return wrapper

    reference = {}
    for ns, tag in ((b0ns(), "b0"), (cns(), "cand")):
        mod_name = ("urban_network_analysis.Engines.AggregateFlow"
                    if tag == "b0" else "una_f1_cand.Engines.AggregateFlow")
        mod = sys.modules[mod_name]
        real_kernel = mod._accumulate_od_flow
        eng, settings = fixtures_f1.build_engine(ns, "base", 1)
        try:
            mod._accumulate_od_flow = make_wrapper()
            with pytest.raises(_Boom):
                eng.Centrality(settings)
        finally:
            mod._accumulate_od_flow = real_kernel
        # reduction never ran: result arrays are the allocated zeros
        assert float(np_abs_max(eng.edge_flow_AB)) == 0.0
        assert float(np_abs_max(eng.edge_flow_BA)) == 0.0
        assert float(np_abs_max(eng.node_flow)) == 0.0
        # same object, clean rerun
        eng.Centrality(settings)
        reference[tag] = fixtures_f1.engine_output_arrays(eng)
    assert_engine_bytes(reference["cand"], reference["b0"], "t10/rerun")
    # and the rerun equals the plain battery output
    eng_b = _run(b0ns(), "base", 1)
    assert_engine_bytes(reference["b0"],
                        fixtures_f1.engine_output_arrays(eng_b),
                        "t10/rerun-vs-battery")


def np_abs_max(arr):
    return float(np.max(np.abs(arr))) if arr is not None and \
        arr.shape[0] else 0.0


_PROGRESS_RE = re.compile(
    r"origin ([0-9,]+)/([0-9,]+) \(([0-9.]+)%, ([0-9]+)/s, "
    r"ETA ([0-9.]+) min\)")


def test_t12_progress_log_content():
    """Progress event content sequence (origin k/n) is deterministic
    and arm-invariant at K=3; rate/ETA fields are compared only
    structurally (present, parseable) — never by value."""
    seqs = {}
    for ns, tag in ((b0ns(), "b0"), (cns(), "cand")):
        eng, settings = fixtures_f1.build_engine(ns, "base", 3)
        eng.Centrality(settings)
        progress = []
        for event, details in eng.topology.logger.log_list:
            if event != "AggregateFlow":
                continue
            m = _PROGRESS_RE.search(details)
            if m:
                done = int(m.group(1).replace(",", ""))
                total = int(m.group(2).replace(",", ""))
                float(m.group(3))  # parseable percent
                float(m.group(4))  # parseable rate
                float(m.group(5))  # parseable ETA
                progress.append((done, total))
        assert progress, f"{tag}: no progress lines captured"
        assert progress[-1][0] == progress[-1][1], \
            f"{tag}: progress sequence does not reach completion"
        seqs[tag] = progress
    assert seqs["cand"] == seqs["b0"], "progress k/n sequence diverged"
