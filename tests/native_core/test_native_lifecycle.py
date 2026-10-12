"""Lifecycle: worker partitioning, cancellation, ownership, determinism.

Worker-count invariance (SPEC.md section 6: outputs are
origin-partitioned, so threads cannot move a bit), cooperative
cancellation at origin boundaries (incomplete KernelOutputs, never
partial publication), owner-lifetime safety across GC, and run-to-run
determinism including scratch reuse.
"""
import gc
import threading
import time

import numpy as np
import pytest

from urban_network_analysis.backends.native import (
    new_cancel_flag,
    route_scope_access,
)
from urban_network_analysis.backends.native._loader import (
    NativeUnavailableError,
    load_core,
)

from conftest import bits, build_case, corrected_case

try:
    CORE = load_core()
except NativeUnavailableError:  # pragma: no cover - unbuilt host
    CORE = None

pytestmark = pytest.mark.skipif(CORE is None,
                                reason="native core not built on this host")


def fold_args(c):
    return (c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
            c["d"], c["dw"], c["dwt"], 0.02, 5.0, 8.0, 0.3, "exponential",
            np.array([0.9, 0.5, 0.25, 0.1]), c["cutoff"])


def multi_origin_case(name="base", k=9):
    """The battery case with K origin rows (identical rows: the
    partition law is per-origin, so identical origins make any
    worker-assignment bit difference impossible to hide)."""
    c = corrected_case(name)
    o = np.repeat(c["o"], k, axis=0)
    ow = np.repeat(c["ow"], k, axis=0)
    return dict(c, o=o, ow=ow)


# ---------------------------------------------------------------------------
# Worker partitioning: threads never move a bit or a counter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile", ["una_legacy", "corrected_v1"])
@pytest.mark.parametrize("threads", [1, 2, 4])
def test_threads_bitwise_and_counter_invariant(profile, threads):
    c = multi_origin_case()
    run = route_scope_access(profile, *fold_args(c), threads=threads)
    assert run.native_engaged is True
    assert run.outputs.status == "complete"
    if threads == 1:
        return  # baseline run; the others must match it bit-for-bit
    base = route_scope_access(profile, *fold_args(c), threads=1)
    assert bits(run.result.reach) == bits(base.result.reach)
    assert bits(run.result.gravity_exponential) == \
        bits(base.result.gravity_exponential)
    assert bits(run.result.gravity_logistic) == bits(base.result.gravity_logistic)
    assert bits(run.result.knn_access) == bits(base.result.knn_access)
    # Engagement is a function of the WORK (origins), not the partition.
    assert run.engagement["heap_pushes"] == base.engagement["heap_pushes"]
    assert run.engagement["heap_pops"] == base.engagement["heap_pops"]
    assert run.engagement["rows_scanned"] == base.engagement["rows_scanned"]
    assert run.engagement["eligible_collected"] == \
        base.engagement["eligible_collected"]
    assert run.engagement["kept_count"] == base.engagement["kept_count"]


# ---------------------------------------------------------------------------
# Cancellation: bounded-boundary polls, incomplete, no partial publication
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile", ["una_legacy", "corrected_v1"])
def test_precancelled_run_incomplete_no_publication(profile):
    c = multi_origin_case(k=3)
    cancel = new_cancel_flag()
    cancel.cancel()
    run = route_scope_access(profile, *fold_args(c), cancel=cancel)
    assert run.native_engaged is True
    assert run.outputs.status == "incomplete"
    assert run.outputs.arrays == ()
    assert run.outputs.next_logical_state == "origin-0"
    assert run.result is None
    assert run.engagement["first_incomplete_origin"] == 0


@pytest.mark.parametrize("profile", ["una_legacy", "corrected_v1"])
def test_midrun_cancellation_lands_between_origins(profile):
    """A watcher cancels a many-origin run in flight; the run reports
    incomplete with a valid first-incomplete origin and publishes
    nothing.  Bounded attempts keep the test time finite on a fast
    host (a completed run before the cancel lands retries larger)."""
    for k in (64, 256, 1024, 4096):
        case = multi_origin_case(k=k)
        cancel = new_cancel_flag()

        def watcher():
            time.sleep(0.001)
            cancel.cancel()

        t = threading.Thread(target=watcher)
        t.start()
        run = route_scope_access(profile, *fold_args(case), cancel=cancel)
        t.join()
        if run.outputs.status == "incomplete":
            o_count = case["o"].shape[0]
            first = int(run.engagement["first_incomplete_origin"])
            assert 0 <= first < o_count
            assert run.outputs.next_logical_state == f"origin-{first}"
            assert run.outputs.arrays == ()
            assert run.result is None
            return
        assert run.outputs.status == "complete"
    pytest.fail("cancellation never landed despite up to 4096 origins")


def test_precancelled_caller_buffer_untouched():
    """Binding level: a pre-cancelled run writes NOTHING into the
    caller's buffer (cancellation is checked before the first origin)."""
    c = multi_origin_case(k=2)
    cancel = CORE.CancelFlag()
    cancel.cancel()
    out = np.zeros((c["o"].shape[0], c["d"].shape[0]), dtype=np.float64)
    rep = CORE.run_corrected(c["ptr"], c["vec"], c["wts"], c["o"], c["ow"],
                             c["d"], c["dw"], c["cutoff"], 1, cancel, out)
    codes = CORE.status_codes()
    assert int(rep["status"]) == int(codes["ERR_CANCELLED"])
    assert int(rep["first_incomplete_origin"]) == 0
    assert not out.any()


# ---------------------------------------------------------------------------
# Ownership: inputs releasable after the call; outputs own their buffers
# ---------------------------------------------------------------------------

def test_owner_arrays_releasable_after_call():
    """The binding borrows inputs for the call only; after the route
    returns, dropping every owner (and collecting) leaves the published
    arrays intact (values snapshotted before release; ASan re-proves
    the no-use-after-free law on the same battery in Phase 2)."""
    c = multi_origin_case(k=2)
    args = fold_args(c)
    expected = route_scope_access("una_legacy", *args).result
    keep = route_scope_access("una_legacy", *args)
    del c, args
    gc.collect()
    assert bits(keep.result.reach) == bits(expected.reach)
    assert bits(keep.result.knn_access) == bits(expected.knn_access)


# ---------------------------------------------------------------------------
# Determinism: repeated runs, scratch reuse, interleaved profiles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile", ["una_legacy", "corrected_v1"])
def test_repeated_runs_identical(profile):
    """Same inputs twice: identical bits AND identical engagement (the
    per-worker scratch is reset per origin — no state leaks between
    runs, none depends on prior buffer contents)."""
    c = multi_origin_case(k=5)
    one = route_scope_access(profile, *fold_args(c))
    two = route_scope_access(profile, *fold_args(c))
    assert bits(one.result.reach) == bits(two.result.reach)
    assert bits(one.result.gravity_exponential) == \
        bits(two.result.gravity_exponential)
    assert bits(one.result.knn_access) == bits(two.result.knn_access)
    assert one.engagement["heap_pushes"] == two.engagement["heap_pushes"]
    assert one.engagement["rows_scanned"] == two.engagement["rows_scanned"]


def test_interleaved_profiles_stable():
    """Legacy and corrected runs interleaved on the same graph disagree
    exactly where their semantics disagree and agree where they must —
    no cross-profile state in the core."""
    c = multi_origin_case(k=2)
    leg_a = route_scope_access("una_legacy", *fold_args(c)).result
    cor_a = route_scope_access("corrected_v1", *fold_args(c)).result
    leg_b = route_scope_access("una_legacy", *fold_args(c)).result
    cor_b = route_scope_access("corrected_v1", *fold_args(c)).result
    assert bits(leg_a.reach) == bits(leg_b.reach)
    assert bits(cor_a.reach) == bits(cor_b.reach)
    # Same validated domain (nonnegative finite costs): the corrected
    # labels are the true shortest walks, so they never exceed the
    # legacy schedule's labels bit-plainly is NOT claimed — only
    # self-stability across interleaving is asserted here (the profiles
    # are distinct semantics, pinned separately).
