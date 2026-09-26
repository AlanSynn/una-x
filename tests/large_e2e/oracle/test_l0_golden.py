"""L0 golden corpus test: the current compiled B0 must reproduce the
frozen golden byte-for-byte, and the golden assets must be intact.

The golden was produced ONLY by golden/build_golden.py from the B0
subprocess; this test re-derives every value from live compiled B0 in
this process and compares dtype+shape+bytes. No candidate code is
loaded."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

import fixtures
from comparator import assert_array_bytes_equal
from runner_arm import run_battery
from support import b0, golden_hashes, golden_meta


@pytest.fixture(scope="module")
def current_battery():
    """Live compiled-B0 battery run in this process (single pass)."""
    b = b0()
    store, meta = {}, {"checks": {}, "warnings": []}
    run_battery(b, store, meta)
    return store, meta


def test_golden_assets_intact():
    h = golden_hashes()
    for name in ("golden_b0.npz", "golden_b0.json"):
        digest = hashlib.sha256((golden_dir() / name).read_bytes()).hexdigest()
        assert digest == h[name], f"{name} mutated after golden build"


def golden_dir():
    from support import GOLDEN_DIR
    return GOLDEN_DIR


def test_battery_covers_expected_keys(current_battery):
    store, _ = current_battery
    digests = golden_hashes()["key_digests"]
    assert len(store) == len(digests)
    # baseline corpus size (a drop means a fixture disappeared)
    assert len(store) >= 1031, len(store)
    # every crafted case family is present
    prefixes = ("l0_", "l1_", "flt_", "metric_", "k/", "flow/")
    for key in digests:
        assert key.startswith(prefixes), key


def test_current_run_matches_golden_bytes(current_battery):
    store, _ = current_battery
    golden = np.load(golden_dir() / "golden_b0.npz")
    assert set(store) == set(golden.files)
    for key in sorted(store):
        assert_array_bytes_equal(np.asarray(store[key]),
                                 np.asarray(golden[key]), key)


def test_engine_row_order_replica_matches_live(current_battery):
    """The fixture-side Python replica of B0's adjacency row order must
    agree with the live engine construction on every case (this is what
    lets kernel-level traces address engine fixtures)."""
    _, meta = current_battery
    bad = [k for k, v in meta["checks"]["engine_vs_replica_row_order"].items()
           if not v]
    assert bad == []


def test_input_immutability_and_output_ownership(current_battery):
    _, meta = current_battery
    assert all(meta["checks"]["input_immutability"].values())
    assert not any(meta["checks"]["scope_output_ownership"].values())


def test_refusal_domains_raise_and_are_recorded(current_battery):
    _, meta = current_battery
    probes = meta["refusal_probes"]
    assert probes["scope_cutoff_str"]["exception_type"] == "TypingError"
    assert probes["scope_cutoff_none"]["exception_type"] == "TypingError"
    assert probes["scope_object_weights"]["exception_type"] == "TypingError"
    assert probes["engine_mismatched_origin_weights"][
        "exception_type"] == "ValueError"


def test_kernel_scope_battery_bytes(current_battery):
    """Direct spot-checks of the adversarial kernel cases against the
    golden (named diagnostics, not just the blanket byte comparison)."""
    store, _ = current_battery
    golden = np.load(golden_dir() / "golden_b0.npz")

    # Duplicate destinations on the ORIGIN row: the row scan is staged
    # (both incidences eligible against the same pop-time labels, last
    # eligible wins) BUT the start node is seeded twice, so the stale
    # seed re-scan re-admits the cheaper incidence and the FINAL label
    # is the cheap value in BOTH orders. The staged-only observable
    # lives on a downstream duplicate row (test_l0_traces /
    # test_negative_mutations); here the converged bytes are pinned.
    cheap_first = store["k/k_dup_dest_cheap_first/scope"]
    assert cheap_first[1] == 2.0, cheap_first
    cheap_last = store["k/k_dup_dest_cheap_last/scope"]
    assert cheap_last[1] == 2.0, cheap_last

    # cutoff boundary: the scope label below cutoff (nextafter(5,-1))
    # plus the 0.5 terminal weight lands ABOVE cutoff after adjustment,
    # so the compiled adjust value is 5.4999... and the destination is
    # NOT retained by the <= cutoff filter (filter_indices empty).
    nxt = store["flt_nextafter_cutoff/adjust/0"]
    below = float(np.nextafter(np.float64(5.0), np.float64(0))) + 0.5
    assert nxt[0] == below, (nxt[0], below)
    assert nxt[0] > 5.0
    assert store["flt_sentinel_two_pow_53/scope/0"][3] == 2.0 ** 53

    for key in ("k/k_dup_dest_cheap_first/scope", "k/k_stale_queue/scope",
                "k/k_seed_equals_sentinel/scope",
                "flt_signed_zero/adjust/0",
                "flt_near_overflow_finite/adjust/0",
                "flt_maxfloat_cutoff/scope/0",
                "flt_inf_incidence_weight/scope/0",
                "metric_knn_logistic_k3/integrated/knn_access"):
        assert_array_bytes_equal(np.asarray(store[key]),
                                 np.asarray(golden[key]), key)


def test_signed_zero_bytes_exact(current_battery):
    store, _ = current_battery
    # flt_signed_zero: scope[1] == -0.0 propagates to dest0 (min(-0,+0))
    # and dest1 (min(+0,-0)); numba-min yields -0.0 in BOTH tie orders —
    # Python's first-arg min would give +0.0 for dest1. Byte 0x80.. is
    # the negative-zero signature.
    adj = np.asarray(store["flt_signed_zero/adjust/0"])
    assert adj[0].tobytes() == np.array(-0.0).tobytes(), adj
    assert adj[1].tobytes() == np.array(-0.0).tobytes(), adj
    assert adj[2].tobytes() == np.array(0.0).tobytes(), adj


def test_specialization_inventory(current_battery):
    """L1: all kernel specializations exercised by the corpus are
    enumerated in the golden meta (both terminal-array layouts)."""
    _, meta = current_battery
    sigs = {k: len(v) for k, v in meta["signatures"].items()}
    assert sigs["compact_vector_node_view_scope"] >= 1
    assert sigs["adjust_destination_distances"] >= 2, (
        "expected both C and F terminal-array layout specializations")
    assert sigs["_accumulate_od_flow"] == 1
