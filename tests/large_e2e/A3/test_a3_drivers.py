"""A3 driver-level battery (proof.md section 7, T1, T7, T8, T10).

Route coverage: the guard-firing variants exercise the tail-free
private route in all four drivers (od/integrated, both engine
families); the refusal variants pin that an A3 refusal keeps the A1
route and an A1 refusal keeps the original loop — every output
byte-identical to B0 (regression pin P4 / public-path invariance T10).
"""
from __future__ import annotations

import numpy as np
import pytest

import fixtures_a3
from harness_a3 import (
    assert_integrated_bytes,
    assert_od_bytes,
    b0ns,
    cns,
    run_od,
)

FIRING_VARIANTS = ("valid", "boundary_values", "shared_terminals",
                   "zero_destinations")
REFUSAL_VARIANTS = ("oob_value", "negative_value", "extra_row",
                    "a1_refusal")
SEEDS = (20260925, 20260926, 20260927)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("variant", FIRING_VARIANTS)
def test_firing_route_byte_equal(variant, seed):
    arrays = getattr(fixtures_a3, variant)(seed=seed)
    assert_od_bytes(arrays, f"{variant}/{seed}")
    assert_integrated_bytes(arrays, f"{variant}/{seed}")


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("variant", REFUSAL_VARIANTS)
def test_refusal_route_byte_equal(variant, seed):
    """A3 refused (or A1 refused) falls through unchanged; outputs stay
    byte-identical to the pre-change build on the same inputs."""
    arrays = getattr(fixtures_a3, variant)(seed=seed)
    assert_od_bytes(arrays, f"{variant}/{seed}")
    assert_integrated_bytes(arrays, f"{variant}/{seed}")


def test_repeated_call_leaves_no_residue():
    """T7 at the callable level: every guarded input is validated up
    front, so no mid-loop exception path exists; the residue claim is
    that repeated and interleaved calls on the same array objects are
    byte-stable and identical to B0 (per-iteration allocation; nothing
    to reset)."""
    arrays = fixtures_a3.valid()
    first_families = {}
    for family in ("acce", "acc"):
        out_b0 = np.asarray(run_od(b0ns(), arrays, family=family))
        first_families[family] = out_b0
        out_c1 = np.asarray(run_od(cns(), arrays, family=family))
        assert out_c1.tobytes() == out_b0.tobytes()

    refused = fixtures_a3.oob_value()
    assert_od_bytes(refused, "residue/interleaved")

    for family in ("acce", "acc"):
        out_b0 = np.asarray(run_od(b0ns(), arrays, family=family))
        out_c2 = np.asarray(run_od(cns(), arrays, family=family))
        assert out_c2.tobytes() == out_b0.tobytes()
        assert out_c2.tobytes() == first_families[family].tobytes()

    assert_integrated_bytes(arrays, "residue/integrated-repeat")


def test_shared_input_arrays_untouched_by_drivers():
    """T5 at the driver level: the driver mutates nothing it was
    given."""
    arrays = fixtures_a3.valid()
    keys = ("adjacency_pointer", "adjacency_vector",
            "adjacency_vector_weights", "adjacynct_vector_network_node",
            "o_terminal_idxs", "o_terminal_weights",
            "d_terminal_idxs", "d_terminal_weights", "d_weights")
    before = {k: np.ascontiguousarray(arrays[k]).tobytes() for k in keys}
    assert_od_bytes(arrays, "readonly/od")
    assert_integrated_bytes(arrays, "readonly/integrated")
    for k in keys:
        assert np.ascontiguousarray(arrays[k]).tobytes() == before[k], k
