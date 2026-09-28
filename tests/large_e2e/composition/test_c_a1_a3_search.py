"""I00 composition: the A1×A3 sibling dispatch matrix (the deep
accessibility interaction; the module carrying the composition proof
obligation).

In the composed tree both tracks live in one module pair: A1's
snapshot-preserving scratch route is admitted first, and A3's tail-free
private scope route is nested INSIDE the A1-admitted branch (both
engine families, od + integrated drivers). The composition risks are
(1) the composed guard cells select the same route B0's single route
would produce byte-for-byte in every reachable cell, and (2) the two
scratch disciplines (A1 two-phase scratch vs A3 private scope) do not
contaminate each other across repeated/interleaved calls.

Per-cell ENGAGEMENT RECEIPTS (h04 I00 plan R3): every cell records the
nopython-resolved guard values (a1_admitted, a1_max_degree,
a3_admitted) into the artifacts tail — byte-equality alone would let a
cell pass vacuously, so the receipt is the route-execution evidence.
All routes produce identical bytes by design; the guards + the nested
dispatch structure are what discriminate which route ran.

KERNEL-POPULATION COVERAGE (h04 I00 plan R4): the A3I admission was on
the full-nd/tailfree-V partition, so the matrix must EXERCISE both
populations; test_matrix_populations walks the whole matrix, tallies
the receipts, and asserts tailfree_V > 0 AND full_nd > 0.

Cells probed per fixture variant:

* (True, True)   → composed A3 tail-free route fires (tailfree-V).
* (True, False)  → A3 refused; A1 scratch route runs (full-nd).
* (False, ·)     → A1 refused; original kernel loop runs (full-nd);
                   A3 is never evaluated by the dispatch (nested guard).
"""
from __future__ import annotations

import numpy as np
import pytest

import fixtures_composition
from fixtures_composition import GUARD_EXPECTATIONS, SEEDS, VARIANTS
from harness_composition import (assert_integrated_bytes, assert_od_bytes,
                                 b0ns, cns, guard_cell, record,
                                 record_cell_receipt, run_od)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_guard_cells_match_expected_dispatch(variant, seed):
    """Every fixture lands in its intended dispatch cell (guards probed
    under nopython in the composed module) and the cell receipt is
    recorded (R3)."""
    arrays = getattr(fixtures_composition, variant)(seed=seed)
    a1_ok, a1_max, a3_ok = guard_cell(arrays)
    exp_a1, exp_a3 = GUARD_EXPECTATIONS[variant]
    assert a1_ok is exp_a1, f"{variant}/{seed}: a1_admitted {a1_ok}"
    if exp_a3 is not None:
        assert a3_ok is exp_a3, f"{variant}/{seed}: a3_admitted {a3_ok}"
    for family in ("acce", "acc"):
        record_cell_receipt(variant, seed, family, a1_ok, a1_max, a3_ok)
    record("guard_cell", variant=variant, seed=seed,
           a1=a1_ok, a1_max_degree=a1_max, a3=a3_ok)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_every_cell_byte_equal_to_b0(variant, seed):
    """Every reachable cell of the composed matrix produces outputs
    byte-identical to B0 (od + integrated, both engine families), with
    the executed route receipt recorded per cell/family (R3)."""
    arrays = getattr(fixtures_composition, variant)(seed=seed)
    a1_ok, a1_max, a3_ok = guard_cell(arrays)
    assert_od_bytes(arrays, f"matrix/{variant}/{seed}")
    assert_integrated_bytes(arrays, f"matrix/{variant}/{seed}")
    for family in ("acce", "acc"):
        record_cell_receipt(variant, seed, family, a1_ok, a1_max, a3_ok)


def test_matrix_populations():
    """R4: walk the ENTIRE matrix in one deterministic pass, verify
    every cell byte-equal to B0 with its receipt recorded, and assert
    BOTH kernel populations were exercised (tailfree_V > 0 — the A3
    route fired somewhere; full_nd > 0 — the A1 fallback / original
    loop served somewhere). The A3I admission partition was
    full-nd/tailfree-V; a matrix missing either population could not
    carry the admission into composition."""
    tallies = {"tailfree_V": 0, "full_nd": 0}
    for seed in SEEDS:
        for variant in VARIANTS:
            arrays = getattr(fixtures_composition, variant)(seed=seed)
            a1_ok, a1_max, a3_ok = guard_cell(arrays)
            exp_a1, exp_a3 = GUARD_EXPECTATIONS[variant]
            assert a1_ok is exp_a1 and (exp_a3 is None or a3_ok is exp_a3), \
                f"{variant}/{seed}: unexpected cell ({a1_ok}, {a3_ok})"
            assert_od_bytes(arrays, f"populations/{variant}/{seed}")
            assert_integrated_bytes(arrays, f"populations/{variant}/{seed}")
            key = "tailfree_V" if (a1_ok and a3_ok) else "full_nd"
            tallies[key] += 1
            for family in ("acce", "acc"):
                record_cell_receipt(variant, seed, family,
                                    a1_ok, a1_max, a3_ok)
    assert tallies["tailfree_V"] > 0, "matrix never exercised the A3 route"
    assert tallies["full_nd"] > 0, "matrix never exercised the full-nd routes"
    record("matrix_populations", **tallies,
           receipt_total=sum(tallies.values()))


@pytest.mark.parametrize("seed", SEEDS)
def test_input_arrays_untouched_by_composed_drivers(seed):
    """T5 at the composition level: no driver in any cell mutates its
    input arrays (A1 scratch ownership + A3 private scope both hold
    under composition)."""
    for variant in VARIANTS:
        arrays = getattr(fixtures_composition, variant)(seed=seed)
        ins = {k: np.asarray(v).copy() for k, v in arrays.items()
               if isinstance(v, np.ndarray)}
        assert_od_bytes(arrays, f"inputs/{variant}/{seed}")
        assert_integrated_bytes(arrays, f"inputs/{variant}/{seed}")
        for k, v in ins.items():
            assert np.asarray(arrays[k]).tobytes() == v.tobytes(), \
                f"{variant}/{seed}: input {k} mutated"


@pytest.mark.parametrize("seed", SEEDS)
def test_interleaved_cells_leave_no_residue(seed):
    """The three scratch disciplines alternate per call in one process
    (A3 route → A1 fallback route → original loop → A3 route): outputs
    byte-stable and identical to fresh B0 runs of the same fixtures."""
    seq = ("both_admit", "a3_refuse_oob", "a1_refusal", "both_admit")
    first = {}
    for i, variant in enumerate(seq):
        arrays = getattr(fixtures_composition, variant)(seed=seed)
        out_c = np.asarray(run_od(cns(), arrays, family="acce"))
        if i in (0, 3):
            first.setdefault(variant, out_c.tobytes())
            assert out_c.tobytes() == first[variant], \
                f"{variant}: repeated A3-route call not byte-stable"
        out_b = np.asarray(run_od(b0ns(), arrays, family="acce"))
        assert out_c.tobytes() == out_b.tobytes(), \
            f"{variant}: interleaved call diverged from B0"
    record("interleaved_cells_clean", seed=seed, sequence=list(seq))
