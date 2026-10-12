"""C12 poison law: the OUTPUT_DEPENDENCIES map is behavioral
(campaign una_10x, dossier 05).

For each decay branch and each poisonable parameter, every output whose
effective dependency list excludes the parameter must keep its bits
under a large poison, and at least one declared witness output must
move.  A cutoff change that flips no retention-filter membership moves
NOTHING (the map names the dependency membership_cutoff, not cutoff);
a membership-flipping cutoff moves every output.
"""
import numpy as np
import pytest

from conftest import bits, mk_case, run_all
from urban_network_analysis.Engines._metric_reuse_scratch import (
    _PARAM_TO_DEP, output_dependencies)

OUTPUTS = ("reach", "gravity_exponential", "gravity_logistic",
           "knn_access")

# poison -> {param override or weights mutation, witness output}
POISONS = {
    "gravity_beta": ({"gravity_beta": 2.5}, "gravity_exponential"),
    "gravity_plateau": ({"gravity_plateau": 2.0},
                        "gravity_exponential"),
    "gravity_growth_rate": ({"gravity_growth_rate": 1.5},
                            "gravity_logistic"),
    "gravity_logistic_midpoint": ({"gravity_logistic_midpoint": 3.0},
                                  "gravity_logistic"),
    "knn_weights": ("weights", "knn_access"),
}


def case_of(**over):
    return mk_case(n=32, seed=13, **over)


def weights_poison(case):
    case = dict(case)
    w = case["knn_weights"].copy()
    w[3] = w[3] * 2.0
    case["knn_weights"] = w
    return case


@pytest.mark.parametrize("decay", ["exponential", "logistic",
                                   "linear"])
@pytest.mark.parametrize("poison_key",
                         ["gravity_beta", "gravity_plateau",
                          "gravity_growth_rate",
                          "gravity_logistic_midpoint", "knn_weights"])
def test_declared_dead_outputs_never_move(combined, splits, decay,
                                          poison_key):
    over, witness = POISONS[poison_key]
    deps = output_dependencies(decay)
    dep = _PARAM_TO_DEP[poison_key]
    base = case_of(decay=decay)
    poisoned = weights_poison(base) if over == "weights" \
        else case_of(decay=decay, **over)
    b_outs, b_sp = run_all(base, combined, splits)
    p_outs, p_sp = run_all(poisoned, combined, splits)
    for out in OUTPUTS:
        if dep not in deps[out]:
            assert bits(p_sp[out]) == bits(b_sp[out]), \
                ("declared dead moved", out, decay, poison_key)
    assert bits(p_sp[witness]) != bits(b_sp[witness]), \
        ("witness did not move", witness, decay, poison_key)


def _membership_frozen_case():
    case = mk_case(n=32, seed=17)
    case["d_distance"] = np.linspace(0.5, 6.0, 32)
    return case


def test_cutoff_entering_only_via_membership(combined, splits):
    """Cutoff moved past the farthest retained distance: the retention
    filter keeps the same set, so every output is bit-identical."""
    base = _membership_frozen_case()
    wider = dict(base, cutoff=6.5)
    b_outs, b_sp = run_all(base, combined, splits)
    w_outs, w_sp = run_all(wider, combined, splits)
    for out in OUTPUTS:
        assert bits(w_sp[out]) == bits(b_sp[out]), out


def test_membership_flip_moves_everything(combined, splits):
    base = _membership_frozen_case()
    cut = dict(base, cutoff=3.0)
    b_outs, b_sp = run_all(base, combined, splits)
    c_outs, c_sp = run_all(cut, combined, splits)
    for out in OUTPUTS:
        assert bits(c_sp[out]) != bits(b_sp[out]), out


def test_kept_distance_moves_decay_outputs_not_reach(combined, splits):
    """A kept-distance byte change with membership fixed moves every
    DECAY output but never reach: reach reads kept weights only (fold
    bytes and the C10 plan_fields table agree), so kept_distances is
    not a reach dependency."""
    base = _membership_frozen_case()
    moved = dict(base)
    d = moved["d_distance"].copy()
    d[7] = d[7] + 0.25
    moved["d_distance"] = d
    b_outs, b_sp = run_all(base, combined, splits)
    m_outs, m_sp = run_all(moved, combined, splits)
    assert bits(m_sp["reach"]) == bits(b_sp["reach"])
    for out in ("gravity_exponential", "gravity_logistic",
                "knn_access"):
        assert bits(m_sp[out]) != bits(b_sp[out]), out
