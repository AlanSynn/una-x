"""C12 typed MetricID identity laws (campaign una_10x, dossier 05).

Every float enters by its IEEE-754 bits, the decay branch as a string,
the coefficient vector as an ordered value list, the SOURCE dtypes as
dtype strings, and the engine fold's bytes as compiler identity.
Provenance has no field: equal bits from different handles are the same
identity.
"""
import numpy as np
import pytest

from conftest import metric_job, mk_case
from urban_network_analysis.Engines import _metric_reuse_scratch as mrs

BASE = dict(d_weights=[1.0, 2.0, 3.0], gravity_beta=1.5,
            gravity_plateau=0.5, gravity_logistic_midpoint=2.0,
            gravity_growth_rate=0.75, knn_decay="exponential",
            knn_weights=[0.9, 0.8, 0.7], cutoff=6.0)


def ident(**over):
    kw = dict(BASE)
    kw.update(over)
    return mrs.metric_identity(**kw)


def test_equal_bits_different_handles_same_id():
    w = np.array(BASE["d_weights"])
    a = ident()
    b = ident(d_weights=np.asfortranarray(w))
    assert a == b
    c = ident(d_weights=w.copy())
    assert a == c


def test_one_ulp_on_every_scalar_moves_id():
    for key in ("gravity_beta", "gravity_plateau",
                "gravity_logistic_midpoint", "gravity_growth_rate",
                "cutoff"):
        bumped = float(np.nextafter(BASE[key], 4.0))
        assert bumped != BASE[key], key
        assert ident(**{key: bumped}) != ident(), key


def test_one_ulp_in_weight_vector_moves_id():
    w = np.array(BASE["d_weights"])
    w[1] = np.nextafter(w[1], 4.0)
    assert ident(d_weights=w) != ident()


def test_one_ulp_in_knn_vector_moves_id():
    k = np.array(BASE["knn_weights"])
    k[0] = np.nextafter(k[0], 4.0)
    assert ident(knn_weights=k) != ident()


def test_signed_zero_distinct():
    w = [0.0] + BASE["d_weights"][1:]
    wz = [-0.0] + BASE["d_weights"][1:]
    assert ident(d_weights=w) != ident(d_weights=wz)


def test_decay_branch_string_distinct():
    assert ident(knn_decay="exponential") != \
        ident(knn_decay="logistic")
    assert ident(knn_decay="logistic") != ident(knn_decay="linear")


def test_source_dtype_distinct():
    assert ident(d_weights=np.array(BASE["d_weights"],
                                    dtype=np.float32)) != ident()
    assert ident(knn_weights=np.array(BASE["knn_weights"],
                                      dtype=np.float32)) != ident()


def test_profile_id_distinct():
    assert ident(profile_id="legacy_default_profile") != \
        ident(profile_id="other_profile")


def test_vector_permutation_distinct():
    """The value list is ORDERED: a permutation is a different metric
    input."""
    w = BASE["d_weights"][::-1]
    assert ident(d_weights=w) != ident()


def test_engine_sha_field_is_live_and_fed():
    import urban_network_analysis.Engines.AccessibilityWElevation \
        as eng
    a = ident()
    other_file = eng.__file__.replace("AccessibilityWElevation",
                                      "_sparse_access_scratch")
    old = eng.__file__
    assert other_file != old
    try:
        eng.__file__ = other_file
        moved = ident()
    finally:
        eng.__file__ = old
    assert moved != a


def test_engine_sha_matches_preserved_table(engine_sha, table):
    """Second derivation: the live engine bytes hash to the value the
    C10 preserved identity table records."""
    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                yield from walk(x)
        elif isinstance(v, list):
            for x in v:
                yield from walk(x)
        elif isinstance(v, str):
            yield v
    assert engine_sha in set(walk(table))


def test_case_to_job_roundtrip():
    """The planner job projection carries exactly the metric inputs of
    a characterization case (no distance field)."""
    case = mk_case(n=8)
    assert "d_distance" not in metric_job(case)
    job = metric_job(case)
    assert set(job) == set(BASE)
