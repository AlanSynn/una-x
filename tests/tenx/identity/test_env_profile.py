"""C10: numerical environment profile and the invalidation law.

The table's numerical_env must fingerprint the LIVE interpreter
(numba/numpy versions), the ACTUAL engine bytes (sha256 recomputed
here straight from the file, independently of the fixture) and the
fold's real decorators.  Every env field is identity-bearing:
perturbing any one flips the env fingerprint (profile invalidation).
"""
import hashlib
import re

import numba
import numpy
import pytest

import _canonical as canon
import _engine_facts as ef


def env_identity(env):
    """Canonical identity over the env profile's fields."""
    return canon.identity([("env_" + k, v)
                           for k, v in sorted(env.items())])


def perturb_value(v):
    if isinstance(v, list):
        assert v
        return [x + "!" for x in v]
    if isinstance(v, str):
        return v + "!"
    raise TypeError(type(v))


def test_numerical_env_matches_live_interpreter(table):
    env = table["numerical_env"]
    assert env["numba_version"] == numba.__version__
    assert env["numpy_version"] == numpy.__version__


def test_engine_sha256_recomputed_from_bytes(table):
    env = table["numerical_env"]
    h = hashlib.sha256()
    with open(ef.ENGINE, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert env["engine_sha256"] == h.hexdigest()
    assert env["engine_sha256"] == table["fold"]["engine_sha256"]


def test_decorators_are_the_real_compiled_flags(facts, table):
    env = table["numerical_env"]
    assert env["fold_decorators"] == facts["decorators"]
    # the fold is actually njit-compiled, through env-driven flag names
    joined = " ".join(facts["decorators"])
    assert "njit" in joined
    assert re.search(r"NUMBA_[A-Z_]+", joined)
    for kw in ("parallel", "cache", "nogil", "fastmath"):
        assert kw + "=" in joined


def test_env_fingerprint_is_exhaustive(table):
    """Perturbing ANY env field flips the env fingerprint."""
    env = table["numerical_env"]
    assert len(env) >= 4
    base = env_identity(env)
    for k, v in env.items():
        pert = dict(env)
        pert[k] = perturb_value(v)
        assert pert[k] != v
        assert env_identity(pert) != base, k


def test_profile_invalidation_law(table, contract):
    """profile_id gates the whole table and enters GraphID identity."""
    assert table["profile_id"]
    gfields = [f for t, fs in contract["id_fields"].items()
               for f in fs if f == "profile_id"]
    assert len(gfields) == 1  # exactly the GraphID carrier
    assert "profile_id" in table["profile_invalidation"]
    assert "invalidates" in table["profile_invalidation"]


def test_numerical_env_is_a_graph_identity_field(contract):
    assert "numerical_env" in contract["id_fields"]["GraphID"]


def test_schema_and_task_agree(table, contract):
    assert table["schema_version"] == contract["schema_version"]
    assert table["task"] == contract["task"] == "C10"


def test_engine_file_present_and_named(facts):
    assert facts["engine_file"] == ef.ENGINE_REL
    assert ef.ENGINE.is_file()
