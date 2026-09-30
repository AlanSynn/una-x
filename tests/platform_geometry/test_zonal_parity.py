"""Bitwise facade-vs-reference parity over the supported zonal surface.

Protocol per dossier 01: run the reference arm TWICE (baseline
determinism), then the facade; the facade digest must equal the
reference digest exactly (floats as bit patterns, geometries as WKB
hashes, dtypes/categories/order as recorded).
"""
from __future__ import annotations

import pytest

from conftest import SCENARIOS

pytestmark = pytest.mark.platform_geometry


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_facade_matches_bridged_reference_bitwise(scenario, arm_runner):
    ref_a = arm_runner(scenario, "reference")
    ref_b = arm_runner(scenario, "reference")
    assert ref_a == ref_b, (
        f"reference arm is not deterministic for {scenario}; the parity "
        f"baseline is invalid and must be investigated before comparing "
        f"the facade")
    facade = arm_runner(scenario, "facade")
    assert facade == ref_a, _first_difference(facade, ref_a)


def _first_difference(a, b, path="root"):
    """Human-targetable locator for the first digest divergence."""
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                return f"{path}.{key}: missing in facade"
            if key not in b:
                return f"{path}.{key}: missing in reference"
            sub = _first_difference(a[key], b[key], f"{path}.{key}")
            if sub:
                return sub
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return (f"{path}: length {len(a)} != {len(b)} "
                    f"(first mismatch at index {min(len(a), len(b))})")
        for i, (x, y) in enumerate(zip(a, b)):
            sub = _first_difference(x, y, f"{path}[{i}]")
            if sub:
                return sub
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"
