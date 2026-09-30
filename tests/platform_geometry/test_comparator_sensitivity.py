"""Comparator sensitivity (mutation kill).

A parity comparator that cannot fail proves nothing.  Each deliberate
facade sabotage below MUST be detected as a digest difference against
the reference arm; each yields well-formed (non-crashing) wrong output
so the comparator itself is what catches it:
  discard_longest  — redundant-edge discard keeps the LONGEST edge
                     (INC06 semantics inverted)
  swap_od_colors   — origin/destination categorical colors swapped
  bump_edge_weight — first edge weight nudged by one ulp
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.platform_geometry

SABOTAGES = {
    "discard_longest": "redundant_discard",
    "swap_od_colors": "grid_basic",
    "bump_edge_weight": "grid_basic",
}


@pytest.mark.parametrize("sabotage,scenario", sorted(SABOTAGES.items()))
def test_sabotaged_facade_is_detected(scenario, sabotage, arm_runner):
    honest = arm_runner(scenario, "facade")
    broken = arm_runner(scenario, "facade", sabotage=sabotage)
    ref = arm_runner(scenario, "reference")
    assert honest == ref, "honest facade diverged (test premise broken)"
    assert broken != ref, (
        f"sabotage {sabotage!r} was NOT detected by the comparator; the "
        f"parity guarantee is vacuous for this dimension")
    assert broken != honest
