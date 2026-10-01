"""Bitwise facade-vs-reference parity over the extended Zonal surface.

For every scenario the reference arm runs TWICE (determinism control),
then the facade; the facade digest must equal the reference digest
bitwise.  Sabotage variants prove the comparator actually selects the
mutated state (a deliberately broken candidate must fail).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.madina_api

from conftest import SCENARIOS  # path set up by conftest

# M4: at least one street edge must carry MULTIPLE inserted nodes — the
# engine-consumption precondition the TOPOLOGY review asked to pin.
M4_SCENARIO = "insert_multinode_edge"


def _first_difference(a, b, path="root"):
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                return f"{path}.{k}: missing in a"
            if k not in b:
                return f"{path}.{k}: missing in b"
            found = _first_difference(a[k], b[k], f"{path}.{k}")
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = _first_difference(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_facade_matches_bridged_reference_bitwise(scenario, arm_runner):
    ref_a = arm_runner(scenario, "reference")
    ref_b = arm_runner(scenario, "reference")
    assert ref_a["state"] == ref_b["state"], \
        _first_difference(ref_a["state"], ref_b["state"], "reference")
    facade = arm_runner(scenario, "facade")
    assert facade["state"] == ref_a["state"], \
        _first_difference(ref_a["state"], facade["state"],
                          f"{scenario} facade")


def test_m4_multiple_inserted_nodes_on_one_edge(arm_runner):
    """TOPOLOGY review handoff M4: the scenario actually produces a
    multi-node-per-edge network (2 origins from the same edge + a
    destination elsewhere), and the digest records per-edge counts."""
    digest = arm_runner(M4_SCENARIO, "facade")["state"]
    per_edge = digest["inserted_per_edge"]
    assert any(count >= 2 for count in per_edge.values()), per_edge
    # the split-edge provenance columns the engines will consume exist
    assert "nearest_edge_id" in digest["nodes"]["values"]
    assert "parent_street_id" in digest["edges"]["values"]
    # add/remove roundtrips on a street node and on a real inserted
    # node: the exact post-roundtrip state is pinned bitwise by the
    # parity test itself (restoration is NOT asserted — review finding
    # 1: node counts alone masked real edge/weight corruption)
    for key in ("roundtrip_street_node", "roundtrip_inserted_node",
                "update_light_graph_after_add",
                "update_light_graph_after_remove"):
        assert key in digest, key


@pytest.mark.parametrize("sabotage,scenario", [
    ("drop_parent_col", "id_semantics"),
    ("drop_last_edge", M4_SCENARIO),
    ("swap_default_color", "style_colors"),
    ("drop_describe_line", "describe_output"),
    ("flip_turn_threshold", "turn_params"),
    ("undercount_layers", "map_deck"),
    ("hide_set_style_error", "quirks"),
])
def test_sabotaged_digests_are_selected(arm_runner, sabotage, scenario):
    """Comparator sensitivity (EXECUTION.md: a deliberately broken
    candidate must fail): the mutated digest must differ from the
    honest reference digest."""
    honest = arm_runner(scenario, "reference")["state"]
    broken = arm_runner(scenario, "reference", sabotage=sabotage)["state"]
    assert broken != honest, f"sabotage {sabotage} was NOT selected"
