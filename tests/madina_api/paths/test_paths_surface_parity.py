"""Bitwise facade-vs-reference parity over the una.paths surface.

For every scenario the reference arm runs TWICE (determinism control),
then the facade; the facade digest must equal the reference digest
bitwise.  Sabotage variants prove the comparator actually selects the
mutated state (a deliberately broken candidate must fail).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.madina_api

from _paths_scenario import SCENARIO_NAMES  # unique module name: bare
# `conftest` imports collide across non-package test dirs (zonal review)


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


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_facade_matches_bridged_reference_bitwise(scenario, arm_runner):
    ref_a = arm_runner(scenario, "reference")
    ref_b = arm_runner(scenario, "reference")
    assert ref_a["state"] == ref_b["state"], \
        _first_difference(ref_a["state"], ref_b["state"], "reference")
    facade = arm_runner(scenario, "facade")
    assert facade["state"] == ref_a["state"], \
        _first_difference(ref_a["state"], facade["state"],
                          f"{scenario} facade")


def test_all_paths_no_hidden_k_limit(arm_runner):
    """Completion condition: the engine's route set equals a full
    independent networkx enumeration of every simple path within the
    detour bound (interior-projected, dedupe semantics matched), per
    destination, with bitwise-equal per-route distances; and the public
    GeoDataFrame materializes exactly that route set (coherence)."""
    digest = arm_runner("alt_detour", "facade")["state"]
    assert digest["engine_equals_independent"] is True, \
        "engine route set differs from independent enumeration"
    assert digest["coherence"]["gdf_counts_equal_independent"] is True
    # multi-alternative reality: at least one destination carries >1
    # route (this is the ALL-paths surface, not a K-shortest view)
    assert any(rec["n_routes"] > 1
               for rec in digest["engine_per_destination"].values())
    shortest = arm_runner("alt_shortest", "facade")["state"]
    assert shortest["coherence"]["gdf_counts_equal_independent"] is True


@pytest.mark.parametrize("sabotage,scenario", [
    ("k_limit_paths", "alt_detour"),
    ("drop_one_path", "alt_detour"),
    ("perturb_distance_bit", "alt_shortest"),
    ("drop_geometry_component", "alt_shortest"),
    ("flip_turn_distance", "turn_params"),
    ("hide_validation_error", "validation"),
])
def test_sabotaged_digests_are_selected(arm_runner, sabotage, scenario):
    """Comparator sensitivity (EXECUTION.md: a deliberately broken
    candidate must fail): the mutated digest must differ from the
    honest reference digest."""
    honest = arm_runner(scenario, "reference")["state"]
    broken = arm_runner(scenario, "reference", sabotage=sabotage)["state"]
    assert broken != honest, f"sabotage {sabotage} was NOT selected"


def test_turn_penalty_changes_distances(arm_runner):
    """The turn-penalty scenario must show the penalty engaging
    (distances differ on/off) and the relaxed threshold re-pinned."""
    state = arm_runner("turn_params", "facade")["state"]
    assert state["gdf_penalty_on"] != state["gdf_penalty_off"]
    assert state["gdf_penalty_on_relaxed"] != state["gdf_penalty_on"]
    assert state["net_params_before"] == [45.0, 30.0]
    assert state["net_params_after"] == [91.0, 15.0]
    # direct turn primitives recorded real geometry angles
    assert state["turn_probes"], "no turn probe recorded"


def test_engine_primitives_all_populated(arm_runner):
    """Every exported engine primitive was exercised for real: the
    enumerators returned routes for at least one destination, the
    distance matrix is populated, and return_paths=False suppresses
    path recording.  bfs_paths_many_targets_iterative is the pinned
    exception: the upstream node-based variant returns EMPTY route
    lists on this fixture while its edge-based sibling enumerates
    fully — recorded honestly and pinned bitwise by the parity test;
    here we only assert the keys exist."""
    state = arm_runner("engine_primitives", "facade")["state"]
    assert state["path_generator_d_idxs_order"], "no destinations reached"
    assert state["distance_matrix"], "distance matrix empty"
    assert any(state["path_generator_edges"][k]
               for k in state["path_generator_edges"])
    assert any(state["bfs_path_edges"][k] for k in state["bfs_path_edges"])
    for k in state["bfs_paths_nodes"]:
        assert isinstance(state["bfs_paths_nodes"][k], list)
    assert any(state["wandering_edges"][k] for k in state["wandering_edges"])
    assert state["scope_paths_empty_when_return_paths_false"] is True
