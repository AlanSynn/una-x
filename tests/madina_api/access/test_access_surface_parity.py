"""Bitwise facade-vs-reference parity over the una accessibility
surface (validate_zonal_ready / accessibility / service_area + the
access-engine trio).

For every scenario the reference arm runs TWICE (determinism control),
then the facade; the facade digest must equal the reference digest
bitwise.  Sabotage variants prove the comparator actually selects the
mutated state (a deliberately broken candidate must fail).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.madina_api

from _access_scenario import SCENARIO_NAMES  # unique module name: bare
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


def _float(v):
    import struct
    return struct.unpack(">d", bytes.fromhex(v))[0]


def test_completion_condition_closest_facility_direction(arm_runner):
    """Completion condition (dossier): closest-facility direction and
    exact numeric fields.  d1's unique closest origin is o3 (source 2,
    100 m) regardless of processing order; the d2 three-way 200 m tie
    is won by the FIRST origin in the seeded shuffle; the unreachable
    iso destination keeps NaN (NA not filled in the destination layer);
    origin reach/gravity are overwritten by the joint-destination sums
    (0 for origins without joint destinations)."""
    state = arm_runner("closest_facility", "facade")["state"]
    cf = state["closest_facility_by_destination_source_id"]
    # direction: d1 (source 0) -> o3 (source 2) at exactly 100
    assert cf["0"][0] == 2 and _float(cf["0"][1]) == 100.0
    # tie: d2 (source 1) -> first processed origin at exactly 200
    first_processed = state["shuffled_origin_order"][0]
    first_source = {9: 0, 10: 1, 11: 2}[first_processed]
    assert cf["1"][0] == first_source and _float(cf["1"][1]) == 200.0
    # unreachable: iso destination keeps NaN in node gdf AND layer
    assert cf["2"] == [None, "7ff8000000000000"]
    dest_layer = state["layers"]["destinations"]
    assert _float(dest_layer["values"]["cfd"][-1]) != _float(
        dest_layer["values"]["cfd"][-1])  # NaN survives the join
    # joint-destination recomputation: source 2 wins BOTH destinations
    # (d1 by strict distance, d2 as first processed) -> its joint reach
    # is the summed destination weights 2+3, the others drop to 0
    reach = state["node"]["values"]["reach"][-6:-3]  # the three origins
    assert sorted(_float(v) for v in reach) == [0.0, 0.0, 5.0]


def test_completion_condition_named_joins_and_return_types(arm_runner):
    """Completion condition (dossier): named joins land under the
    requested names with an int layer index, and service_area returns
    the (destinations, network_edges, scope_gdf) triple with real
    content; the turn_penalty variant engages; below-radius and silent
    int-miss calls return empty structures."""
    state = arm_runner("layer_mutations", "facade")["state"]
    final = state["after_second_name_save"]["origins"]
    assert final["columns"][-3:] == ["reach", "grav", "reach2"]
    assert state["origin_layer_index_int"] == "0"  # int-typed index

    sa = arm_runner("service_areas", "facade")["state"]
    for key in ("single_int_id", "all_origins", "list_ids",
                "turn_penalty_variant"):
        rec = sa[key]
        assert set(rec) == {"destinations", "network_edges", "scope_gdf"}
        assert len(rec["scope_gdf"]["index"]) > 0
    assert sa["turn_engages"] is True
    # scope rows: border+origin per origin, single vs all
    assert len(sa["single_int_id"]["scope_gdf"]["index"]) == 2
    assert len(sa["all_origins"]["scope_gdf"]["index"]) == 6
    # degenerate returns are empty but well-formed (both arms pin them)
    for key in ("below_radius_skip", "int_miss_silent"):
        assert len(sa[key]["scope_gdf"]["index"]) == 0
        assert len(sa[key]["destinations"]["index"]) == 0
    assert "ValueError" in sa["bad_list_error"]["error"]
    # the paths-review N3 strengthening: list-origin alternative_paths
    # digest recorded (multi-origin zonals raise the pinned ValueError)
    n3 = sa["alt_paths_list_origin"]
    assert n3["gdf"] is None and "ValueError" in n3["error"]


def test_completion_condition_exact_numeric_fields(arm_runner):
    """Completion condition (dossier): exact numeric fields — reach 5.0
    weightless, gravity 13.0 under alpha=2/beta=0 with named weights
    [2,3], NaN destination weights hitting fillna(0) (reach 2.0), knn
    string/list form equality and plateau decay engagement."""
    rg = arm_runner("reach_gravity", "facade")["state"]
    reach = rg["reach"]["node"]["values"]["reach"][-5:-2]
    assert all(_float(v) == 5.0 for v in reach)
    grav = rg["gravity"]["node"]["values"]["gravity"][-5:-2]
    assert all(_float(v) == 13.0 for v in grav)
    nan_reach = rg["nan_destination_weight"]["node"]["values"]["reach"][-5:-2]
    assert all(_float(v) == 2.0 for v in nan_reach)
    # review M4: accessibility itself positively exercised with
    # turn_penalty=True (the penalty must change the recorded state)
    assert rg["reach_turn_penalty"]["error"] is None
    assert rg["reach_turn_penalty"]["turn_engages"] is True
    assert rg["reach_turn_penalty"]["progress_lines"] == 3

    knn = arm_runner("knn", "facade")["state"]
    assert knn["str_form_equals_list_form"] is True
    assert knn["plateau_engages_decay"] is True
    assert knn["knn_list_form"]["layers"]["origins"]["columns"][-1] == "knn"


def test_engine_primitives_pins(arm_runner):
    """Engine trio pins: the pristine-network AttributeError, the
    exception-swallowing path, and the parallel consolidation failures
    (num_cores=2 needs beta, and cannot combine with closest_facility)
    — all recorded, all reproduced bitwise by the facade."""
    state = arm_runner("engine_primitives", "facade")["state"]
    assert "AttributeError" in state["gop_reach_gravity"][
        "pristine_network_error"]
    assert "knn_access" in state["gop_knn"]["origin_row"]["values"]
    one = state["one_access_stub_queue"]
    assert one["error"] is None and one["progress_lines"] == 3
    assert one["processed_origins"] == one["queue_order"]
    sw = state["one_access_exception_swallowed"]
    assert sw["returned_error"] is None
    assert sw["notice_in_stdout"] is True and sw["traceback_in_stderr"] is True
    assert len(sw["processed_before_failure"]) == 2
    nc1 = state["parallel_access_num_cores_1"]
    assert nc1["error"] is None and nc1["progress_lines"] == 3
    assert "KeyError: 'gravity'" in state[
        "parallel_access_num_cores_2_beta_none"]["error"]
    assert state["parallel_access_num_cores_2_beta_set"]["error"] is None
    assert "KeyError: 'reach'" in state[
        "parallel_access_num_cores_2_closest_facility"]["error"]


def test_validation_matrix_all_real_errors(arm_runner):
    """The full argument-validation matrix: every probe records an exact
    error type+message (no silent acceptance), including the pinned
    structural quirks (accessibility reads iloc[0] before
    validate_zonal_ready -> IndexError; empty Zonal fails earlier on
    network=None -> AttributeError; service_area validates first)."""
    state = arm_runner("validation", "facade")["state"]
    assert len(state) >= 30
    assert all(v != "NO_ERROR" for v in state.values())
    assert sum(v.startswith("TypeError") for v in state.values()) == 15
    assert sum(v.startswith("ValueError") for v in state.values()) == 13
    assert sum(v.startswith("AttributeError") for v in state.values()) == 2
    assert sum(v.startswith("IndexError") for v in state.values()) == 2
    # accessibility quirk order: no-destination rows fail with IndexError
    # (iloc[0] read) BEFORE validate_zonal_ready would
    assert state["no_origin_rows"].startswith("IndexError")
    assert state["no_destination_rows"].startswith("IndexError")
    # closest_facility=True demands both save names as strings
    assert "save_closest_facility_as" in state[
        "closest_facility_true_without_saves"]
    assert "save_closest_facility_distance_as" in state[
        "closest_facility_true_missing_distance_save"]


@pytest.mark.parametrize("sabotage,scenario", [
    ("alpha_swap", "reach_gravity"),
    ("perturb_gravity_bit", "reach_gravity"),
    ("skip_shuffle", "closest_facility"),
    ("reverse_closest_facility", "closest_facility"),
    ("fillna_closest_distance", "closest_facility"),
    ("hide_validation_error", "validation"),
])
def test_sabotaged_digests_are_selected(arm_runner, sabotage, scenario):
    """Comparator sensitivity (EXECUTION.md: a deliberately broken
    candidate must fail): the mutated run must differ from the honest
    reference digest.  alpha_swap (dossier mutant) is parameter-level;
    skip_shuffle monkeypatches sample(frac=1) to identity in the arm
    process (behavioral); reverse_closest_facility (dossier mutant) and
    the rest are digest-level controls."""
    honest = arm_runner(scenario, "reference")["state"]
    broken = arm_runner(scenario, "reference", sabotage=sabotage)["state"]
    assert broken != honest, f"sabotage {sabotage} was NOT selected"


def test_skip_shuffle_flips_tie_winner(arm_runner):
    """The unseeded sample(frac=1) is load-bearing: removing it flips
    the d2 tie winner (seeded order [11, 10, 9] vs index order).  This
    is the RNG-control evidence: the shuffle is exercised, not
    removed."""
    honest = arm_runner("closest_facility", "reference")["state"]
    broken = arm_runner("closest_facility", "reference",
                        sabotage="skip_shuffle")["state"]
    assert honest["closest_facility_by_destination_source_id"]["1"] != \
        broken["closest_facility_by_destination_source_id"]["1"]
    assert honest["shuffled_origin_order"] == [11, 10, 9]
    assert broken["shuffled_origin_order"] == [9, 10, 11]


def test_node_builder_truncation_pinned(arm_runner):
    """New pinned upstream defect (bounded reproduction): on the broken
    fixture the lexicographic-maximum point C=(100,100) occurs three
    times and the vectorized node builder's truncated final range
    leaves B-C's `end` labeled 0 (F) and drops the A-C diagonal as a
    false redundant edge; o3 (mid B-C) then has an EMPTY accessibility
    scope.  Both arms must reproduce every byte — the facade is a
    faithful copy, bugs included."""
    state = arm_runner("node_builder_truncation", "facade")["state"]
    e2 = state["broken_edge_table"]["2"]
    assert (e2["start"], e2["end"], e2["parent_street_id"]) == (4, 0, 2)
    assert _float(e2["length"]) == 100.0
    assert "5" not in state["broken_edge_table"], "diagonal present?!"
    assert len(state["broken_edge_table"]) == 6
    assert state["o3_scope_empty"] is True
    assert state["diagonal_dropped_in_broken"] is True
    # the clean fixture's edge table is correct (contrast control)
    c2 = state["clean_edge_table"]["2"]
    assert (c2["start"], c2["end"]) == (4, 5)
    assert _float(c2["length"]) == 100.0
    assert len(state["clean_edge_table"]) == 8
    # o1 (unaffected end of the grid) still reaches both destinations
    assert all(_float(v) == 200.0 for v in state["o1_scope"].values())
