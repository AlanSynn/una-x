"""Bitwise facade-vs-reference parity over the una betweenness surface
(parallel_betweenness / one_betweenness_2 / betweenness_exposure /
paralell_betweenness_exposure + the tools.betweenness wrapper).

For every scenario the reference arm runs TWICE (determinism control),
then the facade; the facade digest must equal the reference digest
bitwise.  Sabotage variants prove the comparator actually selects the
mutated state (a deliberately broken candidate must fail).

Model discipline (dossier 01): closest-destination and Huff competition
are DISTINCT models (never aggregate substitution); decay exponent and
power are distinct; the exposure engine's betweenness is route-
attributed per edge (not an aggregate flow).
"""
from __future__ import annotations

import math
import struct

import pytest

pytestmark = pytest.mark.madina_api

from _flow_scenario import SCENARIO_NAMES  # unique module name: bare
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


def _float(v):
    return struct.unpack(">d", bytes.fromhex(v))[0]


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


# ----------------------------------------------------------------------
# completion condition (dossier): "Correct model rather than aggregate
# substitution; layer/edge/state mutation, route attribution and exact
# specified reductions pass."
# ----------------------------------------------------------------------

HUFF_FULL_STATS = {
    "reach", "gravity", "closest_destination_distance",
    "furthest_destination_distance", "mean_path_length",
    "probable_travel_distance", "eligible_destinations", "path_count",
    "path_segment_count", "path_segment_memory", "scope_node_count",
    "chunck_count", "memory_stalls"}


def test_models_distinct_and_stats_quirk(arm_runner):
    """closest_destination=True (default) and the Huff competition are
    bitwise-distinct models.  The pinned stats quirk: with
    closest_destination=True the stats block dies on the unbound
    eligible_destinations_shortest_distance (NameError swallowed by the
    bare except) — reach/gravity ARE written, everything after is not;
    with Huff the full stats set lands."""
    state = arm_runner("flow_closest_huff", "facade")["state"]
    assert state["models_distinguish"] is True
    dc = state["direct_closest"]["origin_gdf"]
    dh = state["direct_huff"]["origin_gdf"]
    assert set(dc["stats_columns_present"]) == {"reach", "gravity"}
    assert set(dh["stats_columns_present"]) == HUFF_FULL_STATS
    # exact specified reductions: every origin reaches both destinations
    assert all(_float(v) == 2.0 for v in dh["values"]["reach"])
    assert all(_float(v) == 2.0 for v in dc["values"]["reach"])
    assert all(_float(v) == 2.0 for v in dh["values"]["eligible_destinations"])
    # closest <= furthest, and closest distance is a real positive number
    for c, f in zip(dh["values"]["closest_destination_distance"],
                    dh["values"]["furthest_destination_distance"]):
        assert 0.0 < _float(c) <= _float(f)
    # route attribution: with closest=True each origin routes to ONE
    # destination; the Huff run spreads mass differently -> at least one
    # edge changes bits (already pinned by models_distinguish, the
    # per-edge record makes the attribution inspectable)
    assert len(state["direct_huff"]["betweenness_by_edge"]) == len(
        state["direct_closest"]["betweenness_by_edge"])
    # the seeded shuffle is load-bearing and recorded
    assert state["huff"]["shuffled_origin_order"] == \
        state["closest_true"]["shuffled_origin_order"]
    assert state["closest_true"]["progress_seen"] is True
    assert state["direct_closest"]["return_keys"] == ["edge_gdf",
                                                      "origin_gdf"]


def test_decay_matrix_and_destination_cap(arm_runner):
    """Decay/penalty/detour engagement on the EXPOSURE engine (whose
    wandering_messenger paths carry real edge sequences): exponent and
    power decay are distinct models, path-detour penalties are
    distinguishable per edge, the detour width engages, and the
    destniation_cap cut (typo param) is engaged by a digest difference."""
    state = arm_runner("flow_decay_matrix", "facade")["state"]
    assert state["exponent_differs_from_no_decay"] is True
    assert state["power_differs_from_exponent"] is True
    assert state["cap_differs_from_uncapped"] is True
    for key in ("no_decay", "exponent", "power", "destniation_cap_1"):
        assert state[key]["error"] is None
        og = state[key]["origin_gdf"]
        assert "reach" in og["values"]
        assert all(_float(v) == 2.0 for v in og["values"]["reach"])
    # route-attribution teeth: the exposure engine's per-edge digests
    # distinguish BOTH detour penalties and the detour width (correct
    # model, not aggregate substitution)
    assert state["exposure_penalty_exponent_differs"] is True
    assert state["exposure_penalty_power_differs"] is True
    assert state["exposure_detour_engages"] is True
    for key in ("penalty_equal_1_5", "penalty_exponent_1_5",
                "penalty_power_1_5", "detour_1_0"):
        assert state[key]["error"] is None
    # power decay is per-path 1/w^2 while exponent decay is uniform at
    # the shortest distance (pinned quirk) -> the two digests differ in
    # more than one edge (at least: an edge carrying multiple paths)
    diffs = sum(1 for k in state["exponent"]["betweenness_by_edge"]
                if state["exponent"]["betweenness_by_edge"][k]
                != state["power"]["betweenness_by_edge"][k])
    assert diffs >= 2


def test_lowlevel_engine_pins(arm_runner):
    """Network-level engine: the PINNED num_cores>1 defect (array_split
    shards the origins frame into ndarray rows -> every worker dies on
    ``origins.index`` -> the as_completed loop swallows it -> sum([])
    silently leaves ALL-ZERO betweenness), the PINNED attribution
    degeneracy (trimmed paths + street-node nearest_edge_id 0 collapse
    all mass onto edge 0, per-OD totals 2*prob_sum*weight — recorded,
    not hidden), the closest-vs-Huff model distinction, the
    retention round-trip reproducing bits with only retained_d_idxs
    populated, the tracker starting at INT 0, and the tiny-radius
    destination_count==0 continue."""
    state = arm_runner("flow_lowlevel_engine", "facade")["state"]
    # pinned upstream defect: nc=2 silently all-zero
    assert state["nc2_equals_nc1_bitwise"] is False
    assert state["nc2_all_zero"] is True
    assert state["nc2_silent_zero"]["error"] is None
    assert state["nc2_silent_zero"]["worker_error_markers"] == 2
    assert state["base_nc1"]["worker_error_markers"] == 0
    # the deterministic per-destination exception swallowing (o:9/d:10)
    assert state["base_nc1"]["swallowed_destination_errors"] >= 1

    # pinned attribution degeneracy: with all mass collapsing onto edge
    # 0, per-OD totals are 2*prob_sum*weight — exponent and power agree
    # bitwise at multiple paths, the penalty is moot exactly at single
    # path (detour 1.0), and the equal branch alone degrades through
    # float32 (python 1/len probability x float32 origin weight)
    assert state["penalty_moot_at_single_path"] is True
    assert state["penalty_exponent_equals_power_at_multi_path"] is True
    assert state["penalty_equal_bitwise_differs_from_exponent"] is True
    assert state["penalty_equal_degradation_magnitude"] < 1e-6
    # detour ratio ENGAGES in the low-level engine: it filters the path
    # set (upstream betweenness.py:297), so huff@1.5 differs from
    # huff@1.0 bitwise (caught-in-review fix: the earlier flag paired
    # 1.0 against 1.0 — vacuous)
    assert state["lowlevel_detour_engages"] is True
    assert state["huff_detour_1_5"]["betweenness_by_edge"] != \
        state["huff_destination_weights"]["betweenness_by_edge"]
    assert state["huff_differs"] is True
    assert state["base_nc1"]["error"] is None
    assert state["base_nc1"]["res_edge_is_network_edges"] is True
    # edge-0 collapse with the documented 2*prob_sum*weight totals
    nonzero = {k: _float(v)
               for k, v in state["base_nc1"]["betweenness_by_edge"].items()
               if _float(v) != 0.0}
    assert set(nonzero) == {"0"}
    # the exposure engine is where REAL route attribution engages (mass
    # on actual edge sequences; the low-level engine's detour/penalty
    # sensitivity never moves mass off edge 0)
    assert "attribution_note" in state

    ret = state["retention"]
    assert ret["first_call_error"] is None
    assert ret["retained_paths_empty"] is True
    assert ret["retained_distances_empty"] is True
    assert set(ret["first_call_return_keys"]) == {
        "edge_gdf", "retained_d_idxs", "retained_paths",
        "retained_distances"}
    assert len(ret["retained_d_idxs"]) == 3  # one entry per origin
    assert all(len(per_origin) == 2  # both destinations retained
               for per_origin in ret["retained_d_idxs"].values())
    assert ret["passthrough_error"] is None
    assert ret["passthrough_reproduces_bits"] is True

    one = state["one_betweenness_2"]
    assert one["error"] is None
    assert one["tracker_types_int_start"] is True
    assert one["tiny_radius_error"] is None
    assert one["tiny_radius_tracker_all_zero"] is True
    # in-process engine: nonzero tracker entries exist (all on edge 0,
    # per the pinned degeneracy)
    assert any(_float(v) > 0.0 for v in one["tracker"].values())


def test_elastic_knn_pins(arm_runner):
    """Elastic trip generation: string and list knn_weight forms agree
    bitwise, the plateau engages the decay, the save join lands the
    knn_weight column, and the pinned clobber resets network.knn_weight
    to None on the next non-elastic call."""
    state = arm_runner("flow_elastic_knn", "facade")["state"]
    assert state["str_form_equals_list_form"] is True
    assert state["plateau_engages"] is True
    assert state["save_elastic_without_elastic_error"].startswith(
        "ValueError")
    ew = state["knn_list_form"]["layers"]["origins"]
    assert "ew" in ew["columns"]
    # elastic weights are in (0, 1] per origin (knn scores)
    assert all(0.0 < _float(v) <= 1.0 for v in ew["values"]["ew"])
    assert state["knn_list_form"]["network_knn_weight"] == "[0.5, 0.25]"
    clob = state["clobber_after_elastic"]
    assert clob["network_knn_weight"] == "None"
    assert clob["network_knn_plateau"] == "0"
    assert state["clobber_equals_fresh_plain"] is True


def test_exposure_diagnostics_pins(arm_runner):
    """Path exposure: the four hazzard origin stats land (including the
    typo column decayed_mean_hazzad), the save join lands expected_
    hazzard_meters under the requested name, keep_diagnostics joins the
    diagnostic columns into the origin layer (wall-clock columns
    presence-pinned, values excluded from digests), and the
    keep_diagnostics-without-save TypeError fires only AFTER a full
    engine run."""
    state = arm_runner("flow_exposure_diagnostics", "facade")["state"]
    og = state["direct_exposure"]["origin_gdf"]
    assert {"mean_hazzard", "decayed_mean_hazzad",
            "expected_hazzard_meters",
            "probable_travel_distance_weighted_hazzard"} <= set(
        og["stats_columns_present"])
    assert og["values"]["decayed_mean_hazzad"] == \
        og["values"]["decayed_mean_hazzad"]  # structure only; bits via parity
    assert all(_float(v) == 2.0 for v in og["values"]["reach"])
    # exposure + closest_destination=True: hazzard stats precede the
    # stats-block NameError -> they ARE written, distances are not
    dc = state["direct_exposure_closest"]["origin_gdf"]
    assert {"reach", "gravity", "mean_hazzard", "decayed_mean_hazzad",
            "expected_hazzard_meters",
            "probable_travel_distance_weighted_hazzard"} <= set(
        dc["stats_columns_present"])
    assert "closest_destination_distance" not in dc["stats_columns_present"]

    diag = state["tools_diagnostics"]
    assert diag["error"] is None
    assert len(diag["renamed_time_columns_present"]) == 4
    origins = diag["layers"]["origins"]
    assert "expo" in origins["columns"]
    assert "bt_decayed_mean_hazzad" in origins["columns"]
    assert "bt_memory_stalls" in origins["columns"]
    assert all(_float(v) >= 0.0 for v in origins["values"]["expo"])
    # pinned quirk: with NO origin save requested, the expo join (and
    # the whole diagnostics join) is silently skipped — no column, no
    # error
    expo_only = state["expo_only_silently_ignored"]
    assert expo_only["error"] is None
    assert expo_only["expo_joined"] is False

    td = state["keep_diagnostics_without_save"]
    assert td["error"].startswith("TypeError")
    assert td["edges_got_betweenness_column"] is True
    assert td["origins_layer_untouched"] is True


def test_repeats_mutations_pins(arm_runner):
    """Dossier combos: repeated identical calls on one Zonal are
    idempotent through the drop+rejoin save path; changed weights change
    the betweenness bits; the clear/reinsert round-trip reproduces a
    fresh build; mixed origin source layers route the save join to the
    FIRST processed origin's layer (seeded shuffle decides which)."""
    state = arm_runner("flow_repeats_mutations", "facade")["state"]
    assert state["repeat_idempotent"] is True
    assert state["repeat_first"]["error"] is None
    assert state["repeat_second"]["error"] is None
    assert state["weight_change_engages"] is True
    assert state["clear_reinsert_reproduces"] is True
    # the weight edit lives on the NETWORK nodes (the engine's input),
    # not the layer gdf, which stays untouched
    assert state["weight_after"]["layers"]["destinations"] == \
        state["weight_before"]["layers"]["destinations"]
    mixed = state["mixed_source_layers"]
    assert mixed["layer_receiving_join"] in (["origins"], ["origins2"])
    assert mixed["layer_receiving_join"] == [
        mixed["first_processed_source_layer"]]
    other = "origins2" if mixed["layer_receiving_join"] == ["origins"] \
        else "origins"
    assert "reach" not in mixed["layers"][other]["columns"]
    # both origin layers contributed origins (4 origins processed)
    assert len(mixed["shuffled_origin_order"]) == 4


def test_edge_cases_pins(arm_runner):
    """No reachable destination (empty d_idxs continue; the tools save
    path KeyErrors on 'reach'), zero-weight origin skip, all-zero
    destination gravities (the continue that skips task_done), zero
    origins, and turn_penalty engagement."""
    state = arm_runner("flow_edge_cases", "facade")["state"]
    tiny = state["tiny_radius"]
    assert tiny["tools_save_reach_error"].startswith("KeyError")
    assert "'reach'" in tiny["tools_save_reach_error"]
    assert tiny["tools_edges_all_zero"] is True
    assert tiny["direct_error"] is None
    assert tiny["direct_betweenness_all_zero"] is True
    assert tiny["direct_origin_gdf"]["stats_columns_present"] == []

    zw = state["zero_weight_origin"]
    assert zw["error"] is None
    reach = zw["layers"]["origins"]["values"]["reach"]
    assert _float(reach[0]) == 0.0  # the zero-weight origin's NaN filled
    assert any(_float(v) == 2.0 for v in reach)

    zg = state["zero_destination_gravity"]
    assert zg["direct_error"] is None
    assert zg["direct_betweenness_all_zero"] is True
    assert zg["direct_origin_gdf"]["stats_columns_present"] == []
    assert zg["tools_save_reach_error"].startswith("KeyError")

    zo = state["zero_origins"]
    # a destination-only zonal: the tools wrapper fails EARLY at
    # validate_zonal_ready (no origin nodes), while the direct engine
    # dies later on the progress print's 0/0 (both pinned verbatim)
    assert zo["tools_error"].startswith("ValueError")
    assert "origin nodes" in zo["tools_error"]
    assert zo["direct_error"].startswith("ZeroDivisionError")

    assert state["turn_engages"] is True
    assert state["turn_penalty"]["error"] is None


def test_nc2_exposure_pins(arm_runner):
    """Exposure engine at num_cores=2: per-origin layer state is
    partition-independent and bitwise-equal across core counts; the edge
    betweenness is recorded structurally only — the shared-queue
    partition carries no bitwise-stability contract upstream
    (load-dependent by construction; empirically stable across three
    retained runs on this fixture — documented, not hidden)."""
    state = arm_runner("flow_nc2_exposure", "facade")["state"]
    assert state["layer_state_equal_across_cores"] is True
    assert state["nc1_progress_seen"] is True
    assert state["nc2_progress_seen"] is True
    assert state["nc2_edges_structural"]["all_finite"] is True
    assert state["nc2_edges_structural"]["nonzero_edges"] > 0
    assert state["nc1_edges_structural"]["nonzero_edges"] > 0


def test_validation_matrix_all_real_errors(arm_runner):
    """The full argument-validation matrix: every probe records an exact
    error type+message (no silent acceptance), the zonal is untouched,
    and the keep_diagnostics TypeError fires only after a full engine
    run (post-engine mutation pinned)."""
    state = arm_runner("flow_validation", "facade")["state"]
    errors = state["errors"]
    assert len(errors) == 27
    assert all(v != "NO_ERROR" for v in errors.values())
    assert sum(v.startswith("TypeError") for v in errors.values()) == 17
    assert sum(v.startswith("ValueError") for v in errors.values()) == 10
    # the duplicated save_gravity_as check uses the identical message
    # template (recorded strings differ only in the parameter name)
    assert errors["save_gravity_as_int"].replace(
        "'save_gravity_as'", "'save_reach_as'") == \
        errors["save_reach_as_int"]
    # the exposure attribute probe lists the legal columns
    assert "not in layer streets" in errors["path_exposure_attr_missing"]
    assert "hazzard" not in errors["path_exposure_attr_missing"]
    assert state["zonal_untouched_after_probes"] is True
    td = state["keep_diagnostics_typeerror_after_run"]
    assert td["error"].startswith("TypeError")
    assert td["edges_got_betweenness_column"] is True
    assert td["origins_layer_untouched"] is True


# ----------------------------------------------------------------------
# comparator sensitivity (EXECUTION.md: a deliberately broken candidate
# must fail).  decay_method_swap and detour_swap are the dossier's
# parameter-level mutants; split_edge_drop and exposure_column_drop are
# the dossier's digest-level mutants.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("sabotage,scenario", [
    ("decay_method_swap", "flow_decay_matrix"),
    ("detour_swap", "flow_lowlevel_engine"),
    ("split_edge_drop", "flow_closest_huff"),
    ("exposure_column_drop", "flow_exposure_diagnostics"),
])
def test_sabotaged_digests_are_selected(arm_runner, sabotage, scenario):
    honest = arm_runner(scenario, "reference")["state"]
    broken = arm_runner(scenario, "reference", sabotage=sabotage)["state"]
    assert broken != honest, f"sabotage {sabotage} was NOT selected"


def test_split_edge_drop_is_a_real_split(arm_runner):
    """The split_edge_drop mutant drops split_probe edge-table row 2,
    the first of the two rows carrying parent_street_id 2 (edge 3's
    parent merged into edge 2's, the pipeline-produced split-street
    shape) — and the dropped row's betweenness bits are genuinely gone
    from the digest.  Also pins the observable split-street behavior of
    tools.betweenness's per-parent join: the FIRST half keeps its
    betweenness on the streets layer, the second half's value (the
    unique 4.0) is silently discarded — street row 3 reads NaN — and
    the join index lands int64 only via the TODO cast."""
    honest = arm_runner("flow_closest_huff", "reference")["state"]
    broken = arm_runner("flow_closest_huff", "reference",
                        sabotage="split_edge_drop")["state"]
    probe = honest["split_probe"]
    assert probe["parents"] == [0, 1, 2, 2, 4, 5, 6, 7]
    assert probe["first_split_row"] == 2
    assert probe["first_split_shared_by"] == 2
    # the per-parent street join: first half wins, second half's value
    # (4.0, unique among edge betweenness) is nowhere on the streets
    # layer; the second half's street row reads NaN
    bte = probe["betweenness_by_edge"]
    assert _float(bte["3"]) != _float(bte["2"])  # halves differ: 4 vs 2
    assert probe["street_bt_index"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert probe["street_bt_index_dtype"] == "int64"
    assert _float(probe["street_bt_values"][2]) == _float(bte["2"])
    assert math.isnan(_float(probe["street_bt_values"][3]))
    assert _float(bte["3"]) not in [_float(v) for v in
                                    probe["street_bt_values"]]
    # the mutant: row 2 is dropped and its bits are gone
    meta = broken["split_probe"]["split_edge_dropped"]
    assert meta == {"row": 2, "parent_street_id": 2, "shared_by": 2}
    assert len(broken["split_probe"]["edges"]["index"]) == \
        len(probe["edges"]["index"]) - 1
    assert len(broken["split_probe"]["betweenness_by_edge"]) == \
        len(probe["betweenness_by_edge"]) - 1
    assert broken["split_probe"]["parents"] == [0, 1, 2, 4, 5, 6, 7]
    assert "2" not in broken["split_probe"]["betweenness_by_edge"]


def test_exposure_column_drop_removes_the_typo_column(arm_runner):
    """The exposure_column_drop mutant omits decayed_mean_hazzad (the
    upstream typo column) from the recorded origin-frame digest — the
    digest shrinks and the stats-presence ledger shrinks with it."""
    honest = arm_runner("flow_exposure_diagnostics", "reference")["state"]
    broken = arm_runner("flow_exposure_diagnostics", "reference",
                        sabotage="exposure_column_drop")["state"]
    h = honest["direct_exposure"]["origin_gdf"]
    b = broken["direct_exposure"]["origin_gdf"]
    assert "decayed_mean_hazzad" in h["columns"]
    assert "decayed_mean_hazzad" not in b["columns"]
    assert "decayed_mean_hazzad" in h["stats_columns_present"]
    assert "decayed_mean_hazzad" not in b["stats_columns_present"]
    assert len(b["columns"]) == len(h["columns"]) - 1
