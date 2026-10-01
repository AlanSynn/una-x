"""MADINA_WORKFLOWS bitwise parity suite: the pairing-workflow surface
(Logger / betweenness_flow_simulation / KNN_accessibility) through the
compat facade vs the pinned upstream reference — SAME interpreter, same
file fixtures, IEEE-754-strict digests.

Structure (mirrors tests/madina_api/flow):
  - one test per scenario: clean digests from both arms must be
    byte-identical modulo the arm tags (documented presence-only
    policies: pydeck HTML bytes, wall-clock columns);
  - sabotage tests: source-level mutants patched into EACH arm's OWN
    workflows.py must (a) be SELECTED (mutant facade != clean facade)
    and (b) be INTEGRITY-checked (mutant facade == mutant reference),
    so a mutant proves comparator sensitivity, not arm noise;
  - completion-condition tests: CSV defaults correct by profile,
    per-row input changes correct by profile, and every expanded API
    row covered by tests or a documented unsupported stub.

The pinned upstream facts exercised here are documented in the facade
header (src/urban_network_analysis/compat/madina/una/workflows.py) and
the scenario module docstring.
"""
from __future__ import annotations

import json

import pytest

from _workflows_scenario import (  # noqa: F401  (unique module name —
    # bare `conftest` imports collide across non-package test dirs,
    # zonal review)
    SCENARIO_NAMES,
    SABOTAGE_SCENARIOS,
)

pytestmark = pytest.mark.madina_api

# scenario digest cache: each (arm, scenario, sabotage) executes once
# per session — the workflow runs are the expensive part
_DIGESTS = {}


def _digest(arm_runner, scenario, arm, sabotage=None):
    key = (scenario, arm, sabotage)
    if key not in _DIGESTS:
        _DIGESTS[key] = arm_runner(scenario, arm, sabotage=sabotage)
    return _DIGESTS[key]


def _strip_arm_tags(digest: dict) -> dict:
    """Digest minus the arm/scenario/sabotage metadata (the scenario
    script records which file each arm loaded; the file NAME is env
    layout, not behavior)."""
    d = json.loads(json.dumps(digest))
    d.pop("arm", None)
    d.pop("scenario", None)
    d.pop("sabotage", None)
    d.get("env_marker", {}).pop("workflows_file", None)
    return d


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


# ----------------------------------------------------------------------
# reflection (surface pin)
# ----------------------------------------------------------------------

def test_reflection_surface_identical(reflection_runner):
    """Module surface (exports, workflow signatures, Logger method
    signatures) captured from each arm's own module must be identical
    (modulo the arm tag)."""
    ref = reflection_runner("reference")
    fac = reflection_runner("facade")
    ref.pop("arm")
    fac.pop("arm")
    assert ref == fac, _first_difference(ref, fac, "reflection")
    assert "Logger" in ref["classes"]
    assert set(ref["public_functions"]) >= {
        "betweenness_flow_simulation", "KNN_accessibility"}


# ----------------------------------------------------------------------
# clean bitwise parity, one test per scenario
# ----------------------------------------------------------------------

@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_scenario_bitwise_parity(arm_runner, scenario):
    """Reference determinism control (reference executed TWICE — the
    second run deliberately bypasses the _DIGESTS cache so it is a real
    re-execution, review catch F1) then facade == reference — same
    discipline as the flow suite."""
    ref_a = _strip_arm_tags(_digest(arm_runner, scenario, "reference"))
    ref_b = _strip_arm_tags(arm_runner(scenario, "reference"))
    assert ref_a == ref_b, \
        _first_difference(ref_a, ref_b, f"{scenario} reference x2")
    fac = _strip_arm_tags(_digest(arm_runner, scenario, "facade"))
    assert fac == ref_a, \
        _first_difference(ref_a, fac, f"{scenario} facade")


# ----------------------------------------------------------------------
# completion condition: CSV defaults and per-row input changes correct
# by profile
# ----------------------------------------------------------------------

def test_csv_defaults_correct_by_profile(arm_runner):
    """Flow profile: save_flow_csv/save_origin_csv default False ->
    those files are ABSENT while the geoJSONs and maps are present.
    KNN profile: origin_record.csv IS a default output.  Both profiles
    pin their defaults independently."""
    d = _strip_arm_tags(_digest(arm_runner, "wf_flow_defaults", "facade"))
    assert d["record_csv_absent_by_default"] is True
    assert d["origin_csv_absent_by_default"] is True
    assert d["origin_geojson_present"] is True
    assert d["flow_maps_present"] == ["flow_map_dark.html",
                                      "flow_map_light.html"]
    k = _strip_arm_tags(_digest(arm_runner, "wf_knn_defaults", "facade"))
    assert "origin_record.csv" in k["files"]
    assert "origin_record.geoJSON" in k["files"]
    assert k["totals_present"] == ["normalized_knn_access",
                                   "total_knn_access"]


def test_per_row_changes_correct_by_profile(arm_runner):
    """Second-row input changes must change the profile, never be
    silently ignored.  Flow profile: same-cost rows flush nodes and
    accumulate a SECOND flow column; a cost change rebuilds the
    network (events: one streets load, TWO topology builds, origin/
    destination layers reused — loaded only with row 1).  KNN profile:
    a Network_File change CRASHES (pinned upstream defect — see
    test_knn_network_file_change_pinned_crash)."""
    two = _strip_arm_tags(
        _digest(arm_runner, "wf_flow_two_pairings", "facade"))
    assert two["both_flow_columns"] == ["flow_a", "flow_b"]
    cost = _strip_arm_tags(_digest(arm_runner, "wf_flow_cost_change",
                                   "facade"))
    assert cost["both_flow_columns"] == ["geo_leg", "len_leg"]
    events = cost["files"]["time_log.csv"]["events"]
    loaded = [e for _, e in events
              if e and e.startswith("network FIle Loaded")]
    built = [e for _, e in events if e == "network topology created"]
    origin_loaded = [e for _, e in events if e and "origins file" in e]
    assert len(loaded) == 1, loaded
    assert len(built) == 2, built
    assert len(origin_loaded) == 1, origin_loaded


def test_knn_network_file_change_pinned_crash(arm_runner):
    """PINNED UPSTREAM DEFECT: the KNN workflow's Network_File-change
    branch calls load_layer on the already-loaded 'streets' label and
    dies with KeyError('Layer with label streets is already in Zonal
    object') BEFORE any row-2 work.  Row 1's per-pairing
    origin_record.csv exists; every final output (totals, geoJSON,
    time_log) is unreachable.  Both arms reproduce it identically (the
    clean-parity test proves that); this test pins the behavior."""
    d = _strip_arm_tags(_digest(arm_runner, "wf_knn_changes", "facade"))
    assert d["workflow_error"] == (
        "KeyError: 'Layer with label streets is already in Zonal object'")
    assert sorted(d["files"]) == ["origin_record.csv"]


def test_knn_guard_and_valueerror_matrix(arm_runner):
    """Pinned verbatim validation behavior at workflow level: the two
    workflows have DIFFERENT city_name errors; explicit folders are
    rejected by KNN without city_name; an empty KNN_Weight CSV cell
    raises the accessibility list message; an empty pairings table
    raises the pandas EmptyDataError; the flow workflow accepts
    explicit folders without city_name."""
    errs = _strip_arm_tags(_digest(arm_runner, "wf_flow_errors",
                                   "facade"))["errors"]
    assert errs["noargs"] == (
        "ValueError: parameter 'city_name' needs to be specified if "
        "`data_folder` and `output_folder` are not provided, or provide "
        "paths to the `data_folder` and `output_folder`")
    knn = _strip_arm_tags(_digest(arm_runner, "wf_knn_defaults", "facade"))
    guard = knn["guard_probes"]
    assert guard["city_name_required"] == (
        "ValueError: parameter 'city_name' needs to be specified")
    assert guard["nan_knn_weight"] == (
        "ValueError: knn_weight should be a list of numerical values "
        "like [0.5, 0.25, 0.25]")
    assert errs["empty_pairings"].startswith("EmptyDataError:")
    assert errs["missing_data_folder"].startswith("FileNotFoundError:")
    assert errs["custom_pairings_file"] is None


def test_closest_destination_stats_defect_pinned(arm_runner):
    """PINNED UPSTREAM DEFECT at workflow level: with the default
    Closest_destination=True the exposure engine's per-origin stats
    block raises UnboundLocalError (betweenness.py:858-859 reads
    eligible_destinations_shortest_distance, assigned ONLY in the Huff
    branch at :612) AFTER reach/gravity — the handler swallows it, the
    workflow 'succeeds', and the origin record carries reach/gravity
    but none of the crashed stats columns."""
    d = _strip_arm_tags(_digest(arm_runner, "wf_flow_defaults", "facade"))
    origin_name = ("huff_flow_O(origins)_D(destinations)/"
                   "origin_record_(origins).geoJSON")
    frame = d["files"][origin_name]["frame"]
    assert d["workflow_error"] is None
    assert [c for c in frame["columns"]
            if c.startswith(("reach_", "gravity_"))] != []
    crashed = [f"huff_flow_{name}" for name in (
        "closest_destination_distance", "furthest_destination_distance",
        "mean_path_length", "probable_travel_distance",
        "eligible_destinations", "path_count")]
    assert [c for c in frame["columns"] if c in crashed] == []


def test_huff_profile_has_healthy_stats(arm_runner):
    """The same stats columns the closest-destination profile loses are
    PRESENT on the Huff profile (closest_destination=False) — the
    defect is branch-specific, and the suite would catch a regression
    in either direction."""
    d = _strip_arm_tags(_digest(arm_runner, "wf_flow_huff_elastic",
                                "facade"))
    assert d["workflow_error"] is None
    assert d["elastic_weight_column_present"] is True
    assert len(d["stats_columns_present"]) == 4
    assert d["reach_gravity_present"] == ["gravity_elastic_huff",
                                          "reach_elastic_huff"]


def test_knn_zero_reach_pinned_crash(arm_runner):
    """PINNED UPSTREAM DEFECT: a pairing with no reachable destination
    (Radius=10) crashes the KNN workflow with KeyError: 'reach' BEFORE
    any output file is written (the empty-output dossier combination
    manifests as this exception)."""
    d = _strip_arm_tags(_digest(arm_runner, "wf_knn_zero_reach", "facade"))
    assert d["workflow_error"] == "KeyError: 'reach'"
    assert d["files"] == {}


def test_alpha_profile_dependence_pinned(arm_runner):
    """PINNED: with Destination_Weight='Count', insert_node receives
    weight_attribute=None -> unit destination NODE weights -> the
    hardcoded gravity alpha is BITWISE-INERT (1**a == 1**1); the
    alpha=999 sensitivity clone must be bitwise-equal over the
    deterministic files.  The alpha ENGAGES on the 'weight' profile
    (wf_knn_weights) — selection proven by the knn_alpha_2 sabotage."""
    d = _strip_arm_tags(_digest(arm_runner, "wf_knn_defaults", "facade"))
    assert d["alpha_sensitivity_error"] is None
    assert d["alpha_999_deterministic_outputs_equal"] is True


def test_nc3_structural_policy_recorded(arm_runner):
    """The default num_cores=8 flow workflow runs the exposure engine at
    num_cores=3: no bitwise contract (queue-partition accumulation),
    so the streets record is digested STRUCTURALLY while the origin
    record is bitwise — both recorded, neither hidden."""
    d = _strip_arm_tags(_digest(arm_runner, "wf_flow_nc3_structural",
                                "facade"))
    assert d["workflow_error"] is None
    assert d["record_structural"]["columns"] == [
        "id", "length", "geometry", "weight", "huff_flow"]
    assert d["record_structural"]["row_count"] == 8


def test_exposure_row_wiring(arm_runner):
    """Exposure_Attribute row: the edge-record exposure column list is
    empty (the exposure lands on the ORIGIN record — the per-parent
    street join is betweenness-only) while the origin record carries
    the hazzard stats INCLUDING the pinned 'decayed_mean_hazzad' typo
    column; save_path_exposure_as's output is present despite the
    reach/gravity gating being satisfied by the workflow's unconditional
    save names."""
    d = _strip_arm_tags(_digest(arm_runner, "wf_flow_exposure_row",
                                "facade"))
    assert d["workflow_error"] is None
    assert d["exposure_column_on_record"] == []
    assert "exposure_expo" in d["origin_hazzard_columns"]
    assert "expo_decayed_mean_hazzad" in d["origin_hazzard_columns"]


# ----------------------------------------------------------------------
# API-row coverage manifest (completion condition)
# ----------------------------------------------------------------------

def test_every_api_row_covered():
    """Every public symbol of the workflows module (and every Logger
    method) has scenario/test coverage or a documented unsupported
    stub.  The module is fully implemented (no unsupported stubs); the
    manifest names the rows that exercise each symbol."""
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    # the verbatim upstream module-top `os.environ['USE_PYGEOS'] = '0'`
    # (upstream line 6) fires on this in-process import; restore the
    # driver process's environment afterwards or every LATER suite that
    # probes env hygiene in a subprocess inherits the variable and
    # fails (observed against platform_geometry's D3 probe in the
    # whole-repo regression — the probe itself is correct)
    had_pygeos = os.environ.pop("USE_PYGEOS", None)
    try:
        import urban_network_analysis.compat.madina.una.workflows as wf_mod
    finally:
        if had_pygeos is not None:
            os.environ["USE_PYGEOS"] = had_pygeos
        else:
            os.environ.pop("USE_PYGEOS", None)
    manifest = {
        "Logger": ["reflection", "wf_flow_defaults (pairing_end outputs)",
                   "time_log events digested in every flow scenario"],
        "Logger.__init__": ["every flow/knn scenario (start log rows)"],
        "Logger.log": ["per-row changes (event sequences)",
                       "test_per_row_changes_correct_by_profile"],
        "Logger.pairing_end": ["wf_flow_defaults (files + CSV defaults)",
                               "wf_flow_two_pairings", "wf_flow_cost_change"],
        "Logger.simulation_end": ["wf_flow_defaults (root record + log)"],
        "Logger.flow_map_template_1": [
            "wf_flow_defaults (maps presence-pinned; pydeck bytes "
            "nondeterministic — documented policy)",
            "facade-env test (actionable ImportError without pydeck)"],
        "betweenness_flow_simulation": [
            "wf_flow_defaults", "wf_flow_huff_elastic",
            "wf_flow_two_pairings", "wf_flow_cost_change",
            "wf_flow_exposure_row", "wf_flow_nc3_structural",
            "wf_flow_errors"],
        "KNN_accessibility": [
            "wf_knn_defaults", "wf_knn_weights", "wf_knn_changes",
            "wf_knn_zero_reach", "guard probes in wf_knn_defaults"],
    }
    # keep the manifest honest against the real surface
    public = sorted(
        n for n in vars(wf_mod)
        if not n.startswith("_") and (
            callable(getattr(wf_mod, n))
            or isinstance(getattr(wf_mod, n), type))
        and getattr(wf_mod, n).__module__ == wf_mod.__name__)
    assert public == ["KNN_accessibility", "Logger",
                      "betweenness_flow_simulation"]
    logger_methods = sorted(
        n for n in vars(wf_mod.Logger)
        if not n.startswith("__")
        and callable(vars(wf_mod.Logger)[n]))
    assert logger_methods == ["flow_map_template_1", "log", "pairing_end",
                              "simulation_end"]
    for symbol in logger_methods:
        assert f"Logger.{symbol}" in manifest, (
            f"Logger.{symbol} missing from coverage manifest")
    for symbol, rows in manifest.items():
        assert rows, f"{symbol} has no coverage row"


# ----------------------------------------------------------------------
# sabotages: selection (mutant != clean) + integrity (mutant arms equal)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("mutant,scenario",
                         sorted((m, s) for m, s in SABOTAGE_SCENARIOS.items()
                                if m != "omit_exposure_column"))
def test_source_mutant_selected_and_integral(arm_runner, mutant, scenario):
    clean = _strip_arm_tags(_digest(arm_runner, scenario, "facade"))
    ref_mut = _strip_arm_tags(
        _digest(arm_runner, scenario, "reference", sabotage=mutant))
    fac_mut = _strip_arm_tags(
        _digest(arm_runner, scenario, "facade", sabotage=mutant))
    assert fac_mut != clean, f"{mutant} NOT selected — comparator blind"
    assert ref_mut == fac_mut, (
        f"{mutant}: mutated arms disagree — the mutant is not testing "
        "the same behavioral change on both sides")


def test_digest_mutant_selected_and_integral(arm_runner):
    """Dossier mutant: omit the exposure column from the recorded
    digest — a comparator that only checks presence/rows would pass a
    broken surface."""
    scenario = SABOTAGE_SCENARIOS["omit_exposure_column"]
    clean = _strip_arm_tags(_digest(arm_runner, scenario, "facade"))
    ref_mut = _strip_arm_tags(
        _digest(arm_runner, scenario, "reference",
                sabotage="omit_exposure_column"))
    fac_mut = _strip_arm_tags(
        _digest(arm_runner, scenario, "facade",
                sabotage="omit_exposure_column"))
    assert fac_mut != clean, "omitted exposure column NOT selected"
    assert ref_mut == fac_mut
    assert fac_mut["origin_hazzard_columns"] == []
