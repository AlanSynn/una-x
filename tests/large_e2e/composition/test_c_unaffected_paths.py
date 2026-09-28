"""I00 composition: unaffected paths.

The composed source surface is exactly the union of the five screened
tracks. Pinned from live bytes at test runtime:

* contract identity — CONTRACT.md is frozen; its sha256 is pinned
  here so contract drift fails loudly BEFORE the census compares the
  tree against the copied expected inventory (h04 I00 plan item R6
  companion);
* package census — of the candidate tree's .py files, exactly three are
  modified vs B0 (Accessibility.py, AccessibilityWElevation.py,
  AggregateFlow.py), exactly two are new (the two private scratch
  modules), and every other file — including AccessibilityWTurns.py,
  _ordered_csr.py, Settings.py, Logger.py — is byte-identical;
* the turns pipeline (turns=True flow engine at K=2) is byte-identical
  across arms and its turns kernel carries identical targetoptions (no
  nogil in either arm);
* the non-turns stats surface is untouched by turns settings (the
  turns driver calls _accumulate_od_flow_turns directly; no F2 wrapper
  engagement);
* A1/A3 dispatch symbols are absent from the turns engine source.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import sys

from harness_composition import (assert_flow_bytes, b0ns, build_flow_arm,
                                 cns, flow_output_arrays, record)

WT = "/Users/alansynn/orca/workspaces/una-x/wt-large-e2e"
CONTRACT_PATH = f"{WT}/campaigns/una_large_e2e/CONTRACT.md"
CONTRACT_SHA256 = ("3283a12e70ec72163556c94f65269bc422cb621f860900000"
                   "e672a7ef80adad1")

EXPECTED_MODIFIED = (
    "Engines/Accessibility.py",
    "Engines/AccessibilityWElevation.py",
    "Engines/AggregateFlow.py",
)
EXPECTED_NEW = (
    "Engines/_large_access_scratch.py",
    "Engines/_large_flow_workspace.py",
)


def test_contract_is_frozen():
    """CONTRACT.md must be byte-identical to the frozen packet bytes
    before any census compares against the inventory derived from it."""
    with open(CONTRACT_PATH, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    assert digest == CONTRACT_SHA256, \
        f"CONTRACT.md drifted from the frozen packet: {digest}"


def _tree_files(root):
    pkg_dir = os.path.join(root, "urban_network_analysis")
    out = {}
    for dirpath, _dirnames, filenames in os.walk(pkg_dir):
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, fn), pkg_dir)
                out[rel] = os.path.join(dirpath, fn)
    return out


def test_package_census_from_bytes():
    """The composed diff surface vs B0 is exactly three modified + two
    new files; all other files byte-identical; no file removed."""
    files_b = _tree_files(b0ns().root)
    files_c = _tree_files(cns().root)
    assert set(files_b) | set(EXPECTED_NEW) == set(files_c), \
        "candidate tree shape diverged from the declared surface"
    modified, identical = [], []
    for rel, path_c in sorted(files_c.items()):
        if rel in EXPECTED_NEW:
            assert rel not in files_b
            continue
        with open(files_b[rel], "rb") as fh:
            bytes_b = fh.read()
        with open(path_c, "rb") as fh:
            bytes_c = fh.read()
        if bytes_b == bytes_c:
            identical.append(rel)
        else:
            modified.append(rel)
    assert sorted(modified) == sorted(EXPECTED_MODIFIED), \
        f"modified set diverged: {modified}"
    record("package_census", modified=sorted(modified),
           new=list(EXPECTED_NEW), identical_count=len(identical))


def _spec():
    from fixtures_composition import flow_spec_oracle
    return flow_spec_oracle()


def test_turns_engine_byte_equal_k2():
    """turns=True engine case at K=2 (executor path of the turns
    driver): end-to-end outputs byte-identical across arms."""
    spec_extra = {"turns": True}
    eng_b, settings_b = build_flow_arm(b0ns(), _spec(), 2,
                                       extra_overrides=spec_extra)
    eng_c, settings_c = build_flow_arm(cns(), _spec(), 2,
                                       extra_overrides=spec_extra)
    eng_b.Centrality(settings_b)
    eng_c.Centrality(settings_c)
    assert_flow_bytes(flow_output_arrays(eng_c),
                      flow_output_arrays(eng_b), "unaffected/turns_true/k2")
    record("turns_engine_k2")


def test_turns_kernel_targetoptions_identical():
    """_accumulate_od_flow_turns targetoptions are IDENTICAL across
    arms (the turns kernel is not part of the change surface)."""
    b0_mod = sys.modules["urban_network_analysis.Engines.AggregateFlow"]
    c_mod = cns().flow_module
    name = "_accumulate_od_flow_turns"
    assert name in vars(b0_mod) and name in vars(c_mod), \
        "turns kernel missing from an arm's module"
    opts_b = dict(vars(b0_mod)[name].targetoptions)
    opts_c = dict(vars(c_mod)[name].targetoptions)
    assert opts_b == opts_c, \
        f"turns kernel targetoptions diverged: {opts_b} vs {opts_c}"
    assert not opts_c.get("nogil", False), \
        "turns kernel unexpectedly carries nogil"


def test_turns_driver_source_has_no_track_hooks():
    """The turns driver source contains no local-route call and no A1/A3
    dispatch symbols: the turns pipeline predates and excludes every
    track (static read of the composed module's live source)."""
    flow_mod = cns().flow_module
    src = inspect.getsource(flow_mod.AggregateFlow
                            ._process_origins_aggregate_turns)
    for token in ("_use_local_route", "_f2_stats", "_a1_", "_a3_",
                  "_lfws"):
        assert token not in src, \
            f"turns driver source references track symbol {token}"


def test_wturns_engine_file_byte_identical():
    """AccessibilityWTurns.py is byte-identical across arms (the engine
    family carries neither A1 nor A3 symbols — F-side turns pipeline
    covered above)."""
    files_b = _tree_files(b0ns().root)
    files_c = _tree_files(cns().root)
    rel = "Engines/AccessibilityWTurns.py"
    with open(files_b[rel], "rb") as fh:
        bytes_b = fh.read()
    with open(files_c[rel], "rb") as fh:
        bytes_c = fh.read()
    assert bytes_b == bytes_c, "AccessibilityWTurns.py diverged"
    assert b"_a1_" not in bytes_c and b"_a3_" not in bytes_c \
        and b"_a2_" not in bytes_c, \
        "AccessibilityWTurns.py references a track symbol"
