"""Published Madina incident reproductions (FAILURES).

Sources (fetched 2026-09-30, repo archived read-only 2026-08-25):
  #7   una service_area: TypeError due to clipping (a-albashir, 2024-05-02)
  #8   error in creating street network (gvvgomez, 2024-12-26; screenshot-
       only) + unmerged PR #10 (TrueOthem, 2025-05-10): GeoPandas >= 1.0
       GeometryArray lost `.data` in node_edge_builder
  #12  pandas 3.0 / numpy 2.0 compatibility (Robinlovelace, 2026-06-29)
       + unmerged PR #13: pd.Series(fastpath=True) removed in pandas 3.0;
       np.array_split turns GeoDataFrame chunks into object ndarrays
  #14  published on TestPyPI but not PyPI (JamesParrott, 2026-07-13)
  #15  dependencies declared in docs/requirements.txt missing from
       pyproject.toml / requirements.txt; networkx missing (JamesParrott,
       2026-07-13)
  PR#9 redundant_edge_treatment: signature default 'discard' contradicts
       the docstring's 'Default "split"' (packet incident INC06)

Causal-intervention protocol per dossier 03: attempt A0 = pinned source
unchanged (original failure); A1 = the single documented correction
applied in memory (failure removed); A2 = deliberate restoration
(correction removed; failure returns within the safe bound).  The pinned
tree on disk is never modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from conftest import MADINA_SRC  # noqa: E402
from incident_runner import (assert_attempt_ok, run_attempt)  # noqa: E402

PAYLOAD_MADINA = str(HERE / "payloads" / "payload_madina_network.py")

pytestmark = pytest.mark.incidents


def _params(madina_src, **kw):
    base = {"madina_src": madina_src,
            "node_snapping_tolerance": 0.0,
            "action": "build_only"}
    base.update(kw)
    return base


# ----------------------------------------------------------------------
# Issue #8 / PR #10 — GeometryArray.data on network creation
# ----------------------------------------------------------------------

def test_issue8_geometryarray_data_causal_chain(attempts_root, madina_src, madina_python):
    d = attempts_root / "issue8"
    # A0: pinned source -> exact reported AttributeError at the reported site
    a0 = run_attempt(PAYLOAD_MADINA, _params(madina_src), d / "a0_pinned",
                     timeout_s=180, python_exe=madina_python)
    assert_attempt_ok(a0, allow_failed=True)
    assert a0["status"] == "failed"
    tb = a0["child_result"]["traceback"]
    assert "'GeometryArray' object has no attribute 'data'" in a0[
        "child_result"]["error"], a0["child_result"]["error"]
    assert "network_utils.py" in tb and "node_edge_builder" in tb

    # A1: the single PR-#10-style correction (in memory) -> failure removed;
    # the pipeline reaches the NEXT reported defect (fastpath, issue #12a)
    a1 = run_attempt(PAYLOAD_MADINA,
                     _params(madina_src, interventions=["I1_geomarray_data"]),
                     d / "a1_corrected", timeout_s=180, python_exe=madina_python)
    assert_attempt_ok(a1, allow_failed=True)
    if a1["status"] == "failed":
        assert "fastpath" in a1["child_result"]["error"], \
            a1["child_result"]["error"]

    # A2: deliberate restoration -> original failure returns
    a2 = run_attempt(PAYLOAD_MADINA, _params(madina_src), d / "a2_restored",
                     timeout_s=180, python_exe=madina_python)
    assert_attempt_ok(a2, allow_failed=True)
    assert a2["status"] == "failed"
    assert "no attribute 'data'" in a2["child_result"]["error"]


# ----------------------------------------------------------------------
# Issue #12a / PR #13 — pd.Series fastpath removed in pandas 3.0
# ----------------------------------------------------------------------

def test_issue12a_fastpath_removed_in_pandas3(attempts_root, madina_src, madina_python):
    d = attempts_root / "issue12a"
    # direct mechanism: fastpath kwarg gone in pandas 3
    import pandas as pd
    import numpy as np
    major = int(pd.__version__.split(".")[0])
    assert major >= 3, f"env is pandas {pd.__version__}; report needs 3.x"
    try:
        pd.Series(np.zeros(3), fastpath=True, index=[0, 1, 2])
        raised = None
    except TypeError as exc:
        raised = str(exc)
    assert raised is not None, "pandas accepted fastpath; report stale"

    # through the real path (needs the #8 correction first: single-chain)
    a0 = run_attempt(PAYLOAD_MADINA,
                     _params(madina_src, interventions=["I1_geomarray_data"]),
                     d / "a0_chain", timeout_s=180, python_exe=madina_python)
    assert_attempt_ok(a0, allow_failed=True)
    assert a0["status"] == "failed", a0["child_result"]
    assert "fastpath" in a0["child_result"]["error"]

    # A1: + I2 (drop fastpath kwarg exactly as PR #13 part 1) -> removed
    a1 = run_attempt(PAYLOAD_MADINA,
                     _params(madina_src,
                             interventions=["I1_geomarray_data",
                                            "I2_fastpath"]),
                     d / "a1_corrected", timeout_s=180, python_exe=madina_python)
    assert_attempt_ok(a1)
    assert a1["status"] == "completed"
    assert a1["child_result"]["result"]["build"]["nodes"] >= 12

    # A2: restore I2 only (keep I1) -> fastpath failure returns
    a2 = run_attempt(PAYLOAD_MADINA,
                     _params(madina_src, interventions=["I1_geomarray_data"]),
                     d / "a2_restored", timeout_s=180, python_exe=madina_python)
    assert_attempt_ok(a2, allow_failed=True)
    assert a2["status"] == "failed"
    assert "fastpath" in a2["child_result"]["error"]


# ----------------------------------------------------------------------
# Issue #12b / PR #13 — np.array_split on GeoDataFrame chunks (numpy 2)
# ----------------------------------------------------------------------

def test_issue12b_arraysplit_object_chunks(attempts_root, madina_src, madina_python):
    d = attempts_root / "issue12b"
    import numpy as np
    import pandas as pd
    major = int(np.__version__.split(".")[0])
    assert major >= 2, f"env is numpy {np.__version__}; report needs 2.x"
    gdf = pd.DataFrame({"a": [1, 2, 3, 4]})
    chunks = np.array_split(gdf, 2)
    chunk_broke = False
    try:
        _ = chunks[0].index
    except AttributeError:
        chunk_broke = True
    if not chunk_broke:
        pytest.skip("numpy/pandas pair keeps DataFrame chunks; report stale "
                    f"(numpy {np.__version__}, pandas {pd.__version__})")

    # full path: parallel_betweenness(num_cores=2) on the built project
    # (I1+I2 keep network creation alive; closest_destination=False isolates
    # the split defect).  Pinned parallel_betweenness PRINTS per-worker
    # errors and returns partial results instead of raising, so the
    # defect evidence is the error report in the captured stdout.
    a0 = run_attempt(PAYLOAD_MADINA,
                     _params(madina_src,
                             interventions=["I1_geomarray_data",
                                            "I2_fastpath"],
                             action="btn_parallel", num_cores=2,
                             closest_destination=False),
                     d / "a0_parallel", timeout_s=300, python_exe=madina_python)
    assert_attempt_ok(a0)
    bp = a0["child_result"]["result"]["btn_parallel"]
    assert bp["completed"] is True  # pinned driver silently returns partial
    assert "'numpy.ndarray' object has no attribute 'index'" in \
        bp["stdout_tail"], bp["stdout_tail"]

    # A1: I3 (.iloc position split, PR #13 part 2) -> split defect removed
    a1 = run_attempt(PAYLOAD_MADINA,
                     _params(madina_src,
                             interventions=["I1_geomarray_data",
                                            "I2_fastpath", "I3_arraysplit"],
                             action="btn_parallel", num_cores=2,
                             closest_destination=False),
                     d / "a1_corrected", timeout_s=300, python_exe=madina_python)
    assert_attempt_ok(a1)
    bp1 = a1["child_result"]["result"]["btn_parallel"]
    # the ndarray-chunk defect must be GONE from the worker error reports
    # (whatever else pinned madina prints downstream)
    assert "'numpy.ndarray' object has no attribute 'index'" not in \
        bp1["stdout_tail"], bp1["stdout_tail"]
    if bp1["completed"] is False:
        assert "numpy.ndarray" not in (bp1.get("raised") or ""), \
            bp1["raised"]


# ----------------------------------------------------------------------
# Issue #7 — service_area clip with GeometryCollection scope
# ----------------------------------------------------------------------

def test_issue7_service_area_geometrycollection(attempts_root, madina_src, madina_python):
    d = attempts_root / "issue7"
    params = _params(madina_src,
                     interventions=["I1_geomarray_data", "I2_fastpath"],
                     action="service_area", search_radius=1000)
    a0 = run_attempt(PAYLOAD_MADINA, params, d / "a0", timeout_s=300, python_exe=madina_python)
    assert_attempt_ok(a0)
    sa = a0["child_result"]["result"]["service_area"]
    assert sa["entered"] is True
    assert "GeometryCollection" in sa["raised"], sa["raised"]
    assert "'mask' should be" in sa["raised"] or "clip" in \
        sa["traceback"], sa["raised"]
    # causal control: the scope is a mixed-dimension union because the
    # destination hull degenerates (few point destinations); the same call
    # on a non-degenerate scope must not raise — recorded via the
    # betweenness arm (same fixture, non-degenerate aggregation) as the
    # intervention belongs to TOPOLOGY, not to this reproduction.


# ----------------------------------------------------------------------
# BTN stats NameError + d_graph poisoning (bug registry: reproduced)
# ----------------------------------------------------------------------

def test_btn_stats_nameerror_and_graph_poisoning(attempts_root, madina_src, madina_python):
    d = attempts_root / "btn_stats"
    params = _params(madina_src,
                     interventions=["I1_geomarray_data", "I2_fastpath"],
                     action="btn_exposure", closest_destination=True,
                     search_radius=1000)
    a0 = run_attempt(PAYLOAD_MADINA, params, d / "a0_closest_true",
                     timeout_s=300, python_exe=madina_python)
    assert_attempt_ok(a0)
    btn = a0["child_result"]["result"]["btn"]
    assert btn["raised"] is None, btn["raised"]  # bare excepts swallow it
    assert btn["per_origin_error_lines"], (
        "expected per-origin statistics errors with closest_destination=True")
    assert all("error collecting origin statistics" in ln
               for ln in btn["per_origin_error_lines"]), btn
    # the swallowed exception's traceback lands on stderr
    assert "UnboundLocalError" in btn["stderr_tail"] and \
        "eligible_destinations_shortest_distance" in btn["stderr_tail"], \
        btn["stderr_tail"]
    # poisoning: ORIGIN nodes the handler failed to remove remain in the
    # shared working graph after the call (destination nodes are
    # pre-inserted into d_graph by create_graph and are not leak evidence)
    assert btn["origin_nodes_still_in_d_graph"] == btn["od_node_ids"][:2], (
        f"expected all origins leaked into d_graph, got "
        f"{btn['origin_nodes_still_in_d_graph']}")
    assert btn["d_graph_nodes_after"] > btn["d_graph_nodes_before"]

    # causal intervention: single setting flip closest_destination=False
    a1 = run_attempt(PAYLOAD_MADINA,
                     dict(params, closest_destination=False),
                     d / "a1_closest_false", timeout_s=300, python_exe=madina_python)
    assert_attempt_ok(a1)
    btn1 = a1["child_result"]["result"]["btn"]
    assert btn1["raised"] is None
    assert not btn1["per_origin_error_lines"], btn1["per_origin_error_lines"]
    assert not btn1["origin_nodes_still_in_d_graph"], (
        "clean run leaked ORIGIN nodes into d_graph")


# ----------------------------------------------------------------------
# Issue #15 — networkx missing from declared dependencies
# ----------------------------------------------------------------------

def test_issue15_networkx_dependency_declaration(madina_src):
    clone_root = Path(madina_src).parent
    declared_files = [clone_root / "pyproject.toml",
                      clone_root / "requirements.txt"]
    for declared in declared_files:
        text = declared.read_text(encoding="utf-8").lower()
        assert "networkx" not in text, (
            f"{declared.name} declares networkx; report #15 stale")
    from madina_bridge import module_source
    # networkx is imported at module level across the una package
    assert "import networkx" in module_source(
        madina_src, "madina/una/betweenness.py")
    assert "import networkx" in module_source(
        madina_src, "madina/una/paths.py")


# ----------------------------------------------------------------------
# Issue #14 — distribution: TestPyPI only (documented report; the local
# observable is the packaging metadata: conda recipe present, no PyPI
# release workflow in the pinned tree)
# ----------------------------------------------------------------------

def test_issue14_distribution_metadata(madina_src):
    clone_root = Path(madina_src).parent
    assert (clone_root / "meta.yaml").is_file(), \
        "pinned clone lost its conda recipe; report #14 evidence stale"
    gh_dir = clone_root / ".github" / "workflows"
    assert not gh_dir.exists() or not any(gh_dir.glob("*.yml")), (
        "pinned tree gained release workflows; report #14 stale")


# ----------------------------------------------------------------------
# PR #9 — redundant_edge_treatment: code default 'discard' vs docstring
# 'Default "split"' (packet incident INC06)
# ----------------------------------------------------------------------

def test_issue9_redundant_edge_default_matches_code_not_docstring(
        attempts_root, madina_src, madina_python):
    # source-level: the signature default and the docstring disagree
    from madina_bridge import module_source
    src = module_source(madina_src, "madina/zonal/zonal.py")
    assert "redundant_edge_treatment: str ='discard'" in src, (
        "pinned signature default changed; PR #9 evidence stale")
    assert 'Default "split"' in src, (
        "pinned docstring no longer claims 'split'; PR #9 evidence stale")

    d = attempts_root / "issue9"
    common = {"madina_src": madina_src, "fixture": "redundant",
              "action": "build_only",
              "interventions": ["I1_geomarray_data", "I2_fastpath"]}
    # default call: the LONGER of the two redundant edges must be gone
    # (discard behavior), not split -- grid 17 + straight diagonal 1 = 18
    a0 = run_attempt(PAYLOAD_MADINA, dict(common), d / "a0_default",
                     timeout_s=180, python_exe=madina_python)
    assert_attempt_ok(a0)
    b0 = a0["child_result"]["result"]["build"]
    assert b0["edges"] == 18 and not any(w > 150.0 for w in b0["weights"]), b0

    # causal contrast: explicit 'split' keeps the SHORTEST redundant edge
    # intact and splits the longer one at its midpoint (+1 node, +2 edges
    # of weight/2 -- pinned network_utils.py "sort_values('weight').index[1:]")
    a1 = run_attempt(PAYLOAD_MADINA,
                     dict(common, redundant_edge_treatment="split"),
                     d / "a1_split", timeout_s=180, python_exe=madina_python)
    assert_attempt_ok(a1)
    b1 = a1["child_result"]["result"]["build"]
    assert (b1["nodes"], b1["edges"]) == (13, 20), b1
    assert not any(w > 150.0 for w in b1["weights"]), b1["weights"]

    # causal contrast: explicit 'keep' retains both (17 + 2), incl. the
    # longer one -- so the docstring's claimed default IS reachable and
    # distinguishable from what the code actually does
    a2 = run_attempt(PAYLOAD_MADINA,
                     dict(common, redundant_edge_treatment="keep"),
                     d / "a2_keep", timeout_s=180, python_exe=madina_python)
    assert_attempt_ok(a2)
    b2 = a2["child_result"]["result"]["build"]
    assert b2["edges"] == 19 and any(w > 150.0 for w in b2["weights"]), b2
