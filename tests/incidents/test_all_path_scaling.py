"""All-path enumeration scaling mechanism (dossier 03 failure class:
"finite but combinatorial all-path work") on square lattices.

Inputs are small and each is executed once per attempt; growth across
(N, detour_ratio) cells is the mechanism evidence (paths generated and
output-size lower bound), with a per-attempt independent watchdog.  A
watchdog hit is recorded as censored data for that cell (capsule
preserved), never as a root cause, and the test does not re-run a
censored cell at a larger size.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from incident_runner import run_attempt  # noqa: E402

PAYLOAD = str(HERE / "payloads" / "payload_allpaths.py")

pytestmark = pytest.mark.incidents

CELLS = [  # (N, detour_ratio): increasing work in BOTH factors
    (2, 1.0), (2, 1.5),
    (3, 1.0), (3, 1.5),
    (4, 1.0), (4, 1.5),
]


def test_all_path_growth_on_lattices(attempts_root, madina_src, madina_python):
    rows = []
    for n, ratio in CELLS:
        rec = run_attempt(
            PAYLOAD, {"madina_src": madina_src, "n": n,
                      "detour_ratio": ratio},
            attempts_root / f"allpaths_n{n}_r{ratio}",
            timeout_s=240, python_exe=madina_python)
        if rec["status"] == "incomplete":
            # censored cell: bounded watchdog outcome, capsule retained;
            # growth beyond this point is not measured (and not claimed)
            rows.append({"n": n, "detour_ratio": ratio,
                         "status": "incomplete(censored)",
                         "wall_s": rec["wall_s"]})
            break
        assert rec["status"] == "completed", rec["child_result"]
        att = rec["child_result"]["result"]["attempts"][0]
        att["status"] = "completed"
        att["wall_s"] = rec["wall_s"]
        rows.append(att)

    completed = [r for r in rows if r["status"] == "completed"]
    assert len(completed) >= 3, "too few bounded cells completed to classify"
    # mechanism: path count grows combinatorially with size and detour
    same_ratio = [r for r in completed if r["detour_ratio"] == 1.0]
    if len(same_ratio) >= 3:
        g = [r["paths_generated"] for r in same_ratio]
        assert g == sorted(g), f"path count must grow with N: {g}"
    big = max(completed, key=lambda r: (r["n"], r["detour_ratio"]))
    small = min(completed, key=lambda r: (r["n"], r["detour_ratio"]))
    factor = big["paths_generated"] / max(small["paths_generated"], 1)
    assert factor > 2.0, (
        f"expected combinatorial growth, got factor {factor}")
    # output-size lower bound documented for the exact-ALL contract
    for r in completed:
        assert r["output_lower_bound_bytes_estimate"] > 0
