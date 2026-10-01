"""NO-ledger surface pin for the workflows module (mirrors the flow
suite's reflection test): signatures and docstring heads captured from
each arm's own module must agree.  The facade body is verbatim, so any
signature drift is a ledger omission."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.madina_api


def test_logger_and_workflow_signatures_match_reference(
        reflection_runner):
    ref = reflection_runner("reference")
    fac = reflection_runner("facade")
    ref.pop("arm")
    fac.pop("arm")
    assert fac == ref
    # the load-bearing pins, asserted explicitly so a digest accident
    # cannot silently void them
    assert fac["signature::betweenness_flow_simulation"] == (
        "(city_name=None, data_folder=None, output_folder=None, "
        "pairings_file='pairings.csv', num_cores=8) -> None")
    assert fac["signature::KNN_accessibility"] == (
        "(city_name=None, data_folder=None, output_folder=None, "
        "pairings_file='pairing.csv', num_cores=8)")
    assert fac["signature::Logger.pairing_end"].startswith(
        "(self, shaqra: <Zonal>, pairing: pandas.Series, "
        "save_flow_map=True")
    assert "save_diagnostics_map=False" in (
        fac["signature::Logger.pairing_end"])
    for m in ("log", "pairing_end", "simulation_end",
              "flow_map_template_1"):
        assert m in fac["Logger::methods"]
    assert fac["docstring_head::betweenness_flow_simulation"].startswith(
        "A workflow to generate trips between pairs of origins")
