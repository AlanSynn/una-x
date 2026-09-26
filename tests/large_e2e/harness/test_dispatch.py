"""Dispatcher unit tests (HARNESS.md behavior 1 + negative tests).

Mock UNA objects assert EXACT per-method call counts, including zero calls
to the wrong methods, and that unsupported analyses error instead of
defaulting to accessibility.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HARNESS_DIR = (Path(__file__).resolve().parents[3] / "benchmarks" / "large_e2e")
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from harness.dispatch import (  # noqa: E402
    ANALYSIS_TO_METHOD,
    UnsupportedAnalysisError,
    execute_analysis,
    extract_signature,
    validate_analysis,
)


class MockUNA:
    """Records every public-method call so exact counts can be asserted.

    Engines are attribute objects, mirroring the real engines whose result
    arrays live on attributes (Base.reach, Base.edge_flow, ...).
    """

    def __init__(self):
        self.calls: list[str] = []
        self.accessibility = None
        self.flow = None

    def RunAccessibility(self):
        self.calls.append("RunAccessibility")
        self.accessibility = SimpleNamespace(
            reach=[1.0, 2.0],
            gravity_exponential=[1.0, 2.0],
            gravity_logistic=[1.0, 2.0],
            knn_access=[1.0, 2.0],
        )

    def RunFlow(self):
        self.calls.append("RunFlow")
        self.flow = SimpleNamespace(
            edge_flow=[1.0], edge_flow_AB=[1.0], edge_flow_BA=[1.0],
            node_flow=None)

    def RunODM(self, format="Sqlite", speed=5.0, file_name=None):
        self.calls.append(f"RunODM(format={format}, speed={speed}, "
                          f"file_name={file_name})")
        return {"od": True}


def test_accessibility_request_invokes_accessibility_only():
    una = MockUNA()
    outcome = execute_analysis(una, "accessibility")
    assert una.calls == ["RunAccessibility"]
    assert outcome.method_name == "RunAccessibility"
    assert outcome.signature["engine_attr"] == "accessibility"
    assert outcome.signature["engine_present"] is True
    assert outcome.signature["result_attrs"]["reach"] == "len=2"


def test_flow_request_invokes_flow_only():
    una = MockUNA()
    outcome = execute_analysis(una, "flow")
    assert una.calls == ["RunFlow"]
    assert outcome.method_name == "RunFlow"
    assert outcome.signature["engine_attr"] == "flow"
    assert outcome.signature["result_attrs"]["edge_flow"] == "len=1"


def test_odm_request_invokes_odm_only():
    una = MockUNA()
    outcome = execute_analysis(una, "odm", {"format": "csv", "speed": 4.0})
    assert una.calls == ["RunODM(format=csv, speed=4.0, file_name=None)"]
    assert outcome.method_name == "RunODM"
    assert outcome.signature["engine_attr"] == "od"


def test_invalid_analysis_errors_and_never_defaults_to_accessibility():
    una = MockUNA()
    for bad in ("centrality", "", None, "ACCESS", "Accessibilty", 3):
        with pytest.raises(UnsupportedAnalysisError):
            execute_analysis(una, bad)
        with pytest.raises(UnsupportedAnalysisError):
            validate_analysis(bad)
    assert una.calls == []  # zero calls of ANY method for bad analyses


def test_analysis_name_normalization_is_lowercase_exact():
    assert validate_analysis("Flow") == "flow"
    assert validate_analysis(" ODM ") == "odm"


def test_method_table_covers_all_supported_analyses():
    assert set(ANALYSIS_TO_METHOD) == {"accessibility", "flow", "odm"}
    assert ANALYSIS_TO_METHOD["accessibility"] == "RunAccessibility"
    assert ANALYSIS_TO_METHOD["flow"] == "RunFlow"
    assert ANALYSIS_TO_METHOD["odm"] == "RunODM"


def test_signature_reports_absent_engine_honestly():
    una = MockUNA()  # no method called: engines still None
    sig = extract_signature(una, "accessibility")
    assert sig["engine_present"] is False
    assert sig["result_attrs"]["reach"] == "absent"


def test_odm_signature_does_not_invent_an_engine_attribute():
    una = MockUNA()
    execute_analysis(una, "odm")
    sig = extract_signature(una, "odm")
    assert sig["engine_present"] is False  # RunODM keeps the engine local
    assert "note" in sig["result_attrs"]
