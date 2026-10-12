"""H00 negative tests: the public-dispatch probe (true API dispatch).

Wraps a synthetic stand-in class (never the real package - the real
dispatch behavior through UNA.RunBatch is owned by the public-chronology
tests); the probe mechanics under test are wrap fidelity, restore
fidelity and the illegal-entry refusal.
"""
from __future__ import annotations

import sys
import types

import pytest

from benchmarks.una_10x.dispatch_probe import PublicDispatchProbe


class _FakeUNA:
    def RunBatch(self, analysis, pairing_file=None, **kw):
        return ("batch", analysis)

    def RunAccessibility(self):
        return "access"

    def RunODM(self, *a, **k):
        return "odm"

    def RunFlow(self):
        return "flow"


@pytest.fixture()
def fake_una_module(monkeypatch):
    mod = types.ModuleType("_h00_fake_una_mod")
    mod.UNA = _FakeUNA
    monkeypatch.setitem(sys.modules, "_h00_fake_una_mod", mod)
    return mod


def test_probe_records_true_calls_and_restores_the_class(fake_una_module):
    original = _FakeUNA.RunBatch
    with PublicDispatchProbe(una_module="_h00_fake_una_mod") as probe:
        assert _FakeUNA.RunBatch is not original
        assert _FakeUNA().RunBatch("flow") == ("batch", "flow")
        _FakeUNA().RunAccessibility()
    assert probe.called == ["UNA.RunBatch", "UNA.RunAccessibility"]
    assert _FakeUNA.RunBatch is original
    assert _FakeUNA().RunAccessibility() == "access"


def test_probe_restores_on_exceptions(fake_una_module):
    original = _FakeUNA.RunFlow
    with pytest.raises(RuntimeError, match="mid-leg"):
        with PublicDispatchProbe(una_module="_h00_fake_una_mod"):
            raise RuntimeError("mid-leg failure")
    assert _FakeUNA.RunFlow is original


def test_probe_refuses_a_class_missing_public_entries(monkeypatch):
    mod = types.ModuleType("_h00_broken_una_mod")
    mod.UNA = type("UNA", (), {})          # no public entries at all
    monkeypatch.setitem(sys.modules, "_h00_broken_una_mod", mod)
    with pytest.raises(AssertionError, match="missing"):
        with PublicDispatchProbe(una_module="_h00_broken_una_mod"):
            pass
    assert not hasattr(mod.UNA, "RunBatch")   # nothing was half-wrapped


def test_probe_only_wraps_legal_public_entries(fake_una_module):
    _FakeUNA.helper = lambda self: "internal"
    try:
        with PublicDispatchProbe(una_module="_h00_fake_una_mod") as probe:
            assert _FakeUNA().helper() == "internal"
        assert probe.called == []          # helpers are not dispatch evidence
    finally:
        del _FakeUNA.helper
