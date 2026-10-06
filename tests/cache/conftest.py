"""Shared pins for the stage-cache conformance suites.

The prepared layer (una_10x P02 increment B, ``batch/prepared.py``)
short-circuits BEFORE stage lookups when enabled (its default), which
would replace this suite's served-hit observables with prepared hits.
Stage-cache semantics are certified HERE in isolation; the prepared
layer's own hit/invalidation/parity semantics are certified by
``tests/tenx/post_p00/preparation/`` (PP02-PP04), including a dedicated
coexistence fixture.
"""
import pytest


@pytest.fixture(autouse=True)
def _prepared_off(monkeypatch):
    monkeypatch.setenv("UNA_PREPARED", "0")
