"""Fixtures for the post-P00 suites (campaign una_10x, P01M onward).

Same self-contained public-API style as the cohort island: the batch is
exercised through the PUBLIC RunBatch API over the frozen 3x3 smoke
grid.  Telemetry tests toggle the opt-in environment (UNA_PHASE_*)
through monkeypatch so spawned workers inherit it at p.start().

The P03C replay/ suite's machinery lives here too: a second conftest.py
in replay/ would shadow THIS module for the island's
``from conftest import tree_bytes`` users.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO / "src"),):
    if p not in sys.path:
        sys.path.insert(0, p)

SMOKE = REPO / "tests" / "execution" / "fixtures"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "tenx_post_p00: post-P00 program tests (una_10x)")


@pytest.fixture(scope="session")
def smoke_fixture() -> Path:
    for f in ("network.geojson", "origins.geojson", "destinations.geojson"):
        if not (SMOKE / f).is_file():
            pytest.skip(f"smoke-grid fixture missing: {f}")
    return SMOKE


@pytest.fixture
def make_batch(smoke_fixture, tmp_path):
    """UNA instance with N independent accessibility rows over the frozen
    smoke grid (same shape as the cohort island's fixture)."""
    from urban_network_analysis import UNA

    def make(n=2, *, out_folder=None, extra=None):
        p = UNA(verbosity=0)
        p.settings.data_folder = str(smoke_fixture)
        p.settings.network_file = "network.geojson"
        p.settings.origins_file = "origins.geojson"
        p.settings.destinations_file = "destinations.geojson"
        p.settings.output_folder = str(out_folder or tmp_path / "out")
        p.settings.output_wStamp = False
        for i in range(n):
            p.settings.name = f"row{i}"
            p.settings.search_radius = 90 + 10 * i
            if extra:
                extra(i, p.settings)
            p.SaveSettingsToProject()
        return p

    return make


@pytest.fixture
def telemetry_extra():
    """The extra['telemetry'] block of a run's batch report (or None)."""
    def get(una):
        extra = getattr(una.batch_report, "extra", None) or {}
        return extra.get("telemetry")
    return get


def tree_bytes(root, skip_hidden=False):
    """relpath -> bytes for every file under root (hidden runtime entries
    excluded on both sides when skip_hidden)."""
    out = {}
    for dirpath, _dirs, names in os.walk(root):
        for name in names:
            if skip_hidden and name.startswith("."):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            if skip_hidden and (rel.startswith(".") or "/." in rel):
                continue
            out[rel] = open(path, "rb").read()
    return out


# ----------------------------------------------------------------------
# P03C compiled-outer-replay machinery (replay/ suite).
#
# The C11 trace island's case machinery is importlib-loaded from its
# file.  The owner numba cache scratch is forced BEFORE the load,
# exactly as the trace island does.  Field names the direct-record
# helper hand-uses are census-asserted against the live dataclasses:
# a mistyped name fails loudly at collection instead of silently
# constructing a wrong record.
# ----------------------------------------------------------------------
import dataclasses
import hashlib
import importlib.util
import types

os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(REPO / "campaign_runs" / "_numba_cache_owner"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

_P03C_SPEC = importlib.util.spec_from_file_location(
    "p03c_trace_conftest",
    REPO / "tests" / "tenx" / "trace" / "conftest.py")
_tc = importlib.util.module_from_spec(_P03C_SPEC)
sys.modules["p03c_trace_conftest"] = _tc
_P03C_SPEC.loader.exec_module(_tc)

import urban_network_analysis.Engines._trace_access_scratch as _trm

_RECORD_FIELDS = {f.name for f in dataclasses.fields(_trm.TraceRecord)}
for _name in (
    "origin_count", "dest_count", "cutoff_value", "cutoff_dtype",
    "metric_dtype", "indptr", "destination_ids", "distance_bits",
    "content_digest",
):
    assert _name in _RECORD_FIELDS, f"field census miss: {_name}"

P03C_FOLD_FIELD = [f.name for f in dataclasses.fields(
    _trm.TraceReplayResult) if "fold" in f.name][0]
P03C_RESULT_FIELDS = _tc.RESULT_FIELDS
P03C_METRIC_PARAMS = _tc.METRIC_PARAMS


def p03c_direct_record(rec0, indptr, ids, dist, cutoff,
                       cutoff_dtype="float64", metric_dtype=None,
                       dest_count=None):
    """A structurally valid record rebuilt from hand-packed arrays via
    dataclasses.replace (census-verified field names) with the packed
    bytes digest recomputed — the same gates must accept it that
    accept a produced record."""
    import numpy as np
    digest = hashlib.sha256(
        np.asarray(indptr).tobytes() + np.asarray(ids).tobytes()
        + np.asarray(dist).tobytes()).hexdigest()
    kw = {
        "origin_count": int(np.asarray(indptr).shape[0]) - 1,
        "cutoff_value": float(cutoff),
        "cutoff_dtype": str(cutoff_dtype),
        "indptr": indptr,
        "destination_ids": ids,
        "distance_bits": dist,
        "content_digest": digest,
    }
    if dest_count is not None:
        kw["dest_count"] = int(dest_count)
    if metric_dtype is not None:
        kw["metric_dtype"] = str(metric_dtype)
    return dataclasses.replace(rec0, **kw)


@pytest.fixture
def tr():
    return _trm


@pytest.fixture
def baseline():
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        integrated_scope_access)
    return integrated_scope_access


@pytest.fixture
def tch():
    """The P03C case-machinery namespace."""
    import numpy as np

    def outbits(rp):
        """Every observable replay surface: the four fold outputs by
        raw bits, the fold count, the route and the refusal reason."""
        return (tuple(np.asarray(getattr(rp, f)).tobytes()
                      for f in P03C_RESULT_FIELDS)
                + (getattr(rp, P03C_FOLD_FIELD), rp.route, rp.refusal))

    return types.SimpleNamespace(
        FOLD_FIELD=P03C_FOLD_FIELD,
        RESULT_FIELDS=P03C_RESULT_FIELDS,
        METRIC_PARAMS=P03C_METRIC_PARAMS,
        driver_args=_tc.driver_args,
        ring_case=_tc.ring_case,
        build=_tc.build,
        replay=_tc.replay,
        result_equal=_tc.result_equal,
        bits=_tc.bits,
        outbits=outbits,
        direct_record=p03c_direct_record,
        np=np)
