"""Fixtures for the P03M typed metric identity suite (campaign una_10x
post_p00, designs/04).

Reuses the C11 trace battery machinery verbatim (admitted cases, build
helper, bit comparators) by importlib-loading
``tests/tenx/trace/conftest.py`` from its bytes — the same single-source
pattern the P03C suite uses, so the admitted domain and the bit
comparators have no drift.  NUMBA_CACHE_DIR follows the C11
owner-owned scratch convention: ``cache=True`` jitting never writes
beside the sources.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

# Owner-owned scratch (C11 convention).
os.environ["NUMBA_CACHE_DIR"] = str(
    Path(__file__).resolve().parents[4] / "campaign_runs"
    / "_numba_cache_owner")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

_TENX = Path(__file__).resolve().parents[2]

_SPEC = importlib.util.spec_from_file_location(
    "p03m_trace_conftest", _TENX / "trace" / "conftest.py")
tck = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tck)

# Re-exported C11 battery helpers (single source, no drift).
admitted_case = tck.admitted_case
driver_args = tck.driver_args
CASE_PARAMS = tck.CASE_PARAMS
RESULT_FIELDS = tck.RESULT_FIELDS
METRIC_PARAMS = tck.METRIC_PARAMS
bits = tck.bits

# The suite metric bundle (C11 replay() law), as in the P03C suite.
BUNDLE = (1.5, 2.0, 3.0, 0.8)


@pytest.fixture
def tr():
    from urban_network_analysis.Engines import _trace_access_scratch
    return _trace_access_scratch


@pytest.fixture
def mid():
    from urban_network_analysis.Engines import _metric_identity
    return _metric_identity


def replay_binding(case, d_weights=None, bundle=BUNDLE, knn_decay="exponential",
                   knn_weights=None, cutoff=None):
    """The EXACT positional binding of ``replay_trace`` for a case."""
    return (case["dwt"] if d_weights is None else d_weights,
            bundle[0], bundle[1], bundle[2], bundle[3], knn_decay,
            METRIC_PARAMS[4] if knn_weights is None else knn_weights,
            case["cutoff"] if cutoff is None else cutoff)


def replay_with(trm, rec, case, **kw):
    """replay_trace with the bundle binding (overrides via kwargs)."""
    return trm.replay_trace(rec, *replay_binding(case, **kw))


def fields_bits(res, fields=RESULT_FIELDS):
    """{field: output bits} for one replay result."""
    return {f: bits(getattr(res, f)) for f in fields}


def changed_fields(a, b, fields=RESULT_FIELDS):
    """Fields whose output bits differ between two results."""
    return [f for f in fields if a[f] != b[f]]
