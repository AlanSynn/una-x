"""Fixtures for the S11 sparse-kernel suite (campaign una_10x).

The case battery is the cpu_kernels battery loaded verbatim from its
single source file (``tests/cpu_kernels/conftest.py``) — the battery
has one definition and this suite imports it.  The battery's
destination-terminal weights carry a -1.0 entry in EVERY case (by
cpu_kernels design, "negative/zero terminal weights"), which lies
OUTSIDE the dossier-02 sparse value domain (finite nonnegative seeds,
terminal costs and destination weights), so ``admitted_case`` clones a
battery case into the admitted domain (``|dw|``) and ``driver_args``
binds the sparse driver's positional signature.  Both behaviors are
gates: the sparse route AND the refusal->unchanged-original fallback
must each be bitwise-exact against the actual compiled baseline.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

_CK_PATH = Path(__file__).resolve().parents[2] / "cpu_kernels" / "conftest.py"
_spec = importlib.util.spec_from_file_location("cpu_kernels_conftest", _CK_PATH)
ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck)

build_case = ck.build_case
CASE_PARAMS = ck.CASE_PARAMS
case_kwargs = ck.case_kwargs
bits = ck.bits

# Metric parameters shared by every driver call in this suite; the fold
# itself is always the engine's own compiled fold (both drivers call it).
METRIC_PARAMS = (1.5, 2.0, 3.0, 0.8,
                 np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))


def metric_args(knn_decay="exponential"):
    return METRIC_PARAMS[:4] + (knn_decay,) + (METRIC_PARAMS[4],)


def admitted_case(name):
    """Battery case cloned into the dossier-02 admitted value domain.

    ``|dw|`` removes the battery's designed -1.0 destination-terminal
    cost; the seed clone replaces non-finite/negative SEED weights with
    1.0 (the nan_seed/neg_seed variants write ``ow`` directly).  Value
    refusals for the UNCLONED battery are covered by the fallback test.
    """
    case = build_case(3 + CASE_PARAMS.index(name), **case_kwargs(name))
    case["dw"] = np.abs(case["dw"])
    case["ow"] = np.where(np.isfinite(case["ow"]), np.abs(case["ow"]),
                          np.float64(1.0))
    return case


def driver_args(case, knn_decay="exponential"):
    """Positional arguments of the 16-parameter driver signature."""
    return (case["o"], case["ow"], case["ptr"], case["vec"], case["wts"],
            case["flag"], case["d"], case["dw"], case["dwt"]) \
        + metric_args(knn_decay) + (case["cutoff"],)


RESULT_FIELDS = ("reach", "gravity_exponential", "gravity_logistic",
                 "knn_access")


def bit_equal_result(res, baseline):
    """Bitwise equality of the four driver outputs."""
    return all(bits(getattr(res, f)) == bits(b)
               for f, b in zip(RESULT_FIELDS, baseline))


@pytest.fixture
def sp():
    from urban_network_analysis.Engines import _sparse_access_scratch
    return _sparse_access_scratch


@pytest.fixture
def baseline():
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        integrated_scope_access)
    return integrated_scope_access
