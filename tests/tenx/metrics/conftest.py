"""Fixtures for the C12 metric-reuse suite (campaign una_10x).

The combined fold and the four generated split folds are imported from
the real installed modules under this worktree's src (the split folds
are AST surgery on the engine bytes — never retyped).  No fold
parameter name is retyped anywhere in this suite: parameter names come
from live introspection of the combined fold's signature, and cases are
built positionally in the C10 extractor order.
"""
import os
import sys
from pathlib import Path

# One numba cache owner, shared with the trace suite; set before numba
# is imported by any collection.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(__file__).resolve().parents[3] / ".numba_cache_owner"))

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "tests" / "tenx" / "identity"))

import _engine_facts as ef  # noqa: E402  (suite-local live-facts module)
import inspect  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import urban_network_analysis.Engines.AccessibilityWElevation as eng_mod
from urban_network_analysis.Engines.AccessibilityWElevation import (  # noqa: E402
    reach_gravity_knn_access as COMBINED)
from urban_network_analysis.Engines import _metric_reuse_scratch as mrs  # noqa: E402

IDENTITY_DIR = REPO / "tests" / "tenx" / "identity" / "tables"
SPLIT_NAMES = ("reach", "gravity_exponential", "gravity_logistic",
               "knn_access")

# C10 extractor order of the 13 fold parameters (the same order the
# identity suite's Driver uses; cross-checked against live introspection
# in test_metric_compiler_parity).
CASE_ORDER = ("d_distance", "d_weights", "cutoff", "gravity_beta",
              "gravity_plateau", "gravity_logistic_midpoint",
              "knn_gravity_plateau", "knn_weights", "knn_gravity_beta",
              "knn_decay", "knn_gravity_logistic_midpoint",
              "gravity_growth_rate", "knn_gravity_growth_rate")


def bits(x):
    return np.asarray(x, dtype=np.float64).tobytes()


@pytest.fixture(scope="session")
def live_params():
    names = list(
        inspect.signature(COMBINED.py_func).parameters)
    assert tuple(names) == CASE_ORDER
    return names


@pytest.fixture(scope="session")
def combined():
    return COMBINED


@pytest.fixture(scope="session")
def splits(live_params):
    out = {}
    for name in SPLIT_NAMES:
        fn = getattr(mrs, "_%s_only" % name)
        assert list(inspect.signature(fn.py_func).parameters) == \
            list(live_params), name
        out[name] = fn
    return out


@pytest.fixture(scope="session")
def table():
    import json
    # Generation 2 (see tests/tenx/identity/conftest.py); generation 1
    # stays preserved untouched as metric_dependencies.json.
    return json.loads(
        (IDENTITY_DIR / "metric_dependencies.gen2.json").read_text())


@pytest.fixture(scope="session")
def engine_sha():
    return mrs._engine_sha256()


def bind(case):
    """Keyword-bind a case dict by the LIVE signature, in CASE_ORDER."""
    return {k: case[k] for k in CASE_ORDER}


def run_all(case, combined, splits):
    """Combined twice (determinism guard) + every split once."""
    c1 = combined(**bind(case))
    c2 = combined(**bind(case))
    assert all(bits(a) == bits(b) for a, b in zip(c1, c2))
    out_vals = dict(zip(("reach", "gravity_exponential",
                         "gravity_logistic", "knn_access"), c1))
    split_vals = {name: fn(**bind(case)) for name, fn in splits.items()}
    return out_vals, split_vals


def mk_case(n=64, seed=7, decay="exponential", **over):
    g = np.random.default_rng(seed)
    case = dict(
        d_distance=g.uniform(0.0, 10.0, n),
        d_weights=g.uniform(0.5, 2.0, n),
        cutoff=6.0,
        gravity_beta=1.5, gravity_plateau=0.5,
        gravity_logistic_midpoint=2.0, knn_gravity_plateau=0.5,
        knn_weights=g.uniform(0.1, 1.0, max(n, 10))[::-1].copy(),
        knn_gravity_beta=1.5, knn_decay=decay,
        knn_gravity_logistic_midpoint=2.0,
        gravity_growth_rate=0.75, knn_gravity_growth_rate=0.75)
    case.update(over)
    return case


@pytest.fixture(scope="session")
def facts():
    return ef.fold_facts("reach_gravity_knn_access")


def metric_job(case):
    """A planner job dict from a characterization case."""
    return {"d_weights": case["d_weights"],
            "gravity_beta": case["gravity_beta"],
            "gravity_plateau": case["gravity_plateau"],
            "gravity_logistic_midpoint":
                case["gravity_logistic_midpoint"],
            "gravity_growth_rate": case["gravity_growth_rate"],
            "knn_decay": case["knn_decay"],
            "knn_weights": case["knn_weights"],
            "cutoff": case["cutoff"]}
