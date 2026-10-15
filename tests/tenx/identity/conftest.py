"""Fixtures for the C10 identity suite (campaign una_10x).

The compiled fold is bound POSITIONALLY in the order extracted from the
engine AST (``facts['params']``) — no fold parameter name is retyped
anywhere in this suite.  The dependency table and the trace contract are
loaded from identity/tables/ (relocated byte-identically from
campaigns/una_10x/identity/) and cross-checked against the
live AST facts.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _engine_facts as ef  # noqa: E402  (suite-local live-facts module)

IDENTITY_DIR = REPO / "tests" / "tenx" / "identity" / "tables"
FOLD_NAME = "reach_gravity_knn_access"


@pytest.fixture(scope="session")
def facts():
    return ef.fold_facts(FOLD_NAME)


@pytest.fixture(scope="session")
def bindings():
    return ef.fold_call_bindings(FOLD_NAME)


@pytest.fixture(scope="session")
def derivation():
    return ef.wrapper_derivation(kwarg_sub="growth")


@pytest.fixture(scope="session")
def contract():
    return json.loads((IDENTITY_DIR / "trace_contract.json").read_text())


@pytest.fixture(scope="session")
def table():
    # Generation 2: the engine evolved by the gated lazy-import change
    # (exl-20261011T164228Z); generation 1 stays preserved untouched as
    # metric_dependencies.json (Z00 campaign-close record).
    return json.loads(
        (IDENTITY_DIR / "metric_dependencies.gen2.json").read_text())


@pytest.fixture(scope="session")
def fold(facts):
    mod = importlib.import_module(
        "urban_network_analysis.Engines.AccessibilityWElevation")
    fn = getattr(mod, facts["fold_name"])
    return fn


class Driver:
    """Positional driver for the 13-param fold, extractor-ordered."""

    def __init__(self, facts, fold):
        self.names = facts["params"]
        self.fold = fold
        self.facts = facts
        self.defaults = {
            0: np.array([1.0, 2.0, 3.0, 8.0]),      # retained distances
            1: np.array([1.0, 2.0, 0.5, 4.0]),      # destination weights
            2: 4.0,                                  # cutoff
            3: 1.5,                                  # gravity beta
            4: 0.5,                                  # gravity plateau
            5: 2.0,                                  # logistic midpoint
            6: 0.5,                                  # knn plateau (alias)
            7: np.array([1.0, 0.9, 0.8, 0.7, 0.6,
                         0.5, 0.4, 0.3, 0.2, 0.1]),  # knn coefficients
            8: 1.5,                                  # knn beta (alias)
            9: "logistic",                           # decay branch value
            10: 2.0,                                 # knn midpoint (alias)
            11: 0.75,                                # growth
            12: 0.75,                                # knn growth (alias)
        }

    def args(self, **overrides):
        """overrides keyed by EXTRACTED param name (never hand-typed)."""
        args = [self.defaults[i] for i in range(len(self.names))]
        for k, v in overrides.items():
            args[self.names.index(k)] = v
        return args

    def __call__(self, **overrides):
        return self.fold(*self.args(**overrides))


@pytest.fixture(scope="session")
def driver(facts, fold):
    return Driver(facts, fold)
