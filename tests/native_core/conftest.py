"""Fixtures for the NATIVE_CORE conformance suite.

Imports the CPU_ALGORITHMS battery (``tests/cpu_kernels/conftest.py``)
so one seed pins one topology across engines, oracle and native core,
and adds the corrected-admissible transform (the battery's
NaN/negative terminal weights are outside corrected_v1's typed domain;
``np.abs`` on the terminal-weight arrays makes them admissible while
keeping every other adversarial property).

This directory is a collection island: run ``pytest tests/native_core``
alone (the conftest manipulates sys.path).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

_CPUK_CONF = (Path(__file__).resolve().parents[1] / "cpu_kernels"
              / "conftest.py")
_spec = importlib.util.spec_from_file_location(
    "_native_core_cpuk_conftest", _CPUK_CONF)
_cpuk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cpuk)

build_case = _cpuk.build_case
CASE_PARAMS = _cpuk.CASE_PARAMS
case_kwargs = _cpuk.case_kwargs
bits = _cpuk.bits


def corrected_case(name):
    """The battery case made corrected_v1-admissible: terminal split
    weights are distances along a host edge, so |.| preserves the case's
    adversarial structure (topology, duplicate arcs, self-loops, seeds
    at/over the cutoff) while clearing the typed cost domain."""
    kw = dict(build_case(3 + CASE_PARAMS.index(name), **case_kwargs(name)))
    kw["ow"] = np.abs(kw["ow"])
    kw["dw"] = np.abs(kw["dw"])
    return kw


# Corrected-admissible battery members (nan_seed_end keeps its NaN under
# abs and weights_f32 keeps its float32 edge weights — both are the
# typed-refusal test rows, not parity rows).
CORRECTED_PARITY_PARAMS = ["base", "neg_seed", "big_seed", "same_terminal",
                           "zero_cutoff"]


@pytest.fixture(scope="session")
def core():
    """The loaded compiled core; skips the suite when it is not built
    (Phase 2 builds it first — a skip here means an unbuilt host, not a
    passing proof)."""
    from urban_network_analysis.backends.native._loader import (
        NativeUnavailableError,
        load_core,
    )
    try:
        return load_core()
    except NativeUnavailableError as exc:  # pragma: no cover - unbuilt host
        pytest.skip(f"native core not built on this host: {exc}")


@pytest.fixture(scope="session")
def engines():
    """The compiled engine modules (numba import; legacy parity refs)."""
    from urban_network_analysis.Engines import _large_access_scratch as LAS
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        compact_vector_node_view_scope,
        integrated_scope_access,
    )
    from types import SimpleNamespace
    return SimpleNamespace(LAS=LAS, compact=compact_vector_node_view_scope,
                           integrated=integrated_scope_access)
