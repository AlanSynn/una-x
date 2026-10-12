"""Engagement honesty: loader refusals, cold import, package failure.

The dossier's fail-sentinel gates: the backend may claim engagement
ONLY when the compiled core is actually loaded and self-identifies;
every refusal condition produces a recorded fallback with
``native_engaged=False`` and NO engagement counters — a shim that
engages "sometimes" or a stale binary that loads "anyway" fails here.
"""
import contextlib
import os
import subprocess
import sys
import types

import numpy as np
import pytest

from urban_network_analysis.backends.native import (
    NativeUnavailableError,
    is_available,
    native_capabilities,
    route_scope_access,
)
from urban_network_analysis.backends.native import _loader

from conftest import REPO_SRC, bits, build_case, corrected_case

CORE_NAME = _loader.MODULE_NAME


def fold_args(c):
    return (c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
            c["d"], c["dw"], c["dwt"], 0.02, 5.0, 8.0, 0.3, "exponential",
            np.array([0.9, 0.5, 0.25, 0.1]), c["cutoff"])


@contextlib.contextmanager
def poisoned_module(stub):
    """Temporarily replace (or remove) the compiled module in
    sys.modules and reset the loader cache, restoring everything."""
    prior = sys.modules.get(CORE_NAME, "__absent__")
    sys.modules[CORE_NAME] = stub
    _loader.reset_loader_cache()
    try:
        yield
    finally:
        if prior == "__absent__":
            sys.modules.pop(CORE_NAME, None)
        else:
            sys.modules[CORE_NAME] = prior
        _loader.reset_loader_cache()


def stub_module(**attrs):
    """A full-symbol stub with a WRONG abi_version (the skew case);
    individual attrs override the placeholders."""
    stub = types.ModuleType(CORE_NAME)
    for name in _loader.REQUIRED_SYMBOLS:
        setattr(stub, name, lambda *a, **k: None)
    stub.abi_version = lambda: 999
    for name, value in attrs.items():
        setattr(stub, name, value)
    return stub


# ---------------------------------------------------------------------------
# Loader refusals
# ---------------------------------------------------------------------------

def test_loader_refuses_foreign_stub_missing_symbols():
    stub = types.ModuleType(CORE_NAME)
    stub.abi_version = lambda: 1
    with poisoned_module(stub):
        with pytest.raises(NativeUnavailableError, match="missing required"):
            _loader.load_core()


def test_loader_refuses_abi_skew():
    stub = stub_module()  # every symbol present, wrong abi_version
    with poisoned_module(stub):
        with pytest.raises(NativeUnavailableError, match="abi skew"):
            _loader.load_core()


def test_loader_refuses_broken_module():
    with poisoned_module(None):  # import halted: sys.modules[name] = None
        with pytest.raises(NativeUnavailableError, match="not importable"):
            _loader.load_core()


def test_route_records_loader_refusal():
    """The route turns a missing/stale binary into the recorded portable
    fallback — never an ImportError leak, never silent engagement."""
    stub = types.ModuleType(CORE_NAME)
    with poisoned_module(stub):
        run = route_scope_access("una_legacy", *fold_args(build_case(5)))
        assert run.native_engaged is False
        assert run.fallback_reason is not None
        assert "native-unavailable" in run.fallback_reason
        assert run.engagement == {}
        assert run.outputs.status == "complete"  # portable route finished


def test_disable_env_is_rechecked_per_load():
    """The environment gate is evaluated on every load, not cached into
    engagement: a disabled load never returns a stale core."""
    c = fold_args(build_case(5))
    assert route_scope_access("una_legacy", *c).native_engaged in \
        (True, False)  # may or may not be built; the check is honest
    old = os.environ.get("UNA_NATIVE_CORE_DISABLE")
    os.environ["UNA_NATIVE_CORE_DISABLE"] = "1"
    _loader.reset_loader_cache()
    try:
        with pytest.raises(NativeUnavailableError, match="disabled"):
            _loader.load_core()
        assert is_available() is False
    finally:
        _loader.reset_loader_cache()
        if old is None:
            os.environ.pop("UNA_NATIVE_CORE_DISABLE", None)
        else:
            os.environ["UNA_NATIVE_CORE_DISABLE"] = old


# ---------------------------------------------------------------------------
# Cold import: the whole flow with the binary absent, in a subprocess
# ---------------------------------------------------------------------------

def test_cold_import_full_fallback_subprocess():
    """Cold interpreter with UNA_NATIVE_CORE_DISABLE=1: the package
    imports, availability reports False, and BOTH profile routes fall
    back bitwise to their pinned portable compositions."""
    code = """
import sys
sys.path.insert(0, {src!r})
sys.path.insert(0, {tst!r})
import numpy as np
import urban_network_analysis.backends.native as nat
assert nat.is_available() is False, "disabled core must not engage"
assert not hasattr(nat, "_native_core"), "compiled module was imported"
from conftest import bits, build_case, corrected_case
from urban_network_analysis.kernels import scope_access as SA
from urban_network_analysis.kernels.reductions import (
    adjust_destination_distances_plan,
    fold_reach_gravity_knn_plan,
)
from urban_network_analysis.reference.search import corrected_scope_labels

KNN = np.array([0.9, 0.5, 0.25, 0.1])

c = build_case(5)
args = (c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
        c["d"], c["dw"], c["dwt"], 0.02, 5.0, 8.0, 0.3, "exponential",
        KNN, c["cutoff"])
oracle = SA.integrated_scope_access_oracle(*args)
run = nat.route_scope_access("una_legacy", *args)
assert run.native_engaged is False
assert run.fallback_reason.startswith("native-unavailable")
assert bits(run.result.reach) == bits(oracle.reach)
assert bits(run.result.knn_access) == bits(oracle.knn_access)

cc = corrected_case("base")
args_c = (cc["o"], cc["ow"], cc["ptr"], cc["vec"], cc["wts"], cc["flag"],
          cc["d"], cc["dw"], cc["dwt"], 0.02, 5.0, 8.0, 0.3,
          "exponential", KNN, cc["cutoff"])
run_c = nat.route_scope_access("corrected_v1", *args_c)
assert run_c.native_engaged is False
rows = np.empty((cc["o"].shape[0], cc["d"].shape[0]), dtype=np.float64)
for i in range(cc["o"].shape[0]):
    lab = corrected_scope_labels(cc["o"][i:i + 1], cc["ow"][i:i + 1],
                                 cc["ptr"], cc["vec"], cc["wts"],
                                 cc["cutoff"])
    rows[i, :] = adjust_destination_distances_plan(lab, cc["d"], cc["dw"])
ge = np.empty(cc["o"].shape[0], dtype=np.float64)
for i in range(cc["o"].shape[0]):
    ge[i] = fold_reach_gravity_knn_plan(
        rows[i], cc["dwt"], cc["cutoff"], 0.02, 5.0, 8.0, 0.3,
        "exponential", KNN)[1]
assert bits(run_c.result.gravity_exponential) == bits(ge)
print("COLD-IMPORT FALLBACK OK")
"""
    tst = str(REPO_SRC.parent / "tests" / "native_core")
    env = dict(os.environ, UNA_NATIVE_CORE_DISABLE="1",
               PYTHONHASHSEED="0")
    # This child runs the DISABLE path: it never imports the compiled
    # core, so it gains nothing from an inherited sanitizer environment —
    # while numba/llvmlite/libcrypto leak reports at ITS exit would flip
    # returncode and mask the actual assertion target.  Sanitizer
    # coverage for the core comes from every other test in the same run.
    for sanitizer_var in ("LD_PRELOAD", "ASAN_OPTIONS", "UBSAN_OPTIONS",
                          "LSAN_OPTIONS"):
        env.pop(sanitizer_var, None)
    proc = subprocess.run([sys.executable, "-c", code.format(src=str(REPO_SRC),
                                                              tst=tst)],
                          env=env, capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "COLD-IMPORT FALLBACK OK" in proc.stdout


# ---------------------------------------------------------------------------
# Capability honesty
# ---------------------------------------------------------------------------

def test_capabilities_reflect_loaded_binary(core):
    cap = native_capabilities()
    assert cap.backend == "native"
    assert set(cap.supported_profiles) == {"una_legacy", "corrected_v1"}
    assert cap.binary_fingerprint == str(core.build_fingerprint())
    # The stated requirement is the COMPILED baseline (x86-64-v1), not
    # the host's reported level — a v4 host must not inflate the claim.
    assert cap.isa_or_device_requirements == \
        (str(core.isa_report()["compiled_baseline"]),)
    assert cap.isa_or_device_requirements[0] == "x86-64-v1"
    # Qualification is a separate calibrated disposition; the claim never
    # self-qualifies (contracts.py: entries never qualify a forced route).
    assert cap.qualified is False
    assert cap.evidence_ref is None


def test_capabilities_refuse_when_disabled():
    old = os.environ.get("UNA_NATIVE_CORE_DISABLE")
    os.environ["UNA_NATIVE_CORE_DISABLE"] = "1"
    _loader.reset_loader_cache()
    try:
        with pytest.raises(NativeUnavailableError):
            native_capabilities()
    finally:
        _loader.reset_loader_cache()
        if old is None:
            os.environ.pop("UNA_NATIVE_CORE_DISABLE", None)
        else:
            os.environ["UNA_NATIVE_CORE_DISABLE"] = old


def test_corrected_fallback_result_is_reference_composition():
    """In-process native-absent run for corrected_v1 equals the exact
    reference composition (not an approximation of it)."""
    from urban_network_analysis.kernels.reductions import (
        adjust_destination_distances_plan,
        fold_reach_gravity_knn_plan,
    )
    from urban_network_analysis.reference.search import corrected_scope_labels

    c = corrected_case("base")
    old = os.environ.get("UNA_NATIVE_CORE_DISABLE")
    os.environ["UNA_NATIVE_CORE_DISABLE"] = "1"
    _loader.reset_loader_cache()
    try:
        run = route_scope_access("corrected_v1", *fold_args(c))
    finally:
        _loader.reset_loader_cache()
        if old is None:
            os.environ.pop("UNA_NATIVE_CORE_DISABLE", None)
        else:
            os.environ["UNA_NATIVE_CORE_DISABLE"] = old
    assert run.native_engaged is False
    assert run.route == "corrected_reference"  # attribution honesty
    assert run.engagement == {}
    rows = np.empty((c["o"].shape[0], c["d"].shape[0]), dtype=np.float64)
    for i in range(c["o"].shape[0]):
        lab = corrected_scope_labels(
            c["o"][i:i + 1], c["ow"][i:i + 1], c["ptr"], c["vec"], c["wts"],
            c["cutoff"])
        rows[i, :] = adjust_destination_distances_plan(lab, c["d"], c["dw"])
    ka = np.empty(c["o"].shape[0], dtype=np.float64)
    for i in range(c["o"].shape[0]):
        ka[i] = fold_reach_gravity_knn_plan(
            rows[i], c["dwt"], c["cutoff"], 0.02, 5.0, 8.0, 0.3,
            "exponential", np.array([0.9, 0.5, 0.25, 0.1]))[3]
    assert bits(run.result.knn_access) == bits(ka)
