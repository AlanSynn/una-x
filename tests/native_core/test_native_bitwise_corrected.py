"""corrected_v1 bitwise battery: native core vs the reference profile.

The native core claims bit-identical execution of the corrected scope
schedule (``reference/search.py::corrected_scope_labels``) and the
builtin-min adjust on the typed-admitted domain.  The pinning chain:

- labels: native vs ``corrected_scope_labels`` (the independent,
  reviewed scalar reference) — byte-for-byte;
- adjust: native ``run_corrected`` rows vs
  ``reductions.adjust_destination_distances_plan`` on the reference
  labels — byte-for-byte (the plan is the frozen PLAN_ADJUST);
- fold: composed from the SAME canonical plan function on both sides
  (identity by construction; its own pins live in tests/cpu_kernels);
- typed refusals: the reference error classes, raised by the route,
  mirrored status codes at the raw binding.

The battery cases are made corrected-admissible with ``|.|`` on the
terminal split weights (distances along a host edge); the two cases
that stay inadmissible under |.| (NaN seed; float32 edge weights) are
the typed-refusal rows.
"""
import numpy as np
import pytest

from urban_network_analysis.backends.native import route_scope_access
from urban_network_analysis.backends.native._loader import (
    NativeUnavailableError,
    load_core,
)
from urban_network_analysis.kernels.reductions import (
    adjust_destination_distances_plan,
)

from conftest import (CORRECTED_PARITY_PARAMS, bits, build_case,
                      case_kwargs, corrected_case)

try:
    CORE = load_core()
except NativeUnavailableError:  # pragma: no cover - unbuilt host
    CORE = None

needs_core = pytest.mark.skipif(CORE is None,
                                reason="native core not built on this host")


def fold_args(c):
    return (c["o"], c["ow"], c["ptr"], c["vec"], c["wts"], c["flag"],
            c["d"], c["dw"], c["dwt"],
            0.02, 5.0, 8.0, 0.3, "exponential",
            np.array([0.9, 0.5, 0.25, 0.1]), c["cutoff"])


def reference_rows(c):
    """(o_count, d_count) reference adjust rows: corrected labels per
    origin + the frozen adjust plan (the composition the native core is
    pinned against, piece by piece)."""
    d_count = c["d"].shape[0]
    rows = np.empty((c["o"].shape[0], d_count), dtype=np.float64)
    for i in range(c["o"].shape[0]):
        label = CORE_ref_labels(c, i)
        rows[i, :] = adjust_destination_distances_plan(label, c["d"], c["dw"])
    return rows


def CORE_ref_labels(c, i):
    from urban_network_analysis.reference.search import corrected_scope_labels
    return corrected_scope_labels(
        c["o"][i:i + 1], c["ow"][i:i + 1], c["ptr"], c["vec"], c["wts"],
        c["cutoff"])


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

@needs_core
@pytest.mark.parametrize("name", CORRECTED_PARITY_PARAMS)
def test_search_labels_corrected_bitwise_vs_reference(name):
    """Native corrected labels == reference labels, byte-for-byte."""
    c = corrected_case(name)
    native = CORE.search_labels_corrected(
        c["o"][0], c["ow"][0], c["ptr"], c["vec"], c["wts"], c["cutoff"])
    reference = CORE_ref_labels(c, 0)
    assert native.dtype == np.float64
    assert native.shape[0] == c["ptr"].shape[0] - 1
    assert bits(native) == bits(reference)


@needs_core
@pytest.mark.parametrize("name", CORRECTED_PARITY_PARAMS)
def test_adjust_rows_bitwise_vs_plan(name):
    """Native run_corrected rows == adjust plan on reference labels."""
    c = corrected_case(name)
    o_count, d_count = c["o"].shape[0], c["d"].shape[0]
    out = np.empty((o_count, d_count), dtype=np.float64)
    rep = CORE.run_corrected(
        c["ptr"], c["vec"], c["wts"], c["o"], c["ow"], c["d"], c["dw"],
        c["cutoff"], 1, None, out)
    assert int(rep["status"]) == int(CORE.status_codes()["OK"])
    assert bits(out) == bits(reference_rows(c))


# ---------------------------------------------------------------------------
# Whole route (native search+adjust composed with the canonical fold)
# ---------------------------------------------------------------------------

@needs_core
@pytest.mark.parametrize("name", CORRECTED_PARITY_PARAMS)
def test_route_corrected_bitwise_and_status(name):
    """The corrected route: native-engaged, complete, and the published
    arrays equal the reference composition (labels pin + adjust pin +
    the same fold plan)."""
    c = corrected_case(name)
    run = route_scope_access("corrected_v1", *fold_args(c))
    assert run.native_engaged is True
    assert run.route == "corrected_native"
    assert run.outputs.status == "complete"
    rows = reference_rows(c)
    # Fold the reference rows with the canonical corrected plan — the
    # exact composition the route performs on the native rows.
    from urban_network_analysis.kernels.reductions import (
        fold_reach_gravity_knn_plan,
    )
    reach = np.empty(c["o"].shape[0], dtype=c["o"].dtype)
    ge = np.empty(c["o"].shape[0], dtype=np.float64)
    gl = np.empty(c["o"].shape[0], dtype=np.float64)
    ka = np.empty(c["o"].shape[0], dtype=np.float64)
    kept = 0
    for i in range(c["o"].shape[0]):
        kept += int(np.count_nonzero(rows[i] <= c["cutoff"]))
        r, g_ex, g_lo, k = fold_reach_gravity_knn_plan(
            rows[i], c["dwt"], c["cutoff"], 0.02, 5.0, 8.0, 0.3,
            "exponential", np.array([0.9, 0.5, 0.25, 0.1]))
        reach[i], ge[i], gl[i], ka[i] = r, g_ex, g_lo, k
    assert bits(run.result.reach) == bits(reach)
    assert bits(run.result.gravity_exponential) == bits(ge)
    assert bits(run.result.gravity_logistic) == bits(gl)
    assert bits(run.result.knn_access) == bits(ka)
    assert run.engagement["kept_count"] == kept
    # At cutoff 0 no seed label is inside the closed radius, so zero heap
    # work is the honest counter there; every other case must show real
    # traversal (the fail-sentinel against a shim).
    if name != "zero_cutoff":
        assert run.engagement["heap_pushes"] > 0


# ---------------------------------------------------------------------------
# Typed refusals: the reference error contract (route raises the
# reference classes; the raw binding mirrors the status codes)
# ---------------------------------------------------------------------------

@needs_core
@pytest.mark.parametrize("case_name,err_name", [
    ("nan_seed_end", "NonFiniteCostError"),
    ("neg_seed", "NegativeCostError"),
    ("weights_f32", "MalformedGraphError"),
])
def test_typed_refusals_route(case_name, err_name):
    """The corrected route refuses the typed domain with the REFERENCE
    error classes (first failure wins, reference order)."""
    from urban_network_analysis.reference import errors as ref_errors
    c = build_case(3 + ["base", "nan_seed_end", "neg_seed", "big_seed",
                        "same_terminal", "zero_cutoff",
                        "weights_f32"].index(case_name),
                   **case_kwargs(case_name))
    err_type = getattr(ref_errors, err_name)
    with pytest.raises(err_type):
        route_scope_access("corrected_v1", *fold_args(c))


@needs_core
@pytest.mark.parametrize("mutate,err_name", [
    ("cutoff_inf", "NonFiniteCostError"),
    ("cutoff_neg", "NegativeCostError"),
    ("oob_origin", "OutOfRangeTerminalError"),
    ("oob_destination", "OutOfRangeTerminalError"),
    ("ptr_not_zero", "MalformedGraphError"),
    ("vec_oob", "MalformedGraphError"),
])
def test_typed_refusals_scenarios(mutate, err_name):
    from urban_network_analysis.reference import errors as ref_errors
    c = corrected_case("base")
    if mutate == "cutoff_inf":
        c = dict(c, cutoff=float("inf"))
    elif mutate == "cutoff_neg":
        c = dict(c, cutoff=-1.0)
    elif mutate == "oob_origin":
        o = c["o"].copy()
        o[0, 0] = 10 ** 6
        c = dict(c, o=o)
    elif mutate == "oob_destination":
        d = c["d"].copy()
        d[1, 1] = -2
        c = dict(c, d=d)
    elif mutate == "ptr_not_zero":
        ptr = c["ptr"].copy()
        ptr[0] = 1
        c = dict(c, ptr=ptr)
    elif mutate == "vec_oob":
        vec = c["vec"].copy()
        vec[0] = c["ptr"].shape[0] - 1  # == node_count, one past the domain
        c = dict(c, vec=vec)
    with pytest.raises(getattr(ref_errors, err_name)):
        route_scope_access("corrected_v1", *fold_args(c))


@needs_core
@pytest.mark.parametrize("case_name,status_name", [
    ("nan_seed_end", "NONFINITE_COST"),
    ("neg_seed", "NEGATIVE_COST"),
    ("oob_origin", "OOB_TERMINAL"),
])
def test_typed_refusals_binding_status_codes(case_name, status_name):
    """The raw binding reports the same refusals as structured status
    codes (translated after GIL reacquire) — no Python exception, no
    partial writes to the out buffer it was handed.  (Off-contract
    dtypes like weights_f32 are refused at the BINDING layout layer
    before the core scan; the wrapper's validators raise the typed
    error first — covered by test_binding_refuses_offcontract_dtypes.)"""
    codes = CORE.status_codes()
    battery = ["base", "nan_seed_end", "neg_seed", "big_seed",
               "same_terminal", "zero_cutoff", "weights_f32"]
    kwargs = case_kwargs(case_name) if case_name in battery else {}
    c = build_case(3 + (battery.index(case_name) if case_name in battery
                        else 0), **kwargs)
    if case_name == "oob_origin":
        o = c["o"].copy()
        o[0, 0] = 10 ** 6
        c = dict(c, o=o)
    out = np.zeros((c["o"].shape[0], c["d"].shape[0]), dtype=np.float64)
    rep = CORE.run_corrected(
        c["ptr"], c["vec"], c["wts"], c["o"], c["ow"], c["d"], c["dw"],
        c["cutoff"], 1, None, out)
    assert int(rep["status"]) == int(codes[status_name])
    # No partial publication: the caller's buffer stays untouched on a
    # refused run.
    assert not out.any()


@needs_core
def test_cutoff_int_widening_is_the_charged_conversion():
    """int cutoff == float cutoff bit-for-bit through the route (the one
    int->float64 widening, mirroring the compiled kernel's parameter)."""
    c = corrected_case("base")
    run_f = route_scope_access("corrected_v1", *fold_args(c))
    run_i = route_scope_access("corrected_v1", *fold_args(c)[:15], 25)
    assert bits(run_i.result.reach) == bits(run_f.result.reach)
    assert bits(run_i.result.gravity_exponential) == \
        bits(run_f.result.gravity_exponential)
