"""Reflection matrix: the facade's public una surface must match the
pinned upstream surface — the delivered accessibility trio and tools
functions with their exact signatures, star-export identity, and the
module namespaces modulo an EXPLICIT ledgered delta.

Reflection is necessary but not sufficient (dossier 01); the behavior
scenarios live in test_access_surface_parity.py.  Post-ACCESS ledgered
deltas here (see compat una/__init__ ledger; MADINA_FLOW extends the
same facade module and retires the rest):

una package, reference-only names
  the five undelivered betweenness functions (parallel_betweenness,
  one_betweenness_2, clockwiseangle_and_distance, betweenness_exposure,
  paralell_betweenness_exposure).  The betweenness module's public
  imports (concurrent, futures, getsizeof, gpd, mp, np, os, pd, psutil,
  random, time, Zonal) and the submodule binding now exist on BOTH
  sides (the facade betweenness.py carries the verbatim import block),
  so they retired from this ledger at ACCESS delivery.
una package, facade-only names
  none expected.
una.tools, reference-only names
  betweenness (the fifth upstream tools function, MADINA_FLOW) and
  paralell_betweenness_exposure (upstream tools re-imports it).
una.tools, facade-only names
  none expected.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.madina_api

# ---- ledgered deltas (see module docstring) -------------------------
REFERENCE_ONLY_UNA_NAMES = {
    "parallel_betweenness", "one_betweenness_2",
    "clockwiseangle_and_distance", "betweenness_exposure",
    "paralell_betweenness_exposure",
}
FACADE_ONLY_UNA_NAMES = set()
REFERENCE_ONLY_TOOLS_NAMES = {"betweenness", "paralell_betweenness_exposure"}
FACADE_ONLY_TOOLS_NAMES = set()

BETWEENNESS_FUNCTIONS = (
    "get_origin_properties",
    "one_access",
    "parallel_access",
)
TOOLS_FUNCTIONS = (
    "validate_zonal_ready",
    "accessibility",
    "service_area",
    "alternative_paths",
)


def _annotation_aliases(sig):
    """Annotation module qualifiers differ between the trees (the
    verbatim port binds ``Zonal`` from the compat package, upstream from
    madina.zonal); strip the qualifier on BOTH sides so the remainder
    must be exactly equal."""
    # longest first: the compat qualifier CONTAINS the upstream one as
    # a substring
    for prefix in ("urban_network_analysis.compat.madina.zonal.network.",
                   "urban_network_analysis.compat.madina.zonal.zonal.",
                   "madina.zonal.network.", "madina.zonal.zonal."):
        sig = sig.replace(prefix, "")
    return sig


def test_reflection_matches_pinned_reference(reflection_runner):
    ref = reflection_runner("reference")
    facade = reflection_runner("facade")
    assert "surface" in ref and "surface" in facade
    assert ref["identity"]["package_has_path"] is True
    assert facade["identity"]["package_has_path"] is True

    rs, fs = ref["surface"], facade["surface"]
    ref_una = set(rs["una_public"]) - REFERENCE_ONLY_UNA_NAMES
    fa_una = set(fs["una_public"]) - FACADE_ONLY_UNA_NAMES
    ref_tools = set(rs["tools_public"]) - REFERENCE_ONLY_TOOLS_NAMES
    fa_tools = set(fs["tools_public"]) - FACADE_ONLY_TOOLS_NAMES

    assert fa_una == ref_una, (
        f"unledgered una delta: only-ref={sorted(ref_una - fa_una)} "
        f"only-facade={sorted(fa_una - ref_una)}")
    assert fa_tools == ref_tools, (
        f"unledgered tools delta: only-ref={sorted(ref_tools - fa_tools)} "
        f"only-facade={sorted(fa_tools - ref_tools)}")

    # delivered surface is a verbatim port: EXACT equality modulo the
    # documented annotation aliases
    for key in ("betweenness_functions", "tools_functions"):
        ref_sigs = {k: _annotation_aliases(v) for k, v in rs[key].items()}
        fa_sigs = {k: _annotation_aliases(v) for k, v in fs[key].items()}
        assert fa_sigs == ref_sigs, f"{key} differ after alias stripping"

    # the betweenness MODULE namespace (public imports carried by the
    # verbatim import block) matches modulo the same ledger — the five
    # undelivered functions are the module's reference-only names too
    ref_btd = set(rs["betweenness_public"]) - REFERENCE_ONLY_UNA_NAMES
    fa_btd = set(fs["betweenness_public"]) - FACADE_ONLY_UNA_NAMES
    assert fa_btd == ref_btd, (
        f"unledgered betweenness-module delta: "
        f"only-ref={sorted(ref_btd - fa_btd)} "
        f"only-facade={sorted(fa_btd - ref_btd)}")


def test_star_exports_and_submodule_identity(reflection_runner):
    """una.<fn> is the module attribute for every delivered function
    (upstream star-import semantics), and the submodule bindings
    resolve."""
    rs = reflection_runner("reference")["surface"]
    fs = reflection_runner("facade")["surface"]
    assert rs["star_exports_are_betweenness_functions"] is True
    assert fs["star_exports_are_betweenness_functions"] is True
    # neither package binds the tools functions at the una level (the
    # tools surface is reachable only through the submodule) — pinned
    # equal and false on both arms
    assert rs["una_level_tools_bindings"] == fs[
        "una_level_tools_bindings"]
    assert set(fs["una_level_tools_bindings"].values()) == {False}
    assert rs["una_betweenness_is_betweenness"] is True
    assert fs["una_betweenness_is_betweenness"] is True
    assert rs["una_tools_is_tools"] is True
    assert fs["una_tools_is_tools"] is True
