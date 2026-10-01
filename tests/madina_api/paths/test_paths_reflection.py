"""Reflection matrix: the facade's public una surface must match the
pinned upstream surface — every paths function with its exact
signature, star-export identity, and the module namespaces modulo an
EXPLICIT ledgered interim delta.

Reflection is necessary but not sufficient (dossier 01); the behavior
scenarios live in test_paths_surface_parity.py.  Interim ledgered
deltas here (see compat una/__init__ ledger; the accessibility and
betweenness facade modules land with their delivering tasks and these
lists shrink to empty then):

una package, reference-only names
  the upstream betweenness-module star-exports (parallel_betweenness,
  one_betweenness_2, clockwiseangle_and_distance, betweenness_exposure,
  paralell_betweenness_exposure, get_origin_properties, one_access,
  parallel_access) plus the submodule binding ``betweenness`` and the
  betweenness module's own public imports (concurrent, futures,
  getsizeof, gpd, mp, np, os, pd, psutil, random, time, Zonal — Zonal
  reaches una through betweenness.py's ``from ..zonal import Zonal``;
  the facade paths.py imports Network the same way, which is why
  Network appears on both sides).
una package, facade-only names
  none expected.
una.tools, reference-only names
  validate_zonal_ready, accessibility, service_area (accessibility
  task), betweenness (betweenness task), paralell_betweenness_exposure
  and parallel_access (upstream tools re-imports them from the
  betweenness module).
una.tools, facade-only names
  none expected.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.madina_api

# ---- ledgered interim deltas (see module docstring) -----------------
REFERENCE_ONLY_UNA_NAMES = {
    "betweenness",
    "parallel_betweenness", "one_betweenness_2",
    "clockwiseangle_and_distance", "betweenness_exposure",
    "paralell_betweenness_exposure", "get_origin_properties",
    "one_access", "parallel_access",
    "concurrent", "futures", "getsizeof", "gpd", "mp", "np", "os",
    "pd", "psutil", "random", "time", "Zonal",
}
FACADE_ONLY_UNA_NAMES = set()
REFERENCE_ONLY_TOOLS_NAMES = {
    "validate_zonal_ready", "accessibility", "service_area",
    "betweenness", "paralell_betweenness_exposure", "parallel_access",
}
FACADE_ONLY_TOOLS_NAMES = set()


def _annotation_aliases(sig):
    """Annotation module qualifiers differ between the trees (the
    verbatim port binds ``Network``/``Zonal`` from the compat package,
    upstream from madina.zonal); strip the qualifier on BOTH sides so
    the remainder must be exactly equal."""
    # longest first: the compat qualifier CONTAINS the upstream one as
    # a substring
    for prefix in ("urban_network_analysis.compat.madina.zonal.network.",
                   "urban_network_analysis.compat.madina.zonal.zonal.",
                   "madina.zonal.network.", "madina.zonal.zonal."):
        sig = sig.replace(prefix, "")
    return sig


def _normalized(state, is_facade):
    una_names = set(state["una_public"])
    tools_names = set(state["tools_public"])
    una_names -= (FACADE_ONLY_UNA_NAMES if is_facade
                  else REFERENCE_ONLY_UNA_NAMES)
    tools_names -= (FACADE_ONLY_TOOLS_NAMES if is_facade
                    else REFERENCE_ONLY_TOOLS_NAMES)
    paths_sigs = {k: _annotation_aliases(v)
                  for k, v in state["paths_functions"].items()}
    tools_sigs = {k: _annotation_aliases(v)
                  for k, v in state["tools_functions"].items()}
    return una_names, tools_names, state["paths_public"], \
        paths_sigs, tools_sigs


def test_reflection_matches_pinned_reference(reflection_runner):
    ref = reflection_runner("reference")
    facade = reflection_runner("facade")
    assert "surface" in ref and "surface" in facade

    # both namespaces are packages (una has no star-import shadowing:
    # madina/__init__ binds una via `from .una import *`, which does
    # not copy the submodule over the package attribute)
    assert ref["identity"]["package_has_path"] is True
    assert facade["identity"]["package_has_path"] is True

    ref_una, ref_tools, ref_paths_names, ref_paths_sigs, ref_tools_sigs = \
        _normalized(ref["surface"], False)
    fa_una, fa_tools, fa_paths_names, fa_paths_sigs, fa_tools_sigs = \
        _normalized(facade["surface"], True)

    # the paths module surface is a verbatim port: EXACT equality
    assert fa_paths_names == ref_paths_names, (
        f"paths namespace delta: only-ref={sorted(set(ref_paths_names) - set(fa_paths_names))} "
        f"only-facade={sorted(set(fa_paths_names) - set(ref_paths_names))}")
    assert fa_paths_sigs == ref_paths_sigs, "paths signatures differ"
    assert fa_tools_sigs == ref_tools_sigs, "tools signatures differ"

    # package namespaces equal modulo the interim ledger
    assert fa_una == ref_una, (
        f"unledgered una delta: only-ref={sorted(ref_una - fa_una)} "
        f"only-facade={sorted(fa_una - ref_una)}")
    assert fa_tools == ref_tools, (
        f"unledgered tools delta: only-ref={sorted(ref_tools - fa_tools)} "
        f"only-facade={sorted(fa_tools - ref_tools)}")


def test_star_exports_and_submodule_identity(reflection_runner):
    """una.<fn> is una.paths.<fn> for every paths function (upstream
    star-import semantics), and the submodule bindings resolve."""
    ref = reflection_runner("reference")["surface"]
    facade = reflection_runner("facade")["surface"]
    assert ref["star_exports_are_paths_functions"] is True
    assert facade["star_exports_are_paths_functions"] is True
    assert ref["una_paths_is_paths"] is True
    assert facade["una_paths_is_paths"] is True
    assert ref["una_tools_is_tools"] is True
    assert facade["una_tools_is_tools"] is True
